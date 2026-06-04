import os
import json
import threading
import time
import hashlib
from pathlib import Path
from logger import get_logger

log = get_logger("rag")

_RAG_ENABLED = False
_COLLECTION = None
_lock = threading.Lock()

_SEARCH_CACHE: dict = {}
_SEARCH_CACHE_TTL = 30


class OllamaEmbeddingFunction:
    """Custom ChromaDB embedding function using Ollama API. Fully offline."""

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        embeddings = [[0.0] * 768] * len(input)
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        
        def fetch_embedding(idx, text):
            try:
                resp = requests.post(
                    f"{ollama_host}/api/embed",
                    json={"model": self.model_name, "input": text},
                    timeout=60,
                )
                resp.raise_for_status()
                return idx, resp.json()["embeddings"][0]
            except Exception as e:
                log.warning("Ollama embedding failed for text (%d chars): %s", len(text), e)
                return idx, [0.0] * 768

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_embedding, i, text) for i, text in enumerate(input)]
            for future in as_completed(futures):
                idx, emb = future.result()
                embeddings[idx] = emb
                
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.__call__([query])[0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self.__call__(documents)

def is_rag_enabled() -> bool:
    """Check if the RAG module is currently active."""
    global _RAG_ENABLED
    return _RAG_ENABLED

def disable_rag():
    """Disables RAG and clears the collection reference."""
    global _RAG_ENABLED, _COLLECTION
    _RAG_ENABLED = False
    _COLLECTION = None

def enable_rag_for_project(project_dir: str) -> str:
    """Initializes ChromaDB, creates embeddings for the project, and enables semantic search."""
    global _RAG_ENABLED, _COLLECTION
    
    try:
        import chromadb
    except ImportError:
        return "ERROR: 'chromadb' is not installed. Please run `pip install chromadb` to use RAG."
        
    try:
        from config import get_embedding_provider, get_ollama_embedding_model

        project_path = Path(project_dir).expanduser().resolve()
        if not project_path.exists():
            return f"Error: Project directory {project_dir} does not exist."
            
        print(f"\n[INFO] Initializing Vector Database for {project_path.name}...")
        
        db_path = project_path / ".argent" / "chroma_db"
        db_path.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(path=str(db_path))
        
        embedding_provider = get_embedding_provider()
        if embedding_provider == "ollama":
            ollama_model = get_ollama_embedding_model()
            print(f"[INFO] Using Ollama embeddings (model: {ollama_model}) — fully offline.")
            ef = OllamaEmbeddingFunction(model_name=ollama_model)
        else:
            try:
                from chromadb.utils import embedding_functions
                print("[INFO] Using sentence-transformers embeddings (all-MiniLM-L6-v2).")
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            except ImportError:
                return "ERROR: 'sentence-transformers' is not installed. Run `pip install sentence-transformers` or switch to Ollama embeddings."
        
        _COLLECTION = client.get_or_create_collection(name="project_codebase", embedding_function=ef)
        
        _index_codebase(project_path, _COLLECTION)
        
        _RAG_ENABLED = True
        return f"Successfully enabled RAG for '{project_path.name}' (embeddings: {embedding_provider}). Indexed files and ready for /search."
        
    except Exception as e:
        import traceback
        _RAG_ENABLED = False
        return f"Failed to enable RAG: {e}\n{traceback.format_exc()}"

def _chunk_python_ast(text: str, file_rel_path: str) -> tuple[list, list]:
    import ast
    docs = []
    metadatas = []
    try:
        tree = ast.parse(text)
    except Exception:
        # Fallback to heuristic on syntax errors
        return _chunk_heuristic(text, file_rel_path)

    chunks = []
    lines = text.splitlines()
    if not lines:
        return [], []

    # Identify top-level classes and functions
    blocks = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", len(lines))
            blocks.append((start, end, type(node).__name__, node.name))

    # Map line number (1-based) to its class/function block
    line_to_block = {}
    for start, end, btype, name in blocks:
        for l in range(start, end + 1):
            line_to_block[l] = (btype, name, start, end)

    current_chunk = []
    chunk_start = 1
    i = 1
    while i <= len(lines):
        line = lines[i - 1]
        if i in line_to_block:
            btype, name, bstart, bend = line_to_block[i]
            # Flush any module level statements gathered so far
            if current_chunk:
                chunks.append(("\n".join(current_chunk), chunk_start, i - 1, "Module level"))
                current_chunk = []
            
            block_lines = lines[bstart - 1 : bend]
            # If the block is very large, chunk it internally but keep the context name
            if len(block_lines) > 80:
                for sub_idx in range(0, len(block_lines), 60):
                    sub_chunk = block_lines[sub_idx : sub_idx + 60]
                    sub_start = bstart + sub_idx
                    sub_end = min(bend, bstart + sub_idx + len(sub_chunk) - 1)
                    chunks.append(("\n".join(sub_chunk), sub_start, sub_end, f"{btype} {name} (Part {sub_idx//60 + 1})"))
            else:
                chunks.append(("\n".join(block_lines), bstart, bend, f"{btype} {name}"))
            
            i = bend + 1
            chunk_start = i
        else:
            current_chunk.append(line)
            if len(current_chunk) >= 60:
                chunks.append(("\n".join(current_chunk), chunk_start, i, "Module level"))
                current_chunk = []
                chunk_start = i + 1
            i += 1

    if current_chunk:
        chunks.append(("\n".join(current_chunk), chunk_start, len(lines), "Module level"))

    for text_block, s_line, e_line, context in chunks:
        header = f"File: {file_rel_path}\nLines {s_line}-{e_line} ({context})\n"
        docs.append(header + text_block)
        metadatas.append({
            "file": str(file_rel_path),
            "start_line": s_line,
            "end_line": e_line,
            "context": context
        })

    return docs, metadatas


def _chunk_heuristic(text: str, file_rel_path: str) -> tuple[list, list]:
    import re
    docs = []
    metadatas = []
    
    lines = text.splitlines()
    if not lines:
        return docs, metadatas
        
    chunks = []
    current_chunk = []
    start_line = 1
    
    # Matches definitions with optional leading spaces: class, def, function, public/private void, etc.
    def_pattern = re.compile(
        r'^\s*(class\s+|def\s+|function\s+|export\s+(?:default\s+)?class\s+|'
        r'public\s+(?:class|struct|interface|void|async|static)|'
        r'private\s+(?:class|struct|interface|void|async|static))'
    )

    for i, line in enumerate(lines):
        if def_pattern.match(line) and len(current_chunk) > 20:
            chunks.append(("\n".join(current_chunk), start_line, i))
            current_chunk = [line]
            start_line = i + 1
        else:
            current_chunk.append(line)
            if len(current_chunk) >= 60:
                chunks.append(("\n".join(current_chunk), start_line, i + 1))
                current_chunk = []
                start_line = i + 2
                
    if current_chunk:
        chunks.append(("\n".join(current_chunk), start_line, len(lines)))
        
    for text_block, s_line, e_line in chunks:
        header = f"File: {file_rel_path}\nLines {s_line}-{e_line}\n"
        docs.append(header + text_block)
        metadatas.append({
            "file": str(file_rel_path),
            "start_line": s_line,
            "end_line": e_line,
            "context": "Heuristic block"
        })
        
    return docs, metadatas


def _chunk_text(text: str, file_rel_path: str) -> list:
    """Chunks text logically (using AST for Python, and regex heuristics for others)."""
    if file_rel_path.endswith(".py"):
        return _chunk_python_ast(text, file_rel_path)
    return _chunk_heuristic(text, file_rel_path)

def _load_argentignore(project_path: Path) -> set:
    """Load .argentignore patterns from the project root."""
    ignore_file = project_path / ".argentignore"
    patterns = set()
    if ignore_file.exists():
        try:
            for line in ignore_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.add(line.rstrip("/"))
        except Exception:
            pass
    return patterns


def _is_ignored(rel_path: str, patterns: set) -> bool:
    """Check if a relative path matches any .argentignore pattern."""
    if not patterns:
        return False
    parts = Path(rel_path).parts
    for pattern in patterns:
        if pattern in parts:
            return True
        if pattern.startswith("*."):
            if rel_path.endswith(pattern[1:]):
                return True
        if rel_path.startswith(pattern):
            return True
    return False


def _index_codebase(project_path: Path, collection):
    """Walks the codebase and chunks files into the vector database."""
    print("[INFO] Indexing codebase. This might take a minute...")
    
    docs = []
    ids = []
    metadatas = []
    
    supported_extensions = {".py", ".cs", ".js", ".ts", ".html", ".css", ".cpp", ".h", ".c"}
    default_ignore = {".git", ".argent", "node_modules", "venv", "env", "Library", "Temp", "Logs"}
    
    ignore_patterns = _load_argentignore(project_path)
    
    doc_id_counter = 0
    
    for root, dirs, files in os.walk(project_path):
        rel_root = str(Path(root).relative_to(project_path))
        dirs[:] = [d for d in dirs
                   if d not in default_ignore
                   and not _is_ignored(str(Path(rel_root) / d) if rel_root != "." else d, ignore_patterns)]
        for file in files:
            file_path = Path(root) / file
            rel = str(file_path.relative_to(project_path))
            if file_path.suffix in supported_extensions and not _is_ignored(rel, ignore_patterns):
                try:
                    with open(file_path, 'r', encoding="utf-8") as f:
                        source = f.read()
                    
                    file_docs, file_metas = _chunk_text(source, rel)
                    for d, m in zip(file_docs, file_metas):
                        docs.append(d)
                        metadatas.append(m)
                        chunk_id = hashlib.md5(f"{rel}_{m.get('start_line', doc_id_counter)}".encode()).hexdigest()
                        ids.append(f"doc_{chunk_id}")
                        doc_id_counter += 1
                except Exception:
                    pass
                    
    if docs:
        print(f"[INFO] Uploading {len(docs)} chunks to ChromaDB...")
        log.info("Indexing %d chunks into ChromaDB", len(docs))
        batch_size = 5000
        for i in range(0, len(docs), batch_size):
            collection.upsert(
                documents=docs[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
    print("[INFO] Indexing complete.")
    log.info("Indexing complete: %d docs, %d ids", len(docs), len(ids))

def update_file_index(file_path: str):
    """Updates the vector index for a single file. Useful for synchronous RAG updates."""
    global _RAG_ENABLED, _COLLECTION
    if not _RAG_ENABLED or _COLLECTION is None:
        return
        
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            return
            
        # 1. Delete old chunks for this file
        # Find current project root - heuristic: look for .argent folder
        project_root = path.parent
        while project_root != project_root.parent and not (project_root / ".argent").exists():
            project_root = project_root.parent
            
        if not (project_root / ".argent").exists():
            return # Root not found
            
        rel_path = str(path.relative_to(project_root))
        _COLLECTION.delete(where={"file": rel_path})
        
        # 2. Add new chunks
        with open(path, 'r', encoding='utf-8') as f:
            source = f.read()
            
        docs, metadatas = _chunk_text(source, rel_path)
        if docs:
            # Generate new IDs based on timestamp/filename to avoid collisions
            import time
            ts = int(time.time())
            ids = [f"upd_{ts}_{i}" for i in range(len(docs))]
            _COLLECTION.upsert(
                documents=docs,
                metadatas=metadatas,
                ids=ids
            )
    except Exception as e:
        print(f"[WARN] Failed to update RAG for {file_path}: {e}")

def semantic_search(query: str, n_results: int = 5) -> str:
    """Tool for the LLM to search the vector database for code snippets using Hybrid Search + RRF."""
    global _RAG_ENABLED, _COLLECTION
    
    if not _RAG_ENABLED or _COLLECTION is None:
        return "Error: RAG is not enabled. Cannot perform semantic search."
    
    cache_key = hashlib.md5(f"{query}:{n_results}".encode()).hexdigest()
    now = time.time()
    if cache_key in _SEARCH_CACHE:
        cached_ts, cached_result = _SEARCH_CACHE[cache_key]
        if now - cached_ts < _SEARCH_CACHE_TTL:
            return cached_result
        
    try:
        # 1. Semantic (Vector) search
        vector_results = _COLLECTION.query(
            query_texts=[query],
            n_results=n_results * 2
        )
        
        vector_docs = []
        if vector_results.get('documents') and vector_results['documents'][0]:
            for i in range(len(vector_results['documents'][0])):
                doc = vector_results['documents'][0][i]
                meta = vector_results['metadatas'][0][i]
                doc_id = vector_results['ids'][0][i]
                vector_docs.append((doc, meta, doc_id))
                
        # 2. Keyword Search
        import re
        keywords = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
        keyword_docs = []
        if keywords:
            try:
                data = _COLLECTION.get(include=["documents", "metadatas"])
                if data and data.get("documents"):
                    scored = []
                    for doc, meta, doc_id in zip(data["documents"], data["metadatas"], data["ids"]):
                        doc_lower = doc.lower()
                        score = sum(doc_lower.count(kw) for kw in keywords)
                        if score > 0:
                            scored.append((score, doc, meta, doc_id))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    keyword_docs = [(doc, meta, doc_id) for _, doc, meta, doc_id in scored[:n_results * 2]]
            except Exception:
                pass
                
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_map = {}
        
        for rank, (doc, meta, doc_id) in enumerate(vector_docs):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
            doc_map[doc_id] = (doc, meta)
            
        for rank, (doc, meta, doc_id) in enumerate(keyword_docs):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
            doc_map[doc_id] = (doc, meta)
            
        if not rrf_scores:
            return f"No relevant code found for query: '{query}'"
            
        # Get top n_results after fusion
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:n_results]
        
        output = [f"### Semantic Search Results for '{query}' ###\n"]
        for i, doc_id in enumerate(sorted_ids):
            doc, meta = doc_map[doc_id]
            context_str = f" | {meta['context']}" if 'context' in meta else ""
            output.append(f"--- Snippet {i+1} | {meta['file']} (Lines {meta['start_line']}-{meta['end_line']}{context_str}) ---")
            output.append("```")
            output.append(doc)
            output.append("```\n")
            
        result = "\n".join(output)
        _SEARCH_CACHE[cache_key] = (now, result)
        
        if len(_SEARCH_CACHE) > 100:
            oldest_key = min(_SEARCH_CACHE, key=lambda k: _SEARCH_CACHE[k][0])
            del _SEARCH_CACHE[oldest_key]
        
        return result
    except Exception as e:
        return f"Error performing semantic search: {e}"
