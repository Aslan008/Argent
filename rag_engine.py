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
_GLOBAL_DB_CLIENT = None
_ACTIVE_KB_COLLECTIONS = []
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

        # Bound concurrency to what a local Ollama can take: firing one request
        # per chunk (or a hardcoded 10) can OOM a small GPU or trigger timeouts
        # on a weak box. Configurable via embedding_concurrency; never more
        # workers than there are texts.
        from config import get_embedding_concurrency
        workers = max(1, min(get_embedding_concurrency(), len(input)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fetch_embedding, i, text) for i, text in enumerate(input)]
            for future in as_completed(futures):
                idx, emb = future.result()
                embeddings[idx] = emb

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.__call__([query])[0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self.__call__(documents)

def init_external_kbs():
    """Initializes external knowledge bases from the global config."""
    global _GLOBAL_DB_CLIENT, _ACTIVE_KB_COLLECTIONS
    try:
        import chromadb
    except ImportError:
        return "ERROR: 'chromadb' is not installed."

    try:
        from config import get_embedding_provider, get_ollama_embedding_model, get_external_kbs
        kbs = get_external_kbs()
        enabled_kbs = [kb for kb in kbs if kb.get("enabled", True)]
        
        if enabled_kbs:
            embedding_provider = get_embedding_provider()
            if embedding_provider == "ollama":
                ollama_model = get_ollama_embedding_model()
                ef = OllamaEmbeddingFunction(model_name=ollama_model)
            else:
                try:
                    from chromadb.utils import embedding_functions
                    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                except ImportError:
                    return "ERROR: 'sentence-transformers' is not installed."

            global_db_path = Path.home() / ".argent_coder_kbs"
            global_db_path.mkdir(parents=True, exist_ok=True)
            _GLOBAL_DB_CLIENT = chromadb.PersistentClient(path=str(global_db_path))
            _ACTIVE_KB_COLLECTIONS = []
            
            for kb in enabled_kbs:
                try:
                    col = _GLOBAL_DB_CLIENT.get_collection(name=f"kb_{kb['id']}", embedding_function=ef)
                    _ACTIVE_KB_COLLECTIONS.append(col)
                    print(f"[INFO] Loaded External Knowledge Base: {kb['name']}")
                except Exception:
                    print(f"[WARN] External KB '{kb['name']}' is enabled but not indexed. Use /kb index {kb['id']} to index it.")
            return "Successfully loaded external Knowledge Bases."
        return "No external Knowledge Bases to load."
    except Exception as e:
        import traceback
        return f"Failed to load external KBs: {e}\n{traceback.format_exc()}"

def is_rag_enabled() -> bool:
    """Check if the RAG module or any external KB is currently active."""
    global _RAG_ENABLED, _ACTIVE_KB_COLLECTIONS
    return _RAG_ENABLED or len(_ACTIVE_KB_COLLECTIONS) > 0

def disable_rag():
    """Disables RAG and clears the collection reference (keeps KBs if loaded)."""
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


def _read_pdf(file_path) -> str | None:
    """Extract text from a PDF via pypdf. Returns None if no parser is available
    or the file can't be read, so binary garbage is never indexed as text."""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            print(f"[WARN] Skipping PDF '{file_path}': install 'pypdf' to index PDFs.")
            return None
    try:
        reader = PdfReader(str(file_path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return text if text.strip() else None
    except Exception as e:
        print(f"[WARN] Could not read PDF '{file_path}': {e}")
        return None


def index_external_kb(kb_dict: dict) -> str:
    """Indexes an external knowledge base directory into the global ChromaDB."""
    global _GLOBAL_DB_CLIENT, _ACTIVE_KB_COLLECTIONS
    try:
        import chromadb
    except ImportError:
        return "ERROR: 'chromadb' is not installed. Please run `pip install chromadb`."
        
    try:
        from config import get_embedding_provider, get_ollama_embedding_model
        kb_path = Path(kb_dict["path"]).expanduser().resolve()
        if not kb_path.exists():
            return f"Error: Knowledge Base path '{kb_path}' does not exist."
            
        global_db_path = Path.home() / ".argent_coder_kbs"
        global_db_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(global_db_path))
        
        embedding_provider = get_embedding_provider()
        if embedding_provider == "ollama":
            ollama_model = get_ollama_embedding_model()
            ef = OllamaEmbeddingFunction(model_name=ollama_model)
        else:
            try:
                from chromadb.utils import embedding_functions
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            except ImportError:
                return "ERROR: 'sentence-transformers' is not installed."
                
        col_name = f"kb_{kb_dict['id']}"
        try:
            client.delete_collection(name=col_name)
        except Exception:
            pass
            
        collection = client.create_collection(name=col_name, embedding_function=ef)
        
        print(f"[INFO] Indexing Knowledge Base at {kb_path}...")
        docs = []
        ids = []
        metadatas = []
        doc_id_counter = 0
        
        for root, dirs, files in os.walk(kb_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in {".md", ".txt", ".html", ".htm", ".pdf", ".json", ".csv"}:
                    try:
                        if file_path.suffix.lower() == ".pdf":
                            source = _read_pdf(file_path)
                            if source is None:
                                continue  # no parser / unreadable — skip, don't index garbage
                        else:
                            with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
                                source = f.read()

                        rel = str(file_path.relative_to(kb_path))

                        if file_path.suffix.lower() in {".html", ".htm"}:
                            # Unity docs get a symbol-aware cleaner (strips the
                            # nav/feedback boilerplate and prepends the API
                            # symbol); other HTML falls back to plain text.
                            try:
                                from src.rag.unity_docs import is_unity_doc, chunk_unity_doc
                                if is_unity_doc(source):
                                    unity_chunks = chunk_unity_doc(source, rel)
                                    if unity_chunks:
                                        file_docs = [c for c, _ in unity_chunks]
                                        file_metas = [m for _, m in unity_chunks]
                                        for d, m in zip(file_docs, file_metas):
                                            docs.append(d)
                                            m["source_type"] = "external_kb"
                                            metadatas.append(m)
                                            import hashlib
                                            chunk_id = hashlib.md5(f"{rel}_{m.get('start_line', doc_id_counter)}".encode()).hexdigest()
                                            ids.append(f"kb_{chunk_id}_{doc_id_counter}")
                                            doc_id_counter += 1
                                        continue  # done with this file
                            except Exception as e:
                                log.debug("Unity-doc chunking failed for %s; falling back: %s", rel, e)
                            try:
                                from bs4 import BeautifulSoup
                                soup = BeautifulSoup(source, "html.parser")
                                source = soup.get_text(separator="\n", strip=True)
                            except ImportError:
                                pass

                        file_docs, file_metas = _chunk_heuristic(source, rel)
                        for d, m in zip(file_docs, file_metas):
                            docs.append(d)
                            m["source_type"] = "external_kb"
                            metadatas.append(m)
                            import hashlib
                            chunk_id = hashlib.md5(f"{rel}_{m.get('start_line', doc_id_counter)}".encode()).hexdigest()
                            ids.append(f"kb_{chunk_id}_{doc_id_counter}")
                            doc_id_counter += 1
                    except Exception as e:
                        log.warning("Skipping KB file %s: %s", file_path, e)
                        
        if docs:
            print(f"[INFO] Uploading {len(docs)} chunks to ChromaDB...")
            batch_size = 5000
            for i in range(0, len(docs), batch_size):
                collection.upsert(documents=docs[i:i+batch_size], metadatas=metadatas[i:i+batch_size], ids=ids[i:i+batch_size])
        
        try:
            from src.rag.keyword_index import clear_cache
            clear_cache()
        except Exception:
            pass
        msg = f"Successfully indexed Knowledge Base '{kb_dict['name']}' ({len(docs)} chunks)."
        # For documentation-heavy KBs, MiniLM is weaker than a doc-tuned embedder.
        if embedding_provider != "ollama":
            msg += ("\nTip: for technical documentation, nomic-embed-text via Ollama retrieves more "
                    "accurately than the default MiniLM. Switch with /rag_provider, then re-run /kb index.")
        return msg
    except Exception as e:
        import traceback
        return f"Failed to index Knowledge Base: {e}\n{traceback.format_exc()}"


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
                except Exception as e:
                    log.warning("Skipping file %s during indexing: %s", rel, e)
                    
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
            try:
                from src.rag.keyword_index import clear_cache
                clear_cache()
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Failed to update RAG for {file_path}: {e}")

def semantic_search(query: str, n_results: int = 5, target_kb: str = "all") -> str:
    """Tool for the LLM to search the vector database for code snippets using Hybrid Search + RRF."""
    global _RAG_ENABLED, _COLLECTION, _ACTIVE_KB_COLLECTIONS
    
    if not _RAG_ENABLED and len(_ACTIVE_KB_COLLECTIONS) == 0:
        return "Error: RAG and Knowledge Bases are disabled. Cannot perform semantic search."
        
    collections_to_search = []
    if target_kb == "all":
        if _COLLECTION is not None:
            collections_to_search.append(_COLLECTION)
        collections_to_search.extend(_ACTIVE_KB_COLLECTIONS)
    elif target_kb == "local":
        if _COLLECTION is not None:
            collections_to_search.append(_COLLECTION)
    else:
        for kb_col in _ACTIVE_KB_COLLECTIONS:
            if kb_col.name == f"kb_{target_kb}":
                collections_to_search.append(kb_col)
                break
                
    if not collections_to_search:
        return f"Error: No active collections found for target '{target_kb}'."
    
    cache_key = hashlib.md5(f"{query}:{n_results}:{target_kb}".encode()).hexdigest()
    now = time.time()
    if cache_key in _SEARCH_CACHE:
        cached_ts, cached_result = _SEARCH_CACHE[cache_key]
        if now - cached_ts < _SEARCH_CACHE_TTL:
            return cached_result
        
    try:
        import re
        keywords = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
        
        rrf_scores = {}
        doc_map = {}
        
        for col in collections_to_search:
            try:
                # 1. Semantic (Vector) search
                v_res = col.query(query_texts=[query], n_results=n_results * 2)
                if v_res.get('documents') and v_res['documents'][0]:
                    for rank, i in enumerate(range(len(v_res['documents'][0]))):
                        doc = v_res['documents'][0][i]
                        meta = v_res['metadatas'][0][i]
                        doc_id = v_res['ids'][0][i]
                        
                        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
                        doc_map[doc_id] = (doc, meta)
                        
                # 2. Keyword Search via a cached BM25 index (built once per
                # collection) instead of scanning the whole corpus each query.
                if keywords:
                    from src.rag.keyword_index import get_index
                    kidx = get_index(col)
                    for rank, (doc_id, doc, meta) in enumerate(kidx.search(query, n_results * 2)):
                        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
                        doc_map[doc_id] = (doc, meta)
            except Exception as e:
                log.debug("Keyword search fusion skipped (vector results still returned): %s", e)
            
        if not rrf_scores:
            return f"No relevant code found for query: '{query}'"
            
        # Get top n_results after fusion
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:n_results]
        
        output = [f"### Semantic Search Results for '{query}' ###\n"]
        for i, doc_id in enumerate(sorted_ids):
            doc, meta = doc_map[doc_id]
            context_str = f" | {meta['context']}" if 'context' in meta else ""
            start = meta.get('start_line', '?')
            end = meta.get('end_line', start)
            output.append(f"--- Snippet {i+1} | {meta['file']} (Lines {start}-{end}{context_str}) ---")
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
