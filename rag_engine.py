import os
from pathlib import Path
from logger import get_logger

log = get_logger("rag")

# IMPORTANT: We DO NOT import chromadb here at the top level!
# It will be imported inside functions (lazy loading) to prevent slow startup times.

_RAG_ENABLED = False
_COLLECTION = None

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
    
    # Lazy Import checks
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        return "ERROR: 'chromadb' is not installed. Please run `pip install chromadb` to use RAG."
        
    try:
        project_path = Path(project_dir).expanduser().resolve()
        if not project_path.exists():
            return f"Error: Project directory {project_dir} does not exist."
            
        print(f"\n[INFO] Initializing Vector Database for {project_path.name}...")
        
        # We store the local DB inside the .argent folder of the user's project
        db_path = project_path / ".argent" / "chroma_db"
        db_path.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(path=str(db_path))
        
        # Use a lightweight embedding function suitable for code (MiniLM is standard)
        sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        _COLLECTION = client.get_or_create_collection(name="project_codebase", embedding_function=sentence_transformer_ef)
        
        # Now we index the codebase
        _index_codebase(project_path, _COLLECTION)
        
        _RAG_ENABLED = True
        return f"Successfully enabled RAG for '{project_path.name}'. Indexed files and ready for /search."
        
    except Exception as e:
        import traceback
        _RAG_ENABLED = False
        return f"Failed to enable RAG: {e}\n{traceback.format_exc()}"

def _chunk_text(text: str, file_rel_path: str) -> list:
    """Chunks text logically (by blocks if possible, otherwise by lines)."""
    docs = []
    metadatas = []
    
    lines = text.splitlines()
    if not lines:
        return docs, metadatas
        
    # Smart chunking: we look for 'class ' or 'def ' to start a new chunk in Python/CS/JS
    # to avoid breaking functions in the middle.
    chunks = []
    current_chunk = []
    start_line = 1
    
    for i, line in enumerate(lines):
        # Heuristic: if line starts with code definition and current chunk is sizeable, break.
        if (line.startswith(("class ", "def ", "function ", "export ", "public class", "private class"))) and len(current_chunk) > 20:
            chunks.append(("\n".join(current_chunk), start_line, i))
            current_chunk = [line]
            start_line = i + 1
        else:
            current_chunk.append(line)
            # Hard limit at 60 lines to avoid massive chunks
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
            "end_line": e_line
        })
        
    return docs, metadatas

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
                        ids.append(f"doc_{doc_id_counter}")
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
    """Tool for the LLM to search the vector database for code snippets."""
    global _RAG_ENABLED, _COLLECTION
    
    if not _RAG_ENABLED or _COLLECTION is None:
        return "Error: RAG is not enabled. Cannot perform semantic search."
        
    try:
        results = _COLLECTION.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return f"No relevant code found for query: '{query}'"
            
        output = [f"### Semantic Search Results for '{query}' ###\n"]
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            output.append(f"--- Snippet {i+1} | {meta['file']} (Lines {meta['start_line']}-{meta['end_line']}) ---")
            output.append("```")
            output.append(doc)
            output.append("```\n")
            
        return "\n".join(output)
    except Exception as e:
        return f"Error performing semantic search: {e}"
