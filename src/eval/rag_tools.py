from src.eval.grep_tools import read_file as _read_file


def vector_search(query: str, index_store, embedder) -> str:
    """Search the vector index using an embedding of the query."""
    try:
        query_emb = embedder.embed([query])[0]
        results = index_store.search(query_emb, top_k=5)
        if not results:
            return "No results found."
        return "\n---\n".join(results)
    except Exception as e:
        return f"Error during vector search: {e}"


def read_file_for_rag(file_path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read file content (same as grep tool, re-exported for RAG agent)."""
    return _read_file(file_path, start_line, end_line)
