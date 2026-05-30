import os
import json
import numpy as np
from pathlib import Path

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class IndexBuilder:
    """Collects Java files from codebase and builds embeddings for each."""

    def __init__(self, codebase_root: str, embedder):
        self.codebase_root = codebase_root
        self.embedder = embedder

    def build_chunks(self) -> list[dict]:
        """Walk Java files, create chunks, embed them."""
        chunks = []
        java_files = []
        for root, _, files in os.walk(self.codebase_root):
            for f in files:
                if f.endswith(".java"):
                    java_files.append(os.path.join(root, f))

        texts = []
        for fpath in sorted(java_files):
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                content = ""
            rel_path = os.path.relpath(fpath, self.codebase_root)
            texts.append(content)
            chunks.append({"file": rel_path, "content": content})

        if not chunks:
            return []

        embeddings = self.embedder.embed(texts)
        for i, emb in enumerate(embeddings):
            chunks[i]["embedding"] = emb

        return chunks


class IndexStore:
    """Save/load a FAISS index plus metadata."""

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def save(self, chunks: list[dict]):
        embeddings = np.array([c["embedding"] for c in chunks], dtype="float32")
        metadata = [{"file": c["file"], "content": c["content"][:2000]} for c in chunks]

        if HAS_FAISS:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            faiss.write_index(index, str(self.index_dir / "index.faiss"))
        else:
            np.save(self.index_dir / "embeddings.npy", embeddings)

        with open(self.index_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

    def load(self) -> dict:
        with open(self.index_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if HAS_FAISS and (self.index_dir / "index.faiss").exists():
            index = faiss.read_index(str(self.index_dir / "index.faiss"))
            return {"index": index, "metadata": metadata}
        else:
            embeddings = np.load(self.index_dir / "embeddings.npy")
            return {"embeddings": embeddings, "metadata": metadata}

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[str]:
        """Return top-k results as formatted strings with file + snippet."""
        data = self.load()
        metadata = data["metadata"]
        q = np.array([query_embedding], dtype="float32")

        if "index" in data:
            faiss.normalize_L2(q)
            scores, indices = data["index"].search(q, min(top_k, len(metadata)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(metadata):
                    meta = metadata[idx]
                    results.append(
                        f"File: {meta['file']}\nScore: {score:.4f}\n```java\n{meta['content'][:1000]}\n```"
                    )
            return results
        else:
            embeddings = data["embeddings"]
            similarity = np.dot(embeddings, q.T).flatten()
            top_indices = np.argsort(similarity)[-top_k:][::-1]
            results = []
            for idx in top_indices:
                if idx < len(metadata):
                    meta = metadata[idx]
                    results.append(
                        f"File: {meta['file']}\nScore: {similarity[idx]:.4f}\n```java\n{meta['content'][:1000]}\n```"
                    )
            return results


if __name__ == "__main__":
    import argparse
    from src.eval.embedder import DeepSeekEmbedder

    parser = argparse.ArgumentParser(description="Build RAG vector index")
    parser.add_argument("--codebase", default="codebases/apollo")
    parser.add_argument("--index-dir", default="data/index")
    args = parser.parse_args()

    embedder = DeepSeekEmbedder()
    builder = IndexBuilder(codebase_root=args.codebase, embedder=embedder)
    chunks = builder.build_chunks()
    print(f"Embedded {len(chunks)} files")

    store = IndexStore(args.index_dir)
    store.save(chunks)
    print(f"Index saved to {args.index_dir}")
