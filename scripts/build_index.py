"""Build the RAG vector index for Apollo codebase."""
import argparse
import sys
sys.path.insert(0, "src")

from src.eval.indexer import IndexBuilder, IndexStore
from src.eval.embedder import DeepSeekEmbedder

parser = argparse.ArgumentParser()
parser.add_argument("--codebase", default="codebases/apollo")
parser.add_argument("--index-dir", default="data/index")
args = parser.parse_args()

print(f"Building index from {args.codebase} ...")
embedder = DeepSeekEmbedder()
builder = IndexBuilder(codebase_root=args.codebase, embedder=embedder)
chunks = builder.build_chunks()
print(f"  Embedded {len(chunks)} files")

store = IndexStore(args.index_dir)
store.save(chunks)
print(f"  Index saved to {args.index_dir}")
print("Done.")
