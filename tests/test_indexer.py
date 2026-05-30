# tests/test_indexer.py
import os
import tempfile
from pathlib import Path
import pytest
from src.eval.indexer import IndexBuilder, IndexStore


@pytest.fixture
def small_java_project():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "com/example"
        src.mkdir(parents=True)
        (src / "Hello.java").write_text("public class Hello { void greet() {} }")
        (src / "World.java").write_text("public class World { void run() {} }")
        yield tmp


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(hash(t) % 100)] * 128 for t in texts]


def test_index_builder_collects_files(small_java_project):
    builder = IndexBuilder(
        codebase_root=small_java_project,
        embedder=FakeEmbedder(),
    )
    chunks = builder.build_chunks()
    assert len(chunks) == 2
    assert all("file" in c and "content" in c and "embedding" in c for c in chunks)


def test_index_store_save_and_load(small_java_project):
    builder = IndexBuilder(
        codebase_root=small_java_project,
        embedder=FakeEmbedder(),
    )
    chunks = builder.build_chunks()

    with tempfile.TemporaryDirectory() as idx_dir:
        store = IndexStore(idx_dir)
        store.save(chunks)

        loaded = store.load()
        assert len(loaded["metadata"]) == 2
        assert loaded["metadata"][0]["file"] == chunks[0]["file"]


def test_index_store_search(small_java_project):
    builder = IndexBuilder(
        codebase_root=small_java_project,
        embedder=FakeEmbedder(),
    )
    chunks = builder.build_chunks()

    with tempfile.TemporaryDirectory() as idx_dir:
        store = IndexStore(idx_dir)
        store.save(chunks)

        results = store.search(query_embedding=chunks[0]["embedding"], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], str)
