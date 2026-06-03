# tests/test_indexer.py
import tempfile
from pathlib import Path
import pytest
from src.eval.indexer import IndexBuilder, IndexStore


@pytest.fixture
def small_java_project():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src" / "main" / "java" / "com" / "example"
        src.mkdir(parents=True)
        (src / "Hello.java").write_text(
            "package com.example;\n\n"
            "public class Hello {\n"
            "  public void greet() {\n"
            "    System.out.println(\"hi\");\n"
            "  }\n"
            "}\n"
        )
        (src / "World.java").write_text(
            "package com.example;\n\n"
            "public class World {\n"
            "  public void run() {\n"
            "    System.out.println(\"run\");\n"
            "  }\n"
            "}\n"
        )
        yield tmp


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(hash(t) % 100)] * 128 for t in texts]


def test_index_builder_produces_symbol_chunks(small_java_project):
    builder = IndexBuilder(
        codebase_root=small_java_project,
        embedder=FakeEmbedder(),
    )
    chunks = builder.build_chunks()
    # Each file: 1 class + 1 method = 2 chunks. 2 files = 4 chunks
    assert len(chunks) == 4
    symbols = {c["symbol"] for c in chunks}
    assert "Hello" in symbols
    assert "Hello.greet" in symbols
    assert "World" in symbols
    assert "World.run" in symbols


def test_index_builder_chunk_metadata(small_java_project):
    builder = IndexBuilder(
        codebase_root=small_java_project,
        embedder=FakeEmbedder(),
    )
    chunks = builder.build_chunks()
    for c in chunks:
        assert "file" in c
        assert "symbol" in c
        assert "symbol_type" in c
        assert "start_line" in c
        assert "end_line" in c
        assert "content" in c
        assert "embedding" in c


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
        assert len(loaded["metadata"]) == 4
        for meta in loaded["metadata"]:
            assert "file" in meta
            assert "symbol" in meta
            assert "start_line" in meta


def test_index_store_search_format(small_java_project):
    builder = IndexBuilder(
        codebase_root=small_java_project,
        embedder=FakeEmbedder(),
    )
    chunks = builder.build_chunks()

    with tempfile.TemporaryDirectory() as idx_dir:
        store = IndexStore(idx_dir)
        store.save(chunks)

        results = store.search(query_embedding=chunks[0]["embedding"], top_k=2)
        assert len(results) >= 1
        for r in results:
            assert "File:" in r
            assert "Symbol:" in r
            assert "Lines:" in r
