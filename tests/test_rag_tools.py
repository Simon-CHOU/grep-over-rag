import os
import tempfile
from src.eval.rag_tools import vector_search, read_file_for_rag


class FakeIndexStore:
    def __init__(self, results=None):
        self._results = results or []

    def search(self, query_embedding, top_k=5):
        return self._results[:top_k]


class FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 128 for _ in texts]


def test_vector_search_returns_results():
    store = FakeIndexStore([
        "File: Foo.java\nScore: 0.95\n```java\nclass Foo {}\n```",
        "File: Bar.java\nScore: 0.80\n```java\nclass Bar {}\n```",
    ])
    embedder = FakeEmbedder()
    results = vector_search("find Foo class", store, embedder)
    assert "Foo.java" in results
    assert "Bar.java" in results


def test_vector_search_empty():
    store = FakeIndexStore([])
    embedder = FakeEmbedder()
    results = vector_search("find nothing", store, embedder)
    assert results == "No results found."


def test_read_file_for_rag():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".java", delete=False) as f:
        f.write("public class Test { void method() {} }")
        fpath = f.name
    try:
        content = read_file_for_rag(fpath)
        assert "public class Test" in content
    finally:
        os.unlink(fpath)
