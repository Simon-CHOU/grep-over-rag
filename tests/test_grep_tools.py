# tests/test_grep_tools.py
import os
import tempfile
from pathlib import Path
import pytest
from src.eval.grep_tools import rg_search, glob_files, read_file


@pytest.fixture
def sample_java_dir():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "com/example"
        src.mkdir(parents=True)
        (src / "OrderService.java").write_text(
            "package com.example;\n\n"
            "public class OrderService {\n"
            "    public Order createOrder(String name) {\n"
            "        return new Order(name);\n"
            "    }\n"
            "}\n"
        )
        (src / "UserService.java").write_text(
            "package com.example;\n\n"
            "public class UserService {\n"
            "    public User findByName(String name) {\n"
            "        return new User(name);\n"
            "    }\n"
            "}\n"
        )
        yield tmp


def test_rg_search_finds_class(sample_java_dir):
    results = rg_search("OrderService", sample_java_dir)
    assert len(results) >= 1
    assert any("OrderService.java" in r["file"] for r in results)
    assert all("file" in r and "line" in r and "content" in r for r in results)


def test_rg_search_no_match(sample_java_dir):
    results = rg_search("NonExistentClass", sample_java_dir)
    assert results == []


def test_rg_search_file_filter(sample_java_dir):
    results = rg_search("OrderService", sample_java_dir, file_pattern="*.java")
    assert len(results) >= 1


def test_glob_files_finds_java(sample_java_dir):
    results = glob_files("**/*.java", sample_java_dir)
    assert len(results) == 2
    assert all(f.endswith(".java") for f in results)


def test_read_file_content(sample_java_dir):
    file_path = os.path.join(sample_java_dir, "com/example/OrderService.java")
    content = read_file(file_path)
    assert "public class OrderService" in content
    assert "createOrder" in content


def test_read_file_with_line_range(sample_java_dir):
    file_path = os.path.join(sample_java_dir, "com/example/OrderService.java")
    content = read_file(file_path, start_line=1, end_line=2)
    lines = content.rstrip("\n").split("\n")
    assert len(lines) <= 2


def test_read_file_nonexistent():
    content = read_file("/nonexistent/file.java")
    assert content.startswith("Error:")
