# RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the RAG agent pipeline (symbol-level chunking + DashScope embedding + FAISS index) so it can run the 50-task Apollo eval and produce results comparable to the Grep agent.

**Architecture:** Java source files are parsed into symbol-level chunks (class/method/interface/enum) via regex. Each chunk is embedded via DashScope's `text-embedding-v4` API and stored in a FAISS index. At runtime, the RAG agent queries the index with `vector_search`, reads files for precision, and answers via `deepseek-v4-flash`.

**Tech Stack:** Python 3.11, FAISS, OpenAI SDK (shared by DeepSeek chat + DashScope embedding), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-06-04-rag-pipeline-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/eval/chunker.py` | CREATE | Symbol-level Java chunker: `Chunk` dataclass, `_strip_comments`, `find_matching_brace`, `chunk_file` |
| `src/eval/embedder.py` | MODIFY | Add `DashScopeEmbedder` class (replaces `DeepSeekEmbedder` in runner) |
| `src/eval/indexer.py` | MODIFY | Use `SymbolChunker` in `build_chunks()`, update metadata and search format |
| `src/eval/rag_tools.py` | MODIFY | No code change needed — `vector_search` delegates to `IndexStore.search` which is updated in indexer |
| `src/eval/rag_agent.py` | MODIFY | Fix `_parse_answer` to strip markdown artifacts |
| `src/eval/runner.py` | MODIFY | Import `DashScopeEmbedder` instead of `DeepSeekEmbedder` |
| `tests/test_chunker.py` | CREATE | Unit tests for chunker |
| `tests/test_embedder.py` | MODIFY | Add DashScope embedder tests |
| `tests/test_indexer.py` | MODIFY | Update assertions for symbol-level chunks |
| `tests/test_rag_agent.py` | MODIFY | Add markdown-artifact parser tests |

---

### Task 1: SymbolChunker — Data Structures and Utilities

**Files:**
- Create: `src/eval/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunker.py
import pytest
from src.eval.chunker import Chunk, _strip_comments, find_matching_brace


class TestStripComments:
    def test_strips_block_comment(self):
        source = "public class Foo {\n  /* comment */\n  void bar() {}\n}"
        result = _strip_comments(source)
        assert "comment" not in result
        assert "void bar()" in result

    def test_strips_line_comment(self):
        source = "public class Foo {\n  // line comment\n  void bar() {}\n}"
        result = _strip_comments(source)
        assert "line comment" not in result
        assert "void bar()" in result

    def test_preserves_line_numbers(self):
        source = "line1\n/* block\ncomment */\nline4"
        result = _strip_comments(source)
        lines = result.split("\n")
        assert len(lines) == 4
        assert "line1" in lines[0]
        assert "line4" in lines[3]

    def test_handles_multiline_javadoc(self):
        source = "/**\n * Javadoc\n * @param foo bar\n */\npublic class Foo {}"
        result = _strip_comments(source)
        assert "Javadoc" not in result
        assert "public class Foo" in result


class TestFindMatchingBrace:
    def test_simple_braces(self):
        lines = ["public class Foo {", "  void bar() {}", "}"]
        source = "\n".join(lines)
        end = find_matching_brace(source, 0)
        assert end == 2

    def test_nested_braces(self):
        lines = ["{", "  {", "    {}", "  }", "}"]
        source = "\n".join(lines)
        end = find_matching_brace(source, 0)
        assert end == 4

    def test_skips_braces_in_strings(self):
        lines = ['{', '  String s = "{";', '}']
        source = "\n".join(lines)
        end = find_matching_brace(source, 0)
        assert end == 2

    def test_skips_braces_in_comments(self):
        lines = ["{", "  // {", "  /* } */", "}"]
        source = "\n".join(lines)
        end = find_matching_brace(source, 0)
        assert end == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.chunker'`

- [ ] **Step 3: Implement dataclass and utility functions**

```python
# src/eval/chunker.py
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    file: str
    symbol: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str


# Match class/interface/enum declarations
CLASS_RE = re.compile(
    r"^\s*"
    r"(?:public\s+|private\s+|protected\s+)?"
    r"(?:static\s+)?(?:final\s+)?(?:abstract\s+)?"
    r"(?:class|interface|enum)\s+(\w+)"
)

# Match method/constructor declarations
METHOD_RE = re.compile(
    r"^\s*"
    r"(?:public|private|protected)\s+"
    r"(?:(?:static|final|abstract|synchronized|native|default)\s+)*"
    r"(?:<[^>]+>\s+)?"
    r"(?:\S+ )+"
    r"(\w+)\s*\("
)


def _strip_comments(source: str) -> str:
    """Strip Java comments while preserving line structure."""
    result = []
    i = 0
    in_block_comment = False
    in_line_comment = False

    while i < len(source):
        if in_block_comment:
            if source[i:i+2] == "*/":
                result.append("  ")
                i += 2
                in_block_comment = False
            else:
                result.append(" " if source[i] != "\n" else "\n")
                i += 1
        elif in_line_comment:
            if source[i] == "\n":
                result.append("\n")
                in_line_comment = False
            else:
                result.append(" ")
            i += 1
        elif source[i:i+2] == "/*":
            result.append("  ")
            i += 2
            in_block_comment = True
        elif source[i:i+2] == "//":
            result.append("  ")
            i += 2
            in_line_comment = True
        else:
            result.append(source[i])
            i += 1

    return "".join(result)


def find_matching_brace(source: str, start_line_idx: int) -> int:
    """Find the line index of the closing } matching the first { from start_line_idx.
    Skips braces in comments, strings, and char literals."""
    lines = source.split("\n")
    depth = 0
    in_block = False
    in_line = False
    in_str = False
    in_chr = False

    for line_idx in range(start_line_idx, len(lines)):
        line = lines[line_idx]
        in_line = False
        j = 0
        while j < len(line):
            if in_block:
                if j + 1 < len(line) and line[j:j+2] == "*/":
                    in_block = False
                    j += 2
                    continue
                j += 1
                continue
            if in_line:
                j += 1
                continue

            c = line[j]
            two = line[j:j+2] if j + 1 < len(line) else ""

            if two == "/*":
                in_block = True
                j += 2
                continue
            if two == "//":
                in_line = True
                j += 2
                continue

            if in_str:
                if c == "\\" and j + 1 < len(line):
                    j += 2
                    continue
                if c == '"':
                    in_str = False
                j += 1
                continue

            if in_chr:
                if c == "\\" and j + 1 < len(line):
                    j += 2
                    continue
                if c == "'":
                    in_chr = False
                j += 1
                continue

            if c == '"':
                in_str = True
                j += 1
                continue
            if c == "'":
                in_chr = True
                j += 1
                continue
            if c == "{":
                depth += 1
                j += 1
                continue
            if c == "}":
                depth -= 1
                if depth == 0:
                    return line_idx
                j += 1
                continue

            j += 1

    return len(lines) - 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/chunker.py tests/test_chunker.py
git commit -m "feat: add chunker dataclass, comment stripper, and brace matcher"
```

---

### Task 2: SymbolChunker — `chunk_file` Function

**Files:**
- Modify: `src/eval/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_chunker.py`:

```python
from src.eval.chunker import chunk_file


class TestChunkFile:
    def test_simple_class_with_methods(self):
        source = (
            "package com.example;\n"
            "\n"
            "public class HelloService {\n"
            "\n"
            "  public String greet(String name) {\n"
            "    return \"Hello \" + name;\n"
            "  }\n"
            "\n"
            "  public void run() {\n"
            "    System.out.println(\"running\");\n"
            "  }\n"
            "}\n"
        )
        chunks = chunk_file(source, "com/example/HelloService.java")
        symbols = {c.symbol: c for c in chunks}
        # Should have class + 2 methods = 3 chunks
        assert len(chunks) == 3
        assert "HelloService" in symbols
        assert symbols["HelloService"].symbol_type == "class"
        assert symbols["HelloService"].start_line == 3
        assert "HelloService.greet" in symbols
        assert symbols["HelloService.greet"].symbol_type == "method"
        assert symbols["HelloService.greet"].start_line == 5
        assert "HelloService.run" in symbols
        assert symbols["HelloService.run"].symbol_type == "method"
        assert symbols["HelloService.run"].start_line == 9

    def test_interface(self):
        source = (
            "public interface FooService {\n"
            "  void doSomething();\n"
            "}\n"
        )
        chunks = chunk_file(source, "FooService.java")
        symbols = {c.symbol: c for c in chunks}
        assert "FooService" in symbols
        assert symbols["FooService"].symbol_type == "interface"

    def test_enum_with_body(self):
        source = (
            "public enum Status {\n"
            "  ACTIVE, INACTIVE;\n"
            "\n"
            "  public String label() {\n"
            "    return name().toLowerCase();\n"
            "  }\n"
            "}\n"
        )
        chunks = chunk_file(source, "Status.java")
        symbols = {c.symbol: c for c in chunks}
        assert "Status" in symbols
        assert symbols["Status"].symbol_type == "enum"
        assert "Status.label" in symbols

    def test_inner_static_class(self):
        source = (
            "public class Outer {\n"
            "\n"
            "  public void outerMethod() {\n"
            "  }\n"
            "\n"
            "  public static class Inner {\n"
            "    public void innerMethod() {\n"
            "    }\n"
            "  }\n"
            "}\n"
        )
        chunks = chunk_file(source, "Outer.java")
        symbols = {c.symbol: c for c in chunks}
        assert "Outer" in symbols
        assert "Outer.outerMethod" in symbols
        assert "Outer.Inner" in symbols
        assert symbols["Outer.Inner"].symbol_type == "class"
        assert "Outer.Inner.innerMethod" in symbols

    def test_skips_comments_with_braces(self):
        source = (
            "/**\n"
            " * This class has { braces } in javadoc.\n"
            " */\n"
            "public class DocClass {\n"
            "  // method with { in comment\n"
            "  public void real() {\n"
            "  }\n"
            "}\n"
        )
        chunks = chunk_file(source, "DocClass.java")
        symbols = {c.symbol: c for c in chunks}
        assert "DocClass" in symbols
        assert "DocClass.real" in symbols

    def test_constructor(self):
        source = (
            "public class Foo {\n"
            "  private String name;\n"
            "\n"
            "  public Foo(String name) {\n"
            "    this.name = name;\n"
            "  }\n"
            "}\n"
        )
        chunks = chunk_file(source, "Foo.java")
        symbols = {c.symbol: c for c in chunks}
        assert "Foo" in symbols
        assert "Foo.Foo" in symbols
        assert symbols["Foo.Foo"].symbol_type == "method"

    def test_annotation_type(self):
        source = (
            "public @interface MyAnnotation {\n"
            "  String value();\n"
            "}\n"
        )
        chunks = chunk_file(source, "MyAnnotation.java")
        symbols = {c.symbol: c for c in chunks}
        assert "MyAnnotation" in symbols
        assert symbols["MyAnnotation"].symbol_type in ("interface", "class")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunker.py::TestChunkFile -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_file'`

- [ ] **Step 3: Implement `chunk_file`**

Append to `src/eval/chunker.py`:

```python
def _detect_brace_depth_at_line(source: str, target_line_idx: int) -> int:
    """Calculate brace depth at the start of target_line_idx by counting
    braces from the beginning of the file (skipping comments/strings)."""
    lines = source.split("\n")
    depth = 0
    in_block = False
    in_line = False
    in_str = False
    in_chr = False

    for line_idx in range(target_line_idx):
        line = lines[line_idx]
        in_line = False
        j = 0
        while j < len(line):
            if in_block:
                if j + 1 < len(line) and line[j:j+2] == "*/":
                    in_block = False
                    j += 2
                    continue
                j += 1
                continue
            if in_line:
                j += 1
                continue

            c = line[j]
            two = line[j:j+2] if j + 1 < len(line) else ""

            if two == "/*":
                in_block = True
                j += 2
                continue
            if two == "//":
                in_line = True
                j += 2
                continue
            if in_str:
                if c == "\\" and j + 1 < len(line):
                    j += 2
                    continue
                if c == '"':
                    in_str = False
                j += 1
                continue
            if in_chr:
                if c == "\\" and j + 1 < len(line):
                    j += 2
                    continue
                if c == "'":
                    in_chr = False
                j += 1
                continue
            if c == '"':
                in_str = True
                j += 1
                continue
            if c == "'":
                in_chr = True
                j += 1
                continue
            if c == "{":
                depth += 1
                j += 1
                continue
            if c == "}":
                depth -= 1
                j += 1
                continue
            j += 1

    return depth


def chunk_file(source: str, file_path: str) -> list[Chunk]:
    """Parse a Java source file and extract symbol-level chunks."""
    lines = source.split("\n")
    stripped = _strip_comments(source)
    stripped_lines = stripped.split("\n")
    chunks = []

    for i, sline in enumerate(stripped_lines):
        # Top-level class/interface/enum (depth 0)
        m = CLASS_RE.match(sline)
        if m:
            depth = _detect_brace_depth_at_line(source, i)
            if depth == 0:
                class_name = m.group(1)
                # Detect type from original line (preserves @interface)
                orig = lines[i]
                if "interface" in orig and "@interface" not in orig:
                    stype = "interface"
                elif "@interface" in orig:
                    stype = "interface"
                elif "enum" in orig:
                    stype = "enum"
                else:
                    stype = "class"
                # Find opening { on this or following line
                body_start = i
                while body_start < len(stripped_lines) and "{" not in stripped_lines[body_start]:
                    body_start += 1
                if body_start < len(stripped_lines):
                    end = find_matching_brace(source, body_start)
                    content = "\n".join(lines[i:end+1])
                    chunks.append(Chunk(
                        file=file_path, symbol=class_name,
                        symbol_type=stype, start_line=i+1, end_line=end+1,
                        content=content,
                    ))
                    # Extract methods inside this class
                    for j in range(i+1, end):
                        mm = METHOD_RE.match(stripped_lines[j])
                        if mm:
                            method_name = mm.group(1)
                            mbody_start = j
                            while mbody_start < len(stripped_lines) and "{" not in stripped_lines[mbody_start]:
                                mbody_start += 1
                            if mbody_start < len(stripped_lines):
                                mend = find_matching_brace(source, mbody_start)
                                mcontent = "\n".join(lines[j:mend+1])
                                chunks.append(Chunk(
                                    file=file_path,
                                    symbol=f"{class_name}.{method_name}",
                                    symbol_type="method",
                                    start_line=j+1, end_line=mend+1,
                                    content=mcontent,
                                ))
        else:
            # Inner class/interface/enum (depth > 0)
            depth = _detect_brace_depth_at_line(source, i)
            if depth >= 1:
                im = CLASS_RE.match(sline)
                if im:
                    inner_name = im.group(1)
                    # Find parent class name from enclosing chunk
                    parent = ""
                    for c in chunks:
                        if c.symbol_type in ("class", "interface", "enum") and "." not in c.symbol:
                            if c.start_line <= i+1 <= c.end_line:
                                parent = c.symbol
                    full_name = f"{parent}.{inner_name}" if parent else inner_name
                    orig = lines[i]
                    if "interface" in orig:
                        istype = "interface"
                    elif "enum" in orig:
                        istype = "enum"
                    else:
                        istype = "class"
                    body_start = i
                    while body_start < len(stripped_lines) and "{" not in stripped_lines[body_start]:
                        body_start += 1
                    if body_start < len(stripped_lines):
                        end = find_matching_brace(source, body_start)
                        content = "\n".join(lines[i:end+1])
                        chunks.append(Chunk(
                            file=file_path, symbol=full_name,
                            symbol_type=istype, start_line=i+1, end_line=end+1,
                            content=content,
                        ))
                        # Extract methods inside inner class
                        for j in range(i+1, end):
                            mm = METHOD_RE.match(stripped_lines[j])
                            if mm:
                                method_name = mm.group(1)
                                mbody_start = j
                                while mbody_start < len(stripped_lines) and "{" not in stripped_lines[mbody_start]:
                                    mbody_start += 1
                                if mbody_start < len(stripped_lines):
                                    mend = find_matching_brace(source, mbody_start)
                                    mcontent = "\n".join(lines[j:mend+1])
                                    chunks.append(Chunk(
                                        file=file_path,
                                        symbol=f"{full_name}.{method_name}",
                                        symbol_type="method",
                                        start_line=j+1, end_line=mend+1,
                                        content=mcontent,
                                    ))

    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: All 15 tests PASS

- [ ] **Step 5: Manual verification on real Apollo file**

```bash
uv run python -c "
from src.eval.chunker import chunk_file
with open('codebases/apollo/apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/ReleaseService.java') as f:
    source = f.read()
chunks = chunk_file(source, 'apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/ReleaseService.java')
print(f'Total chunks: {len(chunks)}')
for c in chunks[:5]:
    print(f'  {c.symbol_type:10s} {c.symbol:50s} lines {c.start_line}-{c.end_line}')
print('  ...')
# Verify findOne method is captured
findOne = [c for c in chunks if 'findOne' in c.symbol]
print(f'findOne chunks: {len(findOne)}')
for c in findOne:
    print(f'  {c.symbol} line {c.start_line}')
"
```

Expected: Multiple chunks extracted, including `ReleaseService.findOne` around line 97-98.

- [ ] **Step 6: Commit**

```bash
git add src/eval/chunker.py tests/test_chunker.py
git commit -m "feat: add chunk_file function for symbol-level Java parsing"
```

---

### Task 3: DashScopeEmbedder

**Files:**
- Modify: `src/eval/embedder.py`
- Test: `tests/test_embedder.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_embedder.py
from unittest.mock import MagicMock, patch
from src.eval.embedder import DashScopeEmbedder


def test_dashscope_embedder_uses_env_vars():
    with patch.dict("os.environ", {
        "DASHSCOPE_API_KEY": "test-key",
        "DASHSCOPE_BASE_URL": "https://test.example.com/v1",
    }):
        embedder = DashScopeEmbedder()
        assert embedder.api_key == "test-key"
        assert embedder.base_url == "https://test.example.com/v1"
        assert embedder.model == "text-embedding-v4"


def test_dashscope_embedder_default_base_url():
    with patch.dict("os.environ", {"DASHSCOPE_API_KEY": "test-key"}, clear=False):
        import os
        os.environ.pop("DASHSCOPE_BASE_URL", None)
        embedder = DashScopeEmbedder()
        assert embedder.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_dashscope_embedder_batch():
    embedder = DashScopeEmbedder(api_key="fake-key", batch_size=2)
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in range(2)
    ]
    embedder._client = MagicMock()
    embedder._client.embeddings.create.return_value = mock_response

    results = embedder.embed(["text1", "text2", "text3"])
    assert len(results) == 3  # 2 in first batch (mocked), need second call
    assert embedder._client.embeddings.create.call_count == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedder.py -v`
Expected: FAIL — `ImportError: cannot import name 'DashScopeEmbedder'`

- [ ] **Step 3: Implement DashScopeEmbedder**

Replace the entire content of `src/eval/embedder.py` with:

```python
# src/eval/embedder.py
import os
import time
from openai import OpenAI


class DeepSeekEmbedder:
    """Embedding client for DeepSeek's embedding API (legacy — DeepSeek
    does not provide an embedding endpoint; kept for backward compat)."""

    def __init__(self, base_url=None, api_key=None, model=None, batch_size=20):
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model or "text-embedding-3-small"
        self.batch_size = batch_size
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in response.data])
            if i + self.batch_size < len(texts):
                time.sleep(0.1)
        return all_embeddings


class DashScopeEmbedder:
    """Embedding client for Alibaba DashScope (text-embedding-v4).
    Uses OpenAI-compatible API format."""

    def __init__(self, base_url=None, api_key=None, model=None, batch_size=6):
        self.base_url = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is required")
        self.model = model or "text-embedding-v4"
        self.batch_size = batch_size
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching if needed."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_results = [d.embedding for d in response.data]
            all_embeddings.extend(batch_results[:len(batch)])
            if i + self.batch_size < len(texts):
                time.sleep(0.2)  # DashScope rate limit courtesy
        return all_embeddings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedder.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/embedder.py tests/test_embedder.py
git commit -m "feat: add DashScopeEmbedder for text-embedding-v4"
```

---

### Task 4: Indexer Integration

**Files:**
- Modify: `src/eval/indexer.py`
- Modify: `tests/test_indexer.py`

- [ ] **Step 1: Update `test_indexer.py` for symbol-level chunks**

Replace `tests/test_indexer.py` entirely:

```python
# tests/test_indexer.py
import tempfile
from pathlib import Path
import pytest
from src.eval.indexer import IndexBuilder, IndexStore


@pytest.fixture
def small_java_project():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "com/example"
        src.mkdir(parents=True)
        (src / "Hello.java").write_text(
            "package com.example;\n\n"
            "public class Hello {\n"
            "  void greet() {\n"
            "    System.out.println(\"hi\");\n"
            "  }\n"
            "}\n"
        )
        (src / "World.java").write_text(
            "package com.example;\n\n"
            "public class World {\n"
            "  void run() {\n"
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
        # Metadata should have symbol info
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
        # Search results should include symbol and line info
        for r in results:
            assert "File:" in r
            assert "Symbol:" in r
            assert "Lines:" in r
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_indexer.py -v`
Expected: FAIL — chunks don't have `symbol`, `symbol_type`, etc.

- [ ] **Step 3: Update `indexer.py`**

Replace `src/eval/indexer.py` entirely:

```python
# src/eval/indexer.py
import os
import json
import numpy as np
from pathlib import Path
from src.eval.chunker import chunk_file

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class IndexBuilder:
    """Collects Java files from codebase, chunks them at symbol level,
    and builds embeddings for each chunk."""

    def __init__(self, codebase_root: str, embedder):
        self.codebase_root = codebase_root
        self.embedder = embedder

    def build_chunks(self) -> list[dict]:
        """Walk Java files, create symbol-level chunks, embed them."""
        java_files = []
        for root, _, files in os.walk(self.codebase_root):
            for f in files:
                if f.endswith(".java") and "src" in root and "main" in root and "test" not in root:
                    java_files.append(os.path.join(root, f))

        chunk_dicts = []
        for fpath in sorted(java_files):
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                continue
            rel_path = os.path.relpath(fpath, self.codebase_root).replace("\\", "/")
            chunks = chunk_file(content, rel_path)
            for c in chunks:
                chunk_dicts.append({
                    "file": c.file,
                    "symbol": c.symbol,
                    "symbol_type": c.symbol_type,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "content": c.content,
                })

        if not chunk_dicts:
            return []

        # Build embedding input: symbol name + type + code preview
        embed_texts = [
            f"{c['symbol']} ({c['symbol_type']})\n{c['content'][:500]}"
            for c in chunk_dicts
        ]
        embeddings = self.embedder.embed(embed_texts)
        for i, emb in enumerate(embeddings):
            chunk_dicts[i]["embedding"] = emb

        return chunk_dicts


class IndexStore:
    """Save/load a FAISS index plus metadata."""

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def save(self, chunks: list[dict]):
        embeddings = np.array([c["embedding"] for c in chunks], dtype="float32")
        metadata = [
            {
                "file": c["file"],
                "symbol": c["symbol"],
                "symbol_type": c["symbol_type"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "content": c["content"][:2000],
            }
            for c in chunks
        ]

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
        """Return top-k results as formatted strings with file, symbol, and line info."""
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
                        f"File: {meta['file']}\n"
                        f"Symbol: {meta['symbol']} ({meta['symbol_type']})\n"
                        f"Lines: {meta['start_line']}-{meta['end_line']}\n"
                        f"Score: {score:.4f}\n"
                        f"```java\n{meta['content'][:1000]}\n```"
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
                        f"File: {meta['file']}\n"
                        f"Symbol: {meta['symbol']} ({meta['symbol_type']})\n"
                        f"Lines: {meta['start_line']}-{meta['end_line']}\n"
                        f"Score: {similarity[idx]:.4f}\n"
                        f"```java\n{meta['content'][:1000]}\n```"
                    )
            return results


if __name__ == "__main__":
    import argparse
    from src.eval.embedder import DashScopeEmbedder

    parser = argparse.ArgumentParser(description="Build RAG vector index")
    parser.add_argument("--codebase", default="codebases/apollo")
    parser.add_argument("--index-dir", default="data/index")
    args = parser.parse_args()

    embedder = DashScopeEmbedder()
    builder = IndexBuilder(codebase_root=args.codebase, embedder=embedder)
    chunks = builder.build_chunks()
    print(f"Created {len(chunks)} symbol-level chunks from Java files")

    store = IndexStore(args.index_dir)
    store.save(chunks)
    print(f"Index saved to {args.index_dir}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_indexer.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/indexer.py tests/test_indexer.py
git commit -m "feat: integrate symbol-level chunker into indexer"
```

---

### Task 5: RAG Agent Parser Fix + Runner Update

**Files:**
- Modify: `src/eval/rag_agent.py:144-148`
- Modify: `src/eval/runner.py:181`
- Test: `tests/test_rag_agent.py`

- [ ] **Step 1: Add parser tests to `test_rag_agent.py`**

Append to `tests/test_rag_agent.py`:

```python
def test_rag_agent_strips_markdown_backticks():
    llm = make_mock_llm([
        ChatResult(content="`apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/ReleaseService.java:98`"),
    ])
    agent = RAGAgent(llm_client=llm, codebase_root="/tmp",
                     index_store=FakeIndexStore(), embedder=FakeEmbedder())
    result = agent.run(symbol="ReleaseService.findOne", symbol_type="method")
    assert result.file_path == "apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/ReleaseService.java"
    assert result.line_number == 98


def test_rag_agent_strips_bold_markdown():
    llm = make_mock_llm([
        ChatResult(content="**apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/ReleaseService.java:195**"),
    ])
    agent = RAGAgent(llm_client=llm, codebase_root="/tmp",
                     index_store=FakeIndexStore(), embedder=FakeEmbedder())
    result = agent.run(symbol="ReleaseService.publish", symbol_type="method")
    assert result.file_path == "apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/ReleaseService.java"
    assert result.line_number == 195


def test_rag_agent_strips_answer_prefix():
    llm = make_mock_llm([
        ChatResult(content="**Answer:** `apollo-portal/src/main/java/com/ctrip/framework/apollo/portal/service/NamespaceService.java:168`"),
    ])
    agent = RAGAgent(llm_client=llm, codebase_root="/tmp",
                     index_store=FakeIndexStore(), embedder=FakeEmbedder())
    result = agent.run(symbol="NamespaceService.deleteNamespace", symbol_type="method")
    assert result.file_path == "apollo-portal/src/main/java/com/ctrip/framework/apollo/portal/service/NamespaceService.java"
    assert result.line_number == 168
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rag_agent.py -v -k "strips"`
Expected: 3 FAIL — parser captures markdown artifacts

- [ ] **Step 3: Fix `_parse_answer` in `rag_agent.py`**

Replace lines 144-148 in `src/eval/rag_agent.py`:

```python
    def _parse_answer(self, text: str) -> tuple[str, int] | None:
        # Try strict path:line pattern first (Java source file)
        match = re.search(r"([\w\-./]+\.java):(\d+)", text)
        if match:
            path = match.group(1).strip().strip("`*'\"")
            return (path, int(match.group(2)))
        # Fallback: any path-like:line
        match = re.search(r"([\w\-./]+):(\d+)", text)
        if match:
            path = match.group(1).strip().strip("`*'\"")
            return (path, int(match.group(2)))
        return None
```

- [ ] **Step 4: Update `runner.py` to use `DashScopeEmbedder`**

In `src/eval/runner.py`, change line 181:

```python
    from src.eval.embedder import DashScopeEmbedder
```

And change lines 196-197:

```python
        embedder = DashScopeEmbedder()
```

- [ ] **Step 5: Run all RAG agent tests**

Run: `uv run pytest tests/test_rag_agent.py -v`
Expected: All 5 tests PASS (2 existing + 3 new)

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS, no regressions

- [ ] **Step 7: Commit**

```bash
git add src/eval/rag_agent.py src/eval/runner.py tests/test_rag_agent.py
git commit -m "fix: strip markdown artifacts in RAG agent parser, switch to DashScopeEmbedder"
```

---

### Task 6: Smoke Test (1 Task)

**Files:** No new files — integration test via CLI

- [ ] **Step 1: Verify `DASHSCOPE_API_KEY` is set**

```bash
echo "DASHSCOPE_API_KEY set: $([ -n \"$DASHSCOPE_API_KEY\" ] && echo yes || echo no)"
```

Expected: `yes`. If `no`, user must set the env var before proceeding.

- [ ] **Step 2: Build index on Apollo codebase**

```bash
uv run python -m src.eval.indexer \
  --codebase codebases/apollo \
  --index-dir data/index
```

Expected output:
```
Created ~3500 symbol-level chunks from Java files
Index saved to data/index
```

If embedding API errors, check `DASHSCOPE_API_KEY` and DashScope API quota.

- [ ] **Step 3: Create 1-task smoke test file**

```bash
echo '{"id":"task_001","symbol":"ApolloEurekaClientConfig","type":"class","file":"apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/eureka/ApolloEurekaClientConfig.java","expected_line":36,"difficulty":"easy","note":"Smoke test"}' > data/smoke_tasks.jsonl
```

- [ ] **Step 4: Run RAG agent smoke test**

```bash
uv run python -m src.eval.runner \
  --mode rag \
  --codebase codebases/apollo \
  --index-dir data/index \
  --tasks data/smoke_tasks.jsonl \
  --output-dir results/smoke_rag
```

Expected: 1/1 success, or at least a valid prediction (non-empty file path).

- [ ] **Step 5: Check results**

```bash
cat results/smoke_rag/rag_raw.csv
```

Expected: `success` column is `true`, `predicted_file` matches `expected_file`.

- [ ] **Step 6: Clean up and commit**

```bash
rm data/smoke_tasks.jsonl
git add -A
git commit -m "chore: smoke test RAG pipeline end-to-end"
```

---

### Task 7: Full Experiment

**Files:** No code changes — execution and reporting

- [ ] **Step 1: Run full RAG experiment (50 tasks)**

```bash
uv run python -m src.eval.runner \
  --mode rag \
  --codebase codebases/apollo \
  --index-dir data/index \
  --tasks data/tasks.jsonl \
  --output-dir results
```

Expected: 50 tasks complete, results in `results/rag_raw.csv` and `results/rag_aggregate.csv`.

- [ ] **Step 2: Compare Grep vs RAG results**

```bash
uv run python -c "
import csv

def summarize(path, mode):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    total = len(rows)
    success = sum(1 for r in rows if r['success'] == 'true')
    lat = [int(r['latency_ms']) for r in rows if r['success'] == 'true']
    steps = [int(r['steps']) for r in rows if r['success'] == 'true']
    print(f'=== {mode.upper()} ===')
    print(f'  Success: {success}/{total} ({success/total:.1%})')
    if lat:
        print(f'  Avg latency (pass): {sum(lat)/len(lat):.0f}ms')
        print(f'  Avg steps (pass): {sum(steps)/len(steps):.1f}')
    print()
    for diff in ['easy', 'medium', 'hard', 'edge']:
        d_rows = [r for r in rows if r['difficulty'] == diff]
        d_pass = sum(1 for r in d_rows if r['success'] == 'true')
        print(f'  {diff:8s}: {d_pass}/{len(d_rows)} = {d_pass/len(d_rows):.1%}')
    print()
    from collections import Counter
    errors = Counter(r['error_type'] for r in rows if r['success'] == 'false')
    for err, count in errors.most_common():
        print(f'  {err:15s}: {count}')
    print()

summarize('results/grep_raw.csv', 'grep')
summarize('results/rag_raw.csv', 'rag')
"
```

- [ ] **Step 3: Write experiment report**

Generate report to `docs/report/exp_res_{timestamp}.log` following the same format as the Grep report. Include:
- Overall results comparison table
- Per-difficulty breakdown for both modes
- Per-symbol-type breakdown
- Error analysis
- Key insights and conclusion

- [ ] **Step 4: Commit report**

```bash
git add docs/report/ results/
git commit -m "docs: add RAG experiment report and Grep vs RAG comparison"
```

- [ ] **Step 5: Push all changes**

```bash
git push origin main
```
