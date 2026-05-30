# Agent Eval (Grep vs RAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated eval system that compares Grep-based vs RAG-based agents at locating Java symbol definitions in the Apollo config center codebase.

**Architecture:** Python CLI with two agent implementations sharing a common LLM client (deepseek-v4-flash via OpenAI-compatible API). Grep Agent uses ripgrep/glob/read tools; RAG Agent uses FAISS vector search/read tools. A runner orchestrates 50 tasks end-to-end and emits CSV + failures JSONL.

**Tech Stack:** Python 3.11+, openai SDK, FAISS, numpy, pytest, uv

---

## File Structure Map

```
src/eval/
  __init__.py           # Package marker
  llm_client.py         # DeepSeek LLM wrapper (OpenAI-compatible function calling)
  grep_tools.py         # ripgrep, glob, file read — executed locally
  rag_tools.py          # FAISS vector_search, file read — executed locally
  grep_agent.py         # Grep Agent: LLM loop + grep tools, returns file:line
  rag_agent.py          # RAG Agent: LLM loop + vector search tools, returns file:line
  indexer.py            # Build FAISS index from Java source files
  runner.py             # Load tasks → run agent per task → score → write CSV
  scorer.py             # Compare prediction vs expected (file match + line ±3)
data/
  tasks.jsonl           # 50 curated symbol-lookup tasks (manually annotated from Apollo)
  index/                # FAISS index artifacts (gitignored)
results/                # Output: grep_summary.csv, rag_summary.csv, failures.jsonl
tests/
  test_scorer.py
  test_grep_tools.py
  test_rag_tools.py
  test_llm_client.py
pyproject.toml
```

---

### Task 1: Python Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/eval/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Write pyproject.toml with dependencies**

```toml
[project]
name = "grep-over-rag"
version = "0.1.0"
description = "Agent Eval: Grep vs RAG for Java symbol lookup"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.0.0",
    "numpy>=1.26.0",
    "faiss-cpu>=1.7.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
data/index/
results/
.pytest_cache/
```

- [ ] **Step 3: Create empty __init__.py**

```python
"""Agent Eval: Grep vs RAG for Java symbol lookup."""
```

- [ ] **Step 4: Install dependencies and verify**

Run: `uv pip install -e ".[dev]"`
Expected: all packages install successfully

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src/
git commit -m "feat: add Python project skeleton with dependencies"
```

---

### Task 2: LLM Client (DeepSeek API Wrapper)

**Files:**
- Create: `src/eval/llm_client.py`
- Create: `tests/test_llm_client.py`

The LLM client wraps the OpenAI SDK pointed at DeepSeek's API. It supports function/tool calling, which both agents use to decide which search tool to invoke.

- [ ] **Step 1: Write test for LLM client**

```python
# tests/test_llm_client.py
import os
import pytest
from src.eval.llm_client import LLMClient

def test_llm_client_initializes_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client = LLMClient()
    assert client.base_url == "https://api.deepseek.com"
    assert client.api_key == "sk-test"
    assert client.model == "deepseek-v4-flash"

def test_llm_client_raises_on_missing_env():
    with pytest.raises(ValueError, match="DEEPSEEK_BASE_URL"):
        LLMClient()

def test_llm_client_chat_basic(mocker):
    mock_create = mocker.patch("openai.resources.chat.Completions.create")
    mock_create.return_value = mocker.Mock(
        choices=[mocker.Mock(message=mocker.Mock(content="answer", tool_calls=None))]
    )
    client = LLMClient(base_url="http://test", api_key="sk-test")
    messages = [{"role": "user", "content": "hello"}]
    result = client.chat(messages, tools=[])
    assert result.content == "answer"
    mock_create.assert_called_once()

def test_llm_client_chat_with_tool_calls(mocker):
    mock_create = mocker.patch("openai.resources.chat.Completions.create")
    mock_create.return_value = mocker.Mock(
        choices=[mocker.Mock(message=mocker.Mock(
            content=None,
            tool_calls=[mocker.Mock(
                id="call_1",
                function=mocker.Mock(name="search", arguments='{"pattern":"OrderService"}')
            )]
        ))]
    )
    client = LLMClient(base_url="http://test", api_key="sk-test")
    result = client.chat([{"role": "user", "content": "find OrderService"}], tools=[])
    assert result.tool_calls[0].function.name == "search"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_llm_client.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement LLMClient**

```python
# src/eval/llm_client.py
import os
from dataclasses import dataclass, field
from openai import OpenAI

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)

class LLMClient:
    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.base_url:
            raise ValueError("DEEPSEEK_BASE_URL environment variable is required")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        self.model = "deepseek-v4-flash"
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResult:
        kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            import json
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return ChatResult(content=msg.content, tool_calls=tool_calls)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_llm_client.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/llm_client.py tests/test_llm_client.py
git commit -m "feat: add LLM client wrapping DeepSeek API with tool calling"
```

---

### Task 3: Grep Tools (ripgrep, glob, read)

**Files:**
- Create: `src/eval/grep_tools.py`
- Create: `tests/test_grep_tools.py`

Pure functions — no LLM involved. These are the tools the Grep Agent's LLM can invoke.

- [ ] **Step 1: Write tests for grep tools**

```python
# tests/test_grep_tools.py
import os
import tempfile
from pathlib import Path
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
    lines = content.split("\n")
    assert len(lines) <= 2

def test_read_file_nonexistent():
    content = read_file("/nonexistent/file.java")
    assert content.startswith("Error:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_grep_tools.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement grep tools**

```python
# src/eval/grep_tools.py
import subprocess
from pathlib import Path
from glob import glob as _glob

def rg_search(pattern: str, codebase_root: str, file_pattern: str = "*.java") -> list[dict]:
    """Run ripgrep and return list of {file, line, content}."""
    cmd = [
        "rg", "--no-heading", "--line-number",
        "--glob", file_pattern,
        "--max-count", "30",
        pattern, codebase_root
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        results = []
        for line in lines:
            parts = line.split(":", 2)
            if len(parts) >= 3:
                results.append({
                    "file": parts[0],
                    "line": int(parts[1]),
                    "content": parts[2].strip()
                })
        return results
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

def glob_files(pattern: str, codebase_root: str) -> list[str]:
    """Find files matching glob pattern relative to codebase root."""
    full_pattern = str(Path(codebase_root) / pattern)
    matches = sorted(_glob(full_pattern, recursive=True))
    return [str(Path(m).relative_to(codebase_root)) for m in matches]

def read_file(file_path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read file content. Optionally specify line range (1-indexed)."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if start_line > 0:
                lines = f.readlines()
                end = end_line if end_line > 0 else len(lines)
                return "".join(lines[start_line - 1:end])
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error: {e}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_grep_tools.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/grep_tools.py tests/test_grep_tools.py
git commit -m "feat: add grep tools (rg_search, glob_files, read_file)"
```

---

### Task 4: Scorer (Success/Failure Evaluation)

**Files:**
- Create: `src/eval/scorer.py`
- Create: `tests/test_scorer.py`

- [ ] **Step 1: Write tests for scorer**

```python
# tests/test_scorer.py
from src.eval.scorer import score_prediction

def test_exact_match():
    assert score_prediction(
        predicted_file="foo/Bar.java",
        predicted_line=42,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is True

def test_line_within_tolerance_plus():
    assert score_prediction(
        predicted_file="foo/Bar.java",
        predicted_line=44,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is True

def test_line_within_tolerance_minus():
    assert score_prediction(
        predicted_file="foo/Bar.java",
        predicted_line=39,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is True

def test_line_outside_tolerance():
    assert score_prediction(
        predicted_file="foo/Bar.java",
        predicted_line=46,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is False

def test_file_mismatch():
    assert score_prediction(
        predicted_file="foo/Baz.java",
        predicted_line=42,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is False

def test_both_wrong():
    assert score_prediction(
        predicted_file="foo/Baz.java",
        predicted_line=99,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_scorer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement scorer**

```python
# src/eval/scorer.py
def score_prediction(
    predicted_file: str,
    predicted_line: int,
    expected_file: str,
    expected_line: int
) -> bool:
    """Return True if file matches exactly and line is within ±3."""
    file_match = predicted_file == expected_file
    line_match = abs(predicted_line - expected_line) <= 3
    return file_match and line_match
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_scorer.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/scorer.py tests/test_scorer.py
git commit -m "feat: add scorer with file match + line ±3 tolerance"
```

---

### Task 5: Grep Agent

**Files:**
- Create: `src/eval/grep_agent.py`
- Create: `tests/test_grep_agent.py`

The Grep Agent runs an LLM loop: system prompt describes the task → LLM chooses tools → tools execute locally → LLM sees results → produces final answer `file_path:line_number`.

- [ ] **Step 1: Write test for Grep Agent**

```python
# tests/test_grep_agent.py
import pytest
from unittest.mock import MagicMock, patch
from src.eval.grep_agent import GrepAgent, GrepAgentResult

def make_mock_llm(responses: list):
    """responses: list of ChatResult or Exception"""
    mock = MagicMock()
    mock.chat = MagicMock(side_effect=responses)
    return mock

def test_grep_agent_finds_symbol_in_one_step():
    from src.eval.llm_client import ChatResult, ToolCall

    llm = make_mock_llm([
        ChatResult(content="apollo-biz/src/main/java/.../NamespaceService.java:42"),
    ])
    agent = GrepAgent(llm_client=llm, codebase_root="/tmp/apollo")
    result = agent.run(
        symbol="NamespaceService.findNamespace",
        symbol_type="method",
    )
    assert result.file_path.endswith("NamespaceService.java")
    assert result.line_number == 42
    assert result.steps == 1

def test_grep_agent_uses_tools_then_answers():
    from src.eval.llm_client import ChatResult, ToolCall

    llm = make_mock_llm([
        ChatResult(
            content=None,
            tool_calls=[ToolCall(id="1", name="rg_search", arguments={"pattern": "findNamespace"})]
        ),
        ChatResult(content="apollo-biz/src/main/java/.../NamespaceService.java:87"),
    ])
    agent = GrepAgent(llm_client=llm, codebase_root="/tmp/apollo")
    result = agent.run(
        symbol="NamespaceService.findNamespace",
        symbol_type="method",
    )
    assert result.line_number == 87
    assert result.steps == 2

def test_grep_agent_enforces_max_steps():
    from src.eval.llm_client import ChatResult, ToolCall

    tool_calls = [ChatResult(
        content=None,
        tool_calls=[ToolCall(id="1", name="rg_search", arguments={"pattern": "findX"})]
    )]
    llm = make_mock_llm(tool_calls * 10)  # infinite loop of tool calls
    agent = GrepAgent(llm_client=llm, codebase_root="/tmp/apollo", max_steps=4)
    result = agent.run(symbol="Something", symbol_type="method")
    assert result.steps <= 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_grep_agent.py -v`
Expected: FAIL

- [ ] **Step 3: Implement GrepAgent**

```python
# src/eval/grep_agent.py
import json
import re
from dataclasses import dataclass
from src.eval.llm_client import LLMClient
from src.eval.grep_tools import rg_search, glob_files, read_file

@dataclass
class GrepAgentResult:
    file_path: str
    line_number: int
    steps: int
    raw_output: str

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rg_search",
            "description": "Search for a pattern in Java source files using ripgrep. Returns list of {file, line, content}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": "Find Java files matching a glob pattern (e.g., **/NamespaceService.java).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern relative to codebase root"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content of a Java source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "Optional: start line (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Optional: end line (1-indexed)"},
                },
                "required": ["file_path"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a code search agent. Your task: find the exact definition of a Java symbol.

## Rules
- You MUST output the final answer as: file_path:line_number
- The file_path is relative to the codebase root
- The line_number is where the class/method/enum is DEFINED (not where it's called)
- Do NOT guess — only answer after examining the source code
- You may use rg_search, glob_files, and read_file tools
- Maximum 4 steps total

## Symbol types
- "class": Find the class declaration line
- "method": Find the method declaration line within its class
- "interface": Find the interface declaration line
- "enum": Find the enum declaration line
"""

class GrepAgent:
    def __init__(self, llm_client=None, codebase_root="codebases/apollo", max_steps=4):
        self.llm = llm_client or LLMClient()
        self.codebase_root = codebase_root
        self.max_steps = max_steps

    def run(self, symbol: str, symbol_type: str) -> GrepAgentResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Find the definition of {symbol_type} `{symbol}`."},
        ]

        steps = 0
        for _ in range(self.max_steps):
            steps += 1
            result = self.llm.chat(messages, tools=TOOLS)

            if result.content and not result.tool_calls:
                parsed = self._parse_answer(result.content)
                if parsed:
                    return GrepAgentResult(
                        file_path=parsed[0],
                        line_number=parsed[1],
                        steps=steps,
                        raw_output=result.content,
                    )

            if result.tool_calls:
                for tc in result.tool_calls:
                    tool_result = self._execute_tool(tc.name, tc.arguments)
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })

        return GrepAgentResult(
            file_path="",
            line_number=0,
            steps=steps,
            raw_output="Max steps exceeded",
        )

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "rg_search":
            results = rg_search(args["pattern"], self.codebase_root)
            if not results:
                return "No matches found."
            return "\n".join(
                f"{r['file']}:{r['line']}: {r['content']}" for r in results[:20]
            )
        elif name == "glob_files":
            results = glob_files(args["pattern"], self.codebase_root)
            if not results:
                return "No files found."
            return "\n".join(results[:30])
        elif name == "read_file":
            results = read_file(
                args["file_path"],
                args.get("start_line", 0),
                args.get("end_line", 0),
            )
            return results
        return f"Unknown tool: {name}"

    def _parse_answer(self, text: str) -> tuple[str, int] | None:
        match = re.search(r"(.+?):(\d+)", text)
        if match:
            return (match.group(1).strip(), int(match.group(2)))
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_grep_agent.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/grep_agent.py tests/test_grep_agent.py
git commit -m "feat: add Grep Agent with LLM-driven tool calling loop"
```

---

### Task 6: RAG Indexer

**Files:**
- Create: `src/eval/indexer.py`
- Create: `tests/test_indexer.py`

Builds a FAISS index from all Java source files. Each file is chunked → embedded → stored with metadata.

- [ ] **Step 1: Write test for indexer**

```python
# tests/test_indexer.py
import os
import tempfile
import numpy as np
from pathlib import Path
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
        assert len(loaded) == 2
        assert loaded[0]["file"] == chunks[0]["file"]

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
        assert isinstance(results[0], str)  # serialized result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_indexer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement indexer**

```python
# src/eval/indexer.py
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
        """Walk Java files under src/main/, create chunks, embed them."""
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
            index = faiss.IndexFlatIP(dim)  # inner product (cosine after normalization)
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
```

- [ ] **Step 4: Add __main__ block for CLI usage**

Append to `src/eval/indexer.py`:
```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_indexer.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add src/eval/indexer.py tests/test_indexer.py
git commit -m "feat: add RAG indexer (FAISS + embedding-based file indexing)"
```

---

### Task 7: RAG Tools + RAG Agent

**Files:**
- Create: `src/eval/rag_tools.py`
- Create: `src/eval/rag_agent.py`
- Create: `tests/test_rag_tools.py`
- Create: `tests/test_rag_agent.py`

RAG tools: `vector_search` and `read_file`. The agent runs the same LLM loop pattern as GrepAgent but with vector_search instead of rg_search.

- [ ] **Step 1: Write tests for RAG tools**

```python
# tests/test_rag_tools.py
import os
import tempfile
from pathlib import Path
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
```

- [ ] **Step 2: Write tests for RAG Agent**

```python
# tests/test_rag_agent.py
from unittest.mock import MagicMock
from src.eval.llm_client import ChatResult, ToolCall
from src.eval.rag_agent import RAGAgent

class FakeIndexStore:
    def search(self, query_embedding, top_k=5):
        return [
            "File: NamespaceService.java\nScore: 0.95\n```java\npublic Namespace findNamespace() {\n```"
        ]

class FakeEmbedder:
    def embed(self, texts):
        return [[0.1] * 128 for _ in texts]

def make_mock_llm(responses):
    mock = MagicMock()
    mock.chat = MagicMock(side_effect=responses)
    return mock

def test_rag_agent_finds_symbol_with_vector_search():
    llm = make_mock_llm([
        ChatResult(
            content=None,
            tool_calls=[ToolCall(id="1", name="vector_search", arguments={"query": "NamespaceService.findNamespace"})]
        ),
        ChatResult(content="NamespaceService.java:42"),
    ])
    agent = RAGAgent(
        llm_client=llm,
        codebase_root="/tmp",
        index_store=FakeIndexStore(),
        embedder=FakeEmbedder(),
    )
    result = agent.run(symbol="NamespaceService.findNamespace", symbol_type="method")
    assert result.file_path == "NamespaceService.java"
    assert result.line_number == 42
    assert result.steps == 2

def test_rag_agent_direct_answer():
    llm = make_mock_llm([
        ChatResult(content="NamespaceService.java:87"),
    ])
    agent = RAGAgent(
        llm_client=llm,
        codebase_root="/tmp",
        index_store=FakeIndexStore(),
        embedder=FakeEmbedder(),
    )
    result = agent.run(symbol="NamespaceService.findNamespace", symbol_type="method")
    assert result.line_number == 87
    assert result.steps == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_rag_tools.py tests/test_rag_agent.py -v`
Expected: FAIL

- [ ] **Step 4: Implement RAG tools**

```python
# src/eval/rag_tools.py
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
```

- [ ] **Step 5: Implement RAG Agent**

```python
# src/eval/rag_agent.py
import json
import re
from dataclasses import dataclass
from src.eval.llm_client import LLMClient
from src.eval.rag_tools import vector_search, read_file_for_rag

@dataclass
class RAGAgentResult:
    file_path: str
    line_number: int
    steps: int
    raw_output: str

RAG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Search the codebase using semantic vector search. Best for finding where a symbol is defined. Returns top files with snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query, e.g., 'find where OrderService.createOrder is defined'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content of a Java source file to verify the exact definition line.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "Optional: start line (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Optional: end line (1-indexed)"},
                },
                "required": ["file_path"],
            },
        },
    },
]

RAG_SYSTEM_PROMPT = """You are a code search agent using semantic vector search. Your task: find the exact definition of a Java symbol.

## Rules
- You MUST output the final answer as: file_path:line_number
- The file_path is relative to the codebase root
- The line_number is where the class/method/enum is DEFINED
- Do NOT guess — verify by reading the source file
- You may use vector_search and read_file tools
- Maximum 4 steps total

## Symbol types
- "class": Find the class declaration line
- "method": Find the method declaration line within its class
- "interface": Find the interface declaration line
- "enum": Find the enum declaration line
"""

class RAGAgent:
    def __init__(self, llm_client=None, codebase_root="codebases/apollo",
                 index_store=None, embedder=None, max_steps=4):
        self.llm = llm_client or LLMClient()
        self.codebase_root = codebase_root
        self.index_store = index_store
        self.embedder = embedder
        self.max_steps = max_steps

    def run(self, symbol: str, symbol_type: str) -> RAGAgentResult:
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Find the definition of {symbol_type} `{symbol}`."},
        ]

        steps = 0
        for _ in range(self.max_steps):
            steps += 1
            result = self.llm.chat(messages, tools=RAG_TOOLS)

            if result.content and not result.tool_calls:
                parsed = self._parse_answer(result.content)
                if parsed:
                    return RAGAgentResult(
                        file_path=parsed[0],
                        line_number=parsed[1],
                        steps=steps,
                        raw_output=result.content,
                    )

            if result.tool_calls:
                for tc in result.tool_calls:
                    tool_result = self._execute_tool(tc.name, tc.arguments)
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })

        return RAGAgentResult(
            file_path="",
            line_number=0,
            steps=steps,
            raw_output="Max steps exceeded",
        )

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "vector_search":
            return vector_search(args["query"], self.index_store, self.embedder)
        elif name == "read_file":
            return read_file_for_rag(
                args["file_path"],
                args.get("start_line", 0),
                args.get("end_line", 0),
            )
        return f"Unknown tool: {name}"

    def _parse_answer(self, text: str) -> tuple[str, int] | None:
        match = re.search(r"(.+?):(\d+)", text)
        if match:
            return (match.group(1).strip(), int(match.group(2)))
        return None
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_rag_tools.py tests/test_rag_agent.py -v`
Expected: 5 PASS

- [ ] **Step 7: Commit**

```bash
git add src/eval/rag_tools.py src/eval/rag_agent.py tests/test_rag_tools.py tests/test_rag_agent.py
git commit -m "feat: add RAG Agent with vector_search + read_file tools"
```

---

### Task 8: Embedder (DeepSeek Embedding API)

**Files:**
- Create: `src/eval/embedder.py`

Wraps DeepSeek's embedding API (or OpenAI text-embedding-3-small as fallback) for the RAG pipeline. Used by both indexer and RAG agent at query time.

- [ ] **Step 1: Implement embedder**

```python
# src/eval/embedder.py
import os
import time
from openai import OpenAI


class DeepSeekEmbedder:
    """Embedding client for DeepSeek's embedding API."""

    def __init__(self, base_url=None, api_key=None, model=None, batch_size=20):
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model or "text-embedding-3-small"
        self.batch_size = batch_size
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching if needed. Returns list of embedding vectors."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in response.data])
            if i + self.batch_size < len(texts):
                time.sleep(0.1)  # rate limit courtesy
        return all_embeddings
```

- [ ] **Step 2: Commit**

```bash
git add src/eval/embedder.py
git commit -m "feat: add DeepSeek embedder for RAG index + query"
```

---

### Task 9: Runner (Orchestrator)

**Files:**
- Create: `src/eval/runner.py`

The runner reads tasks from JSONL, dispatches to the appropriate agent, scores results, and writes CSV + failures JSONL.

- [ ] **Step 1: Write test for runner**

```python
# tests/test_runner.py
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.eval.runner import Runner

def make_mock_agent(results):
    """Create a mock agent that returns the given results in sequence."""
    mock = MagicMock()
    mock.run = MagicMock(side_effect=results)
    return mock

class MockAgentResult:
    def __init__(self, file_path, line_number, steps, raw_output):
        self.file_path = file_path
        self.line_number = line_number
        self.steps = steps
        self.raw_output = raw_output

def test_runner_processes_all_tasks(tmp_path):
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("\n".join([
        json.dumps({"id": "t1", "symbol": "Foo.bar", "type": "method", "file": "Foo.java", "expected_line": 10, "difficulty": "easy", "note": ""}),
        json.dumps({"id": "t2", "symbol": "Baz.qux", "type": "class", "file": "Baz.java", "expected_line": 3, "difficulty": "medium", "note": ""}),
        json.dumps({"id": "t3", "symbol": "Qux.fred", "type": "method", "file": "Qux.java", "expected_line": 25, "difficulty": "easy", "note": ""}),
    ]))

    agent = make_mock_agent([
        MockAgentResult("Foo.java", 10, 1, "ok"),
        MockAgentResult("Baz.java", 4, 2, "ok"),   # line 4 vs expected 3 → within ±3
        MockAgentResult("Wrong.java", 25, 3, "wrong file"),  # wrong file
    ])

    runner = Runner(agent_factory=lambda: agent, mode="grep")
    summary_path = tmp_path / "summary.csv"
    failures_path = tmp_path / "failures.jsonl"

    runner.run(tasks_file=str(tasks_file), summary_path=str(summary_path), failures_path=str(failures_path))

    # Verify summary CSV
    lines = summary_path.read_text().strip().split("\n")
    assert len(lines) == 4  # header + 3 tasks
    assert lines[0] == "task_id,difficulty,symbol,success,steps,latency_ms,predicted_file,predicted_line"
    assert "t1,easy,Foo.bar,true,1" in lines[1]
    assert "t2,medium,Baz.qux,true,2" in lines[2]
    assert "t3,easy,Qux.fred,false,3" in lines[3]

    # Verify failures JSONL
    failures = failures_path.read_text().strip().split("\n")
    assert len(failures) == 1
    fail = json.loads(failures[0])
    assert fail["task_id"] == "t3"

def test_runner_stats_summary(tmp_path):
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("\n".join([
        json.dumps({"id": "t1", "symbol": "Foo.bar", "type": "method", "file": "Foo.java", "expected_line": 10, "difficulty": "easy", "note": ""}),
        json.dumps({"id": "t2", "symbol": "Baz.qux", "type": "class", "file": "Baz.java", "expected_line": 3, "difficulty": "medium", "note": ""}),
    ]))

    agent = make_mock_agent([
        MockAgentResult("Foo.java", 10, 1, "ok"),
        MockAgentResult("Wrong.java", 99, 4, "miss"),
    ])

    runner = Runner(agent_factory=lambda: agent, mode="grep")
    summary_path = tmp_path / "summary.csv"

    runner.run(tasks_file=str(tasks_file), summary_path=str(summary_path), failures_path=str(tmp_path / "fail.jsonl"))
    assert runner.stats["total"] == 2
    assert runner.stats["success"] == 1
    assert runner.stats["failed"] == 1
    assert runner.stats["success_rate"] == 0.5
    assert runner.stats["avg_steps"] == 2.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement runner**

```python
# src/eval/runner.py
import csv
import json
import time
import sys
from pathlib import Path
from src.eval.scorer import score_prediction


class Runner:
    def __init__(self, agent_factory, mode="grep"):
        self.agent_factory = agent_factory
        self.mode = mode
        self.stats = {}

    def run(self, tasks_file: str, summary_path: str, failures_path: str):
        tasks = self._load_tasks(tasks_file)
        rows = []
        failures = []
        success_count = 0
        total_steps = 0
        total_latency = 0

        for task in tasks:
            agent = self.agent_factory()
            start = time.perf_counter()
            result = agent.run(symbol=task["symbol"], symbol_type=task["type"])
            latency_ms = int((time.perf_counter() - start) * 1000)

            success = score_prediction(
                predicted_file=result.file_path,
                predicted_line=result.line_number,
                expected_file=task["file"],
                expected_line=task["expected_line"],
            )

            rows.append({
                "task_id": task["id"],
                "difficulty": task.get("difficulty", ""),
                "symbol": task["symbol"],
                "success": str(success).lower(),
                "steps": result.steps,
                "latency_ms": latency_ms,
                "predicted_file": result.file_path,
                "predicted_line": result.line_number,
            })

            if success:
                success_count += 1
            else:
                failures.append({
                    "task_id": task["id"],
                    "symbol": task["symbol"],
                    "type": task["type"],
                    "expected_file": task["file"],
                    "expected_line": task["expected_line"],
                    "predicted_file": result.file_path,
                    "predicted_line": result.line_number,
                    "steps": result.steps,
                    "latency_ms": latency_ms,
                    "raw_output": result.raw_output,
                })

            total_steps += result.steps
            total_latency += latency_ms

        # Write summary CSV
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "task_id", "difficulty", "symbol", "success",
                "steps", "latency_ms", "predicted_file", "predicted_line",
            ])
            writer.writeheader()
            writer.writerows(rows)

        # Write failures JSONL
        if failures:
            Path(failures_path).parent.mkdir(parents=True, exist_ok=True)
            with open(failures_path, "w") as f:
                for fail in failures:
                    f.write(json.dumps(fail, ensure_ascii=False) + "\n")

        total = len(tasks)
        self.stats = {
            "total": total,
            "success": success_count,
            "failed": total - success_count,
            "success_rate": success_count / total if total > 0 else 0,
            "avg_steps": total_steps / total if total > 0 else 0,
            "avg_latency_ms": total_latency / total if total > 0 else 0,
        }

    def _load_tasks(self, tasks_file: str) -> list[dict]:
        tasks = []
        with open(tasks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        return tasks


def main():
    import argparse
    from src.eval.grep_agent import GrepAgent
    from src.eval.rag_agent import RAGAgent
    from src.eval.indexer import IndexStore
    from src.eval.embedder import DeepSeekEmbedder

    parser = argparse.ArgumentParser(description="Agent Eval: Grep vs RAG")
    parser.add_argument("--mode", choices=["grep", "rag"], required=True)
    parser.add_argument("--codebase", default="codebases/apollo")
    parser.add_argument("--tasks", default="data/tasks.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    summary_file = f"{args.output_dir}/{args.mode}_summary.csv"
    failures_file = f"{args.output_dir}/failures.jsonl"

    if args.mode == "grep":
        agent_factory = lambda: GrepAgent(codebase_root=args.codebase)
    else:
        store = IndexStore(args.index_dir)
        embedder = DeepSeekEmbedder()
        agent_factory = lambda: RAGAgent(
            codebase_root=args.codebase,
            index_store=store,
            embedder=embedder,
        )

    runner = Runner(agent_factory=agent_factory, mode=args.mode)
    runner.run(
        tasks_file=args.tasks,
        summary_path=summary_file,
        failures_path=failures_file,
    )

    print(f"\n=== {args.mode.upper()} Results ===")
    print(f"Total: {runner.stats['total']}")
    print(f"Success: {runner.stats['success']}")
    print(f"Failed: {runner.stats['failed']}")
    print(f"Success Rate: {runner.stats['success_rate']:.1%}")
    print(f"Avg Steps: {runner.stats['avg_steps']:.1f}")
    print(f"Avg Latency: {runner.stats['avg_latency_ms']:.0f}ms")
    print(f"Summary: {summary_file}")
    if runner.stats["failed"] > 0:
        print(f"Failures: {failures_file}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_runner.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/runner.py tests/test_runner.py
git commit -m "feat: add eval runner with CSV + failures JSONL output"
```

---

### Task 10: Task Dataset (50 Curated Apollo Symbols)

**Files:**
- Create: `data/tasks.jsonl`

Manually curate 50 symbol-lookup tasks from Apollo source. This step requires reading Apollo Java files and recording exact line numbers.

**Curate approach:**
1. Sample across all 7 modules proportionally to file count
2. Cover all difficulty tiers: 20 easy, 15 medium, 10 hard, 5 edge
3. Select real symbols from Apollo's actual code

**Module file distribution (approximate):**
- apollo-biz: ~120 files (24%)
- apollo-portal: ~110 files (22%)
- apollo-adminservice: ~70 files (14%)
- apollo-configservice: ~60 files (12%)
- apollo-common: ~60 files (12%)
- apollo-audit: ~50 files (10%)
- apollo-assembly: ~40 files (8%)

**Symbol selection criteria by difficulty:**
- **easy (20)**: Unique class names like `AdminServiceHealthIndicator`, `CommitController`, `AuditEntity`
- **medium (15)**: Common method names appearing in multiple files, e.g., `findByAppId`, `getConfig`, `update`
- **hard (10)**: Methods with overloads, inner classes, long files (600+ lines), e.g., methods in `ReleaseService.java`
- **edge (5)**: Enum definitions, interface methods, abstract class methods

- [ ] **Step 1: Curate 20 easy tasks**

Scan Apollo source for classes with globally unique simple names. Each task record:
```json
{
  "id": "task_001",
  "symbol": "AdminServiceHealthIndicator",
  "type": "class",
  "file": "apollo-adminservice/src/main/java/com/ctrip/framework/apollo/adminservice/AdminServiceHealthIndicator.java",
  "expected_line": <actual_line>,
  "difficulty": "easy",
  "note": "Globally unique class name"
}
```

Working: extract 20 such classes from Apollo, record the exact line where `public class X` appears.

- [ ] **Step 2: Curate 15 medium tasks**

Find methods whose names appear in multiple files (e.g., `findByAppId`, `delete`, `save`, `findOne`, `handleMessage`). Verify with `rg` that the name has ≥3 occurrences. Task records specify the exact class + method.

```json
{
  "id": "task_021",
  "symbol": "AppRepository.findByAppId",
  "type": "method",
  "file": "apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/repository/AppRepository.java",
  "expected_line": <actual_line>,
  "difficulty": "medium",
  "note": "findByAppId appears in 5 repository files"
}
```

Working: `rg "findByAppId" codebases/apollo/ --glob '*.java'` to find candidates, record the exact method declaration lines.

- [ ] **Step 3: Curate 10 hard tasks**

Select from:
- Methods with overloads: same method name, different parameters within the same class
- Inner class definitions: `public class Outer { public class Inner {} }`
- Methods in large files (600+ lines): `ReleaseService.java`, `OpenApiModelConverters.java`

```json
{
  "id": "task_036",
  "symbol": "ReleaseService.publish",
  "type": "method",
  "file": "apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/ReleaseService.java",
  "expected_line": <actual_line>,
  "difficulty": "hard",
  "note": "Method in 633-line file, multiple overloaded publish methods"
}
```

- [ ] **Step 4: Curate 5 edge tasks**

Target edge cases:
- Enum constants
- `@interface` (annotation type) definitions
- Interface methods with `default` implementation
- Abstract class protected constructors

```json
{
  "id": "task_046",
  "symbol": "ConfigFileFormat",
  "type": "enum",
  "file": "apollo-common/src/main/java/com/ctrip/framework/apollo/common/dto/ConfigFileFormat.java",
  "expected_line": <actual_line>,
  "difficulty": "edge",
  "note": "Enum definition"
}
```

- [ ] **Step 5: Validate dataset**

Run: `uv run python -c "
import json
with open('data/tasks.jsonl') as f:
    tasks = [json.loads(l) for l in f if l.strip()]
assert len(tasks) == 50, f'Expected 50, got {len(tasks)}'
from collections import Counter
diffs = Counter(t['difficulty'] for t in tasks)
assert diffs['easy'] == 20
assert diffs['medium'] == 15
assert diffs['hard'] == 10
assert diffs['edge'] == 5
print('Dataset valid: 50 tasks, correct distribution')
"`

- [ ] **Step 6: Commit**

```bash
git add data/tasks.jsonl
git commit -m "feat: add 50 curated symbol-lookup tasks from Apollo source"
```

---

### Task 11: Integration — Build RAG Index + End-to-End Smoke Test

**Files:**
- Create: `scripts/build_index.py` (thin CLI for indexer)
- Create: `scripts/smoke_test.sh` (quick end-to-end)

- [ ] **Step 1: Create index builder CLI**

```python
# scripts/build_index.py
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
```

- [ ] **Step 2: Verify index build works on small subset**

Run (on a subset of 10 files): `uv run python -c "
import sys; sys.path.insert(0, 'src')
from src.eval.indexer import IndexBuilder, IndexStore
from src.eval.embedder import DeepSeekEmbedder
import os, tempfile

# Collect 10 files
java_files = []
for root, _, files in os.walk('codebases/apollo'):
    for f in files:
        if f.endswith('.java') and '/src/main/' in root:
            java_files.append(os.path.join(root, f))
        if len(java_files) >= 10:
            break
    if len(java_files) >= 10:
        break

with tempfile.TemporaryDirectory() as tmp:
    # write 10 files to tmp
    import shutil
    for jf in java_files[:10]:
        dst = os.path.join(tmp, os.path.basename(jf))
        shutil.copy(jf, dst)

    class FakeEmbedder:
        def embed(self, texts):
            return [[0.1] * 128 for _ in texts]

    builder = IndexBuilder(tmp, FakeEmbedder())
    chunks = builder.build_chunks()
    assert len(chunks) == 10
    store = IndexStore(os.path.join(tmp, 'index'))
    store.save(chunks)
    loaded = store.load()
    print(f'OK: {len(loaded[\"metadata\"])} documents in index')
"` (Requires DEEPSEEK_BASE_URL and DEEPSEEK_API_KEY env vars)

- [ ] **Step 3: Commit**

```bash
git add scripts/build_index.py
git commit -m "feat: add index builder CLI and smoke test"
```

---

## Implementation Order

| Order | Task | Dependencies |
|---|---|---|
| 1 | Project skeleton | none |
| 2 | LLM Client | Task 1 |
| 3 | Grep Tools | Task 1 |
| 4 | Scorer | Task 1 |
| 5 | Grep Agent | Tasks 2, 3 |
| 6 | RAG Indexer | Task 1 |
| 7 | RAG Tools + RAG Agent | Tasks 2, 6 |
| 8 | Embedder | Task 1 |
| 9 | Runner | Tasks 4, 5, 7, 8 |
| 10 | Task Dataset | Apollo codebase |
| 11 | Integration test | Tasks 9, 10 |
