# RAG Pipeline Implementation Design

**Date:** 2026-06-04
**Status:** Approved
**Context:** Grep agent experiment complete (68% success on 50 tasks). RAG pipeline needed for fair comparison.

---

## Problem Statement

The experiment compares two code search paradigms (Grep vs RAG) for locating Java symbol definitions in the Apollo codebase. The Grep agent is complete. The RAG agent has code scaffolding but cannot run due to:

1. **No embedding model** — DeepSeek API does not offer an embedding endpoint
2. **Coarse chunking** — current indexer treats each `.java` file as one chunk; 57% of files exceed the 2000-char metadata truncation threshold
3. **Parser bug** — `rag_agent.py` has the same markdown-artifact parsing bug that was fixed in `grep_agent.py`

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding provider | Alibaba DashScope `text-embedding-v4` | OpenAI-compatible API, no new SDK needed |
| Chunking strategy | Symbol-level (class/method/interface/enum) | Gives RAG a fair shot at method localization |
| Chunking implementation | Regex-based Java parsing | Zero dependencies, sufficient for Apollo's consistent code style |
| Fairness principle | Both paradigms at their best | Grep uses `rg` (best text search tool); RAG uses symbol-level embedding (best semantic retrieval) |

## Architecture

```
Build Phase (one-time):
  .java files → SymbolChunker → Chunk[]
  Chunk[] → DashScopeEmbedder → embedding vectors
  vectors + metadata → FAISS IndexStore

Runtime Phase (per task):
  tasks.jsonl → Runner → RAGAgent
    → vector_search(query) → top-5 chunks with line numbers
    → read_file (if needed for precise line)
    → deepseek-v4-flash → answer
```

Unchanged components: `Runner`, `LLMClient`, `Scorer`, `Classifier`, `tasks.jsonl`.

## Components

### 1. SymbolChunker (`src/eval/chunker.py`) — NEW

Scans `.java` files and extracts symbol definitions as chunks.

**Output per chunk:**

```python
@dataclass
class Chunk:
    file: str          # relative path from codebase root
    symbol: str        # fully qualified name, e.g. "ReleaseService.publish"
    symbol_type: str   # "class" | "method" | "interface" | "enum"
    start_line: int    # 1-indexed definition start line
    end_line: int      # 1-indexed definition end line
    content: str       # full source text from declaration to closing }
```

**Parsing logic:**

1. Line-by-line scan with regex matching:
   - Class/interface/enum: `(public|private|protected)?\s*(static)?\s*(abstract)?\s*(class|interface|enum)\s+\w+`
   - Method: `(public|private|protected)\s+(static\s+)?(final\s+)?(\S+\s+)+\w+\s*\(`
2. From declaration line, count `{`/`}` pairs until balanced
3. Skip braces inside block comments (`/*...*/`), line comments (`//`), and string literals
4. Top-level class/interface/enum → one chunk; each method within → separate chunk

**Expected output:** 510 Java files → ~3000-4000 chunks.

**Known limitations:** Does not handle nested inner classes beyond 2 levels, anonymous inner classes, or lambda-defined methods.

### 2. DashScopeEmbedder (`src/eval/embedder.py`) — MODIFIED

Add `DashScopeEmbedder` class alongside existing `DeepSeekEmbedder`.

```python
class DashScopeEmbedder:
    def __init__(self, base_url=None, api_key=None, model=None, batch_size=6):
        self.base_url = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.model = model or "text-embedding-v4"
        self.batch_size = batch_size  # conservative for DashScope rate limits
```

Uses the same `OpenAI` client (DashScope is OpenAI-compatible). Batch size set to 6 due to DashScope's stricter rate limits vs OpenAI.

### 3. Indexer (`src/eval/indexer.py`) — MODIFIED

Changes to `IndexBuilder.build_chunks()`:

- Replace whole-file reading with `SymbolChunker`
- Embedding input = `f"{symbol} ({symbol_type})\n{content[:500]}"` — gives the embedding model semantic context (symbol name + type) alongside code
- Metadata stores: `file`, `symbol`, `symbol_type`, `start_line`, `end_line`, `content` (full, no truncation)

Changes to `IndexStore.search()` return format:

```
File: apollo-biz/src/main/java/.../ReleaseService.java
Symbol: ReleaseService.publish (method)
Lines: 195-240
Score: 0.8734
```java
public ReleaseDTO publish(...) {
    ...
}
```
```

### 4. RAG Agent Parser Fix (`src/eval/rag_agent.py`) — MODIFIED

Replace `_parse_answer` with the same fix applied to `grep_agent.py`:

```python
def _parse_answer(self, text: str) -> tuple[str, int] | None:
    match = re.search(r"([\w\-./]+\.java):(\d+)", text)
    if match:
        path = match.group(1).strip().strip("`*'\"")
        return (path, int(match.group(2)))
    match = re.search(r"([\w\-./]+):(\d+)", text)
    if match:
        path = match.group(1).strip().strip("`*'\"")
        return (path, int(match.group(2)))
    return None
```

### 5. RAG Tools (`src/eval/rag_tools.py`) — MODIFIED

`vector_search()` return format updated to include symbol name, type, and line range from metadata (aligned with indexer changes above).

## Environment Variables

| Variable | Purpose |
|---|---|
| `DEEPSEEK_BASE_URL` | LLM API for chat (deepseek-v4-flash) |
| `DEEPSEEK_API_KEY` | LLM API key |
| `DASHSCOPE_API_KEY` | Embedding API key |
| `DASHSCOPE_BASE_URL` | Optional override (defaults to `https://dashscope.aliyuncs.com/compatible-mode/v1`) |

## Execution Flow

```bash
# 1. Set environment
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_API_KEY="sk-xxx"
export DASHSCOPE_API_KEY="sk-xxx"

# 2. Build RAG index (symbol-level chunking + DashScope embedding)
uv run python -m src.eval.indexer \
  --codebase codebases/apollo \
  --index-dir data/index

# 3. Run RAG agent
uv run python -m src.eval.runner \
  --mode rag \
  --codebase codebases/apollo \
  --index-dir data/index \
  --tasks data/tasks.jsonl

# 4. Compare
# results/grep_raw.csv vs results/rag_raw.csv
```

## File Change Summary

| File | Action | Description |
|---|---|---|
| `src/eval/chunker.py` | NEW | Symbol-level Java chunker |
| `src/eval/embedder.py` | MODIFY | Add `DashScopeEmbedder` class |
| `src/eval/indexer.py` | MODIFY | Integrate chunker, update metadata format |
| `src/eval/rag_tools.py` | MODIFY | Update search result format with line numbers |
| `src/eval/rag_agent.py` | MODIFY | Fix `_parse_answer` |
| `src/eval/grep_agent.py` | NO CHANGE | Already complete |
| `src/eval/runner.py` | NO CHANGE | Already complete |

## Success Criteria

- RAG agent runs 50 tasks end-to-end without crashes
- Results CSV comparable format to Grep results
- No regression in Grep agent

## Out of Scope

- Modifying Grep agent
- Hyperparameter tuning (top-k, batch size, embedding dimensions)
- Building a web UI or dashboard for results
- Multi-run statistical analysis (runs > 1)
