# Failure Mode Experiment — Design Spec

> **核心论点 (C): 失败模式论** — Grep 失败是"找不到"(empty_result)，RAG 失败是"找错"(wrong_file)。对于 2B Agent Orchestration，silent failure < misleading answer。

## 1. 实验矩阵

```
50 tasks × 2 modes (grep / rag) × 5 runs = 500 次 Agent 调用
```

**N=5 的理由：**
- McNemar 检验在 50 对样本、中等效应量（success rate 差 ≥10%）下有 ≥80% power
- 5 runs 足够区分：always fails (5/5) / sporadic (1-4/5) / always passes (0/5)
- 每次 LLM 调用独立（新 Agent 实例，不共享上下文）

## 2. 新增模块

### 2.1 Error Classifier (`src/eval/classifier.py`)

```python
def classify(predicted_file, predicted_line, expected_file, expected_line) -> str:
    # Returns: "correct" | "wrong_file" | "wrong_line" | "empty_result"

def has_valid_result(predicted_file, predicted_line) -> bool:
    # True if agent produced a non-empty prediction
```

### 2.2 Statistical Reporter (`src/eval/stats.py`)

三个统计问题：

| # | 问题 | 方法 | 输出 |
|---|---|---|---|
| 1 | Grep vs RAG 是否有显著差异？ | McNemar + paired bootstrap 95% CI | p-value, effect size, CI |
| 2 | 差异集中在哪个难度？ | Per-layer Fisher exact (easy/medium/hard) | 分层 p-values |
| 3 | 失败模式分布是否不同？ | Fisher-Freeman-Halton (2×3 table) | p-value, 频数表 |

模块接口：
```python
class ComparisonStats:
    def compute(self, grep_aggregate: list[dict], rag_aggregate: list[dict]) -> dict:
        return {
            "overall": { ... },
            "by_difficulty": { ... },
            "failure_patterns": { ... },
        }
```

### 2.3 Case Study Extractor (`scripts/extract_cases.py`)

自动筛选最具叙事价值的失败案例：

**筛选优先级：**
1. **Divergence case** — Grep 5/5 pass AND RAG 0/5 fail
2. **Wrong_file case** — RAG error_type = wrong_file
3. **High confidence failure** — RAG 预测了具体行号但与 ground truth 偏差大

**输出 K=5（top 5 案例）：**
- 2 divergence cases
- 2 wrong_file cases
- 1 RAG wins case（展示完整性）

## 3. 修改模块

### 3.1 Runner (`src/eval/runner.py`)

**改动：**
- 新增 `--runs N` 参数（默认 1，向后兼容）
- 每个 task 跑 N 次，每个 run 创建新 Agent 实例
- 将 `scorer.score_prediction` 替换为 `classifier.classify`
- 单次 API 失败不中断 batch（记录 `error_type="api_error"`）
- 输出两级 CSV：

**Raw CSV** (每行 = 一次 run):
```
run_id, task_id, difficulty, symbol, type,
expected_file, expected_line,
predicted_file, predicted_line,
success, error_type, steps, latency_ms
```

**Aggregate CSV** (每行 = 一个 task):
```
task_id, difficulty, symbol,
success_count, success_rate,
mean_steps, std_steps,
mean_latency_ms, std_latency_ms,
failure_pattern, dominant_error_type
```

### 3.2 Scorer (`src/eval/scorer.py`)

保留 `score_prediction()`（向后兼容），新增 `has_valid_result()`。

## 4. 输出结构

```
results/
├── grep_raw.csv           # 250 rows
├── rag_raw.csv            # 250 rows
├── grep_aggregate.csv     # 50 rows
├── rag_aggregate.csv      # 50 rows
├── comparison.csv         # 50 rows: task_id, grep_rate, rag_rate, delta
├── summary.json           # Top-level stats + significance + per-difficulty
├── failures_grep.jsonl    # All failed runs
├── failures_rag.jsonl     # All failed runs
└── case_studies/
    ├── case_01.md ... case_05.md
```

## 5. 一键实验流程

```bash
bash scripts/run_experiment.sh
```

等价于：
```bash
# Step 1: Build RAG index
uv run python scripts/build_index.py

# Step 2: Grep Agent (50 × 5 runs)
uv run python -m src.eval.runner --mode grep --runs 5

# Step 3: RAG Agent (50 × 5 runs)
uv run python -m src.eval.runner --mode rag --runs 5

# Step 4: Statistical report
uv run python -m src.eval.stats results/

# Step 5: Case studies
uv run python scripts/extract_cases.py results/ --top 5
```

## 6. 最终交付物与论证映射

| 交付文件 | 论证点 |
|---|---|
| `comparison.csv` | Head-to-head per task |
| `summary.json` | 是否有统计显著差异？效应量多大？ |
| `case_studies/*.md` | **工程师叙事**：具体展示 RAG 为什么找错 |
| Failures breakdown (wrong_file/empty) | **失败模式分布**：Grep fail-silent vs RAG mislead |
| Per-difficulty stats | 差异集中在 medium/hard（同名符号多文件） |
| Latency comparison | RAG 多一次 embedding API 调用，但主论点在失败模式 |
