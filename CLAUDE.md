# CLAUDE.md — Agent Eval (Grep vs RAG)

## 1. Eval 目标（唯一）

> **在给定 Java 代码库中，定位特定 Java 符号（Class / Method）的定义位置。**

不评估：
- 代码理解
- 代码生成
- 重构建议
- 多跳推理

✅ **只评估：能否精确找到符号定义。**

---

## 2. LLM Provider（强制约束）

Agent **必须**通过以下环境变量调用 LLM：

| 环境变量 | 用途 |
|---|---|
| `DEEPSEEK_BASE_URL` | LLM API Base URL |
| `DEEPSEEK_API_KEY` | LLM API Secret Key |

**模型名称（固定）：**
```
deepseek-v4-flash
```

❌ 禁止使用其他模型  
❌ 禁止硬编码 Key / URL  

---

## 3. 被测对象（严格对等）

| 维度 | Grep Agent | RAG Agent |
|---|---|---|
| 检索方式 | `rg (ripgrep)` | Vector Search |
| 工具 | `grep`, `glob`, `read` | `vector_search`, `read` |
| 索引 | 无 | 预构建 |
| LLM | deepseek-v4-flash | deepseek-v4-flash |

---

## 4. 任务定义（OneShot）

每个 Task 为一个 JSONL 样本：

```json
{
  "id": "task_001",
  "symbol": "OrderService.createOrder",
  "type": "method",
  "file": "com/example/order/OrderService.java",
  "expected_line": 42
}
```

### 合法 Symbol 类型
- Class
- Interface
- Method
- Enum

---

## 5. Agent 行为规范（硬约束）

### 通用
- 只允许 **1 次最终回答**
- 不允许猜测
- 回答必须是 `file_path:line_number`
- 使用 `deepseek-v4-flash`

### Grep Agent
✅ 允许：
- `rg "SymbolName"`
- `glob **/*.java`
- `read file`

❌ 禁止：
- Embedding
- Vector DB

### RAG Agent
✅ 允许：
- Embedding（`text-embedding-3-small` 或 DeepSeek Embedding）
- Top-K 检索

❌ 禁止：
- 原生 grep
- 文件系统扫描

---

## 6. 成功判定（自动化）

```python
success = (
    predicted_file == expected_file
    and abs(predicted_line - expected_line) <= 3
)
```

| 指标 | 要求 |
|---|---|
| Success | ✅ / ❌ |
| Max Steps | ≤ 4 |
| Hallucination | 0 |
| Latency | < 10s / task |

---

## 7. 数据集（固定）

- **代码库**：单个 Spring Boot 项目
- **规模**：≤ 500 Java 文件
- **样本数**：50

---

## 8. 执行方式

```bash
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_API_KEY="sk-xxxxxxxx"

uv run python -m src.eval.runner \
  --mode grep \
  --tasks data/tasks.jsonl
```

```bash
uv run python -m src.eval.runner \
  --mode rag \
  --tasks data/tasks.jsonl
```

---

## 9. 输出（机器可读）

```text
results/
├── grep_summary.csv
├── rag_summary.csv
└── failures.jsonl
```

CSV 字段：
```csv
task_id,success,steps,latency_ms
```

---

## 10. 禁止项（红线）

❌ 多轮对话  
❌ Agent Memory  
❌ Human-in-the-loop  
❌ 更换模型  
❌ 动态修改任务  

---

## 11. 结束条件

Eval 结束当且仅当：
- 50 个样本全部完成
- 无人工干预
- 结果可复现

---

> **本文件为 Agent 执行的唯一事实来源。**