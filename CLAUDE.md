# CLAUDE.md — Agent Eval (Grep vs RAG)

## 1. Eval 目标（唯一）

> **在 Apollo 配置中心 Java 代码库中，定位特定 Java 符号（Class / Method）的定义位置。**

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

## 4. 被测代码库

**项目：** [Apollo 配置中心](https://github.com/apolloconfig/apollo)（携程开源）

**管理方式：** Git Submodule，位于 `codebases/apollo/`

```bash
git submodule add https://github.com/apolloconfig/apollo.git codebases/apollo
```

**规模统计：**
- 主源代码（src/main）：510 个 `.java` 文件
- 测试代码（src/test）：208 个 `.java` 文件（不计入 eval 范围）
- 模块：7 个核心模块

**模块列表（按代码量排序）：**

| 模块 | 路径 | 职责 |
|---|---|---|
| apollo-biz | `codebases/apollo/apollo-biz/` | 核心业务逻辑 |
| apollo-adminservice | `codebases/apollo/apollo-adminservice/` | 管理后台 |
| apollo-configservice | `codebases/apollo/apollo-configservice/` | 配置服务 |
| apollo-portal | `codebases/apollo/apollo-portal/` | 管理门户 |
| apollo-common | `codebases/apollo/apollo-common/` | 公共组件 |
| apollo-audit | `codebases/apollo/apollo-audit/` | 审计模块 |
| apollo-assembly | `codebases/apollo/apollo-assembly/` | 打包模块 |

**包路径：** 所有代码位于 `com/ctrip/framework/apollo/` 下。

**Eval 范围限定：** 仅 `codebases/apollo/*/src/main/java/**/*.java`（不含 test）。

---

## 5. 任务定义（OneShot）

每个 Task 为一个 JSONL 样本，从 Apollo 代码库中真实提取：

```json
{
  "id": "task_001",
  "symbol": "NamespaceService.findNamespace",
  "type": "method",
  "file": "apollo-biz/src/main/java/com/ctrip/framework/apollo/biz/service/NamespaceService.java",
  "expected_line": 87,
  "difficulty": "medium",
  "note": "findNamespace 出现在多个 Service 中"
}
```

**新增字段：**
- `difficulty`：`easy` / `medium` / `hard`
- `note`：标注该样本的挑战点（与 Grep / RAG 的对比分析相关）

### 合法 Symbol 类型
- Class
- Interface
- Method
- Enum

### 难度定义

| 难度 | 特征 | 样本数 |
|---|---|---|
| `easy` | 全局唯一类名或方法名，单次搜索即可命中 | 20 |
| `medium` | 同名符号多个文件中出现，需结合类名/包名过滤 | 15 |
| `hard` | 方法重载 / 内部类 / 长文件 / 继承层次中的定义 | 10 |
| `edge` | 极端场景：注解处理器、泛型、lambda 内定义等 | 5 |

---

## 6. Agent 行为规范（硬约束）

### 通用
- 只允许 **1 次最终回答**
- 不允许猜测
- 回答必须是 `file_path:line_number`
- 使用 `deepseek-v4-flash`
- 文件路径相对于代码库根目录（即 `codebases/apollo/` 之后的相对路径）

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
- 预构建向量索引

❌ 禁止：
- 原生 grep
- 文件系统扫描

---

## 7. 成功判定（自动化）

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

## 8. 数据集（固定）

- **代码库**：Apollo 配置中心（Git Submodule）
- **规模**：510 个 Java 源文件（仅 src/main）
- **样本数**：50
- **任务文件**：`data/tasks.jsonl`
- **标注方式**：从 Apollo 源码手动选取 50 个符号，记录精确行号

---

## 9. 执行方式

```bash
# 1. 克隆仓库（含 submodule）
git clone --recurse-submodules <this-repo>

# 2. 设置环境变量
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_API_KEY="sk-xxxxxxxx"

# 3. 构建 RAG 索引（仅 RAG 模式需要）
uv run python -m src.eval.indexer \
  --codebase codebases/apollo \
  --index-dir data/index

# 4. 运行 Grep Agent
uv run python -m src.eval.runner \
  --mode grep \
  --codebase codebases/apollo \
  --tasks data/tasks.jsonl

# 5. 运行 RAG Agent
uv run python -m src.eval.runner \
  --mode rag \
  --codebase codebases/apollo \
  --index-dir data/index \
  --tasks data/tasks.jsonl
```

---

## 10. 输出（机器可读）

```text
results/
├── grep_summary.csv
├── rag_summary.csv
└── failures.jsonl
```

CSV 字段：
```csv
task_id,difficulty,symbol,success,steps,latency_ms,predicted_file,predicted_line
```

---

## 11. 禁止项（红线）

❌ 多轮对话  
❌ Agent Memory  
❌ Human-in-the-loop  
❌ 更换模型  
❌ 动态修改任务  

---

## 12. 结束条件

Eval 结束当且仅当：
- 50 个样本全部完成
- 无人工干预
- 结果可复现

---

> **本文件为 Agent 执行的唯一事实来源。**
