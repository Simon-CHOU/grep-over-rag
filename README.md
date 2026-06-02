# Grep vs RAG — Agent Eval

评估 Grep（文本搜索）vs RAG（向量检索）两种方式在**精确 Java 符号定位**任务上的表现差异。

核心论点：Grep 失败是"找不到"（可感知），RAG 失败是"找错"（误导）——对于 2B Agent Orchestration 场景，silent failure 优于 misleading answer。

被测代码库：[Apollo 配置中心](https://github.com/apolloconfig/apollo)（携程开源），510 个主源码 Java 文件。

## Quick Startup

```bash
# 1. 克隆仓库（含 Apollo 子模块）
git clone --recurse-submodules https://github.com/Simon-CHOU/grep-over-rag.git
cd grep-over-rag

# 2. 安装依赖
uv sync

# 3. 配置 LLM 环境变量
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_API_KEY="sk-xxxxxxxx"

# 4. 运行完整实验（50 tasks × 2 modes × 5 runs = 500 次 Agent 调用）
bash scripts/run_experiment.sh 5

# 5. 查看结果
cat results/summary.json
ls results/case_studies/
```

## 输出

```
results/
├── grep_raw.csv              # Grep 每次运行明细 (250 rows)
├── rag_raw.csv               # RAG 每次运行明细 (250 rows)
├── grep_aggregate.csv        # Grep 每 task 聚合 (50 rows)
├── rag_aggregate.csv         # RAG 每 task 聚合 (50 rows)
├── comparison.csv            # Grep vs RAG head-to-head
├── summary.json              # 统计检验结论
└── case_studies/             # 5 篇失败案例分析
    ├── case_01.md
    └── ...
```

## 项目结构

```
src/eval/
├── grep_agent.py             # Grep Agent (ripgrep + LLM)
├── rag_agent.py              # RAG Agent (FAISS + LLM)
├── indexer.py                # 向量索引构建
├── runner.py                 # 实验编排 (multi-run)
├── classifier.py             # 失败模式分类
├── stats.py                  # 统计检验
└── scorer.py                 # 精度判定
data/
└── tasks.jsonl               # 50 个标注任务
```
