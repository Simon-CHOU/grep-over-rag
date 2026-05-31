#!/usr/bin/env bash
set -euo pipefail

RUNS="${1:-5}"
echo "=== Agent Eval: Grep vs RAG (${RUNS} runs per task) ==="

# Step 1: Build RAG index
echo ""
echo "--- Step 1: Building RAG index ---"
uv run python scripts/build_index.py

# Step 2: Run Grep Agent
echo ""
echo "--- Step 2: Grep Agent (${RUNS} runs) ---"
uv run python -m src.eval.runner --mode grep --runs "${RUNS}"

# Step 3: Run RAG Agent
echo ""
echo "--- Step 3: RAG Agent (${RUNS} runs) ---"
uv run python -m src.eval.runner --mode rag --runs "${RUNS}"

# Step 4: Statistical report
echo ""
echo "--- Step 4: Statistical Report ---"
uv run python -m src.eval.stats results/

# Step 5: Case studies
echo ""
echo "--- Step 5: Extracting Case Studies ---"
uv run python scripts/extract_cases.py results/ --top 5

echo ""
echo "=== Experiment Complete ==="
echo "Results: results/"
echo "  grep_raw.csv, rag_raw.csv"
echo "  grep_aggregate.csv, rag_aggregate.csv"
echo "  comparison.csv, summary.json"
echo "  case_studies/"
