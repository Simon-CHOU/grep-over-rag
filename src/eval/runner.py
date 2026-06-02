# src/eval/runner.py
import csv
import json
import time
import sys
from pathlib import Path
from collections import Counter
from src.eval.classifier import classify, has_valid_result


class Runner:
    def __init__(self, agent_factory, mode="grep"):
        self.agent_factory = agent_factory
        self.mode = mode
        self.stats = {}

    def run(self, tasks_file: str, output_dir: str, num_runs: int = 1):
        tasks = self._load_tasks(tasks_file)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        raw_path = out / f"{self.mode}_raw.csv"
        agg_path = out / f"{self.mode}_aggregate.csv"
        failures_path = out / f"failures_{self.mode}.jsonl"

        raw_rows = []
        failures = []

        with open(raw_path, "w", newline="") as raw_f:
            raw_writer = csv.DictWriter(raw_f, fieldnames=[
                "run_id", "task_id", "difficulty", "symbol", "type",
                "expected_file", "expected_line",
                "predicted_file", "predicted_line",
                "success", "error_type", "steps", "latency_ms",
            ])
            raw_writer.writeheader()

            total_latency = 0
            total_steps = 0
            total_runs = 0
            success_runs = 0

            for task in tasks:
                task_runs = []
                for run_idx in range(num_runs):
                    run_id = f"{task['id']}_r{run_idx + 1}"
                    agent = self.agent_factory()

                    start = time.perf_counter()
                    try:
                        result = agent.run(symbol=task["symbol"], symbol_type=task["type"])
                        latency_ms = int((time.perf_counter() - start) * 1000)
                        error_type = classify(
                            result.file_path, result.line_number,
                            task["file"], task["expected_line"],
                        )
                        success = error_type == "correct"
                        row = {
                            "run_id": run_id,
                            "task_id": task["id"],
                            "difficulty": task.get("difficulty", ""),
                            "symbol": task["symbol"],
                            "type": task["type"],
                            "expected_file": task["file"],
                            "expected_line": task["expected_line"],
                            "predicted_file": result.file_path,
                            "predicted_line": result.line_number,
                            "success": str(success).lower(),
                            "error_type": error_type,
                            "steps": result.steps,
                            "latency_ms": latency_ms,
                        }
                    except Exception as e:
                        latency_ms = int((time.perf_counter() - start) * 1000)
                        error_type = "api_error"
                        row = {
                            "run_id": run_id,
                            "task_id": task["id"],
                            "difficulty": task.get("difficulty", ""),
                            "symbol": task["symbol"],
                            "type": task["type"],
                            "expected_file": task["file"],
                            "expected_line": task["expected_line"],
                            "predicted_file": "",
                            "predicted_line": 0,
                            "success": "false",
                            "error_type": error_type,
                            "steps": 0,
                            "latency_ms": latency_ms,
                        }
                    raw_writer.writerow(row)
                    raw_rows.append(row)

                    if error_type != "correct" and error_type != "api_error":
                        failures.append(row)

                    total_latency += row["latency_ms"]
                    total_steps += row["steps"]
                    total_runs += 1
                    if row["success"] == "true":
                        success_runs += 1
                    task_runs.append(row)

        with open(agg_path, "w", newline="") as agg_f:
            agg_writer = csv.DictWriter(agg_f, fieldnames=[
                "task_id", "difficulty", "symbol",
                "success_count", "success_rate",
                "mean_steps", "std_steps",
                "mean_latency_ms", "std_latency_ms",
                "failure_pattern", "dominant_error_type",
            ])
            agg_writer.writeheader()

            for task in tasks:
                task_rows = [r for r in raw_rows if r["task_id"] == task["id"]]
                successes = [r for r in task_rows if r["success"] == "true"]
                sc = len(successes)
                sr = sc / num_runs if num_runs > 0 else 0
                steps = [r["steps"] for r in task_rows]
                lats = [r["latency_ms"] for r in task_rows]
                errors = [r["error_type"] for r in task_rows if r["error_type"] != "correct"]

                mean_steps = sum(steps) / len(steps) if steps else 0
                mean_lat = sum(lats) / len(lats) if lats else 0

                if sc == num_runs:
                    pattern = "always_passes"
                elif sc == 0:
                    pattern = "always_fails"
                else:
                    pattern = "sporadic"

                dominant = Counter(errors).most_common(1)
                dominant_error = dominant[0][0] if dominant else "n/a"

                agg_writer.writerow({
                    "task_id": task["id"],
                    "difficulty": task.get("difficulty", ""),
                    "symbol": task["symbol"],
                    "success_count": sc,
                    "success_rate": str(sr),
                    "mean_steps": str(mean_steps),
                    "std_steps": "",
                    "mean_latency_ms": str(mean_lat),
                    "std_latency_ms": "",
                    "failure_pattern": pattern,
                    "dominant_error_type": dominant_error,
                })

        if failures:
            with open(failures_path, "w") as f:
                for fail in failures:
                    f.write(json.dumps(fail, ensure_ascii=False) + "\n")

        self.stats = {
            "total_runs": total_runs,
            "total_tasks": len(tasks),
            "num_runs_per_task": num_runs,
            "success_runs": success_runs,
            "failed_runs": total_runs - success_runs,
            "success_rate": success_runs / total_runs if total_runs > 0 else 0,
            "avg_steps": total_steps / total_runs if total_runs > 0 else 0,
            "avg_latency_ms": total_latency / total_runs if total_runs > 0 else 0,
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
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--codebase", default="codebases/apollo")
    parser.add_argument("--tasks", default="data/tasks.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

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
        output_dir=args.output_dir,
        num_runs=args.runs,
    )

    print(f"\n=== {args.mode.upper()} Results ===")
    print(f"Tasks: {runner.stats['total_tasks']}")
    print(f"Runs per task: {runner.stats['num_runs_per_task']}")
    print(f"Total runs: {runner.stats['total_runs']}")
    print(f"Success runs: {runner.stats['success_runs']}")
    print(f"Failed runs: {runner.stats['failed_runs']}")
    print(f"Success Rate: {runner.stats['success_rate']:.1%}")
    print(f"Avg Steps: {runner.stats['avg_steps']:.1f}")
    print(f"Avg Latency: {runner.stats['avg_latency_ms']:.0f}ms")
    print(f"Raw CSV: {args.output_dir}/{args.mode}_raw.csv")
    print(f"Aggregate CSV: {args.output_dir}/{args.mode}_aggregate.csv")


if __name__ == "__main__":
    main()
