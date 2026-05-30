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
