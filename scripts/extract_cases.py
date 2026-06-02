#!/usr/bin/env python3
"""
scripts/extract_cases.py -- Extract compelling failure/success case studies
from raw CSV outputs of the agent eval runner.

Usage:
    uv run python scripts/extract_cases.py results/ --top 5

Output:
    results/case_studies/case_01.md through case_05.md

Selection algorithm (priority-based):
    1. Divergence: grep 100% success, RAG 0% success
    2. RAG error_type = wrong_file with delta > 30%
    3. RAG error_type = wrong_line with delta > 0
    4. At least 1 RAG-wins case for balance
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def load_raw(results_dir: str) -> dict[str, list[dict]]:
    """Load raw CSV files for both grep and rag modes.

    Returns {"grep": [row_dict, ...], "rag": [row_dict, ...]}.
    """
    data: dict[str, list[dict]] = {}
    for mode in ("grep", "rag"):
        path = Path(results_dir) / f"{mode}_raw.csv"
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            data[mode] = []
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            data[mode] = rows
            print(f"  Loaded {len(rows)} rows from {path}")
    return data


def group_by_task(rows: list[dict]) -> dict[str, list[dict]]:
    """Group raw rows by task_id."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        tid = row["task_id"]
        groups.setdefault(tid, []).append(row)
    return groups


def compute_task_metrics(grep_rows: list[dict], rag_rows: list[dict]) -> dict:
    """Compute comparison metrics for a task across grep and RAG runs.

    Returns a dict with aggregated success rates, dominant error types,
    and representative rows for markdown generation.
    """
    grep_successes = sum(1 for r in grep_rows if r["success"] == "true")
    rag_successes = sum(1 for r in rag_rows if r["success"] == "true")
    grep_total = len(grep_rows) or 1
    rag_total = len(rag_rows) or 1
    grep_sr = grep_successes / grep_total
    rag_sr = rag_successes / rag_total
    delta = grep_sr - rag_sr

    grep_errors = [r["error_type"] for r in grep_rows if r["error_type"] != "correct"]
    rag_errors = [r["error_type"] for r in rag_rows if r["error_type"] != "correct"]

    dominant_grep_counter = Counter(grep_errors).most_common(1)
    dominant_rag_counter = Counter(rag_errors).most_common(1)

    dominant_grep = dominant_grep_counter[0][0] if dominant_grep_counter else "correct"
    dominant_rag = dominant_rag_counter[0][0] if dominant_rag_counter else "correct"

    # Use first row for symbol info (consistent across runs)
    ref = (grep_rows or rag_rows)[0]

    return {
        "task_id": ref["task_id"],
        "symbol": ref["symbol"],
        "type": ref["type"],
        "expected_file": ref["expected_file"],
        "expected_line": int(ref["expected_line"]),
        "difficulty": ref.get("difficulty", ""),
        "grep_total_runs": grep_total,
        "grep_success_count": grep_successes,
        "grep_success_rate": grep_sr,
        "grep_dominant_error": dominant_grep,
        "rag_total_runs": rag_total,
        "rag_success_count": rag_successes,
        "rag_success_rate": rag_sr,
        "rag_dominant_error": dominant_rag,
        "delta": delta,
        # Representative rows for markdown generation
        "grep_sample": grep_rows[0] if grep_rows else None,
        "rag_sample": rag_rows[0] if rag_rows else None,
    }


def compute_priority(metrics: dict) -> tuple:
    """Compute priority tuple for case selection sorting.

    Returns (priority_group, sort_key_within_group)
    where a lower priority_group value = higher overall priority.

    Priority groups:
        1: Divergence (grep 100%, RAG 0%)
        2: RAG wrong_file with delta > 0.30
        3: RAG wrong_line with delta > 0
        4: RAG wins (delta < 0) — for balance
        5: Everything else
    """
    delta = metrics["delta"]
    rag_error = metrics["rag_dominant_error"]

    # Priority 1: Divergence — grep always succeeds, RAG never does
    if metrics["grep_success_rate"] == 1.0 and metrics["rag_success_rate"] == 0.0:
        return (1, -delta)

    # Priority 2: RAG wrong_file with meaningful gap
    if rag_error == "wrong_file" and delta > 0.30:
        return (2, -delta)

    # Priority 3: RAG wrong_line with any gap
    if rag_error == "wrong_line" and delta > 0:
        return (3, -delta)

    # Priority 4: RAG outperforms grep (interesting counter-example)
    if delta < 0:
        return (4, delta)

    # Priority 5: remaining cases
    return (5, -delta)


def select_cases(
    grep_data: list[dict], rag_data: list[dict], top_n: int = 5
) -> list[dict]:
    """Select the most compelling case studies from the raw data.

    Uses priority-based scoring and ensures at least one RAG-wins case.
    """
    grep_by_task = group_by_task(grep_data)
    rag_by_task = group_by_task(rag_data)

    common_tasks = set(grep_by_task.keys()) & set(rag_by_task.keys())
    if not common_tasks:
        print("Error: No common tasks between grep and rag results", file=sys.stderr)
        sys.exit(1)

    # Compute metrics for each common task
    all_metrics = []
    for tid in sorted(common_tasks):
        metrics = compute_task_metrics(grep_by_task[tid], rag_by_task[tid])
        all_metrics.append(metrics)

    print(f"  Computed metrics for {len(all_metrics)} common tasks")

    # Sort by priority
    scored = [(compute_priority(m), m) for m in all_metrics]
    scored.sort(key=lambda x: x[0])
    sorted_metrics = [m for _, m in scored]

    # Select top_n
    selected = sorted_metrics[:top_n]

    # Ensure at least one RAG-wins case
    has_rag_wins = any(m["delta"] < 0 for m in selected)
    if not has_rag_wins:
        rag_wins = [m for m in all_metrics if m["delta"] < 0]
        if rag_wins:
            rag_wins.sort(key=lambda m: m["delta"])  # most negative = biggest RAG win
            # Replace the lowest-priority selected item
            selected[-1] = rag_wins[0]
            # Re-sort by priority
            selected.sort(key=lambda m: compute_priority(m))

    return selected[:top_n]


def generate_markdown(metrics: dict, case_num: int) -> str:
    """Generate a case study markdown file for the given task metrics."""
    lines = []
    lines.append(f"# Case Study {case_num:02d}: {metrics['symbol']}\n")
    lines.append("\n")

    # --- Symbol Information ---
    lines.append("## Symbol Information\n")
    lines.append("\n")
    lines.append(f"- **Task ID:** {metrics['task_id']}\n")
    lines.append(f"- **Symbol:** `{metrics['symbol']}`\n")
    lines.append(f"- **Type:** {metrics['type']}\n")
    lines.append(f"- **Difficulty:** {metrics['difficulty']}\n")
    lines.append(
        f"- **Expected Location:** `{metrics['expected_file']}:{metrics['expected_line']}`\n"
    )
    lines.append("\n")

    # --- Comparison Table ---
    lines.append("## Comparison Table\n")
    lines.append("\n")
    lines.append("| Metric | Grep | RAG |\n")
    lines.append("|---|---|---|\n")
    lines.append(
        f"| Success Rate | {metrics['grep_success_rate']:.0%} "
        f"({metrics['grep_success_count']}/{metrics['grep_total_runs']}) "
        f"| {metrics['rag_success_rate']:.0%} "
        f"({metrics['rag_success_count']}/{metrics['rag_total_runs']}) |\n"
    )
    lines.append(
        f"| Dominant Error | {metrics['grep_dominant_error']} "
        f"| {metrics['rag_dominant_error']} |\n"
    )
    lines.append(f"| Delta (Grep - RAG) | {metrics['delta']:+.0%} | |\n")
    lines.append("\n")

    # --- Representative Run Details ---
    lines.append("## Representative Run Details\n")
    lines.append("\n")

    sample = metrics.get("grep_sample")
    if sample:
        lines.append(
            f"- **Grep prediction:** "
            f"`{sample.get('predicted_file', 'N/A')}:{sample.get('predicted_line', 'N/A')}`\n"
        )
        lines.append(f"- **Grep steps:** {sample.get('steps', 'N/A')}\n")
        lines.append(f"- **Grep latency:** {sample.get('latency_ms', 'N/A')}ms\n")

    sample = metrics.get("rag_sample")
    if sample:
        lines.append(
            f"- **RAG prediction:** "
            f"`{sample.get('predicted_file', 'N/A')}:{sample.get('predicted_line', 'N/A')}`\n"
        )
        lines.append(f"- **RAG steps:** {sample.get('steps', 'N/A')}\n")
        lines.append(f"- **RAG latency:** {sample.get('latency_ms', 'N/A')}ms\n")

    lines.append("\n")

    # --- Failure Analysis ---
    lines.append("## Failure Analysis\n")
    lines.append("\n")

    if metrics["rag_dominant_error"] != "correct":
        rag_error = metrics["rag_dominant_error"]
        if rag_error == "wrong_file":
            lines.append(
                "**RAG consistently retrieves the wrong file.** "
                "The agent's vector search likely returns a semantically similar "
                "but incorrect file, failing to distinguish between related symbols "
                "that share context or naming patterns.\n"
            )
        elif rag_error == "wrong_line":
            lines.append(
                "**RAG finds the right file but pinpoints the wrong line.** "
                "The embedding-based chunk retrieval may not align with exact "
                "definition boundaries, causing the agent to land on a nearby "
                "but incorrect line.\n"
            )
        elif rag_error == "empty_result":
            lines.append(
                "**RAG produces empty results.** "
                "The agent may fail to construct a valid query or the index "
                "may not contain relevant matches for this symbol.\n"
            )
        elif rag_error == "api_error":
            lines.append(
                "**RAG encountered an API error.** "
                "This may indicate a transient failure or a rate-limit issue "
                "during the evaluation run.\n"
            )
        else:
            lines.append(
                f"**RAG fails with error type: {rag_error}.**\n"
            )
    else:
        lines.append("RAG succeeded on this task.\n")

    if metrics["grep_dominant_error"] != "correct":
        lines.append("\n")
        lines.append(
            f"**Grep also struggles with this task** "
            f"(dominant error: {metrics['grep_dominant_error']}). "
            "This suggests the symbol is inherently difficult to locate "
            "via string matching — perhaps it uses a common name or "
            "appears in many files.\n"
        )

    lines.append("\n")

    # --- Success Analysis ---
    lines.append("## Success Analysis\n")
    lines.append("\n")

    if metrics["delta"] > 0:
        lines.append(
            "**Why Grep wins:** Grep performs exact string matching, which "
            "is reliable when the symbol name is unique and follows a "
            "consistent naming convention. The `rg_search` tool pinpoints "
            "the definition line precisely with no ambiguity.\n"
        )
        lines.append("\n")
        lines.append(
            "**Why RAG loses:** Semantic search can retrieve contextually "
            "related content from the wrong file or miss the exact definition "
            "boundary. The embedding model groups similar concepts together, "
            "which can lead to file-level confusion when multiple classes "
            "share related responsibilities.\n"
        )
    elif metrics["delta"] < 0:
        lines.append(
            "**Why RAG wins:** RAG's semantic understanding allows it to "
            "locate symbols even when the exact name doesn't match a unique "
            "pattern. This is especially valuable for symbols with generic "
            "names, overloaded methods, or when the definition uses "
            "different naming conventions than references.\n"
        )
        lines.append("\n")
        lines.append(
            "**Why Grep loses:** Grep relies on exact pattern matching. "
            "If the symbol name appears in variable references, comments, "
            "or unrelated methods, the agent may be distracted by false "
            "positives and pick the wrong occurrence.\n"
        )
    else:
        lines.append(
            "Both agents perform equally on this task, suggesting the "
            "symbol is straightforward to locate regardless of approach.\n"
        )

    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract failure-mode case studies from raw CSV results"
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing grep_raw.csv and rag_raw.csv",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of case studies to generate (default: 5)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: {results_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Load raw data
    data = load_raw(str(results_dir))

    if not data["grep"]:
        print("Error: grep_raw.csv is missing or empty", file=sys.stderr)
        sys.exit(1)
    if not data["rag"]:
        print("Error: rag_raw.csv is missing or empty", file=sys.stderr)
        sys.exit(1)

    # Select cases
    selected = select_cases(data["grep"], data["rag"], top_n=args.top)

    if not selected:
        print("No case studies selected")
        sys.exit(1)

    # Generate markdown files
    out_dir = results_dir / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, metrics in enumerate(selected, 1):
        md = generate_markdown(metrics, i)
        case_path = out_dir / f"case_{i:02d}.md"
        case_path.write_text(md, encoding="utf-8")
        print(f"  Generated {case_path}")

    print(
        f"\nDone. {len(selected)} case studies written to {out_dir}/"
    )


if __name__ == "__main__":
    main()
