"""src/eval/stats.py -- Statistical tests on aggregate eval CSVs.

Reads grep and rag aggregate CSVs produced by the runner, computes:

  - McNemar test (paired binary) comparing majority-vote outcomes per task
  - Paired bootstrap 95 % CI on success-rate delta (grep - rag)
  - Per-difficulty Fisher exact test (easy, medium, hard)
  - Failure-pattern distribution (wrong_file / wrong_line / empty_result)

Outputs ``comparison.csv`` (per-task paired outcomes) and
``summary.json`` (all statistical results).

Usage::

    uv run python -m src.eval.stats results/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2 as chi2_dist, fisher_exact


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ComparisonStats:
    """Container for comparison statistics between grep and rag modes.

    Attributes:
        num_runs: Number of runs per task.
        num_tasks: Number of tasks compared.
        mcnemar: McNemar test results dict.
        bootstrap_ci: Bootstrap CI results dict.
        per_difficulty_fisher: Per-difficulty Fisher exact test results.
        failure_patterns: Failure pattern distributions.
    """

    def __init__(
        self,
        num_runs: int,
        num_tasks: int,
        mcnemar: dict,
        bootstrap_ci: dict,
        per_difficulty_fisher: dict,
        failure_patterns: dict,
    ):
        self.num_runs = num_runs
        self.num_tasks = num_tasks
        self.mcnemar = mcnemar
        self.bootstrap_ci = bootstrap_ci
        self.per_difficulty_fisher = per_difficulty_fisher
        self.failure_patterns = failure_patterns


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def read_aggregate_csv(path: str) -> list[dict[str, Any]]:
    """Read an aggregate CSV and convert numeric columns."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["success_count"] = int(row["success_count"])
            row["success_rate"] = float(row["success_rate"])
            rows.append(row)
    return rows


def read_raw_csv(path: str) -> list[dict[str, Any]]:
    """Read a raw (per-run) CSV and convert relevant columns."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["success"] = row["success"].strip().lower() == "true"
            row["steps"] = int(row["steps"])
            row["latency_ms"] = int(row["latency_ms"])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_num_runs(agg_rows: list[dict]) -> int:
    """Infer *num_runs* from aggregate CSV data.

    Uses the relationship ``success_rate = success_count / num_runs``.
    Falls back to 1 when no useful row is found.
    """
    for row in agg_rows:
        sc = row["success_count"]
        sr = row["success_rate"]
        if sc > 0 and sr > 0:
            return round(sc / sr)
    return 1


def majority_pass(success_count: int, num_runs: int) -> bool:
    """Return True when *success_count* represents a strict majority."""
    return success_count > num_runs / 2


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def compute_mcnemar(
    grep_agg: list[dict],
    rag_agg: list[dict],
    num_runs: int,
) -> dict[str, Any]:
    """Continuity-corrected McNemar test on paired binary outcomes.

    Majority vote (> *num_runs* / 2) determines pass/fail per task.
    Returns the 2x2 contingency table, chi-squared statistic, p-value, and
    a significance flag.
    """
    grep_by_id = {r["task_id"]: r for r in grep_agg}
    rag_by_id = {r["task_id"]: r for r in rag_agg}

    a = b = c = d = 0
    for tid in grep_by_id:
        if tid not in rag_by_id:
            continue
        g_pass = majority_pass(grep_by_id[tid]["success_count"], num_runs)
        r_pass = majority_pass(rag_by_id[tid]["success_count"], num_runs)

        if g_pass and r_pass:
            a += 1
        elif g_pass and not r_pass:
            b += 1
        elif not g_pass and r_pass:
            c += 1
        else:
            d += 1

    n_discordant = b + c
    if n_discordant == 0:
        chi2_val = 0.0
    else:
        chi2_val = (abs(b - c) - 1.0) ** 2 / n_discordant

    p_value = 1.0 - chi2_dist.cdf(chi2_val, 1)

    return {
        "table": {"a": a, "b": b, "c": c, "d": d},
        "chi2": chi2_val,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
    }


def bootstrap_ci(
    grep_agg: list[dict],
    rag_agg: list[dict],
    num_runs: int,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap confidence interval on success-rate delta (grep - rag).

    For each task the per-mode success rate is ``success_count / num_runs``.
    The statistic of interest is the mean across tasks of
    ``rate_grep - rate_rag``.  Bootstrap resampling is seeded for
    reproducibility.
    """
    grep_by_id = {r["task_id"]: r for r in grep_agg}
    rag_by_id = {r["task_id"]: r for r in rag_agg}

    task_ids = [tid for tid in grep_by_id if tid in rag_by_id]
    deltas = np.array([
        grep_by_id[tid]["success_rate"] - rag_by_id[tid]["success_rate"]
        for tid in task_ids
    ])
    observed_mean = float(np.mean(deltas))

    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, alpha / 2 * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha / 2) * 100))

    return {
        "observed_mean_delta": observed_mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_bootstrap": n_bootstrap,
        "confidence_level": ci,
    }


def compute_fisher_exact(
    grep_agg: list[dict],
    rag_agg: list[dict],
    num_runs: int,
) -> dict[str, dict[str, Any]]:
    """Per-difficulty Fisher exact test on the paired 2x2 table.

    Returns a dict keyed by difficulty, each containing the contingency
    table, odds ratio, and p-value.
    """
    grep_by_id = {r["task_id"]: r for r in grep_agg}
    rag_by_id = {r["task_id"]: r for r in rag_agg}

    tables: dict[str, dict[str, int]] = {}
    for tid in grep_by_id:
        if tid not in rag_by_id:
            continue
        diff = grep_by_id[tid].get("difficulty", "")
        if not diff:
            continue
        if diff not in tables:
            tables[diff] = {"a": 0, "b": 0, "c": 0, "d": 0}

        g_pass = majority_pass(grep_by_id[tid]["success_count"], num_runs)
        r_pass = majority_pass(rag_by_id[tid]["success_count"], num_runs)

        if g_pass and r_pass:
            tables[diff]["a"] += 1
        elif g_pass and not r_pass:
            tables[diff]["b"] += 1
        elif not g_pass and r_pass:
            tables[diff]["c"] += 1
        else:
            tables[diff]["d"] += 1

    results: dict[str, dict] = {}
    for diff, tbl in sorted(tables.items()):
        contingency = [[tbl["a"], tbl["b"]], [tbl["c"], tbl["d"]]]
        if tbl["a"] + tbl["b"] == 0 or tbl["c"] + tbl["d"] == 0:
            odds_ratio_val = None
            p_val = 1.0
        else:
            odds_ratio_val, p_val = fisher_exact(contingency)

        results[diff] = {
            "table": tbl,
            "odds_ratio": float(odds_ratio_val) if odds_ratio_val is not None else None,
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05) if p_val is not None else False,
        }

    return results


def compute_failure_patterns(
    grep_raw: list[dict],
    rag_raw: list[dict],
) -> dict[str, dict[str, dict]]:
    """Distribution of failure error types for each mode.

    Counts every run whose ``success`` is False, grouped by ``error_type``.
    Returns a dict like::

        {"grep": {"wrong_file": {"count": 5, "percentage": 55.6}, ...},
         "rag":  {"wrong_file": {"count": 3, "percentage": 50.0}, ...}}
    """
    result: dict[str, dict[str, dict]] = {"grep": {}, "rag": {}}

    for mode, rows in [("grep", grep_raw), ("rag", rag_raw)]:
        counter: Counter[str] = Counter()
        for r in rows:
            if not r["success"]:
                counter[r["error_type"]] += 1
        total = sum(counter.values())
        if total > 0:
            for err_type in sorted(counter):
                cnt = counter[err_type]
                result[mode][err_type] = {
                    "count": cnt,
                    "percentage": round(cnt / total * 100, 2),
                }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point.

    Usage::

        uv run python -m src.eval.stats results/
    """
    parser = argparse.ArgumentParser(
        description="Statistical tests on agent eval aggregate CSVs"
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing grep/rag aggregate and raw CSVs",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f"Error: not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # ---- read data -------------------------------------------------------
    grep_agg = read_aggregate_csv(str(results_dir / "grep_aggregate.csv"))
    rag_agg = read_aggregate_csv(str(results_dir / "rag_aggregate.csv"))
    grep_raw = read_raw_csv(str(results_dir / "grep_raw.csv"))
    rag_raw = read_raw_csv(str(results_dir / "rag_raw.csv"))

    num_runs = infer_num_runs(grep_agg + rag_agg)

    # ---- compute statistics ----------------------------------------------
    mcnemar = compute_mcnemar(grep_agg, rag_agg, num_runs)
    boot = bootstrap_ci(grep_agg, rag_agg, num_runs)
    fisher = compute_fisher_exact(grep_agg, rag_agg, num_runs)
    fail_pats = compute_failure_patterns(grep_raw, rag_raw)

    # ---- write comparison.csv --------------------------------------------
    grep_by_id = {r["task_id"]: r for r in grep_agg}
    rag_by_id = {r["task_id"]: r for r in rag_agg}
    all_tids = sorted(tid for tid in grep_by_id if tid in rag_by_id)

    cmp_path = results_dir / "comparison.csv"
    with open(cmp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id", "difficulty", "symbol",
            "grep_success", "rag_success",
            "grep_success_rate", "rag_success_rate",
        ])
        for tid in all_tids:
            g = grep_by_id[tid]
            r = rag_by_id[tid]
            g_pass = majority_pass(g["success_count"], num_runs)
            r_pass = majority_pass(r["success_count"], num_runs)
            writer.writerow([
                tid,
                g.get("difficulty", ""),
                g.get("symbol", ""),
                str(g_pass).lower(),
                str(r_pass).lower(),
                g["success_rate"],
                r["success_rate"],
            ])

    # ---- write summary.json ----------------------------------------------
    summary: dict[str, Any] = {
        "num_runs": num_runs,
        "num_tasks": len(all_tids),
        "mcnemar_test": mcnemar,
        "bootstrap_ci": boot,
        "per_difficulty_fisher": fisher,
        "failure_patterns": fail_pats,
    }

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- console report --------------------------------------------------
    print(f"Results directory: {results_dir}")
    print(f"Tasks: {summary['num_tasks']}, Runs per task: {num_runs}")
    print()
    print(f"McNemar test:")
    print(f"  table:      {mcnemar['table']}")
    print(f"  chi-squared: {mcnemar['chi2']:.4f}")
    print(f"  p-value:    {mcnemar['p_value']:.4f}")
    print(f"  significant: {mcnemar['significant']}")
    print()
    print(f"Bootstrap 95% CI (grep - rag):")
    print(f"  observed mean delta: {boot['observed_mean_delta']:.4f}")
    print(f"  [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
    print()
    print(f"Comparison CSV: {cmp_path}")
    print(f"Summary JSON:   {summary_path}")


if __name__ == "__main__":
    main()
