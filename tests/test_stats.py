"""Tests for src/eval/stats.py -- statistical tests on aggregate eval CSVs."""

import json
import csv
import pytest
import numpy as np
from pathlib import Path
from io import StringIO
from src.eval.stats import (
    read_aggregate_csv,
    read_raw_csv,
    infer_num_runs,
    majority_pass,
    compute_mcnemar,
    bootstrap_ci,
    compute_fisher_exact,
    compute_failure_patterns,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AGG_FIELDS = [
    "task_id", "difficulty", "symbol",
    "success_count", "success_rate",
    "mean_steps", "std_steps",
    "mean_latency_ms", "std_latency_ms",
    "failure_pattern", "dominant_error_type",
]

RAW_FIELDS = [
    "run_id", "task_id", "difficulty", "symbol", "type",
    "expected_file", "expected_line",
    "predicted_file", "predicted_line",
    "success", "error_type", "steps", "latency_ms",
]


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _make_agg_rows():
    """6 tasks: 2 easy, 2 medium, 2 hard, num_runs=3.

    Returns types matching what ``read_aggregate_csv`` would emit
    (``success_count`` int, ``success_rate`` float).
    """
    return [
        {"task_id": "t1", "difficulty": "easy",   "symbol": "Foo.bar",     "success_count": 3, "success_rate": 1.0,                 "mean_steps": "1", "std_steps": "", "mean_latency_ms": "100", "std_latency_ms": "", "failure_pattern": "always_passes", "dominant_error_type": "n/a"},
        {"task_id": "t2", "difficulty": "easy",   "symbol": "Baz.qux",     "success_count": 2, "success_rate": 2/3,                  "mean_steps": "2", "std_steps": "", "mean_latency_ms": "200", "std_latency_ms": "", "failure_pattern": "sporadic",     "dominant_error_type": "wrong_line"},
        {"task_id": "t3", "difficulty": "easy",   "symbol": "Qux.corge",   "success_count": 0, "success_rate": 0.0,                 "mean_steps": "4", "std_steps": "", "mean_latency_ms": "300", "std_latency_ms": "", "failure_pattern": "always_fails", "dominant_error_type": "wrong_file"},
        {"task_id": "t4", "difficulty": "medium", "symbol": "Grault.garply","success_count": 3, "success_rate": 1.0,                "mean_steps": "1", "std_steps": "", "mean_latency_ms": "150", "std_latency_ms": "", "failure_pattern": "always_passes", "dominant_error_type": "n/a"},
        {"task_id": "t5", "difficulty": "medium", "symbol": "Waldo.thud",  "success_count": 1, "success_rate": 1/3,                  "mean_steps": "3", "std_steps": "", "mean_latency_ms": "250", "std_latency_ms": "", "failure_pattern": "sporadic",     "dominant_error_type": "wrong_file"},
        {"task_id": "t6", "difficulty": "hard",   "symbol": "Plugh.xyzzy", "success_count": 0, "success_rate": 0.0,                 "mean_steps": "4", "std_steps": "", "mean_latency_ms": "350", "std_latency_ms": "", "failure_pattern": "always_fails", "dominant_error_type": "empty_result"},
    ]


def _make_raw_rows():
    """18 raw rows (6 tasks x 3 runs) consistent with _make_agg_rows().

    Returns types matching what ``read_raw_csv`` would emit
    (``success`` bool, ``steps`` int, ``latency_ms`` int).

    t1: 3 correct
    t2: 2 correct, 1 wrong_line
    t3: 3 wrong_file
    t4: 3 correct
    t5: 1 correct, 2 wrong_file
    t6: 3 empty_result
    """
    spec = [
        ("t1", "easy",   "Foo.bar",     True,  "correct",     "Foo.java",   "10"),
        ("t1", "easy",   "Foo.bar",     True,  "correct",     "Foo.java",   "10"),
        ("t1", "easy",   "Foo.bar",     True,  "correct",     "Foo.java",   "10"),
        ("t2", "easy",   "Baz.qux",     True,  "correct",     "Baz.java",   "3"),
        ("t2", "easy",   "Baz.qux",     True,  "correct",     "Baz.java",   "3"),
        ("t2", "easy",   "Baz.qux",     False, "wrong_line",  "Baz.java",   "99"),
        ("t3", "easy",   "Qux.corge",   False, "wrong_file",  "Other.java", "0"),
        ("t3", "easy",   "Qux.corge",   False, "wrong_file",  "Other.java", "0"),
        ("t3", "easy",   "Qux.corge",   False, "wrong_file",  "Other.java", "0"),
        ("t4", "medium", "Grault.garply",True, "correct",     "Grault.java","42"),
        ("t4", "medium", "Grault.garply",True, "correct",     "Grault.java","42"),
        ("t4", "medium", "Grault.garply",True, "correct",     "Grault.java","42"),
        ("t5", "medium", "Waldo.thud",  True,  "correct",     "Waldo.java", "15"),
        ("t5", "medium", "Waldo.thud",  False, "wrong_file",  "Other.java", "0"),
        ("t5", "medium", "Waldo.thud",  False, "wrong_file",  "Other.java", "0"),
        ("t6", "hard",   "Plugh.xyzzy", False, "empty_result","",           "0"),
        ("t6", "hard",   "Plugh.xyzzy", False, "empty_result","",           "0"),
        ("t6", "hard",   "Plugh.xyzzy", False, "empty_result","",           "0"),
    ]
    rows = []
    for run_idx, (tid, diff, sym, succ, err, pf, pl) in enumerate(spec, start=1):
        rows.append({
            "run_id": f"{tid}_r{((run_idx - 1) % 3) + 1}",
            "task_id": tid,
            "difficulty": diff,
            "symbol": sym,
            "type": "method",
            "expected_file": f"{sym.split('.')[0]}.java",
            "expected_line": pl,
            "predicted_file": pf,
            "predicted_line": pl if succ else "0",
            "success": succ,
            "error_type": err,
            "steps": 1 if succ else 4,
            "latency_ms": 100,
        })
    return rows


# ===================================================================
# 1. CSV reading
# ===================================================================

class TestReadAggregateCsv:
    def test_reads_all_rows(self, tmp_path):
        path = tmp_path / "grep_aggregate.csv"
        _write_csv(path, AGG_FIELDS, _make_agg_rows())
        result = read_aggregate_csv(str(path))
        assert len(result) == 6

    def test_type_conversion(self, tmp_path):
        rows = [{"task_id": "t1", "difficulty": "easy", "symbol": "F.b",
                 "success_count": "2", "success_rate": "0.6666666666666666",
                 "mean_steps": "2", "std_steps": "", "mean_latency_ms": "200",
                 "std_latency_ms": "", "failure_pattern": "sporadic",
                 "dominant_error_type": "wrong_line"}]
        _write_csv(tmp_path / "a.csv", AGG_FIELDS, rows)
        result = read_aggregate_csv(str(tmp_path / "a.csv"))
        assert result[0]["success_count"] == 2
        assert isinstance(result[0]["success_count"], int)
        assert isinstance(result[0]["success_rate"], float)
        assert result[0]["success_rate"] == pytest.approx(2 / 3)


class TestReadRawCsv:
    def test_reads_all_rows(self, tmp_path):
        _write_csv(tmp_path / "r.csv", RAW_FIELDS, _make_raw_rows())
        result = read_raw_csv(str(tmp_path / "r.csv"))
        assert len(result) == 18

    def test_type_conversion(self, tmp_path):
        rows = [{"run_id": "t1_r1", "task_id": "t1", "difficulty": "easy",
                 "symbol": "F.b", "type": "method",
                 "expected_file": "F.java", "expected_line": "10",
                 "predicted_file": "F.java", "predicted_line": "10",
                 "success": "true", "error_type": "correct",
                 "steps": "1", "latency_ms": "100"}]
        _write_csv(tmp_path / "r.csv", RAW_FIELDS, rows)
        result = read_raw_csv(str(tmp_path / "r.csv"))
        assert result[0]["success"] is True
        assert isinstance(result[0]["steps"], int)
        assert isinstance(result[0]["latency_ms"], int)

    def test_false_string_converted(self, tmp_path):
        rows = [{"run_id": "t1_r1", "task_id": "t1", "difficulty": "easy",
                 "symbol": "F.b", "type": "method",
                 "expected_file": "F.java", "expected_line": "10",
                 "predicted_file": "", "predicted_line": "0",
                 "success": "false", "error_type": "wrong_file",
                 "steps": "4", "latency_ms": "200"}]
        _write_csv(tmp_path / "r.csv", RAW_FIELDS, rows)
        result = read_raw_csv(str(tmp_path / "r.csv"))
        assert result[0]["success"] is False


# ===================================================================
# 2. Helper functions
# ===================================================================

class TestHelperFunctions:
    def test_infer_num_runs_from_success_rate(self):
        rows = [{"success_count": 3, "success_rate": 1.0},
                {"success_count": 0, "success_rate": 0.0},
                {"success_count": 2, "success_rate": 2 / 3}]
        assert infer_num_runs(rows) == 3

    def test_infer_num_runs_fallback(self):
        rows = [{"success_count": 0, "success_rate": 0.0},
                {"success_count": 0, "success_rate": 0.0}]
        assert infer_num_runs(rows) == 1

    def test_majority_pass_odd(self):
        assert majority_pass(3, 5) is True
        assert majority_pass(2, 5) is False

    def test_majority_pass_even(self):
        assert majority_pass(3, 4) is True
        assert majority_pass(2, 4) is False
        assert majority_pass(1, 4) is False


# ===================================================================
# 3. McNemar test
# ===================================================================

class TestMcNemar:
    def test_perfect_agreement(self):
        """If both modes agree on every task -> a=3, d=3, b=c=0, chi2=0."""
        rows = _make_agg_rows()
        result = compute_mcnemar(rows, rows, num_runs=3)
        assert result["table"] == {"a": 3, "b": 0, "c": 0, "d": 3}
        assert result["chi2"] == 0.0
        assert result["p_value"] == 1.0

    def test_asymmetric(self):
        """grep all pass, rag 3 pass 3 fail (by majority).

        t1-t3 pass in rag, t4-t6 fail in rag.
        a=3, b=3, c=0, d=0.
        chi2 = (|3-0|-1)^2 / (3+0) = 4/3
        """
        g = _make_agg_rows()
        r_list = _make_agg_rows()
        for i in range(6):
            sc = 3 if i < 3 else 0
            r_list[i] = dict(r_list[i], success_count=sc)
        for i in range(6):
            g[i] = dict(g[i], success_count=3)
        result = compute_mcnemar(g, r_list, num_runs=3)
        assert result["table"] == {"a": 3, "b": 3, "c": 0, "d": 0}
        expected_chi2 = (abs(3 - 0) - 1) ** 2 / 3
        assert result["chi2"] == pytest.approx(expected_chi2)

    def test_discordant_pairs(self):
        """Create a mix of concordant and discordant pairs.

        t1: g=pass(3), r=pass(3)  -> a
        t2: g=pass(3), r=fail(1)  -> b
        t3: g=fail(1), r=pass(3)  -> c
        t4: g=fail(0), r=fail(0)  -> d

        a=1, b=1, c=1, d=1
        chi2 = (|1-1|-1)^2 / (1+1) = 1/2 = 0.5
        """
        g_rows = [
            {"task_id": "t1", "difficulty": "easy",   "symbol": "A", "success_count": 3, "success_rate": 1.0},
            {"task_id": "t2", "difficulty": "easy",   "symbol": "B", "success_count": 3, "success_rate": 1.0},
            {"task_id": "t3", "difficulty": "medium", "symbol": "C", "success_count": 1, "success_rate": 1/3},
            {"task_id": "t4", "difficulty": "medium", "symbol": "D", "success_count": 0, "success_rate": 0.0},
        ]
        r_rows = [
            {"task_id": "t1", "difficulty": "easy",   "symbol": "A", "success_count": 3, "success_rate": 1.0},
            {"task_id": "t2", "difficulty": "easy",   "symbol": "B", "success_count": 1, "success_rate": 1/3},
            {"task_id": "t3", "difficulty": "medium", "symbol": "C", "success_count": 3, "success_rate": 1.0},
            {"task_id": "t4", "difficulty": "medium", "symbol": "D", "success_count": 0, "success_rate": 0.0},
        ]
        result = compute_mcnemar(g_rows, r_rows, num_runs=3)
        assert result["table"] == {"a": 1, "b": 1, "c": 1, "d": 1}
        assert result["chi2"] == pytest.approx(0.5)

    def test_p_value_calculation(self):
        """chi2=0.5 with 1 df -> p = 1 - chi2.cdf(0.5, 1) ≈ 0.4795."""
        from scipy.stats import chi2
        g_rows = [
            {"task_id": "t1", "difficulty": "easy",   "symbol": "A", "success_count": 3, "success_rate": 1.0},
            {"task_id": "t2", "difficulty": "medium", "symbol": "B", "success_count": 3, "success_rate": 1.0},
            {"task_id": "t3", "difficulty": "easy",   "symbol": "C", "success_count": 1, "success_rate": 1/3},
            {"task_id": "t4", "difficulty": "hard",   "symbol": "D", "success_count": 0, "success_rate": 0.0},
        ]
        r_rows = [
            {"task_id": "t1", "difficulty": "easy",   "symbol": "A", "success_count": 3, "success_rate": 1.0},
            {"task_id": "t2", "difficulty": "medium", "symbol": "B", "success_count": 1, "success_rate": 1/3},
            {"task_id": "t3", "difficulty": "easy",   "symbol": "C", "success_count": 3, "success_rate": 1.0},
            {"task_id": "t4", "difficulty": "hard",   "symbol": "D", "success_count": 0, "success_rate": 0.0},
        ]
        result = compute_mcnemar(g_rows, r_rows, num_runs=3)
        expected_p = 1 - chi2.cdf(0.5, 1)
        assert result["p_value"] == pytest.approx(expected_p, abs=1e-4)


# ===================================================================
# 4. Bootstrap CI
# ===================================================================

class TestBootstrapCi:
    def test_returns_expected_keys(self):
        result = bootstrap_ci(_make_agg_rows(), _make_agg_rows(), num_runs=3, n_bootstrap=100, seed=42)
        assert "observed_mean_delta" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["observed_mean_delta"] <= result["ci_upper"]

    def test_zero_delta_when_identical(self):
        result = bootstrap_ci(_make_agg_rows(), _make_agg_rows(), num_runs=3, n_bootstrap=100, seed=42)
        assert result["observed_mean_delta"] == pytest.approx(0.0)

    def test_positive_delta_grep_better(self):
        g = [dict(r, success_count=3, success_rate=1.0) for r in _make_agg_rows()]
        r = [dict(r, success_count=0, success_rate=0.0) for r in _make_agg_rows()]
        result = bootstrap_ci(g, r, num_runs=3, n_bootstrap=200, seed=42)
        assert result["observed_mean_delta"] == pytest.approx(1.0)
        assert result["ci_lower"] > 0

    def test_reproducible_seed(self):
        r1 = bootstrap_ci(_make_agg_rows(), _make_agg_rows(), num_runs=3, n_bootstrap=200, seed=42)
        r2 = bootstrap_ci(_make_agg_rows(), _make_agg_rows(), num_runs=3, n_bootstrap=200, seed=42)
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]

    def test_partial_improvement(self):
        """3 tasks: grep improves on 2, ties on 1."""
        g = [
            {"task_id": "t1", "difficulty": "easy",   "symbol": "A", "success_count": 3, "success_rate": 1.0},
            {"task_id": "t2", "difficulty": "medium", "symbol": "B", "success_count": 3, "success_rate": 1.0},
            {"task_id": "t3", "difficulty": "hard",   "symbol": "C", "success_count": 0, "success_rate": 0.0},
        ]
        r = [
            {"task_id": "t1", "difficulty": "easy",   "symbol": "A", "success_count": 3, "success_rate": 1.0},
            {"task_id": "t2", "difficulty": "medium", "symbol": "B", "success_count": 0, "success_rate": 0.0},
            {"task_id": "t3", "difficulty": "hard",   "symbol": "C", "success_count": 0, "success_rate": 0.0},
        ]
        result = bootstrap_ci(g, r, num_runs=3, n_bootstrap=1000, seed=42)
        assert result["observed_mean_delta"] == pytest.approx(1 / 3)


# ===================================================================
# 5. Fisher exact test per difficulty
# ===================================================================

class TestFisherExact:
    def test_returns_per_difficulty(self):
        result = compute_fisher_exact(_make_agg_rows(), _make_agg_rows(), num_runs=3)
        assert set(result.keys()) == {"easy", "medium", "hard"}
        for diff in ("easy", "medium", "hard"):
            assert "odds_ratio" in result[diff]
            assert "p_value" in result[diff]
            assert "table" in result[diff]

    def test_default_data_tables(self):
        """With identical _make_agg_rows() rows for both:
        easy:   t1(a), t2(a), t3(d)           -> a=2, b=0, c=0, d=1
        medium: t4(a), t5(d)                   -> a=1, b=0, c=0, d=1
        hard:   t6(d)                          -> a=0, b=0, c=0, d=1
        """
        result = compute_fisher_exact(_make_agg_rows(), _make_agg_rows(), num_runs=3)
        assert result["easy"]["table"] == {"a": 2, "b": 0, "c": 0, "d": 1}
        assert result["medium"]["table"] == {"a": 1, "b": 0, "c": 0, "d": 1}
        assert result["hard"]["table"] == {"a": 0, "b": 0, "c": 0, "d": 1}

    def test_asymmetric_modes(self):
        """Rag makes t3 and t5 pass.
        easy:   t1(a), t2(a), t3(g=fail,r=pass=c)  -> a=2, b=0, c=1, d=0
        medium: t4(a), t5(g=fail,r=pass=c)          -> a=1, b=0, c=1, d=0
        hard:   t6(d)                                -> a=0, b=0, c=0, d=1
        """
        g_rows = _make_agg_rows()
        r_rows = _make_agg_rows()
        r_rows[2] = dict(r_rows[2], success_count=3, success_rate=1.0)
        r_rows[4] = dict(r_rows[4], success_count=3, success_rate=1.0)
        result = compute_fisher_exact(g_rows, r_rows, num_runs=3)
        assert result["easy"]["table"] == {"a": 2, "b": 0, "c": 1, "d": 0}
        assert result["medium"]["table"] == {"a": 1, "b": 0, "c": 1, "d": 0}
        assert result["hard"]["table"] == {"a": 0, "b": 0, "c": 0, "d": 1}


# ===================================================================
# 6. Failure pattern distribution
# ===================================================================

class TestFailurePatterns:
    def test_counts_match_raw_data(self):
        raw = _make_raw_rows()
        result = compute_failure_patterns(raw, raw)
        # Total failures: t2(1 wrong_line) + t3(3 wrong_file) + t5(2 wrong_file) + t6(3 empty_result) = 9
        assert result["grep"]["wrong_file"]["count"] == 5
        assert result["grep"]["wrong_line"]["count"] == 1
        assert result["grep"]["empty_result"]["count"] == 3

    def test_percentages(self):
        raw = _make_raw_rows()
        result = compute_failure_patterns(raw, [])
        fp = result["grep"]
        assert fp["wrong_file"]["percentage"] == pytest.approx(5 / 9 * 100, abs=0.1)
        assert fp["wrong_line"]["percentage"] == pytest.approx(1 / 9 * 100, abs=0.1)
        assert fp["empty_result"]["percentage"] == pytest.approx(3 / 9 * 100, abs=0.1)

    def test_empty_raw(self):
        result = compute_failure_patterns([], [])
        assert result == {"grep": {}, "rag": {}}

    def test_only_correct(self):
        raw = [{"success": True, "error_type": "correct"}]
        result = compute_failure_patterns(raw, raw)
        assert result["grep"] == {}
        assert result["rag"] == {}

    def test_rag_side_independent(self):
        """Rag raw can have a different failure distribution."""
        grep_raw = _make_raw_rows()
        rag_raw = [dict(r, error_type="wrong_line") for r in _make_raw_rows() if not r["success"]]
        result = compute_failure_patterns(grep_raw, rag_raw)
        assert result["grep"]["wrong_file"]["count"] == 5
        assert result["rag"]["wrong_line"]["count"] == 9


# ===================================================================
# 7. Integration: end-to-end CLI pipeline
# ===================================================================

class TestMainPipeline:
    def _run_main(self, tmp_path, grep_agg=None, rag_agg=None, grep_raw=None, rag_raw=None):
        _write_csv(tmp_path / "grep_aggregate.csv", AGG_FIELDS, grep_agg or _make_agg_rows())
        _write_csv(tmp_path / "rag_aggregate.csv", AGG_FIELDS, rag_agg or _make_agg_rows())
        _write_csv(tmp_path / "grep_raw.csv", RAW_FIELDS, grep_raw or _make_raw_rows())
        _write_csv(tmp_path / "rag_raw.csv", RAW_FIELDS, rag_raw or _make_raw_rows())

        old_argv = __import__("sys").argv
        old_stdout = __import__("sys").stdout
        try:
            __import__("sys").argv = ["stats.py", str(tmp_path)]
            __import__("sys").stdout = StringIO()
            main()
        finally:
            __import__("sys").argv = old_argv
            __import__("sys").stdout = old_stdout

    def test_writes_output_files(self, tmp_path):
        self._run_main(tmp_path)
        assert (tmp_path / "comparison.csv").exists()
        assert (tmp_path / "summary.json").exists()

    def test_comparison_csv_columns(self, tmp_path):
        grep_agg = _make_agg_rows()
        rag_agg = list(_make_agg_rows())
        rag_agg[2] = dict(rag_agg[2], success_count=3, success_rate=1.0)
        self._run_main(tmp_path, grep_agg=grep_agg, rag_agg=rag_agg)

        with open(tmp_path / "comparison.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 6
        assert reader.fieldnames == [
            "task_id", "difficulty", "symbol",
            "grep_success", "rag_success",
            "grep_success_rate", "rag_success_rate",
        ]
        # t3: grep fails (0/3), rag passes (3/3)
        t3 = [r for r in rows if r["task_id"] == "t3"][0]
        assert t3["grep_success"] == "false"
        assert t3["rag_success"] == "true"

    def test_summary_json_structure(self, tmp_path):
        self._run_main(tmp_path)
        with open(tmp_path / "summary.json") as f:
            summary = json.load(f)

        assert summary["num_runs"] == 3
        assert summary["num_tasks"] == 6
        assert "mcnemar_test" in summary
        assert "bootstrap_ci" in summary
        assert "per_difficulty_fisher" in summary
        assert "failure_patterns" in summary
        m = summary["mcnemar_test"]
        assert "table" in m and "chi2" in m and "p_value" in m and "significant" in m

    def test_failure_patterns_in_summary(self, tmp_path):
        self._run_main(tmp_path)
        with open(tmp_path / "summary.json") as f:
            summary = json.load(f)
        fp = summary["failure_patterns"]
        assert "grep" in fp
        assert "rag" in fp
        assert fp["grep"]["wrong_file"]["count"] == fp["rag"]["wrong_file"]["count"]
        assert fp["grep"]["wrong_line"]["count"] == fp["rag"]["wrong_line"]["count"]
        assert fp["grep"]["empty_result"]["count"] == fp["rag"]["empty_result"]["count"]

    def test_handles_missing_difficulty(self, tmp_path):
        base = _make_agg_rows()
        for r in base:
            r["difficulty"] = ""
        self._run_main(tmp_path, grep_agg=base, rag_agg=base)
        with open(tmp_path / "summary.json") as f:
            summary = json.load(f)
        assert summary["per_difficulty_fisher"] == {}
