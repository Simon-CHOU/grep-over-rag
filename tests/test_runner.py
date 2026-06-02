# tests/test_runner.py
import json
import csv
from pathlib import Path
from unittest.mock import MagicMock
from src.eval.runner import Runner


def make_mock_agent(results):
    mock = MagicMock()
    mock.run = MagicMock(side_effect=results)
    return mock


class MockAgentResultError:
    def __init__(self, file_path="", line_number=0, steps=0, raw_output=""):
        self.file_path = file_path
        self.line_number = line_number
        self.steps = steps
        self.raw_output = raw_output


def test_runner_single_run_per_task(tmp_path):
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("\n".join([
        json.dumps({"id": "t1", "symbol": "Foo.bar", "type": "method", "file": "Foo.java", "expected_line": 10, "difficulty": "easy", "note": ""}),
        json.dumps({"id": "t2", "symbol": "Baz.qux", "type": "class", "file": "Baz.java", "expected_line": 3, "difficulty": "medium", "note": ""}),
    ]))

    agent = make_mock_agent([
        MockAgentResultError("Foo.java", 10, 1, "ok"),
        MockAgentResultError("Wrong.java", 25, 3, "wrong file"),
    ])

    runner = Runner(agent_factory=lambda: agent, mode="grep")
    runner.run(tasks_file=str(tasks_file), output_dir=str(tmp_path), num_runs=1)

    # Verify raw CSV
    raw = Path(tmp_path) / "grep_raw.csv"
    reader = csv.DictReader(raw.open())
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["error_type"] == "correct"
    assert rows[1]["error_type"] == "wrong_file"

    # Verify aggregate CSV
    agg = Path(tmp_path) / "grep_aggregate.csv"
    reader = csv.DictReader(agg.open())
    agg_rows = list(reader)
    assert len(agg_rows) == 2
    assert agg_rows[0]["success_count"] == "1"
    assert agg_rows[0]["success_rate"] == "1.0"
    assert agg_rows[1]["success_count"] == "0"


def test_runner_multi_run(tmp_path):
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(json.dumps(
        {"id": "t1", "symbol": "Foo.bar", "type": "method", "file": "Foo.java", "expected_line": 10, "difficulty": "easy", "note": ""}
    ) + "\n")

    # 3 runs: pass, fail (wrong_file), pass
    agent = make_mock_agent([
        MockAgentResultError("Foo.java", 10, 1, "ok"),
        MockAgentResultError("Bar.java", 10, 2, "wrong"),
        MockAgentResultError("Foo.java", 11, 1, "ok"),
    ])

    runner = Runner(agent_factory=lambda: agent, mode="grep")
    runner.run(tasks_file=str(tasks_file), output_dir=str(tmp_path), num_runs=3)

    raw = Path(tmp_path) / "grep_raw.csv"
    reader = csv.DictReader(raw.open())
    rows = list(reader)
    assert len(rows) == 3
    assert rows[1]["error_type"] == "wrong_file"

    agg = Path(tmp_path) / "grep_aggregate.csv"
    reader = csv.DictReader(agg.open())
    agg_rows = list(reader)
    assert agg_rows[0]["success_count"] == "2"
    assert agg_rows[0]["success_rate"] == "0.6666666666666666"
    assert agg_rows[0]["failure_pattern"] == "sporadic"
    assert agg_rows[0]["dominant_error_type"] == "wrong_file"


def test_runner_api_error_resilience(tmp_path):
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(json.dumps(
        {"id": "t1", "symbol": "Foo.bar", "type": "method", "file": "Foo.java", "expected_line": 10, "difficulty": "easy", "note": ""}
    ) + "\n")

    agent = make_mock_agent([RuntimeError("API timeout")])

    runner = Runner(agent_factory=lambda: agent, mode="grep")
    runner.run(tasks_file=str(tasks_file), output_dir=str(tmp_path), num_runs=1)

    raw = Path(tmp_path) / "grep_raw.csv"
    reader = csv.DictReader(raw.open())
    rows = list(reader)
    assert rows[0]["error_type"] == "api_error"
    assert rows[0]["success"] == "false"
    assert rows[0]["latency_ms"] == "0"
