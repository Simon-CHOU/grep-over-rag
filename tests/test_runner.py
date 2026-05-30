import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from src.eval.runner import Runner


def make_mock_agent(results):
    mock = MagicMock()
    mock.run = MagicMock(side_effect=results)
    return mock


class MockAgentResult:
    def __init__(self, file_path, line_number, steps, raw_output):
        self.file_path = file_path
        self.line_number = line_number
        self.steps = steps
        self.raw_output = raw_output


def test_runner_processes_all_tasks(tmp_path):
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("\n".join([
        json.dumps({"id": "t1", "symbol": "Foo.bar", "type": "method", "file": "Foo.java", "expected_line": 10, "difficulty": "easy", "note": ""}),
        json.dumps({"id": "t2", "symbol": "Baz.qux", "type": "class", "file": "Baz.java", "expected_line": 3, "difficulty": "medium", "note": ""}),
        json.dumps({"id": "t3", "symbol": "Qux.fred", "type": "method", "file": "Qux.java", "expected_line": 25, "difficulty": "easy", "note": ""}),
    ]))

    agent = make_mock_agent([
        MockAgentResult("Foo.java", 10, 1, "ok"),
        MockAgentResult("Baz.java", 4, 2, "ok"),   # line 4 vs expected 3 → within +-3
        MockAgentResult("Wrong.java", 25, 3, "wrong file"),  # wrong file
    ])

    runner = Runner(agent_factory=lambda: agent, mode="grep")
    summary_path = str(tmp_path / "summary.csv")
    failures_path = str(tmp_path / "failures.jsonl")

    runner.run(tasks_file=str(tasks_file), summary_path=summary_path, failures_path=failures_path)

    # Verify summary CSV
    lines = Path(summary_path).read_text().strip().split("\n")
    assert len(lines) == 4  # header + 3 tasks
    assert lines[0] == "task_id,difficulty,symbol,success,steps,latency_ms,predicted_file,predicted_line"
    # t1: Foo.java:10 matches Foo.java:10 → success
    assert "t1,easy,Foo.bar,true,1" in lines[1]
    # t2: Baz.java:4 vs Baz.java:3 → within +-3 → success
    assert "t2,medium,Baz.qux,true,2" in lines[2]
    # t3: Wrong.java vs Qux.java → fail
    assert "t3,easy,Qux.fred,false,3" in lines[3]

    # Verify failures JSONL
    failures = Path(failures_path).read_text().strip().split("\n")
    assert len(failures) == 1
    fail = json.loads(failures[0])
    assert fail["task_id"] == "t3"


def test_runner_stats_summary(tmp_path):
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("\n".join([
        json.dumps({"id": "t1", "symbol": "Foo.bar", "type": "method", "file": "Foo.java", "expected_line": 10, "difficulty": "easy", "note": ""}),
        json.dumps({"id": "t2", "symbol": "Baz.qux", "type": "class", "file": "Baz.java", "expected_line": 3, "difficulty": "medium", "note": ""}),
    ]))

    agent = make_mock_agent([
        MockAgentResult("Foo.java", 10, 1, "ok"),
        MockAgentResult("Wrong.java", 99, 4, "miss"),
    ])

    runner = Runner(agent_factory=lambda: agent, mode="grep")
    summary_path = str(tmp_path / "summary.csv")

    runner.run(tasks_file=str(tasks_file), summary_path=summary_path, failures_path=str(tmp_path / "fail.jsonl"))
    assert runner.stats["total"] == 2
    assert runner.stats["success"] == 1
    assert runner.stats["failed"] == 1
    assert runner.stats["success_rate"] == 0.5
    assert runner.stats["avg_steps"] == 2.5
