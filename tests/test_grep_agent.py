# tests/test_grep_agent.py
import pytest
from unittest.mock import MagicMock
from src.eval.grep_agent import GrepAgent
from src.eval.llm_client import ChatResult, ToolCall


def make_mock_llm(responses: list):
    mock = MagicMock()
    mock.chat = MagicMock(side_effect=responses)
    return mock


def test_grep_agent_finds_symbol_in_one_step():
    llm = make_mock_llm([
        ChatResult(content="NamespaceService.java:42"),
    ])
    agent = GrepAgent(llm_client=llm, codebase_root="/tmp/apollo")
    result = agent.run(symbol="NamespaceService.findNamespace", symbol_type="method")
    assert result.file_path.endswith("NamespaceService.java")
    assert result.line_number == 42
    assert result.steps == 1


def test_grep_agent_uses_tools_then_answers():
    llm = make_mock_llm([
        ChatResult(
            content=None,
            tool_calls=[ToolCall(id="1", name="rg_search", arguments={"pattern": "findNamespace"})]
        ),
        ChatResult(content="NamespaceService.java:87"),
    ])
    agent = GrepAgent(llm_client=llm, codebase_root="/tmp/apollo")
    result = agent.run(symbol="NamespaceService.findNamespace", symbol_type="method")
    assert result.line_number == 87
    assert result.steps == 2


def test_grep_agent_enforces_max_steps():
    tool_calls = ChatResult(
        content=None,
        tool_calls=[ToolCall(id="1", name="rg_search", arguments={"pattern": "findX"})]
    )
    llm = make_mock_llm([tool_calls] * 10)
    agent = GrepAgent(llm_client=llm, codebase_root="/tmp/apollo", max_steps=4)
    result = agent.run(symbol="Something", symbol_type="method")
    assert result.steps <= 4
