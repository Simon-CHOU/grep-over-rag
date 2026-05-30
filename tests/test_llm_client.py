# tests/test_llm_client.py
import os
import pytest
from src.eval.llm_client import LLMClient


def test_llm_client_initializes_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    client = LLMClient()
    assert client.base_url == "https://api.deepseek.com"
    assert client.api_key == "sk-test"
    assert client.model == "deepseek-v4-flash"


def test_llm_client_raises_on_missing_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_BASE_URL"):
        LLMClient()


def test_llm_client_chat_basic(mocker):
    mock_create = mocker.patch("openai.resources.chat.Completions.create")
    mock_create.return_value = mocker.Mock(
        choices=[mocker.Mock(message=mocker.Mock(content="answer", tool_calls=None))]
    )
    client = LLMClient(base_url="http://test", api_key="sk-test")
    messages = [{"role": "user", "content": "hello"}]
    result = client.chat(messages, tools=[])
    assert result.content == "answer"
    mock_create.assert_called_once()


def test_llm_client_chat_with_tool_calls(mocker):
    mock_create = mocker.patch("openai.resources.chat.Completions.create")
    tool_call_mock = mocker.Mock()
    tool_call_mock.id = "call_1"
    tool_call_mock.function.name = "search"
    tool_call_mock.function.arguments = '{"pattern":"OrderService"}'
    mock_create.return_value = mocker.Mock(
        choices=[mocker.Mock(message=mocker.Mock(
            content=None,
            tool_calls=[tool_call_mock]
        ))]
    )
    client = LLMClient(base_url="http://test", api_key="sk-test")
    result = client.chat([{"role": "user", "content": "find OrderService"}], tools=[])
    assert result.tool_calls[0].name == "search"
