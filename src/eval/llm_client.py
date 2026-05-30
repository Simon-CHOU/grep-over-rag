# src/eval/llm_client.py
import os
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient:
    """OpenAI-compatible client wrapper pointed at DeepSeek's API.

    Reads DEEPSEEK_BASE_URL and DEEPSEEK_API_KEY from environment by default,
    but accepts explicit values for testing.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.base_url:
            raise ValueError("DEEPSEEK_BASE_URL environment variable is required")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is required")
        self.model = "deepseek-v4-flash"
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, messages: list[dict], tools: list[dict]) -> ChatResult:
        """Send a chat completion request with optional tool calling."""
        import json

        kwargs: dict = dict(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return ChatResult(content=msg.content, tool_calls=tool_calls)
