# src/eval/grep_agent.py
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.eval.llm_client import LLMClient
from src.eval.grep_tools import rg_search, glob_files, read_file


@dataclass
class GrepAgentResult:
    file_path: str
    line_number: int
    steps: int
    raw_output: str


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rg_search",
            "description": "Search for a pattern in Java source files using ripgrep. Returns list of {file, line, content}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob_files",
            "description": "Find Java files matching a glob pattern (e.g., **/NamespaceService.java).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern relative to codebase root"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content of a Java source file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "Optional: start line (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Optional: end line (1-indexed)"},
                },
                "required": ["file_path"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a code search agent. Your task: find the exact definition of a Java symbol.

## Rules
- You MUST output the final answer as: file_path:line_number
- The file_path is relative to the codebase root
- The line_number is where the class/method/enum is DEFINED (not where it's called)
- Do NOT guess -- only answer after examining the source code
- You may use rg_search, glob_files, and read_file tools
- Maximum 4 steps total

## Symbol types
- "class": Find the class declaration line
- "method": Find the method declaration line within its class
- "interface": Find the interface declaration line
- "enum": Find the enum declaration line
"""


class GrepAgent:
    def __init__(self, llm_client=None, codebase_root="codebases/apollo", max_steps=4):
        self.llm = llm_client or LLMClient()
        self.codebase_root = codebase_root
        self.max_steps = max_steps

    def run(self, symbol: str, symbol_type: str) -> GrepAgentResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Find the definition of {symbol_type} `{symbol}`."},
        ]

        steps = 0
        for _ in range(self.max_steps):
            steps += 1
            result = self.llm.chat(messages, tools=TOOLS)

            if result.content and not result.tool_calls:
                parsed = self._parse_answer(result.content)
                if parsed:
                    return GrepAgentResult(
                        file_path=parsed[0],
                        line_number=parsed[1],
                        steps=steps,
                        raw_output=result.content,
                    )

            if result.tool_calls:
                # Collect all tool results first, batch into one assistant message
                tool_results = []
                for tc in result.tool_calls:
                    tool_results.append((tc, self._execute_tool(tc.name, tc.arguments)))

                assistant_msg = {
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": [],
                }
                for tc, _ in tool_results:
                    assistant_msg["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    })
                messages.append(assistant_msg)

                for tc, tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tr,
                    })

        return GrepAgentResult(
            file_path="",
            line_number=0,
            steps=steps,
            raw_output="Max steps exceeded",
        )

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "rg_search":
            results = rg_search(args["pattern"], self.codebase_root)
            if not results:
                return "No matches found."
            return "\n".join(
                f"{r['file']}:{r['line']}: {r['content']}" for r in results[:20]
            )
        elif name == "glob_files":
            results = glob_files(args["pattern"], self.codebase_root)
            if not results:
                return "No files found."
            return "\n".join(results[:30])
        elif name == "read_file":
            full_path = str(Path(self.codebase_root) / args["file_path"])
            results = read_file(
                full_path,
                args.get("start_line", 0),
                args.get("end_line", 0),
            )
            return results
        return f"Unknown tool: {name}"

    def _parse_answer(self, text: str) -> tuple[str, int] | None:
        match = re.search(r"(.+?):(\d+)", text)
        if match:
            return (match.group(1).strip(), int(match.group(2)))
        return None
