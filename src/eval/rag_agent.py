import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.eval.llm_client import LLMClient
from src.eval.rag_tools import vector_search, read_file_for_rag


@dataclass
class RAGAgentResult:
    file_path: str
    line_number: int
    steps: int
    raw_output: str


RAG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Search the codebase using semantic vector search. Best for finding where a symbol is defined. Returns top files with snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query, e.g., 'find where OrderService.createOrder is defined'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content of a Java source file to verify the exact definition line.",
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

RAG_SYSTEM_PROMPT = """You are a code search agent using semantic vector search. Your task: find the exact definition of a Java symbol.

## Rules
- You MUST output the final answer as: file_path:line_number
- The file_path is relative to the codebase root
- The line_number is where the class/method/enum is DEFINED
- Do NOT guess -- verify by reading the source file
- You may use vector_search and read_file tools
- Maximum 4 steps total

## Symbol types
- "class": Find the class declaration line
- "method": Find the method declaration line within its class
- "interface": Find the interface declaration line
- "enum": Find the enum declaration line
"""


class RAGAgent:
    def __init__(self, llm_client=None, codebase_root="codebases/apollo",
                 index_store=None, embedder=None, max_steps=4):
        self.llm = llm_client or LLMClient()
        self.codebase_root = codebase_root
        self.index_store = index_store
        self.embedder = embedder
        self.max_steps = max_steps

    def run(self, symbol: str, symbol_type: str) -> RAGAgentResult:
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"Find the definition of {symbol_type} `{symbol}`."},
        ]

        steps = 0
        for _ in range(self.max_steps):
            steps += 1
            result = self.llm.chat(messages, tools=RAG_TOOLS)

            if result.content and not result.tool_calls:
                parsed = self._parse_answer(result.content)
                if parsed:
                    return RAGAgentResult(
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

        return RAGAgentResult(
            file_path="",
            line_number=0,
            steps=steps,
            raw_output="Max steps exceeded",
        )

    def _execute_tool(self, name: str, args: dict) -> str:
        if name == "vector_search":
            return vector_search(args["query"], self.index_store, self.embedder)
        elif name == "read_file":
            full_path = str(Path(self.codebase_root) / args["file_path"])
            return read_file_for_rag(
                full_path,
                args.get("start_line", 0),
                args.get("end_line", 0),
            )
        return f"Unknown tool: {name}"

    def _parse_answer(self, text: str) -> tuple[str, int] | None:
        # Try strict path:line pattern first (Java source file)
        match = re.search(r"([\w\-./]+\.java):(\d+)", text)
        if match:
            path = match.group(1).strip().strip("`*'\"")
            return (path, int(match.group(2)))
        # Fallback: any path-like:line
        match = re.search(r"([\w\-./]+):(\d+)", text)
        if match:
            path = match.group(1).strip().strip("`*'\"")
            return (path, int(match.group(2)))
        return None
