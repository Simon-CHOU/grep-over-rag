# src/eval/grep_tools.py
import subprocess
from pathlib import Path
from glob import glob as _glob


def rg_search(pattern: str, codebase_root: str, file_pattern: str = "*.java") -> list[dict]:
    """Run ripgrep and return list of {file, line, content}."""
    cmd = [
        "rg", "--no-heading", "--line-number",
        "--glob", file_pattern,
        "--max-count", "30",
        pattern, codebase_root
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        results = []
        for line in lines:
            parts = line.rsplit(":", 2)
            if len(parts) >= 3:
                results.append({
                    "file": parts[0],
                    "line": int(parts[1]),
                    "content": parts[2].strip()
                })
        return results
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def glob_files(pattern: str, codebase_root: str) -> list[str]:
    """Find files matching glob pattern relative to codebase root."""
    full_pattern = str(Path(codebase_root) / pattern)
    matches = sorted(_glob(full_pattern, recursive=True))
    return [str(Path(m).relative_to(codebase_root)) for m in matches]


def read_file(file_path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Read file content. Optionally specify line range (1-indexed)."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if start_line > 0:
                lines = f.readlines()
                end = end_line if end_line > 0 else len(lines)
                return "".join(lines[start_line - 1:end])
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error: {e}"
