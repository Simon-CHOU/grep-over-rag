# src/eval/chunker.py
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    file: str
    symbol: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str


# Match class/interface/enum declarations
CLASS_RE = re.compile(
    r"^\s*"
    r"(?:public\s+|private\s+|protected\s+)?"
    r"(?:static\s+)?(?:final\s+)?(?:abstract\s+)?"
    r"(?:class|interface|enum)\s+(\w+)"
)

# Match method/constructor declarations
METHOD_RE = re.compile(
    r"^\s*"
    r"(?:public|private|protected)\s+"
    r"(?:(?:static|final|abstract|synchronized|native|default)\s+)*"
    r"(?:<[^>]+>\s+)?"
    r"(?:\S+ )+"
    r"(\w+)\s*\("
)


def _strip_comments(source: str) -> str:
    """Strip Java comments while preserving line structure."""
    result = []
    i = 0
    in_block_comment = False
    in_line_comment = False

    while i < len(source):
        if in_block_comment:
            if source[i:i+2] == "*/":
                result.append("  ")
                i += 2
                in_block_comment = False
            else:
                result.append(" " if source[i] != "\n" else "\n")
                i += 1
        elif in_line_comment:
            if source[i] == "\n":
                result.append("\n")
                in_line_comment = False
            else:
                result.append(" ")
            i += 1
        elif source[i:i+2] == "/*":
            result.append("  ")
            i += 2
            in_block_comment = True
        elif source[i:i+2] == "//":
            result.append("  ")
            i += 2
            in_line_comment = True
        else:
            result.append(source[i])
            i += 1

    return "".join(result)


def find_matching_brace(source: str, start_line_idx: int) -> int:
    """Find the line index of the closing } matching the first { from start_line_idx.
    Skips braces in comments, strings, and char literals."""
    lines = source.split("\n")
    depth = 0
    in_block = False
    in_line = False
    in_str = False
    in_chr = False

    for line_idx in range(start_line_idx, len(lines)):
        line = lines[line_idx]
        in_line = False
        j = 0
        while j < len(line):
            if in_block:
                if j + 1 < len(line) and line[j:j+2] == "*/":
                    in_block = False
                    j += 2
                    continue
                j += 1
                continue
            if in_line:
                j += 1
                continue

            c = line[j]
            two = line[j:j+2] if j + 1 < len(line) else ""

            if two == "/*":
                in_block = True
                j += 2
                continue
            if two == "//":
                in_line = True
                j += 2
                continue

            if in_str:
                if c == "\\" and j + 1 < len(line):
                    j += 2
                    continue
                if c == '"':
                    in_str = False
                j += 1
                continue

            if in_chr:
                if c == "\\" and j + 1 < len(line):
                    j += 2
                    continue
                if c == "'":
                    in_chr = False
                j += 1
                continue

            if c == '"':
                in_str = True
                j += 1
                continue
            if c == "'":
                in_chr = True
                j += 1
                continue
            if c == "{":
                depth += 1
                j += 1
                continue
            if c == "}":
                depth -= 1
                if depth == 0:
                    return line_idx
                j += 1
                continue

            j += 1

    return len(lines) - 1
