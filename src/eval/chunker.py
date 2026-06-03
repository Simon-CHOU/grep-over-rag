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


# Match class/interface/enum declarations (including @interface)
CLASS_RE = re.compile(
    r"^\s*"
    r"(?:public\s+|private\s+|protected\s+)?"
    r"(?:static\s+)?(?:final\s+)?(?:abstract\s+)?"
    r"(?:class|@?interface|enum)\s+(\w+)"
)

# Match method declarations (requires return type)
METHOD_RE = re.compile(
    r"^\s*"
    r"(?:public|private|protected)\s+"
    r"(?:(?:static|final|abstract|synchronized|native|default)\s+)*"
    r"(?:<[^>]+>\s+)?"
    r"(?:\S+ )+"
    r"(\w+)\s*\("
)

# Match constructor declarations (no return type)
CONSTRUCTOR_RE = re.compile(
    r"^\s*"
    r"(?:public|private|protected)\s+"
    r"(?:(?:static|final|abstract|synchronized|native|default)\s+)*"
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


def _detect_brace_depth_at_line(source: str, target_line_idx: int) -> int:
    """Calculate brace depth at the start of target_line_idx by counting
    braces from the beginning of the file (skipping comments/strings)."""
    lines = source.split("\n")
    depth = 0
    in_block = False
    in_line = False
    in_str = False
    in_chr = False

    for line_idx in range(target_line_idx):
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
                j += 1
                continue
            j += 1

    return depth


def chunk_file(source: str, file_path: str) -> list[Chunk]:
    """Parse a Java source file and extract symbol-level chunks."""
    lines = source.split("\n")
    stripped = _strip_comments(source)
    stripped_lines = stripped.split("\n")
    chunks: list[Chunk] = []

    # --- Pass 1: find all class/interface/enum declarations ---
    class_decls: list[tuple[int, int, str, str]] = []  # (start_idx, end_idx, name, stype)

    for i, sline in enumerate(stripped_lines):
        m = CLASS_RE.match(sline)
        if not m:
            continue
        body_start = i
        while body_start < len(stripped_lines) and "{" not in stripped_lines[body_start]:
            body_start += 1
        if body_start >= len(stripped_lines):
            continue
        end = find_matching_brace(source, body_start)

        name = m.group(1)
        orig = lines[i]
        if "@interface" in orig:
            stype = "interface"
        elif "interface" in orig:
            stype = "interface"
        elif "enum" in orig:
            stype = "enum"
        else:
            stype = "class"

        class_decls.append((i, end, name, stype))

    # --- Pass 2: build full names and create class chunks ---
    class_full_names: dict[int, str] = {}

    for (start, end, name, stype) in class_decls:
        depth = _detect_brace_depth_at_line(source, start)
        if depth == 0:
            full_name = name
        else:
            # Find innermost enclosing class (longest full name that encloses this line)
            parent_name = ""
            for (ps, pe, pn, pt) in class_decls:
                if ps < start and pe > start:
                    candidate = class_full_names.get(ps, pn)
                    if len(candidate) > len(parent_name):
                        parent_name = candidate
            full_name = f"{parent_name}.{name}" if parent_name else name

        class_full_names[start] = full_name
        content = "\n".join(lines[start:end + 1])
        chunks.append(Chunk(
            file=file_path,
            symbol=full_name,
            symbol_type=stype,
            start_line=start + 1,
            end_line=end + 1,
            content=content,
        ))

    # --- Pass 3: extract methods for each class ---
    for (start, end, name, stype) in class_decls:
        full_name = class_full_names[start]

        # Collect ranges of directly nested inner classes to skip
        nested_ranges: list[tuple[int, int]] = []
        for (os_, oe_, on_, ot_) in class_decls:
            if os_ > start and oe_ < end:
                nested_ranges.append((os_, oe_))

        for j in range(start + 1, end):
            # Skip lines inside nested inner classes
            skip = False
            for (ns, ne) in nested_ranges:
                if ns <= j <= ne:
                    skip = True
                    break
            if skip:
                continue

            # Try METHOD_RE first (has return type), then CONSTRUCTOR_RE
            method_name = None
            mm = METHOD_RE.match(stripped_lines[j])
            if mm:
                method_name = mm.group(1)
            else:
                cm = CONSTRUCTOR_RE.match(stripped_lines[j])
                if cm:
                    method_name = cm.group(1)

            if method_name is None:
                continue

            mbody_start = j
            while mbody_start < len(stripped_lines) and "{" not in stripped_lines[mbody_start]:
                mbody_start += 1
            if mbody_start < len(stripped_lines):
                mend = find_matching_brace(source, mbody_start)
                mcontent = "\n".join(lines[j:mend + 1])
                chunks.append(Chunk(
                    file=file_path,
                    symbol=f"{full_name}.{method_name}",
                    symbol_type="method",
                    start_line=j + 1,
                    end_line=mend + 1,
                    content=mcontent,
                ))

    return chunks
