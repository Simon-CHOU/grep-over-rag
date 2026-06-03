# tests/test_chunker.py
import pytest
from src.eval.chunker import Chunk, _strip_comments, find_matching_brace


class TestStripComments:
    def test_strips_block_comment(self):
        source = "public class Foo {\n  /* comment */\n  void bar() {}\n}"
        result = _strip_comments(source)
        assert "comment" not in result
        assert "void bar()" in result

    def test_strips_line_comment(self):
        source = "public class Foo {\n  // line comment\n  void bar() {}\n}"
        result = _strip_comments(source)
        assert "line comment" not in result
        assert "void bar()" in result

    def test_preserves_line_numbers(self):
        source = "line1\n/* block\ncomment */\nline4"
        result = _strip_comments(source)
        lines = result.split("\n")
        assert len(lines) == 4
        assert "line1" in lines[0]
        assert "line4" in lines[3]

    def test_handles_multiline_javadoc(self):
        source = "/**\n * Javadoc\n * @param foo bar\n */\npublic class Foo {}"
        result = _strip_comments(source)
        assert "Javadoc" not in result
        assert "public class Foo" in result


class TestFindMatchingBrace:
    def test_simple_braces(self):
        lines = ["public class Foo {", "  void bar() {}", "}"]
        source = "\n".join(lines)
        end = find_matching_brace(source, 0)
        assert end == 2

    def test_nested_braces(self):
        lines = ["{", "  {", "    {}", "  }", "}"]
        source = "\n".join(lines)
        end = find_matching_brace(source, 0)
        assert end == 4

    def test_skips_braces_in_strings(self):
        lines = ['{', '  String s = "{";', '}']
        source = "\n".join(lines)
        end = find_matching_brace(source, 0)
        assert end == 2

    def test_skips_braces_in_comments(self):
        lines = ["{", "  // {", "  /* } */", "}"]
        source = "\n".join(lines)
        end = find_matching_brace(source, 0)
        assert end == 3
