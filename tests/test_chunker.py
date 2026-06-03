# tests/test_chunker.py
import pytest
from src.eval.chunker import Chunk, _strip_comments, find_matching_brace
from src.eval.chunker import chunk_file


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


class TestChunkFile:
    def test_simple_class_with_methods(self):
        source = (
            "package com.example;\n"
            "\n"
            "public class HelloService {\n"
            "\n"
            "  public String greet(String name) {\n"
            "    return \"Hello \" + name;\n"
            "  }\n"
            "\n"
            "  public void run() {\n"
            "    System.out.println(\"running\");\n"
            "  }\n"
            "}\n"
        )
        chunks = chunk_file(source, "com/example/HelloService.java")
        symbols = {c.symbol: c for c in chunks}
        # Should have class + 2 methods = 3 chunks
        assert len(chunks) == 3
        assert "HelloService" in symbols
        assert symbols["HelloService"].symbol_type == "class"
        assert symbols["HelloService"].start_line == 3
        assert "HelloService.greet" in symbols
        assert symbols["HelloService.greet"].symbol_type == "method"
        assert symbols["HelloService.greet"].start_line == 5
        assert "HelloService.run" in symbols
        assert symbols["HelloService.run"].symbol_type == "method"
        assert symbols["HelloService.run"].start_line == 9

    def test_interface(self):
        source = (
            "public interface FooService {\n"
            "  void doSomething();\n"
            "}\n"
        )
        chunks = chunk_file(source, "FooService.java")
        symbols = {c.symbol: c for c in chunks}
        assert "FooService" in symbols
        assert symbols["FooService"].symbol_type == "interface"

    def test_enum_with_body(self):
        source = (
            "public enum Status {\n"
            "  ACTIVE, INACTIVE;\n"
            "\n"
            "  public String label() {\n"
            "    return name().toLowerCase();\n"
            "  }\n"
            "}\n"
        )
        chunks = chunk_file(source, "Status.java")
        symbols = {c.symbol: c for c in chunks}
        assert "Status" in symbols
        assert symbols["Status"].symbol_type == "enum"
        assert "Status.label" in symbols

    def test_inner_static_class(self):
        source = (
            "public class Outer {\n"
            "\n"
            "  public void outerMethod() {\n"
            "  }\n"
            "\n"
            "  public static class Inner {\n"
            "    public void innerMethod() {\n"
            "    }\n"
            "  }\n"
            "}\n"
        )
        chunks = chunk_file(source, "Outer.java")
        symbols = {c.symbol: c for c in chunks}
        assert "Outer" in symbols
        assert "Outer.outerMethod" in symbols
        assert "Outer.Inner" in symbols
        assert symbols["Outer.Inner"].symbol_type == "class"
        assert "Outer.Inner.innerMethod" in symbols

    def test_skips_comments_with_braces(self):
        source = (
            "/**\n"
            " * This class has { braces } in javadoc.\n"
            " */\n"
            "public class DocClass {\n"
            "  // method with { in comment\n"
            "  public void real() {\n"
            "  }\n"
            "}\n"
        )
        chunks = chunk_file(source, "DocClass.java")
        symbols = {c.symbol: c for c in chunks}
        assert "DocClass" in symbols
        assert "DocClass.real" in symbols

    def test_constructor(self):
        source = (
            "public class Foo {\n"
            "  private String name;\n"
            "\n"
            "  public Foo(String name) {\n"
            "    this.name = name;\n"
            "  }\n"
            "}\n"
        )
        chunks = chunk_file(source, "Foo.java")
        symbols = {c.symbol: c for c in chunks}
        assert "Foo" in symbols
        assert "Foo.Foo" in symbols
        assert symbols["Foo.Foo"].symbol_type == "method"

    def test_annotation_type(self):
        source = (
            "public @interface MyAnnotation {\n"
            "  String value();\n"
            "}\n"
        )
        chunks = chunk_file(source, "MyAnnotation.java")
        symbols = {c.symbol: c for c in chunks}
        assert "MyAnnotation" in symbols
        assert symbols["MyAnnotation"].symbol_type in ("interface", "class")
