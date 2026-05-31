# tests/test_classifier.py
from src.eval.classifier import classify, has_valid_result


def test_classify_correct():
    assert classify("Foo.java", 42, "Foo.java", 42) == "correct"


def test_classify_correct_within_tolerance():
    assert classify("Foo.java", 44, "Foo.java", 42) == "correct"


def test_classify_wrong_file():
    assert classify("Bar.java", 42, "Foo.java", 42) == "wrong_file"


def test_classify_wrong_line():
    assert classify("Foo.java", 99, "Foo.java", 42) == "wrong_line"


def test_classify_empty_file():
    assert classify("", 42, "Foo.java", 42) == "empty_result"


def test_classify_zero_line():
    assert classify("Foo.java", 0, "Foo.java", 42) == "empty_result"


def test_classify_both_empty():
    assert classify("", 0, "Foo.java", 42) == "empty_result"


def test_has_valid_result_true():
    assert has_valid_result("Foo.java", 42) is True


def test_has_valid_result_empty_file():
    assert has_valid_result("", 42) is False


def test_has_valid_result_zero_line():
    assert has_valid_result("Foo.java", 0) is False
