# tests/test_scorer.py
from src.eval.scorer import score_prediction


def test_exact_match():
    assert score_prediction(
        predicted_file="foo/Bar.java",
        predicted_line=42,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is True


def test_line_within_tolerance_plus():
    assert score_prediction(
        predicted_file="foo/Bar.java",
        predicted_line=44,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is True


def test_line_within_tolerance_minus():
    assert score_prediction(
        predicted_file="foo/Bar.java",
        predicted_line=39,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is True


def test_line_outside_tolerance():
    assert score_prediction(
        predicted_file="foo/Bar.java",
        predicted_line=46,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is False


def test_file_mismatch():
    assert score_prediction(
        predicted_file="foo/Baz.java",
        predicted_line=42,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is False


def test_both_wrong():
    assert score_prediction(
        predicted_file="foo/Baz.java",
        predicted_line=99,
        expected_file="foo/Bar.java",
        expected_line=42
    ) is False


from src.eval.scorer import has_valid_result


def test_has_valid_result_true():
    assert has_valid_result("Foo.java", 42) is True


def test_has_valid_result_false():
    assert has_valid_result("", 0) is False
