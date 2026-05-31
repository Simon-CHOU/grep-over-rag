# src/eval/classifier.py
def classify(
    predicted_file: str,
    predicted_line: int,
    expected_file: str,
    expected_line: int
) -> str:
    """Classify a prediction into one of four categories."""
    if not predicted_file or predicted_line == 0:
        return "empty_result"
    if predicted_file != expected_file:
        return "wrong_file"
    if abs(predicted_line - expected_line) > 3:
        return "wrong_line"
    return "correct"


def has_valid_result(predicted_file: str, predicted_line: int) -> bool:
    """True if agent produced a non-empty prediction."""
    return bool(predicted_file) and predicted_line > 0
