# src/eval/scorer.py
def score_prediction(
    predicted_file: str,
    predicted_line: int,
    expected_file: str,
    expected_line: int
) -> bool:
    """Return True if file matches exactly and line is within +-3."""
    file_match = predicted_file == expected_file
    line_match = abs(predicted_line - expected_line) <= 3
    return file_match and line_match
