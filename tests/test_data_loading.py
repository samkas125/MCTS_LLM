"""Tests for data preprocessing."""

from src.data.preprocess import extract_boxed_answer, extract_gsm8k_answer


def test_gsm8k_answer_extraction():
    assert extract_gsm8k_answer("some steps\n#### 42") == "42"
    assert extract_gsm8k_answer("#### 1,000") == "1000"


def test_math_answer_extraction():
    assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_boxed_answer("\\boxed{42}") == "42"
    assert extract_boxed_answer("no answer") == "no answer"


def test_nested_boxed():
    text = "\\boxed{\\frac{a+b}{c+d}}"
    result = extract_boxed_answer(text)
    assert result == "\\frac{a+b}{c+d}"
