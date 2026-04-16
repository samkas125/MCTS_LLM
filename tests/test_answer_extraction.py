"""Tests for answer extraction from model outputs."""

from src.rewards.answer_extraction import (
    extract_answer,
    extract_boxed_answer,
    extract_gsm8k_answer,
)


# --- extract_boxed_answer ---


def test_simple_boxed():
    assert extract_boxed_answer("The answer is \\boxed{42}") == "42"


def test_fraction_boxed():
    assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_nested_braces():
    assert extract_boxed_answer("\\boxed{\\frac{a+b}{c}}") == "\\frac{a+b}{c}"


def test_multiple_boxed_takes_last():
    text = "First \\boxed{wrong} then \\boxed{correct}"
    assert extract_boxed_answer(text) == "correct"


def test_no_boxed():
    assert extract_boxed_answer("no boxed answer here") is None


def test_empty_boxed():
    assert extract_boxed_answer("\\boxed{}") == ""


def test_boxed_with_latex():
    text = "\\boxed{x^2 + 2x + 1}"
    assert extract_boxed_answer(text) == "x^2 + 2x + 1"


def test_boxed_negative_number():
    assert extract_boxed_answer("\\boxed{-7}") == "-7"


# --- extract_gsm8k_answer ---


def test_gsm8k_integer():
    assert extract_gsm8k_answer("blah blah\n#### 72") == "72"


def test_gsm8k_with_comma():
    assert extract_gsm8k_answer("#### 1,234") == "1234"


def test_gsm8k_decimal():
    assert extract_gsm8k_answer("#### 3.14") == "3.14"


def test_gsm8k_no_marker():
    assert extract_gsm8k_answer("the answer is 42") is None


# --- extract_answer (combined) ---


def test_extract_prefers_boxed():
    text = "Long reasoning...\n#### 42\n\\boxed{42}"
    assert extract_answer(text, source="gsm8k") == "42"


def test_extract_falls_back_to_gsm8k():
    text = "Some reasoning\n#### 72"
    assert extract_answer(text, source="gsm8k") == "72"


def test_extract_from_math():
    text = "Therefore \\boxed{\\frac{1}{3}}"
    assert extract_answer(text, source="math") == "\\frac{1}{3}"


def test_extract_no_answer():
    assert extract_answer("I don't know", source="math") is None
