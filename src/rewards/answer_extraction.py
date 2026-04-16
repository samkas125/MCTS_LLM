"""Extract final answers from model outputs in various formats."""

import re


def extract_boxed_answer(text: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces.

    Takes the LAST \\boxed occurrence (the final answer).
    """
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None

    last_match = matches[-1]
    start = last_match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[start : i - 1].strip()
    return None


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract numeric answer after #### in GSM8K format."""
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip().replace(",", "")
    return None


def extract_answer(text: str, source: str = "math") -> str | None:
    """Extract answer from model output based on dataset source.

    Tries \\boxed{} first (works for both MATH and GSM8K when model uses it),
    then falls back to #### for GSM8K.
    """
    # Always try boxed first (the model is prompted to use it)
    answer = extract_boxed_answer(text)
    if answer is not None:
        return answer

    # Fallback for GSM8K format
    if source == "gsm8k":
        answer = extract_gsm8k_answer(text)
        if answer is not None:
            return answer

    # Last resort: try to extract a number at the end of the response
    # Look for patterns like "The answer is 42" or "= 42"
    patterns = [
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s*:?\s*(.+?)\.?\s*$",
        r"=\s*(\S+)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1).strip()

    return None
