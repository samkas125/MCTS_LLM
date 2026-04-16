"""Preprocess GSM8K and MATH datasets into unified format for MCTS and GRPO.

Reads/writes JSONL to avoid Arrow save_to_disk deadlocks on some systems.
"""

import re
from pathlib import Path

from datasets import Dataset, load_dataset


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract numeric answer from GSM8K format ('#### <number>')."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


def extract_boxed_answer(solution_text: str) -> str:
    """Extract answer from \\boxed{...} in MATH format, handling nested braces."""
    matches = list(re.finditer(r"\\boxed\{", solution_text))
    if not matches:
        return solution_text.strip()

    last_match = matches[-1]
    start = last_match.end()
    depth = 1
    i = start
    while i < len(solution_text) and depth > 0:
        if solution_text[i] == "{":
            depth += 1
        elif solution_text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return solution_text[start : i - 1].strip()
    return solution_text.strip()


def preprocess_gsm8k(dataset: Dataset) -> Dataset:
    """Convert GSM8K split to unified format."""

    def process(example, idx):
        return {
            "problem_id": f"gsm8k_{idx}",
            "problem": example["question"],
            "solution": extract_gsm8k_answer(example["answer"]),
            "source": "gsm8k",
            "level": 0,
            "subject": "arithmetic",
            "difficulty": 0.0,
        }

    return dataset.map(process, with_indices=True, remove_columns=dataset.column_names)


def preprocess_math500(dataset: Dataset) -> Dataset:
    """Convert MATH-500 eval set to unified format."""

    def process(example, idx):
        level = example.get("level", 1)
        if isinstance(level, str):
            level_match = re.search(r"(\d+)", level)
            level = int(level_match.group(1)) if level_match else 1

        return {
            "problem_id": f"math500_{idx}",
            "problem": example["problem"],
            "solution": example.get("answer", extract_boxed_answer(example.get("solution", ""))),
            "source": "math500",
            "level": level,
            "subject": example.get("subject", "unknown"),
            "difficulty": level / 5.0,
        }

    return dataset.map(process, with_indices=True, remove_columns=dataset.column_names)


def _load_jsonl(path: str | Path) -> Dataset:
    return load_dataset("json", data_files=str(path), split="train")


def preprocess_all(raw_dir: str | Path, processed_dir: str | Path) -> dict:
    """Load raw datasets, preprocess, and save as JSONL to processed_dir."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess GSM8K
    gsm8k_train = preprocess_gsm8k(_load_jsonl(raw_dir / "gsm8k_train.jsonl"))
    gsm8k_test = preprocess_gsm8k(_load_jsonl(raw_dir / "gsm8k_test.jsonl"))

    result = {
        "gsm8k_train": gsm8k_train,
        "gsm8k_test": gsm8k_test,
        "train_combined": gsm8k_train,
    }

    # Load MATH-500 if available
    math500_path = raw_dir / "math500_test.jsonl"
    if math500_path.exists():
        math500_test = preprocess_math500(_load_jsonl(math500_path))
        result["math500_test"] = math500_test

    # Save all as JSONL
    for name, ds in result.items():
        ds.to_json(str(processed_dir / f"{name}.jsonl"))

    from rich.console import Console
    console = Console()
    console.print("[bold green]Preprocessing complete![/bold green]")
    for name, ds in result.items():
        console.print(f"  {name}: {len(ds)} examples")

    return result


if __name__ == "__main__":
    preprocess_all("data/raw", "data/processed")
