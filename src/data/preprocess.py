"""Preprocess GSM8K and MATH datasets into unified format for MCTS and GRPO."""

import re
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk


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
    """Convert GSM8K split to unified format.

    Returns Dataset with columns:
        problem_id, problem, solution, source, level, subject, difficulty
    """

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


def preprocess_math(dataset: Dataset) -> Dataset:
    """Convert MATH split to unified format.

    Returns Dataset with columns:
        problem_id, problem, solution, source, level, subject, difficulty
    """

    def process(example, idx):
        level_str = example.get("level", "Level 1")
        level_match = re.search(r"(\d+)", level_str)
        level = int(level_match.group(1)) if level_match else 1

        return {
            "problem_id": f"math_{idx}",
            "problem": example["problem"],
            "solution": extract_boxed_answer(example["solution"]),
            "source": "math",
            "level": level,
            "subject": example.get("type", "unknown"),
            "difficulty": level / 5.0,
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


def create_combined_training_set(
    gsm8k_train: Dataset,
    math_train: Dataset,
    max_size: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Combine GSM8K and MATH training sets.

    Args:
        gsm8k_train: Preprocessed GSM8K training split.
        math_train: Preprocessed MATH training split.
        max_size: If set, subsample to this size (proportionally).
        seed: Random seed for subsampling.
    """
    combined = concatenate_datasets([gsm8k_train, math_train])

    if max_size and len(combined) > max_size:
        combined = combined.shuffle(seed=seed).select(range(max_size))

    return combined


def preprocess_all(raw_dir: str | Path, processed_dir: str | Path) -> dict:
    """Load raw datasets, preprocess, and save to processed_dir.

    Returns dict with all processed datasets.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load raw datasets
    gsm8k = load_from_disk(str(raw_dir / "gsm8k"))
    math_ds = load_from_disk(str(raw_dir / "math"))
    math500 = load_from_disk(str(raw_dir / "math500"))

    # Preprocess
    gsm8k_train = preprocess_gsm8k(gsm8k["train"])
    gsm8k_test = preprocess_gsm8k(gsm8k["test"])
    math_train = preprocess_math(math_ds["train"])
    math500_test = preprocess_math500(math500["test"])

    # Combined training set
    train_combined = create_combined_training_set(gsm8k_train, math_train)

    # Save
    result = {
        "gsm8k_train": gsm8k_train,
        "gsm8k_test": gsm8k_test,
        "math_train": math_train,
        "math500_test": math500_test,
        "train_combined": train_combined,
    }

    for name, ds in result.items():
        ds.save_to_disk(str(processed_dir / name))

    from rich.console import Console

    console = Console()
    console.print("[bold green]Preprocessing complete![/bold green]")
    for name, ds in result.items():
        console.print(f"  {name}: {len(ds)} examples")

    return result


if __name__ == "__main__":
    preprocess_all("data/raw", "data/processed")
