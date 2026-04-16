"""Dataset loading and formatting for TRL trainers."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk

from src.inference.prompt_builder import SYSTEM_PROMPT_COT


def prepare_grpo_dataset(
    problems: Dataset,
) -> Dataset:
    """Prepare prompt-only dataset for GRPO training.

    GRPOTrainer generates its own completions — it only needs prompts
    and ground truth answers (for the accuracy reward function).

    Returns Dataset with columns:
        prompt: list[dict] (conversational format)
        solution: str (ground truth answer)
        problem_id: str
    """

    def format_row(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT_COT},
                {"role": "user", "content": example["problem"]},
            ],
            "solution": example["solution"],
            "problem_id": example["problem_id"],
        }

    return problems.map(format_row, remove_columns=problems.column_names)


def prepare_sft_dataset(
    problems: Dataset,
    mcts_traces: dict[str, list[dict]],
) -> Dataset:
    """Prepare prompt-completion dataset for SFT baseline.

    Uses best MCTS trajectory as the supervised target.

    Returns Dataset with columns:
        messages: list[dict] (full conversation for SFT)
    """
    records = []
    for i in range(len(problems)):
        pid = problems[i]["problem_id"]
        if pid not in mcts_traces or not mcts_traces[pid]:
            continue

        best_traj = mcts_traces[pid][0]
        trajectory_text = (
            best_traj.get("trajectory_text", "")
            if isinstance(best_traj, dict)
            else best_traj.trajectory_text
        )

        if not trajectory_text:
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_COT},
            {"role": "user", "content": problems[i]["problem"]},
            {"role": "assistant", "content": trajectory_text},
        ]
        records.append({"messages": messages})

    return Dataset.from_list(records)


def _load_any(path: str) -> Dataset:
    """Load a dataset from Arrow directory or JSONL file."""
    p = Path(path)
    jsonl = Path(str(path) + ".jsonl")
    if jsonl.exists():
        return load_dataset("json", data_files=str(jsonl), split="train")
    if p.exists():
        return load_from_disk(str(p))
    raise FileNotFoundError(f"Dataset not found at {path} or {jsonl}")


def load_processed_datasets(processed_dir: str) -> dict[str, Dataset]:
    """Load all preprocessed datasets from disk (Arrow or JSONL).

    Returns dict with keys: train_combined, gsm8k_test, math500_test (optional)
    """
    result = {
        "train_combined": _load_any(f"{processed_dir}/train_combined"),
        "gsm8k_test": _load_any(f"{processed_dir}/gsm8k_test"),
    }
    try:
        result["math500_test"] = _load_any(f"{processed_dir}/math500_test")
    except FileNotFoundError:
        result["math500_test"] = None
    return result
