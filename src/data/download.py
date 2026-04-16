"""Download GSM8K and MATH-500 datasets from HuggingFace.

Saves as JSONL files to avoid Arrow save_to_disk deadlocks on some systems.
"""

from pathlib import Path

from datasets import load_dataset


def download_gsm8k(output_dir: str | Path) -> dict:
    """Download GSM8K dataset and save as JSONL.

    Returns dict with 'train' and 'test' splits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main")
    ds["train"].to_json(str(output_dir / "gsm8k_train.jsonl"))
    ds["test"].to_json(str(output_dir / "gsm8k_test.jsonl"))
    return ds


def download_math500(output_dir: str | Path) -> dict:
    """Download MATH-500 evaluation subset and save as JSONL.

    Returns dict with 'test' split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceH4/MATH-500")
    ds["test"].to_json(str(output_dir / "math500_test.jsonl"))
    return ds


def download_all_datasets(output_dir: str | Path) -> dict:
    """Download all datasets and return them."""
    output_dir = Path(output_dir)

    from rich.console import Console
    console = Console()

    console.print("[bold]Downloading GSM8K...[/bold]")
    gsm8k = download_gsm8k(output_dir)

    console.print("[bold]Downloading MATH-500...[/bold]")
    try:
        math500 = download_math500(output_dir)
        console.print(f"  MATH-500: test={len(math500['test'])}")
    except Exception as e:
        console.print(f"[yellow]MATH-500 download failed (skipping): {e}[/yellow]")
        math500 = None

    console.print("[bold green]Datasets downloaded successfully![/bold green]")
    console.print(f"  GSM8K: train={len(gsm8k['train'])}, test={len(gsm8k['test'])}")

    return {"gsm8k": gsm8k, "math500": math500}


if __name__ == "__main__":
    download_all_datasets("data/raw")
