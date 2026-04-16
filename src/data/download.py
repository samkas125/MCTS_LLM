"""Download GSM8K, MATH, and MATH-500 datasets from HuggingFace."""

from pathlib import Path

from datasets import load_dataset


def download_gsm8k(output_dir: str | Path) -> dict:
    """Download GSM8K dataset.

    Returns dict with 'train' and 'test' splits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main")
    ds.save_to_disk(str(output_dir / "gsm8k"))
    return ds


def download_math(output_dir: str | Path) -> dict:
    """Download MATH (competition_math) dataset.

    Returns dict with 'train' and 'test' splits.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("hendrycks/competition_math")
    ds.save_to_disk(str(output_dir / "math"))
    return ds


def download_math500(output_dir: str | Path) -> dict:
    """Download MATH-500 evaluation subset.

    Returns dict with 'test' split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceH4/MATH-500")
    ds.save_to_disk(str(output_dir / "math500"))
    return ds


def download_all_datasets(output_dir: str | Path) -> dict:
    """Download all datasets and return them.

    Returns:
        dict with keys 'gsm8k', 'math', 'math500', each containing dataset splits.
    """
    output_dir = Path(output_dir)

    from rich.console import Console

    console = Console()

    console.print("[bold]Downloading GSM8K...[/bold]")
    gsm8k = download_gsm8k(output_dir)

    console.print("[bold]Downloading MATH...[/bold]")
    math = download_math(output_dir)

    console.print("[bold]Downloading MATH-500...[/bold]")
    math500 = download_math500(output_dir)

    console.print("[bold green]All datasets downloaded successfully![/bold green]")
    console.print(f"  GSM8K:   train={len(gsm8k['train'])}, test={len(gsm8k['test'])}")
    console.print(f"  MATH:    train={len(math['train'])}, test={len(math['test'])}")
    console.print(f"  MATH-500: test={len(math500['test'])}")

    return {"gsm8k": gsm8k, "math": math, "math500": math500}


if __name__ == "__main__":
    download_all_datasets("data/raw")
