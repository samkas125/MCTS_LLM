"""Metrics computation and summary across rounds."""

from __future__ import annotations


def compute_improvement(results_log: list[dict]) -> dict:
    """Compute summary metrics across all training rounds.

    Args:
        results_log: List of dicts, each with 'pre_eval' and 'post_eval' keys
                     containing per-benchmark accuracy dicts.

    Returns:
        Summary dict with trajectories and improvements.
    """
    if not results_log:
        return {}

    summary = {
        "num_rounds": len(results_log),
    }

    for benchmark in ["gsm8k", "math500"]:
        pre_scores = []
        post_scores = []

        for entry in results_log:
            pre = entry.get("pre_eval", {}).get(benchmark, {})
            post = entry.get("post_eval", {}).get(benchmark, {})
            pre_scores.append(pre.get("accuracy", 0.0))
            post_scores.append(post.get("accuracy", 0.0))

        summary[f"{benchmark}_pre_trajectory"] = pre_scores
        summary[f"{benchmark}_post_trajectory"] = post_scores
        summary[f"{benchmark}_baseline"] = pre_scores[0] if pre_scores else 0.0
        summary[f"{benchmark}_final"] = post_scores[-1] if post_scores else 0.0
        summary[f"{benchmark}_improvement"] = (
            summary[f"{benchmark}_final"] - summary[f"{benchmark}_baseline"]
        )

    return summary


def format_results_table(results_log: list[dict]) -> str:
    """Format results log as a readable markdown table."""
    lines = [
        "| Round | GSM8K (pre) | GSM8K (post) | MATH-500 (pre) | MATH-500 (post) |",
        "|-------|-------------|--------------|----------------|-----------------|",
    ]

    for entry in results_log:
        r = entry.get("round", "?")
        gsm_pre = entry.get("pre_eval", {}).get("gsm8k", {}).get("accuracy", 0)
        gsm_post = entry.get("post_eval", {}).get("gsm8k", {}).get("accuracy", 0)
        math_pre = entry.get("pre_eval", {}).get("math500", {}).get("accuracy", 0)
        math_post = entry.get("post_eval", {}).get("math500", {}).get("accuracy", 0)
        lines.append(
            f"| {r} | {gsm_pre:.4f} | {gsm_post:.4f} | {math_pre:.4f} | {math_post:.4f} |"
        )

    return "\n".join(lines)
