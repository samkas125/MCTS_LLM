"""Evaluation pipeline for GSM8K and MATH-500 benchmarks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from datasets import Dataset

from src.inference.prompt_builder import SYSTEM_PROMPT_COT
from src.rewards.accuracy import check_answer_equivalence
from src.rewards.answer_extraction import extract_answer

logger = logging.getLogger(__name__)


def _resolve_vllm_model_path(model_path: str) -> str:
    """Return a path vLLM can load — resolves LoRA adapters to their base model."""
    adapter_cfg = Path(model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        with open(adapter_cfg) as f:
            base = json.load(f)["base_model_name_or_path"]
        logger.warning(
            f"{model_path} is a LoRA adapter — evaluating base model {base} instead"
        )
        return base
    return model_path


def evaluate_model(
    model_path: str,
    gsm8k_test: Dataset | None = None,
    math500_test: Dataset | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 2048,
    vllm_base_url: str | None = None,
) -> dict:
    """Evaluate a model on GSM8K test and/or MATH-500.

    Args:
        model_path: HuggingFace model ID or local checkpoint path.
        vllm_base_url: If provided, use an already-running vLLM API server
                       instead of loading the model in-process.
    """
    if vllm_base_url:
        return _evaluate_via_api(
            model_path, gsm8k_test, math500_test, temperature, max_tokens, vllm_base_url
        )

    from vllm import LLM, SamplingParams

    vllm_path = _resolve_vllm_model_path(model_path)
    logger.info(f"Loading model for evaluation: {vllm_path}")
    llm = LLM(
        model=vllm_path,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0 if temperature == 0.0 else 0.8,
    )

    results = {}
    if gsm8k_test is not None:
        logger.info(f"Evaluating on GSM8K test ({len(gsm8k_test)} examples)")
        results["gsm8k"] = _evaluate_dataset(llm, gsm8k_test, sampling_params, "gsm8k")
        logger.info(f"GSM8K accuracy: {results['gsm8k']['accuracy']:.4f}")
    if math500_test is not None:
        logger.info(f"Evaluating on MATH-500 ({len(math500_test)} examples)")
        results["math500"] = _evaluate_dataset(llm, math500_test, sampling_params, "math")
        logger.info(f"MATH-500 accuracy: {results['math500']['accuracy']:.4f}")
    return results


def _evaluate_via_api(
    model_path: str,
    gsm8k_test: Dataset | None,
    math500_test: Dataset | None,
    temperature: float,
    max_tokens: int,
    base_url: str,
) -> dict:
    """Evaluate using an already-running vLLM OpenAI-compatible server."""
    import asyncio
    from openai import AsyncOpenAI

    vllm_model = _resolve_vllm_model_path(model_path)
    client = AsyncOpenAI(base_url=base_url, api_key="dummy")
    logger.info(f"Evaluating via API server ({base_url}) model={vllm_model}")

    async def generate_batch(dataset: Dataset) -> list[str]:
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(vllm_model)
        except Exception:
            pass

        async def single(example):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_COT},
                {"role": "user", "content": example["problem"]},
            ]
            try:
                resp = await client.chat.completions.create(
                    model=vllm_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception:
                return ""

        return await asyncio.gather(*[single(dataset[i]) for i in range(len(dataset))])

    results = {}
    for name, dataset, source in [
        ("gsm8k", gsm8k_test, "gsm8k"),
        ("math500", math500_test, "math"),
    ]:
        if dataset is None:
            continue
        logger.info(f"Evaluating on {name} ({len(dataset)} examples)")
        outputs = asyncio.run(generate_batch(dataset))
        correct, total = 0, len(dataset)
        for i, generated in enumerate(outputs):
            predicted = extract_answer(generated, source=source)
            if predicted and check_answer_equivalence(predicted, dataset[i]["solution"]):
                correct += 1
        accuracy = correct / total if total > 0 else 0.0
        logger.info(f"{name} accuracy: {accuracy:.4f}")
        results[name] = {"accuracy": accuracy, "correct": correct, "total": total}
    return results


def _evaluate_dataset(
    llm,
    dataset: Dataset,
    sampling_params,
    source: str,
) -> dict:
    """Evaluate on a single dataset.

    Returns dict with accuracy, correct count, total, and per-level breakdown.
    """
    from vllm import LLM, SamplingParams

    # Build prompts
    tokenizer = llm.get_tokenizer()
    prompts = []
    for example in dataset:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_COT},
            {"role": "user", "content": example["problem"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Score
    correct = 0
    total = len(dataset)
    per_level: dict[str, dict] = {}
    per_subject: dict[str, dict] = {}
    wrong_examples = []

    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        predicted = extract_answer(generated, source=source)
        gt = dataset[i]["solution"]

        is_correct = predicted is not None and check_answer_equivalence(
            predicted, gt
        )

        if is_correct:
            correct += 1

        # Per-level tracking
        level = str(dataset[i].get("level", "unknown"))
        if level not in per_level:
            per_level[level] = {"correct": 0, "total": 0}
        per_level[level]["total"] += 1
        if is_correct:
            per_level[level]["correct"] += 1

        # Per-subject tracking
        subject = dataset[i].get("subject", "unknown")
        if subject not in per_subject:
            per_subject[subject] = {"correct": 0, "total": 0}
        per_subject[subject]["total"] += 1
        if is_correct:
            per_subject[subject]["correct"] += 1

        # Log some wrong examples for debugging
        if not is_correct and len(wrong_examples) < 5:
            wrong_examples.append(
                {
                    "problem": dataset[i]["problem"][:200],
                    "ground_truth": gt,
                    "predicted": predicted,
                    "generated": generated[:300],
                }
            )

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "per_level": {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
            for k, v in sorted(per_level.items())
        },
        "per_subject": {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
            for k, v in sorted(per_subject.items())
        },
        "wrong_examples": wrong_examples,
    }


def save_eval_results(
    results: dict,
    output_dir: str,
    model_name: str,
    round_num: int,
) -> str:
    """Save evaluation results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"eval_{model_name}_round_{round_num}.json"
    path = output_dir / filename

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Eval results saved to {path}")
    return str(path)
