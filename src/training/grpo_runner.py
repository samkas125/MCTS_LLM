"""GRPO training pipeline using TRL's GRPOTrainer with QLoRA."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger(__name__)


def load_grpo_config(config_path: str = "configs/grpo_config.yaml") -> dict:
    """Load GRPO config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_grpo_training(
    model_name_or_path: str,
    train_dataset: Dataset,
    reward_funcs: list,
    reward_weights: list[float] | None = None,
    output_dir: str = "outputs/checkpoints",
    round_num: int = 0,
    config_path: str = "configs/grpo_config.yaml",
    eval_dataset: Dataset | None = None,
) -> str:
    """Run one round of GRPO training.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        train_dataset: Prompt-only dataset (must have 'prompt' and 'solution' columns).
        reward_funcs: List of reward functions (TRL-compatible callables).
        reward_weights: Optional weights for each reward function.
        output_dir: Base output directory for checkpoints.
        round_num: Current round number (for naming).
        config_path: Path to GRPO config YAML.
        eval_dataset: Optional eval dataset.

    Returns:
        Path to the saved model checkpoint.
    """
    cfg = load_grpo_config(config_path)
    run_output_dir = str(Path(output_dir) / f"grpo_round_{round_num}")

    logger.info(f"Starting GRPO training round {round_num}")
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Output: {run_output_dir}")

    # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["qlora"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["qlora"]["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["qlora"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=cfg["qlora"]["bnb_4bit_use_double_quant"],
    )

    peft_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        task_type="CAUSAL_LM",
        target_modules=cfg["lora"]["target_modules"],
    )

    # GRPO training config
    training_args = GRPOConfig(
        output_dir=run_output_dir,
        # Generation
        num_generations=cfg["grpo"]["num_generations"],
        max_completion_length=cfg["grpo"]["max_completion_length"],
        max_prompt_length=cfg["grpo"]["max_prompt_length"],
        temperature=cfg["grpo"]["temperature"],
        # vLLM
        use_vllm=cfg["vllm"]["use_vllm"],
        vllm_mode=cfg["vllm"]["vllm_mode"],
        vllm_gpu_memory_utilization=cfg["vllm"]["gpu_memory_utilization"],
        # GRPO loss
        loss_type=cfg["grpo"]["loss_type"],
        beta=cfg["grpo"]["beta"],
        epsilon=cfg["grpo"]["epsilon"],
        scale_rewards=cfg["grpo"]["scale_rewards"],
        num_iterations=cfg["grpo"]["num_iterations"],
        # Reward weights
        reward_weights=reward_weights,
        # Training
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        bf16=cfg["training"]["bf16"],
        # Logging
        report_to=cfg["logging"]["report_to"],
        logging_steps=cfg["logging"]["logging_steps"],
        log_completions=cfg["logging"]["log_completions"],
        run_name=f"grpo_round_{round_num}",
        # Saving
        save_steps=cfg["saving"]["save_steps"],
        save_total_limit=cfg["saving"]["save_total_limit"],
    )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save final checkpoint
    final_path = str(Path(run_output_dir) / "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info(f"GRPO round {round_num} complete. Model saved to {final_path}")
    return final_path
