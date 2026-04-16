"""SFT baseline training pipeline using TRL's SFTTrainer."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def load_sft_config(config_path: str = "configs/sft_config.yaml") -> dict:
    """Load SFT config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_sft_training(
    model_name_or_path: str,
    train_dataset: Dataset,
    output_dir: str = "outputs/checkpoints",
    round_num: int = 0,
    config_path: str = "configs/sft_config.yaml",
) -> str:
    """Run SFT baseline training using MCTS-generated trajectories.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        train_dataset: Dataset with 'messages' column (full conversations).
        output_dir: Base output directory.
        round_num: Current round number.
        config_path: Path to SFT config YAML.

    Returns:
        Path to the saved model checkpoint.
    """
    cfg = load_sft_config(config_path)
    run_output_dir = str(Path(output_dir) / f"sft_round_{round_num}")

    logger.info(f"Starting SFT training round {round_num}")
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Train size: {len(train_dataset)}")

    # QLoRA config (same as GRPO for fair comparison)
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

    training_args = SFTConfig(
        output_dir=run_output_dir,
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        learning_rate=cfg["training"]["learning_rate"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        bf16=cfg["training"]["bf16"],
        report_to=cfg["logging"]["report_to"],
        logging_steps=cfg["logging"]["logging_steps"],
        run_name=f"sft_round_{round_num}",
        save_steps=cfg["saving"]["save_steps"],
        save_total_limit=cfg["saving"]["save_total_limit"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    trainer.train()

    final_path = str(Path(run_output_dir) / "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info(f"SFT round {round_num} complete. Model saved to {final_path}")
    return final_path
