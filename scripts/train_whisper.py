#!/usr/bin/env python3
"""Fine-tune Whisper large-v3 with LoRA for multi-task ASR+ST.

Pipeline A (Cascaded): This trains the ASR component.
For X→eng pairs, also trains direct ST via Whisper's translate task.
For X→non-eng pairs (arn-spa, que-spa, bho-hin), trains ASR only
(translation handled downstream by NLLB).

Key techniques:
- LoRA rank 128 on all attention + FFN layers
- Multi-task learning: ASR (30%) + ST (70%) when target is English
- SpecAugment + speed perturbation + noise injection
- Gradient checkpointing for A100 40GB
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import WhisperSTDataset, whisper_collate_fn
from src.data.augmentation import AudioAugmentor


def load_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, help="Language pair ID (e.g., bem-eng)")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--lang_config", default="configs/language_pairs.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume_from", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.lang_config) as f:
        lang_cfg = yaml.safe_load(f)["language_pairs"][args.pair]

    return args, train_cfg, lang_cfg


@dataclass
class WhisperDataCollator:
    """Data collator that handles padding and forced decoder input IDs."""
    processor: Any
    task: str = "transcribe"
    language: str = "en"

    def __call__(self, features):
        input_features = torch.stack([f["input_features"] for f in features])

        # Pad labels
        label_lengths = [f["labels"].shape[0] for f in features]
        max_label_len = max(label_lengths)
        labels = torch.full((len(features), max_label_len), -100, dtype=torch.long)
        for i, f in enumerate(features):
            labels[i, :f["labels"].shape[0]] = f["labels"]

        return {
            "input_features": input_features,
            "labels": labels,
        }


def compute_max_steps(data_hours, cfg):
    """Scale max_steps based on dataset size."""
    base_steps = cfg["whisper"]["training"]["max_steps"]
    # Rough heuristic: 1 hour of data ≈ ~120 utterances
    # With batch size 32 and 3 epochs: steps = 120 * hours * 3 / 32
    estimated_samples = data_hours * 120
    estimated_steps = int(estimated_samples * 5 / 32)  # ~5 epochs
    return min(max(estimated_steps, 1000), base_steps * 2)


def main():
    args, train_cfg, lang_cfg = load_configs()
    wcfg = train_cfg["whisper"]
    pair_id = args.pair

    # Determine output directory
    output_dir = args.output_dir or f"{train_cfg['paths']['output_root']}/whisper/{pair_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Data paths
    data_dir = f"{train_cfg['paths']['data_root']}/processed/{pair_id}"
    train_manifest = f"{data_dir}/train.jsonl"
    dev_manifest = f"{data_dir}/dev.jsonl"

    if not os.path.exists(train_manifest):
        print(f"ERROR: Training manifest not found: {train_manifest}")
        sys.exit(1)

    # Determine task mode
    target_is_english = lang_cfg["tgt_lang"] == "eng"
    whisper_lang = lang_cfg.get("whisper_lang")

    print(f"{'='*60}")
    print(f"Training Whisper LoRA for: {pair_id}")
    print(f"  Target is English: {target_is_english}")
    print(f"  Whisper lang code: {whisper_lang or 'N/A (will fine-tune)'}")
    print(f"  Task mode: {'ASR+ST' if target_is_english else 'ASR-only'}")
    print(f"  Output: {output_dir}")

    # Load model and processor
    cache_dir = train_cfg["paths"]["model_cache"]
    processor = WhisperProcessor.from_pretrained(
        wcfg["model_name"], cache_dir=cache_dir
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        wcfg["model_name"],
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    )

    # Enable gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # Apply LoRA
    lora_cfg = wcfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Setup augmentation
    aug_cfg = wcfg["augmentation"]
    augmentor = AudioAugmentor(
        speed_perturbation=aug_cfg["speed_perturbation"],
        speed_factors=aug_cfg["speed_factors"],
        noise_injection=aug_cfg["noise_injection"],
        noise_snr_db=aug_cfg["noise_snr_db"],
    )

    # Create datasets
    task_mode = "both" if target_is_english else "asr"
    train_dataset = WhisperSTDataset(
        manifest_path=train_manifest,
        processor=processor,
        language=whisper_lang or lang_cfg["src_lang"],
        task=task_mode,
        asr_weight=wcfg["multitask"]["asr_weight"],
        max_audio_len_sec=wcfg["training"]["max_audio_length_sec"],
        augmentor=augmentor,
        target_is_english=target_is_english,
    )

    eval_dataset = None
    if os.path.exists(dev_manifest):
        eval_dataset = WhisperSTDataset(
            manifest_path=dev_manifest,
            processor=processor,
            language=whisper_lang or lang_cfg["src_lang"],
            task="asr" if not target_is_english else "translate",
            max_audio_len_sec=wcfg["training"]["max_audio_length_sec"],
            target_is_english=target_is_english,
        )

    # Compute max steps based on data size
    data_hours = lang_cfg.get("data_hours", 30)
    max_steps = compute_max_steps(data_hours, train_cfg)
    print(f"  Max training steps: {max_steps} (based on {data_hours}h of data)")

    # Training arguments
    t_cfg = wcfg["training"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        warmup_steps=t_cfg["warmup_steps"],
        fp16=t_cfg["fp16"],
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=t_cfg["eval_steps"] if eval_dataset else None,
        save_steps=t_cfg["save_steps"],
        logging_steps=t_cfg["logging_steps"],
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="tensorboard",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        dataloader_num_workers=t_cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # Data collator
    collator = WhisperDataCollator(
        processor=processor,
        task="transcribe",
        language=whisper_lang or "en",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor.feature_extractor,
    )

    # Train
    print(f"\nStarting training...")
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save final model
    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    print(f"\nTraining complete. Model saved to: {final_dir}")

    # Save training info
    info = {
        "pair_id": pair_id,
        "model": wcfg["model_name"],
        "lora_rank": lora_cfg["rank"],
        "max_steps": max_steps,
        "task_mode": task_mode,
        "target_is_english": target_is_english,
    }
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
