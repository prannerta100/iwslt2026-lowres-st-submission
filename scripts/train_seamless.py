#!/usr/bin/env python3
"""Fine-tune SeamlessM4T v2 large with LoRA for end-to-end speech translation.

Pipeline B (E2E): Direct speech-to-text translation for all language pairs.
SeamlessM4T v2 natively supports many target languages including English, Spanish, Hindi.

Key techniques:
- LoRA rank 64 on attention + FFN layers
- Gradient checkpointing for A100 40GB
- Lower learning rate (5e-5) to avoid catastrophic forgetting
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
import yaml
from transformers import (
    SeamlessM4Tv2ForSpeechToText,
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import SeamlessSTDataset, seamless_collate_fn
from src.data.augmentation import AudioAugmentor


def load_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
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
class SeamlessDataCollator:
    """Collator for SeamlessM4T with variable-length inputs."""
    processor: Any

    def __call__(self, features):
        # Pad input features (audio)
        max_audio_len = max(f["input_features"].shape[-1] for f in features)
        max_label_len = max(f["labels"].shape[0] for f in features)

        batch_input_features = []
        batch_attention_mask = []
        batch_labels = torch.full(
            (len(features), max_label_len), -100, dtype=torch.long
        )

        for i, f in enumerate(features):
            feat = f["input_features"]
            pad_len = max_audio_len - feat.shape[-1]

            if pad_len > 0:
                feat = torch.nn.functional.pad(feat, (0, pad_len))

            batch_input_features.append(feat)

            mask = torch.ones(feat.shape[-1], dtype=torch.long)
            if pad_len > 0:
                mask[-pad_len:] = 0
            batch_attention_mask.append(mask)

            batch_labels[i, : f["labels"].shape[0]] = f["labels"]

        return {
            "input_features": torch.stack(batch_input_features),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": batch_labels,
        }


def main():
    args, train_cfg, lang_cfg = load_configs()
    scfg = train_cfg["seamless"]
    pair_id = args.pair

    output_dir = args.output_dir or f"{train_cfg['paths']['output_root']}/seamless/{pair_id}"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = f"{train_cfg['paths']['data_root']}/processed/{pair_id}"
    train_manifest = f"{data_dir}/train.jsonl"
    dev_manifest = f"{data_dir}/dev.jsonl"

    if not os.path.exists(train_manifest):
        print(f"ERROR: Training manifest not found: {train_manifest}")
        sys.exit(1)

    src_lang = lang_cfg["seamless_src_lang"]
    tgt_lang = lang_cfg["seamless_tgt_lang"]

    print(f"{'='*60}")
    print(f"Training SeamlessM4T v2 LoRA for: {pair_id}")
    print(f"  {src_lang} -> {tgt_lang}")
    print(f"  Output: {output_dir}")

    # Load model and processor
    cache_dir = train_cfg["paths"]["model_cache"]
    processor = AutoProcessor.from_pretrained(
        scfg["model_name"], cache_dir=cache_dir
    )
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        scfg["model_name"],
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    )

    # Gradient checkpointing
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Apply LoRA
    lora_cfg = scfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Augmentation
    augmentor = AudioAugmentor(
        speed_perturbation=True,
        speed_factors=[0.9, 1.0, 1.1],
        noise_injection=True,
        noise_snr_db=[10, 15, 20],
    )

    # Datasets
    train_dataset = SeamlessSTDataset(
        manifest_path=train_manifest,
        processor=processor,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_audio_len_sec=scfg["training"]["max_audio_length_sec"],
        augmentor=augmentor,
    )

    eval_dataset = None
    if os.path.exists(dev_manifest):
        eval_dataset = SeamlessSTDataset(
            manifest_path=dev_manifest,
            processor=processor,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_audio_len_sec=scfg["training"]["max_audio_length_sec"],
        )

    # Scale steps by data size
    data_hours = lang_cfg.get("data_hours", 30)
    max_steps = min(int(data_hours * 120 * 5 / 32), scfg["training"]["max_steps"] * 2)
    max_steps = max(max_steps, 800)

    print(f"  Max training steps: {max_steps}")

    # Training arguments
    t_cfg = scfg["training"]
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
        generation_max_length=256,
        report_to="tensorboard",
        load_best_model_at_end=True if eval_dataset else False,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    collator = SeamlessDataCollator(processor=processor)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print(f"\nStarting training...")
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    # Save
    final_dir = f"{output_dir}/final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    print(f"\nTraining complete. Model saved to: {final_dir}")

    info = {
        "pair_id": pair_id,
        "model": scfg["model_name"],
        "lora_rank": lora_cfg["rank"],
        "max_steps": max_steps,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    }
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
