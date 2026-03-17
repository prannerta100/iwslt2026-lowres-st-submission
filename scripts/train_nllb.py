#!/usr/bin/env python3
"""Fine-tune NLLB-200 1.3B distilled for MT with intra-distillation.

Used in the cascaded pipeline: Whisper ASR output → NLLB MT → target text.
Also used standalone for any pair where we have parallel text.

Key techniques:
- Intra-distillation: regularize fine-tuning with pre-trained model's predictions
  Loss = (1-alpha) * CE_loss + alpha * KL(fine-tuned || pretrained)
- Fine-tune on source transcriptions → target translations
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import NLLBDataset


def load_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--lang_config", default="configs/language_pairs.yaml")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.lang_config) as f:
        lang_cfg = yaml.safe_load(f)["language_pairs"][args.pair]

    return args, train_cfg, lang_cfg


def intra_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Combined CE + KL divergence loss for intra-distillation.

    Args:
        student_logits: Fine-tuned model logits (B, T, V)
        teacher_logits: Frozen pre-trained model logits (B, T, V)
        labels: Target token IDs (B, T), -100 for padding
        alpha: Weight for distillation loss
        temperature: Softmax temperature for KL divergence
    """
    # Standard cross-entropy loss
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    # KL divergence loss (only on non-padding positions)
    mask = (labels != -100).unsqueeze(-1).float()

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    kl_loss = F.kl_div(
        student_log_probs * mask,
        teacher_probs * mask,
        reduction="batchmean",
    ) * (temperature ** 2)

    return (1 - alpha) * ce_loss + alpha * kl_loss


def main():
    args, train_cfg, lang_cfg = load_configs()
    ncfg = train_cfg["nllb"]
    pair_id = args.pair

    # Check if NLLB supports this source language
    nllb_src = lang_cfg.get("nllb_src_lang")
    nllb_tgt = lang_cfg.get("nllb_tgt_lang")

    if not nllb_src:
        print(f"WARNING: {pair_id} source language not in NLLB. "
              "Will attempt fine-tuning with a placeholder language code.")
        nllb_src = "eng_Latn"  # placeholder, will be overridden by fine-tuning

    output_dir = args.output_dir or f"{train_cfg['paths']['output_root']}/nllb/{pair_id}"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = f"{train_cfg['paths']['data_root']}/processed/{pair_id}"
    train_manifest = f"{data_dir}/train.jsonl"
    dev_manifest = f"{data_dir}/dev.jsonl"

    if not os.path.exists(train_manifest):
        print(f"ERROR: Training manifest not found: {train_manifest}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Training NLLB for: {pair_id}")
    print(f"  {nllb_src} -> {nllb_tgt}")
    print(f"  Intra-distillation: {ncfg['intra_distillation']['enabled']}")
    print(f"  Output: {output_dir}")

    cache_dir = train_cfg["paths"]["model_cache"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ncfg["model_name"], cache_dir=cache_dir
    )

    # Load student (fine-tuned) model
    student = AutoModelForSeq2SeqLM.from_pretrained(
        ncfg["model_name"], cache_dir=cache_dir
    ).to(device)

    # Load teacher (frozen pre-trained) model for intra-distillation
    teacher = None
    distill_cfg = ncfg["intra_distillation"]
    if distill_cfg["enabled"]:
        teacher = AutoModelForSeq2SeqLM.from_pretrained(
            ncfg["model_name"], cache_dir=cache_dir
        ).to(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        print("  Teacher model loaded for intra-distillation")

    # Dataset
    t_cfg = ncfg["training"]
    train_dataset = NLLBDataset(
        manifest_path=train_manifest,
        tokenizer=tokenizer,
        src_lang=nllb_src,
        tgt_lang=nllb_tgt,
        max_length=t_cfg["max_source_length"],
    )

    eval_dataset = None
    if os.path.exists(dev_manifest):
        eval_dataset = NLLBDataset(
            manifest_path=dev_manifest,
            tokenizer=tokenizer,
            src_lang=nllb_src,
            tgt_lang=nllb_tgt,
            max_length=t_cfg["max_source_length"],
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=t_cfg["per_device_train_batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=0.01,
    )

    num_epochs = t_cfg["num_epochs"]
    total_steps = len(train_loader) * num_epochs // t_cfg["gradient_accumulation_steps"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=t_cfg["warmup_steps"],
        num_training_steps=total_steps,
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if t_cfg["fp16"] else None

    # Training loop
    print(f"\nStarting training: {num_epochs} epochs, {total_steps} steps")
    global_step = 0
    best_eval_loss = float("inf")

    for epoch in range(num_epochs):
        student.train()
        epoch_loss = 0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(progress):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=t_cfg["fp16"]):
                student_outputs = student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                if teacher is not None and distill_cfg["enabled"]:
                    with torch.no_grad():
                        teacher_outputs = teacher(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                    loss = intra_distillation_loss(
                        student_logits=student_outputs.logits,
                        teacher_logits=teacher_outputs.logits,
                        labels=batch["labels"],
                        alpha=distill_cfg["alpha"],
                        temperature=distill_cfg["temperature"],
                    )
                else:
                    loss = student_outputs.loss

                loss = loss / t_cfg["gradient_accumulation_steps"]

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % t_cfg["gradient_accumulation_steps"] == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * t_cfg["gradient_accumulation_steps"]
            num_batches += 1
            progress.set_postfix(loss=f"{epoch_loss/num_batches:.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Eval
        if eval_dataset:
            eval_loss = evaluate(student, eval_dataset, device, t_cfg)
            print(f"  Eval loss: {eval_loss:.4f}")

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                student.save_pretrained(f"{output_dir}/best")
                tokenizer.save_pretrained(f"{output_dir}/best")
                print(f"  Saved best model (eval_loss={eval_loss:.4f})")

    # Save final
    student.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Delete teacher to free memory
    if teacher is not None:
        del teacher
        torch.cuda.empty_cache()

    print(f"\nTraining complete. Model saved to: {output_dir}/final")

    info = {
        "pair_id": pair_id,
        "model": ncfg["model_name"],
        "nllb_src": nllb_src,
        "nllb_tgt": nllb_tgt,
        "epochs": num_epochs,
        "intra_distillation": distill_cfg["enabled"],
        "best_eval_loss": best_eval_loss if eval_dataset else None,
    }
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(info, f, indent=2)


def evaluate(model, eval_dataset, device, t_cfg):
    model.eval()
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=t_cfg["per_device_train_batch_size"],
        num_workers=2,
    )
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=t_cfg["fp16"]):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            total_loss += outputs.loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    main()
