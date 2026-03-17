"""Unified dataset classes for all language pairs."""

import json
import os
import random
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from torch.utils.data import Dataset

from src.data.augmentation import AudioAugmentor


class STManifest:
    """Loads a standardized manifest file (JSONL) with fields:
    audio_path, src_text (transcription), tgt_text (translation), duration
    """

    @staticmethod
    def load(manifest_path: str) -> list[dict]:
        entries = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                entries.append(entry)
        return entries

    @staticmethod
    def save(entries: list[dict], manifest_path: str):
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class WhisperSTDataset(Dataset):
    """Dataset for Whisper LoRA fine-tuning with multi-task ASR+ST.

    Each sample returns either an ASR or ST example based on the configured ratio.
    ASR uses <|transcribe|> task token, ST uses <|translate|> task token.
    For non-English targets, ASR-only mode is used (ST handled by NLLB downstream).
    """

    def __init__(
        self,
        manifest_path: str,
        processor,
        language: str,
        task: str = "both",  # "asr", "translate", "both"
        asr_weight: float = 0.3,
        max_audio_len_sec: float = 30.0,
        augmentor: Optional[AudioAugmentor] = None,
        target_is_english: bool = True,
    ):
        self.entries = STManifest.load(manifest_path)
        self.processor = processor
        self.language = language
        self.task = task
        self.asr_weight = asr_weight
        self.max_audio_len_sec = max_audio_len_sec
        self.augmentor = augmentor
        self.target_is_english = target_is_english
        self.sample_rate = 16000

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        waveform, sr = torchaudio.load(entry["audio_path"])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)  # mono

        # Trim to max length
        max_samples = int(self.max_audio_len_sec * self.sample_rate)
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]

        # Augmentation
        if self.augmentor is not None:
            waveform = self.augmentor(waveform, self.sample_rate)

        # Decide task for this sample
        if self.task == "both":
            do_asr = random.random() < self.asr_weight
        else:
            do_asr = self.task == "asr"

        if do_asr:
            target_text = entry["src_text"]
            task_token = "transcribe"
        else:
            if self.target_is_english:
                target_text = entry["tgt_text"]
                task_token = "translate"
            else:
                # For non-English targets, Whisper does ASR only
                target_text = entry["src_text"]
                task_token = "transcribe"

        # Process audio
        input_features = self.processor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features[0]

        # Process labels
        labels = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            padding=False,
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
            "task": task_token,
        }


class SeamlessSTDataset(Dataset):
    """Dataset for SeamlessM4T v2 fine-tuning for direct speech-to-text translation."""

    def __init__(
        self,
        manifest_path: str,
        processor,
        src_lang: str,
        tgt_lang: str,
        max_audio_len_sec: float = 30.0,
        augmentor: Optional[AudioAugmentor] = None,
    ):
        self.entries = STManifest.load(manifest_path)
        self.processor = processor
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_audio_len_sec = max_audio_len_sec
        self.augmentor = augmentor
        self.sample_rate = 16000

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        waveform, sr = torchaudio.load(entry["audio_path"])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)

        max_samples = int(self.max_audio_len_sec * self.sample_rate)
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]

        if self.augmentor is not None:
            waveform = self.augmentor(waveform, self.sample_rate)

        inputs = self.processor(
            audios=waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            src_lang=self.src_lang,
        )

        with self.processor.as_target_processor():
            labels = self.processor(
                text=entry["tgt_text"],
                return_tensors="pt",
                src_lang=self.tgt_lang,
            ).input_ids[0]

        return {
            "input_features": inputs["input_features"][0],
            "attention_mask": inputs["attention_mask"][0] if "attention_mask" in inputs else None,
            "labels": labels,
        }


class NLLBDataset(Dataset):
    """Text-only parallel dataset for NLLB fine-tuning."""

    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 256,
    ):
        self.entries = STManifest.load(manifest_path)
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        self.tokenizer.src_lang = self.src_lang
        source = self.tokenizer(
            entry["src_text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        self.tokenizer.src_lang = self.tgt_lang
        with self.tokenizer.as_target_tokenizer():
            target = self.tokenizer(
                entry["tgt_text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        labels = target.input_ids[0].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source.input_ids[0],
            "attention_mask": source.attention_mask[0],
            "labels": labels,
        }


def whisper_collate_fn(batch):
    """Custom collate for Whisper that handles variable-length labels."""
    input_features = torch.stack([x["input_features"] for x in batch])
    label_lengths = [x["labels"].shape[0] for x in batch]
    max_label_len = max(label_lengths)

    labels = torch.full((len(batch), max_label_len), -100, dtype=torch.long)
    for i, x in enumerate(batch):
        labels[i, : x["labels"].shape[0]] = x["labels"]

    return {
        "input_features": input_features,
        "labels": labels,
    }


def seamless_collate_fn(batch):
    """Custom collate for SeamlessM4T with variable-length audio."""
    max_audio_len = max(x["input_features"].shape[-1] for x in batch)
    max_label_len = max(x["labels"].shape[0] for x in batch)

    input_features = []
    attention_masks = []
    labels = torch.full((len(batch), max_label_len), -100, dtype=torch.long)

    for i, x in enumerate(batch):
        feat = x["input_features"]
        pad_len = max_audio_len - feat.shape[-1]
        if pad_len > 0:
            feat = torch.nn.functional.pad(feat, (0, pad_len))
        input_features.append(feat)

        mask = torch.ones(max_audio_len, dtype=torch.long)
        if pad_len > 0:
            mask[-pad_len:] = 0
        attention_masks.append(mask)

        labels[i, : x["labels"].shape[0]] = x["labels"]

    return {
        "input_features": torch.stack(input_features),
        "attention_mask": torch.stack(attention_masks),
        "labels": labels,
    }
