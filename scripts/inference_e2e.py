#!/usr/bin/env python3
"""End-to-end inference using fine-tuned SeamlessM4T v2.

Generates N-best hypotheses for MBR decoding.
"""

import argparse
import json
import os
import sys

import torch
import torchaudio
import yaml
from tqdm import tqdm
from transformers import (
    SeamlessM4Tv2ForSpeechToText,
    AutoProcessor,
)
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import STManifest


def load_audio(audio_path, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.mean(dim=0)


def seamless_translate(
    model, processor, audio_paths, src_lang, tgt_lang,
    batch_size=4, device="cuda"
):
    """1-best translation using SeamlessM4T v2."""
    translations = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="SeamlessM4T ST"):
        batch_paths = audio_paths[i : i + batch_size]
        batch_audio = [load_audio(p).numpy() for p in batch_paths]

        inputs = processor(
            audios=batch_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            src_lang=src_lang,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                tgt_lang=tgt_lang,
                max_new_tokens=256,
            )

        decoded = processor.batch_decode(generated[0], skip_special_tokens=True)
        translations.extend(decoded)

    return translations


def seamless_translate_nbest(
    model, processor, audio_paths, src_lang, tgt_lang,
    num_beams=10, num_return=50, batch_size=2, device="cuda"
):
    """N-best translation using SeamlessM4T v2 for MBR."""
    all_nbest = []
    num_ret = min(num_return, num_beams)

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="SeamlessM4T N-best"):
        batch_paths = audio_paths[i : i + batch_size]
        batch_audio = [load_audio(p).numpy() for p in batch_paths]

        inputs = processor(
            audios=batch_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            src_lang=src_lang,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                tgt_lang=tgt_lang,
                max_new_tokens=256,
                num_beams=num_beams,
                num_return_sequences=num_ret,
                do_sample=False,
            )

        # Handle output format - SeamlessM4T may return tuple
        if isinstance(generated, tuple):
            token_ids = generated[0]
        else:
            token_ids = generated

        for b in range(len(batch_paths)):
            start = b * num_ret
            end = start + num_ret
            hypotheses = processor.batch_decode(
                token_ids[start:end], skip_special_tokens=True
            )
            all_nbest.append(hypotheses)

    return all_nbest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--lang_config", default="configs/language_pairs.yaml")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--nbest", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.lang_config) as f:
        lang_cfg = yaml.safe_load(f)["language_pairs"][args.pair]

    pair_id = args.pair
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or f"{train_cfg['paths']['output_root']}/ensemble/{pair_id}"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = f"{train_cfg['paths']['data_root']}/processed/{pair_id}"
    manifest_path = f"{data_dir}/{args.split}.jsonl"
    entries = STManifest.load(manifest_path)
    audio_paths = [e["audio_path"] for e in entries]

    src_lang = lang_cfg["seamless_src_lang"]
    tgt_lang = lang_cfg["seamless_tgt_lang"]

    print(f"{'='*60}")
    print(f"E2E inference: {pair_id} ({len(entries)} utterances)")
    print(f"  {src_lang} -> {tgt_lang}")

    # Load model
    cache_dir = train_cfg["paths"]["model_cache"]
    scfg = train_cfg["seamless"]
    model_dir = args.model_dir or f"{train_cfg['paths']['output_root']}/seamless/{pair_id}/final"

    print(f"\nLoading SeamlessM4T from: {model_dir}")
    processor = AutoProcessor.from_pretrained(scfg["model_name"], cache_dir=cache_dir)

    base_model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        scfg["model_name"], cache_dir=cache_dir, torch_dtype=torch.float16
    ).to(device)

    if os.path.exists(f"{model_dir}/adapter_config.json"):
        model = PeftModel.from_pretrained(base_model, model_dir)
    else:
        model = base_model
    model.eval()

    # 1-best
    print("\nRunning 1-best translation...")
    translations = seamless_translate(
        model, processor, audio_paths, src_lang, tgt_lang, device=device
    )

    with open(f"{output_dir}/e2e_1best_{args.split}.txt", "w") as f:
        for t in translations:
            f.write(t.strip() + "\n")

    # N-best for MBR
    print("\nRunning N-best translation...")
    nbest = seamless_translate_nbest(
        model, processor, audio_paths, src_lang, tgt_lang,
        num_beams=args.num_beams, num_return=args.nbest, device=device
    )

    with open(f"{output_dir}/e2e_nbest_{args.split}.json", "w") as f:
        json.dump(nbest, f, ensure_ascii=False)

    del model, base_model
    torch.cuda.empty_cache()

    print(f"\nE2E inference complete. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
