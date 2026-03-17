#!/usr/bin/env python3
"""Cascaded inference: Whisper ASR → NLLB MT.

Generates N-best hypotheses for MBR decoding.
For X→eng pairs with Whisper translate capability, also generates direct ST hypotheses.
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
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import STManifest


def load_audio(audio_path, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.mean(dim=0)


def whisper_asr_inference(
    model, processor, audio_paths, language=None, batch_size=8, device="cuda"
):
    """Run Whisper ASR on a list of audio files. Returns transcriptions."""
    transcriptions = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Whisper ASR"):
        batch_paths = audio_paths[i : i + batch_size]
        batch_audio = [load_audio(p).numpy() for p in batch_paths]

        inputs = processor(
            batch_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(device)

        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        ) if language else None

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
            )

        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        transcriptions.extend(decoded)

    return transcriptions


def whisper_st_nbest(
    model, processor, audio_paths, language=None,
    num_beams=5, num_return=50, batch_size=4, device="cuda"
):
    """Run Whisper direct ST with N-best output for MBR."""
    all_nbest = []

    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Whisper ST N-best"):
        batch_paths = audio_paths[i : i + batch_size]
        batch_audio = [load_audio(p).numpy() for p in batch_paths]

        inputs = processor(
            batch_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(device)

        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task="translate"
        ) if language else None

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225,
                num_beams=num_beams,
                num_return_sequences=min(num_return, num_beams),
                do_sample=False,
            )

        # Reshape: (batch * num_return, seq) -> (batch, num_return, seq)
        num_ret = min(num_return, num_beams)
        for b in range(len(batch_paths)):
            start = b * num_ret
            end = start + num_ret
            hypotheses = processor.batch_decode(
                generated[start:end], skip_special_tokens=True
            )
            all_nbest.append(hypotheses)

    return all_nbest


def nllb_translate(
    model, tokenizer, texts, src_lang, tgt_lang, batch_size=32, device="cuda"
):
    """Translate texts using NLLB. Returns translations."""
    translations = []
    tokenizer.src_lang = src_lang

    # Get target language token ID
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    for i in tqdm(range(0, len(texts), batch_size), desc="NLLB MT"):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_new_tokens=256,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(decoded)

    return translations


def nllb_translate_nbest(
    model, tokenizer, texts, src_lang, tgt_lang,
    num_beams=5, num_return=50, batch_size=16, device="cuda"
):
    """Translate with N-best output for MBR."""
    all_nbest = []
    tokenizer.src_lang = src_lang
    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    num_ret = min(num_return, num_beams)

    for i in tqdm(range(0, len(texts), batch_size), desc="NLLB N-best"):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,
                max_new_tokens=256,
                num_beams=num_beams,
                num_return_sequences=num_ret,
                do_sample=False,
            )

        for b in range(len(batch_texts)):
            start = b * num_ret
            end = start + num_ret
            hypotheses = tokenizer.batch_decode(
                generated[start:end], skip_special_tokens=True
            )
            all_nbest.append(hypotheses)

    return all_nbest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--lang_config", default="configs/language_pairs.yaml")
    parser.add_argument("--whisper_model_dir", default=None)
    parser.add_argument("--nllb_model_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--nbest", type=int, default=50, help="N-best for MBR")
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

    # Load test manifest
    data_dir = f"{train_cfg['paths']['data_root']}/processed/{pair_id}"
    manifest_path = f"{data_dir}/{args.split}.jsonl"
    entries = STManifest.load(manifest_path)
    audio_paths = [e["audio_path"] for e in entries]

    print(f"{'='*60}")
    print(f"Cascaded inference: {pair_id} ({len(entries)} utterances)")

    # --- Step 1: Whisper ASR ---
    cache_dir = train_cfg["paths"]["model_cache"]
    wcfg = train_cfg["whisper"]
    whisper_dir = args.whisper_model_dir or f"{train_cfg['paths']['output_root']}/whisper/{pair_id}/final"

    print(f"\nLoading Whisper from: {whisper_dir}")
    processor = WhisperProcessor.from_pretrained(wcfg["model_name"], cache_dir=cache_dir)

    base_model = WhisperForConditionalGeneration.from_pretrained(
        wcfg["model_name"], cache_dir=cache_dir, torch_dtype=torch.float16
    ).to(device)

    if os.path.exists(f"{whisper_dir}/adapter_config.json"):
        model = PeftModel.from_pretrained(base_model, whisper_dir)
    else:
        model = base_model
    model.eval()

    whisper_lang = lang_cfg.get("whisper_lang")

    # ASR transcriptions
    print("\nRunning ASR...")
    transcriptions = whisper_asr_inference(
        model, processor, audio_paths, language=whisper_lang, device=device
    )

    # Save ASR output
    with open(f"{output_dir}/asr_{args.split}.txt", "w") as f:
        for t in transcriptions:
            f.write(t.strip() + "\n")

    # N-best from Whisper direct ST (if target is English)
    cascade_nbest = None
    if lang_cfg.get("whisper_can_translate"):
        print("\nRunning Whisper direct ST N-best...")
        cascade_nbest = whisper_st_nbest(
            model, processor, audio_paths, language=whisper_lang,
            num_beams=args.num_beams, num_return=args.nbest, device=device
        )

    # Free Whisper memory
    del model, base_model
    torch.cuda.empty_cache()

    # --- Step 2: NLLB MT ---
    nllb_src = lang_cfg.get("nllb_src_lang")
    nllb_tgt = lang_cfg.get("nllb_tgt_lang")

    if nllb_src:
        ncfg = train_cfg["nllb"]
        nllb_dir = args.nllb_model_dir or f"{train_cfg['paths']['output_root']}/nllb/{pair_id}/best"

        if not os.path.exists(nllb_dir):
            nllb_dir = f"{train_cfg['paths']['output_root']}/nllb/{pair_id}/final"

        print(f"\nLoading NLLB from: {nllb_dir}")
        nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_dir)
        nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_dir).to(device)
        nllb_model.eval()

        # 1-best translations
        print("\nRunning NLLB MT (1-best)...")
        translations_1best = nllb_translate(
            nllb_model, nllb_tokenizer, transcriptions,
            nllb_src, nllb_tgt, device=device
        )

        # N-best translations for MBR
        print("\nRunning NLLB MT (N-best)...")
        nllb_nbest = nllb_translate_nbest(
            nllb_model, nllb_tokenizer, transcriptions,
            nllb_src, nllb_tgt,
            num_beams=args.num_beams, num_return=args.nbest, device=device
        )

        del nllb_model
        torch.cuda.empty_cache()
    else:
        # No NLLB for this pair (e.g., arn-spa where Mapuzugun not in NLLB)
        translations_1best = transcriptions  # fallback
        nllb_nbest = [[t] for t in transcriptions]

    # Save cascade outputs
    with open(f"{output_dir}/cascade_1best_{args.split}.txt", "w") as f:
        for t in translations_1best:
            f.write(t.strip() + "\n")

    with open(f"{output_dir}/cascade_nbest_{args.split}.json", "w") as f:
        json.dump(nllb_nbest, f, ensure_ascii=False)

    if cascade_nbest:
        with open(f"{output_dir}/whisper_st_nbest_{args.split}.json", "w") as f:
            json.dump(cascade_nbest, f, ensure_ascii=False)

    print(f"\nCascade inference complete. Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
