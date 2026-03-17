#!/usr/bin/env python3
"""Preprocess downloaded data into standardized JSONL manifests.

Each pair gets its own preprocessing logic due to different data formats.
Output: /workspace/data/processed/{pair}/{split}.jsonl

Each line in the manifest:
{"audio_path": "/abs/path.wav", "src_text": "transcription", "tgt_text": "translation", "duration": 5.2}
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import soundfile as sf
import yaml

RAW_DIR = "/workspace/data/raw"
PROCESSED_DIR = "/workspace/data/processed"


def get_duration(audio_path):
    """Get audio duration in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        return 0.0


def convert_to_wav_16k(input_path, output_path):
    """Convert any audio file to 16kHz mono WAV."""
    if os.path.exists(output_path):
        return output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = f'ffmpeg -y -i "{input_path}" -ar 16000 -ac 1 -f wav "{output_path}" -loglevel error'
    subprocess.run(cmd, shell=True, check=True)
    return output_path


def write_manifest(entries, output_path):
    """Write manifest JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(entries)} entries to {output_path}")


# ============================================================
# Per-pair preprocessing functions
# ============================================================

def preprocess_arn_spa(raw_dir, proc_dir):
    """Mapuzugun-Spanish: HuggingFace dataset."""
    from datasets import load_from_disk

    entries_by_split = {}
    for split in ["train", "validation", "test"]:
        split_dir = f"{raw_dir}/{split}"
        if not os.path.exists(split_dir):
            # Try loading from cache
            cache_dir = f"{raw_dir}/cache"
            from datasets import load_dataset
            ds = load_dataset("mengct00/Mapudungun_iwslt26", cache_dir=cache_dir)
            for s in ds:
                ds[s].save_to_disk(f"{raw_dir}/{s}")

        try:
            ds = load_from_disk(split_dir)
        except Exception:
            continue

        wav_dir = f"{proc_dir}/wavs/{split}"
        os.makedirs(wav_dir, exist_ok=True)

        entries = []
        for i, sample in enumerate(ds):
            # Adapt to actual dataset columns
            audio = sample.get("audio", {})
            audio_array = audio.get("array") if isinstance(audio, dict) else None
            sr = audio.get("sampling_rate", 16000) if isinstance(audio, dict) else 16000

            wav_path = f"{wav_dir}/{i:06d}.wav"
            if audio_array is not None and not os.path.exists(wav_path):
                import numpy as np
                sf.write(wav_path, np.array(audio_array), sr)

            src_text = sample.get("transcription", sample.get("src_text", ""))
            tgt_text = sample.get("translation", sample.get("tgt_text", ""))
            duration = len(audio_array) / sr if audio_array is not None else 0

            if src_text and tgt_text:
                entries.append({
                    "audio_path": wav_path,
                    "src_text": src_text.strip(),
                    "tgt_text": tgt_text.strip(),
                    "duration": round(duration, 2),
                })

        entries_by_split[split] = entries

    for split, entries in entries_by_split.items():
        out_split = "dev" if split == "validation" else split
        write_manifest(entries, f"{proc_dir}/{out_split}.jsonl")


def preprocess_github_generic(pair_id, raw_dir, proc_dir):
    """Generic preprocessing for GitHub-sourced datasets.

    Discovers the data format and attempts to parse:
    - TSV files with columns like: audio_file, transcription, translation
    - Separate text files with line-aligned transcriptions/translations
    - JSON/JSONL metadata files
    """
    # Find the cloned repo directory
    repo_dirs = [d for d in Path(raw_dir).iterdir() if d.is_dir() and d.name != "cache"]
    if not repo_dirs:
        print(f"  No data directory found for {pair_id}")
        return

    repo_dir = repo_dirs[0]
    print(f"  Processing repo: {repo_dir}")

    # Look for common data organization patterns
    # Pattern 1: TSV/CSV files
    tsv_files = list(repo_dir.rglob("*.tsv")) + list(repo_dir.rglob("*.csv"))
    json_files = list(repo_dir.rglob("*.json")) + list(repo_dir.rglob("*.jsonl"))

    # Pattern 2: Separate directories for audio, transcription, translation
    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.flac"]:
        audio_files.extend(repo_dir.rglob(ext))

    # Pattern 3: Line-aligned text files
    txt_files = list(repo_dir.rglob("*.txt"))

    print(f"  Found: {len(tsv_files)} TSV, {len(json_files)} JSON, "
          f"{len(audio_files)} audio, {len(txt_files)} TXT files")

    # Try TSV first (most common format for IWSLT datasets)
    for tsv_file in tsv_files:
        try:
            entries = parse_tsv_manifest(tsv_file, repo_dir, proc_dir)
            if entries:
                split = guess_split(tsv_file.name)
                write_manifest(entries, f"{proc_dir}/{split}.jsonl")
        except Exception as e:
            print(f"  Failed to parse {tsv_file}: {e}")

    # Try JSONL
    for jf in json_files:
        if jf.suffix == ".jsonl":
            try:
                entries = parse_jsonl_manifest(jf, repo_dir, proc_dir)
                if entries:
                    split = guess_split(jf.name)
                    write_manifest(entries, f"{proc_dir}/{split}.jsonl")
            except Exception as e:
                print(f"  Failed to parse {jf}: {e}")


def parse_tsv_manifest(tsv_path, repo_dir, proc_dir):
    """Parse a TSV file into manifest entries."""
    import csv
    entries = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = reader.fieldnames
        print(f"  TSV headers: {headers}")

        for row in reader:
            # Common column name patterns
            audio_col = next((h for h in headers if h.lower() in
                            ["audio", "audio_path", "path", "wav", "file", "audio_file"]), None)
            src_col = next((h for h in headers if h.lower() in
                          ["transcription", "src_text", "source", "transcript", "src"]), None)
            tgt_col = next((h for h in headers if h.lower() in
                          ["translation", "tgt_text", "target", "tgt", "en", "eng", "english",
                           "spanish", "spa", "hindi", "hin"]), None)

            if not (audio_col and (src_col or tgt_col)):
                continue

            audio_path = row.get(audio_col, "")
            if not os.path.isabs(audio_path):
                # Try relative to repo dir
                for base in [repo_dir, tsv_path.parent]:
                    candidate = os.path.join(base, audio_path)
                    if os.path.exists(candidate):
                        audio_path = candidate
                        break

            src_text = row.get(src_col, "") if src_col else ""
            tgt_text = row.get(tgt_col, "") if tgt_col else ""

            if os.path.exists(audio_path) and (src_text or tgt_text):
                # Convert to 16kHz WAV
                wav_name = Path(audio_path).stem + ".wav"
                wav_path = f"{proc_dir}/wavs/{wav_name}"
                try:
                    convert_to_wav_16k(audio_path, wav_path)
                    duration = get_duration(wav_path)
                except Exception:
                    continue

                entries.append({
                    "audio_path": wav_path,
                    "src_text": src_text.strip(),
                    "tgt_text": tgt_text.strip(),
                    "duration": round(duration, 2),
                })

    return entries


def parse_jsonl_manifest(jsonl_path, repo_dir, proc_dir):
    """Parse a JSONL file into manifest entries."""
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            audio_path = row.get("audio_path", row.get("audio", row.get("path", "")))
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(repo_dir, audio_path)

            src_text = row.get("src_text", row.get("transcription", ""))
            tgt_text = row.get("tgt_text", row.get("translation", ""))

            if os.path.exists(audio_path) and (src_text or tgt_text):
                entries.append({
                    "audio_path": audio_path,
                    "src_text": src_text.strip(),
                    "tgt_text": tgt_text.strip(),
                    "duration": row.get("duration", get_duration(audio_path)),
                })
    return entries


def guess_split(filename):
    """Guess the split from a filename."""
    name = filename.lower()
    if "test" in name:
        return "test"
    elif "dev" in name or "valid" in name or "val" in name:
        return "dev"
    else:
        return "train"


def preprocess_hf_african_celtic(pair_id, raw_dir, proc_dir, subset):
    """Process McGill-NLP/african_celtic_dataset for ibo/hau/yor."""
    from datasets import load_dataset

    print(f"  Loading McGill-NLP/african_celtic_dataset subset={subset}")
    ds = load_dataset("McGill-NLP/african_celtic_dataset", subset,
                      cache_dir=f"{raw_dir}/cache")

    for split_name, split_data in ds.items():
        out_split = "dev" if split_name == "validation" else split_name
        wav_dir = f"{proc_dir}/wavs/{out_split}"
        os.makedirs(wav_dir, exist_ok=True)

        entries = []
        for i, sample in enumerate(split_data):
            audio = sample.get("audio", {})
            audio_array = audio.get("array") if isinstance(audio, dict) else None
            sr = audio.get("sampling_rate", 16000) if isinstance(audio, dict) else 16000

            wav_path = f"{wav_dir}/{i:06d}.wav"
            if audio_array is not None and not os.path.exists(wav_path):
                import numpy as np
                sf.write(wav_path, audio_array if isinstance(audio_array, list) else audio_array, sr)

            src_text = sample.get("transcription", sample.get("src_text", ""))
            tgt_text = sample.get("translation", sample.get("tgt_text", ""))
            duration = len(audio_array) / sr if audio_array is not None else 0

            if src_text and tgt_text:
                entries.append({
                    "audio_path": wav_path,
                    "src_text": src_text.strip(),
                    "tgt_text": tgt_text.strip(),
                    "duration": round(duration, 2),
                })

        write_manifest(entries, f"{proc_dir}/{out_split}.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Preprocess IWSLT 2026 LR-ST data")
    parser.add_argument("--config", default="configs/language_pairs.yaml")
    parser.add_argument("--pairs", nargs="*", help="Specific pairs (default: all)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pairs = config["language_pairs"]
    if args.pairs:
        pairs = {k: v for k, v in pairs.items() if k in args.pairs}

    for pair_id, cfg in pairs.items():
        raw_dir = f"{RAW_DIR}/{pair_id}"
        proc_dir = f"{PROCESSED_DIR}/{pair_id}"
        os.makedirs(proc_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Preprocessing: {pair_id} ({cfg['name']})")

        if not os.path.exists(raw_dir):
            print(f"  Raw data not found at {raw_dir} — skipping")
            continue

        try:
            if pair_id == "arn-spa":
                preprocess_arn_spa(raw_dir, proc_dir)
            elif pair_id in ("ibo-eng", "hau-eng", "yor-eng"):
                subset = cfg.get("hf_subset", pair_id.split("-")[0])
                preprocess_hf_african_celtic(pair_id, raw_dir, proc_dir, subset)
            else:
                preprocess_github_generic(pair_id, raw_dir, proc_dir)
        except Exception as e:
            print(f"  ERROR preprocessing {pair_id}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*60}")
    print("Preprocessing Summary:")
    for pair_id in pairs:
        proc_dir = f"{PROCESSED_DIR}/{pair_id}"
        for split in ["train", "dev", "test"]:
            manifest = f"{proc_dir}/{split}.jsonl"
            if os.path.exists(manifest):
                count = sum(1 for _ in open(manifest))
                print(f"  {pair_id}/{split}: {count} entries")


if __name__ == "__main__":
    main()
