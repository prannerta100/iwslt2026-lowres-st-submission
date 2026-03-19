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
from huggingface_hub import login

# Try to login with HF_TOKEN if available
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("Logged in to HuggingFace Hub")
    except Exception as e:
        print(f"Warning: Could not login to HuggingFace: {e}")

RAW_DIR = os.path.expanduser("~/workspace/data/raw")
PROCESSED_DIR = os.path.expanduser("~/workspace/data/processed")


def get_duration(audio_path):
    """Get audio duration in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        return 0.0


def convert_to_wav_16k(input_path, output_path):
    """Convert any audio file to 16kHz mono WAV using librosa/soundfile."""
    if os.path.exists(output_path):
        return output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        import librosa
        import numpy as np
        # Load audio with librosa (handles resampling)
        audio, sr = librosa.load(input_path, sr=16000, mono=True)
        # Write as WAV
        sf.write(output_path, audio, 16000)
    except Exception as e:
        # Fallback: try to just copy if it's already a compatible WAV
        import shutil
        try:
            info = sf.info(input_path)
            if info.samplerate == 16000:
                shutil.copy2(input_path, output_path)
            else:
                # Try reading and resampling with soundfile only
                audio, sr = sf.read(input_path)
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)  # Convert to mono
                # Simple resampling (not ideal but works)
                import scipy.signal
                if sr != 16000:
                    num_samples = int(len(audio) * 16000 / sr)
                    audio = scipy.signal.resample(audio, num_samples)
                sf.write(output_path, audio, 16000)
        except Exception as e2:
            raise RuntimeError(f"Could not convert {input_path}: {e}, {e2}")
    
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
            # Try loading from cache - but this dataset may require HF access approval
            cache_dir = f"{raw_dir}/cache"
            try:
                from datasets import load_dataset
                ds = load_dataset("mengct00/Mapudungun_iwslt26", cache_dir=cache_dir)
                for s in ds:
                    ds[s].save_to_disk(f"{raw_dir}/{s}")
            except Exception as e:
                print(f"  Could not download arn-spa dataset: {e}")
                print("  NOTE: This dataset may require access approval at https://huggingface.co/datasets/mengct00/Mapudungun_iwslt26")
                return

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


def preprocess_bho_hin(raw_dir, proc_dir):
    """Process Bhojpuri-Hindi dataset with separate txt files for translations."""
    repo_dir = Path(raw_dir)
    data_dir = None
    
    # Find the data directory
    for d in repo_dir.rglob("iwslt2024-2025_bho-hi"):
        data_dir = d
        break
    
    if not data_dir:
        print(f"  Could not find iwslt2024-2025_bho-hi directory")
        return
    
    for split_name in ["train", "dev", "test-2024", "test-2025"]:
        split_dir = data_dir / split_name
        if not split_dir.exists():
            continue
        
        out_split = "test" if "test" in split_name else split_name
        wav_dir = Path(proc_dir) / "wavs" / out_split
        os.makedirs(wav_dir, exist_ok=True)
        
        # Read the TSV (no header: path, offset, duration)
        tsv_file = split_dir / "stamped.tsv"
        if not tsv_file.exists():
            continue
        
        # Read Hindi translations
        hi_file = split_dir / "txt" / f"{split_name}.hi"
        hi_lines = []
        if hi_file.exists():
            with open(hi_file, "r", encoding="utf-8") as f:
                hi_lines = [line.strip() for line in f]
        
        entries = []
        with open(tsv_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                
                audio_rel_path = parts[0]  # e.g., wav/filename.wav
                audio_path = split_dir / audio_rel_path
                
                if not audio_path.exists():
                    continue
                
                # Get Hindi translation if available
                tgt_text = hi_lines[i] if i < len(hi_lines) else ""
                # For Bhojpuri, we don't have transcription - use empty or placeholder
                src_text = tgt_text  # Use Hindi as both for now (will be ASR target)
                
                duration = float(parts[2]) if len(parts) > 2 else 0
                
                # Copy/link audio to output dir
                out_wav = wav_dir / f"{i:06d}.wav"
                if not out_wav.exists():
                    try:
                        convert_to_wav_16k(str(audio_path), str(out_wav))
                    except Exception:
                        continue
                
                if tgt_text:
                    entries.append({
                        "audio_path": str(out_wav),
                        "src_text": src_text,
                        "tgt_text": tgt_text,
                        "duration": round(duration, 2),
                    })
        
        if entries:
            write_manifest(entries, f"{proc_dir}/{out_split}.jsonl")
            print(f"    {split_name}: {len(entries)} entries")


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

        # Common column name patterns - detect once outside the loop
        audio_col = next((h for h in headers if h.lower() in
                        ["audio", "audio_path", "path", "wav", "file", "audio_file", "audio_id"]), None)
        src_col = next((h for h in headers if h.lower() in
                      ["transcription", "src_text", "source", "transcript", "src", "sentence",
                       "bem_transcription", "text", "que"]), None)
        tgt_col = next((h for h in headers if h.lower() in
                      ["translation", "tgt_text", "target", "tgt", "en", "eng", "english",
                       "spanish", "spa", "hindi", "hin", "en_translation"]), None)

        if not (audio_col and (src_col or tgt_col)):
            print(f"  Could not find required columns. audio={audio_col}, src={src_col}, tgt={tgt_col}")
            return entries

        print(f"  Using columns: audio={audio_col}, src={src_col}, tgt={tgt_col}")

        for row in reader:

            audio_path = row.get(audio_col, "")
            if not os.path.isabs(audio_path):
                # Try various locations for the audio file
                found = False
                search_bases = [
                    repo_dir,
                    tsv_path.parent,
                    tsv_path.parent.parent,  # Go up one level
                ]
                # Also search in common audio subdirectories
                for base in search_bases:
                    for subdir in ["", "audio", "wavs", "wav", "../audio"]:
                        candidate = os.path.join(base, subdir, audio_path)
                        if os.path.exists(candidate):
                            audio_path = candidate
                            found = True
                            break
                    if found:
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


def preprocess_hf_african_celtic(pair_id, raw_dir, proc_dir, lang_filter):
    """Process McGill-NLP/african_celtic_dataset for ibo/hau/yor."""
    from datasets import load_from_disk, load_dataset

    print(f"  Loading african_celtic_dataset for language: {lang_filter}")
    
    # Try to load from disk first (saved by download_data.py)
    ds = None
    for split in ["train", "validation", "test"]:
        split_dir = f"{raw_dir}/{split}"
        if os.path.exists(split_dir):
            try:
                if ds is None:
                    ds = {}
                ds[split] = load_from_disk(split_dir)
                print(f"    Loaded {split} from disk: {len(ds[split])} samples")
            except Exception as e:
                print(f"    Could not load {split} from disk: {e}")
    
    # If not on disk, download directly
    if ds is None or len(ds) == 0:
        print(f"  Downloading dataset directly...")
        ds = load_dataset("McGill-NLP/african_celtic_dataset", cache_dir=f"{raw_dir}/cache")
        # Filter by language
        ds = ds.filter(lambda x: x.get("language", "").lower() == lang_filter.lower())

    for split_name, split_data in ds.items():
        out_split = "dev" if split_name == "validation" else split_name
        wav_dir = f"{proc_dir}/wavs/{out_split}"
        os.makedirs(wav_dir, exist_ok=True)

        entries = []
        for i, sample in enumerate(split_data):
            audio = sample.get("audio", {})
            audio_array = audio.get("array") if isinstance(audio, dict) else None
            sr = audio.get("sampling_rate", 48000) if isinstance(audio, dict) else 48000

            wav_path = f"{wav_dir}/{i:06d}.wav"
            if audio_array is not None and not os.path.exists(wav_path):
                import numpy as np
                sf.write(wav_path, np.array(audio_array), sr)

            # This dataset has 'text' field for transcription, no translation
            src_text = sample.get("text", "")
            # For ASR-only datasets, we use the same text as placeholder
            tgt_text = src_text  # Will need translation model or parallel data
            duration = sample.get("duration", len(audio_array) / sr if audio_array is not None else 0)

            if src_text:
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
            elif pair_id == "bho-hin":
                preprocess_bho_hin(raw_dir, proc_dir)
            elif pair_id in ("ibo-eng", "hau-eng", "yor-eng"):
                lang_filter = cfg.get("hf_lang_filter", pair_id.split("-")[0])
                preprocess_hf_african_celtic(pair_id, raw_dir, proc_dir, lang_filter)
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
