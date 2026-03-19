#!/usr/bin/env python3
"""Download and organize all 10 language pair datasets.

Handles HuggingFace datasets, GitHub repos, and direct downloads.
Outputs standardized JSONL manifests to /workspace/data/processed/{pair}/
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from datasets import load_dataset
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


def run_cmd(cmd, cwd=None):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
    return result


def download_hf_dataset(pair_id, cfg):
    """Download from HuggingFace using datasets library."""
    out_dir = f"{RAW_DIR}/{pair_id}"
    cache_dir = f"{out_dir}/cache"
    os.makedirs(out_dir, exist_ok=True)

    dataset_name = cfg["hf_dataset"]
    subset = cfg.get("hf_subset")
    lang_filter = cfg.get("hf_lang_filter")

    print(f"  Downloading HF dataset: {dataset_name} (subset={subset}, lang_filter={lang_filter})")

    try:
        if subset:
            ds = load_dataset(dataset_name, subset, cache_dir=cache_dir)
        else:
            ds = load_dataset(dataset_name, cache_dir=cache_dir)
        
        # Filter by language if specified
        if lang_filter:
            print(f"  Filtering by language: {lang_filter}")
            ds = ds.filter(lambda x: x.get("language", "").lower() == lang_filter.lower())
        
        print(f"  Dataset loaded. Splits: {list(ds.keys())}")

        # Save info
        with open(f"{out_dir}/info.json", "w") as f:
            json.dump({"splits": list(ds.keys()), "num_rows": {k: len(v) for k, v in ds.items()}}, f)

        # Save each split
        for split_name, split_data in ds.items():
            split_dir = f"{out_dir}/{split_name}"
            os.makedirs(split_dir, exist_ok=True)
            split_data.save_to_disk(split_dir)
            print(f"    Saved {split_name}: {len(split_data)} examples")
    except Exception as e:
        print(f"  ERROR downloading HF dataset: {e}")
    
    return out_dir


def download_github_repo(pair_id, cfg):
    """Clone a GitHub repository."""
    out_dir = f"{RAW_DIR}/{pair_id}"
    os.makedirs(out_dir, exist_ok=True)

    url = cfg["github_url"]
    repo_name = url.rstrip("/").split("/")[-1]
    clone_dir = f"{out_dir}/{repo_name}"

    if os.path.exists(clone_dir):
        print(f"  Already cloned: {clone_dir}")
        return clone_dir

    print(f"  Cloning: {url}")
    run_cmd(f"git clone --depth 1 {url} {clone_dir}")
    return clone_dir


def download_ckb_data(pair_id, cfg):
    """Download Central Kurdish data from LIUM."""
    out_dir = f"{RAW_DIR}/{pair_id}"
    os.makedirs(out_dir, exist_ok=True)
    print("  NOTE: CKB data requires manual download from LIUM website.")
    print(f"  URL: {cfg.get('download_url', 'https://lium.univ-lemans.fr/en/corpus-commute-kurdish/')}")
    print(f"  Please download and extract to: {out_dir}/")
    return out_dir


def download_pair(pair_id, cfg):
    """Download data for a single language pair."""
    source = cfg["data_source"]
    print(f"\n{'='*60}")
    print(f"Downloading: {pair_id} ({cfg['name']})")
    print(f"  Source: {source}, Est. hours: {cfg.get('data_hours', '?')}")

    if source == "huggingface":
        return download_hf_dataset(pair_id, cfg)
    elif source == "github":
        return download_github_repo(pair_id, cfg)
    elif source == "web":
        return download_ckb_data(pair_id, cfg)
    else:
        print(f"  Unknown source: {source}")
        return None


def discover_audio_files(data_dir):
    """Recursively find all audio files in a directory."""
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                audio_files.append(os.path.join(root, f))
    return sorted(audio_files)


def discover_text_files(data_dir):
    """Find TSV, CSV, or JSON files that might contain transcriptions/translations."""
    extensions = {".tsv", ".csv", ".json", ".jsonl", ".txt"}
    text_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                text_files.append(os.path.join(root, f))
    return sorted(text_files)


def create_manifest_from_discovered(pair_id, data_dir):
    """Attempt to create a manifest by discovering the data format.

    This is a best-effort function. Each dataset has a different format,
    so the preprocess.py script handles the per-pair specifics.
    """
    proc_dir = f"{PROCESSED_DIR}/{pair_id}"
    os.makedirs(proc_dir, exist_ok=True)

    audio_files = discover_audio_files(data_dir)
    text_files = discover_text_files(data_dir)

    info = {
        "pair_id": pair_id,
        "data_dir": data_dir,
        "num_audio_files": len(audio_files),
        "text_files": text_files[:20],
        "sample_audio": audio_files[:5],
    }

    info_path = f"{proc_dir}/discovery_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Found {len(audio_files)} audio files, {len(text_files)} text files")
    print(f"  Discovery info saved to: {info_path}")
    return info


def main():
    parser = argparse.ArgumentParser(description="Download IWSLT 2026 LR-ST data")
    parser.add_argument("--config", default="configs/language_pairs.yaml")
    parser.add_argument("--pairs", nargs="*", help="Specific pairs to download (default: all)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    pairs = config["language_pairs"]
    if args.pairs:
        pairs = {k: v for k, v in pairs.items() if k in args.pairs}

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    results = {}
    for pair_id, cfg in pairs.items():
        try:
            data_dir = download_pair(pair_id, cfg)
            if data_dir:
                info = create_manifest_from_discovered(pair_id, data_dir)
                results[pair_id] = {"status": "ok", "dir": data_dir, **info}
            else:
                results[pair_id] = {"status": "skipped"}
        except Exception as e:
            print(f"  ERROR: {e}")
            results[pair_id] = {"status": "error", "error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary:")
    for pair_id, r in results.items():
        status = r["status"]
        extra = f" ({r.get('num_audio_files', '?')} audio files)" if status == "ok" else ""
        print(f"  {pair_id}: {status}{extra}")

    summary_path = f"{RAW_DIR}/download_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
