#!/usr/bin/env python3
"""Evaluate ST outputs using BLEU, chrF++, and COMET.

Follows IWSLT evaluation protocol: lowercase, no punctuation.
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.metrics import evaluate_all, normalize_text
from src.data.dataset import STManifest


def load_hypotheses(path):
    with open(path, "r") as f:
        return [line.strip() for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--lang_config", default="configs/language_pairs.yaml")
    parser.add_argument("--hyp_file", default=None, help="Hypothesis file to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all available outputs")
    parser.add_argument("--no_comet", action="store_true", help="Skip COMET (faster)")
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)

    pair_id = args.pair
    data_dir = f"{train_cfg['paths']['data_root']}/processed/{pair_id}"
    ensemble_dir = f"{train_cfg['paths']['output_root']}/ensemble/{pair_id}"

    # Load references
    manifest_path = f"{data_dir}/{args.split}.jsonl"
    if not os.path.exists(manifest_path):
        print(f"ERROR: Reference manifest not found: {manifest_path}")
        sys.exit(1)

    entries = STManifest.load(manifest_path)
    references = [e["tgt_text"] for e in entries]
    sources = [e.get("src_text", "") for e in entries]

    print(f"{'='*60}")
    print(f"Evaluation: {pair_id} ({args.split})")
    print(f"  References: {len(references)} utterances")

    # Determine which files to evaluate
    hyp_files = {}
    if args.hyp_file:
        hyp_files["custom"] = args.hyp_file
    elif args.all:
        # Find all available output files
        candidates = [
            ("cascade_1best", f"{ensemble_dir}/cascade_1best_{args.split}.txt"),
            ("e2e_1best", f"{ensemble_dir}/e2e_1best_{args.split}.txt"),
            ("mbr", f"{ensemble_dir}/mbr_{args.split}.txt"),
        ]
        for name, path in candidates:
            if os.path.exists(path):
                hyp_files[name] = path
    else:
        # Default: evaluate MBR output
        mbr_path = f"{ensemble_dir}/mbr_{args.split}.txt"
        if os.path.exists(mbr_path):
            hyp_files["mbr"] = mbr_path
        else:
            print("No MBR output found. Use --hyp_file or --all.")
            sys.exit(1)

    # Evaluate each
    all_results = {}
    for name, path in hyp_files.items():
        print(f"\n--- {name}: {path} ---")
        hypotheses = load_hypotheses(path)

        if len(hypotheses) != len(references):
            print(f"  WARNING: {len(hypotheses)} hypotheses vs {len(references)} references")
            min_len = min(len(hypotheses), len(references))
            hypotheses = hypotheses[:min_len]
            refs = references[:min_len]
            srcs = sources[:min_len]
        else:
            refs = references
            srcs = sources

        results = evaluate_all(
            hypotheses, refs, srcs,
            compute_comet_score=not args.no_comet,
        )

        all_results[name] = results
        print(f"  BLEU:  {results['bleu']:.2f}")
        print(f"  chrF++: {results['chrf++']:.2f}")
        if "comet" in results:
            print(f"  COMET: {results['comet']:.4f}")

    # Save results
    results_path = f"{ensemble_dir}/eval_results_{args.split}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'System':<20} {'BLEU':>8} {'chrF++':>8} {'COMET':>8}")
        print("-" * 50)
        for name, r in all_results.items():
            comet = f"{r['comet']:.4f}" if "comet" in r else "N/A"
            print(f"{name:<20} {r['bleu']:>8.2f} {r['chrf++']:>8.2f} {comet:>8}")


if __name__ == "__main__":
    main()
