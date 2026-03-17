#!/usr/bin/env python3
"""Prepare submission files in IWSLT format.

Format: [team_name].[task].[type].[label].[language-pair].txt
Example: myteam.st.unconstrained.primary.bem-eng.txt
"""

import argparse
import os
import shutil
import sys

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team_name", required=True)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--lang_config", default="configs/language_pairs.yaml")
    parser.add_argument("--output_dir", default="/workspace/outputs/submissions")
    parser.add_argument("--pairs", nargs="*", help="Specific pairs (default: all)")
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)
    with open(args.lang_config) as f:
        lang_cfg = yaml.safe_load(f)

    pairs = lang_cfg["language_pairs"]
    if args.pairs:
        pairs = {k: v for k, v in pairs.items() if k in args.pairs}

    os.makedirs(args.output_dir, exist_ok=True)
    submitted = []

    for pair_id, cfg in pairs.items():
        ensemble_dir = f"{train_cfg['paths']['output_root']}/ensemble/{pair_id}"

        # Primary: MBR ensemble
        mbr_file = f"{ensemble_dir}/mbr_test.txt"
        if os.path.exists(mbr_file):
            dst = f"{args.output_dir}/{args.team_name}.st.unconstrained.primary.{pair_id}.txt"
            shutil.copy2(mbr_file, dst)
            submitted.append(dst)
            print(f"  Primary (MBR): {dst}")

        # Contrastive 1: Cascade 1-best
        cascade_file = f"{ensemble_dir}/cascade_1best_test.txt"
        if os.path.exists(cascade_file):
            dst = f"{args.output_dir}/{args.team_name}.st.unconstrained.contrastive1.{pair_id}.txt"
            shutil.copy2(cascade_file, dst)
            submitted.append(dst)
            print(f"  Contrastive1 (cascade): {dst}")

        # Contrastive 2: E2E 1-best
        e2e_file = f"{ensemble_dir}/e2e_1best_test.txt"
        if os.path.exists(e2e_file):
            dst = f"{args.output_dir}/{args.team_name}.st.unconstrained.contrastive2.{pair_id}.txt"
            shutil.copy2(e2e_file, dst)
            submitted.append(dst)
            print(f"  Contrastive2 (E2E): {dst}")

        # ASR output (optional)
        asr_file = f"{ensemble_dir}/asr_test.txt"
        if os.path.exists(asr_file):
            dst = f"{args.output_dir}/{args.team_name}.asr.unconstrained.primary.{pair_id}.txt"
            shutil.copy2(asr_file, dst)
            submitted.append(dst)

    print(f"\n{'='*60}")
    print(f"Submission files prepared: {len(submitted)} files in {args.output_dir}")
    print(f"\nTo submit, email all files to: iwslt.2026.lowres.submissions@gmail.com")
    print("Include a brief system description in the email body.")


if __name__ == "__main__":
    main()
