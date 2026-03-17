#!/usr/bin/env python3
"""MBR decoding: combine N-best from cascade and E2E systems.

Loads N-best hypotheses from both pipelines, combines them,
and selects the best hypothesis per utterance using MBR with chrF.
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.mbr import mbr_decode_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--scoring_metric", default="chrf", choices=["chrf", "bleu"])
    args = parser.parse_args()

    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)

    pair_id = args.pair
    input_dir = args.input_dir or f"{train_cfg['paths']['output_root']}/ensemble/{pair_id}"
    output_dir = args.output_dir or input_dir
    split = args.split

    print(f"{'='*60}")
    print(f"MBR Decoding: {pair_id}")

    # Load N-best from cascade
    cascade_nbest_path = f"{input_dir}/cascade_nbest_{split}.json"
    e2e_nbest_path = f"{input_dir}/e2e_nbest_{split}.json"

    cascade_nbest = []
    e2e_nbest = []

    if os.path.exists(cascade_nbest_path):
        with open(cascade_nbest_path) as f:
            cascade_nbest = json.load(f)
        print(f"  Loaded cascade N-best: {len(cascade_nbest)} utterances")

    if os.path.exists(e2e_nbest_path):
        with open(e2e_nbest_path) as f:
            e2e_nbest = json.load(f)
        print(f"  Loaded E2E N-best: {len(e2e_nbest)} utterances")

    # Also check for Whisper direct ST N-best
    whisper_st_path = f"{input_dir}/whisper_st_nbest_{split}.json"
    whisper_st_nbest = []
    if os.path.exists(whisper_st_path):
        with open(whisper_st_path) as f:
            whisper_st_nbest = json.load(f)
        print(f"  Loaded Whisper ST N-best: {len(whisper_st_nbest)} utterances")

    # Determine number of utterances
    num_utts = max(len(cascade_nbest), len(e2e_nbest), len(whisper_st_nbest))
    if num_utts == 0:
        print("  ERROR: No N-best hypotheses found!")
        sys.exit(1)

    # Combine hypotheses
    combined_nbest = []
    for i in range(num_utts):
        hyps = []
        if i < len(cascade_nbest):
            hyps.extend(cascade_nbest[i])
        if i < len(e2e_nbest):
            hyps.extend(e2e_nbest[i])
        if i < len(whisper_st_nbest):
            hyps.extend(whisper_st_nbest[i])
        combined_nbest.append(hyps)

    print(f"  Combined: {num_utts} utterances, "
          f"avg {sum(len(h) for h in combined_nbest)/num_utts:.1f} hypotheses each")

    # MBR decode
    print(f"\n  Running MBR with {args.scoring_metric}...")
    results = mbr_decode_batch(combined_nbest, scoring_metric=args.scoring_metric)

    # Save MBR output
    output_path = f"{output_dir}/mbr_{split}.txt"
    with open(output_path, "w") as f:
        for t in results:
            f.write(t.strip() + "\n")

    print(f"\n  MBR output saved to: {output_path}")
    print(f"  {len(results)} translations generated")

    # Also save individual 1-best fallbacks for comparison
    fallback_sources = {
        "cascade": f"{input_dir}/cascade_1best_{split}.txt",
        "e2e": f"{input_dir}/e2e_1best_{split}.txt",
    }
    for name, path in fallback_sources.items():
        if os.path.exists(path):
            print(f"  Fallback available: {name} -> {path}")


if __name__ == "__main__":
    main()
