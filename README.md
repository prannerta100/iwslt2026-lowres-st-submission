# IWSLT 2026 Low-Resource Speech Translation Submission

Dual-pipeline system for the IWSLT 2026 Low-Resource ST shared task, targeting all 10 language pairs with a unified approach that combines cascaded and end-to-end speech translation with MBR ensemble decoding.

## Architecture

```
                    ┌──────────────────────────────┐
                    │        Source Audio           │
                    └──────────┬───────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    ┌─────────────────────┐       ┌─────────────────────────┐
    │  Pipeline A: Cascade │       │  Pipeline B: End-to-End  │
    │                     │       │                         │
    │  Whisper large-v3   │       │  SeamlessM4T v2 large   │
    │  (LoRA, MTL ASR+ST) │       │  (LoRA fine-tuned)      │
    │         │           │       │                         │
    │         ▼           │       │    Direct ST output     │
    │  NLLB-200 1.3B     │       │    + N-best hypotheses  │
    │  (intra-distill.)   │       │                         │
    │         │           │       └────────────┬────────────┘
    │    N-best hyps      │                    │
    └─────────┬───────────┘                    │
              │                                │
              └────────────┬───────────────────┘
                           ▼
              ┌─────────────────────────┐
              │  MBR Decoding (chrF)    │
              │  50+50 hypotheses       │
              │  Select best per utt.   │
              └────────────┬────────────┘
                           ▼
                    Final Translation
```

## Language Pairs

| Pair | Direction | Data | Pipeline Notes |
|------|-----------|------|----------------|
| arn-spa | Mapuzugun → Spanish | ~130h | E2E primary (not in NLLB) |
| bem-eng | Bemba → English | ~180h | Cascade + E2E |
| bho-hin | Bhojpuri → Hindi | ~26h | Pseudo-translation trick |
| ckb-eng | C. Kurdish → English | ~30h | Cascade + E2E |
| gle-eng | Irish → English | ~25h | Cascade + E2E |
| ibo-eng | Igbo → English | ~20h | Cascade + E2E |
| hau-eng | Hausa → English | ~20h | Cascade + E2E |
| que-spa | Quechua → Spanish | ~58h | Cascade + E2E |
| ca-en | Catalan → English | ~30h | Whisper direct ST available |
| yor-eng | Yoruba → English | ~20h | Cascade + E2E |

## Key Techniques

- **Whisper large-v3 + LoRA (rank 128)**: Multi-task ASR+ST, SpecAugment, speed perturbation
- **SeamlessM4T v2 large + LoRA (rank 64)**: End-to-end ST for all pairs
- **NLLB-200 1.3B + intra-distillation**: Regularized MT fine-tuning
- **MBR decoding**: Combine N-best from both pipelines, scored by chrF
- **Audio augmentation**: Speed perturbation (0.9/1.0/1.1), noise injection, SpecAugment
- **Pseudo-translation**: For bho-hin (Bhojpuri~Hindi proximity)

## Quick Start

### 1. Setup (on A100 machine)

```bash
git clone <this-repo> && cd iwslt2026-lowres-st-submission
bash setup.sh          # Install deps, download pretrained models (~15GB)
```

### 2. Run Full Pipeline

```bash
# All 10 pairs (takes ~3-4 days on 1x A100)
bash run_all.sh all

# Or specific pairs
bash run_all.sh "bem-eng que-spa bho-hin"

# Set team name
TEAM_NAME=myteam bash run_all.sh all
```

### 3. Run Individual Steps

```bash
# Download data
python scripts/download_data.py --pairs bem-eng que-spa

# Preprocess
python scripts/preprocess.py --pairs bem-eng

# Train Whisper
python scripts/train_whisper.py --pair bem-eng

# Train SeamlessM4T
python scripts/train_seamless.py --pair bem-eng

# Train NLLB
python scripts/train_nllb.py --pair bem-eng

# Inference
python scripts/inference_cascade.py --pair bem-eng --split test
python scripts/inference_e2e.py --pair bem-eng --split test

# MBR ensemble
python scripts/mbr_decode.py --pair bem-eng --split test

# Evaluate
python scripts/evaluate.py --pair bem-eng --split dev --all

# Prepare submission
python scripts/prepare_submission.py --team_name myteam
```

## Resource Requirements

| Resource | Requirement |
|----------|-------------|
| GPU | 1x A100 40GB (or 80GB for larger batches) |
| Disk | 100GB minimum |
| RAM | 32GB+ |
| Time | ~3-4 days for all 10 pairs |

### Disk Budget

| Component | Size |
|-----------|------|
| Pretrained models (Whisper + SeamlessM4T + NLLB) | ~15GB |
| Audio data (all 10 pairs) | ~50GB |
| LoRA checkpoints | ~5GB |
| Working space + outputs | ~25GB |

### GPU Memory (per phase)

| Phase | VRAM |
|-------|------|
| Whisper LoRA training | ~22GB |
| SeamlessM4T LoRA training | ~28GB |
| NLLB training | ~8GB |
| Inference (any model) | ~20GB |

## Project Structure

```
├── configs/
│   ├── language_pairs.yaml    # All 10 pairs: data sources, lang codes
│   └── training.yaml          # Hyperparameters for all models
├── scripts/
│   ├── download_data.py       # Download from HF, GitHub, LIUM
│   ├── preprocess.py          # Standardize to JSONL manifests
│   ├── train_whisper.py       # Whisper LoRA MTL fine-tuning
│   ├── train_seamless.py      # SeamlessM4T LoRA fine-tuning
│   ├── train_nllb.py          # NLLB with intra-distillation
│   ├── inference_cascade.py   # Whisper ASR → NLLB MT (N-best)
│   ├── inference_e2e.py       # SeamlessM4T direct ST (N-best)
│   ├── mbr_decode.py          # MBR ensemble decoding
│   ├── evaluate.py            # BLEU, chrF++, COMET
│   └── prepare_submission.py  # Format for IWSLT submission
├── src/
│   ├── data/
│   │   ├── dataset.py         # PyTorch datasets (Whisper, Seamless, NLLB)
│   │   └── augmentation.py    # SpecAugment, speed perturb, noise
│   ├── utils/
│   │   ├── metrics.py         # BLEU, chrF++, COMET evaluation
│   │   └── mbr.py             # MBR decoding algorithm
├── setup.sh                   # One-click environment setup
├── run_all.sh                 # Master orchestration script
└── requirements.txt
```

## Submission Format

Files are named: `[team].[task].[condition].[label].[pair].txt`

- **Primary**: MBR ensemble (cascade + E2E combined)
- **Contrastive1**: Cascade 1-best (Whisper ASR → NLLB MT)
- **Contrastive2**: E2E 1-best (SeamlessM4T direct)

Submit to: iwslt.2026.lowres.submissions@gmail.com

## Approach vs. Prior IWSLT Winners

| Technique | JHU 2024 | KIT 2024 | GMU 2025 | **Ours** |
|-----------|----------|----------|----------|----------|
| Whisper version | v2 | v2 | - | **v3** |
| SeamlessM4T | v2 | v2 | v2 | **v2** |
| NLLB size | 600M | 1B | - | **1.3B** |
| LoRA rank | 200 | 8 | - | **128** |
| Multi-task ASR+ST | Yes | No | No | **Yes** |
| Intra-distillation | Yes (NLLB) | No | No | **Yes** |
| MBR ensemble | No | Yes | No | **Yes** |
| Systems combined | Pick best | Cascade+E2E | E2E only | **Both+MBR** |
| Pseudo-translation | bho/mar only | No | No | **bho-hin** |
| Speed perturbation | Yes | No | No | **Yes** |

## Evaluation

Official metrics (lowercased, no punctuation):
- **BLEU** (sacreBLEU)
- **chrF++** (sacreBLEU, word_order=2)
- **COMET** (wmt22-comet-da)
