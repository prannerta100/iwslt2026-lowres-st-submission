#!/bin/bash
set -euo pipefail

# ============================================================
# IWSLT 2026 Low-Resource ST — Full Pipeline Orchestration
# ============================================================
# Run on: 1x A100 40GB, 100GB disk
# Estimated total time: ~3-4 days
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# All 10 language pairs
ALL_PAIRS="arn-spa bem-eng bho-hin ckb-eng gle-eng ibo-eng hau-eng que-spa ca-en yor-eng"

# Override: set specific pairs to run (or "all" for everything)
PAIRS="${1:-all}"
if [ "$PAIRS" = "all" ]; then
    PAIRS="$ALL_PAIRS"
fi

TEAM_NAME="${TEAM_NAME:-iwslt2026}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="/workspace/outputs/logs/$TIMESTAMP"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline.log"
}

run_step() {
    local step_name="$1"
    shift
    log "START: $step_name"
    local start_time=$(date +%s)

    if "$@" 2>&1 | tee -a "$LOG_DIR/${step_name}.log"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log "DONE: $step_name (${duration}s)"
    else
        log "FAILED: $step_name"
        return 1
    fi
}

# ============================================================
# Phase 0: Setup (if not already done)
# ============================================================
if [ ! -d "/workspace/models" ] || [ -z "$(ls -A /workspace/models 2>/dev/null)" ]; then
    log "=== Phase 0: Environment Setup ==="
    run_step "setup" bash setup.sh
fi

# ============================================================
# Phase 1: Data Download & Preprocessing (~4-6 hours)
# ============================================================
log "=== Phase 1: Data Download & Preprocessing ==="

run_step "download_data" python scripts/download_data.py --pairs $PAIRS
run_step "preprocess_data" python scripts/preprocess.py --pairs $PAIRS

# Verify data
log "Data verification:"
for pair in $PAIRS; do
    if [ -f "/workspace/data/processed/$pair/train.jsonl" ]; then
        count=$(wc -l < "/workspace/data/processed/$pair/train.jsonl")
        log "  $pair: $count training samples"
    else
        log "  $pair: WARNING - no training manifest found"
    fi
done

# ============================================================
# Phase 2: Whisper LoRA Fine-tuning (~15-20 hours)
# ============================================================
log "=== Phase 2: Whisper LoRA MTL Fine-tuning ==="

for pair in $PAIRS; do
    if [ -f "/workspace/data/processed/$pair/train.jsonl" ]; then
        run_step "whisper_$pair" python scripts/train_whisper.py --pair "$pair"
    else
        log "  SKIP Whisper $pair: no training data"
    fi
done

# ============================================================
# Phase 3: SeamlessM4T LoRA Fine-tuning (~15-20 hours)
# ============================================================
log "=== Phase 3: SeamlessM4T v2 LoRA Fine-tuning ==="

for pair in $PAIRS; do
    if [ -f "/workspace/data/processed/$pair/train.jsonl" ]; then
        run_step "seamless_$pair" python scripts/train_seamless.py --pair "$pair"
    else
        log "  SKIP SeamlessM4T $pair: no training data"
    fi
done

# ============================================================
# Phase 4: NLLB Fine-tuning (~5-8 hours)
# ============================================================
log "=== Phase 4: NLLB Fine-tuning with Intra-Distillation ==="

for pair in $PAIRS; do
    if [ -f "/workspace/data/processed/$pair/train.jsonl" ]; then
        run_step "nllb_$pair" python scripts/train_nllb.py --pair "$pair"
    else
        log "  SKIP NLLB $pair: no training data"
    fi
done

# ============================================================
# Phase 5: Inference — Cascade + E2E (~3-5 hours)
# ============================================================
log "=== Phase 5: Inference ==="

for pair in $PAIRS; do
    for split in dev test; do
        if [ -f "/workspace/data/processed/$pair/$split.jsonl" ]; then
            run_step "cascade_${pair}_${split}" python scripts/inference_cascade.py \
                --pair "$pair" --split "$split"

            run_step "e2e_${pair}_${split}" python scripts/inference_e2e.py \
                --pair "$pair" --split "$split"
        fi
    done
done

# ============================================================
# Phase 6: MBR Decoding (~1 hour)
# ============================================================
log "=== Phase 6: MBR Decoding ==="

for pair in $PAIRS; do
    for split in dev test; do
        ensemble_dir="/workspace/outputs/ensemble/$pair"
        if [ -f "$ensemble_dir/cascade_nbest_${split}.json" ] || \
           [ -f "$ensemble_dir/e2e_nbest_${split}.json" ]; then
            run_step "mbr_${pair}_${split}" python scripts/mbr_decode.py \
                --pair "$pair" --split "$split"
        fi
    done
done

# ============================================================
# Phase 7: Evaluation (~1 hour)
# ============================================================
log "=== Phase 7: Evaluation ==="

for pair in $PAIRS; do
    if [ -f "/workspace/data/processed/$pair/dev.jsonl" ]; then
        run_step "eval_${pair}" python scripts/evaluate.py \
            --pair "$pair" --split dev --all
    fi
done

# ============================================================
# Phase 8: Prepare Submission
# ============================================================
log "=== Phase 8: Prepare Submission ==="

run_step "prepare_submission" python scripts/prepare_submission.py \
    --team_name "$TEAM_NAME" --pairs $PAIRS

# ============================================================
# Final Summary
# ============================================================
log ""
log "============================================================"
log "PIPELINE COMPLETE"
log "============================================================"
log ""
log "Submission files: /workspace/outputs/submissions/"
log "Evaluation results: /workspace/outputs/ensemble/*/eval_results_dev.json"
log "Logs: $LOG_DIR/"
log ""

# Print eval results summary
log "=== Results Summary ==="
for pair in $PAIRS; do
    results_file="/workspace/outputs/ensemble/$pair/eval_results_dev.json"
    if [ -f "$results_file" ]; then
        log "  $pair:"
        python3 -c "
import json
with open('$results_file') as f:
    results = json.load(f)
for sys_name, metrics in results.items():
    bleu = metrics.get('bleu', 0)
    chrf = metrics.get('chrf++', 0)
    comet = metrics.get('comet', 'N/A')
    if isinstance(comet, float):
        comet = f'{comet:.4f}'
    print(f'    {sys_name:20s} BLEU={bleu:6.2f}  chrF++={chrf:6.2f}  COMET={comet}')
" 2>/dev/null || true
    fi
done

log ""
log "Disk usage:"
du -sh /workspace/data/ /workspace/outputs/ /workspace/models/ 2>/dev/null || true
df -h /workspace 2>/dev/null || true
