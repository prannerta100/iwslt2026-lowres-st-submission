#!/bin/bash
set -euo pipefail

echo "=== IWSLT 2026 Low-Resource ST — Environment Setup ==="

# Create workspace directories
mkdir -p /workspace/{data,outputs,models}
mkdir -p /workspace/data/{raw,processed}
mkdir -p /workspace/outputs/{whisper,seamless,nllb,ensemble,submissions}

# Install system deps
apt-get update && apt-get install -y ffmpeg sox libsndfile1 git-lfs bc

# Python environment
pip install --upgrade pip
pip install -r requirements.txt

# Pre-download models to cache
echo "=== Downloading pretrained models ==="
python -c "
from transformers import WhisperForConditionalGeneration, WhisperProcessor
print('Downloading Whisper large-v3...')
WhisperProcessor.from_pretrained('openai/whisper-large-v3', cache_dir='/workspace/models')
WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3', cache_dir='/workspace/models')
print('Done.')
"

python -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
print('Downloading NLLB-200 1.3B distilled...')
AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-1.3B', cache_dir='/workspace/models')
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-1.3B', cache_dir='/workspace/models')
print('Done.')
"

python -c "
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
print('Downloading SeamlessM4T v2 large...')
AutoProcessor.from_pretrained('facebook/seamless-m4t-v2-large', cache_dir='/workspace/models')
SeamlessM4Tv2ForSpeechToText.from_pretrained('facebook/seamless-m4t-v2-large', cache_dir='/workspace/models')
print('Done.')
"

echo "=== Setup complete ==="
echo "Disk usage:"
du -sh /workspace/models/
df -h /workspace
