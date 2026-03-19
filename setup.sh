#!/bin/bash

# Load HF_TOKEN from bashrc if not already set
source ~/.bashrc 2>/dev/null || true

echo "=== IWSLT 2026 Low-Resource ST - Folder Setup ==="

# Create workspace directories
mkdir -p ~/workspace/{data,outputs,models}
mkdir -p ~/workspace/data/{raw,processed}
mkdir -p ~/workspace/outputs/{whisper,seamless,nllb,ensemble,submissions}

# Pre-download models to cache
echo "=== Downloading pretrained models ==="

python3 << 'EOF'
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor
print('Downloading Whisper large-v3...')
cache_dir = os.path.expanduser('~/workspace/models')
WhisperProcessor.from_pretrained('openai/whisper-large-v3', cache_dir=cache_dir)
WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v3', cache_dir=cache_dir)
print('Done.')
EOF

python3 << 'EOF'
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
print('Downloading NLLB-200 1.3B distilled...')
cache_dir = os.path.expanduser('~/workspace/models')
AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-1.3B', cache_dir=cache_dir)
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-1.3B', cache_dir=cache_dir)
print('Done.')
EOF

python3 << 'EOF'
import os
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
print('Downloading SeamlessM4T v2 large...')
cache_dir = os.path.expanduser('~/workspace/models')
AutoProcessor.from_pretrained('facebook/seamless-m4t-v2-large', cache_dir=cache_dir)
SeamlessM4Tv2ForSpeechToText.from_pretrained('facebook/seamless-m4t-v2-large', cache_dir=cache_dir)
print('Done.')
EOF

echo "=== Setup complete ==="
echo "Disk usage:"
du -sh ~/workspace/models/
df -h ~/workspace
