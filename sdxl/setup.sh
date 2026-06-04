#!/bin/bash
set -e

echo "================================================="
echo " Product Imagery - SDXL DoRA + Pivotal Tuning"
echo " Setup"
echo "================================================="

export SDXL_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "SDXL dir: $SDXL_DIR"

# System packages (Ubuntu 22 / 24 both)
apt-get update -qq
apt-get install -y -qq git wget curl libglib2.0-0 \
    libgl1-mesa-glx 2>/dev/null || apt-get install -y -qq libgl1 2>/dev/null || true

# Python deps
# - diffusers from source: the advanced training script lives in examples/ and
#   tracks the diffusers main branch
# - peft >= 0.11: required for DoRA (use_dora)
# - openai: GPT-4o vision captioning
echo "Installing Python dependencies..."
pip install \
    "transformers>=4.44.0" \
    "peft>=0.11.0" \
    accelerate \
    datasets \
    safetensors \
    sentencepiece \
    pillow \
    openai \
    tensorboard \
    prodigyopt \
    --quiet

echo "Installing diffusers from source..."
pip install git+https://github.com/huggingface/diffusers.git --quiet

# Download the advanced SDXL DreamBooth LoRA training script (self-contained)
echo "Downloading advanced SDXL training script..."
wget -q "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py" \
    -O "$SDXL_DIR/train_dreambooth_lora_sdxl_advanced.py"

# Non-interactive accelerate config (single GPU)
echo "Configuring accelerate..."
accelerate config default 2>/dev/null || true

# HuggingFace login (SDXL base is public, but login avoids rate limits)
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to HuggingFace..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
fi

# Sanity checks
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "  WARNING: OPENAI_API_KEY not set."
    echo "  Captioning needs it. Export before running:"
    echo "    export OPENAI_API_KEY=sk-..."
    echo ""
fi

mkdir -p "$SDXL_DIR/output"

echo ""
echo "================================================="
echo " Setup complete."
echo " Next: export OPENAI_API_KEY=... && bash run.sh"
echo "================================================="
