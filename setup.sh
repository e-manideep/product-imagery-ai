#!/bin/bash
set -e

echo "================================================="
echo " Product Imagery AI - Setup"
echo "================================================="

export BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Base directory: $BASE_DIR"

# System packages
apt-get update -qq
apt-get install -y -qq git wget curl libgl1-mesa-glx libglib2.0-0

# Install all dependencies
# diffusers==0.32.0: supports FLUX, compatible with PyTorch 2.4.0
echo "Installing dependencies..."
pip install \
    "diffusers==0.32.0" \
    "transformers>=4.44.0" \
    "peft>=0.11.0" \
    accelerate \
    sentencepiece \
    protobuf \
    pillow \
    pyyaml \
    bitsandbytes \
    tqdm \
    prodigyopt \
    --quiet

# Download the official HuggingFace DreamBooth FLUX training script
echo "Downloading DreamBooth FLUX training script..."
TRAIN_SCRIPT="$BASE_DIR/train_dreambooth_lora_flux.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    wget -q "https://raw.githubusercontent.com/huggingface/diffusers/v0.32.0/examples/dreambooth/train_dreambooth_lora_flux.py" \
        -O "$TRAIN_SCRIPT"
    echo "  Downloaded train_dreambooth_lora_flux.py"
else
    echo "  Training script already exists."
fi

# Configure accelerate (non-interactive, single GPU)
echo "Configuring accelerate..."
accelerate config default --mixed_precision bf16 2>/dev/null || \
accelerate config default 2>/dev/null || true

# HuggingFace login
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo ""
    echo "  WARNING: HF_TOKEN not set."
    echo "  Export it before running: export HF_TOKEN=your_token"
    echo ""
fi

# Create directories
mkdir -p "$BASE_DIR/data/product"
mkdir -p "$BASE_DIR/output/product_lora"
mkdir -p "$BASE_DIR/output/inference"

echo ""
echo "================================================="
echo " Setup complete!"
echo " Next: export HF_TOKEN=... && bash run_all.sh"
echo "================================================="
