#!/bin/bash
set -e

echo "================================================="
echo " Product Imagery AI - Setup"
echo "================================================="

export BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Base directory: $BASE_DIR"

# System packages (handle Ubuntu 22 and 24 both)
apt-get update -qq
apt-get install -y -qq git wget curl libglib2.0-0 \
    libgl1-mesa-glx 2>/dev/null || apt-get install -y -qq libgl1 2>/dev/null || true

# Install Python dependencies
echo "Installing Python dependencies..."
pip install \
    "transformers>=4.44.0" \
    "peft>=0.11.0" \
    accelerate \
    sentencepiece \
    protobuf \
    pillow \
    pyyaml \
    bitsandbytes \
    tqdm \
    --quiet

# Install diffusers from source (required by latest dreambooth script)
echo "Installing diffusers from source..."
pip install git+https://github.com/huggingface/diffusers.git --quiet

# Download the official HuggingFace DreamBooth FLUX training script
echo "Downloading DreamBooth FLUX training script..."
wget -q "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux.py" \
    -O "$BASE_DIR/train_dreambooth_lora_flux.py"

# HuggingFace login
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to HuggingFace..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
else
    echo ""
    echo "  WARNING: HF_TOKEN not set. Export it first:"
    echo "    export HF_TOKEN=your_token"
    echo ""
fi

# Create directories
mkdir -p "$BASE_DIR/data/product"
mkdir -p "$BASE_DIR/output/product_lora"
mkdir -p "$BASE_DIR/output/inference"

echo ""
echo "================================================="
echo " Setup complete! Now run: bash run_all.sh"
echo "================================================="
