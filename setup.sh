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

# Clone or update ai-toolkit
if [ ! -d "$BASE_DIR/ai-toolkit" ]; then
    echo "Cloning ai-toolkit..."
    git clone https://github.com/ostris/ai-toolkit.git "$BASE_DIR/ai-toolkit"
else
    echo "Updating ai-toolkit..."
    cd "$BASE_DIR/ai-toolkit" && git pull && cd "$BASE_DIR"
fi

# Install ai-toolkit requirements
echo "Installing ai-toolkit dependencies..."
pip install -r "$BASE_DIR/ai-toolkit/requirements.txt" --quiet

# Install pipeline dependencies
echo "Installing pipeline dependencies..."
pip install \
    "transformers>=4.48.0" \
    accelerate \
    "diffusers>=0.30.0" \
    sentencepiece \
    protobuf \
    pillow \
    pyyaml \
    bitsandbytes \
    tqdm \
    --quiet

# Clear HF modules cache to avoid stale Florence-2 dynamic module conflicts
rm -rf ~/.cache/huggingface/modules/

# HuggingFace login
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo ""
    echo "  WARNING: HF_TOKEN not set."
    echo "  Run this before training:"
    echo "    export HF_TOKEN=your_token"
    echo "    huggingface-cli login --token \$HF_TOKEN"
    echo ""
fi

# Create required directories
mkdir -p "$BASE_DIR/data/product"
mkdir -p "$BASE_DIR/output"
mkdir -p "$BASE_DIR/output/inference"

echo ""
echo "================================================="
echo " Setup complete!"
echo " Next steps:"
echo "   1. Upload your images to: $BASE_DIR/data/product/"
echo "   2. Set HF_TOKEN and run: bash run_all.sh"
echo "================================================="
