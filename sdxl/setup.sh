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

# Python deps - PINNED for torch 2.4 compatibility.
# transformers 5.x / latest diffusers reference torch.float8_e8m0fnu (torch 2.7+),
# which breaks on the torch 2.4 pods. This pinned set is known-good and supports
# DoRA (--use_dora) + pivotal tuning. The training script is pulled from the
# matching diffusers release tag so check_min_version passes.
DIFFUSERS_VERSION="0.32.0"
echo "Installing Python dependencies (pinned for torch 2.4)..."
pip install \
    "transformers==4.46.3" \
    "diffusers==${DIFFUSERS_VERSION}" \
    "peft==0.13.2" \
    accelerate \
    datasets \
    safetensors \
    sentencepiece \
    pillow \
    openai \
    tensorboard \
    prodigyopt \
    --quiet

# Download the advanced SDXL training script from the matching release tag
echo "Downloading advanced SDXL training script (v${DIFFUSERS_VERSION})..."
wget -q "https://raw.githubusercontent.com/huggingface/diffusers/v${DIFFUSERS_VERSION}/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py" \
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
