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
# diffusers==0.32.0: has FLUX support, compatible with PyTorch 2.4.0
# (0.33.0+ adds WanTransformer3DModel but breaks with PyTorch 2.4 flash_attn_3)
echo "Installing pipeline dependencies..."
pip install \
    "transformers>=4.48.0" \
    accelerate \
    "diffusers==0.32.0" \
    sentencepiece \
    protobuf \
    pillow \
    pyyaml \
    bitsandbytes \
    tqdm \
    --quiet

# Patch ai-toolkit: make WanTransformer3DModel optional (added in diffusers 0.33+,
# not needed for FLUX LoRA training, incompatible with diffusers 0.32 + PyTorch 2.4)
echo "Patching ai-toolkit for diffusers 0.32 compatibility..."
LORA_SPECIAL="$BASE_DIR/ai-toolkit/toolkit/lora_special.py"
python3 - "$LORA_SPECIAL" << 'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    c = f.read()
old = 'from diffusers import UNet2DConditionModel, PixArtTransformer2DModel, AuraFlowTransformer2DModel, WanTransformer3DModel'
new = ('from diffusers import UNet2DConditionModel, PixArtTransformer2DModel, AuraFlowTransformer2DModel\n'
       'try:\n    from diffusers import WanTransformer3DModel\nexcept ImportError:\n    WanTransformer3DModel = None')
if old in c:
    with open(path, 'w') as f:
        f.write(c.replace(old, new))
    print('  lora_special.py patched.')
else:
    print('  lora_special.py already patched.')
PYEOF

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
