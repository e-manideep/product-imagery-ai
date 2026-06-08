#!/bin/bash
set -e

echo "================================================="
echo " Product Imagery - FLUX.2 [klein] 9B"
echo " Setup"
echo "================================================="

export FLUX2_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "FLUX2 dir: $FLUX2_DIR"

# System packages
apt-get update -qq
apt-get install -y -qq git wget curl libglib2.0-0 \
    libgl1-mesa-glx 2>/dev/null || apt-get install -y -qq libgl1 2>/dev/null || true

# ── torch version gate ───────────────────────────────────────────────────────
# FLUX.2 needs the latest transformers, which references torch.float8_e8m0fnu
# (added in torch 2.7). On older torch the import crashes. FP8 training also
# needs a GPU with compute capability >= 8.9 (H100/H200/Blackwell/RTX 4090+).
python - <<'PY'
import torch
v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
print(f"Detected torch {torch.__version__}")
if v < (2, 7):
    print("\n  ******************************************************************")
    print("  WARNING: torch < 2.7 detected. FLUX.2 will likely FAIL to import.")
    print("  Launch a RunPod 'PyTorch 2.8' (CUDA 12.4+) pod instead, then re-run.")
    print("  ******************************************************************\n")
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print(f"GPU compute capability: {cc[0]}.{cc[1]}")
    if cc < (8, 9):
        print("  NOTE: FP8 training (--do_fp8_training) needs cc>=8.9.")
        print("  Set FP8=0 in run.sh to fall back to NF4 4-bit (bitsandbytes).")
PY

# ── dependencies (latest, FLUX.2 is bleeding edge) ───────────────────────────
echo "Installing diffusers from source..."
pip install git+https://github.com/huggingface/diffusers.git --quiet

echo "Installing supporting libraries..."
pip install -U \
    transformers \
    accelerate \
    peft \
    datasets \
    safetensors \
    sentencepiece \
    pillow \
    openai \
    tensorboard \
    prodigyopt \
    bitsandbytes \
    torchao \
    --quiet

# ── training script + flux requirements from diffusers main ──────────────────
echo "Downloading FLUX.2 [klein] training script..."
wget -q "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux2_klein.py" \
    -O "$FLUX2_DIR/train_dreambooth_lora_flux2_klein.py"
wget -q "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/requirements_flux.txt" \
    -O "$FLUX2_DIR/requirements_flux.txt" 2>/dev/null || true
[ -f "$FLUX2_DIR/requirements_flux.txt" ] && pip install -r "$FLUX2_DIR/requirements_flux.txt" --quiet || true

# Non-interactive accelerate config (single GPU)
echo "Configuring accelerate..."
accelerate config default 2>/dev/null || true

# HuggingFace login (FLUX.2-klein-9B is GATED)
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to HuggingFace..."
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
else
    echo ""
    echo "  WARNING: HF_TOKEN not set. FLUX.2-klein-9B is a gated model."
    echo "  export HF_TOKEN=hf_... before running."
    echo ""
fi

mkdir -p "$FLUX2_DIR/output"

echo ""
echo "================================================="
echo " Setup complete."
echo " 1. Accept the license (once) at:"
echo "      https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
echo " 2. export HF_TOKEN=hf_...   (if not already)"
echo " 3. bash run.sh"
echo "================================================="
