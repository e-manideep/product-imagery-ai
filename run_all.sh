#!/bin/bash
# Full pipeline: caption → config → train → inference
# Override any variable by exporting it before running this script.
#
# Example:
#   export HF_TOKEN=hf_xxx
#   export TRIGGER_WORD=xtbll
#   export TRAIN_STEPS=1500   # use 1500 if you have fewer than 10 images
#   bash run_all.sh

set -e

# ── Config ──────────────────────────────────────────────────────────────────
export BASE_DIR="${BASE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
export TRIGGER_WORD="${TRIGGER_WORD:-xtbll}"
export LORA_RANK="${LORA_RANK:-16}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000}"
export RUN_NAME="${RUN_NAME:-product_lora}"
export DATA_DIR="${DATA_DIR:-$BASE_DIR/data/product}"
export LR="${LR:-1e-4}"
export QUANTIZE="${QUANTIZE:-true}"    # true for 24GB GPU (RTX 4090); set false for 80GB+
export SAMPLE_EVERY="${SAMPLE_EVERY:-250}"

echo "================================================="
echo " Product Imagery AI Pipeline"
echo "================================================="
echo "  BASE_DIR:     $BASE_DIR"
echo "  TRIGGER_WORD: $TRIGGER_WORD"
echo "  LORA_RANK:    $LORA_RANK"
echo "  TRAIN_STEPS:  $TRAIN_STEPS"
echo "  RUN_NAME:     $RUN_NAME"
echo "  DATA_DIR:     $DATA_DIR"
echo "  QUANTIZE:     $QUANTIZE"
echo ""

# ── Preflight checks ────────────────────────────────────────────────────────
if [ ! -d "$BASE_DIR/ai-toolkit" ]; then
    echo "ERROR: ai-toolkit not found. Run setup.sh first."
    exit 1
fi

IMAGE_COUNT=$(find "$DATA_DIR" \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) 2>/dev/null | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "ERROR: No images found in $DATA_DIR"
    echo "Upload your product images there first."
    exit 1
fi
echo "Found $IMAGE_COUNT product images."
echo ""

# ── Step 1: Caption ──────────────────────────────────────────────────────────
echo "================================================="
echo " [1/4] Captioning images with Florence-2..."
echo "================================================="
python "$BASE_DIR/caption.py" \
    --data_dir "$DATA_DIR" \
    --trigger_word "$TRIGGER_WORD"

echo ""

# ── Step 2: Generate training config ────────────────────────────────────────
echo "================================================="
echo " [2/4] Generating training config..."
echo "================================================="
python "$BASE_DIR/generate_config.py"
CONFIG_PATH="$BASE_DIR/configs/active_config.yaml"
echo ""

# ── Step 3: Train ────────────────────────────────────────────────────────────
echo "================================================="
echo " [3/4] Training LoRA (est. 30-60 min on A100)..."
echo "================================================="
cd "$BASE_DIR/ai-toolkit"
python run.py "$CONFIG_PATH"
cd "$BASE_DIR"
echo ""

# ── Step 4: Inference ────────────────────────────────────────────────────────
echo "================================================="
echo " [4/4] Running inference test..."
echo "================================================="
LATEST_LORA=$(find "$BASE_DIR/output/$RUN_NAME" -name "*.safetensors" 2>/dev/null | sort | tail -1)
if [ -z "$LATEST_LORA" ]; then
    echo "ERROR: No trained LoRA found in output/$RUN_NAME/"
    exit 1
fi
echo "Using checkpoint: $LATEST_LORA"

python "$BASE_DIR/inference.py" \
    --lora_path "$LATEST_LORA" \
    --trigger_word "$TRIGGER_WORD" \
    --output_dir "$BASE_DIR/output/inference" \
    --num_images 6

echo ""
echo "================================================="
echo " Pipeline complete!"
echo ""
echo "  Trained LoRA : $LATEST_LORA"
echo "  Test images  : $BASE_DIR/output/inference/"
echo "================================================="
