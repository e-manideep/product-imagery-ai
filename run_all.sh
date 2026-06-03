#!/bin/bash
# Full pipeline: caption → train → inference
# Override any variable by exporting before running.
#
# Example:
#   export HF_TOKEN=hf_xxx
#   export TRIGGER_WORD=xtbll
#   export TRAIN_STEPS=1000
#   bash run_all.sh

set -e

# ── Config ──────────────────────────────────────────────────────────────────
export BASE_DIR="${BASE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
export TRIGGER_WORD="${TRIGGER_WORD:-xtbll}"
export LORA_RANK="${LORA_RANK:-16}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000}"
export RUN_NAME="${RUN_NAME:-product_lora}"
export DATA_DIR="${DATA_DIR:-$BASE_DIR/data/product}"
export OUTPUT_DIR="${OUTPUT_DIR:-$BASE_DIR/output/$RUN_NAME}"

echo "================================================="
echo " Product Imagery AI Pipeline"
echo "================================================="
echo "  BASE_DIR:     $BASE_DIR"
echo "  TRIGGER_WORD: $TRIGGER_WORD"
echo "  LORA_RANK:    $LORA_RANK"
echo "  TRAIN_STEPS:  $TRAIN_STEPS"
echo "  RUN_NAME:     $RUN_NAME"
echo "  DATA_DIR:     $DATA_DIR"
echo ""

# ── Preflight ────────────────────────────────────────────────────────────────
if [ ! -f "$BASE_DIR/train_dreambooth_lora_flux.py" ]; then
    echo "ERROR: train_dreambooth_lora_flux.py not found. Run setup.sh first."
    exit 1
fi

IMAGE_COUNT=$(find "$DATA_DIR" \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) 2>/dev/null | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "ERROR: No images found in $DATA_DIR"
    exit 1
fi
echo "Found $IMAGE_COUNT product images."
echo ""

# ── Step 1: Caption ──────────────────────────────────────────────────────────
echo "================================================="
echo " [1/3] Captioning images..."
echo "================================================="
python "$BASE_DIR/caption.py" \
    --data_dir "$DATA_DIR" \
    --trigger_word "$TRIGGER_WORD"

echo ""

# ── Step 2: Train ─────────────────────────────────────────────────────────────
echo "================================================="
echo " [2/3] Training FLUX DreamBooth LoRA..."
echo "================================================="
mkdir -p "$OUTPUT_DIR"

# Instance prompt: used for all images
INSTANCE_PROMPT="a photo of ${TRIGGER_WORD} perfume bottle"

accelerate launch "$BASE_DIR/train_dreambooth_lora_flux.py" \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
    --instance_data_dir="$DATA_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --mixed_precision="bf16" \
    --instance_prompt="$INSTANCE_PROMPT" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps="$TRAIN_STEPS" \
    --rank="$LORA_RANK" \
    --seed=42 \
    --checkpointing_steps=250 \
    --validation_prompt="a photo of ${TRIGGER_WORD} perfume bottle on a marble table, product photography" \
    --validation_epochs=500

echo ""

# ── Step 3: Inference ────────────────────────────────────────────────────────
echo "================================================="
echo " [3/3] Running inference test..."
echo "================================================="
LATEST_LORA=$(find "$OUTPUT_DIR" -name "pytorch_lora_weights.safetensors" 2>/dev/null | head -1)
if [ -z "$LATEST_LORA" ]; then
    # Try checkpoint subdirs
    LATEST_LORA=$(find "$OUTPUT_DIR" -name "*.safetensors" 2>/dev/null | sort | tail -1)
fi

if [ -z "$LATEST_LORA" ]; then
    echo "ERROR: No trained LoRA found in $OUTPUT_DIR"
    exit 1
fi
echo "Using LoRA: $LATEST_LORA"

python "$BASE_DIR/inference.py" \
    --lora_path "$OUTPUT_DIR" \
    --trigger_word "$TRIGGER_WORD" \
    --output_dir "$BASE_DIR/output/inference" \
    --num_images 6

echo ""
echo "================================================="
echo " Pipeline complete!"
echo "  Trained LoRA : $OUTPUT_DIR"
echo "  Test images  : $BASE_DIR/output/inference/"
echo "================================================="
