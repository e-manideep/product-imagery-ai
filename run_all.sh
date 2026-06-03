#!/bin/bash
set -e

export BASE_DIR="${BASE_DIR:-$(cd "$(dirname "$0")" && pwd)}"
export TRIGGER_WORD="${TRIGGER_WORD:-xtbll}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000}"
export LORA_RANK="${LORA_RANK:-16}"
export RUN_NAME="${RUN_NAME:-product_lora}"
export DATA_DIR="${DATA_DIR:-$BASE_DIR/data/product}"
export OUTPUT_DIR="${OUTPUT_DIR:-$BASE_DIR/output/$RUN_NAME}"

echo "================================================="
echo " Product Imagery AI Pipeline"
echo "================================================="
echo "  TRIGGER_WORD: $TRIGGER_WORD"
echo "  TRAIN_STEPS:  $TRAIN_STEPS"
echo "  LORA_RANK:    $LORA_RANK"
echo ""

# Preflight
if [ ! -f "$BASE_DIR/train_dreambooth_lora_flux.py" ]; then
    echo "ERROR: Run setup.sh first."
    exit 1
fi

IMAGE_COUNT=$(find "$DATA_DIR" \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) 2>/dev/null | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "ERROR: No images found in $DATA_DIR"
    exit 1
fi
echo "Found $IMAGE_COUNT images."

# Create clean image-only directory (dreambooth script loads ALL files as images)
CLEAN_DIR="$BASE_DIR/output/train_images_clean"
rm -rf "$CLEAN_DIR" && mkdir -p "$CLEAN_DIR"
find "$DATA_DIR" \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) \
    -exec cp {} "$CLEAN_DIR/" \;
echo "Copied $IMAGE_COUNT images to clean training dir."
echo ""

# Train
echo "================================================="
echo " Training FLUX DreamBooth LoRA..."
echo "================================================="
mkdir -p "$OUTPUT_DIR"

accelerate launch "$BASE_DIR/train_dreambooth_lora_flux.py" \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
    --instance_data_dir="$CLEAN_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --mixed_precision="bf16" \
    --instance_prompt="a photo of ${TRIGGER_WORD} perfume bottle" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps="$TRAIN_STEPS" \
    --rank="$LORA_RANK" \
    --seed=42 \
    --checkpointing_steps=250

echo ""
echo "================================================="
echo " Done! LoRA saved to: $OUTPUT_DIR"
echo "================================================="
