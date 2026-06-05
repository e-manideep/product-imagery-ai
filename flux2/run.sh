#!/bin/bash
# Run 003 - FLUX.2 klein 9B DreamBooth LoRA (plain LoRA, bf16, quality-first)
# Pipeline: build dataset -> train -> inference
#
#   export HF_TOKEN=hf_xxx     # must have accepted the klein license
#   bash run.sh
#
# Captions are reused from captions.jsonl (scene-only). No API key needed here.

set -e

export FLUX2_DIR="${FLUX2_DIR:-$(cd "$(dirname "$0")" && pwd)}"
export REPO_ROOT="$(cd "$FLUX2_DIR/.." && pwd)"

export TRIGGER_WORD="${TRIGGER_WORD:-xtbll}"
export RANK="${RANK:-32}"
export MAX_STEPS="${MAX_STEPS:-1500}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export RESOLUTION="${RESOLUTION:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export REPORT_TO="${REPORT_TO:-tensorboard}"

export SRC_DIR="${SRC_DIR:-$REPO_ROOT/data/product}"
export CAPTIONS="${CAPTIONS:-$FLUX2_DIR/captions.jsonl}"
export CLEAN_DIR="${CLEAN_DIR:-$FLUX2_DIR/dataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-$FLUX2_DIR/output/product_flux2}"

echo "================================================="
echo " Run 003 - FLUX.2 klein 9B (bf16, plain LoRA)"
echo "================================================="
echo "  TRIGGER_WORD:  $TRIGGER_WORD"
echo "  RANK:          $RANK"
echo "  MAX_STEPS:     $MAX_STEPS"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  RESOLUTION:    $RESOLUTION"
echo ""

if [ ! -f "$FLUX2_DIR/train_dreambooth_lora_flux2_klein.py" ]; then
    echo "ERROR: training script missing. Run setup.sh first."
    exit 1
fi
IMG_COUNT=$(find "$SRC_DIR" \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) 2>/dev/null | wc -l)
if [ "$IMG_COUNT" -eq 0 ]; then
    echo "ERROR: no images in $SRC_DIR"
    exit 1
fi
echo "Found $IMG_COUNT source images."
echo ""

# ── Step 1: build dataset (images + metadata.jsonl) ──────────────────────────
echo "================================================="
echo " [1/3] Build dataset"
echo "================================================="
if [ ! -f "$CAPTIONS" ]; then
    echo "ERROR: captions.jsonl missing at $CAPTIONS"
    exit 1
fi
python "$FLUX2_DIR/build_dataset.py" \
    --src_dir "$SRC_DIR" \
    --captions "$CAPTIONS" \
    --out_dir "$CLEAN_DIR"
echo ""

# ── Step 2: train ────────────────────────────────────────────────────────────
echo "================================================="
echo " [2/3] Train FLUX.2 klein LoRA (bf16)"
echo "================================================="
mkdir -p "$OUTPUT_DIR"

accelerate launch "$FLUX2_DIR/train_dreambooth_lora_flux2_klein.py" \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.2-klein-9B" \
    --dataset_name="$CLEAN_DIR" \
    --caption_column="prompt" \
    --instance_prompt="a photo of ${TRIGGER_WORD} perfume bottle" \
    --output_dir="$OUTPUT_DIR" \
    --mixed_precision="bf16" \
    --resolution="$RESOLUTION" \
    --train_batch_size="$BATCH_SIZE" \
    --gradient_accumulation_steps="$GRAD_ACCUM" \
    --gradient_checkpointing \
    --cache_latents \
    --optimizer="AdamW" \
    --learning_rate="$LEARNING_RATE" \
    --guidance_scale=1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=100 \
    --rank="$RANK" \
    --max_train_steps="$MAX_STEPS" \
    --checkpointing_steps=500 \
    --seed=42 \
    --report_to="$REPORT_TO"
echo ""

# ── Step 3: inference ────────────────────────────────────────────────────────
echo "================================================="
echo " [3/3] Inference (10 example shots)"
echo "================================================="
python "$FLUX2_DIR/inference.py" \
    --lora_dir "$OUTPUT_DIR" \
    --output_dir "$FLUX2_DIR/output/inference" \
    --trigger_word "$TRIGGER_WORD" \
    --examples
echo ""

echo "================================================="
echo " Run 003 complete."
echo "  Weights: $OUTPUT_DIR"
echo "  Images:  $FLUX2_DIR/output/inference"
echo "================================================="
