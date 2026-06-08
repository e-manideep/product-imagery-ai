#!/bin/bash
# Air Jordan 1 "Chicago" — FLUX.2 [klein] 9B DreamBooth LoRA (Plan B), quality-first.
# Pipeline: build dataset -> train -> inference. No label text, no compositing
# (a sneaker has no spelled label; identity is shape + colorway + swoosh).
#
#   export HF_TOKEN=hf_xxx     # must have accepted the FLUX.2-klein-9B license
#   bash run.sh
#
# Identity is anchored to a unique token + the real product name
# ("tjkzx Air Jordan 1 Chicago sneaker") so the base model's strong, accurate
# prior for this shoe does the heavy lifting and the LoRA sharpens to our refs.

set -e

export FLUX2_DIR="${FLUX2_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# Token + real name anchor. Used identically in captions and inference.
export IDENTIFIER="${IDENTIFIER:-tjkzx Air Jordan 1 Chicago sneaker}"
export RANK="${RANK:-64}"
export MAX_STEPS="${MAX_STEPS:-2500}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export RESOLUTION="${RESOLUTION:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
# klein 9B is small enough to train in bf16 for best quality (default).
# Set FP8=1 (needs GPU compute capability >= 8.9) only to save VRAM on a 24GB card.
export FP8="${FP8:-0}"

export SRC_DIR="${SRC_DIR:-$FLUX2_DIR/data/product}"
export CAPTIONS="${CAPTIONS:-$FLUX2_DIR/captions.jsonl}"
export CLEAN_DIR="${CLEAN_DIR:-$FLUX2_DIR/dataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-$FLUX2_DIR/output/product_jordan}"

# A validation prompt rendered during training so we can pick the best checkpoint.
export VAL_PROMPT="${VAL_PROMPT:-a photo of ${IDENTIFIER} on a city street at golden hour, cinematic}"

echo "================================================="
echo " Air Jordan 1 Chicago — FLUX.2 [klein] 9B LoRA"
echo "================================================="
echo "  IDENTIFIER:    $IDENTIFIER"
echo "  RANK:          $RANK"
echo "  MAX_STEPS:     $MAX_STEPS"
echo "  LEARNING_RATE: $LEARNING_RATE"
echo "  RESOLUTION:    $RESOLUTION"
echo "  FP8 training:  $FP8"
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
    echo "ERROR: captions missing at $CAPTIONS  (run caption.py locally first)"
    exit 1
fi
python "$FLUX2_DIR/build_dataset.py" \
    --src_dir "$SRC_DIR" \
    --captions "$CAPTIONS" \
    --out_dir "$CLEAN_DIR"
echo ""

# ── Step 2: train ────────────────────────────────────────────────────────────
echo "================================================="
echo " [2/3] Train FLUX.2 [klein] 9B LoRA"
echo "================================================="
mkdir -p "$OUTPUT_DIR"

PRECISION_FLAGS=""
if [ "$FP8" = "1" ]; then
    PRECISION_FLAGS="--do_fp8_training"
    echo "FP8 enabled — lower VRAM (needs compute capability >= 8.9)."
else
    echo "Training base in bf16 (best quality; klein 9B fits comfortably)."
fi

accelerate launch "$FLUX2_DIR/train_dreambooth_lora_flux2_klein.py" \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.2-klein-9B" \
    --dataset_name="$CLEAN_DIR" \
    --caption_column="prompt" \
    --instance_prompt="a photo of ${IDENTIFIER}" \
    --output_dir="$OUTPUT_DIR" \
    --mixed_precision="bf16" \
    $PRECISION_FLAGS \
    --offload \
    --cache_latents \
    --gradient_checkpointing \
    --use_8bit_adam \
    --resolution="$RESOLUTION" \
    --train_batch_size="$BATCH_SIZE" \
    --gradient_accumulation_steps="$GRAD_ACCUM" \
    --optimizer="adamW" \
    --learning_rate="$LEARNING_RATE" \
    --guidance_scale=1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=100 \
    --rank="$RANK" \
    --lora_alpha="$RANK" \
    --max_train_steps="$MAX_STEPS" \
    --checkpointing_steps=500 \
    --validation_prompt="$VAL_PROMPT" \
    --validation_epochs=20 \
    --num_validation_images=2 \
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
    --identifier "$IDENTIFIER" \
    --examples
echo ""

echo "================================================="
echo " Done."
echo "  Weights: $OUTPUT_DIR/pytorch_lora_weights.safetensors"
echo "  Checkpoints (pick best): $OUTPUT_DIR/checkpoint-*"
echo "  Images:  $FLUX2_DIR/output/inference"
echo "================================================="
