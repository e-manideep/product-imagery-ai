#!/bin/bash
# Single-garment test — WES Navy Leaf print shirt — FLUX.2 [klein] 9B DreamBooth LoRA.
# Pipeline: build dataset -> train -> inference. One garment, a few real variations,
# so the trigger "prtx" learns this exact shirt (same idea as the Jordan shoe run).
#
#   export HF_TOKEN=hf_xxx     # must have accepted the FLUX.2-klein-9B license
#   bash run.sh

set -e

export FLUX2_DIR="${FLUX2_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# Trigger + class. The trigger absorbs the specific shirt; captions are scene-only.
export IDENTIFIER="${IDENTIFIER:-prtx shirt}"
# Single-garment defaults: 4 real images (augmented to 12) overfit fast, so a
# moderate rank, short training, and frequent checkpoints to pick the best one.
export RANK="${RANK:-64}"
export MAX_STEPS="${MAX_STEPS:-1200}"
export CKPT_STEPS="${CKPT_STEPS:-200}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export RESOLUTION="${RESOLUTION:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export REPORT_TO="${REPORT_TO:-tensorboard}"
# klein 9B trains in bf16 for best quality. FP8 only saves VRAM on a 24GB card.
export FP8="${FP8:-0}"
# Big GPU: no offload + full AdamW (best quality). LOW_VRAM=1 restores both.
export LOW_VRAM="${LOW_VRAM:-0}"

export SRC_DIR="${SRC_DIR:-$FLUX2_DIR/data/product}"
export CAPTIONS="${CAPTIONS:-$FLUX2_DIR/captions.jsonl}"
export CLEAN_DIR="${CLEAN_DIR:-$FLUX2_DIR/dataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-$FLUX2_DIR/output/product_shirt}"

# Validation prompt rendered during training so we can pick the best checkpoint.
export VAL_PROMPT="${VAL_PROMPT:-a photo of ${IDENTIFIER} on a man walking down a city street, golden hour}"

echo "================================================="
echo " Single garment — FLUX.2 [klein] 9B LoRA"
echo "================================================="
echo "  IDENTIFIER:    $IDENTIFIER"
echo "  RANK:          $RANK"
echo "  MAX_STEPS:     $MAX_STEPS  (checkpoint every $CKPT_STEPS)"
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
    echo "ERROR: captions missing at $CAPTIONS"
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

# Big-GPU: full model resident + full AdamW. LOW_VRAM=1 re-enables the savers.
MEM_FLAGS=""
if [ "$LOW_VRAM" = "1" ]; then
    MEM_FLAGS="--offload --use_8bit_adam"
    echo "LOW_VRAM — CPU offload + 8-bit Adam enabled."
else
    echo "Big-GPU mode — no offload, full AdamW (max quality)."
fi

# Key training flags, for review:
#   --dataset_name / --caption_column : images + the per-image scene-only captions
#   --instance_prompt                 : fallback prompt; the trigger carries identity
#   --rank / --lora_alpha             : LoRA capacity (64 captures the print, limits overfit)
#   --max_train_steps / --checkpointing_steps : short run + frequent checkpoints to pick the best
#   --guidance_scale 1                : klein is guidance-distilled, trained at guidance 1
#   --upcast_before_saving            : final adapter saved in fp32 (no precision loss)
#   --validation_prompt / _epochs     : sample images mid-training to watch the shirt lock in
accelerate launch "$FLUX2_DIR/train_dreambooth_lora_flux2_klein.py" \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.2-klein-9B" \
    --dataset_name="$CLEAN_DIR" \
    --caption_column="prompt" \
    --instance_prompt="a photo of ${IDENTIFIER}" \
    --output_dir="$OUTPUT_DIR" \
    --mixed_precision="bf16" \
    $PRECISION_FLAGS \
    $MEM_FLAGS \
    --cache_latents \
    --gradient_checkpointing \
    --resolution="$RESOLUTION" \
    --train_batch_size="$BATCH_SIZE" \
    --gradient_accumulation_steps="$GRAD_ACCUM" \
    --optimizer="adamW" \
    --learning_rate="$LEARNING_RATE" \
    --guidance_scale=1 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=50 \
    --rank="$RANK" \
    --lora_alpha="$RANK" \
    --max_train_steps="$MAX_STEPS" \
    --checkpointing_steps="$CKPT_STEPS" \
    --upcast_before_saving \
    --validation_prompt="$VAL_PROMPT" \
    --validation_epochs=25 \
    --num_validation_images=2 \
    --seed=42 \
    --report_to="$REPORT_TO"
echo ""

# ── Step 3: inference ────────────────────────────────────────────────────────
echo "================================================="
echo " [3/3] Inference (example shots)"
echo "================================================="
INFER_FLAGS="--no_offload"
[ "$LOW_VRAM" = "1" ] && INFER_FLAGS=""
python "$FLUX2_DIR/inference.py" \
    --lora_dir "$OUTPUT_DIR" \
    --output_dir "$FLUX2_DIR/output/inference" \
    --identifier "$IDENTIFIER" \
    $INFER_FLAGS \
    --examples
echo ""

echo "================================================="
echo " Done."
echo "  Weights: $OUTPUT_DIR/pytorch_lora_weights.safetensors"
echo "  Checkpoints (pick best): $OUTPUT_DIR/checkpoint-*"
echo "  Images:  $FLUX2_DIR/output/inference"
echo "================================================="
