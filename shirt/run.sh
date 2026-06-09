#!/bin/bash
# Single-garment test — WES Navy Leaf print shirt — FLUX.2 [klein] 9B DreamBooth LoRA.
# Best-output configuration. Pipeline: build dataset -> train -> inference.
#
#   export HF_TOKEN=hf_xxx     # must have accepted the FLUX.2-klein-9B license
#   bash run.sh

set -e

export FLUX2_DIR="${FLUX2_DIR:-$(cd "$(dirname "$0")" && pwd)}"

# Trigger + class. The trigger absorbs the specific shirt; captions are scene-only.
export IDENTIFIER="${IDENTIFIER:-prtx shirt}"

# Best-output config for a single garment from only ~4 real shots. The quality
# lever here is NOT a tiny rank/step count -- it is prior preservation, which
# regularises training with generated class images so the LoRA learns this exact
# shirt's print while keeping the base model's ability to render people/scenes.
# That lets us train longer (and pick the best checkpoint) without overfitting.
export RANK="${RANK:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
export MAX_STEPS="${MAX_STEPS:-2500}"
export CKPT_STEPS="${CKPT_STEPS:-250}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export RESOLUTION="${RESOLUTION:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export REPORT_TO="${REPORT_TO:-tensorboard}"

# Prior preservation -- the main quality lever. Set PRIOR=0 to disable.
export PRIOR="${PRIOR:-1}"
export NUM_CLASS_IMAGES="${NUM_CLASS_IMAGES:-200}"
export CLASS_PROMPT="${CLASS_PROMPT:-a photo of a man wearing a casual shirt}"
export CLASS_DIR="${CLASS_DIR:-$FLUX2_DIR/class_images}"

# klein 9B trains in bf16 for best quality. FP8 only saves VRAM on a 24GB card.
export FP8="${FP8:-0}"
# Big GPU: no offload + full AdamW (best quality). LOW_VRAM=1 restores both.
export LOW_VRAM="${LOW_VRAM:-0}"

export SRC_DIR="${SRC_DIR:-$FLUX2_DIR/data/product}"
export CAPTIONS="${CAPTIONS:-$FLUX2_DIR/captions.jsonl}"
export CLEAN_DIR="${CLEAN_DIR:-$FLUX2_DIR/dataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-$FLUX2_DIR/output/product_shirt}"

export VAL_PROMPT="${VAL_PROMPT:-a photo of ${IDENTIFIER} on a man walking down a city street, golden hour}"

echo "================================================="
echo " Single garment (best output) — FLUX.2 [klein] 9B LoRA"
echo "================================================="
echo "  IDENTIFIER:    $IDENTIFIER"
echo "  RANK:          $RANK  (dropout $LORA_DROPOUT)"
echo "  MAX_STEPS:     $MAX_STEPS  (checkpoint every $CKPT_STEPS)"
echo "  PRIOR:         $PRIOR  ($NUM_CLASS_IMAGES class images)"
echo "  LEARNING_RATE: $LEARNING_RATE"
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

MEM_FLAGS=""
if [ "$LOW_VRAM" = "1" ]; then
    MEM_FLAGS="--offload --use_8bit_adam"
    echo "LOW_VRAM — CPU offload + 8-bit Adam enabled."
else
    echo "Big-GPU mode — no offload, full AdamW (max quality)."
fi

# Prior preservation flags (spaced class prompt -> use an array so it stays one arg).
PRIOR_ARGS=()
if [ "$PRIOR" = "1" ]; then
    PRIOR_ARGS=(--with_prior_preservation --prior_loss_weight=1.0 \
        --class_prompt="$CLASS_PROMPT" --class_data_dir="$CLASS_DIR" \
        --num_class_images="$NUM_CLASS_IMAGES")
    echo "Prior preservation ON — generating/using $NUM_CLASS_IMAGES class images."
fi

# Key flags, for review:
#   --with_prior_preservation : the quality lever; regularises with class images
#   --rank / --lora_dropout   : moderate capacity + dropout to avoid overfit on a tiny set
#   --random_flip             : extra on-the-fly augmentation
#   --checkpointing_steps     : frequent checkpoints; pick the cleanest one afterwards
#   --upcast_before_saving    : final adapter saved in fp32
accelerate launch "$FLUX2_DIR/train_dreambooth_lora_flux2_klein.py" \
    --pretrained_model_name_or_path="black-forest-labs/FLUX.2-klein-9B" \
    --dataset_name="$CLEAN_DIR" \
    --caption_column="prompt" \
    --instance_prompt="a photo of ${IDENTIFIER}" \
    --output_dir="$OUTPUT_DIR" \
    --mixed_precision="bf16" \
    $PRECISION_FLAGS \
    $MEM_FLAGS \
    "${PRIOR_ARGS[@]}" \
    --cache_latents \
    --gradient_checkpointing \
    --random_flip \
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
    --lora_dropout="$LORA_DROPOUT" \
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
