#!/bin/bash
# Run 002 - SDXL DreamBooth with DoRA + Pivotal Tuning
# Pipeline: GPT-4o captioning -> train -> inference
#
#   export HF_TOKEN=hf_xxx
#   export OPENAI_API_KEY=sk-xxx
#   bash run.sh
#
# Every knob below is overridable via env var.

set -e

export SDXL_DIR="${SDXL_DIR:-$(cd "$(dirname "$0")" && pwd)}"
export REPO_ROOT="$(cd "$SDXL_DIR/.." && pwd)"

export TRIGGER_WORD="${TRIGGER_WORD:-xtbll}"
export RANK="${RANK:-32}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"   # UNet DoRA + token embeddings
export TEXT_ENCODER_LR="${TEXT_ENCODER_LR:-5e-6}"
export TI_TOKENS="${TI_TOKENS:-2}"              # new tokens per concept (<s0><s1>)
export TI_FRAC="${TI_FRAC:-0.5}"                # fraction of steps that train the TI embeddings
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export RESOLUTION="${RESOLUTION:-1024}"
export REPORT_TO="${REPORT_TO:-tensorboard}"

export SRC_DIR="${SRC_DIR:-$REPO_ROOT/data/product}"
export CLEAN_DIR="${CLEAN_DIR:-$SDXL_DIR/dataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-$SDXL_DIR/output/product_dora}"

echo "================================================="
echo " Run 002 - SDXL DoRA + Pivotal Tuning"
echo "================================================="
echo "  TRIGGER_WORD:    $TRIGGER_WORD"
echo "  RANK:            $RANK"
echo "  MAX_STEPS:       $MAX_STEPS"
echo "  LEARNING_RATE:   $LEARNING_RATE  (UNet DoRA + token embeddings)"
echo "  TEXT_ENCODER_LR: $TEXT_ENCODER_LR"
echo "  TI tokens:       $TI_TOKENS   (<s0>..<s$((TI_TOKENS-1))>)"
echo "  SRC_DIR:         $SRC_DIR"
echo "  OUTPUT_DIR:      $OUTPUT_DIR"
echo ""

# Preflight
if [ ! -f "$SDXL_DIR/train_dreambooth_lora_sdxl_advanced.py" ]; then
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

# ── Step 1: GPT-4o captioning -> clean dataset folder + metadata.jsonl ────────
echo "================================================="
echo " [1/3] Captioning with GPT-4o (scene-only)..."
echo "================================================="
rm -rf "$CLEAN_DIR"
python "$SDXL_DIR/caption.py" \
    --src_dir "$SRC_DIR" \
    --out_dir "$CLEAN_DIR" \
    --trigger_word "$TRIGGER_WORD"
echo ""

# ── Step 2: Train SDXL DoRA + Pivotal Tuning ─────────────────────────────────
echo "================================================="
echo " [2/3] Training (DoRA + Pivotal Tuning)..."
echo "================================================="
mkdir -p "$OUTPUT_DIR"

accelerate launch "$SDXL_DIR/train_dreambooth_lora_sdxl_advanced.py" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
    --dataset_name="$CLEAN_DIR" \
    --instance_prompt="$TRIGGER_WORD" \
    --caption_column="prompt" \
    --token_abstraction="$TRIGGER_WORD" \
    --num_new_tokens_per_abstraction="$TI_TOKENS" \
    --train_text_encoder_ti \
    --train_text_encoder_ti_frac="$TI_FRAC" \
    --use_dora \
    --output_dir="$OUTPUT_DIR" \
    --mixed_precision="bf16" \
    --resolution="$RESOLUTION" \
    --train_batch_size="$BATCH_SIZE" \
    --repeats=1 \
    --gradient_accumulation_steps="$GRAD_ACCUM" \
    --gradient_checkpointing \
    --learning_rate="$LEARNING_RATE" \
    --text_encoder_lr="$TEXT_ENCODER_LR" \
    --optimizer="adamW" \
    --snr_gamma=5.0 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --rank="$RANK" \
    --max_train_steps="$MAX_STEPS" \
    --checkpointing_steps=500 \
    --seed=42 \
    --report_to="$REPORT_TO"
echo ""

# ── Step 3: Inference ────────────────────────────────────────────────────────
echo "================================================="
echo " [3/3] Inference..."
echo "================================================="
python "$SDXL_DIR/inference.py" \
    --lora_dir "$OUTPUT_DIR" \
    --output_dir "$SDXL_DIR/output/inference" \
    --num_images 6
echo ""

echo "================================================="
echo " Run 002 complete."
echo "  Weights:  $OUTPUT_DIR"
echo "  Images:   $SDXL_DIR/output/inference"
echo "================================================="
