# Air Jordan 1 "Chicago" — FLUX.2 [klein] 9B DreamBooth LoRA

Plan B, quality-first. Train a LoRA on **FLUX.2 [klein] 9B** so we can generate
this exact sneaker in any scene. End-to-end: `clone -> setup -> run`.
**No label-text step, no compositing** — a sneaker's identity is shape + colorway
+ swoosh (a graphic, not spelled text), so the text-rendering problem is gone.

Tuned for **maximum quality on a large GPU** (>= 48GB, e.g. a 96GB card): bf16
base (no FP8), full AdamW (no 8-bit), no CPU offload, real batch size, fp32 LoRA
save, and frequent checkpoints so we keep the sharpest one.

Identity is anchored to a unique token **+ the real product name**
(`tjkzx Air Jordan 1 Chicago sneaker`) in both captions and inference, so the
base model's prior for this famous shoe helps and the LoRA sharpens it to our refs.

## Dataset

`data/product/` — 16 curated images of the AJ1 High OG "Chicago" (555088-101):
- **001-012:** 12 distinct studio angles from the official 360 spin (clean white
  background, watermark-free, consistent).
- **013-016:** 4 verified lifestyle shots (varied backgrounds) so the LoRA
  generalizes beyond white seamless.

## Requirements

- **GPU (recommended):** a large card — **48GB+** (L40S/A6000) up to **96GB**.
  The defaults below assume plenty of VRAM and maximize quality.
- **Low-VRAM (24GB):** run `LOW_VRAM=1 FP8=1 BATCH_SIZE=1 GRAD_ACCUM=4 bash run.sh`
  to re-enable offload + 8-bit Adam + FP8.
- **Pod:** **PyTorch 2.8 / CUDA 12.4+** (FLUX.2 needs torch >= 2.7).
- **HuggingFace:** klein 9B is **gated** — accept the license at
  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B and use an HF token.

## Run it

```bash
cd /workspace
git clone <repo-url>
cd product-imagery/jordan
export HF_TOKEN=hf_xxx
bash setup.sh
bash run.sh
```

Captions (`captions.jsonl`) are generated locally with `caption.py` and committed,
so the pod never needs an API key.

## Config (env overrides) — max-quality defaults

| Variable | Default | Notes |
|---|---|---|
| `IDENTIFIER` | `tjkzx Air Jordan 1 Chicago sneaker` | Token + real name anchor |
| `RANK` | `96` | LoRA rank (more capacity for fine shoe detail) |
| `MAX_STEPS` | `2500` | Checkpoints every 250 — pick the best |
| `BATCH_SIZE` | `4` | Real batch (no accumulation) — smoother gradients |
| `GRAD_ACCUM` | `1` | |
| `LEARNING_RATE` | `1e-4` | |
| `RESOLUTION` | `1024` | |
| `FP8` | `0` | bf16 base (best quality); `1` only to save VRAM |
| `LOW_VRAM` | `0` | `1` re-enables CPU offload + 8-bit Adam |

Training also uses `--upcast_before_saving` (fp32 LoRA weights) and validation
images every ~200 steps so you can watch identity lock in.

## Output

```
output/product_jordan/pytorch_lora_weights.safetensors   # final LoRA (fp32)
output/product_jordan/checkpoint-250 / -500 / ...         # pick the best
output/inference/                                         # 10 example shots
```

## Inference (custom prompts)

klein is distilled — low steps, low guidance. Use `{id}` or write the name directly:

```bash
python inference.py --lora_dir output/product_jordan --output_dir output/mine \
  --steps 8 --guidance_scale 1.0 --no_offload \
  --prompt "a product photo of {id} on a concrete block, studio light, e-commerce" \
  --prompt "{id} on a basketball court under spotlights, dramatic shadows"
```

Notes:
- klein is distilled — `--steps 4` to `8`, `--guidance_scale 1.0`.
- `--no_offload` keeps the model on the GPU for faster generation (use on 40GB+).
