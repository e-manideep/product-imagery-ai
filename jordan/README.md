# Air Jordan 1 "Chicago" — FLUX.2 [dev] 32B DreamBooth LoRA

Plan B, quality-first. Train a LoRA on the full **FLUX.2 [dev] (32B)** model so we
can generate this exact sneaker in any scene. End-to-end: `clone -> setup -> run`.
**No label-text step, no compositing** — a sneaker's identity is shape + colorway
+ swoosh (a graphic, not spelled text), so the text-rendering problem is gone.

Identity is anchored to a unique token **+ the real product name**
(`tjkzx Air Jordan 1 Chicago sneaker`) in both captions and inference, so the
base model's strong, accurate prior for this shoe does the heavy lifting and the
LoRA sharpens it to our reference set.

## Dataset

`data/product/` — 16 curated images of the AJ1 High OG "Chicago" (555088-101):
- **001-012:** 12 distinct studio angles from the official 360 spin (clean white
  background, watermark-free, consistent).
- **013-016:** 4 verified lifestyle shots (varied backgrounds) so the LoRA
  generalizes beyond white seamless.

## Requirements

- **GPU:** FLUX.2 [dev] is 32B and memory-heavy. Train on **H100 / H200 80GB+**
  (FP8 needs compute capability >= 8.9). Uses FP8 base + CPU-offloaded text
  encoder + cached latents + gradient checkpointing + 8-bit Adam.
- **Pod:** **PyTorch 2.8 / CUDA 12.4+** (FLUX.2 needs torch >= 2.7).
- **HuggingFace:** dev is **gated** — accept the license at
  https://huggingface.co/black-forest-labs/FLUX.2-dev and use an HF token.

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

## Config (env overrides)

| Variable | Default | Notes |
|---|---|---|
| `IDENTIFIER` | `tjkzx Air Jordan 1 Chicago sneaker` | Token + real name anchor |
| `RANK` | `64` | LoRA rank (higher = more identity capacity) |
| `MAX_STEPS` | `2500` | Checkpoints every 500 — pick the best |
| `LEARNING_RATE` | `1e-4` | |
| `RESOLUTION` | `1024` | |
| `GRAD_ACCUM` | `4` | Effective batch = 4 |
| `FP8` | `1` | FP8 base; set `0` for bf16 on H200/B200 |

## Output

```
output/product_jordan/pytorch_lora_weights.safetensors   # final LoRA
output/product_jordan/checkpoint-*                        # intermediate (pick best)
output/inference/                                         # 10 example shots
```

## Inference (custom prompts)

Use `{id}` (expands to the identifier) or write the name directly:

```bash
python inference.py --lora_dir output/product_jordan --output_dir output/mine \
  --steps 50 --guidance_scale 4.0 \
  --prompt "a product photo of {id} on a concrete block, studio light, e-commerce" \
  --prompt "{id} on a basketball court under spotlights, dramatic shadows"
```

Notes:
- dev is the full model — `--steps 50`, `--guidance_scale 4.0`.
- On a very large GPU add `--no_offload` for faster generation.
