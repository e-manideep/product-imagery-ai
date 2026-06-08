# Air Jordan 1 "Chicago" — FLUX.2 [klein] 9B DreamBooth LoRA

Plan B, quality-first. Train a LoRA on **FLUX.2 [klein] 9B** so we can generate
this exact sneaker in any scene. End-to-end: `clone -> setup -> run`.
**No label-text step, no compositing** — a sneaker's identity is shape + colorway
+ swoosh (a graphic, not spelled text), so the text-rendering problem is gone.

klein 9B is the distilled, memory-efficient FLUX.2 variant — far cheaper and
faster than dev (32B), and it runs on consumer-class GPUs. We train the base in
**bf16 (no FP8) for best quality** since the model is small enough.

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

- **GPU:** klein 9B is light. **L40S / A100 40GB / RTX A6000 48GB** is comfortable
  in bf16. A 24GB RTX 4090 works with `FP8=1` (needs compute capability >= 8.9).
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

## Config (env overrides)

| Variable | Default | Notes |
|---|---|---|
| `IDENTIFIER` | `tjkzx Air Jordan 1 Chicago sneaker` | Token + real name anchor |
| `RANK` | `64` | LoRA rank (higher = more identity capacity) |
| `MAX_STEPS` | `2500` | Checkpoints every 500 — pick the best |
| `LEARNING_RATE` | `1e-4` | |
| `RESOLUTION` | `1024` | |
| `GRAD_ACCUM` | `4` | Effective batch = 4 |
| `FP8` | `0` | bf16 base (best quality); set `1` to save VRAM on 24GB |

## Output

```
output/product_jordan/pytorch_lora_weights.safetensors   # final LoRA
output/product_jordan/checkpoint-*                        # intermediate (pick best)
output/inference/                                         # 10 example shots
```

## Inference (custom prompts)

klein is distilled — low steps, low guidance. Use `{id}` or write the name directly:

```bash
python inference.py --lora_dir output/product_jordan --output_dir output/mine \
  --steps 8 --guidance_scale 1.0 \
  --prompt "a product photo of {id} on a concrete block, studio light, e-commerce" \
  --prompt "{id} on a basketball court under spotlights, dramatic shadows"
```

Notes:
- klein is distilled — `--steps 4` to `8`, `--guidance_scale 1.0`.
- On a 40GB+ GPU add `--no_offload` for faster generation.
