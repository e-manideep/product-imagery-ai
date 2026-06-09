# Single-Garment Test — Navy Leaf Print Shirt (FLUX.2 klein 9B LoRA)

A single-product DreamBooth LoRA on **FLUX.2 [klein] 9B**, trained on one garment
(the WES Navy Leaf print shirt) so the trigger `prtx` reproduces that exact shirt
in any scene. Same approach as the Jordan shoe run, scaled down for one item.

End-to-end: `clone -> setup -> run`.

## Dataset

`data/product/` — **12 images** of one shirt: the 4 real catalogue variations
(on-model front, on-model 3/4, on-model side/back, print close-up) each expanded
x3 (original + mirror + gentle zoom-crop), since the catalogue only provides 4
angles per product and DreamBooth needs a handful of samples.

Captions (`captions.jsonl`) are **scene-only** — they describe the presentation
("worn by a male model, plain studio background"), never the print, so the
trigger `prtx` learns the shirt itself.

## Run it

```bash
cd /workspace
git clone <repo-url>
cd product-imagery/shirt
export HF_TOKEN=hf_xxx        # accepted the FLUX.2-klein-9B license
bash setup.sh
bash run.sh
```

## Configuration — and why

| Setting | Value | Reason |
|---|---|---|
| Base model | FLUX.2-klein-9B | the 9B model that worked on the Jordan run |
| Precision | bf16, no FP8 | best quality; klein 9B fits comfortably on a large GPU |
| Adapter | LoRA, **rank 64** (alpha 64) | enough to capture the leaf print, low enough to limit overfit on a tiny set |
| Trigger / identity | `prtx shirt`, **scene-only captions** | the trigger absorbs the specific shirt (the captions never describe the print) |
| Steps | **1,200**, checkpoint every **200** | one garment overfits fast — train short and keep 6 checkpoints to pick the cleanest |
| Optimiser | full AdamW, lr 1e-4, constant, 50 warmup | proven klein settings; full (not 8-bit) AdamW on a big GPU |
| Batch | 2 (grad accum 1) | small set; real batch keeps gradients steady |
| Resolution | 1024 | FLUX native |
| Guidance (train) | 1 | klein is guidance-distilled |
| Memory | no CPU offload, gradient checkpointing on | big-GPU max quality; checkpointing is quality-neutral |
| Saved weights | fp32 (`--upcast_before_saving`) | no precision loss in the adapter |
| Validation | every 25 epochs, 2 images | watch the shirt lock in, helps pick the best checkpoint |

`LOW_VRAM=1` re-enables CPU offload + 8-bit Adam for a 24GB card; `FP8=1` adds
FP8 training (needs compute capability >= 8.9).

## Output

```
output/product_shirt/pytorch_lora_weights.safetensors   # final LoRA
output/product_shirt/checkpoint-*                        # pick the best
output/inference/                                        # example shots
```

## Inference (custom prompts)

The shirt is on-model in training, so prompts place it on a person:

```bash
python inference.py --lora_dir output/product_shirt --output_dir output/mine \
  --steps 8 --guidance_scale 1.0 --no_offload \
  --prompt "{id} worn by a man on a beach at sunset, warm light" \
  --prompt "a studio e-commerce photo of {id} worn by a man, soft lighting"
```

Notes:
- klein is distilled — `--steps 4` to `8`, `--guidance_scale 1.0`.
- `{id}` expands to `prtx shirt`.
- Pick the best `checkpoint-*` with `--lora_dir output/product_shirt/checkpoint-800`.
