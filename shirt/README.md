# Single-Garment Test — Navy Leaf Print Shirt (FLUX.2 klein 9B LoRA)

A single-product DreamBooth LoRA on **FLUX.2 [klein] 9B**, trained on one garment
(the WES Navy Leaf print shirt) so the trigger `prtx` reproduces that exact shirt
in any scene. Tuned for **best output**, not just "safe".

End-to-end: `clone -> setup -> run`.

## Dataset

`data/product/` — **12 images** of one shirt: the 4 real catalogue variations
(on-model front, 3/4, side/back, print close-up) each expanded x3 (original +
mirror + gentle zoom-crop), since the catalogue only provides 4 angles per
product. Captions (`captions.jsonl`) are **scene-only** — they describe the
presentation, never the print, so the trigger `prtx` learns the shirt itself.

## Run it

```bash
cd /workspace
git clone <repo-url>
cd product-imagery/shirt
export HF_TOKEN=hf_xxx        # accepted the FLUX.2-klein-9B license
bash setup.sh
bash run.sh
```

## Configuration — and why (best-output)

With only ~4 real shots the enemy is overfitting, not rank/steps. So the main
quality lever is **prior preservation**, not a tiny rank/step count.

| Setting | Value | Reason |
|---|---|---|
| Base model | FLUX.2-klein-9B, bf16 | the 9B that worked on Jordan; best quality |
| **Prior preservation** | **ON, 200 class images** ("a man wearing a casual shirt") | the key lever — regularises so the LoRA learns only the print and keeps the model's ability to render people/scenes; stops overfit and enables longer training |
| Adapter | LoRA **rank 96** (Jordan config), **dropout 0.1** | high capacity; prior preservation + dropout guard against overfit on the small set |
| Trigger / identity | `prtx shirt`, scene-only captions | the trigger absorbs the exact shirt |
| Steps | **2,500**, checkpoint every **250** | prior preservation makes longer training safe; pick the cleanest checkpoint |
| Augmentation | pre-baked 12 + `--random_flip` | more variety from few real shots |
| Optimiser | full AdamW, lr 1e-4, constant, 100 warmup | proven klein settings |
| Batch / res | 2 / 1024 | small set, FLUX-native res |
| Guidance (train) | 1 | klein is guidance-distilled |
| Memory | no offload, gradient checkpointing, fp32 save | big-GPU max quality |
| Inference | klein: 8 steps, guidance 1.0 | distilled model sweet spot |

Toggles: `PRIOR=0` disables prior preservation; `NUM_CLASS_IMAGES` controls how
many class images are generated; `LOW_VRAM=1` re-enables offload + 8-bit Adam;
`FP8=1` adds FP8 training (needs compute capability >= 8.9).

Note: with prior preservation the run first **generates the class images**
(~15–25 min on klein) before training — that is the cost of the better output.

## Output

```
output/product_shirt/pytorch_lora_weights.safetensors   # final LoRA
output/product_shirt/checkpoint-*                        # pick the best
class_images/                                            # generated class set (regularisation)
output/inference/                                        # example shots
```

## Inference (custom prompts)

```bash
python inference.py --lora_dir output/product_shirt --output_dir output/mine \
  --steps 8 --guidance_scale 1.0 --no_offload \
  --prompt "{id} worn by a man on a beach at sunset, warm light" \
  --prompt "a studio e-commerce photo of {id} worn by a man, soft lighting"
```

- klein is distilled — `--steps 4` to `8`, `--guidance_scale 1.0`.
- `{id}` expands to `prtx shirt`.
- Try a specific checkpoint: `--lora_dir output/product_shirt/checkpoint-1500`.
