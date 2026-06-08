# Progress Tracker — Air Jordan 1 "Chicago" product-imagery LoRA

_Last updated: 2026-06-08_

| # | Stage | Status | Notes |
|---|-------|--------|-------|
| 1 | Product chosen | DONE | AJ1 High OG "Chicago" (555088-101) — rigid shape, no text label |
| 2 | Dataset sourced + curated | DONE | 16 imgs: 12 studio 360 angles + 4 lifestyle; watermark-free, colorway-verified |
| 3 | Captions (local, Gemini 2.5 Pro) | DONE | scene-only, identifier-anchored; captions.jsonl (16/16) |
| 4 | Training scripts | DONE | setup.sh, run.sh, inference.py, build_dataset.py, caption.py |
| 5 | Push to repo | TODO | so it is clonable on RunPod |
| 6 | Train on GPU (RunPod) | TODO | H100/H200 80GB, FP8, ~2500 steps, ckpt every 500 |
| 7 | Inference + pick best checkpoint | TODO | 10 example scenes (inference.py --examples) |
| 8 | Review outputs | TODO | compare checkpoints, choose best |

## Best-output config
- Base: FLUX.2-dev 32B (FP8 + offload + cache_latents + grad checkpointing + 8-bit Adam)
- Identity anchor: "tjkzx Air Jordan 1 Chicago sneaker" (unique token + real name -> leverages model prior)
- rank 64, lora_alpha 64, lr 1e-4, constant scheduler, 2500 steps, resolution 1024
- guidance 1 (train) / guidance 4.0 + 50 steps (inference)
- validation every 20 epochs to pick the best checkpoint
- NO label-text / NO compositing (sneaker has no spelled label)

## Locations
- Pipeline folder: C:\Users\MANID\product-imagery-ai\jordan\
- Training images: jordan\data\product\ (001-016.jpg)
- Original dataset: C:\Users\MANID\Downloads\aj1_chicago_dataset\
- Contact sheet: C:\Users\MANID\Downloads\aj1_chicago_contact_sheet.png

## Next action
Push `jordan/` to the repo, then on a RunPod H100/H200: `bash setup.sh && bash run.sh`.
