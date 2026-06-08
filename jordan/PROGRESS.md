# Progress Tracker — Air Jordan 1 "Chicago" product-imagery LoRA

_Last updated: 2026-06-08 — base model switched to FLUX.2 [klein] 9B_

| # | Stage | Status | Notes |
|---|-------|--------|-------|
| 1 | Product chosen | DONE | AJ1 High OG "Chicago" (555088-101) — rigid shape, no text label |
| 2 | Dataset sourced + curated | DONE | 16 imgs: 12 studio 360 angles + 4 lifestyle; watermark-free, colorway-verified |
| 3 | Captions (local, Gemini 2.5 Pro) | DONE | scene-only, identifier-anchored; captions.jsonl (16/16) |
| 4 | Training scripts | DONE | setup.sh, run.sh, inference.py — now targeting klein 9B |
| 5 | Push to repo | DONE | e-manideep/product-imagery-ai |
| 6 | Train on GPU (RunPod) | TODO | L40S / A100 40GB / A6000 48GB, bf16, ~2500 steps |
| 7 | Inference + pick best checkpoint | TODO | 10 scenes (inference.py --examples), klein: 8 steps / g=1 |
| 8 | Review outputs | TODO | compare checkpoints, choose best |

## Best-output config (klein 9B)
- Base: FLUX.2-klein-9B, **bf16 (no FP8) for best quality** (small enough)
- Memory: offload + cache_latents + grad checkpointing + 8-bit Adam
- Identity anchor: "tjkzx Air Jordan 1 Chicago sneaker" (unique token + real name)
- rank 64, lora_alpha 64, lr 1e-4, constant scheduler, 2500 steps, resolution 1024
- guidance 1 (train) / guidance 1.0 + 8 steps (inference, distilled)
- validation every 20 epochs to pick the best checkpoint
- NO label-text / NO compositing (sneaker has no spelled label)

## Why klein 9B (vs dev 32B)
- Much smaller download (~18-29 GB vs ~120 GB) and far cheaper/faster GPUs
- Runs in bf16 on 24-48 GB cards; training ~1-2 hrs vs ~3-5 hrs
- Trade-off: less raw capacity than dev, but a rigid sneaker + name anchor +
  clean dataset make this very feasible. We compare checkpoints to pick the best.

## Locations
- Pipeline folder: C:\Users\MANID\product-imagery-ai\jordan\
- Training images: jordan\data\product\ (001-016.jpg)
- Original dataset: C:\Users\MANID\Downloads\aj1_chicago_dataset\
- Contact sheet: C:\Users\MANID\Downloads\aj1_chicago_contact_sheet.png

## Next action
On a RunPod L40S/A100 (PyTorch 2.8): clone, `bash setup.sh && bash run.sh`.
