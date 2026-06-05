# FLUX.2 klein 9B — DreamBooth LoRA (Run 003)

Third iteration. Swaps the base model from SDXL to **FLUX.2 klein 9B** for much
stronger native text rendering (the SDXL labels came out garbled) and higher
overall image quality. Trained in **bf16 (no fp8)** for maximum quality.

Trade-offs vs the SDXL run: this is plain LoRA — **no DoRA, no pivotal tuning**
(the klein training script doesn't implement them). The stronger base model is
expected to carry product identity from plain LoRA + the same scene-only captions.

## Requirements

- **GPU:** A100 80GB or H100 80GB recommended (bf16, rank 32, no resource limit).
  Minimum RTX 4090 24GB with smaller rank.
- **Pod:** a **PyTorch 2.8 / CUDA 12.4+** template. FLUX.2 needs recent
  transformers that require torch >= 2.7 — the torch 2.4 pods will crash on import.
- **HuggingFace:** klein 9B is **gated**. Accept the license once at
  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B and use an HF token.

## Run it

```bash
cd /workspace
git clone https://github.com/e-manideep/product-imagery-ai.git
cd product-imagery-ai/flux2
export HF_TOKEN=hf_xxx
bash setup.sh
bash run.sh
```

Source images come from `../data/product/` and captions from `captions.jsonl`
(the same scene-only captions used in Run 002).

## Config (env overrides)

| Variable | Default | Notes |
|---|---|---|
| `TRIGGER_WORD` | `xtbll` | Trigger word used literally in prompts |
| `RANK` | `32` | LoRA rank |
| `MAX_STEPS` | `1500` | Training steps |
| `LEARNING_RATE` | `1e-4` | |
| `RESOLUTION` | `1024` | |
| `GRAD_ACCUM` | `2` | Effective batch = 2 |

## Output

```
output/product_flux2/pytorch_lora_weights.safetensors   # trained LoRA
output/inference/                                        # 10 example shots
```

## Inference (custom prompts)

Plain LoRA, so just use the trigger word directly in prompts:

```bash
python inference.py --lora_dir output/product_flux2 --output_dir output/mine \
  --steps 8 --guidance_scale 1.0 \
  --prompt "a bottle of xtbll on a cafe table next to a coffee cup, morning light" \
  --prompt "close-up of a man's hand holding a bottle of xtbll, blurred street, golden hour"
```

Notes:
- klein is a distilled fast model — `--steps 4` to `8`, `--guidance_scale 1.0`.
- On an 80GB GPU add `--no_offload` for faster generation.
