#!/usr/bin/env python3
"""
Inference for the single-garment (navy leaf print shirt) FLUX.2 [klein] 9B LoRA.

The trigger "prtx" was trained to be this exact shirt, so reference it directly
in prompts (e.g. "prtx shirt"). klein is distilled: ~8 steps, guidance ~1.0.
"""

import os
import glob
import argparse
from pathlib import Path
from datetime import datetime

import torch

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

# Example scenes. {id} -> the identifier (e.g. "prtx shirt"). The dataset is
# on-model, so prompts place the shirt on a person.
EXAMPLES = [
    "a fashion e-commerce photo of {id} worn by a man, plain studio background, soft even lighting, sharp focus",
    "{id} worn by a man walking down a city street at golden hour, candid street style, shallow depth of field",
    "{id} worn by a man on a sunny beach, ocean behind, bright natural light, lifestyle photography",
    "{id} worn by a man sitting at a cafe table, warm morning light from a window, relaxed mood",
    "a full-length lookbook shot of {id} worn by a man against a pastel wall, editorial fashion photography",
    "{id} worn by a man in a green park, soft daylight, casual lifestyle shot",
    "close-up of {id} on a man's torso, showing the print and fabric detail, studio light",
    "{id} worn by a man on a rooftop at dusk, city lights bokeh behind, cinematic lifestyle photography",
]


def find_one(pattern, where, label):
    matches = sorted(glob.glob(os.path.join(where, pattern)))
    if not matches:
        raise FileNotFoundError(f"{label} not found ({pattern}) in {where}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", required=True, help="Training output dir with pytorch_lora_weights.safetensors")
    parser.add_argument("--output_dir", default="output/inference")
    parser.add_argument("--steps", type=int, default=8, help="klein is distilled; 4-8 is the sweet spot")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--identifier", default="prtx shirt", help="Trigger + class used in every prompt")
    parser.add_argument("--prompt", action="append", default=None,
                        help="Custom prompt (repeatable). Use {id} or write the identifier directly.")
    parser.add_argument("--prompts_file", default=None,
                        help="Text file of prompts, one per line (blank lines and # comments ignored). Use {id}.")
    parser.add_argument("--examples", action="store_true", help="Generate the built-in example shots.")
    parser.add_argument("--no_offload", action="store_true", help="Disable CPU offload (use only on large GPUs)")
    args = parser.parse_args()

    from diffusers import Flux2KleinPipeline

    print(f"Loading {MODEL_ID} (bf16)...")
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    if args.no_offload:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    lora_path = find_one("pytorch_lora_weights.safetensors", args.lora_dir, "LoRA weights")
    print(f"LoRA weights: {lora_path}")
    pipe.load_lora_weights(lora_path)

    idv = args.identifier
    if args.prompt:
        prompts = [p.replace("{id}", idv) for p in args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
        prompts = [ln.replace("{id}", idv) for ln in lines if ln and not ln.startswith("#")]
        if not prompts:
            raise ValueError(f"No prompts found in {args.prompts_file}")
    else:
        prompts = [e.format(id=idv) for e in EXAMPLES]

    n = len(prompts)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nGenerating {n} images (steps={args.steps}, guidance={args.guidance_scale})...\n")
    for i in range(n):
        prompt = prompts[i]
        print(f"[{i+1}/{n}] {prompt[:80]}...")
        image = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator("cpu").manual_seed(42 + i),
        ).images[0]
        path = out / f"{ts}_result_{i+1:02d}.png"
        image.save(path)
        print(f"  saved {path.name}")

    print(f"\nDone. {n} images in {out}")


if __name__ == "__main__":
    main()
