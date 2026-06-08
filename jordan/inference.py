#!/usr/bin/env python3
"""
Inference for the Air Jordan 1 "Chicago" FLUX.2 [klein] 9B DreamBooth LoRA.

Identity is anchored to a unique token + the real name
("tjkzx Air Jordan 1 Chicago sneaker"), used directly in prompts. FLUX.2 [klein]
is a distilled fast model: low guidance (~1.0) and a low step count (~4-8).
"""

import os
import glob
import argparse
from pathlib import Path
from datetime import datetime

import torch

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

# 10 ready-to-use example scenes. {id} -> the identifier (token + real name).
# For richer, campaign-grade prompts see prompts_sophisticated.txt (run with --prompts_file).
EXAMPLES = [
    "a product photo of {id} on a white marble podium, soft studio softbox lighting, subtle shadow, e-commerce hero shot, 85mm, sharp focus",
    "{id} floating against a clean pastel gradient background, dramatic studio lighting, premium sneaker advertisement, ultra detailed",
    "{id} on wet city asphalt at night, neon signs reflecting in puddles, cinematic street photography, shallow depth of field",
    "a top-down flat lay of {id} on a concrete floor surrounded by crumpled paper and a shoebox lid, editorial streetwear styling",
    "{id} on a rocky mountain ledge at golden hour, warm backlight, outdoor lifestyle photography, epic landscape behind",
    "a close-up of {id} worn on a person's feet walking down a sunny urban sidewalk, motion, candid street style",
    "{id} on a polished glass surface with blue and purple neon rim lighting, futuristic product ad, high contrast",
    "{id} resting on a basketball court hardwood floor under arena spotlights, dramatic shadows, sports editorial",
    "{id} on a sandy skatepark ledge with graffiti walls behind, bright daylight, youthful streetwear vibe",
    "{id} displayed on a minimalist wooden shelf in a boutique, soft window light, warm bokeh background, premium retail",
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
    parser.add_argument("--identifier", default="tjkzx Air Jordan 1 Chicago sneaker",
                        help="Token + real name phrase used in every prompt")
    parser.add_argument("--prompt", action="append", default=None,
                        help="Custom prompt (repeatable). Use {id} or write the identifier directly.")
    parser.add_argument("--examples", action="store_true", help="Generate the 10 built-in example shots.")
    parser.add_argument("--prompts_file", default=None,
                        help="Path to a text file of prompts, one per line (blank lines and # comments ignored). "
                             "Use {id} for the identifier.")
    parser.add_argument("--no_offload", action="store_true", help="Disable CPU offload (use only on very large GPUs)")
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
