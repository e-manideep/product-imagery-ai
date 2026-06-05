#!/usr/bin/env python3
"""
Inference for the FLUX.2 klein 9B DreamBooth LoRA run.

Plain LoRA (no pivotal tuning), so the product is referenced directly by the
trigger word (e.g. "xtbll") in prompts. klein is a distilled fast model:
guidance_scale ~1.0 and a low step count (~4-8).
"""

import os
import glob
import argparse
from pathlib import Path
from datetime import datetime

import torch

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

# Default 6 scenes (parity with earlier runs). {tok} -> trigger word.
SCENES = [
    "a photo of {tok} on a white marble table, soft diffused natural light from a left window, subtle shadow, luxury product photography",
    "a photo of {tok} on a dark wooden surface, dramatic moody lighting, bokeh background, editorial product shot",
    "a photo of {tok} outdoors, golden hour warm sunlight from behind, nature in the background, lifestyle product photography",
    "a photo of {tok} as a flat lay on white linen cloth, overhead view, even studio softbox lighting, minimalist e-commerce product photo",
    "a photo of {tok} close up with water droplets on the surface, macro shot, studio lighting, ultra detailed",
    "a photo of {tok} on black velvet, dramatic rim lighting from behind, dark luxury aesthetic, product photography",
]

# 10 ready-to-use examples. The product is the explicit subject so it always shows.
EXAMPLES = [
    "a product photo of a bottle of {tok} on a white marble countertop, soft window light, tiny water droplets, e-commerce photography, 85mm, sharp focus",
    "close-up of a man's hand holding a bottle of {tok}, the bottle in sharp focus, blurred city street at golden hour, cinematic depth of field",
    "a bottle of {tok} on a wet black stone with splashing water, dramatic studio lighting, dark luxury fragrance ad, ultra detailed",
    "a bottle of {tok} on a sandy beach at sunset, ocean waves in the background, warm golden light, lifestyle product photography",
    "a top-down flat lay of a bottle of {tok} surrounded by green leaves and sliced citrus, fresh natural concept, bright daylight",
    "a bottle of {tok} on a wooden bathroom shelf next to a folded white towel, soft morning light, minimalist interior",
    "a bottle of {tok} on a reflective glass surface with neon blue and purple lighting, modern futuristic ad, high contrast",
    "a bottle of {tok} held in a woman's hand with an elegant manicure, soft pink background, beauty product photography, studio light",
    "a bottle of {tok} on a snowy surface with pine branches, cool winter tones, festive luxury gift concept",
    "a bottle of {tok} on a marble pedestal in a luxury boutique, spotlight, soft bokeh background, premium product display",
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
    parser.add_argument("--num_images", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8, help="klein is distilled; 4-8 is the sweet spot")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--trigger_word", default="xtbll")
    parser.add_argument("--prompt", action="append", default=None,
                        help="Custom prompt (repeatable). Use the trigger word for the product.")
    parser.add_argument("--examples", action="store_true", help="Generate the 10 built-in example shots.")
    parser.add_argument("--label_text", default=None,
                        help="If set, the exact label text to render, e.g. \"Dior SAUVAGE\". "
                             "A legible-text clause is appended to every prompt.")
    parser.add_argument("--no_offload", action="store_true", help="Disable CPU offload (use only on 80GB+ GPUs)")
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

    tok = args.trigger_word
    if args.prompt:
        prompts = list(args.prompt)
    elif args.examples:
        prompts = [e.format(tok=tok) for e in EXAMPLES]
    else:
        prompts = [s.format(tok=tok) for s in SCENES[: args.num_images]]

    # Run 004: explicitly tell the model the exact label text to render.
    if args.label_text:
        clause = f', the front label clearly reads "{args.label_text}" in sharp, legible, correctly-spelled lettering'
        prompts = [p + clause for p in prompts]

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
