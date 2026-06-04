#!/usr/bin/env python3
"""
Inference for the SDXL DoRA + Pivotal Tuning run.

Loads the trained DoRA weights AND the learned textual-inversion embeddings,
then generates the same 6 scenes used in Run 001 for a fair comparison.

The trained concept lives in the new tokens <s0><s1>..., so prompts reference
the product with that token string (not the raw trigger word).
"""

import os
import glob
import argparse
from pathlib import Path
from datetime import datetime

import torch
from safetensors.torch import load_file

# Same 6 scenes as Run 001 (FLUX). {tok} is replaced with the learned token string.
SCENES = [
    "a photo of {tok} on a white marble table, soft diffused natural light from a left window, subtle shadow, luxury product photography, 8K",
    "a photo of {tok} on a dark wooden surface, dramatic moody lighting, bokeh background, editorial product shot, high resolution",
    "a photo of {tok} outdoors, golden hour warm sunlight from behind, nature in the background, lifestyle product photography",
    "a photo of {tok} as a flat lay on white linen cloth, overhead view, even studio softbox lighting, minimalist e-commerce product photo",
    "a photo of {tok} close up with water droplets on the surface, macro shot, studio lighting, ultra detailed",
    "a photo of {tok} on black velvet, dramatic rim lighting from behind, dark luxury aesthetic, product photography",
]


def find_one(pattern, where, label):
    matches = sorted(glob.glob(os.path.join(where, pattern)))
    if not matches:
        raise FileNotFoundError(f"{label} not found ({pattern}) in {where}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir", required=True, help="Training output dir (DoRA weights + _emb.safetensors)")
    parser.add_argument("--output_dir", default="output/inference")
    parser.add_argument("--num_images", type=int, default=6)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args = parser.parse_args()

    from diffusers import DiffusionPipeline, AutoencoderKL

    print("Loading SDXL base + fp16-fix VAE...")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    lora_path = find_one("pytorch_lora_weights.safetensors", args.lora_dir, "LoRA weights")
    emb_path = find_one("*_emb.safetensors", args.lora_dir, "TI embeddings")
    print(f"DoRA weights: {lora_path}")
    print(f"Embeddings:   {emb_path}")

    pipe.load_lora_weights(lora_path)

    # Load the learned tokens. Number of tokens is inferred from the embedding rows.
    state_dict = load_file(emb_path)
    num_tokens = state_dict["clip_l"].shape[0]
    tokens = [f"<s{i}>" for i in range(num_tokens)]
    tok_str = "".join(tokens)
    print(f"Learned tokens: {tok_str}")

    pipe.load_textual_inversion(
        state_dict["clip_l"], token=tokens,
        text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer,
    )
    pipe.load_textual_inversion(
        state_dict["clip_g"], token=tokens,
        text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n = min(args.num_images, len(SCENES))

    print(f"\nGenerating {n} images...\n")
    for i in range(n):
        prompt = SCENES[i].format(tok=tok_str)
        print(f"[{i+1}/{n}] {prompt[:80]}...")
        image = pipe(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator("cuda").manual_seed(42 + i),
        ).images[0]
        path = out / f"{ts}_result_{i+1:02d}.png"
        image.save(path)
        print(f"  saved {path.name}")

    print(f"\nDone. {n} images in {out}")


if __name__ == "__main__":
    main()
