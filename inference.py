#!/usr/bin/env python3
"""
Run inference with a trained product LoRA on FLUX.1-dev.
Generates test images across 6 different product photography scenes.
"""

import os
import sys
import argparse
import glob
from pathlib import Path
from datetime import datetime

import torch


PROMPT_TEMPLATES = [
    "photo of {trigger} perfume bottle on a white marble table, soft diffused natural light from left window, subtle shadow, Canon EOS R5 85mm f/2.8, luxury product photography, 8K",
    "photo of {trigger} perfume bottle on dark wooden surface, dramatic moody lighting, bokeh background, editorial product shot, high resolution",
    "photo of {trigger} perfume bottle outdoors, golden hour warm sunlight from behind, nature in background, lifestyle product photography",
    "photo of {trigger} perfume bottle flat lay on white linen cloth, overhead view, even studio softbox lighting, minimalist e-commerce product photo",
    "photo of {trigger} perfume bottle close-up detail, water droplets on glass surface, macro shot, studio lighting, ultra detailed",
    "photo of {trigger} perfume bottle on black velvet surface, dramatic rim lighting from behind, dark luxury aesthetic, product photography",
]


def find_latest_lora(output_dir, run_name):
    patterns = [
        os.path.join(output_dir, run_name, "*.safetensors"),
        os.path.join(output_dir, "*.safetensors"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--trigger_word", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_images", type=int, default=6)
    parser.add_argument("--lora_scale", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    args = parser.parse_args()

    base_dir = Path(
        os.environ.get("BASE_DIR", Path(__file__).parent.resolve())
    ).resolve()
    run_name = os.environ.get("RUN_NAME", "product_lora")
    trigger_word = args.trigger_word or os.environ.get("TRIGGER_WORD", "xtbll")
    output_dir = Path(args.output_dir or base_dir / "output" / "inference")

    lora_path = args.lora_path
    if lora_path is None:
        lora_path = find_latest_lora(str(base_dir / "output"), run_name)
        if lora_path is None:
            print("ERROR: no .safetensors LoRA found.")
            print(f"  Looked in: {base_dir}/output/{run_name}/")
            print("  Either training hasn't finished or pass --lora_path explicitly.")
            sys.exit(1)
        print(f"Auto-detected LoRA: {lora_path}")

    from diffusers import FluxPipeline

    print("Loading FLUX.1-dev base model (this takes ~2 min first time)...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA (scale={args.lora_scale})...")
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.to("cuda")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num = min(args.num_images, len(PROMPT_TEMPLATES))

    print(f"\nGenerating {num} test images...")
    print(f"Trigger word: '{trigger_word}'")
    print(f"Output: {output_dir}\n")

    for i, template in enumerate(PROMPT_TEMPLATES[:num]):
        prompt = template.format(trigger=trigger_word)
        print(f"[{i+1}/{num}] {prompt[:90]}...")

        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=1024,
                width=1024,
                generator=torch.Generator("cuda").manual_seed(42 + i),
            ).images[0]

        out_path = output_dir / f"{timestamp}_result_{i+1:02d}.png"
        image.save(out_path)
        print(f"  Saved: {out_path.name}")

    print(f"\n✓ Done — {num} images saved to {output_dir}")
    print(f"\nHow to evaluate:")
    print(f"  1. Does the bottle shape match your training images?")
    print(f"  2. Is the label/logo preserved?")
    print(f"  3. Does the product change correctly across different scenes?")
    print(f"\nIf product identity is weak: increase LORA_RANK=32 or TRAIN_STEPS=1500 and retrain.")
    print(f"If overfitting (looks exactly like training image): reduce lora_scale to 0.6 or reduce steps.")


if __name__ == "__main__":
    main()
