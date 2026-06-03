#!/usr/bin/env python3
"""
Auto-caption product images using BLIP (blip-image-captioning-large).
Writes a .txt file alongside each image with the trigger word prepended.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model(device):
    print("Loading BLIP captioner...")
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    print("BLIP loaded.")
    return processor, model


def caption_single(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            min_length=10,
        )

    caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
    return caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/product")
    parser.add_argument("--trigger_word", default="xtbll")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing caption files"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    images = sorted(
        [f for f in data_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
    )
    if not images:
        print(f"ERROR: no images found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images")
    print(f"Trigger word: '{args.trigger_word}'")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_model(device)

    captioned = 0
    skipped = 0

    for img_path in images:
        txt_path = img_path.with_suffix(".txt")

        if txt_path.exists() and not args.overwrite:
            print(f"  [skip] {img_path.name}")
            skipped += 1
            continue

        try:
            caption = caption_single(img_path, processor, model, device)
            full_caption = f"{args.trigger_word} {caption}"
            txt_path.write_text(full_caption, encoding="utf-8")
            print(f"  [ok]  {img_path.name}")
            print(f"        {full_caption[:100]}...")
            captioned += 1
        except Exception as e:
            print(f"  [err] {img_path.name}: {e}")

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone: {captioned} captioned, {skipped} skipped.")


if __name__ == "__main__":
    main()
