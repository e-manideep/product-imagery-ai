#!/usr/bin/env python3
"""
Caption product images for DreamBooth training.
Writes a .txt caption file alongside each image.

Uses simple rule-based captions by default (no GPU needed).
Pass --use-blip to use BLIP model captioning instead.
"""

import os
import sys
import argparse
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Simple captions that rotate across images.
# Good enough for DreamBooth — the trigger word binding is what matters most.
CAPTION_TEMPLATES = [
    "a photo of {trigger} perfume bottle, product photography, studio lighting, high quality",
    "a photo of {trigger} perfume bottle, detailed product shot, professional photography",
    "a photo of {trigger} perfume bottle on a surface, product photography",
    "a photo of {trigger} perfume bottle, luxury fragrance, editorial photography",
    "a photo of {trigger} perfume bottle, close up detail, product shot",
    "a photo of {trigger} perfume bottle, lifestyle product photography",
    "a photo of {trigger} perfume bottle, high resolution product image",
    "a photo of {trigger} perfume bottle, professional studio photo",
    "a photo of {trigger} perfume bottle, commercial product photography",
    "a photo of {trigger} perfume bottle, elegant product shot",
]


def write_simple_captions(data_dir, trigger_word, overwrite=False):
    data_dir = Path(data_dir)
    images = sorted([f for f in data_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS])

    if not images:
        print(f"ERROR: no images found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(images)} images")
    print(f"Trigger word: '{trigger_word}'")
    print()

    captioned = 0
    skipped = 0

    for i, img_path in enumerate(images):
        txt_path = img_path.with_suffix(".txt")

        if txt_path.exists() and not overwrite:
            print(f"  [skip] {img_path.name}")
            skipped += 1
            continue

        template = CAPTION_TEMPLATES[i % len(CAPTION_TEMPLATES)]
        caption = template.format(trigger=trigger_word)
        txt_path.write_text(caption, encoding="utf-8")
        print(f"  [ok]  {img_path.name}")
        print(f"        {caption}")
        captioned += 1

    print(f"\nDone: {captioned} captioned, {skipped} skipped.")


def write_blip_captions(data_dir, trigger_word, overwrite=False):
    import torch
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration

    data_dir = Path(data_dir)
    images = sorted([f for f in data_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS])

    if not images:
        print(f"ERROR: no images found in {data_dir}")
        sys.exit(1)

    # Use CPU if CUDA is unavailable or busy
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            device = "cuda"
        except Exception:
            device = "cpu"
    else:
        device = "cpu"

    print(f"Device: {device}")
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("Loading BLIP captioner...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    print("BLIP loaded.\n")

    captioned = 0
    skipped = 0

    for img_path in images:
        txt_path = img_path.with_suffix(".txt")

        if txt_path.exists() and not overwrite:
            print(f"  [skip] {img_path.name}")
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device, dtype)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=80, num_beams=3)
            caption = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
            full_caption = f"{trigger_word} {caption}"
            txt_path.write_text(full_caption, encoding="utf-8")
            print(f"  [ok]  {img_path.name}: {full_caption[:90]}...")
            captioned += 1
        except Exception as e:
            print(f"  [err] {img_path.name}: {e}")

    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone: {captioned} captioned, {skipped} skipped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/product")
    parser.add_argument("--trigger_word", default="xtbll")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-blip", action="store_true", help="Use BLIP model (requires GPU)")
    args = parser.parse_args()

    if args.use_blip:
        write_blip_captions(args.data_dir, args.trigger_word, args.overwrite)
    else:
        write_simple_captions(args.data_dir, args.trigger_word, args.overwrite)


if __name__ == "__main__":
    main()
