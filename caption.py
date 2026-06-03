#!/usr/bin/env python3
"""
Auto-caption product images using Florence-2-large.
Writes a .txt file alongside each image with the trigger word prepended.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model(device):
    print("Loading Florence-2-large captioner...")
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    print("Florence-2 loaded.")
    return processor, model


def caption_single(image_path, processor, model, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text="<MORE_DETAILED_CAPTION>",
        images=image,
        return_tensors="pt",
    ).to(device, torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=256,
            do_sample=False,
            num_beams=3,
        )

    raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        raw,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(image.width, image.height),
    )
    return result["<MORE_DETAILED_CAPTION>"].strip()


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

    # Free GPU memory before training starts
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nDone: {captioned} captioned, {skipped} skipped.")


if __name__ == "__main__":
    main()
