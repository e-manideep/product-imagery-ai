#!/usr/bin/env python3
"""
Caption product images with GPT-4o vision for pivotal-tuning DreamBooth.

The captioning rule (this is the whole point):
  - Refer to the product ONLY by the trigger token (e.g. "xtbll").
  - NEVER describe the product's color, shape, material, cap, label, or text.
    Those attributes must be captured by the trained token, not the caption.
  - DO describe everything else: surface, background, lighting, setting,
    other objects, composition.

Builds a clean image folder with a metadata.jsonl that the SDXL advanced
training script consumes via --dataset_name + --caption_column.
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

SYSTEM_RULES = """You write training captions for a DreamBooth fine-tune of a single specific product.

The product is a perfume bottle that MUST be referred to ONLY by the exact token "{trigger}" (lowercase).

Hard rules:
- Refer to the bottle as exactly "{trigger}". Never call it a "perfume bottle", "bottle", "fragrance", or anything descriptive.
- DO NOT describe the bottle itself: not its color, shape, size, material, glass, cap, liquid, label, branding, or any text on it. Those are learned by the token.
- DO describe the rest of the scene: the surface it sits on, the background, the lighting, the setting/mood, any other objects, and the camera angle or composition.
- One sentence. Begin the sentence with "{trigger}". Maximum 25 words. No lists, no quotes."""


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def caption_one(client, image_path, trigger, model):
    b64 = encode_image(image_path)
    ext = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_RULES.format(trigger=trigger)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Write the training caption for this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                    },
                ],
            },
        ],
        max_tokens=80,
        temperature=0.4,
    )
    caption = resp.choices[0].message.content.strip().strip('"')

    # Safety net: guarantee the trigger token is present and leads the caption.
    if trigger not in caption:
        caption = f"{trigger} {caption}"
    if not caption.lower().startswith(trigger):
        caption = f"{trigger}, {caption}"
    return caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="Folder of source product images")
    parser.add_argument("--out_dir", required=True, help="Clean folder to build (images + metadata.jsonl)")
    parser.add_argument("--trigger_word", default="xtbll")
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package missing. Run: pip install openai")
        sys.exit(1)

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    images = sorted([f for f in src.iterdir() if f.suffix.lower() in SUPPORTED_EXTS])
    if not images:
        print(f"ERROR: no images in {src}")
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    # Clear any stale metadata so a re-run is clean
    meta_path = out / "metadata.jsonl"
    if meta_path.exists():
        meta_path.unlink()

    client = OpenAI()
    print(f"Captioning {len(images)} images with {args.model}")
    print(f"Trigger token: '{args.trigger_word}'\n")

    records = []
    for i, img in enumerate(images, 1):
        dst_name = f"{i:03d}{img.suffix.lower()}"
        dst = out / dst_name
        dst.write_bytes(img.read_bytes())

        try:
            caption = caption_one(client, img, args.trigger_word, args.model)
        except Exception as e:
            print(f"  [err] {img.name}: {e}")
            caption = f"{args.trigger_word} on a plain surface, soft studio lighting"

        records.append({"file_name": dst_name, "prompt": caption})
        print(f"  [{i:02d}/{len(images)}] {caption}")

    with open(meta_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(records)} captions to {meta_path}")


if __name__ == "__main__":
    main()
