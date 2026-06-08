#!/usr/bin/env python3
"""
Generate scene-only training captions with a vision model.

Run this ONCE locally. It writes captions.jsonl, which is committed so the GPU
pod never needs an API key.

Backend:
  - OpenRouter if OPENROUTER_API_KEY is set (base_url = openrouter.ai/api/v1)
  - OpenAI    if OPENAI_API_KEY is set

Captioning rule:
  - Refer to the product ONLY by the fixed identifier (token + real name), e.g.
    "tjkzx Air Jordan 1 Chicago sneaker".
  - NEVER describe the product's colorway, materials, logo, or stitching. Those
    are carried by the identifier + the images, not the caption.
  - DO describe everything else: surface, background, lighting, setting,
    composition, any other objects.
"""

import os
import sys
import json
import base64
import argparse
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

SYSTEM_RULES = """You write training captions for a DreamBooth fine-tune of one specific product.

The product is a sneaker that MUST be referred to ONLY by the exact identifier "{identifier}".

Hard rules:
- Refer to the shoe as exactly "{identifier}". Never call it just a "sneaker", "shoe", "trainer", or anything else descriptive.
- DO NOT describe the shoe itself: not its colorway, colors, materials, leather, laces, sole, swoosh, logo, branding, or any text on it. Those are learned by the identifier and the images.
- DO describe the rest of the scene: the surface it sits on, the background, the lighting, the setting/mood, any other objects, and the camera angle or composition.
- One sentence. Begin the sentence with "{identifier}". Maximum 30 words. No lists, no quotes."""


def make_client():
    from openai import OpenAI

    if os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        return client, "google/gemini-2.5-pro"
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(), "gpt-4o"

    print("ERROR: set OPENROUTER_API_KEY or OPENAI_API_KEY.")
    sys.exit(1)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def caption_one(client, model, image_path, identifier, retries=3):
    b64 = encode_image(image_path)
    ext = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_RULES.format(identifier=identifier)},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Write the training caption for this image."},
                            {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                        ],
                    },
                ],
                max_tokens=2000,
                temperature=0.4,
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                caption = [ln for ln in content.strip().splitlines() if ln.strip()][-1]
                caption = caption.strip().strip('"').strip()
                break
            last_err = "empty response from model"
        except Exception as e:
            last_err = str(e)
    else:
        raise ValueError(last_err or "captioning failed")

    if identifier.split()[0] not in caption:
        caption = f"{identifier} {caption}"
    if not caption.lower().startswith(identifier.split()[0].lower()):
        caption = f"{identifier}, {caption}"
    return caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="Folder of source product images")
    parser.add_argument("--out", required=True, help="captions.jsonl to write")
    parser.add_argument("--identifier", default="tjkzx Air Jordan 1 Chicago sneaker",
                        help="Fixed token+name phrase that every caption must start with")
    parser.add_argument("--model", default=None, help="Override the vision model")
    args = parser.parse_args()

    src = Path(args.src_dir)
    images = sorted([f for f in src.iterdir() if f.suffix.lower() in SUPPORTED_EXTS])
    if not images:
        print(f"ERROR: no images in {src}")
        sys.exit(1)

    client, default_model = make_client()
    model = args.model or default_model

    print(f"Captioning {len(images)} images with {model}")
    print(f"Identifier: '{args.identifier}'\n")

    records = []
    for i, img in enumerate(images, 1):
        canonical = f"{i:03d}{img.suffix.lower()}"
        try:
            caption = caption_one(client, model, img, args.identifier)
        except Exception as e:
            print(f"  [err] {img.name}: {e}")
            caption = f"{args.identifier} on a plain surface, soft studio lighting"
        records.append({"file_name": canonical, "prompt": caption})
        print(f"  [{i:02d}/{len(images)}] {caption}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} captions to {args.out}")


if __name__ == "__main__":
    main()
