#!/usr/bin/env python3
"""
Generate scene-only training captions with a vision model.

Run this ONCE (locally or anywhere with a key). It writes captions.jsonl, which
is committed to the repo so the GPU pod never needs an API key.

Backend:
  - OpenRouter if OPENROUTER_API_KEY is set (base_url = openrouter.ai/api/v1)
  - OpenAI    if OPENAI_API_KEY is set

The captioning rule (the whole point of Run 002):
  - Refer to the product ONLY by the trigger token (e.g. "xtbll").
  - NEVER describe the product's color, shape, material, cap, label, or text.
    Those attributes are learned by the token, not the caption.
  - DO describe everything else: surface, background, lighting, setting, composition.
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


def make_client():
    """Return (client, default_model) for whichever backend has a key."""
    from openai import OpenAI

    if os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        return client, "openai/gpt-4o"
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAI(), "gpt-4o"

    print("ERROR: set OPENROUTER_API_KEY or OPENAI_API_KEY.")
    sys.exit(1)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def caption_one(client, model, image_path, trigger, retries=3):
    b64 = encode_image(image_path)
    ext = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_RULES.format(trigger=trigger)},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Write the training caption for this image."},
                            {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
                        ],
                    },
                ],
                # Large budget so reasoning models (Gemini 2.5/3.x) finish thinking
                # and still emit the caption -- reasoning counts against max_tokens.
                max_tokens=2000,
                temperature=0.4,
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                # Last non-empty line, in case any preamble slips through.
                caption = [ln for ln in content.strip().splitlines() if ln.strip()][-1]
                caption = caption.strip().strip('"').strip()
                break
            last_err = "empty response from model"
        except Exception as e:
            last_err = str(e)
    else:
        raise ValueError(last_err or "captioning failed")

    # Guarantee the trigger token leads the caption.
    if trigger not in caption:
        caption = f"{trigger} {caption}"
    if not caption.lower().startswith(trigger):
        caption = f"{trigger}, {caption}"
    return caption


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="Folder of source product images")
    parser.add_argument("--out", required=True, help="captions.jsonl to write")
    parser.add_argument("--trigger_word", default="xtbll")
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
    print(f"Trigger token: '{args.trigger_word}'\n")

    records = []
    for i, img in enumerate(images, 1):
        canonical = f"{i:03d}{img.suffix.lower()}"
        try:
            caption = caption_one(client, model, img, args.trigger_word)
        except Exception as e:
            print(f"  [err] {img.name}: {e}")
            caption = f"{args.trigger_word} on a plain surface, soft studio lighting"
        records.append({"file_name": canonical, "prompt": caption})
        print(f"  [{i:02d}/{len(images)}] {caption}")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(records)} captions to {args.out}")


if __name__ == "__main__":
    main()
