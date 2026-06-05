#!/usr/bin/env python3
"""
Build the training dataset folder from source images + a committed captions.jsonl.
No API key needed -- this runs on the GPU pod.

Produces:
  <out_dir>/001.jpg, 002.jpg, ...        (canonical-named copies, sorted order)
  <out_dir>/metadata.jsonl               (file_name + prompt, for --caption_column)
"""

import os
import sys
import json
import argparse
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--captions", required=True, help="captions.jsonl")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    images = sorted([f for f in src.iterdir() if f.suffix.lower() in SUPPORTED_EXTS])
    if not images:
        print(f"ERROR: no images in {src}")
        sys.exit(1)

    captions = {}
    with open(args.captions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                captions[rec["file_name"]] = rec["prompt"]

    if len(captions) != len(images):
        print(f"WARNING: {len(images)} images but {len(captions)} captions. Matching by sorted order.")

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):
        old.unlink()

    records = []
    for i, img in enumerate(images, 1):
        canonical = f"{i:03d}{img.suffix.lower()}"
        (out / canonical).write_bytes(img.read_bytes())
        prompt = captions.get(canonical)
        if prompt is None:
            # fall back to positional match if names drifted
            prompt = list(captions.values())[i - 1] if i - 1 < len(captions) else "a photo"
        records.append({"file_name": canonical, "prompt": prompt})

    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Built dataset: {len(records)} images + captions in {out}")


if __name__ == "__main__":
    main()
