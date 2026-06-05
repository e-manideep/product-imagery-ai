#!/usr/bin/env python3
"""
Run 004: add the bottle's real label text to every training caption.

Reads the scene-only captions (captions.jsonl) and appends an explicit
label-text clause, so the LoRA learns to associate the trigger word with that
exact text. At inference you then reinforce it with --label_text.

  python make_text_captions.py --in captions.jsonl --out captions_text.jsonl --label "Dior SAUVAGE"
"""

import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="src", default="captions.jsonl")
    parser.add_argument("--out", dest="dst", default="captions_text.jsonl")
    parser.add_argument("--label", default="Dior SAUVAGE",
                        help="Exact label text on the product")
    args = parser.parse_args()

    clause = f', the front label clearly reads "{args.label}" in sharp legible text'

    rows = []
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["prompt"] = rec["prompt"].rstrip(". ") + clause
            rows.append(rec)

    with open(args.dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} text-augmented captions to {args.dst}")
    print(f"Example: {rows[0]['prompt']}")


if __name__ == "__main__":
    main()
