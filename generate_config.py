#!/usr/bin/env python3
"""
Generate ai-toolkit training config YAML from environment variables.
All paths are resolved to absolute to avoid working-directory issues.
"""

import os
import sys
import yaml
from pathlib import Path


def generate():
    base_dir = Path(
        os.environ.get("BASE_DIR", Path(__file__).parent.resolve())
    ).resolve()

    run_name = os.environ.get("RUN_NAME", "product_lora")
    trigger_word = os.environ.get("TRIGGER_WORD", "xtbll")
    lora_rank = int(os.environ.get("LORA_RANK", "16"))
    train_steps = int(os.environ.get("TRAIN_STEPS", "1000"))
    lr = float(os.environ.get("LR", "1e-4"))
    quantize = os.environ.get("QUANTIZE", "false").lower() == "true"
    sample_every = int(os.environ.get("SAMPLE_EVERY", "250"))

    data_dir = Path(
        os.environ.get("DATA_DIR", str(base_dir / "data" / "product"))
    ).resolve()

    output_dir = str(base_dir / "output")

    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_count = sum(1 for f in data_dir.iterdir() if f.suffix.lower() in image_exts)
    if image_count == 0:
        print(f"ERROR: no images found in {data_dir}")
        sys.exit(1)

    config = {
        "job": "extension",
        "config": {
            "name": run_name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": output_dir,
                    "device": "cuda:0",
                    "trigger_word": trigger_word,
                    "network": {
                        "type": "lora",
                        "linear": lora_rank,
                        "linear_alpha": lora_rank,
                        "use_dora": True,
                    },
                    "save": {
                        "dtype": "float16",
                        "save_every": sample_every,
                        "max_step_saves_to_keep": 4,
                    },
                    "datasets": [
                        {
                            "folder_path": str(data_dir),
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": False,
                            "resolution": [512, 768, 1024],
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": train_steps,
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": lr,
                        "lr_scheduler": "cosine",
                        "lr_warmup_steps": 100,
                        "dtype": "bf16",
                        "ema_config": {
                            "use_ema": True,
                            "ema_decay": 0.99,
                        },
                    },
                    "model": {
                        "name_or_path": "black-forest-labs/FLUX.1-dev",
                        "is_flux": True,
                        "quantize": quantize,
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": sample_every,
                        "width": 1024,
                        "height": 1024,
                        "prompts": [
                            "photo of [trigger] perfume bottle on white marble table, soft natural window light, luxury product photography",
                            "photo of [trigger] perfume bottle on dark wooden surface, dramatic moody lighting, editorial shot",
                            "photo of [trigger] perfume bottle outdoors, golden hour light, lifestyle product photography",
                            "photo of [trigger] perfume bottle overhead flat lay, white linen, minimalist e-commerce photo",
                        ],
                        "neg": "",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 4,
                        "sample_steps": 20,
                    },
                }
            ],
        },
        "meta": {
            "name": "[name]",
            "version": "1.0",
        },
    }

    config_path = base_dir / "configs" / "active_config.yaml"
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Config written: {config_path}")
    print(f"  run_name:     {run_name}")
    print(f"  trigger_word: {trigger_word}")
    print(f"  lora_rank:    {lora_rank}")
    print(f"  train_steps:  {train_steps}")
    print(f"  data_dir:     {data_dir}  ({image_count} images)")
    print(f"  output_dir:   {output_dir}")
    print(f"  quantize:     {quantize}  (set QUANTIZE=true for 40GB GPUs)")

    return str(config_path)


if __name__ == "__main__":
    generate()
