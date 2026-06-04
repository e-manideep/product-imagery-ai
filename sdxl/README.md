# SDXL DoRA + Pivotal Tuning (Run 002)

Second iteration of the product imagery pipeline. Fixes the core issues from Run 001
(FLUX + plain LoRA + generic captions) with three changes that actually move the needle:

1. **GPT-4o captioning** — describes only the scene, never the bottle. The product's
   look is captured by the trained token, not leaked into text. This was the main Run 001 failure.
2. **DoRA** instead of plain LoRA — better object-identity binding at equal rank.
3. **Pivotal Tuning** — adds dedicated new tokens (`<s0><s1>`) to the tokenizer and learns
   them as embeddings, so the trigger is a clean single concept instead of a token that
   gets split by the tokenizer.

Base model is SDXL because it is the one well-established stack that supports DoRA,
pivotal tuning, and clean token abstraction in a single maintained training script.

## Run it

```bash
cd sdxl
export HF_TOKEN=hf_xxx
export OPENAI_API_KEY=sk-xxx
bash setup.sh
bash run.sh
```

Source images are read from `../data/product/` (the same 20 used in Run 001).

## What changed vs Run 001

| | Run 001 | Run 002 |
|---|---|---|
| Base model | FLUX.1-dev | SDXL 1.0 |
| Adapter | LoRA | DoRA |
| Token | `xtbll` (raw) | Pivotal tuning `<s0><s1>` |
| Captions | Generic templates describing the bottle | GPT-4o, scene-only, token for the bottle |
| Rank | 16 | 32 |
| Steps | 1000 | 2000 |
| Optimizer | AdamW 8-bit | AdamW |
| Text-encoder LR | n/a | 5e-6 |

## Config (env overrides)

| Variable | Default | Notes |
|---|---|---|
| `TRIGGER_WORD` | `xtbll` | Concept abstraction, swapped for `<s0><s1>` in training |
| `RANK` | `32` | DoRA rank |
| `MAX_STEPS` | `2000` | Training steps |
| `LEARNING_RATE` | `1e-4` | UNet DoRA + token embeddings |
| `TEXT_ENCODER_LR` | `5e-6` | Text encoder rate |
| `TI_TOKENS` | `2` | New tokens per concept |
| `TI_FRAC` | `0.5` | Fraction of steps that train the TI embeddings |

## Output

```
output/product_dora/
├── pytorch_lora_weights.safetensors    # DoRA weights
└── product_dora_emb.safetensors        # learned <s0><s1> embeddings
output/inference/                       # 6 test images
```

Inference loads BOTH the DoRA weights and the embeddings, then prompts with `<s0><s1>`.
