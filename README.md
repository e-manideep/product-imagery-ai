# Product Imagery AI

Fine-tune FLUX.1-dev on any physical product using DreamBooth + DoRA LoRA.  
Give it 10–20 product photos → get photorealistic images of that product in any scene.

**Stack:** FLUX.1-dev · DreamBooth LoRA · DoRA · Florence-2 · ostris/ai-toolkit · RunPod A100

---

## How it works

1. You upload product photos (10–20 images, multiple angles)
2. Florence-2 auto-captions them with your trigger word
3. FLUX.1-dev is fine-tuned via DoRA LoRA to bind the product to that trigger word
4. You prompt: `"photo of xtbll perfume bottle on a marble table"` → gets your exact bottle

---

## RunPod Setup

### 1. Launch a pod

- Template: search **"ai-toolkit"** (ostris official) in RunPod templates
- GPU: **A100 SXM 80GB** (spot, ~$1.89/hr)
- Disk: **50 GB** minimum (FLUX.1-dev base model = ~23 GB)

### 2. Clone this repo

```bash
cd /workspace
git clone https://github.com/e-manideep/product-imagery-ai.git
cd product-imagery-ai
```

### 3. Run setup

```bash
export HF_TOKEN=your_huggingface_token   # needs FLUX.1-dev access
bash setup.sh
```

> Get your HF token at https://huggingface.co/settings/tokens  
> Accept FLUX.1-dev license at https://huggingface.co/black-forest-labs/FLUX.1-dev

### 4. Upload your product images

Upload 10–20 photos of your product to `data/product/`.  
Use RunPod's file manager, or:

```bash
# From your local machine:
rsync -avz ./your_images/ root@<pod-ip>:/workspace/product-imagery-ai/data/product/
```

### 5. Run the full pipeline

```bash
bash run_all.sh
```

That's it. The script will:
- Caption all images automatically
- Generate the training config
- Train for ~45 min
- Run 6 test inference images

---

## Configuration

Override any setting by exporting before `run_all.sh`:

| Variable | Default | Description |
|---|---|---|
| `TRIGGER_WORD` | `xtbll` | Unique token bound to your product |
| `TRAIN_STEPS` | `1000` | Training steps (use 1500 for < 10 images) |
| `LORA_RANK` | `16` | LoRA rank (try 32 if product not captured) |
| `RUN_NAME` | `product_lora` | Name for this training run |
| `LR` | `1e-4` | Learning rate |
| `QUANTIZE` | `false` | Set `true` for 40GB GPUs |

Example:
```bash
export TRIGGER_WORD=zrvpx
export TRAIN_STEPS=1500
export LORA_RANK=32
bash run_all.sh
```

---

## Output

After training, you get:

```
output/
├── product_lora/
│   ├── product_lora_000250.safetensors   ← checkpoint at step 250
│   ├── product_lora_000500.safetensors
│   ├── product_lora_000750.safetensors
│   ├── product_lora_001000.safetensors   ← final (use this)
│   └── samples/                          ← auto-generated previews during training
└── inference/
    ├── 20250603_120000_result_01.png     ← marble table scene
    ├── 20250603_120000_result_02.png     ← dark wooden surface
    ├── 20250603_120000_result_03.png     ← outdoor golden hour
    ├── 20250603_120000_result_04.png     ← flat lay overhead
    ├── 20250603_120000_result_05.png     ← macro detail
    └── 20250603_120000_result_06.png     ← dark luxury
```

---

## Testing / Evaluation

Check the 6 test images and ask:

1. **Product identity preserved?** Does the bottle shape, color, label match your input photos?
2. **Scene following?** Does the background/lighting change correctly across prompts?
3. **Quality?** Sharp, photorealistic, no artifacts?

### Tuning if results are off

| Problem | Fix |
|---|---|
| Product not recognized | Increase `LORA_RANK=32`, `TRAIN_STEPS=1500` |
| Product bleeds into everything | Reduce steps to 750, lower `lora_scale` to 0.6 in inference.py |
| Wrong product shape | Add more diverse training images (different angles) |
| Blurry output | Not a LoRA issue — check inference `--steps 28` and `guidance_scale 3.5` |

### Running inference manually with different prompts

```bash
python inference.py \
  --lora_path output/product_lora/product_lora_001000.safetensors \
  --trigger_word xtbll \
  --lora_scale 0.8
```

---

## Running steps individually

```bash
# Caption only
python caption.py --data_dir data/product --trigger_word xtbll

# Generate config only
python generate_config.py

# Train only (after config is generated)
cd ai-toolkit && python run.py ../configs/active_config.yaml && cd ..

# Inference only
python inference.py --lora_path output/product_lora/product_lora_001000.safetensors
```

---

## Cost estimate (RunPod)

| Run | GPU | Time | Cost |
|---|---|---|---|
| First run (1000 steps) | A100 80GB spot | ~45 min | ~$1.50 |
| With re-runs (3 iterations) | A100 80GB spot | ~2.5 hrs | ~$5 |

Download your `.safetensors` and **stop the pod immediately** after training to avoid idle billing.
