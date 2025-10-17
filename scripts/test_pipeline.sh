#!/bin/bash
# Quick test of the evaluation pipeline with 500 samples

set -e

RAE_DIR="/opt/tiger/RAE"
cd "${RAE_DIR}" || exit 1

# Activate venv
source .venv/bin/activate

echo "=========================================================================="
echo "Testing RAE Evaluation Pipeline (500 samples)"
echo "=========================================================================="
echo ""

# Test: Download small subset
echo "Step 1: Downloading 500 validation samples..."
python3 << 'EOF'
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

VAL_DIR = Path("/opt/tiger/RAE/data/imagenet_val_test")
NUM_SAMPLES = 500

VAL_DIR.mkdir(parents=True, exist_ok=True)

print("Loading validation set...")
dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=False)

print(f"Downloading {NUM_SAMPLES} samples...")
for idx, item in enumerate(tqdm(dataset, total=NUM_SAMPLES)):
    if idx >= NUM_SAMPLES:
        break

    image = item['image']
    label = item['label']

    class_dir = VAL_DIR / str(label)
    class_dir.mkdir(exist_ok=True)

    image_path = class_dir / f"{idx:06d}.JPEG"
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(image_path, 'JPEG', quality=95)

print(f"Downloaded {NUM_SAMPLES} images to {VAL_DIR}")
EOF

echo ""
echo "Step 2: Reconstructing samples..."
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc_per_node=8 \
    src/stage1_sample_ddp.py \
    --config configs/stage1/pretrained/SigLIP2.yaml \
    --data-path data/imagenet_val_test \
    --sample-dir evaluations/test_reconstructions \
    --image-size 256 \
    --precision bf16 \
    --per-proc-batch-size 4 \
    --num-workers 4

echo ""
echo "Step 3: Preparing reference set..."
python3 scripts/03_prepare_reference_set.py \
    --data-path data/imagenet_val_test \
    --output-path evaluations/test_reference_set_256.npz \
    --image-size 256

echo ""
echo "Step 4: Evaluating PSNR..."
python3 scripts/06_evaluate_psnr.py \
    --reference-npz evaluations/test_reference_set_256.npz \
    --reconstruction-npz evaluations/test_reconstructions/RAE-pretrained-bs4-bf16.npz \
    --max-value 255.0

echo ""
echo "=========================================================================="
echo "Test pipeline complete!"
echo "=========================================================================="
echo "If this test succeeded, you can run the full 50k evaluation with:"
echo "  bash scripts/run_full_evaluation.sh"
echo "=========================================================================="
