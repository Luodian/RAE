#!/bin/bash
# Download ImageNet validation set (50k images) from HuggingFace

RAE_DIR="/opt/tiger/RAE"
VAL_DIR="${RAE_DIR}/data/imagenet_val"
HF_DATASET="ILSVRC/imagenet-1k"

# Activate virtual environment
if [ -d "${RAE_DIR}/.venv" ]; then
    source "${RAE_DIR}/.venv/bin/activate"
fi

cd "${RAE_DIR}" || exit 1

echo "Downloading ImageNet validation set (50k images)..."

python3 << 'EOF'
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

# Configuration
VAL_DIR = Path("/opt/tiger/RAE/data/imagenet_val")
HF_DATASET = "ILSVRC/imagenet-1k"

# Create output directory
VAL_DIR.mkdir(parents=True, exist_ok=True)

# Load validation split
print(f"Loading validation set from {HF_DATASET}...")
dataset = load_dataset(HF_DATASET, split="validation", streaming=False)

print(f"Total validation images: {len(dataset)}")

# Save images organized by class (ImageFolder format)
print(f"Saving images to {VAL_DIR}...")
for idx, item in enumerate(tqdm(dataset, desc="Downloading validation set")):
    # Get image and label
    image = item['image']
    label = item['label']

    # Create class directory
    class_dir = VAL_DIR / str(label)
    class_dir.mkdir(exist_ok=True)

    # Save image
    image_path = class_dir / f"{idx:06d}.JPEG"

    # Convert to RGB if needed (some images might be grayscale)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image.save(image_path, 'JPEG', quality=95)

print(f"Successfully downloaded {len(dataset)} validation images to {VAL_DIR}")
EOF

echo "Done! Validation set saved to ${VAL_DIR}"
