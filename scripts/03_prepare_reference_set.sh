#!/bin/bash
# Prepare reference set for rFID evaluation

RAE_DIR="/opt/tiger/RAE"
VAL_DIR="${RAE_DIR}/data/imagenet_val"
OUTPUT_PATH="${RAE_DIR}/evaluations/reference_set_256.npz"
IMAGE_SIZE=256

# Activate virtual environment
if [ -d "${RAE_DIR}/.venv" ]; then
    source "${RAE_DIR}/.venv/bin/activate"
fi

cd "${RAE_DIR}" || exit 1

# Verify validation set exists
if [ ! -d "${VAL_DIR}" ]; then
    echo "Error: Validation set not found at ${VAL_DIR}"
    echo "Please run 01_download_imagenet_val.sh first"
    exit 1
fi

echo "Preparing reference set for rFID evaluation..."
echo "Input: ${VAL_DIR}"
echo "Output: ${OUTPUT_PATH}"
echo "Image size: ${IMAGE_SIZE}"

python3 scripts/03_prepare_reference_set.py \
    --data-path "${VAL_DIR}" \
    --output-path "${OUTPUT_PATH}" \
    --image-size "${IMAGE_SIZE}"

if [ $? -eq 0 ]; then
    echo "Reference set preparation complete!"
    ls -lh "${OUTPUT_PATH}"
else
    echo "Error: Reference set preparation failed"
    exit 1
fi
