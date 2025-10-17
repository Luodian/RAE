#!/bin/bash
# Reconstruct ImageNet validation set (50k images) using RAE

RAE_DIR="/opt/tiger/RAE"
CONFIG_PATH="${RAE_DIR}/configs/stage1/pretrained/SigLIP2.yaml"
VAL_DIR="${RAE_DIR}/data/imagenet_val"
RECON_DIR="${RAE_DIR}/evaluations/reconstructions"
IMAGE_SIZE=256
NPROC=8
BATCH_SIZE=8  # Larger batch for validation set

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

# Verify config file exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo "Error: Config file not found at ${CONFIG_PATH}"
    exit 1
fi

# Create reconstruction directory
mkdir -p "${RECON_DIR}"

echo "Starting RAE reconstruction on validation set..."
echo "Config: ${CONFIG_PATH}"
echo "Input: ${VAL_DIR}"
echo "Output: ${RECON_DIR}"
echo "Image size: ${IMAGE_SIZE}"
echo "Number of GPUs: ${NPROC}"
echo "Batch size per GPU: ${BATCH_SIZE}"

# Run reconstruction with NCCL fix
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc_per_node="${NPROC}" \
    src/stage1_sample_ddp.py \
    --config "${CONFIG_PATH}" \
    --data-path "${VAL_DIR}" \
    --sample-dir "${RECON_DIR}" \
    --image-size "${IMAGE_SIZE}" \
    --precision bf16 \
    --per-proc-batch-size "${BATCH_SIZE}" \
    --num-workers 4

if [ $? -eq 0 ]; then
    echo "Reconstruction complete!"
    echo "Results saved to: ${RECON_DIR}"
    echo ""
    echo "NPZ file containing all reconstructions:"
    find "${RECON_DIR}" -name "*.npz" -exec ls -lh {} \;
else
    echo "Error: Reconstruction failed"
    exit 1
fi
