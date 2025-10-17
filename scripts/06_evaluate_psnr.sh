#!/bin/bash
# Evaluate PSNR between original and reconstructed images

RAE_DIR="/opt/tiger/RAE"
REFERENCE_NPZ="${RAE_DIR}/evaluations/reference_set_256.npz"
RECONSTRUCTION_NPZ="${RAE_DIR}/evaluations/reconstructions/RAE-pretrained-bs8-bf16.npz"

# Using uv for environment management
# No need to activate venv, uv handles it

cd "${RAE_DIR}" || exit 1

# Verify files exist
if [ ! -f "${REFERENCE_NPZ}" ]; then
    echo "Error: Reference set not found at ${REFERENCE_NPZ}"
    echo "Please run 03_prepare_reference_set.sh first"
    exit 1
fi

if [ ! -f "${RECONSTRUCTION_NPZ}" ]; then
    echo "Error: Reconstruction NPZ not found at ${RECONSTRUCTION_NPZ}"
    echo "Please run 02_reconstruct_val_set.sh first"
    echo "Note: Update RECONSTRUCTION_NPZ path in this script if your NPZ has a different name"
    exit 1
fi

echo "Evaluating PSNR..."
echo "Reference: ${REFERENCE_NPZ}"
echo "Reconstruction: ${RECONSTRUCTION_NPZ}"
echo ""

uv run python scripts/06_evaluate_psnr.py \
    --reference-npz "${REFERENCE_NPZ}" \
    --reconstruction-npz "${RECONSTRUCTION_NPZ}" \
    --max-value 255.0

if [ $? -eq 0 ]; then
    echo ""
    echo "PSNR evaluation complete!"
else
    echo ""
    echo "Error: PSNR evaluation failed"
    exit 1
fi
