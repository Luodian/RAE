#!/bin/bash
# Master script to run full rFID and PSNR evaluation pipeline

set -e  # Exit on error

RAE_DIR="/opt/tiger/RAE"
SCRIPTS_DIR="${RAE_DIR}/scripts"

cd "${RAE_DIR}" || exit 1

echo "=========================================================================="
echo "RAE Full Evaluation Pipeline - rFID and PSNR"
echo "=========================================================================="
echo ""

# Step 1: Download ImageNet validation set
echo "Step 1/6: Downloading ImageNet validation set (50k images)..."
if [ ! -d "${RAE_DIR}/data/imagenet_val" ] || [ -z "$(ls -A ${RAE_DIR}/data/imagenet_val)" ]; then
    bash "${SCRIPTS_DIR}/01_download_imagenet_val.sh"
else
    echo "Validation set already exists, skipping download..."
fi
echo ""

# Step 2: Reconstruct validation set
echo "Step 2/6: Reconstructing validation set with RAE..."
bash "${SCRIPTS_DIR}/02_reconstruct_val_set.sh"
echo ""

# Step 3: Prepare reference set
echo "Step 3/6: Preparing reference set for rFID..."
if [ ! -f "${RAE_DIR}/evaluations/reference_set_256.npz" ]; then
    bash "${SCRIPTS_DIR}/03_prepare_reference_set.sh"
else
    echo "Reference set already exists, skipping preparation..."
fi
echo ""

# Step 4: Setup ADM evaluation suite
echo "Step 4/6: Setting up ADM evaluation suite..."
if [ ! -d "${RAE_DIR}/evaluations/adm_suite" ]; then
    bash "${SCRIPTS_DIR}/04_setup_adm_evaluation.sh"
else
    echo "ADM suite already setup, skipping..."
fi
echo ""

# Step 5: Evaluate rFID
echo "Step 5/6: Evaluating rFID..."
bash "${SCRIPTS_DIR}/05_evaluate_rfid.sh"
echo ""

# Step 6: Evaluate PSNR
echo "Step 6/6: Evaluating PSNR..."
bash "${SCRIPTS_DIR}/06_evaluate_psnr.sh"
echo ""

echo "=========================================================================="
echo "Full evaluation pipeline complete!"
echo "=========================================================================="
echo "Results are saved in: ${RAE_DIR}/evaluations/"
echo ""
echo "Summary:"
echo "  - Reference set: ${RAE_DIR}/evaluations/reference_set_256.npz"
echo "  - Reconstructions: ${RAE_DIR}/evaluations/reconstructions/"
echo "  - PSNR results: ${RAE_DIR}/evaluations/reconstructions/psnr_results.txt"
echo "=========================================================================="
