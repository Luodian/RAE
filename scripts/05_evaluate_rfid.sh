#!/bin/bash
# Evaluate rFID using ADM suite

RAE_DIR="/opt/tiger/RAE"
ADM_DIR="${RAE_DIR}/evaluations/adm_suite"
REFERENCE_NPZ="${RAE_DIR}/evaluations/reference_set_256.npz"
RECONSTRUCTION_NPZ="${RAE_DIR}/evaluations/reconstructions/RAE-pretrained-bs8-bf16.npz"
BATCH_SIZE=256

# Activate virtual environment
if [ -d "${RAE_DIR}/.venv" ]; then
    source "${RAE_DIR}/.venv/bin/activate"
fi

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

if [ ! -d "${ADM_DIR}" ]; then
    echo "Error: ADM suite not found at ${ADM_DIR}"
    echo "Please run 04_setup_adm_evaluation.sh first"
    exit 1
fi

echo "Computing rFID score..."
echo "Reference: ${REFERENCE_NPZ}"
echo "Reconstruction: ${RECONSTRUCTION_NPZ}"
echo "Batch size: ${BATCH_SIZE}"
echo ""

# Check if using ADM's evaluator or pytorch-fid
if [ -f "${ADM_DIR}/evaluations/evaluator.py" ]; then
    echo "Using ADM evaluator..."
    cd "${ADM_DIR}" || exit 1
    python evaluations/evaluator.py \
        "${REFERENCE_NPZ}" \
        "${RECONSTRUCTION_NPZ}" \
        --batch-size "${BATCH_SIZE}"
else
    # Fallback to pytorch-fid if ADM structure is different
    echo "ADM evaluator not found, using pytorch-fid as fallback..."

    # Install pytorch-fid if not available
    uv pip install pytorch-fid 2>/dev/null || pip install pytorch-fid

    # Use pytorch-fid (note: expects directory or npz files)
    python -m pytorch_fid \
        "${REFERENCE_NPZ}" \
        "${RECONSTRUCTION_NPZ}" \
        --batch-size "${BATCH_SIZE}"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "rFID evaluation complete!"
else
    echo ""
    echo "Error: rFID evaluation failed"
    exit 1
fi
