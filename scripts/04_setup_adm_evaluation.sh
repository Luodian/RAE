#!/bin/bash
# Setup ADM evaluation suite for rFID computation

RAE_DIR="/opt/tiger/RAE"
ADM_DIR="${RAE_DIR}/evaluations/adm_suite"

# Activate virtual environment
if [ -d "${RAE_DIR}/.venv" ]; then
    source "${RAE_DIR}/.venv/bin/activate"
fi

cd "${RAE_DIR}" || exit 1

# Create evaluations directory
mkdir -p "${RAE_DIR}/evaluations"

# Clone ADM evaluation suite if not exists
if [ ! -d "${ADM_DIR}" ]; then
    echo "Cloning ADM evaluation suite..."
    git clone https://github.com/openai/guided-diffusion.git "${ADM_DIR}"
else
    echo "ADM suite already exists at ${ADM_DIR}"
fi

cd "${ADM_DIR}" || exit 1

# Install dependencies for evaluation
echo "Installing ADM suite dependencies..."
uv pip install blobfile
uv pip install scipy

# Check if evaluator.py exists
if [ -f "evaluations/evaluator.py" ]; then
    echo "ADM evaluation suite setup complete!"
    echo "Location: ${ADM_DIR}"
else
    echo "Warning: evaluator.py not found in expected location"
    echo "The ADM suite structure might have changed"
    echo "Please check: ${ADM_DIR}/evaluations/"
fi

echo ""
echo "You can now use 05_evaluate_rfid.sh to compute rFID scores"
