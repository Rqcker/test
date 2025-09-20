#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
ENV_NAME="uag_env"          # Name of the conda environment
PYTHON_VERSION="3.11"       # Python version
TASK="GSM8K"               # Default task to run
DATA_PATH="GSM8K_input.jsonl" # Path to the input data
RECORD_PATH="${TASK}_output.jsonl" # Path to save the output

# --- Environment Setup ---
echo "[INFO] Setting up the environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda is not installed. Please install it first." >&2
    exit 1
fi

# Check if the environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Conda environment '${ENV_NAME}' already exists. Activating it."
else
    echo "[INFO] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python=${PYTHON_VERSION} -y
fi

# Activate the environment
source activate "${ENV_NAME}"

# --- Dependency Installation ---
echo "[INFO] Installing dependencies from requirements.txt..."

pip install -r requirements.txt

# --- Pre-computation/Model Download (Optional but recommended) ---
# It's better to pre-download the models before running the main script
echo "[INFO] Pre-downloading models (this may take a while)..."
python -c "from transformers import AutoModel, AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3'); AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True)" \
    || echo "[WARNING] Model pre-download failed, will attempt download during run. This may be due to needing to accept license terms."

# --- Run Experiment ---
echo "[INFO] Running UAG for task: ${TASK}..."

python code/uag.py \
    --task "${TASK}" \
    --data-path "${DATA_PATH}" \
    --record-path "${RECORD_PATH}" \
    --theta 16 \
    --max-length 2048

echo "[SUCCESS] Experiment finished! Results saved to ${RECORD_PATH}."

# Deactivate the environment
conda deactivate
