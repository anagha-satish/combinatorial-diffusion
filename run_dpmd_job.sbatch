#!/bin/bash
#SBATCH --job-name=dpmd_hiv
#SBATCH --output=logs/dpmd_hiv_%j.out
#SBATCH --error=logs/dpmd_hiv_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=seas_gpu
# If your cluster requires an account, keep the working one here:
# #SBATCH --account=YOUR_ACCOUNT_NAME

set -e

PROJECT_DIR="/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion"
VENV_DIR="$PROJECT_DIR/dpmd_venv"

# Slurm args: THRESHOLD, BUDGET, SEED, N_EVAL
THRESHOLD="${1:-300}"
BUDGET="${2:-5}"
SEED="${3:-0}"
N_EVAL="${4:-10}"

echo "============================================================"
echo " DPMD HIV job (venv)"
echo "  PROJECT_DIR : $PROJECT_DIR"
echo "  VENV_DIR    : $VENV_DIR"
echo "  THRESHOLD   : $THRESHOLD"
echo "  BUDGET      : $BUDGET"
echo "  SEED        : $SEED"
echo "  N_EVAL      : $N_EVAL"
echo "============================================================"
echo ""

cd "$PROJECT_DIR"
mkdir -p logs

# Load CUDA module (same as before)
if command -v module &> /dev/null; then
  module load cuda/12.4.1-fasrc01 2>/dev/null || true
fi

# Activate your venv (this is the key change)
source "$VENV_DIR/bin/activate"

echo "[env] Python executable: $(which python)"
echo "[env] Python version:"
python --version
echo ""

# Optional: quick torch check
python - << 'PY'
import torch
print("torch version:", getattr(torch, "__version__", "not installed"))
print("CUDA available:", torch.cuda.is_available())
PY
echo ""

# Run your driver with full settings
echo "[run] Starting driver_disease_dpmd.py ..."
python driver_disease_dpmd.py \
  -D HIV \
  -T "$THRESHOLD" \
  -B "$BUDGET" \
  -V "$N_EVAL" \
  -s "$SEED"

echo ""
echo "[done] Job finished."
