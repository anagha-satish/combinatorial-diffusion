#!/bin/bash
#SBATCH --job-name=dpmd_hiv
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=seas_gpu

#SBATCH --chdir=/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion
#SBATCH --output=/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion/logs/dpmd_hiv_%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion/logs/dpmd_hiv_%j.err

set -euo pipefail

WORK_DIR="${WORK_DIR:-/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion}"
AFEG_ENV_NAME="${AFEG_ENV_NAME:-AFEG_DP}"

mkdir -p "$WORK_DIR/logs"
cd "$WORK_DIR"

module purge
module load cuda/12.4.1-fasrc01 2>/dev/null || true

# Conda init (site Miniforge)
if [ -f "/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh" ]; then
  source "/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh"
elif [ -f "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh" ]; then
  source "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh"
else
  echo "ERROR: Could not find site Miniforge conda.sh" >&2
  exit 1
fi

conda activate "$AFEG_ENV_NAME"

# --- GUROBI LICENSE (choose ONE method) ---
export GUROBI_LICENSE_SERVER="${GUROBI_LICENSE_SERVER:-PORT@SERVER}"
# export GRB_LICENSE_FILE="${GRB_LICENSE_FILE:-$HOME/.gurobi/gurobi.lic}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ----------------------------
# Args
# ----------------------------
# Profile: A or B
PROFILE="${1:-A}"

# Environment params
THRESHOLD="${2:-300}"
BUDGET="${3:-5}"
SEED="${4:-0}"
N_EVAL="${5:-10}"

# ----------------------------
# Debug (keep until stable)
# ----------------------------
echo "=== JOB DEBUG ==="
echo "HOST: $(hostname)"
echo "PWD:  $(pwd)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "GUROBI_LICENSE_SERVER: ${GUROBI_LICENSE_SERVER:-<unset>}"
echo "GRB_LICENSE_FILE: ${GRB_LICENSE_FILE:-<unset>}"
which python
python --version

python - <<'PY'
import sys
print("sys.executable:", sys.executable)

import numpy, pandas, scipy, networkx, tqdm, matplotlib
print("Core imports: OK")

import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (torch):", getattr(torch.version, "cuda", None))
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0))

import torch_geometric
print("torch_geometric:", torch_geometric.__version__)

import gurobipy as gp
print("gurobipy:", gp.gurobi.version())
m = gp.Model()
print("Gurobi model creation: OK")
PY

echo "nvidia-smi (if available):"
nvidia-smi || true
echo "================="

# ----------------------------
# Param profiles
# ----------------------------
COMMON_FLAGS=(
  -D HIV
  -T "$THRESHOLD"
  -B "$BUDGET"
  -V "$N_EVAL"
  -s "$SEED"
)

if [ "$PROFILE" = "A" ]; then
  # Set A: Stability-first
  EXTRA_FLAGS=(
    --lr 2e-4
    --num_particles 24
    --lambda_start 6.0 --lambda_end 2.5 --lambda_steps 20000
    --w_clip 1.5
    --kappa_exec 300.0
    --kappa_smooth 220.0
    --M_smooth 24 --J_smooth 2
    --flow_steps 48
    --tau 0.003 --delay_update 4
    --q_norm_clip 2.0 --q_running_beta 0.02
    --warmup_steps 4000
    --batch_size 64
    --train_updates 4000
  )
elif [ "$PROFILE" = "B" ]; then
  # Set B: More exploration early, still low noise
  EXTRA_FLAGS=(
    --lr 3e-4
    --num_particles 32
    --lambda_start 10.0 --lambda_end 3.0 --lambda_steps 30000
    --w_clip 2.0
    --kappa_exec 220.0
    --kappa_smooth 180.0
    --M_smooth 32 --J_smooth 1
    --flow_steps 48
    --tau 0.005 --delay_update 2
    --q_norm_clip 2.5 --q_running_beta 0.03
    --warmup_steps 6000
    --batch_size 96
    --train_updates 5000
  )
else
  echo "ERROR: PROFILE must be A or B (got '$PROFILE')" >&2
  exit 2
fi

echo "Running profile: $PROFILE"
echo "driver flags: ${COMMON_FLAGS[*]} ${EXTRA_FLAGS[*]}"

# Run
python driver_disease_dpmd.py "${COMMON_FLAGS[@]}" "${EXTRA_FLAGS[@]}"
