#!/bin/bash
#SBATCH --job-name=srl_HIV_T100_B5
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=seas_gpu

#SBATCH --chdir=/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion
#SBATCH --output=/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion/logs/srl_HIV_T100_B5_%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion/logs/srl_HIV_T100_B5_%j.err

set -euo pipefail

# ============================================================
# Paths + environment
# ============================================================
WORK_DIR="/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion"
RESULTS_DIR="/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion/results_SRL_HIV_100_5"
ENV_NAME="AFEG_DP"

mkdir -p "$WORK_DIR/logs"
mkdir -p "$RESULTS_DIR"
cd "$WORK_DIR"

# ============================================================
# Modules + Conda (same as previous jobs)
# ============================================================
module purge
module load cuda/12.4.1-fasrc01 2>/dev/null || true

if [ -f "/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh" ]; then
  source "/n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh"
elif [ -f "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh" ]; then
  source "/n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh"
else
  echo "ERROR: Could not find site Miniforge conda.sh" >&2
  exit 1
fi

conda activate "$ENV_NAME"

# ============================================================
# Gurobi + CUDA
# ============================================================
export GUROBI_LICENSE_SERVER="${GUROBI_LICENSE_SERVER:-PORT@SERVER}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ============================================================
# Debug info (matches your older scripts)
# ============================================================
echo "================ JOB DEBUG ================"
echo "HOST: $(hostname)"
echo "PWD:  $(pwd)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
which python3
python3 --version

python3 - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

nvidia-smi || true
echo "=========================================="

# ============================================================
# Run SRL baseline (match DPMD run settings)
#   DPMD command was:
#     -D HIV -T 100 -B 5 --warmup_steps 1000 --train_updates 200 --results_dir ... --gamma default(0.99)
#
# Here we mirror those same top-level knobs:
#   -D, -T, -B, --warmup_steps, --train_updates, --gamma, --seed, --results_dir, --log_every, -V
# Everything else uses SRL driver defaults (as DPMD does).
# ============================================================
python3 srl_driver.py \
  -D HIV \
  -T 100 \
  -B 5 \
  -s 0 \
  --gamma 0.99 \
  --warmup_steps 1000 \
  --train_updates 200 \
  --log_every 10 \
  -V 10 \
  --results_dir "$RESULTS_DIR"

echo "[DONE]"
