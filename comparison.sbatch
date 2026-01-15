#!/bin/bash
#SBATCH --job-name=comparison_dpmd_srl
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=seas_gpu

#SBATCH --chdir=/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion
#SBATCH --output=/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion/logs/comparison_%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion/logs/comparison_%j.err

set -euo pipefail

# ============================================================
# Paths + environment
# ============================================================
WORK_DIR="/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion"
ENV_NAME="AFEG_DP"

# DPMD results directory you already have
DPMD_RESULTS_DIR="${WORK_DIR}/results_HIV_100_5"

# SRL results directory you already have
SRL_RESULTS_DIR="${WORK_DIR}/results_SRL_HIV_100_5"

mkdir -p "${WORK_DIR}/logs"
cd "${WORK_DIR}"

# ============================================================
# Modules + Conda (matches your previous sbatch style)
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
# Experiment config
# ============================================================
STD="HIV"
THRESHOLD=100
BUDGET=5
GAMMA=0.99
SEED=0

TRAIN_UPDATES=200
FLOW_STEPS=36
NUM_PARTICLES=12

# ---- choose the DPMD best eval episode you want to plot ----
# (set this to the best episode printed by DPMD, e.g. 140)
BEST_EP_DPMD=140

# ---- SRL best-episode trajectories (you provided this exact path) ----
SRL_TRAJ="/n/netscratch/tambe_lab/Lab/anagha/combinatorial-diffusion/results_SRL_HIV_100_5/trajectories_ep200_HIV_T100_B5_gamma0.99_seed0_train200_k5_m32.csv"

# Baselines write into ./results (hardcoded in your baseline scripts)
BASELINES_ROOT="${WORK_DIR}/results"
RAND_PREFIX="rand_"
GREEDY_PREFIX="greedy_"

# Use more baseline eval episodes for smoother curves (edit if you want)
BASELINE_EVAL_EPISODES=50

# ============================================================
# Debug info
# ============================================================
echo "================ JOB DEBUG ================"
echo "HOST: $(hostname)"
echo "PWD:  $(pwd)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
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

echo "CONFIG:"
echo "  DPMD ep=${BEST_EP_DPMD} from: ${DPMD_RESULTS_DIR}"
echo "  SRL traj: ${SRL_TRAJ}"
echo "  STD=${STD} T=${THRESHOLD} B=${BUDGET} gamma=${GAMMA} seed=${SEED}"
echo "  Baselines: N_EVAL=${BASELINE_EVAL_EPISODES}"
echo ""

# ============================================================
# 1) Select the DPMD trajectories at BEST_EP_DPMD
# ============================================================
RUN_TAG="${STD}_T${THRESHOLD}_B${BUDGET}_gamma${GAMMA}_seed${SEED}_train${TRAIN_UPDATES}_flow${FLOW_STEPS}_K${NUM_PARTICLES}"
DPMD_TRAJ="${DPMD_RESULTS_DIR}/trajectories_ep${BEST_EP_DPMD}_${RUN_TAG}.csv"

if [ ! -f "$DPMD_TRAJ" ]; then
  echo "ERROR: Missing DPMD trajectories file:"
  echo "  $DPMD_TRAJ"
  echo "Available trajectories (head):"
  ls -1 "${DPMD_RESULTS_DIR}"/trajectories_ep*_"${RUN_TAG}".csv 2>/dev/null | head -n 10 || true
  exit 2
fi

echo "✓ DPMD trajectories: $DPMD_TRAJ"
echo ""

# ============================================================
# 1b) Confirm SRL trajectories exists
# ============================================================
if [ ! -f "$SRL_TRAJ" ]; then
  echo "ERROR: Missing SRL trajectories file:"
  echo "  $SRL_TRAJ"
  echo "Available SRL trajectories (head):"
  ls -1 "${SRL_RESULTS_DIR}"/trajectories_ep*.csv 2>/dev/null | head -n 20 || true
  exit 9
fi
echo "✓ SRL trajectories: $SRL_TRAJ"
echo ""

# ============================================================
# 2) Random baseline
# ============================================================
echo "============================================================"
echo "2) Running Random baseline"
echo "============================================================"

python3 driver_disease_random.py \
  -D "$STD" -T "$THRESHOLD" -I 0 -B "$BUDGET" -s "$SEED" -V "$BASELINE_EVAL_EPISODES" \
  -p "$RAND_PREFIX" \
  -G "$GAMMA"

RAND_RUN_DIR=$(ls -td "${BASELINES_ROOT}/${RAND_PREFIX}${STD}_T${THRESHOLD}_B${BUDGET}_seed${SEED}_"* | head -n 1)
RAND_TRAJ="${RAND_RUN_DIR}/trajectories.csv"

if [ ! -f "$RAND_TRAJ" ]; then
  echo "ERROR: Missing random trajectories: $RAND_TRAJ"
  exit 3
fi

echo "✓ Random run dir: $RAND_RUN_DIR"
echo "✓ Random trajectories: $RAND_TRAJ"
echo ""

# ============================================================
# 3) Greedy baseline
# ============================================================
echo "============================================================"
echo "3) Running Greedy Positive Neighbors baseline"
echo "============================================================"

python3 driver_disease_greedy.py \
  -D "$STD" -T "$THRESHOLD" -I 0 -B "$BUDGET" -s "$SEED" -V "$BASELINE_EVAL_EPISODES" \
  -p "$GREEDY_PREFIX" \
  -G "$GAMMA"

GREEDY_RUN_DIR=$(ls -td "${BASELINES_ROOT}/${GREEDY_PREFIX}${STD}_GreedyPositive_T${THRESHOLD}_B${BUDGET}_seed${SEED}_"* | head -n 1)
GREEDY_TRAJ="${GREEDY_RUN_DIR}/trajectories.csv"

if [ ! -f "$GREEDY_TRAJ" ]; then
  echo "ERROR: Missing greedy trajectories: $GREEDY_TRAJ"
  exit 4
fi

echo "✓ Greedy run dir: $GREEDY_RUN_DIR"
echo "✓ Greedy trajectories: $GREEDY_TRAJ"
echo ""

# ============================================================
# 4) Comparison plot (DPMD + Random + Greedy + SRL)
# ============================================================
echo "============================================================"
echo "4) Plotting comparison (DPMD ep ${BEST_EP_DPMD} + SRL + baselines)"
echo "============================================================"

OUT_PNG="${DPMD_RESULTS_DIR}/comparison_dpmd_ep${BEST_EP_DPMD}_srl.png"

python3 plot_comparison.py \
  --dpmd "$DPMD_TRAJ" \
  --random "$RAND_TRAJ" \
  --greedy "$GREEDY_TRAJ" \
  --srl "$SRL_TRAJ" \
  -o "$OUT_PNG" \
  --title "Comparison @ DPMD ep ${BEST_EP_DPMD} (and SRL best) | ${STD}, T=${THRESHOLD}, B=${BUDGET}, seed=${SEED}, gamma=${GAMMA}" \
  --dpi 300

echo ""
echo "==================== OUTPUT ===================="
echo "Comparison plot saved to:"
echo "  $OUT_PNG"
echo "DPMD trajectories:"
echo "  $DPMD_TRAJ"
echo "SRL trajectories:"
echo "  $SRL_TRAJ"
echo "Random run dir:"
echo "  $RAND_RUN_DIR"
echo "Greedy run dir:"
echo "  $GREEDY_RUN_DIR"
echo "================================================"
echo "[DONE]"