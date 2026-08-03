#!/bin/bash

set -euo pipefail

# ==================== ENVIRONMENT SETUP ====================
export SERVER="${SERVER:-cc}"

if [[ "$SERVER" == "cc" ]]; then
    # Load modules
    module load python/3.10
    module load cuda/12.6
    
    # Activate virtual environment
    VENV_DIR="${VENV:-$HOME/pareconv}"
    source "$VENV_DIR/bin/activate"
    
    # Set repository root
    REPO_ROOT="${REPO_CREATE_COR_ROOT:-$HOME/pc-registration/silico/PARENet/experiments/my3DMatch_TTA}"
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
else
    echo "This script is intended for Compute Canada (SERVER=cc)."
    exit 1
fi

# ==================== PYTHON ENVIRONMENT ====================
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONFAULTHANDLER=1

# ==================== SCRATCH SPACE SETUP ====================
# Use /scratch for heavy I/O and caching
SCRATCH_BASE="/scratch/$USER"
RUN_SCRATCH="$SCRATCH_BASE/runs/${SLURM_JOB_ID:-manual}"

export XDG_CACHE_HOME="$SCRATCH_BASE/.cache"
export XDG_CONFIG_HOME="$SCRATCH_BASE/.config"
export XDG_DATA_HOME="$SCRATCH_BASE/.local/share"
export TMPDIR="$SCRATCH_BASE/tmp"

mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$TMPDIR" "$RUN_SCRATCH"

# ==================== GRID SEARCH PARAMETERS ====================
# These are passed from the launcher script
COR="${COR:-uniform}"
SEV="${SEV:-3}"

# ==================== DIAGNOSTIC INFO ====================
echo "=================================================="
echo "SLURM Job Information"
echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Job Name: ${SLURM_JOB_NAME:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo ""
echo "Environment:"
echo "  REPO_ROOT: $REPO_ROOT"
echo "  VENV: $VENV_DIR"

echo ""
echo "Evaluation on corruption:"
echo "  Corruption : $COR"
echo "  Severity: $SEV"
echo "=================================================="

# ==================== GPU INFO ====================
nvidia-smi
echo ""

# ==================== RUN TRAINING ====================
cd "$REPO_ROOT"

echo "Starting evaluation on corruption with associated severity..."
echo ""

# Run the training script with the modified config
python -u create_corrupted_dataset_with_slurm.py \
    --COR "$COR" \
    --SEV "$SEV" \
    2>&1 | tee "$RUN_SCRATCH/test_log.txt"

echo ""
echo "=================================================="
echo "Testing completed"
echo "=================================================="
echo "Log saved to: $RUN_SCRATCH/test_log.txt"