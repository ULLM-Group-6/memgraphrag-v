#!/bin/bash
# Validation, merging and stage stamps need no GPU/model loading.
#SBATCH --job-name=webqa-par-cpu
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=webqa-par-cpu-%j.log
set -euo pipefail
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
exec "$MGRV_TEXT_PYTHON" scripts/webqa_parallel.py "${1:?action required}" \
  --config "$MGRV_RUN_CONFIG" --artifacts "$MGRV_ARTIFACTS" \
  --split "${2:-dev}" --run-id "${3:-parallel-dev-v1}" --candidate "${4:-0}"
