#!/bin/bash
# One GPU per task. Only submit through submit_webqa_parallel.sh.
#SBATCH --job-name=webqa-par-gpu
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=webqa-par-%A_%a.log
set -euo pipefail
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
ACTION=${1:?action required}
case "$ACTION" in
  ground) PYTHON="$MGRV_GROUND_PYTHON" ;;
  embed) PYTHON="$MGRV_VISION_PYTHON" ;;
  assemble|qa|dev-retrieve|test-qa) PYTHON="$MGRV_TEXT_PYTHON" ;;
  *) echo "Unknown GPU action: $ACTION"; exit 2 ;;
esac
exec "$PYTHON" scripts/webqa_parallel.py "$ACTION" \
  --config "$MGRV_RUN_CONFIG" --artifacts "$MGRV_ARTIFACTS" \
  --worker "${SLURM_ARRAY_TASK_ID:-0}" --split "${2:-dev}" --run-id "${3:-parallel-dev-v1}" \
  --candidate "${4:-0}"
