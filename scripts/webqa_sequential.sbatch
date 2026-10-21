#!/bin/bash
# Submit through submit_webqa_sequential.sh, not directly.
#SBATCH --job-name=webqa-seq
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=webqa-seq-%j.log
set -euo pipefail
MODE=${1:?embed, assemble, tune or test required}
RUN_ID=${2:?existing run ID required}
REGISTRATION=${3:?submission record required}
PREVIOUS=${4:--}
ATTEMPT=${5:-0}
# The launcher submits matching limits. Leave ten minutes for checkpoint cleanup.
case "$MODE" in
  embed) WINDOW=14400 ;;
  assemble) WINDOW=21600 ;;
  tune|test) WINDOW=10800 ;;
  *) echo "Unknown mode: $MODE"; exit 2 ;;
esac
export MGRV_SEQ_DEADLINE=$(( $(date +%s) + WINDOW - 600 ))
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export OMP_NUM_THREADS=9
exec "$MGRV_TEXT_PYTHON" scripts/webqa_sequential.py run \
  --mode "$MODE" --run-id "$RUN_ID" --registration "$REGISTRATION" \
  --previous "$PREVIOUS" --attempt "$ATTEMPT"
