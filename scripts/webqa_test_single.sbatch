#!/bin/bash
# Submit with --dependency=afterok:TUNING_JOB_ID; do not edit existing scripts.
#SBATCH --job-name=webqa-test-single
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=webqa-test-single-%j.log
set -euo pipefail
BASE_RUN=${1:-webqa-tuned-v1}
OUTPUT_RUN=${2:-${BASE_RUN}-test-single}
: "${MGRV_ARTIFACTS:?}"
: "${MGRV_RUN_CONFIG:?}"
: "${MGRV_TEXT_PYTHON:?}"
: "${MGRV_VISION_PYTHON:?}"
: "${MGRV_READER_MODEL:?}"
export MGRV_READER_PYTHON=${MGRV_READER_PYTHON:-$MGRV_VISION_PYTHON}
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=9
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
srun --exact --nodes=1 --ntasks=1 --cpus-per-task=9 --gpus=1 --mem=96G \
  --kill-on-bad-exit=1 "$MGRV_TEXT_PYTHON" scripts/webqa_test_single.py run \
  --config "$MGRV_RUN_CONFIG" --artifacts "$MGRV_ARTIFACTS" \
  --run-id "$BASE_RUN" --output-run-id "$OUTPUT_RUN"
