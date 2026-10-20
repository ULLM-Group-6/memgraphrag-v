#!/bin/bash
# Submit through submit_webqa_grouped.sh. Two GPUs remain allocated between steps.
#SBATCH --job-name=webqa-grouped
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=2
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=webqa-grouped-%j.log
set -euo pipefail
MODE=${1:?index or qa required}
RUN_ID=${2:?run ID required}
START=${3:?starting action required}
case "$MODE:$START" in
  index:embed|index:assemble|qa:tune|qa:test|qa:collect-test) ;;
  *) echo "Invalid mode/start: $MODE/$START" >&2; exit 2 ;;
esac
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "${SLURM_SUBMIT_DIR:?}"
: "${MGRV_ARTIFACTS:?}"
: "${MGRV_RUN_CONFIG:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
WORK="$MGRV_ARTIFACTS/index/parallel-v2"
mkdir -p "$WORK"
# Additional whole-allocation lock; existing Python per-phase/worker locks remain.
exec 8>"$WORK/grouped.lock"
flock -n 8 || { echo 'Another grouped allocation is running' >&2; exit 2; }
LOGS="$SLURM_SUBMIT_DIR/logs/webqa-grouped-$SLURM_JOB_ID"
mkdir -p "$LOGS"
SEQUENCE=0
step() {
  local tasks=$1 action=$2 candidate=${3:-0}
  SEQUENCE=$((SEQUENCE + 1))
  # Separate task logs avoid mixing two workers' progress bars.
  local prefix="$LOGS/$(printf '%02d' "$SEQUENCE")-$action-candidate$candidate"
  echo "$(date -Is) START $action candidate=$candidate workers=$tasks logs=$prefix-task%t.log"
  local status=0
  srun --exact --nodes=1 --ntasks="$tasks" --cpus-per-task=18 \
    --gpus="$tasks" --gpus-per-task=1 --gpu-bind=single:1 --mem="$((96 * tasks))G" \
    --kill-on-bad-exit=1 --output="$prefix-task%t.log" --error="$prefix-task%t.log" \
    bash scripts/webqa_grouped_step.sh "$action" "$RUN_ID" "$candidate" || status=$?
  if (( status != 0 )); then
    echo "$(date -Is) FAILED $action candidate=$candidate exit=$status; stopping" >&2
    return "$status"
  fi
  echo "$(date -Is) DONE $action candidate=$candidate"
}
step 1 prepare-auto
if [[ "$MODE" == index ]]; then
  if [[ "$START" == embed ]]; then
    step 2 embed
    step 1 embed-finalize
  fi
  step 1 assemble
else
  if [[ "$START" == tune ]]; then
    for candidate in 0 1 2; do
      step 2 dev-retrieve "$candidate"
      step 1 dev-collect "$candidate"
    done
    step 1 select-beta
  fi
  if [[ "$START" != collect-test ]]; then
    step 2 test-qa
  fi
  step 1 test-collect
fi
echo "$(date -Is) COMPLETED $MODE run=$RUN_ID"
