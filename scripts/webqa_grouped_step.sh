#!/bin/bash
# Called by srun inside webqa_grouped.sbatch, once per worker/task.
set -euo pipefail
ACTION=${1:?action required}
RUN_ID=${2:?run ID required}
CANDIDATE=${3:-0}
case "$ACTION" in
  embed) PYTHON="$MGRV_VISION_PYTHON" ;;
  prepare-auto|embed-finalize|assemble|dev-retrieve|dev-collect|select-beta|test-qa|test-collect)
    PYTHON="$MGRV_TEXT_PYTHON" ;;
  *) echo "Unsupported grouped action: $ACTION" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-18}"
# srun gives each task ONE visible GPU. The Python code uses local device 0.
# Do not overwrite CUDA_VISIBLE_DEVICES with SLURM_PROCID.
exec "$PYTHON" scripts/webqa_parallel.py "$ACTION" \
  --config "$MGRV_RUN_CONFIG" --artifacts "$MGRV_ARTIFACTS" \
  --worker "${SLURM_PROCID:-0}" --split dev --run-id "$RUN_ID" \
  --candidate "$CANDIDATE"
