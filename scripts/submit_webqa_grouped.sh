#!/bin/bash
# Usage: bash scripts/submit_webqa_grouped.sh RUN_ID [embed|assemble|tune|test|collect-test]
# Default: two allocations, finish indexing then automatic dev tuning/test.
# Requires the EXISTING completed ground stage. Never starts text or grounding.
set -euo pipefail
RUN_ID=${1:?Supply the existing automatic run ID, e.g. webqa-tuned-v1}
START=${2:-embed}
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo 'Invalid run ID'; exit 2; }
[[ "$START" =~ ^(embed|assemble|tune|test|collect-test)$ ]] || { echo 'Invalid start'; exit 2; }
: "${MGRV_ARTIFACTS:?Use the existing partially built artifacts directory}"
: "${MGRV_TEXT_PYTHON:?}"
: "${MGRV_VISION_PYTHON:?}"
: "${MGRV_TEXT_EMBEDDING_MODEL:?Keep the original BGE path}"
: "${MGRV_MEMGRAPHRAG:?}"
: "${MGRV_MG2_RAG:?}"
: "${MGRV_SAM3_CHECKPOINT:?Keep the original checkpoint}"
: "${MGRV_READER_MODEL:?}"
export MGRV_GROUND_PYTHON=${MGRV_GROUND_PYTHON:-$MGRV_VISION_PYTHON}
export MGRV_READER_PYTHON=${MGRV_READER_PYTHON:-$MGRV_VISION_PYTHON}
export MGRV_RUN_CONFIG=${MGRV_RUN_CONFIG:-$PWD/configs/webqa_run.yaml}
for python in "$MGRV_TEXT_PYTHON" "$MGRV_VISION_PYTHON" "$MGRV_READER_PYTHON"; do
  test -x "$python" || { echo "Missing Python: $python"; exit 2; }
done
for file in scripts/webqa_grouped.sbatch scripts/webqa_grouped_step.sh \
  scripts/webqa_parallel.py scripts/webqa_run.py scripts/webqa_tuning.py \
  "$MGRV_ARTIFACTS/index/manifest.json" "$MGRV_RUN_CONFIG"; do
  test -f "$file" || { echo "Missing file: $file"; exit 2; }
done
WORK="$MGRV_ARTIFACTS/index/parallel-v2"
mkdir -p "$WORK"
exec 9>"$WORK/submission.lock"
flock -n 9 || { echo 'Another submission is in progress'; exit 2; }
active=$(squeue -h -u "$USER" -o '%F')
for job_file in "$WORK/jobs.tsv" "$MGRV_ARTIFACTS/index/parallel-v1/jobs.tsv"; do
  if [[ -s "$job_file" ]]; then
    while read -r action id; do
      if printf '%s\n' "$active" | grep -Fxq "$id"; then
        echo "Previous job $id ($action) is still active. Wait/cancel that chain first."
        exit 2
      fi
    done < "$job_file"
  fi
done
# Keep the same registry so the old launcher also refuses an active grouped chain.
if [[ -s "$WORK/jobs.tsv" ]]; then
  cp "$WORK/jobs.tsv" "$WORK/jobs-before-grouped-$(date +%Y%m%d-%H%M%S)-$$.tsv"
fi
: > "$WORK/jobs.tsv"
PREVIOUS=
submit() {
  local mode=$1 start=$2 limit=$3
  local dep=() job
  if [[ -n "$PREVIOUS" ]]; then dep=(--dependency="afterok:$PREVIOUS"); fi
  if ! job=$(sbatch --parsable --export=ALL --kill-on-invalid-dep=yes \
    "${dep[@]}" --time="$limit" --job-name="webqa-grp-$mode" \
    scripts/webqa_grouped.sbatch "$mode" "$RUN_ID" "$start"); then
    echo "Submission failed. Already accepted jobs are recorded in $WORK/jobs.tsv." >&2
    echo 'Do not resubmit until you inspect that file and squeue.' >&2
    return 1
  fi
  PREVIOUS=${job%%;*}
  printf 'grouped-%s\t%s\n' "$mode" "$PREVIOUS" | tee -a "$WORK/jobs.tsv"
}
if [[ "$START" == embed || "$START" == assemble ]]; then
  submit index "$START" "${MGRV_GROUPED_INDEX_TIME:-08:00:00}"
  START=tune
fi
submit qa "$START" "${MGRV_GROUPED_QA_TIME:-08:00:00}"
echo "Submitted grouped continuation; maximum TWO GPUs concurrently."
echo "Job IDs: $WORK/jobs.tsv"
echo 'Eight-hour defaults are allocation limits, not measured runtime estimates.'
