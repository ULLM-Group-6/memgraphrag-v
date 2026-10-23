#!/bin/bash
# Automatic: bash scripts/submit_webqa_parallel.sh auto RUN_ID [ground|embed|assemble|tune|test|collect-test]
# Legacy single-value QA: bash scripts/submit_webqa_parallel.sh dev|test RUN_ID [ground|embed|assemble|qa|collect]
# Assumes no legacy serial indexing/QA job is writing to these artifacts.
set -euo pipefail
MODE=${1:-auto}
SPLIT=$MODE
if [[ "$MODE" == auto ]]; then SPLIT=dev; fi
RUN_ID=${2:?Supply a NEW result run ID}
START=${3:-ground}
[[ "$MODE" =~ ^(auto|dev|test)$ ]] || { echo 'Mode must be auto, dev or test'; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo 'Invalid run ID'; exit 2; }
if [[ "$MODE" == auto ]]; then
  [[ "$START" =~ ^(ground|embed|assemble|tune|test|collect-test)$ ]] || { echo 'Invalid automatic start stage'; exit 2; }
else
  [[ "$START" =~ ^(ground|embed|assemble|qa|collect)$ ]] || { echo 'Invalid start stage'; exit 2; }
fi
: "${MGRV_ARTIFACTS:?Use the EXISTING partially built artifacts directory}"
: "${MGRV_TEXT_PYTHON:?Set text Python}"
: "${MGRV_VISION_PYTHON:?Set vision Python}"
: "${MGRV_TEXT_EMBEDDING_MODEL:?Keep the BGE path used during text indexing}"
: "${MGRV_MEMGRAPHRAG:?Set upstream checkout}"
: "${MGRV_MG2_RAG:?Set MG2 checkout}"
: "${MGRV_SAM3_CHECKPOINT:?Set existing SAM3 checkpoint}"
: "${MGRV_READER_MODEL:?Set your own Qwen2.5-VL model directory}"
export MGRV_GROUND_PYTHON=${MGRV_GROUND_PYTHON:-$MGRV_VISION_PYTHON}
export MGRV_READER_PYTHON=${MGRV_READER_PYTHON:-$MGRV_VISION_PYTHON}
export MGRV_RUN_CONFIG=${MGRV_RUN_CONFIG:-$PWD/configs/webqa_run.yaml}
for python in "$MGRV_TEXT_PYTHON" "$MGRV_VISION_PYTHON" "$MGRV_GROUND_PYTHON" "$MGRV_READER_PYTHON"; do
  test -x "$python" || { echo "Missing Python: $python"; exit 2; }
done
test -f "$MGRV_ARTIFACTS/index/manifest.json"
test -f "$MGRV_RUN_CONFIG"
test -f scripts/webqa_parallel_gpu.sbatch
WORK="$MGRV_ARTIFACTS/index/parallel-v2"
mkdir -p "$WORK"
exec 9>"$WORK/submission.lock"
flock -n 9 || { echo 'Another submission is in progress'; exit 2; }
# Refuse a second chain while any job from the last one is pending/running.
active=$(squeue -h -u "$USER" -o '%F')
for job_file in "$WORK/jobs.tsv" "$MGRV_ARTIFACTS/index/parallel-v1/jobs.tsv"; do
 if [[ -s "$job_file" ]]; then
  while read -r action id; do
    if printf '%s\n' "$active" | grep -Fxq "$id"; then
      echo "Previous job $id ($action) is still active. Finish/cancel that chain first."
      exit 2
    fi
  done < "$job_file"
 fi
done
if [[ -s "$WORK/jobs.tsv" ]]; then
  cp "$WORK/jobs.tsv" "$WORK/jobs-$(date +%Y%m%d-%H%M%S).tsv"
fi
: > "$WORK/jobs.tsv"
PREVIOUS=
CANDIDATE=0
submit() {
  local action=$1 script=$2
  shift 2
  local dep=()
  if [[ -n "$PREVIOUS" ]]; then dep=(--dependency="afterok:$PREVIOUS"); fi
  local job
  job=$(sbatch --parsable --export=ALL --kill-on-invalid-dep=yes \
    "${dep[@]}" "$@" "$script" "$action" "$SPLIT" "$RUN_ID" "$CANDIDATE")
  PREVIOUS=${job%%;*}
  local label=$action
  if [[ "$action" == dev-* ]]; then label="$action-$CANDIDATE"; fi
  printf '%s\t%s\n' "$label" "$PREVIOUS" | tee -a "$WORK/jobs.tsv"
}
# Prepare/check frozen inputs on a CPU node, avoiding heavy hashing on the login node.
if [[ "$MODE" == auto ]]; then
  submit prepare-auto scripts/webqa_parallel_cpu.sbatch
else
  submit prepare scripts/webqa_parallel_cpu.sbatch
fi
if [[ "$START" == ground ]]; then
  submit ground scripts/webqa_parallel_gpu.sbatch --array=0-1%2 --time="${MGRV_GROUND_TIME:-08:00:00}"
  submit ground-finalize scripts/webqa_parallel_cpu.sbatch
  START=embed
fi
if [[ "$START" == embed ]]; then
  submit embed scripts/webqa_parallel_gpu.sbatch --array=0-1%2 --time="${MGRV_EMBED_TIME:-08:00:00}"
  submit embed-finalize scripts/webqa_parallel_cpu.sbatch
  START=assemble
fi
if [[ "$START" == assemble ]]; then
  submit assemble scripts/webqa_parallel_gpu.sbatch --time="${MGRV_ASSEMBLE_TIME:-04:00:00}"
  START=qa
  if [[ "$MODE" == auto ]]; then START=tune; fi
fi
if [[ "$MODE" == auto ]]; then
  if [[ "$START" == tune ]]; then
    # The CPU preflight verifies the three-entry grid before any GPU job runs.
    for CANDIDATE in 0 1 2; do
      submit dev-retrieve scripts/webqa_parallel_gpu.sbatch --array=0-1%2 --time="${MGRV_TUNE_TIME:-04:00:00}"
      submit dev-collect scripts/webqa_parallel_cpu.sbatch
    done
    CANDIDATE=0
    submit select-beta scripts/webqa_parallel_cpu.sbatch
    START=test
  fi
  if [[ "$START" == test ]]; then
    submit test-qa scripts/webqa_parallel_gpu.sbatch --array=0-1%2 --time="${MGRV_QA_TIME:-08:00:00}"
  fi
  submit test-collect scripts/webqa_parallel_cpu.sbatch
  echo "Automatic dev tuning -> test chain submitted; at most TWO GPUs. Job IDs: $WORK/jobs.tsv"
  exit 0
fi
if [[ "$START" == qa ]]; then
  submit qa scripts/webqa_parallel_gpu.sbatch --array=0-1%2 --time="${MGRV_QA_TIME:-08:00:00}"
fi
submit collect scripts/webqa_parallel_cpu.sbatch
echo "Submitted one chain; at most TWO GPUs. Job IDs: $WORK/jobs.tsv"
