#!/bin/bash
# Usage: bash scripts/submit_webqa_sequential.sh RUN_ID [index|assemble|tune|test]
# index: 5h embed -> optional 5h backup; success -> 6h assembly -> 3h tune -> 3h test.
set -euo pipefail
RUN_ID=${1:?Supply the existing run ID, e.g. webqa-tuned-v1}
START=${2:-index}
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo 'Invalid run ID'; exit 2; }
[[ "$START" =~ ^(index|assemble|tune|test)$ ]] || { echo 'Invalid start'; exit 2; }
[[ -z "${SLURM_JOB_ID:-}" ]] || { echo 'Submit from a login shell, outside an allocation'; exit 2; }
: "${MGRV_ARTIFACTS:?Use your ORIGINAL artifacts directory}"
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
for file in scripts/webqa_sequential.sbatch scripts/webqa_sequential.py \
  scripts/webqa_embed_checkpointed.py scripts/webqa_validation_cache.py scripts/webqa_fast_validation.py \
  scripts/webqa_fast_stage.py scripts/webqa_fast_qa.py scripts/webqa_parallel.py scripts/webqa_run.py \
  scripts/webqa_tuning.py "$MGRV_ARTIFACTS/index/manifest.json" "$MGRV_RUN_CONFIG"; do
  test -f "$file" || { echo "Missing file: $file"; exit 2; }
done
WORK="$MGRV_ARTIFACTS/index/parallel-v2"
mkdir -p "$WORK/sequential-v1"
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
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
REGISTRATION="$WORK/sequential-v1/submission-$STAMP.json"
"$MGRV_TEXT_PYTHON" scripts/webqa_sequential.py register --run-id "$RUN_ID" --registration "$REGISTRATION"
if [[ -s "$WORK/jobs.tsv" ]]; then cp "$WORK/jobs.tsv" "$WORK/jobs-before-sequential-$STAMP.tsv"; fi
: > "$WORK/jobs.tsv"
LAST_JOB=
submit() {
  local mode=$1 label=$2 limit=$3 dependency=$4 predecessor=${5:--} attempt=${6:-0} job
  local dep=()
  if [[ -n "$dependency" ]]; then
    dep=(--dependency="$dependency")
  fi
  # CLI resource values also prevent exported SBATCH_* variables from silently
  # changing the request. --signal reaches the Python coordinator before timeout.
  if ! job=$(sbatch --parsable --export=ALL --kill-on-invalid-dep=yes --no-requeue \
    --partition=gpu_a100 --nodes=1 --ntasks=1 --cpus-per-task=9 --gpus-per-node=1 \
    --mem=96G --time="$limit" --signal=B:USR1@120 --job-name="webqa-seq-$label" "${dep[@]}" \
    scripts/webqa_sequential.sbatch "$mode" "$RUN_ID" "$REGISTRATION" "$predecessor" "$attempt"); then
    echo "Submission failed; inspect already accepted jobs in $WORK/jobs.tsv before retrying." >&2
    return 1
  fi
  LAST_JOB=${job%%;*}
  [[ "$LAST_JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job"; exit 2; }
  printf '%s\t%s\n' "$label" "$LAST_JOB" | tee -a "$WORK/jobs.tsv"
}
if [[ "$START" == index ]]; then
  submit embed embed-1 04:00:00 '' - 1
  FIRST=$LAST_JOB
  submit embed embed-2 04:00:00 "afternotok:$FIRST" "$FIRST" 2
  SECOND=$LAST_JOB
  # OR dependency: assembly needs successful completion of EITHER attempt.
  # If FIRST succeeds, SECOND is impossible and Slurm cancels it without running.
  submit assemble assemble 06:00:00 "afterok:$FIRST?afterok:$SECOND"
elif [[ "$START" == assemble ]]; then
  submit assemble assemble 06:00:00 ''
fi
if [[ "$START" != test ]]; then
  DEP=${LAST_JOB:+afterok:$LAST_JOB}
  submit tune tune 03:00:00 "$DEP"
fi
DEP=${LAST_JOB:+afterok:$LAST_JOB}
submit test test 03:00:00 "$DEP"
echo 'Submitted conditional continuation. At most ONE GPU runs in this chain.'
echo "Job IDs: $WORK/jobs.tsv"
echo 'Embedding: up to 5h + optional 5h. Assembly: 6h. Tune/test: 3h each.'
echo 'Assembly requires embedding SUCCESS; checkpoint exit 75 or two timeouts never release it.'
