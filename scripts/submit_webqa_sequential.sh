#!/bin/bash
# Usage: bash scripts/submit_webqa_sequential.sh RUN_ID [index|tune|test]
# index: three continuation slices -> dev beta tuning -> final test QA.
set -euo pipefail
RUN_ID=${1:?Supply the existing run ID, e.g. webqa-tuned-v1}
START=${2:-index}
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo 'Invalid run ID'; exit 2; }
[[ "$START" =~ ^(index|tune|test)$ ]] || { echo 'Invalid start'; exit 2; }
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
  scripts/webqa_embed_checkpointed.py scripts/webqa_parallel.py scripts/webqa_run.py \
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
PREVIOUS=
submit() {
  local mode=$1 label=$2 dependency=$3 last=$4 predecessor=- job
  local dep=()
  if [[ -n "$PREVIOUS" ]]; then
    dep=(--dependency="$dependency:$PREVIOUS")
    # Only indexing continuation accepts TIMEOUT and needs an accounting gate.
    if [[ "$mode" == index ]]; then predecessor=$PREVIOUS; fi
  fi
  # CLI resource values also prevent exported SBATCH_* variables from silently
  # changing the requested allocation. All five jobs request identical resources.
  if ! job=$(sbatch --parsable --export=ALL --kill-on-invalid-dep=yes --no-requeue \
    --partition=gpu_a100 --nodes=1 --ntasks=1 --cpus-per-task=9 --gpus-per-node=1 \
    --mem=96G --time=03:00:00 --job-name="webqa-seq-$label" "${dep[@]}" \
    scripts/webqa_sequential.sbatch "$mode" "$RUN_ID" "$REGISTRATION" "$predecessor" "$last"); then
    echo "Submission failed; inspect already accepted jobs in $WORK/jobs.tsv before retrying." >&2
    return 1
  fi
  PREVIOUS=${job%%;*}
  [[ "$PREVIOUS" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job"; exit 2; }
  printf '%s\t%s\n' "$label" "$PREVIOUS" | tee -a "$WORK/jobs.tsv"
}
if [[ "$START" == index ]]; then
  submit index index-1 afterany 0
  submit index index-2 afterany 0
  submit index index-3 afterany 1
fi
if [[ "$START" != test ]]; then submit tune tune afterok 0; fi
submit test test afterok 0
echo 'Submitted sequential continuation. At most ONE GPU runs in this chain.'
echo "Job IDs: $WORK/jobs.tsv"
echo 'Three hours per job is a limit, not a guarantee that each phase will finish.'
