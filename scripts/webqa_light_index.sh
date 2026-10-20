#!/bin/bash
# sbatch --export=ALL scripts/webqa_light_index.sh LARGE_INDEX_JOB_ID [RUN_ID]
# One-GPU alternative; isolated outputs; stop when the watched job starts.
#SBATCH --job-name=webqa-light
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=webqa-light-%j.log
set -euo pipefail
WATCH_ID=${1:?Supply the already submitted LARGE INDEX job ID}
RUN_ID=${2:-webqa-tuned-v1}
[[ "$WATCH_ID" =~ ^[0-9]+$ ]] || { echo 'Expected a numeric job/array ID'; exit 2; }
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo 'Invalid run ID'; exit 2; }
: "${SLURM_JOB_ID:?Submit this script with sbatch, not bash}"
[[ "$WATCH_ID" != "$SLURM_JOB_ID" ]] || { echo 'Cannot watch itself'; exit 2; }
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
: "${MGRV_ARTIFACTS:?Set the EXISTING original artifact directory}"
: "${MGRV_TEXT_PYTHON:?}"
: "${MGRV_VISION_PYTHON:?}"
: "${MGRV_TEXT_EMBEDDING_MODEL:?}"
: "${MGRV_MEMGRAPHRAG:?}"
: "${MGRV_SAM3_CHECKPOINT:?}"
export MGRV_RUN_CONFIG=${MGRV_RUN_CONFIG:-$PWD/configs/webqa_run.yaml}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
export OMP_NUM_THREADS=9 MKL_NUM_THREADS=9 OPENBLAS_NUM_THREADS=9 NUMEXPR_NUM_THREADS=9
for executable in "$MGRV_TEXT_PYTHON" "$MGRV_VISION_PYTHON"; do test -x "$executable"; done
for utility in rsync squeue sacct scancel srun realpath; do command -v "$utility" >/dev/null; done
test -f scripts/webqa_parallel.py
test -f "$MGRV_RUN_CONFIG"
SOURCE=$(realpath -e "$MGRV_ARTIFACTS")
DEST=$(realpath -m "${MGRV_LIGHT_ARTIFACTS:-${SOURCE}-light-${SLURM_JOB_ID}}")
case "$DEST/" in "$SOURCE/"*) echo 'Light output must be outside the original artifact directory'; exit 2 ;; esac
case "$SOURCE/" in "$DEST/"*) echo 'Light output must not contain the original directory'; exit 2 ;; esac
[[ ! -e "$DEST" ]] || { echo "Destination already exists; refusing to overwrite: $DEST"; exit 2; }
test -f "$SOURCE/index/manifest.json"
test -d "$SOURCE/index/grounding"
"$MGRV_TEXT_PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); assert "ground" in m["stages"], "Grounding must already be finalized"' "$SOURCE/index/manifest.json"

# Return 0 if the large job has started, 1 if still pending or cancelled before
# starting, and 2 if its state/ownership cannot be established. Never cancel it.
large_started() {
  local rows owner state started found=0 pending=0
  if ! rows=$(squeue -h -j "$WATCH_ID" -o '%u|%T' 2>/dev/null); then rows=; fi
  if [[ -n "$rows" ]]; then
    while IFS='|' read -r owner state; do
      [[ "$owner" == "$USER" ]] || return 2
      case "$state" in
        RUNNING|COMPLETING|SUSPENDED|CONFIGURING) return 0 ;;
        PENDING) pending=1 ;;
        *) return 2 ;;
      esac
    done <<< "$rows"
    (( pending == 1 )) && return 1
  fi
  # Also catch a job that started and finished between checks. -X omits steps.
  rows=$(sacct -X -n -P -j "$WATCH_ID" --format=User,State,Start 2>/dev/null) || return 2
  while IFS='|' read -r owner state started; do
    [[ -n "$owner" ]] || continue
    [[ "$owner" == "$USER" ]] || return 2
    found=1
    case "$state" in
      PENDING*) ;;
      RUNNING*|COMPLETING*|COMPLETED*|SUSPENDED*|CONFIGURING*) return 0 ;;
      *) [[ -n "$started" && "$started" != Unknown && "$started" != None && "$started" != N/A ]] && return 0 ;;
    esac
  done <<< "$rows"
  (( found == 1 )) && return 1
  return 2
}
check_large() {
  local rc=0
  large_started || rc=$?
  case "$rc" in
    0) echo "$(date -Is) Large job $WATCH_ID has started; yielding to it."; return 1 ;;
    1) return 0 ;;
    *) echo "Cannot verify job $WATCH_ID and its owner; stopping rather than running unmonitored." >&2; return 1 ;;
  esac
}
# If the larger allocation won first, exit before copying or loading models.
check_large || exit 0

MAIN_PID=$BASHPID
ACTIVE_PID=
MONITOR_PID=
cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; fi
  if [[ -n "$ACTIVE_PID" ]]; then kill "$ACTIVE_PID" 2>/dev/null || true; wait "$ACTIVE_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
monitor() {
  while sleep 60; do
    if ! check_large; then
      echo "$(date -Is) Cancelling ONLY light job $SLURM_JOB_ID. Original job remains untouched."
      scancel "$SLURM_JOB_ID" || true
      # Fallback if cancellation could not be delivered immediately.
      kill -TERM "$MAIN_PID" 2>/dev/null || true
      return
    fi
  done
}
monitor &
MONITOR_PID=$!
run_command() {
  local rc=0
  check_large || exit 0
  "$@" &
  ACTIVE_PID=$!
  wait "$ACTIVE_PID" || rc=$?
  ACTIVE_PID=
  return "$rc"
}

echo "$(date -Is) Original artifacts: $SOURCE"
echo "$(date -Is) ISOLATED light artifacts: $DEST"
echo "Watching larger job $WATCH_ID every 60 seconds. Copying consumes part of the three-hour limit."
mkdir "$DEST"
# Copy every mutable artifact, dereferencing links so writes cannot reach the
# original index. Share only inputs that this embedding/assembly run never writes.
run_command rsync -aL \
  --exclude='/corpus' --exclude='/models' --exclude='/webqa' --exclude='/runs' \
  --exclude='/index/grounding' --exclude='/index/parallel-v2/jobs*' \
  --exclude='*.lock' "$SOURCE/" "$DEST/"
for relative in corpus models webqa index/grounding; do
  if [[ -e "$SOURCE/$relative" ]]; then
    mkdir -p "$(dirname "$DEST/$relative")"
    ln -s "$(realpath -e "$SOURCE/$relative")" "$DEST/$relative"
  fi
done
check_large || exit 0
export MGRV_ARTIFACTS="$DEST"
printf 'source=%s\nwatched_job=%s\nlight_job=%s\n' "$SOURCE" "$WATCH_ID" "$SLURM_JOB_ID" > "$DEST/light-origin.txt"
LOGS="$SLURM_SUBMIT_DIR/logs/webqa-light-$SLURM_JOB_ID"
mkdir -p "$LOGS"
step() {
  local action=$1 worker=${2:-0} python="$MGRV_TEXT_PYTHON"
  [[ "$action" != embed ]] || python="$MGRV_VISION_PYTHON"
  echo "$(date -Is) START $action worker=$worker; log=$LOGS/$action-$worker.log"
  run_command srun --exact --nodes=1 --ntasks=1 --cpus-per-task=9 \
    --gpus=1 --gpus-per-task=1 --gpu-bind=single:1 --mem=96G \
    --kill-on-bad-exit=1 --output="$LOGS/$action-$worker.log" --error="$LOGS/$action-$worker.log" \
    "$python" scripts/webqa_parallel.py "$action" \
    --config "$MGRV_RUN_CONFIG" --artifacts "$DEST" \
    --worker "$worker" --split dev --run-id "$RUN_ID"
  echo "$(date -Is) DONE $action worker=$worker"
}
# Same frozen plan and algorithms. No text/ground commands and no --force.
step prepare-auto
step embed 0
step embed 1
step embed-finalize
step assemble
echo "$(date -Is) INDEX COMPLETE in $DEST"
echo 'No QA was submitted. Do not merge these files into an index another job is writing.'
