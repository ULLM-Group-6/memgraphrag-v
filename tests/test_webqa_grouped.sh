#!/bin/bash
# CPU-only: fake scheduler and fake model interpreters. Never submits real jobs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
mkdir -p "$TMP/bin" "$TMP/repo/scripts" "$TMP/artifacts/index/parallel-v2"
cp "$ROOT"/scripts/*grouped* "$TMP/repo/scripts/"
touch "$TMP/repo/scripts/webqa_parallel.py" "$TMP/repo/scripts/webqa_run.py" \
  "$TMP/repo/scripts/webqa_tuning.py" "$TMP/artifacts/index/manifest.json" "$TMP/config.yaml"
cat > "$TMP/bin/sbatch" <<'MOCK'
#!/bin/bash
if [[ -n "${FAIL_SUBMIT:-}" && "$*" == *'webqa-grp-qa'* ]]; then exit 1; fi
n=$(cat "$MOCK_ROOT/counter" 2>/dev/null || echo 1000)
n=$((n + 1))
echo "$n" > "$MOCK_ROOT/counter"
printf '%s\n' "$*" >> "$MOCK_ROOT/submissions"
echo "$n"
MOCK
cat > "$TMP/bin/squeue" <<'MOCK'
#!/bin/bash
if [[ -f "$MOCK_ROOT/active" ]]; then cat "$MOCK_ROOT/active"; fi
MOCK
cat > "$TMP/bin/module" <<'MOCK'
#!/bin/bash
exit 0
MOCK
# Locks are stubbed for portability (Git Bash has no flock). Linux lock
# semantics and actual Slurm/GPU binding are not exercised by this test.
cat > "$TMP/bin/flock" <<'MOCK'
#!/bin/bash
exit 0
MOCK
cat > "$TMP/bin/srun" <<'MOCK'
#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >> "$MOCK_ROOT/steps"
tasks=0
while [[ "$1" != bash ]]; do
  case "$1" in --ntasks=*) tasks=${1#*=} ;; esac
  shift
done
[[ "$tasks" == 1 || "$tasks" == 2 ]]
pids=()
for ((w=0; w<tasks; w++)); do
  SLURM_PROCID=$w CUDA_VISIBLE_DEVICES=$w "$@" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
exit "$status"
MOCK
cat > "$TMP/bin/text-python" <<'MOCK'
#!/bin/bash
set -euo pipefail
action=$2
printf '%s|%s|%s|%s|%s\n' "${0##*/}" "$action" "${SLURM_PROCID:-0}" \
  "${CUDA_VISIBLE_DEVICES:-none}" "$*" >> "$MOCK_ROOT/actions"
if [[ "$action" == "${FAIL_ACTION:-}" && "${SLURM_PROCID:-0}" == "${FAIL_WORKER:-1}" ]]; then exit 7; fi
MOCK
cp "$TMP/bin/text-python" "$TMP/bin/vision-python"
chmod +x "$TMP/bin/"*
export MOCK_ROOT="$TMP" PATH="$TMP/bin:$PATH" USER=${USER:-test-user}
export MGRV_ARTIFACTS="$TMP/artifacts" MGRV_RUN_CONFIG="$TMP/config.yaml"
export MGRV_TEXT_PYTHON="$TMP/bin/text-python" MGRV_VISION_PYTHON="$TMP/bin/vision-python"
export MGRV_READER_PYTHON="$TMP/bin/text-python"
export MGRV_TEXT_EMBEDDING_MODEL=x MGRV_MEMGRAPHRAG=x MGRV_MG2_RAG=x
export MGRV_SAM3_CHECKPOINT=x MGRV_READER_MODEL=x
export SLURM_SUBMIT_DIR="$TMP/repo" SLURM_JOB_ID=9000 SLURM_CPUS_PER_TASK=18
cd "$TMP/repo"
for file in scripts/*grouped*; do bash -n "$file"; done
bash scripts/submit_webqa_grouped.sh webqa-tuned-v1 > "$TMP/submit.log"
test "$(wc -l < "$TMP/submissions")" -eq 2
head -n 1 "$TMP/submissions" | grep -q 'index webqa-tuned-v1 embed'
tail -n 1 "$TMP/submissions" | grep -q -- '--dependency=afterok:1001'
tail -n 1 "$TMP/submissions" | grep -q 'qa webqa-tuned-v1 tune'
test "$(wc -l < "$MGRV_ARTIFACTS/index/parallel-v2/jobs.tsv")" -eq 2
echo 1002 > "$TMP/active"
if bash scripts/submit_webqa_grouped.sh webqa-tuned-v1 > "$TMP/refusal.log" 2>&1; then
  echo 'ERROR: active chain allowed'; exit 1
fi
: > "$TMP/active"
# Index: successful pair must finish before finalization and assembly.
bash scripts/webqa_grouped.sbatch index webqa-tuned-v1 embed > "$TMP/index.log"
test "$(wc -l < "$TMP/steps")" -eq 4
grep -q 'vision-python|embed|0|0|' "$TMP/actions"
grep -q 'vision-python|embed|1|1|' "$TMP/actions"
tail -n 2 "$TMP/actions" | head -n 1 | grep -q '|embed-finalize|'
tail -n 1 "$TMP/actions" | grep -q '|assemble|'
! grep -q '|ground' "$TMP/actions"
grep -q -- '--ntasks=2.*--gpus-per-task=1.*--gpu-bind=single:1.*--mem=192G' "$TMP/steps"
# QA: all three candidates, collection, selection, then test and final collection.
: > "$TMP/steps"
: > "$TMP/actions"
bash scripts/webqa_grouped.sbatch qa webqa-tuned-v1 tune > "$TMP/qa.log"
test "$(wc -l < "$TMP/steps")" -eq 10
test "$(grep -c '|dev-retrieve|' "$TMP/actions")" -eq 6
test "$(grep -c '|dev-collect|' "$TMP/actions")" -eq 3
test "$(grep -c '|test-qa|' "$TMP/actions")" -eq 2
for c in 0 1 2; do
  test "$(grep '|dev-retrieve|' "$TMP/actions" | grep -c -- "--candidate $c")" -eq 2
done
sed -n '8p' "$TMP/steps" | grep -q 'select-beta'
sed -n '9p' "$TMP/steps" | grep -q 'test-qa'
sed -n '10p' "$TMP/steps" | grep -q 'test-collect'
# Either worker's error blocks finalization and all later actions.
for worker in 0 1; do
  : > "$TMP/actions"
  if FAIL_ACTION=embed FAIL_WORKER=$worker bash scripts/webqa_grouped.sbatch index run embed > "$TMP/fail-$worker.log" 2>&1; then
    echo 'ERROR: worker failure ignored'; exit 1
  fi
  ! grep -q '|embed-finalize|' "$TMP/actions"
  ! grep -q '|assemble|' "$TMP/actions"
done
: > "$TMP/actions"
if FAIL_ACTION=select-beta FAIL_WORKER=0 bash scripts/webqa_grouped.sbatch qa run tune > "$TMP/fail-selection.log" 2>&1; then
  echo 'ERROR: selection failure ignored'; exit 1
fi
! grep -q '|test-qa|' "$TMP/actions"
# Resume submits QA only and skips tuning when requested.
: > "$TMP/submissions"
MGRV_GROUPED_QA_TIME=04:00:00 bash scripts/submit_webqa_grouped.sh webqa-tuned-v1 test > "$TMP/resume.log"
test "$(wc -l < "$TMP/submissions")" -eq 1
grep -q -- '--time=04:00:00' "$TMP/submissions"
! grep -q -- '--dependency=' "$TMP/submissions"
: > "$TMP/actions"
bash scripts/webqa_grouped.sbatch qa webqa-tuned-v1 test > "$TMP/test.log"
! grep -q '|dev-retrieve|' "$TMP/actions"
test "$(grep -c '|test-qa|' "$TMP/actions")" -eq 2
# A partial submission retains the accepted job in the common registry.
if FAIL_SUBMIT=1 bash scripts/submit_webqa_grouped.sh webqa-tuned-v1 > "$TMP/partial.log" 2>&1; then
  echo 'ERROR: second submission failure ignored'; exit 1
fi
test "$(wc -l < "$MGRV_ARTIFACTS/index/parallel-v2/jobs.tsv")" -eq 1
grep -q 'grouped-index' "$MGRV_ARTIFACTS/index/parallel-v2/jobs.tsv"
echo "Grouped checks passed: submission, two workers, ordering, failure gates, resume, partial submission. Fixture: $TMP"
