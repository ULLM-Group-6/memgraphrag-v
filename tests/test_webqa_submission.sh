#!/bin/bash
# CPU-only submission test: all scheduler commands below are fake; no real jobs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
# Keep the tiny temporary fixture for inspection; no recursive cleanup commands.
mkdir -p "$TMP/bin" "$TMP/artifacts/index" "$TMP/repo/scripts"
cp "$ROOT/scripts/submit_webqa_parallel.sh" "$TMP/repo/scripts/"
touch "$TMP/repo/scripts/webqa_parallel_gpu.sbatch" "$TMP/artifacts/index/manifest.json" "$TMP/config.yaml"
cat > "$TMP/bin/sbatch" <<'MOCK'
#!/bin/bash
n=$(cat "$MOCK_ROOT/counter" 2>/dev/null || echo 1000)
n=$((n + 1))
echo "$n" > "$MOCK_ROOT/counter"
printf '%s\n' "$*" >> "$MOCK_ROOT/calls"
echo "$n"
MOCK
cat > "$TMP/bin/squeue" <<'MOCK'
#!/bin/bash
if [[ -f "$MOCK_ROOT/active" ]]; then cat "$MOCK_ROOT/active"; fi
MOCK
cat > "$TMP/bin/flock" <<'MOCK'
#!/bin/bash
exit 0
MOCK
chmod +x "$TMP/bin/"*
export MOCK_ROOT="$TMP" PATH="$TMP/bin:$PATH"
export USER=${USER:-test-user}
export MGRV_ARTIFACTS="$TMP/artifacts" MGRV_RUN_CONFIG="$TMP/config.yaml"
export MGRV_TEXT_PYTHON="$TMP/bin/sbatch" MGRV_VISION_PYTHON="$TMP/bin/sbatch"
export MGRV_GROUND_PYTHON="$TMP/bin/sbatch" MGRV_READER_PYTHON="$TMP/bin/sbatch"
export MGRV_TEXT_EMBEDDING_MODEL=x MGRV_MEMGRAPHRAG=x MGRV_MG2_RAG=x
export MGRV_SAM3_CHECKPOINT=x MGRV_READER_MODEL=x
cd "$TMP/repo"
bash scripts/submit_webqa_parallel.sh dev test-run > "$TMP/submission-output"
test "$(wc -l < "$TMP/calls")" -eq 8
test "$(grep -c -- '--array=0-1%2' "$TMP/calls")" -eq 3
test "$(grep -c -- '--kill-on-invalid-dep=yes' "$TMP/calls")" -eq 8
! head -n 1 "$TMP/calls" | grep -q -- '--dependency'
for n in $(seq 2 8); do
  previous=$((999 + n))
  sed -n "${n}p" "$TMP/calls" | grep -q -- "--dependency=afterok:$previous "
done
echo 1002 > "$TMP/active"
if bash scripts/submit_webqa_parallel.sh dev test-run > "$TMP/duplicate-output" 2>&1; then
  echo 'ERROR: duplicate active chain was allowed'; exit 1
fi
test "$(wc -l < "$TMP/calls")" -eq 8
: > "$TMP/active"
bash scripts/submit_webqa_parallel.sh dev test-run qa > "$TMP/resume-output"
test "$(wc -l < "$TMP/calls")" -eq 11
tail -n 3 "$TMP/calls" | head -n 1 | grep -q 'prepare dev test-run'
# Automatic chain: each beta's retrieval collection gates the next candidate;
# selection must succeed before test workers can run.
: > "$TMP/calls"
echo 2000 > "$TMP/counter"
bash scripts/submit_webqa_parallel.sh auto auto-run > "$TMP/auto-output"
test "$(wc -l < "$TMP/calls")" -eq 15
test "$(grep -c -- '--array=0-1%2' "$TMP/calls")" -eq 6
test "$(grep -c -- 'dev-retrieve dev auto-run' "$TMP/calls")" -eq 3
test "$(grep -c -- 'dev-collect dev auto-run' "$TMP/calls")" -eq 3
sed -n '1p' "$TMP/calls" | grep -q 'prepare-auto dev auto-run'
sed -n '7p' "$TMP/calls" | grep -q 'dev-retrieve dev auto-run 0'
sed -n '9p' "$TMP/calls" | grep -q 'dev-retrieve dev auto-run 1'
sed -n '11p' "$TMP/calls" | grep -q 'dev-retrieve dev auto-run 2'
sed -n '13p' "$TMP/calls" | grep -q 'select-beta dev auto-run'
sed -n '14p' "$TMP/calls" | grep -q 'test-qa dev auto-run'
sed -n '15p' "$TMP/calls" | grep -q 'test-collect dev auto-run'
for n in $(seq 2 15); do
  previous=$((1999 + n))
  sed -n "${n}p" "$TMP/calls" | grep -q -- "--dependency=afterok:$previous "
done
test "$(grep -c -- '--kill-on-invalid-dep=yes' "$TMP/calls")" -eq 15
: > "$TMP/calls"
bash scripts/submit_webqa_parallel.sh auto auto-run test > "$TMP/test-resume-output"
test "$(wc -l < "$TMP/calls")" -eq 3
! grep -q 'dev-retrieve' "$TMP/calls"
# A still-active job recorded by the old bundle must also block submission.
mkdir -p "$TMP/artifacts/index/parallel-v1"
printf 'ground\t9999\n' > "$TMP/artifacts/index/parallel-v1/jobs.tsv"
echo 9999 > "$TMP/active"
if bash scripts/submit_webqa_parallel.sh auto another-run > "$TMP/old-chain-output" 2>&1; then
  echo 'ERROR: active v1 chain was ignored'; exit 1
fi
test "$(wc -l < "$TMP/calls")" -eq 3
echo "Submission tests passed: automatic dev tuning/test dependencies, two-GPU cap, legacy mode, resume, duplicate/old-chain refusal. Fixture: $TMP"
