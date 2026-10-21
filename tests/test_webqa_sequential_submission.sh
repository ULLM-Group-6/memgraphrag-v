#!/bin/bash
# Fake scheduler only. Does not submit jobs or load models.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
mkdir -p "$TMP/bin" "$TMP/repo/scripts" "$TMP/artifacts/index/parallel-v2"
cp "$ROOT"/scripts/*.py "$ROOT"/scripts/*.sh "$ROOT"/scripts/*.sbatch "$TMP/repo/scripts/"
touch "$TMP/repo/scripts/webqa_parallel.py" "$TMP/repo/scripts/webqa_run.py" \
  "$TMP/repo/scripts/webqa_tuning.py" "$TMP/artifacts/index/manifest.json" "$TMP/config.yaml"
cat > "$TMP/bin/sbatch" <<'MOCK'
#!/bin/bash
n=$(cat "$MOCK_ROOT/counter" 2>/dev/null || echo 1000)
n=$((n + 1)); echo "$n" > "$MOCK_ROOT/counter"
if [[ "${MOCK_FAIL_AT:-}" == "$n" ]]; then exit 1; fi
echo "$*" >> "$MOCK_ROOT/calls"
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
cat > "$TMP/bin/fake-python" <<'MOCK'
#!/bin/bash
echo "$*" >> "$MOCK_ROOT/python-calls"
MOCK
chmod +x "$TMP/bin/"*
export MOCK_ROOT="$TMP" PATH="$TMP/bin:$PATH" USER=${USER:-test-user}
export MGRV_ARTIFACTS="$TMP/artifacts" MGRV_RUN_CONFIG="$TMP/config.yaml"
export MGRV_TEXT_PYTHON="$TMP/bin/fake-python" MGRV_VISION_PYTHON="$TMP/bin/fake-python"
export MGRV_READER_PYTHON="$TMP/bin/fake-python" MGRV_TEXT_EMBEDDING_MODEL=x MGRV_MEMGRAPHRAG=x
export MGRV_MG2_RAG=x MGRV_SAM3_CHECKPOINT=x MGRV_READER_MODEL=x
unset SLURM_JOB_ID
cd "$TMP/repo"
bash -n scripts/submit_webqa_sequential.sh
bash -n scripts/webqa_sequential.sbatch
bash scripts/submit_webqa_sequential.sh webqa-tuned-v1 > "$TMP/output"
test "$(wc -l < "$TMP/calls")" -eq 5
! sed -n '1p' "$TMP/calls" | grep -q -- '--dependency='
sed -n '2p' "$TMP/calls" | grep -Fq -- '--dependency=afternotok:1001'
sed -n '3p' "$TMP/calls" | grep -Fq -- '--dependency=afterok:1001?afterok:1002'
sed -n '4p' "$TMP/calls" | grep -q -- '--dependency=afterok:1003'
sed -n '5p' "$TMP/calls" | grep -q -- '--dependency=afterok:1004'
sed -n '1p' "$TMP/calls" | grep -q -- '--time=05:00:00'
sed -n '2p' "$TMP/calls" | grep -q -- '--time=05:00:00'
sed -n '3p' "$TMP/calls" | grep -q -- '--time=06:00:00'
sed -n '2p' "$TMP/calls" | grep -q '1001 2$'
sed -n '3p' "$TMP/calls" | grep -q 'webqa_sequential.sbatch assemble '
test "$(grep -c -- '--cpus-per-task=9 --gpus-per-node=1 --mem=96G' "$TMP/calls")" -eq 5
test "$(grep -c -- '--time=03:00:00' "$TMP/calls")" -eq 2
test "$(grep -c -- '--kill-on-invalid-dep=yes' "$TMP/calls")" -eq 5
test "$(grep -c -- '--signal=B:USR1@120' "$TMP/calls")" -eq 5
test "$(wc -l < "$MGRV_ARTIFACTS/index/parallel-v2/jobs.tsv")" -eq 5
echo 1001 > "$TMP/active"
if bash scripts/submit_webqa_sequential.sh webqa-tuned-v1 > "$TMP/refused" 2>&1; then
  echo 'ERROR: active old chain was ignored'; exit 1
fi
: > "$TMP/active"
: > "$TMP/calls"
bash scripts/submit_webqa_sequential.sh webqa-tuned-v1 assemble > "$TMP/assemble"
test "$(wc -l < "$TMP/calls")" -eq 3
sed -n '1p' "$TMP/calls" | grep -q 'webqa_sequential.sbatch assemble '
! sed -n '1p' "$TMP/calls" | grep -q -- '--dependency='
: > "$TMP/calls"
bash scripts/submit_webqa_sequential.sh webqa-tuned-v1 tune > "$TMP/tune"
test "$(wc -l < "$TMP/calls")" -eq 2
sed -n '1p' "$TMP/calls" | grep -q 'webqa_sequential.sbatch tune '
sed -n '2p' "$TMP/calls" | grep -q -- '--dependency=afterok:'
: > "$TMP/calls"
bash scripts/submit_webqa_sequential.sh webqa-tuned-v1 test > "$TMP/test"
test "$(wc -l < "$TMP/calls")" -eq 1
grep -q 'webqa_sequential.sbatch test ' "$TMP/calls"
# A mid-chain submission failure preserves IDs of the jobs already accepted.
export MOCK_FAIL_AT=$(( $(cat "$TMP/counter") + 2 ))
if bash scripts/submit_webqa_sequential.sh webqa-tuned-v1 > "$TMP/partial" 2>&1; then
  echo 'ERROR: submission failure ignored'; exit 1
fi
test "$(wc -l < "$MGRV_ARTIFACTS/index/parallel-v2/jobs.tsv")" -eq 1
echo "Submission checks passed: 5h/5h/6h/3h/3h, conditional backup, OR success dependency, resume modes, active/partial-chain guards. Fixture: $TMP"
