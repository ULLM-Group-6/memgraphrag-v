#!/bin/bash
# Download the WebQA corpus images (4,221 files, a few GB) and write the
# manifests. Wikimedia rate-limits downloads to roughly 10 images a minute,
# so this takes about 7 hours. Needs internet, so it runs on the staging partition; running
#   python -m memgraphrag_v.webqa download && python -m memgraphrag_v.webqa manifests
# on a login node works too. Resumable: re-submit to retry what failed.
#   sbatch scripts/webqa_download.sbatch
# Needs: MGRV_ARTIFACTS exported, Misha's files in $MGRV_ARTIFACTS/webqa/
# (see docs/interfaces.md) and a venv at .venv-eva.
#SBATCH --job-name=webqa-download
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=webqa-download-%j.log

set -euo pipefail
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "$SLURM_SUBMIT_DIR"
source .venv-eva/bin/activate
: "${MGRV_ARTIFACTS:?export MGRV_ARTIFACTS before sbatch}"

python -u -m memgraphrag_v.webqa download
python -u -m memgraphrag_v.webqa manifests
