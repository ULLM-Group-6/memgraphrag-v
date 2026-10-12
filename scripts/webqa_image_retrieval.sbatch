#!/bin/bash
# System D on the WebQA subset: embed every corpus image and question with
# EVA-CLIP, run direct retrieval on dev and test, and write the run reports.
#   sbatch scripts/webqa_image_retrieval.sbatch
# Needs: MGRV_ARTIFACTS exported, the models downloaded
# (scripts/download_eva_clip.py), the images and manifests from
# scripts/webqa_download.sbatch, and a venv at .venv-eva.
# Read the log's "image_recall_at_5" per split, then open
# $MGRV_ARTIFACTS/runs/d-{dev,test}/report.html.
#SBATCH --job-name=webqa-d
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=webqa-d-%j.log

set -euo pipefail
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "$SLURM_SUBMIT_DIR"
source .venv-eva/bin/activate
: "${MGRV_ARTIFACTS:?export MGRV_ARTIFACTS before sbatch}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

nvidia-smi
srun python -u -m memgraphrag_v.embeddings image
srun python -u -m memgraphrag_v.embeddings query
for split in dev test; do
  python -u -m memgraphrag_v.image_retrieval --run-id "d-$split" --split "$split"
  python -u scripts/show_retrieval.py --run-id "d-$split"
done
