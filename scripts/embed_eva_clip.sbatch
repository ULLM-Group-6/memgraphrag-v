#!/bin/bash
# Embed corpus images and questions with EVA-CLIP on one A100.
#   sbatch scripts/embed_eva_clip.sbatch            # everything
#   sbatch scripts/embed_eva_clip.sbatch --limit 20 # smoke run
# Needs: MGRV_ARTIFACTS exported, models downloaded (scripts/download_eva_clip.py),
# and a venv at .venv-eva with `pip install -e ".[eva]"`.
#SBATCH --job-name=eva-embed
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=eva-embed-%j.log

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
srun python -u -m memgraphrag_v.embeddings image "$@"
srun python -u -m memgraphrag_v.embeddings query "$@"
