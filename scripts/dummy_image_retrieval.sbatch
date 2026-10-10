#!/bin/bash
# End-to-end test of whole-image retrieval with real EVA-CLIP on the synthetic
# dummy dataset (memgraphrag_v/dummy.py): build it, embed images and questions,
# run direct retrieval. EVA-CLIP should put the gold document first for
# (nearly) all 12 questions; see "gold_doc_at_1" at the end of the log.
#   sbatch scripts/dummy_image_retrieval.sbatch
# Needs: MGRV_ARTIFACTS exported with the models downloaded
# (scripts/download_eva_clip.py) and a venv at .venv-eva. Writes to
# $MGRV_ARTIFACTS/dummy, leaving the real artifacts untouched.
#SBATCH --job-name=eva-dummy
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=eva-dummy-%j.log

set -euo pipefail
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "$SLURM_SUBMIT_DIR"
source .venv-eva/bin/activate
: "${MGRV_ARTIFACTS:?export MGRV_ARTIFACTS before sbatch}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

DUMMY="$MGRV_ARTIFACTS/dummy"
rm -rf "$DUMMY"
python -m memgraphrag_v.dummy "$DUMMY"
ln -s "$MGRV_ARTIFACTS/models" "$DUMMY/models"
export MGRV_ARTIFACTS="$DUMMY"

nvidia-smi
srun python -u -m memgraphrag_v.embeddings image
srun python -u -m memgraphrag_v.embeddings query
python -u -m memgraphrag_v.image_retrieval --run-id dummy-direct
python -u scripts/show_retrieval.py --run-id dummy-direct
