#!/bin/bash
#SBATCH --job-name=eva-smoke
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/eva-smoke-%j.log

set -e
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
cd "$SLURM_SUBMIT_DIR"
source .venv-eva/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
nvidia-smi
python -m pip freeze > "outputs/eva-environment-${SLURM_JOB_ID}.txt"
srun python -u scripts/smoke_eva_clip.py --output "outputs/eva_smoke_${SLURM_JOB_ID}"
