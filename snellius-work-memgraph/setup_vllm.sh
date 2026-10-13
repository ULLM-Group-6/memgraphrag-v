#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:40:00
#SBATCH --job-name=setup_vllm
#SBATCH --output=slurm_%x_%A.out
#SBATCH --error=slurm_%x_%A.err

module purge; module load 2023
module load Python/3.11.3-GCCcore-12.3.0
python -m venv ~/venvs/vllm && source ~/venvs/vllm/bin/activate
pip install -q --upgrade pip
pip install vllm huggingface_hub || echo "VLLM INSTALL FAILED"

export HF_HOME=$HOME/hf_cache
python - << 'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-7B-Instruct",
    allow_patterns=["*.safetensors","*.json","*.txt"])
print("downloaded to", p)
PY
python -c "import vllm; print('vllm', vllm.__version__)" || echo "VLLM IMPORT FAILED"
du -sh $HOME/hf_cache
echo "== done"
