#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:20:00
#SBATCH --job-name=dl_test
#SBATCH --output=slurm_%x_%A.out
#SBATCH --error=slurm_%x_%A.err

echo "== host: $(hostname)  job: $SLURM_JOB_ID"

echo "== internet checks"
curl -sS -m 15 -o /dev/null -w "huggingface.co: %{http_code}\n" https://huggingface.co || echo "huggingface.co: FAIL"
curl -sS -m 15 -o /dev/null -w "pypi.org: %{http_code}\n" https://pypi.org/simple/ || echo "pypi.org: FAIL"
curl -sS -m 15 -o /dev/null -w "github.com: %{http_code}\n" https://github.com || echo "github.com: FAIL"

module purge; module load 2023
module load Python/3.11.3-GCCcore-12.3.0

python -m venv ~/venvs/dl && source ~/venvs/dl/bin/activate
pip install -q huggingface_hub || echo "PIP INSTALL FAILED"

export HF_HOME=$HOME/hf_cache
python - << 'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("BAAI/bge-large-en-v1.5",
    allow_patterns=["*.safetensors","*.json","*.txt"])
print("downloaded to", p)
PY
du -sh $HOME/hf_cache
echo "== done"
