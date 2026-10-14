#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:40:00
#SBATCH --job-name=setup_mg
#SBATCH --output=slurm_%x_%A.out
#SBATCH --error=slurm_%x_%A.err

module purge; module load 2023
module load Python/3.11.3-GCCcore-12.3.0
python -m venv ~/venvs/memgraph && source ~/venvs/memgraph/bin/activate
pip install -q --upgrade pip

cd ~/snellius-work/MemGraphRAG
grep -v -i vllm requirements.txt > ~/snellius-work/req_memgraph.txt   # vllm is only for offline OpenIE
pip install -r ~/snellius-work/req_memgraph.txt || echo "REQ INSTALL FAILED"
pip install gritlm || echo "GRITLM INSTALL FAILED"

echo "== versions"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== import test"
python -c "import sys; sys.path.insert(0,'.'); import src.MemGraphRAG; print('import OK')" || echo "IMPORT FAILED"

echo "== script options"
python code/index.py --help 2>&1 | head -60
python code/retrieval_dataset_test.py --help 2>&1 | head -60
echo "== done"
