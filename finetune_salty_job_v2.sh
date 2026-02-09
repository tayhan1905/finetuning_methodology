#!/bin/bash
#SBATCH --job-name=finetune-salty
#SBATCH --output=finetune-salty.out
#SBATCH --error=finetune-salty.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -t 3-00:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu-long

set -euo pipefail

source /home/t/tayhan/Finetuning/finetuning_methodology/myenv/bin/activate

echo "Python:" $(which python)
python -c "import torch; print('cuda avail:', torch.cuda.is_available()); print('count:', torch.cuda.device_count()); print('torch cuda:', torch.version.cuda)"
nvidia-smi

python finetuning_salty_v2.py
