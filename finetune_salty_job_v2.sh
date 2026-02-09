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

# Load CUDA only if your cluster requires it for driver libraries
module load cuda/11.1

# Activate your venv (IMPORTANT)
source /home/t/tayhan/Finetuning/finetuning_methodology/myenv/bin/activate

# Sanity checks: show python + torch + GPU visibility
echo "=== Python being used ==="
which python
python -c "import sys; print(sys.version)"

echo "=== CUDA_VISIBLE_DEVICES ==="
echo "${CUDA_VISIBLE_DEVICES:-<not set>}"

echo "=== Torch CUDA check ==="
python -c "import torch; \
print('torch:', torch.__version__); \
print('torch.cuda.is_available:', torch.cuda.is_available()); \
print('torch.cuda.device_count:', torch.cuda.device_count()); \
print('torch.version.cuda:', torch.version.cuda)"

echo "=== nvidia-smi ==="
nvidia-smi

# Run training
python finetuning_salty_v2.py
