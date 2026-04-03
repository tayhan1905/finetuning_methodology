#!/bin/bash
#SBATCH --job-name=finetune-salty-v3          # Job name
#SBATCH --output=logs/%x_%j.out               # Stdout log  (logs/finetune-salty-v3_JOBID.out)
#SBATCH --error=logs/%x_%j.err                # Stderr log  (logs/finetune-salty-v3_JOBID.err)
#SBATCH --ntasks=1                            # Single task
#SBATCH --cpus-per-task=4                     # CPU cores per task
#SBATCH --gres=gpu:nv:1                       # 1 GPU  (change to a100-40, h100-47, etc.)
#SBATCH -t 3-00:00:00                         # Max walltime: 3 days (QNLI is large)
#SBATCH --mem=32G                             # Memory — bumped to 32G for 3-dataset sweep
#SBATCH --partition=gpu                       # GPU partition

# ---------------------------------------------------------------------------
# finetuning_salty_v3.py — Dataset Sweep
# Trains SALTEDORA-V4 + full-FT on SST-2, RTE, and QNLI.
# Saves per-epoch QKV weights and runs principal-angles analysis for each task.
#
# Results land in:
#   results_v3/bert-base-uncased/<mode>/r_8/et_0.90/<task>/
#   results_v3/principal_angles/v3/<task>/
#   results_v3/summary_v3_dataset_sweep.csv
# ---------------------------------------------------------------------------

# Create logs dir if it doesn't exist
mkdir -p logs
mkdir -p results_v3/bert-base-uncased
mkdir -p results_v3/principal_angles/v3

# Load required modules (adjust versions to your cluster)
module load python/3.10
module load cuda/12.1

# Activate virtual environment if you use one
# source venv/bin/activate
# source ~/envs/myenv/bin/activate

# Install dependencies if needed (uncomment on first run)
# pip install -r requirements.txt

# Run the dataset sweep script
srun python finetuning_salty_v3.py