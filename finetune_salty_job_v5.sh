#!/bin/bash
#SBATCH --job-name=finetune-salty-v5          # Job name
#SBATCH --output=logs/%x_%j.out               # Stdout log  (logs/finetune-salty-v5_JOBID.out)
#SBATCH --error=logs/%x_%j.err                # Stderr log  (logs/finetune-salty-v5_JOBID.err)
#SBATCH --ntasks=1                            # Single task
#SBATCH --cpus-per-task=4                     # CPU cores per task
#SBATCH --gres=gpu:nv:1                       # 1 GPU  (change to a100-40, h100-47, etc.)
#SBATCH -t 3-00:00:00                         
#SBATCH --mem=16G                             # Memory — SST-2 only, 16G is sufficient
#SBATCH --partition=gpu                       # GPU partition

# ---------------------------------------------------------------------------
# finetuning_salty_v5.py — Hyperparameter Tuning for SALTEdoraLinearV4
#
# Sweeps rank × learning rate × weight decay to find the best HP combination.
# Uses eigen dispersion throughout (no fixed energy fraction).
# Runs full_ft once as a reference ceiling.
#
# Grid  : 5 ranks × 4 lr × 3 wd = 60 SALTEDORA runs + 1 full_ft = 61 total
# Task  : SST-2 only (fastest GLUE task, ideal for HP search)
#
# Results land in:
#   results_v5/bert-base-uncased/saltedora_v4/r_<r>/lr_<lr>/wd_<wd>/glue_sst2/
#   results_v5/bert-base-uncased/full_ft/reference/glue_sst2/
#   results_v5/summary_v5_hp_tuning.csv  ← sorted best → worst accuracy
# ---------------------------------------------------------------------------

# Create required directories
mkdir -p logs
mkdir -p results_v5/bert-base-uncased/saltedora_v4
mkdir -p results_v5/bert-base-uncased/full_ft/reference

# Load required modules (adjust versions to your cluster)
module load python/3.10
module load cuda/12.1

# Activate virtual environment if you use one
# source venv/bin/activate
# source ~/envs/myenv/bin/activate

# Install dependencies if needed (uncomment on first run)
# pip install -r requirements.txt

# Run the HP tuning script
srun python finetuning_salty_v5.py
