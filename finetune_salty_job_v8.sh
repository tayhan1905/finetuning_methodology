#!/bin/bash
#SBATCH --job-name=finetune-salty-v8          # Job name
#SBATCH --output=logs/%x_%j.out               # Stdout log
#SBATCH --error=logs/%x_%j.err                # Stderr log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nv:1
#SBATCH -t 3-00:00:00                         # Max walltime: 3 days
#SBATCH --mem=16G
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# finetuning_salty_v8.py — SALTEDORA-V4 (Knee, r=128) vs Full Fine-Tuning
#
# Trains two models on SST-2 and computes principal angles between their
# weight subspaces at each epoch and at the final checkpoint.
#
# Method  : SALTEdoraLinearV4 (cumulative energy knee, min_frac=0.50, max_frac=0.90)
# Rank    : 128
# Dataset : glue/sst2
# Epochs  : 5
#
# Results : results_v8/
#   bert-base-uncased/full_ft/reference/glue_sst2/
#   bert-base-uncased/saltedora_v4_knee/r_128/glue_sst2/
#   principal_angles/  (final + per-epoch PA summaries + .npz per layer)
#   summary_v8.csv
# ---------------------------------------------------------------------------

mkdir -p logs
mkdir -p results_v8

module load python/3.10
module load cuda/12.1

# source venv/bin/activate

srun python finetuning_salty_v8.py
