#!/bin/bash
#SBATCH --job-name=finetune-salty-v9          # Job name
#SBATCH --output=logs/%x_%j.out               # Stdout log
#SBATCH --error=logs/%x_%j.err                # Stderr log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nv:1
#SBATCH -t 3-00:00:00                         # Max walltime: 3 days
#SBATCH --mem=16G
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# finetuning_salty_v9.py — SALTEDORA-V4 (Fixed Energy Cuts, r=64) vs Full FT
#
# Trains full_ft once as baseline, then sweeps 6 energy cuts for SALTEDORA.
# Computes principal angles vs full_ft at each epoch and final checkpoint.
#
# Method  : SALTEdoraLinearV4 (r_top_override = fixed head fraction)
# Rank    : 64
# Cuts    : [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Dataset : glue/sst2
# Epochs  : 5
#
# Results : results_v9/
#   bert-base-uncased/full_ft/reference/glue_sst2/
#   bert-base-uncased/saltedora_v4_cut_<X>/r_64/glue_sst2/
#   principal_angles/cut_<X>/  (final + per-epoch PA + .npz per layer)
#   summary_v9_energy_cuts.csv
# ---------------------------------------------------------------------------

mkdir -p logs
mkdir -p results_v9

module load python/3.10
module load cuda/12.1

# source venv/bin/activate

srun python finetuning_salty_v9.py
