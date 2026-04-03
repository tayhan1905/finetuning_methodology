#!/bin/bash
#SBATCH --job-name=finetune-salty-v4          # Job name
#SBATCH --output=logs/%x_%j.out               # Stdout log  (logs/finetune-salty-v4_JOBID.out)
#SBATCH --error=logs/%x_%j.err                # Stderr log  (logs/finetune-salty-v4_JOBID.err)
#SBATCH --ntasks=1                            # Single task
#SBATCH --cpus-per-task=4                     # CPU cores per task
#SBATCH --gres=gpu:nv:1                       # 1 GPU  (change to a100-40, h100-47, etc.)
#SBATCH -t 3-00:00:00                         # Max walltime: 3 days (lr_rank sweep = 20 runs)
#SBATCH --mem=32G                             # Memory
#SBATCH --partition=gpu                       # GPU partition

# ---------------------------------------------------------------------------
# finetuning_salty_v4.py — Hyperparameter Tuning
# Sweeps lr × rank (and optionally wd × bs) for SALTEDORA-V4 on SST-2.
# Runs full-FT once as a fixed reference baseline.
# Runs principal-angles analysis (final + epoch trajectory) per HP config.
#
# --sweep options:
#   lr_rank  — 4 lr × 5 ranks = 20 runs          (default, ~1-2 days)
#   wd_bs    — 3 wd × 3 bs   =  9 runs           (~12 hours)
#   all      — 180 runs, extend walltime to 7-00:00:00 if using this
#
# Results land in:
#   results/bert-base-uncased/saltedora_v4/lr_<lr>/r_<r>/wd_<wd>/bs_<bs>/glue_sst2/
#   results/bert-base-uncased/full_ft/reference/glue_sst2/
#   results/principal_angles/v4/hp_<tag>/
#   results/summary_v4_hp_tuning_<sweep>.csv
# ---------------------------------------------------------------------------

# Set the sweep type here: lr_rank | wd_bs | all
SWEEP=${1:-lr_rank}

# Create logs dir if it doesn't exist
mkdir -p logs

# Load required modules (adjust versions to your cluster)
module load python/3.10
module load cuda/12.1

# Activate virtual environment if you use one
# source venv/bin/activate
# source ~/envs/myenv/bin/activate

# Install dependencies if needed (uncomment on first run)
# pip install -r requirements.txt

# Run the HP tuning script
srun python finetuning_salty_v4.py --sweep "$SWEEP"
