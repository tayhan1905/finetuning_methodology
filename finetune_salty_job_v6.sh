#!/bin/bash
#SBATCH --job-name=finetune-salty-v6          # Job name
#SBATCH --output=logs/%x_%j.out               # Stdout log  (logs/finetune-salty-v6_JOBID.out)
#SBATCH --error=logs/%x_%j.err                # Stderr log  (logs/finetune-salty-v6_JOBID.err)
#SBATCH --ntasks=1                            # Single task
#SBATCH --cpus-per-task=4                     # CPU cores per task
#SBATCH --gres=gpu:nv:1                       # 1 GPU  (change to a100-40, h100-47, etc.)
#SBATCH -t 3-00:00:00                         # Max walltime: 3 days (20 runs across 4 datasets)
#SBATCH --mem=32G                             # Memory — 32G to handle MNLI (~393K examples)
#SBATCH --partition=gpu                       # GPU partition

# ---------------------------------------------------------------------------
# finetuning_salty_v6.py — SALTEdoraLinearV4 Dataset × Rank Sweep
#
# Benchmarks SALTEdoraLinearV4 (eigen dispersion, energy_threshold=0.8) across
# four GLUE classification tasks and five rank values = 20 total runs.
#
# Method  : SALTEdoraLinearV4 (r_top_override=None, eigen dispersion)
# Datasets: SST-2 | RTE | QNLI | MNLI
# Ranks   : 8, 16, 32, 64, 128
# ET      : 0.8  (tighter fallback than v3's 0.9)
#
# No weight matrices are saved — optimised for speed.
# Outputs : accuracy, eval_loss, runtime, trainable params per run.
#
# Results land in:
#   results_v6/bert-base-uncased/saltedora_v4/r_<r>/<task>/
#   results_v6/summary_v6_saltedora_sweep.csv
# ---------------------------------------------------------------------------

# Create required directories
mkdir -p logs
mkdir -p results_v6/bert-base-uncased

# Load required modules (adjust versions to your cluster)
module load python/3.10
module load cuda/12.1

# Activate virtual environment if you use one
# source venv/bin/activate
# source ~/envs/myenv/bin/activate

# Install dependencies if needed (uncomment on first run)
# pip install -r requirements.txt

# Run the SALTEdoraLinearV4 sweep script
srun python finetuning_salty_v6.py
