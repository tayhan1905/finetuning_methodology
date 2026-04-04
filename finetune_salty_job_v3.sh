#!/bin/bash
#SBATCH --job-name=finetune-salty-v3          # Job name
#SBATCH --output=logs/%x_%j.out               # Stdout log  (logs/finetune-salty-v3_JOBID.out)
#SBATCH --error=logs/%x_%j.err                # Stderr log  (logs/finetune-salty-v3_JOBID.err)
#SBATCH --ntasks=1                            # Single task
#SBATCH --cpus-per-task=4                     # CPU cores per task
#SBATCH --gres=gpu:nv:1                       # 1 GPU  (change to a100-40, h100-47, etc.)
#SBATCH -t 2-00:00:00                         # Max walltime: 2 days (80 runs across 4 datasets)
#SBATCH --mem=32G                             # Memory — 32G to handle MNLI (~393K examples)
#SBATCH --partition=gpu                       # GPU partition

# ---------------------------------------------------------------------------
# finetuning_salty_v3.py — Performance Comparison Across Datasets
#
# Benchmarks 4 adapter methods across 4 GLUE datasets, 5 ranks = 80 total runs.
#
# Methods : LoRA | DoRA | SALT | SALTEdoraLinearV4 (eigen dispersion)
# Datasets: SST-2 | RTE | QNLI | MNLI
# Ranks   : 8, 16, 32, 64, 128
#
# No weight matrices are saved — optimised for speed.
# Outputs : accuracy, eval_loss, runtime, trainable params per run.
#
# Results land in:
#   results_v3/bert-base-uncased/<mode>/r_<r>/<task>/
#   results_v3/summary_v3_performance_comparison.csv
# ---------------------------------------------------------------------------

# Create required directories
mkdir -p logs
mkdir -p results_v3/bert-base-uncased

# Load required modules (adjust versions to your cluster)
module load python/3.10
module load cuda/12.1

# Activate virtual environment if you use one
# source venv/bin/activate
# source ~/envs/myenv/bin/activate

# Install dependencies if needed (uncomment on first run)
# pip install -r requirements.txt

# Run the performance comparison script
srun python finetuning_salty_v3.py
