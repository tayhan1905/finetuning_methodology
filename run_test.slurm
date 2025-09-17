#!/bin/bash
#SBATCH --job-name=adapter_test        # Job name
#SBATCH --output=logs/%x_%j.out        # Stdout/stderr written here (logs/adapter_test_JOBID.out)
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00                # Walltime (hh:mm:ss)
#SBATCH --mem=16G                      # Memory
#SBATCH --cpus-per-task=4              # Number of CPU cores
#SBATCH --gres=gpu:a100-40:1           # Request 1 GPU (change to nv, h100-47, etc.)

# Optional: if you want to restrict GPU family (example: CUDA 7.5)
# #SBATCH -C cuda75

# Load modules (depends on your cluster setup)
module load python/3.10
module load cuda/12.1

# Activate your virtual environment if you use one
# source ~/envs/myenv/bin/activate

# Run your Python script
srun python adapter_experiments.py
