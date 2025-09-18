#!/bin/bash
#SBATCH --job-name=finetune-salty      # Job name
#SBATCH --output=finetune-salty.out    # Standard output and error log
#SBATCH --error=finetune-salty.err     # Standard error log
#SBATCH --ntasks=1                     # Number of tasks (e.g., 1 task for a single node)
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --gres=gpu:nv:1                # Request 1 GPU of the 'nv' type (modify if necessary)
#SBATCH --time=24:00:00                # Maximum runtime (HH:MM:SS)
#SBATCH --mem=16G                      # Memory per node (16 GB)
#SBATCH --partition=gpu                # Specify the partition to run the job (if applicable)

# Load any required modules (e.g., CUDA, PyTorch, etc.)
module load python/3.8.5
module load cuda/11.1

# Create a virtual environment (if needed)
# python -m venv venv
# source venv/bin/activate

# Install required dependencies if not already done (uncomment the next line if necessary)
# pip install -r requirements.txt

# Activate your Python environment (if you use one)
# source venv/bin/activate

# Run the Python script
python finetuning_salty.py
