#!/bin/bash
#SBATCH --job-name=gpu_test        # Job name
#SBATCH --output=gpu_test_%j.out   # Output file (will store stdout/err)
#SBATCH --error=gpu_test_%j.err    # Error file (stderr)
#SBATCH --time=00:05:00            # Max runtime (adjust as needed)
#SBATCH --gres=gpu:nv:1            # Request one GPU (or change type)
#SBATCH --mem=8G                   # Memory request (adjust as needed)

# Load necessary modules (if any)
module load python/3.8            # Example, modify based on your cluster's environment

# Print details about the job and GPU availability
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$(python -c 'import torch; print(torch.cuda.is_available())')" == "True" ]; then
    python -c "import torch; print(f'Using GPU: {torch.cuda.current_device()}'); print(f'GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}')"
fi
