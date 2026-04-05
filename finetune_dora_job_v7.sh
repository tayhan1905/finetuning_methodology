#!/bin/bash
#SBATCH --job-name=dora_v7
#SBATCH --output=logs/dora_v7_%j.out
#SBATCH --error=logs/dora_v7_%j.err
#SBATCH -t 3-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

mkdir -p logs

echo "======================================"
echo "DoRA v7 — r=64 vs Full FT — SST-2"
echo "Started: $(date)"
echo "======================================"

python finetuning_dora_v7.py

echo "======================================"
echo "Finished: $(date)"
echo "======================================"
