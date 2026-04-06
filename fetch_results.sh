#!/bin/bash
# fetch_results.sh
# ================
# Pulls only CSV files from all results folders on the remote cluster.
# Preserves the full directory structure locally.
#
# NOTE on Windows paths (WSL):
#   D:\TH_FYP  →  /mnt/d/TH_FYP
#   C:\Users\foo  →  /mnt/c/Users/foo
#
# Usage:
#   bash fetch_results.sh

# ── Config ────────────────────────────────────────────────────────
REMOTE_USER="tayhan"
REMOTE_HOST="xlogin.comp.nus.edu.sg"
REMOTE_DIR="/home/t/tayhan/Finetuning/finetuning_methodology/results_v9"
LOCAL_DIR="/mnt/d/TH_FYP/results_v9"
# ──────────────────────────────────────────────────────────────────

mkdir -p "$LOCAL_DIR"

echo "======================================"
echo "Fetching CSV files from $REMOTE_HOST"
echo "Remote : $REMOTE_DIR"
echo "Local  : $LOCAL_DIR"
echo "======================================"

rsync -avzr --include="*/" --include="*.csv" --exclude="*" --prune-empty-dirs "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/"

echo ""
echo "======================================"
echo "Done. Files saved to: $LOCAL_DIR"
echo "======================================"
