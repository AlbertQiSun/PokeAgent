#!/bin/bash
#SBATCH --job-name=poke_sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=sfscai
#SBATCH --gres=gpu:a800:1
#SBATCH --exclude=gpu191
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

# ── No internet on compute nodes ──
# Model is loaded from local filesystem, no HF downloads needed.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Activate environment ──
source ~/.bashrc
conda activate pokemon

cd /gpfsnyu/scratch/qs2196/PokeAgent/bot

# ── Local model path ──
MODEL_PATH="/gpfsnyu/scratch/qs2196/.cache/models/Qwen3-8B"

echo "============================================"
echo "  PokeAgent SFT Training"
echo "  Model:     $MODEL_PATH"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Started:   $(date)"
echo "============================================"

python train_sft.py train \
    --data sft_data.jsonl \
    --model "$MODEL_PATH" \
    --epochs 3 \
    --batch-size 16 \
    --lr 1e-4 \
    --lora-rank 16

echo "Finished: $(date)"
