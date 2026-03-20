#!/bin/bash
#SBATCH --job-name=train_ppo_spa
#SBATCH --output=logs/train_ppo_%j.out
#SBATCH --error=logs/train_ppo_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ============================================================
# PPO Training Script for Lenovo LiCO H100 Cluster
# Architecture: CNN+LSTM Feature Extractor + PPO
# ============================================================

# Project directory (adjust to your cluster path)
cd /nfs-share-stgnode/home/663380531-2/projek1
mkdir -p logs data/models data/eval

echo "=========================================="
echo " CNN+LSTM PPO Training — H100 GPU"
echo "=========================================="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "Time    : $(date)"
echo ""

# GPU diagnostics
nvidia-smi
echo ""

# Install dependencies if needed
pip install --user pyarrow ccxt scipy matplotlib seaborn 2>/dev/null
echo ""

# ============================================================
# Train the CNN+LSTM PPO Agent
# ============================================================
# Key params:
#   --timesteps 5000000    = ~6h on H100
#   --eval_every_steps 50k = evaluate OOS every 50k steps
#   --device cuda          = use H100 GPU
#   --cooldown_steps 3     = prevent over-trading
#   --inactivity_penalty 1 = penalize doing nothing too long
# ============================================================

python src/train/train_ppo_spa.py \
    --features              "data/features/btc_1h_spa.parquet" \
    --output                "data/models/ppo_spa_btc_1h" \
    --timesteps             2500000 \
    --seed                  42 \
    --eval_every_steps      50000 \
    --train_split           0.8 \
    --device                "cuda" \
    --flat_penalty_bps      0.0 \
    --inactivity_steps      24 \
    --inactivity_penalty_bps 5.0 \
    --turnover_reward_coeff 0.0 \
    --trade_threshold       0.01 \
    --deadband_frac         0.05 \
    --min_hold_steps        0 \
    --cooldown_steps        3

echo ""
echo "=========================================="
echo " Training Complete"
echo " Model saved to: data/models/ppo_spa_btc_1h.zip"
echo "=========================================="
echo "Time: $(date)"
