#!/bin/bash

# E&M Tutor Project - Micro Speedrun
# Trains a tiny model on CPU/MPS to demonstrate the pipeline.

# 1. Setup Environment
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Install uv if needed
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
[ -d ".venv" ] || uv venv
# Sync dependencies with cpu extra
uv sync --extra cpu
source .venv/bin/activate

# Setup Rust for Tokenizer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# WANDB (Dummy by default)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

echo "--- STARTING MICRO SPEEDRUN ---"

# 2. Reset Report
python -m nanochat.report reset

# 3. Tokenizer (Train on small subset)
echo "--- TRAIN TOKENIZER ---"
python -m nanochat.dataset -n 1 # Download just 1 shard
python -m scripts.tok_train --max_chars=50000000 # 50M chars
python -m scripts.tok_eval

# 4. Base Training (Tiny Model)
echo "--- BASE TRAINING ---"
# depth=4, small batches, few iterations
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=256 \
    --device_batch_size=4 \
    --total_batch_size=128 \
    --eval_every=10 \
    --eval_tokens=1024 \
    --num_iterations=20 \
    --run=$WANDB_RUN

# Base Loss Eval
python -m scripts.base_loss --device_batch_size=4 --split_tokens=1024

# 5. Mid Training
echo "--- MID TRAINING ---"
# Download identity data if needed
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
    curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

python -m scripts.mid_train \
    --max_seq_len=256 \
    --device_batch_size=4 \
    --total_batch_size=128 \
    --eval_every=10 \
    --num_iterations=20 \
    --run=$WANDB_RUN

# 6. SFT (Baseline Chat)
echo "--- SFT BASELINE ---"
python -m scripts.chat_sft \
    --device_batch_size=4 \
    --target_examples_per_step=16 \
    --num_iterations=20 \
    --eval_steps=2 \
    --run=$WANDB_RUN

echo "--- GENERATING REPORT ---"
python -m nanochat.report generate

echo "--- MICRO SPEEDRUN COMPLETE ---"
echo "You can now chat with the baseline model using: python -m scripts.chat_cli"
