#!/usr/bin/env bash
#
# Start (or re-attach to) the Character-AI Studio tmux + Streamlit server.
#
set -Eeuo pipefail

TMUX_SESSION=${TMUX_SESSION:-character-ai-studio}
VENV_DIR=${VENV_DIR:-../.venv}           # relative to app/
export STREAMLIT_PORT=${STREAMLIT_PORT:-8888}
export STREAMLIT_ADDR=${STREAMLIT_ADDR:-0.0.0.0}

# Default model env vars – override freely
export INFERENCE_ENGINE=${INFERENCE_ENGINE:-vllm}
export VLLM_MODEL=${VLLM_MODEL:-ArliAI/QwQ-32B-ArliAI-RpR-v4}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

export VLLM_GPU_MEMORY_UTILIZATION=0.85  # Leave 15% for KV cache
export VLLM_MAX_NUM_SEQS=256              # Higher concurrent sequences
export VLLM_MAX_BATCHED_TOKENS=8192       # Larger batch token limit
export MAX_MODEL_LEN=4096                 # Context length

# Dataset generation settings
export VLLM_DATASET_BATCH_SIZE=128        # Prevents KV cache preemption

# Advanced optimizations
export VLLM_ATTENTION_BACKEND=FLASHINFER  # Use FlashInfer for better memory efficiency
export VLLM_USE_MODELSCOPE_HUB=False      # Disable ModelScope for better performance
export VLLM_WORKER_USE_RAY=False          # Disable Ray for single-GPU setups

# CUDA optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

run_server() {
  # shellcheck source=/dev/null
  source "../.venv/bin/activate"
  streamlit run app.py \
       --server.address "$STREAMLIT_ADDR" \
       --server.port "$STREAMLIT_PORT" \
       --server.enableCORS false \
       --server.enableXsrfProtection false \
       --server.headless true \
       --browser.gatherUsageStats false \
       --server.fileWatcherType none \
       --logger.level debug
}

if [[ -n ${TMUX:-} ]]; then
  # Already inside tmux – just run.
  run_server
  exit
fi

# Create or attach, depending on whether we have a TTY.
has_tty() { [[ -t 0 && -t 1 ]]; }

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  has_tty && exec tmux attach -t "$TMUX_SESSION" || {
    echo "[INFO] tmux session exists; leaving it running." ; exit 0 ; }
else
  # Fresh session
  tmux new-session -d -s "$TMUX_SESSION" -c "$PWD" "bash -c '$(declare -f run_server); run_server'"
  has_tty && exec tmux attach -t "$TMUX_SESSION" || {
    echo "[INFO] tmux session started (non-interactive shell)." ; exit 0 ; }
fi
