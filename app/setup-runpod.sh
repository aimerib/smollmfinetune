#!/usr/bin/env bash
#
# Character-AI Training Studio - RunPod bootstrap
# -----------------------------------------------
# This script prepares a brand-new RunPod GPU instance (or resumes one)
# so you can reconnect via tmux and run long jobs safely.
#
# Usage:
#   chmod +x runpod_bootstrap.sh
#   ./runpod_bootstrap.sh        # first time / resume
#
#   ./runpod_bootstrap.sh --no-attach   # skip auto-attach to tmux
#
# • All config knobs can be overridden through the environment.
# • Designed to be re-entrant: you may run it multiple times without harm.

if [[ -n "${BASH_SOURCE[0]}" && -f "${BASH_SOURCE[0]}" ]]; then
    SELF=$(readlink -f "${BASH_SOURCE[0]}")
else
    SELF=$(mktemp --tmpdir runpod_bootstrap.XXXX.sh)
    cat >"$SELF" <&0          # write the current script from STDIN
    chmod +x "$SELF"
    exec "$SELF" "$@"         # restart so $0 is now the real path
fi


set -Eeuo pipefail

### ------------------------------------------------------------------ ###
### 1. Configuration (override from CLI or env)                        ###
### ------------------------------------------------------------------ ###
export DEBIAN_FRONTEND=${DEBIAN_FRONTEND:-noninteractive}

# tmux
TMUX_SESSION=${TMUX_SESSION:-character-ai-studio}

# repo
GIT_REPO_URL=${GIT_REPO_URL:-https://github.com/aimerib/smollmfinetune.git}
GIT_REPO_DIR=${GIT_REPO_DIR:-smollmfinetune}

# python & venv
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
VENV_DIR=${VENV_DIR:-.venv}

# inference defaults
export INFERENCE_ENGINE=${INFERENCE_ENGINE:-vllm}
export VLLM_MODEL=${VLLM_MODEL:-PocketDoc/Dans-PersonalityEngine-V1.3.0-24b}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-2048}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

# streamlit
STREAMLIT_PORT=${STREAMLIT_PORT:-8888}
STREAMLIT_ADDRESS=${STREAMLIT_ADDRESS:-0.0.0.0}

# colours
NC=$(printf '\033[0m')
INFO()    { printf "\033[0;34m[INFO]\033[0m    %s\n" "$*"; }
SUCCESS() { printf "\033[0;32m[SUCCESS]\033[0m %s\n" "$*"; }
WARN()    { printf "\033[1;33m[WARN]\033[0m   %s\n" "$*"; }
ERROR()   { printf "\033[0;31m[ERROR]\033[0m  %s\n" "$*"; }

### ------------------------------------------------------------------ ###
### 2. Helpers                                                         ###
### ------------------------------------------------------------------ ###
die() { ERROR "$*"; exit 1; }

trap 'die "Uncaught error on line $LINENO."' ERR

need_cmd() {
  command -v "$1" &>/dev/null || die "Required command '$1' not found."
}

apt_install() {
  # Install only if the package is missing.
  local pkgs=("$@")
  local missing=()
  for p in "${pkgs[@]}"; do
    dpkg -s "$p" &>/dev/null || missing+=("$p")
  done
  if ((${#missing[@]})); then
    INFO "Installing: ${missing[*]} ..."
    apt-get update -qq
    apt-get install -y "${missing[@]}"
    SUCCESS "Packages installed."
  else
    SUCCESS "All packages already present."
  fi
}

### ------------------------------------------------------------------ ###
### 3. tmux bootstrap                                                  ###
### ------------------------------------------------------------------ ###
ensure_tmux() {
  need_cmd tmux
  [[ -n "${TMUX:-}" ]] && return      # already inside

  local SCRIPT_PATH=$SELF      # already guaranteed to be a real file
  if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    INFO "Creating tmux session '$TMUX_SESSION' ..."
    tmux new-session -d -s "$TMUX_SESSION" -c "$PWD" \
      "bash \"$SCRIPT_PATH\" --inside-tmux \"$@\" \
       |& tee \"$HOME/${TMUX_SESSION}.log\" ; \
       echo; echo '[tmux] bootstrap complete – type exit to close'; bash"
  fi

  # wait (max 5 s) for the session to materialise
  for i in {1..10}; do
    tmux has-session -t "$TMUX_SESSION" 2>/dev/null && break
    sleep 0.5
  done || die "tmux session vanished – check ${TMUX_SESSION}.log"

  INFO "Attaching to tmux session '$TMUX_SESSION' ..."
  exec tmux attach -t "$TMUX_SESSION"
}

### ------------------------------------------------------------------ ###
### 4. Core setup (runs **inside** tmux)                               ###
### ------------------------------------------------------------------ ###
setup_repo() {
  INFO "Fetching project repository ..."
  if [[ -d $GIT_REPO_DIR/.git ]]; then
    git -C "$GIT_REPO_DIR" pull --ff-only
  else
    git clone --depth=1 "$GIT_REPO_URL" "$GIT_REPO_DIR"
  fi
  SUCCESS "Repository ready: $GIT_REPO_DIR"
}

setup_uv() {
  if ! command -v uv &>/dev/null; then
    INFO "Installing uv package manager ..."
    curl -sSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    SUCCESS "uv installed."
  fi
}

setup_venv() {
  cd "$GIT_REPO_DIR"
  if [[ ! -d $VENV_DIR ]]; then
    INFO "Creating Python $PYTHON_VERSION virtualenv ($VENV_DIR) ..."
    uv venv --python "$PYTHON_VERSION"
  fi
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  SUCCESS "Virtualenv activated."

  INFO "Installing Python dependencies ..."
  uv pip install -r app/requirements-runpod.txt
  SUCCESS "Dependencies installed."
}

gpu_info() {
  if command -v nvidia-smi &>/dev/null; then
    INFO "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)"
  else
    WARN "nvidia-smi not present (GPU info skipped)."
  fi
}

start_streamlit() {
  INFO "Launching Streamlit ..."
  cd "$GIT_REPO_DIR/app"
  streamlit run app.py \
    --server.address "$STREAMLIT_ADDRESS" \
    --server.port "$STREAMLIT_PORT" \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.headless true \
    --browser.gatherUsageStats false
}

### ------------------------------------------------------------------ ###
### 5. Main entry-point                                                ###
### ------------------------------------------------------------------ ###
main() {
  local opt_no_attach=0 opt_inside_tmux=0
  while [[ $# -gt 0 ]]; do
    case $1 in
      --inside-tmux) opt_inside_tmux=1 ;;
      --no-attach)   opt_no_attach=1   ;;
      *) die "Unknown option: $1" ;;
    esac
    shift
  done

  # Install system tools (only missing ones).
  apt_install vim tmux curl git build-essential

  # If not yet in tmux, fork into one.
  ((opt_inside_tmux)) || ensure_tmux "$@"

  # ---------------- inside tmux from here ---------------- #
  setup_repo
  setup_uv
  setup_venv
  mkdir -p training_output/{adapters,prompts}
  gpu_info

  cat <<EOF

══════════════════════════════════════════════════════════════
🎉  Environment ready
    • Repo:        $PWD
    • Python:      $(python --version 2>&1)
    • Inference:   $INFERENCE_ENGINE → $VLLM_MODEL
    • tmux:        $TMUX_SESSION
    • Streamlit:   http://$STREAMLIT_ADDRESS:$STREAMLIT_PORT
══════════════════════════════════════════════════════════════
EOF

  ((opt_no_attach)) || start_streamlit
}

main "$@"