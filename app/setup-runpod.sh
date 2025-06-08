#!/usr/bin/env bash
#
# Bootstrap a fresh RunPod container.
# Safe to re-run; does NOT touch tmux or run the app.
#
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive

##### Config – override with env vars if you like #############################
REPO_URL=${REPO_URL:-https://github.com/aimerib/smollmfinetune.git}
WORKDIR=${WORKDIR:-/workspace}
REPO_DIR=${REPO_DIR:-smollmfinetune}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
VENV_DIR=${VENV_DIR:-.venv}
##### helpers #################################################################
info()    { printf "\033[0;34m[INFO]\033[0m    %s\n" "$*"; }
success() { printf "\033[0;32m[SUCCESS]\033[0m %s\n" "$*"; }
need_cmd() { command -v "$1" &>/dev/null || { echo >&2 "Missing $1"; exit 1; }; }

apt_install() {
  local pkgs=("$@") missing=()
  for p in "${pkgs[@]}"; do dpkg -s "$p" &>/dev/null || missing+=("$p"); done
  ((${#missing[@]})) || { success "Apt packages already present"; return; }
  info "Installing apt packages: ${missing[*]}"
  apt-get update -qq
  apt-get install -y "${missing[@]}"
  success "Apt install done"
}
###############################################################################

apt_install vim tmux curl git build-essential

mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [[ -d $REPO_DIR/.git ]]; then
  info "Updating repo…"
  git -C "$REPO_DIR" pull --ff-only
else
  info "Cloning repo…"
  git clone --depth=1 "$REPO_URL" "$REPO_DIR"
fi
success "Repo ready → $WORKDIR/$REPO_DIR"

cd "$REPO_DIR"

if ! command -v uv &>/dev/null; then
  info "Installing uv…"
  pip install uv
fi

if ! command -v uv &>/dev/null; then
  error "uv not found"
  exit 1
fi

if [[ ! -d $VENV_DIR ]]; then
  info "Creating Python $PYTHON_VERSION virtualenv…"
  uv venv --python "$PYTHON_VERSION"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
success "Virtualenv activated"

info "Installing Python deps…"
uv pip install -r app/requirements-runpod.txt
success "Python deps installed"

mkdir -p training_output/{adapters,prompts}

info "GPU check:"
command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1 || echo "  (nvidia-smi not found)"

cd app
cat <<EOF

═════════════════════════════════════════════════════════
Bootstrap finished ✔
You are now in: $(pwd)
Next steps (interactive shell):
    ./startup.sh          # create/attach tmux & launch Streamlit
═════════════════════════════════════════════════════════
EOF

# Keep the shell open for convenience when running interactively.
exec bash