#!/bin/bash

# Character AI Training Studio - RunPod Setup Script
# This script automates the complete setup process on RunPod instances

set -e  # Exit on any error

echo "🎭 Character AI Training Studio - RunPod Setup"
echo "=============================================="
echo "🚀 Setting up your AI training environment..."
echo ""

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Update system packages
print_status "Updating system packages..."
apt-get update -qq > /dev/null 2>&1
print_success "System packages updated"

# Install essential tools
print_status "Installing vim and tmux..."
apt-get install -y vim tmux curl git > /dev/null 2>&1
print_success "Essential tools installed"

# Check if we're already in tmux
if [ -z "$TMUX" ]; then
    print_status "Starting tmux session..."
    # Create a new tmux session and run the rest of the script inside it
    tmux new-session -d -s "character-ai-studio" -c "$HOME" "$0 --inside-tmux"
    
    print_success "Created tmux session 'character-ai-studio'"
    echo ""
    echo "🎯 Setup is continuing in tmux session 'character-ai-studio'"
    echo "📋 To attach to the session: tmux attach -t character-ai-studio"
    echo "📋 To list sessions: tmux ls"
    echo "📋 To detach from session: Ctrl+B, then D"
    echo ""
    
    # Wait a moment and then attach to the session
    sleep 2
    exec tmux attach -t "character-ai-studio"
else
    print_success "Already inside tmux session"
fi

# If we get here, we're inside tmux
if [[ "$1" == "--inside-tmux" ]]; then
    print_status "Continuing setup inside tmux..."
    echo ""
fi

# Navigate to workspace
cd /workspace || cd "$HOME"
WORKSPACE=$(pwd)
print_status "Working in: $WORKSPACE"

# Clone the repository
print_status "Cloning smollmfinetune repository..."
if [ -d "smollmfinetune" ]; then
    print_warning "Repository already exists, pulling latest changes..."
    cd smollmfinetune
    git pull origin main
else
    git clone https://github.com/aimerib/smollmfinetune.git
    cd smollmfinetune
fi
print_success "Repository ready"

# Install uv (fast Python package manager)
print_status "Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
    print_success "uv installed successfully"
else
    print_success "uv already installed"
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
uv venv --python 3.11 > /dev/null 2>&1
print_success "Virtual environment created"

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate
print_success "Virtual environment activated"

# Install dependencies
print_status "Installing Python dependencies (this may take a few minutes)..."
cd app
uv pip install -r requirements-runpod.txt > /dev/null 2>&1
print_success "Dependencies installed"

# Set up environment variables for RunPod
print_status "Configuring environment variables..."
export INFERENCE_ENGINE=vllm
export VLLM_MODEL=PocketDoc/Dans-PersonalityEngine-V1.3.0-24b
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=4096
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
print_success "Environment configured"

# Create output directories
print_status "Creating output directories..."
mkdir -p training_output/adapters training_output/prompts
print_success "Output directories ready"

# Get GPU information
print_status "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_success "GPU detected: $GPU_INFO"
else
    print_warning "nvidia-smi not found - GPU detection skipped"
fi

# Final setup summary
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📊 Environment Summary:"
echo "  • Repository: $(pwd)"
echo "  • Python: $(python --version)"
echo "  • Virtual Environment: Activated (.venv)"
echo "  • Inference Engine: vLLM with PersonalityEngine-24B"
echo "  • Tmux Session: character-ai-studio"
echo ""
echo "🚀 Starting Character AI Training Studio..."
echo "  • Access URL: http://0.0.0.0:8888"
echo "  • Local access: http://localhost:8888"
echo ""
echo "📋 Useful Commands:"
echo "  • Detach from tmux: Ctrl+B, then D"
echo "  • Reattach to tmux: tmux attach -t character-ai-studio"
echo "  • View logs: Check the terminal output below"
echo ""

# Start Streamlit
print_status "Launching Streamlit application..."
streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8888 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.headless true \
    --browser.gatherUsageStats false

# If we reach here, Streamlit has stopped
print_warning "Streamlit application stopped"
echo ""
echo "🔄 To restart the application:"
echo "   tmux attach -t character-ai-studio"
echo "   cd /workspace/smollmfinetune/app"
echo "   source .venv/bin/activate"
echo "   streamlit run app.py --server.address 0.0.0.0 --server.port 8888 --server.enableCORS false --server.enableXsrfProtection false" 