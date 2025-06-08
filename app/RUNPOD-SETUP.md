# 🚀 RunPod Quick Setup Guide

## One-Command Setup

```bash
curl -sSL https://raw.githubusercontent.com/aimerib/smollmfinetune/main/app/setup-runpod.sh | bash
```

## Manual Setup

1. **Upload the script** to your RunPod instance
2. **Make it executable** and run:
   ```bash
   chmod +x setup-runpod.sh
   ./setup-runpod.sh
   ```

## What the Script Does

✅ **System Setup**
- Updates apt packages
- Installs vim, tmux, curl, git

✅ **Environment Setup**  
- Creates tmux session `character-ai-studio`
- Clones repository from [aimerib/smollmfinetune](https://github.com/aimerib/smollmfinetune.git)
- Installs uv (fast Python package manager)
- Creates Python 3.11 virtual environment

✅ **Dependencies**
- Installs optimized RunPod requirements
- Configures vLLM with PersonalityEngine-24B
- Sets up environment variables

✅ **Application Launch**
- Starts Streamlit on port **8888**
- Configures for RunPod networking
- Provides access at `http://0.0.0.0:8888`

## 🎯 Tmux Commands

The setup runs everything in a tmux session for persistence:

```bash
# Attach to the session
tmux attach -t character-ai-studio

# Detach from session (app keeps running)
Ctrl+B, then D

# List all sessions
tmux ls

# Kill the session
tmux kill-session -t character-ai-studio
```

## 🔧 Manual Restart

If you need to restart the application:

```bash
tmux attach -t character-ai-studio
cd /workspace/smollmfinetune/app
source .venv/bin/activate
streamlit run app.py --server.address 0.0.0.0 --server.port 8888 --server.enableCORS false --server.enableXsrfProtection false
```

## 🎛️ Environment Variables

The script automatically configures:

```bash
INFERENCE_ENGINE=vllm
VLLM_MODEL=PocketDoc/Dans-PersonalityEngine-V1.3.0-24b
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=4096
CUDA_VISIBLE_DEVICES=0
```

## 🚨 Troubleshooting

**Port Issues**: RunPod exposes port 8888 by default. If using different port, update RunPod port mapping.

**GPU Memory**: For 24B model, ensure your RunPod instance has at least 48GB VRAM.

**Dependencies**: If vLLM fails to install, the app will fall back to transformers automatically.

**tmux Session Lost**: Re-run the setup script to recreate the session.

## 🎭 Ready to Train!

Once setup completes, access your Character AI Training Studio at:
- **Public URL**: Your RunPod instance URL on port 8888
- **Local**: `http://localhost:8888` (if port forwarding)

Happy training! ✨ 