---
sidebar_position: 2
---

# Setup Guide

This guide will help you get the Character Creation Platform up and running from scratch. We'll cover system requirements, installation, configuration, and troubleshooting.

<div style={{background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', padding: '2rem', borderRadius: '12px', color: 'white', marginBottom: '2rem'}}>
  <h2 style={{marginTop: 0}}>Quick Install</h2>
  <p>If you just want to get started quickly, run this single command:</p>
  <pre style={{background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '8px'}}>
    <code>git clone &lt;repository-url&gt; && cd smollmfinetune && ./launch-client.sh</code>
  </pre>
</div>

## System Requirements

### Minimum Requirements

:::warning[Minimum Requirements]
<div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginBottom: '2rem'}}>
  <div style={{backgroundColor: '#FFFFFF22', padding: '0.7rem', borderRadius: '8px'}}>
    <h4>Hardware</h4>
    <ul>
      <li><strong>CPU</strong>: 4+ cores (8+ recommended)</li>
      <li><strong>RAM</strong>: 8GB minimum (16GB recommended)</li>
      <li><strong>Storage</strong>: 20GB free space</li>
      <li><strong>GPU</strong>: Optional for inference (NVIDIA/Apple Silicon)</li>
    </ul>
  </div>
  
  <div style={{backgroundColor: '#FFFFFF22', padding: '0.7rem', borderRadius: '8px'}}>
    <h4>Software</h4>
    <ul>
      <li><strong>OS</strong>: Windows 10+, macOS 11+, Linux</li>
      <li><strong>Python</strong>: 3.11 or higher</li>
      <li><strong>Node.js</strong>: 16+ (for React client)</li>
      <li><strong>Git</strong>: For cloning repository</li>
    </ul>
  </div>
</div>
:::

### Recommended Specifications

For optimal performance, especially when training models:

- **GPU**: NVIDIA RTX 3070+ with 8GB+ VRAM or Apple M1/M2 with 16GB+ unified memory
- **RAM**: 32GB for comfortable multi-model training
- **Storage**: 50GB+ for multiple models and datasets
- **Network**: Stable internet for downloading models (initial setup ~5GB)

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd smollmfinetune
```

### Step 2: Set Up Python Environment

We recommend using a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Python Dependencies

```bash
# Install main application dependencies
cd app
pip install -r requirements.txt

# For GPU acceleration (NVIDIA):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Apple Silicon (M1/M2):
pip install torch torchvision torchaudio

# Return to project root
cd ..
```

### Step 4: Install Node.js Dependencies

```bash
# Install React client dependencies
cd client
npm install

# Return to project root
cd ..
```

### Step 5: Download Base Models

The platform will automatically download required models on first use, but you can pre-download them:

```bash
# Create models directory
mkdir -p models

# Download base model (optional - will auto-download on first use)
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('unsloth/Llama-3.2-1B'); \
AutoModelForCausalLM.from_pretrained('unsloth/Llama-3.2-1B')"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Configuration
INFERENCE_HOST=localhost
INFERENCE_PORT=8000
INFERENCE_WORKERS=4

# Client Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000

# Model Configuration
DEFAULT_MODEL=unsloth/Llama-3.2-1B
MODEL_CACHE_DIR=./models

# GPU Configuration (optional)
CUDA_VISIBLE_DEVICES=0  # For multi-GPU systems
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Database Configuration
DATABASE_URL=sqlite:///./app/users.db
```

### GPU Configuration

#### NVIDIA GPUs

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Configure memory allocation for large models
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### Apple Silicon

```bash
# Check MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# No additional configuration needed - MPS is auto-detected
```

## Launching the Platform

### Option 1: All-in-One Launch (Recommended)

Use our convenient launch script:

```bash
./launch-client.sh
```

This will:
1. Start the Character Devkit (Streamlit app)
2. Launch the Inference Server
3. Start the React Client
4. Open your browser automatically

### Option 2: Launch Components Separately

For development or debugging, you can launch each component individually:

#### Terminal 1: Character Devkit
```bash
cd app
./startup.sh
# Or directly: streamlit run app.py
```
Access at: http://localhost:8501

#### Terminal 2: Inference Server
```bash
python scripts/run_inference_server.py --port 8000 --workers 4
```
API available at: http://localhost:8000

#### Terminal 3: React Client
```bash
cd client
npm start
```
Access at: http://localhost:3000

## Verifying Installation

### Health Checks

1. **Devkit Health**: Navigate to http://localhost:8501
   - You should see the main Character Creation interface
   - Check that all sidebar options are visible

2. **Inference API Health**: Navigate to http://localhost:8000/health
   - Should return: `{"status": "healthy", "gpu_available": true/false}`

3. **React Client**: Navigate to http://localhost:3000
   - Should show the character selection screen
   - Test a character conversation

### Quick Test Flow

1. **Create a Test Character**:
   - Open Devkit (http://localhost:8501)
   - Use Conversational Builder
   - Create a simple character

2. **Generate Training Data**:
   - Navigate to Dataset Studio
   - Generate 50 samples (quick test)

3. **Quick Training**:
   - Go to Training Config
   - Set epochs to 1
   - Start training (should take ~5 minutes)

4. **Test in Client**:
   - Open React Client (http://localhost:3000)
   - Select your character
   - Have a conversation!

## Troubleshooting

### Common Issues

<details>
<summary><strong>ModuleNotFoundError when starting</strong></summary>

Make sure you've activated the virtual environment and installed all dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
cd app && pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Port already in use</strong></summary>

Kill existing processes:
```bash
# Find processes
lsof -i :8501  # Streamlit
lsof -i :8000  # Inference server
lsof -i :3000  # React client

# Kill process (replace PID)
kill -9 <PID>
```
</details>

<details>
<summary><strong>CUDA/GPU not detected</strong></summary>

1. Verify GPU drivers are installed
2. Check PyTorch CUDA version matches your CUDA installation:
```bash
nvidia-smi  # Check CUDA version
python -c "import torch; print(torch.version.cuda)"  # Check PyTorch CUDA
```
</details>

<details>
<summary><strong>React client can't connect to API</strong></summary>

1. Verify inference server is running: `curl http://localhost:8000/health`
2. Check environment variables in `.env`
3. Ensure no firewall is blocking local connections
</details>

### Performance Optimization

#### For Training
- **Reduce batch size** if running out of memory
- **Use gradient checkpointing** for large models
- **Enable mixed precision training** for faster training

#### For Inference
- **Enable attention caching** (default)
- **Adjust worker count** based on CPU cores
- **Use GPU if available** for 5-10x speedup

## Docker Installation (Alternative)

For containerized deployment:

```bash
# Build containers
docker-compose -f docker-compose.prod.yml build

# Launch everything
docker-compose -f docker-compose.prod.yml up

# Access services
# - Devkit: http://localhost:8501
# - API: http://localhost:8000
# - Client: http://localhost:3000
```

## Server Deployment

### Local Server Setup

For production-like local deployment with optimized performance:

#### Option 1: Production Docker Stack

```bash
# Clone the repository
git clone <repository-url>
cd smollmfinetune

# Create production environment file
cat > .env.prod << EOF
# Production Configuration
ENVIRONMENT=production
REDIS_URL=redis://redis:6379/0
DATABASE_URL=sqlite:///data/platform.db

# Inference Configuration
INFERENCE_ENGINE=vllm
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=4096
CUDA_VISIBLE_DEVICES=0

# Security
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EOF

# Launch production stack
docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d

# Verify deployment
curl http://localhost:8888/health
```

#### Option 2: Manual Production Setup

```bash
# 1. Create production virtual environment
python3 -m venv venv-prod
source venv-prod/bin/activate

# 2. Install production dependencies
cd app
pip install -r requirements-prod.txt

# 3. Configure for production
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export PYTHONUNBUFFERED=1

# 4. Start with production settings
streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8888 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.headless true \
  --browser.gatherUsageStats false
```

### RunPod Deployment

RunPod provides powerful GPU instances perfect for AI character training and inference.

#### Quick Setup (Recommended)

The fastest way to get started on RunPod:

```bash
# 1. Create a new RunPod instance
# - Choose: RTX A5000, RTX 4090, or A100
# - Template: PyTorch 2.1 or Ubuntu 22.04
# - Container Disk: 100GB minimum
# - Volume: 500GB for models and data

# 2. Once connected via SSH, run our setup script
curl -sSL https://raw.githubusercontent.com/aimerib/smollmfinetune/main/app/setup-runpod.sh | bash

# 3. The script will:
# - Install all dependencies
# - Clone the repository
# - Create a Python virtual environment
# - Set up tmux session
# - Configure optimized settings

# 4. Start the application
cd /workspace/smollmfinetune/app
./startup.sh
```

#### Manual RunPod Setup

For more control over the setup process:

```bash
# 1. Update system and install dependencies
apt-get update && apt-get install -y \
  git curl build-essential tmux vim \
  python3.11 python3.11-venv python3.11-dev

# 2. Create workspace
mkdir -p /workspace
cd /workspace

# 3. Clone repository
git clone https://github.com/aimerib/smollmfinetune.git
cd smollmfinetune

# 4. Create optimized virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 5. Install RunPod-optimized dependencies
pip install -r app/requirements-runpod.txt

# 6. Configure for RunPod environment
cat > .env.runpod << EOF
# RunPod Configuration
INFERENCE_ENGINE=vllm
VLLM_MODEL=PocketDoc/Dans-PersonalityEngine-V1.3.0-24b
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_MAX_MODEL_LEN=4096
CUDA_VISIBLE_DEVICES=0

# Performance Settings
PYTHONUNBUFFERED=1
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_PORT=8888
STREAMLIT_SERVER_HEADLESS=true

# Security
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
EOF

# 7. Create startup script
cat > start-runpod.sh << 'EOF'
#!/bin/bash
set -e

# Activate environment
source .venv/bin/activate

# Load environment
source .env.runpod

# Create necessary directories
mkdir -p training_output/{adapters,prompts}

# Start in tmux session
tmux new-session -d -s character-ai -c "$(pwd)/app" \
  "streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8888 \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.headless true"

echo "✅ Character AI Studio started!"
echo "📱 Access via RunPod's public URL on port 8888"
echo "🔗 Or use: tmux attach -t character-ai"
EOF

chmod +x start-runpod.sh

# 8. Start the application
./start-runpod.sh
```

#### RunPod Docker Deployment

For containerized RunPod deployment:

```bash
# 1. Create a custom Docker image
cat > Dockerfile.runpod << 'EOF'
FROM runpod/pytorch:2.1.0-py3.11-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl tmux vim \
    && rm -rf /var/lib/apt/lists/*

# Clone repository
RUN git clone https://github.com/aimerib/smollmfinetune.git
WORKDIR /workspace/smollmfinetune

# Install Python dependencies
RUN pip install -r app/requirements-runpod.txt

# Create startup script
RUN echo '#!/bin/bash\n\
cd /workspace/smollmfinetune/app\n\
exec streamlit run app.py \\\n\
  --server.address 0.0.0.0 \\\n\
  --server.port 8888 \\\n\
  --server.enableCORS false \\\n\
  --server.enableXsrfProtection false \\\n\
  --server.headless true' > /start.sh && chmod +x /start.sh

# Expose port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8888/_stcore/health || exit 1

# Start application
CMD ["/start.sh"]
EOF

# 2. Build and push to registry
docker build -f Dockerfile.runpod -t your-registry/character-ai-studio:runpod .
docker push your-registry/character-ai-studio:runpod

# 3. Deploy on RunPod using your custom image
```

### Advanced Server Configuration

#### Performance Optimization

```bash
# GPU Memory Optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# vLLM Configuration for different GPU types
# RTX A5000 (24GB)
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=4096

# RTX 4090 (24GB)
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLLM_MAX_MODEL_LEN=8192

# A100 (80GB)
export VLLM_GPU_MEMORY_UTILIZATION=0.95
export VLLM_MAX_MODEL_LEN=16384
```

#### Load Balancing Setup

For high-traffic deployments:

```nginx
# nginx.conf
upstream character_ai_backend {
    least_conn;
    server app1:8888 max_fails=3 fail_timeout=30s;
    server app2:8888 max_fails=3 fail_timeout=30s;
    server app3:8888 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://character_ai_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
```

#### Monitoring and Logging

```bash
# Set up structured logging
export STREAMLIT_LOGGER_LEVEL=INFO
export PYTHONUNBUFFERED=1

# Enable health monitoring
curl -s http://localhost:8888/_stcore/health | jq

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

### Security Configuration

#### Production Security Settings

```bash
# Create secure environment configuration
cat > .env.secure << EOF
# Disable debug features
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Security headers
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# Database security
DATABASE_URL=postgresql://user:password@localhost:5432/character_ai

# API rate limiting
API_RATE_LIMIT=100  # requests per minute
EOF
```

#### Firewall Configuration

```bash
# Ubuntu/Debian firewall setup
ufw enable
ufw allow ssh
ufw allow 8888/tcp  # Streamlit
ufw allow 8000/tcp  # API (if separate)
ufw deny 6379/tcp   # Redis (internal only)
```

### Backup and Recovery

#### Automated Backup Script

```bash
#!/bin/bash
# backup-character-ai.sh

BACKUP_DIR="/workspace/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup database
sqlite3 /workspace/smollmfinetune/app/users.db ".backup $BACKUP_DIR/database_$DATE.db"

# Backup training outputs
tar -czf "$BACKUP_DIR/training_outputs_$DATE.tar.gz" /workspace/smollmfinetune/training_output/

# Backup character configs
tar -czf "$BACKUP_DIR/characters_$DATE.tar.gz" /workspace/smollmfinetune/characters/

# Clean old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.db" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete

echo "✅ Backup completed: $DATE"
```

#### Recovery Process

```bash
# Restore from backup
BACKUP_DATE="20241201_143000"  # Replace with your backup date

# Stop application
tmux kill-session -t character-ai

# Restore database
cp "/workspace/backups/database_$BACKUP_DATE.db" /workspace/smollmfinetune/app/users.db

# Restore training outputs
tar -xzf "/workspace/backups/training_outputs_$BACKUP_DATE.tar.gz" -C /

# Restart application
cd /workspace/smollmfinetune
./start-runpod.sh
```

## Cloud Deployment

### Deploy to Hugging Face Spaces

1. Fork the repository
2. Create new Space on Hugging Face
3. Connect to your fork
4. Configure with provided `app/Dockerfile`

### Deploy to Runpod

1. Use provided `app/Dockerfile.prod`
2. Set environment variables in Runpod dashboard
3. Configure persistent storage for models
4. Use `app/requirements-runpod.txt` for optimized dependencies

## Quick Reference

### Essential Commands

```bash
# Local Development
./launch-client.sh                    # Start everything locally
cd app && ./startup.sh              # Start just the Streamlit app
streamlit run app.py --server.port 8888  # Manual Streamlit start

# RunPod Deployment
curl -sSL https://raw.githubusercontent.com/aimerib/smollmfinetune/main/app/setup-runpod.sh | bash
cd /workspace/smollmfinetune/app && ./startup.sh

# Docker Production
docker-compose -f docker-compose.prod.yml up -d
docker-compose -f docker-compose.prod.yml logs -f

# Health Checks
curl http://localhost:8888/_stcore/health  # Streamlit health
curl http://localhost:8000/health          # Inference API health
nvidia-smi                                 # GPU status
```

### Port Reference

| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| Streamlit Devkit | 8888 | http://localhost:8888 | Main character creation interface |
| Inference API | 8000 | http://localhost:8000 | Model inference endpoint |
| React Client | 3000 | http://localhost:3000 | Modern chat interface |
| Prometheus | 9090 | http://localhost:9090 | Metrics collection |
| Grafana | 3000 | http://localhost:3000 | Monitoring dashboards |

### Environment Variables

```bash
# Core Configuration
export INFERENCE_ENGINE=vllm
export VLLM_MODEL=PocketDoc/Dans-PersonalityEngine-V1.3.0-24b
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export STREAMLIT_PORT=8888

# Performance Tuning
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export VLLM_MAX_MODEL_LEN=4096
export CUDA_VISIBLE_DEVICES=0

# Security (Production)
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
```

## Next Steps

Now that your platform is running:

1. **[Getting Started Guide](./getting-started)** - Create your first character
2. **[Training Guide](./training-guide)** - Deep dive into model training
3. **[Client Guide](./client-guide)** - Master the React interface
4. **[Core Concepts](./core-concepts)** - Understand the platform architecture
5. **[Production Deployment](./deploy)** - Scale to production with monitoring and security

---
:::tip[Installation Complete!]
<div>
  <p>Your Character Creation Platform is ready. Time to bring your characters to life!</p>
  <a href="./getting-started" style={{background: '#4caf50', color: 'white', padding: '0.2rem 1rem', borderRadius: '4px', textDecoration: 'none', display: 'inline-block', justifyContent: 'center', alignItems: 'center', textAlign: 'center'}}>
    <span style={{fontSize: '1.2rem'}}>Start Creating →</span>
  </a>
</div> 
:::