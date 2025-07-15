#!/bin/bash
# Bootstrap script for spot instances
# Installs dependencies, clones repo, restores checkpoint, and starts training

set -e  # Exit on any error

# Log everything to both stdout and file
exec &> >(tee -a /var/log/spot-training.log)

echo "=== Spot Instance Bootstrap Started at $(date) ==="

# Function to write heartbeat
write_heartbeat() {
    if [ -n "$S3_HEARTBEAT_BUCKET" ] && [ -n "$RUN_ID" ]; then
        TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        INSTANCE_ID=$(ec2-metadata --instance-id 2>/dev/null | cut -d " " -f 2 || echo "unknown")
        
        cat > /tmp/heartbeat.json <<EOF
{
    "timestamp": "$TIMESTAMP",
    "instance_id": "$INSTANCE_ID",
    "run_id": "$RUN_ID",
    "status": "alive"
}
EOF
        
        aws s3 cp /tmp/heartbeat.json "s3://$S3_HEARTBEAT_BUCKET/heartbeats/$RUN_ID" || true
    fi
}

# Parse command line arguments
REPO_URL=${REPO_URL:-"https://github.com/aimerib/smollmfinetune.git"}
GIT_SHA=""
RESUME_FROM=""
S3_CHECKPOINT_BUCKET=${S3_CHECKPOINT_BUCKET:-"training-checkpoints"}
S3_HEARTBEAT_BUCKET=${S3_HEARTBEAT_BUCKET:-$S3_CHECKPOINT_BUCKET}
RUN_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --repo-url)
            REPO_URL="$2"
            shift 2
            ;;
        --git-sha)
            GIT_SHA="$2"
            shift 2
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --s3-checkpoint-bucket)
            S3_CHECKPOINT_BUCKET="$2"
            shift 2
            ;;
        --s3-heartbeat-bucket)
            S3_HEARTBEAT_BUCKET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Generate run ID if not provided
if [ -z "$RUN_ID" ]; then
    RUN_ID="run_$(date +%Y%m%d_%H%M%S)_$(head -c 4 /dev/urandom | od -An -tx4 | tr -d ' ')"
fi

echo "Configuration:"
echo "  REPO_URL: $REPO_URL"
echo "  GIT_SHA: ${GIT_SHA:-latest}"
echo "  RESUME_FROM: ${RESUME_FROM:-none}"
echo "  RUN_ID: $RUN_ID"
echo "  S3_CHECKPOINT_BUCKET: $S3_CHECKPOINT_BUCKET"

# Update system packages
echo "=== Installing System Dependencies ==="
apt-get update -qq
apt-get install -y -qq \
    git \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    curl \
    awscli \
    ec2-metadata \
    tmux \
    htop \
    nvtop || true  # nvtop might not be available on all systems

# Install CUDA if not present (for GPU instances)
if ! command -v nvidia-smi &> /dev/null; then
    echo "=== Installing CUDA ==="
    # This would be more complex in production, simplified for now
    echo "GPU not detected or CUDA not installed. Proceeding without GPU support."
else
    echo "=== GPU Detected ==="
    nvidia-smi
fi

# Setup working directory
WORK_DIR="/home/ubuntu/training"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository
echo "=== Cloning Repository ==="
if [ -d "repo" ]; then
    cd repo
    git fetch origin
    if [ -n "$GIT_SHA" ]; then
        git checkout "$GIT_SHA"
    else
        git pull origin main
    fi
    cd ..
else
    git clone "$REPO_URL" repo
    cd repo
    if [ -n "$GIT_SHA" ]; then
        git checkout "$GIT_SHA"
    fi
    cd ..
fi

# Setup Python environment
echo "=== Setting Up Python Environment ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip wheel setuptools

# Install requirements
echo "=== Installing Python Dependencies ==="
cd repo
if [ -f "app/requirements.txt" ]; then
    pip install -r app/requirements.txt
fi

# Install telemetry SDK for heartbeat
pip install boto3 click httpx

# Setup environment variables
export PYTHONUNBUFFERED=1
export RUN_ID=$RUN_ID
export INSTANCE_ID=$(ec2-metadata --instance-id 2>/dev/null | cut -d " " -f 2 || echo "unknown")
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}

# Create heartbeat script
cat > /home/ubuntu/heartbeat.py <<'EOF'
#!/usr/bin/env python3
import os
import sys
import time
import json
import boto3
from datetime import datetime

def write_heartbeat():
    bucket = os.getenv('S3_HEARTBEAT_BUCKET')
    run_id = os.getenv('RUN_ID')
    instance_id = os.getenv('INSTANCE_ID', 'unknown')
    
    if not bucket or not run_id:
        return
    
    s3 = boto3.client('s3')
    heartbeat = {
        'timestamp': datetime.utcnow().isoformat(),
        'instance_id': instance_id,
        'run_id': run_id,
        'status': 'alive'
    }
    
    try:
        s3.put_object(
            Bucket=bucket,
            Key=f'heartbeats/{run_id}',
            Body=json.dumps(heartbeat),
            ContentType='application/json'
        )
        print(f"Heartbeat written at {heartbeat['timestamp']}")
    except Exception as e:
        print(f"Failed to write heartbeat: {e}")

if __name__ == '__main__':
    while True:
        write_heartbeat()
        time.sleep(300)  # 5 minutes
EOF

chmod +x /home/ubuntu/heartbeat.py

# Start heartbeat monitor in background
echo "=== Starting Heartbeat Monitor ==="
nohup python3 /home/ubuntu/heartbeat.py > /var/log/heartbeat.log 2>&1 &
HEARTBEAT_PID=$!
echo $HEARTBEAT_PID > /var/run/heartbeat.pid

# Restore checkpoint if resuming
if [ -n "$RESUME_FROM" ]; then
    echo "=== Restoring Checkpoint from $RESUME_FROM ==="
    
    # Create cache directory for checkpoints
    CHECKPOINT_CACHE="/home/ubuntu/checkpoint_cache"
    mkdir -p "$CHECKPOINT_CACHE"
    
    # If it's an S3 path, we'll let the training script handle it
    # Otherwise, download it
    if [[ "$RESUME_FROM" == s3://* ]]; then
        echo "S3 checkpoint detected, will be handled by training script"
    else
        echo "Local checkpoint path: $RESUME_FROM"
    fi
fi

# Prepare training command
echo "=== Preparing Training Command ==="
cd /home/ubuntu/training/repo/app

TRAINING_CMD="python scripts/run_sft.py \
    --model-name 'HuggingFaceTB/SmolLM2-360M-Instruct' \
    --max-steps 10000 \
    --save-steps 500 \
    --shard-size-gb 2.0 \
    --s3-bucket '$S3_CHECKPOINT_BUCKET' \
    --s3-prefix 'runs/$RUN_ID/'"

if [ -n "$RESUME_FROM" ]; then
    TRAINING_CMD="$TRAINING_CMD --resume-from '$RESUME_FROM'"
fi

# Write training command to file for reference
echo "$TRAINING_CMD" > /home/ubuntu/training_command.sh
chmod +x /home/ubuntu/training_command.sh

# Start training in tmux session
echo "=== Starting Training in tmux ==="
tmux new-session -d -s training "
    cd /home/ubuntu/training/repo/app
    source /home/ubuntu/training/venv/bin/activate
    export PYTHONUNBUFFERED=1
    export RUN_ID=$RUN_ID
    export S3_HEARTBEAT_BUCKET=$S3_HEARTBEAT_BUCKET
    export S3_CHECKPOINT_BUCKET=$S3_CHECKPOINT_BUCKET
    $TRAINING_CMD 2>&1 | tee /var/log/training.log
"

echo "=== Bootstrap Complete ==="
echo "Training started in tmux session 'training'"
echo "To attach: tmux attach -t training"
echo "Logs available at: /var/log/training.log"
echo "Heartbeat PID: $HEARTBEAT_PID"

# Write final heartbeat
write_heartbeat

# Keep the instance alive
echo "=== Waiting for training to complete ==="
while tmux has-session -t training 2>/dev/null; do
    sleep 60
    write_heartbeat
done

echo "=== Training completed at $(date) ==="

# Final heartbeat
write_heartbeat

# Optional: Shutdown instance after training
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "=== Auto-shutdown enabled, terminating instance in 5 minutes ==="
    sleep 300
    sudo shutdown -h now
fi 