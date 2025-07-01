# Checkpoint Sharding for Distributed Training

Enable cheap, long-running weekend training by saving Narrative-LLM checkpoints as sharded `.safetensors` files across S3-compatible object stores, allowing interruptions and resume on spot instances.

## Overview

Checkpoint sharding splits large model checkpoints into multiple smaller files (shards) that can be stored and transferred more efficiently. This is especially useful for:

- **Spot Instance Training**: Resume seamlessly after interruptions
- **Large Models**: Avoid single-file size limits and transfer issues
- **Cost Optimization**: Use cheaper storage and spot compute instances
- **Fault Tolerance**: Partial downloads can resume from where they left off

## Quick Start

### Basic Usage

Enable checkpoint sharding by adding the `--shard-size-gb` parameter to your training script:

```bash
python scripts/run_sft.py \
    --model-name "HuggingFaceTB/SmolLM2-360M-Instruct" \
    --max-steps 1000 \
    --shard-size-gb 2.0 \
    --output-dir "./training_output"
```

### With S3 Storage

For automatic upload to S3-compatible storage:

```bash
python scripts/run_sft.py \
    --model-name "HuggingFaceTB/SmolLM2-360M-Instruct" \
    --max-steps 1000 \
    --shard-size-gb 2.0 \
    --s3-bucket "my-training-bucket" \
    --s3-prefix "experiments/character_alice/" \
    --output-dir "./training_output"
```

### Resume from Sharded Checkpoint

Resume training from a sharded checkpoint (local or S3):

```bash
python scripts/run_sft.py \
    --resume-from "./training_output/checkpoint-500/shards/checkpoint.json" \
    --shard-size-gb 2.0
```

Or from S3:

```bash
python scripts/run_sft.py \
    --resume-from "s3://my-bucket/experiments/character_alice/checkpoint-500/checkpoint.json" \
    --shard-size-gb 2.0 \
    --s3-bucket "my-training-bucket"
```

## Configuration Options

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--shard-size-gb` | float | 0 | Maximum shard size in GB (0 = disabled) |
| `--s3-bucket` | string | None | S3 bucket for checkpoint storage |
| `--s3-prefix` | string | None | S3 folder prefix for checkpoints |
| `--s3-endpoint-url` | string | None | Custom S3 endpoint (MinIO, Wasabi, etc.) |
| `--resume-from` | string | None | Path to checkpoint manifest or directory |

### Environment Variables

Configure S3 access using standard AWS environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"

# For custom S3 endpoints (MinIO, Wasabi, etc.)
export AWS_ENDPOINT_URL="https://s3.wasabisys.com"
```

### Training Configuration

Add sharding parameters to your training config dictionary:

```python
config = {
    'max_steps': 1000,
    'save_steps': 100,
    'shard_size_gb': 2.0,           # Enable sharding with 2GB shards
    's3_bucket': 'my-bucket',       # Optional S3 upload
    's3_prefix': 'training/char1/', # S3 folder structure
    's3_endpoint_url': None,        # Use AWS S3 by default
}
```

## S3 Provider Examples

### AWS S3

```bash
python scripts/run_sft.py \
    --shard-size-gb 4.0 \
    --s3-bucket "my-aws-bucket" \
    --s3-prefix "training/models/"
```

### MinIO (Self-hosted)

```bash
export AWS_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"

python scripts/run_sft.py \
    --shard-size-gb 2.0 \
    --s3-bucket "training-checkpoints" \
    --s3-endpoint-url "http://localhost:9000"
```

### Wasabi Cloud Storage

```bash
export AWS_ENDPOINT_URL="https://s3.wasabisys.com"
export AWS_ACCESS_KEY_ID="your-wasabi-key"
export AWS_SECRET_ACCESS_KEY="your-wasabi-secret"

python scripts/run_sft.py \
    --shard-size-gb 4.0 \
    --s3-bucket "my-wasabi-bucket" \
    --s3-endpoint-url "https://s3.wasabisys.com"
```

## File Structure

Sharded checkpoints create the following directory structure:

```
training_output/
└── checkpoint-500/
    ├── training_metadata.json          # Training configuration
    ├── training_summary.json           # Metrics and logs
    └── shards/                         # Sharded checkpoint files
        ├── checkpoint.json             # Manifest file
        ├── pytorch_model-00001-of-00003.safetensors
        ├── pytorch_model-00002-of-00003.safetensors
        └── pytorch_model-00003-of-00003.safetensors
```

### Manifest Format

The `checkpoint.json` manifest contains metadata about the shards:

```json
{
  "shards": [
    "pytorch_model-00001-of-00003.safetensors",
    "pytorch_model-00002-of-00003.safetensors", 
    "pytorch_model-00003-of-00003.safetensors"
  ],
  "total_size": 8589934592,
  "num_shards": 3,
  "shard_size_gb": 4.0,
  "created_at": 1703123456.789,
  "tensor_count": 1247
}
```

## Cost Analysis

### Storage Costs

Sharded checkpoints can reduce storage costs by:

- **Deduplication**: Identical shards across checkpoints can be deduplicated
- **Compression**: Better compression ratios for smaller files
- **Lifecycle Policies**: Move older shards to cheaper storage tiers

**Example**: Training a 1.3B parameter model for 48 hours

| Storage Type | Cost per GB/month | Checkpoint Size | Monthly Cost |
|--------------|-------------------|-----------------|--------------|
| AWS S3 Standard | $0.023 | 5.2 GB | $0.12 |
| AWS S3 IA | $0.0125 | 5.2 GB | $0.065 |
| Wasabi | $0.0059 | 5.2 GB | $0.031 |

### Compute Costs

Spot instance savings with fault-tolerant checkpointing:

| Instance Type | On-Demand | Spot Price | Savings | Interruption Risk |
|---------------|-----------|------------|---------|-------------------|
| g4dn.xlarge | $0.526/hr | $0.158/hr | 70% | ~5% per hour |
| p3.2xlarge | $3.06/hr | $0.918/hr | 70% | ~5% per hour |
| g5.2xlarge | $1.006/hr | $0.302/hr | 70% | ~5% per hour |

**Training Time Recovery**: With 5-minute checkpointing, average recovery time is 2.5 minutes per interruption.

### Weekend Training Example

Training a 1.3B model over 48 hours on weekends:

- **Without Spot**: $3.06 × 48 = $146.88
- **With Spot + Sharding**: $0.918 × 48 = $44.06
- **Savings**: $102.82 (70% reduction)

## Advanced Usage

### Custom Shard Loading

```python
from app.utils.checkpointing import ShardLoader

# Load from local sharded checkpoint
loader = ShardLoader()
state_dict = loader.load_from_directory("./checkpoint-500/shards")

# Load from S3 with caching
state_dict = loader.load_from_s3(
    bucket="my-bucket",
    prefix="training/checkpoint-500/",
    local_cache_dir="./cache"
)
```

### Manual Sharding

```python
from app.utils.checkpointing import ShardWriter

# Create shards from state dict
writer = ShardWriter(shard_size_gb=2.0)
manifest_path = writer.save_shards(model.state_dict(), "./shards")

# Upload to S3
success = writer.upload_to_s3(
    local_dir="./shards",
    bucket="my-bucket", 
    prefix="checkpoints/",
    endpoint_url="https://s3.wasabisys.com"
)
```

### Checkpoint Verification

```python
from app.utils.checkpointing import ShardLoader

loader = ShardLoader()
is_valid = loader.verify_checkpoint_integrity("./checkpoint-500/shards")
if not is_valid:
    print("❌ Checkpoint corruption detected!")
```

## Troubleshooting

### Common Issues

**Error: "Shard file not found"**
- Check that all shard files are present in the directory
- Verify S3 credentials and bucket permissions
- Ensure manifest file is valid JSON

**Error: "Out of memory during sharding"**
- Reduce shard size: `--shard-size-gb 1.0`
- Enable CPU offloading if available
- Monitor memory usage during checkpointing

**Error: "S3 upload failed"**
- Check AWS credentials: `aws s3 ls` 
- Verify bucket exists and is accessible
- Check network connectivity and endpoint URL

### Performance Tips

1. **Optimal Shard Size**: 2-4GB provides good balance of transfer speed and file management
2. **Parallel Uploads**: Large shards will use multipart upload automatically
3. **Local Caching**: Keep a local cache of recent shards for faster resume
4. **Network**: Use instances with enhanced networking for faster S3 transfers

### Monitoring

Monitor sharding progress and health:

```python
# Check shard creation progress
import logging
logging.getLogger('app.utils.checkpointing').setLevel(logging.INFO)

# Monitor S3 upload progress
def progress_callback(current, total, filename):
    percent = (current / total) * 100
    print(f"Upload progress: {filename} {percent:.1f}%")

writer.upload_to_s3(..., progress_callback=progress_callback)
```

## Integration with Existing Workflows

### CI/CD Pipelines

```yaml
# GitHub Actions example
- name: Train with Sharding
  run: |
    python scripts/run_sft.py \
      --shard-size-gb 2.0 \
      --s3-bucket "${{ secrets.S3_BUCKET }}" \
      --s3-prefix "ci/run-${{ github.run_id }}/"
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
    AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

### Slurm/SBATCH Scripts

```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Setup environment
export AWS_ACCESS_KEY_ID=$SLURM_S3_KEY
export AWS_SECRET_ACCESS_KEY=$SLURM_S3_SECRET

# Run training with sharding
python scripts/run_sft.py \
  --shard-size-gb 4.0 \
  --s3-bucket "hpc-training-checkpoints" \
  --s3-prefix "job-$SLURM_JOB_ID/" \
  --max-steps 10000
```

## Best Practices

1. **Regular Checkpointing**: Use `save_steps` of 50-100 for spot instances
2. **Unique Prefixes**: Include timestamps or run IDs in S3 prefixes
3. **Cleanup**: Remove old checkpoints to control storage costs
4. **Testing**: Verify resume functionality before long training runs
5. **Monitoring**: Set up alerts for failed uploads or corrupted checkpoints

For more information, see the API documentation in the `app.utils.checkpointing` module. 