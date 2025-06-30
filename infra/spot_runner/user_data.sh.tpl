#!/bin/bash
# User data script for spot instances - Terraform template
# This script is processed by Terraform to inject variables

set -e

# Export environment variables
export REPO_URL="https://github.com/aimerib/smollmfinetune.git"
export S3_CHECKPOINT_BUCKET="${s3_checkpoint_bucket}"
export S3_HEARTBEAT_BUCKET="${s3_checkpoint_bucket}"
export AWS_DEFAULT_REGION="${region}"
export RUN_ID="${run_id}"

# Call the main bootstrap script with parameters
curl -sSL https://raw.githubusercontent.com/aimerib/smollmfinetune/main/scripts/spot_boot.sh | bash -s -- \
    --git-sha "${git_sha}" \
    --resume-from "${resume_from}" \
    --run-id "${run_id}" \
    --s3-checkpoint-bucket "${s3_checkpoint_bucket}" \
    --s3-heartbeat-bucket "${s3_checkpoint_bucket}" 