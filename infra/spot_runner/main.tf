terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "instance_type" {
  description = "EC2 instance type for spot instance"
  type        = string
  default     = "p4d.24xlarge"
}

variable "spot_price" {
  description = "Maximum spot price per hour"
  type        = number
  default     = 10.0
}

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "key_name" {
  description = "EC2 key pair name for SSH access"
  type        = string
  default     = "training-key"
}

variable "s3_checkpoint_bucket" {
  description = "S3 bucket for checkpoint storage"
  type        = string
  default     = "training-checkpoints"
}

variable "daily_budget" {
  description = "Daily budget limit in USD"
  type        = number
  default     = 100.0
}

variable "git_sha" {
  description = "Git commit SHA to checkout"
  type        = string
  default     = ""
}

variable "resume_from" {
  description = "Checkpoint path to resume from"
  type        = string
  default     = ""
}

variable "run_id" {
  description = "Unique run identifier"
  type        = string
  default     = ""
}

# Provider configuration
provider "aws" {
  region = var.region
}

# Data sources
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch * (Ubuntu 20.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

# IAM role for the instance
resource "aws_iam_role" "training_instance" {
  name = "training-spot-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# IAM policy for S3 access
resource "aws_iam_policy" "s3_access" {
  name        = "training-s3-access"
  description = "S3 access for training checkpoints and heartbeats"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_checkpoint_bucket}",
          "arn:aws:s3:::${var.s3_checkpoint_bucket}/*"
        ]
      }
    ]
  })
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "s3_access" {
  role       = aws_iam_role.training_instance.name
  policy_arn = aws_iam_policy.s3_access.arn
}

# Instance profile
resource "aws_iam_instance_profile" "training_instance" {
  name = "training-instance-profile"
  role = aws_iam_role.training_instance.name
}

# Security group
resource "aws_security_group" "training_instance" {
  name        = "training-spot-instances"
  description = "Security group for training spot instances"

  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Restrict this in production
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "training-spot-instances"
  }
}

# S3 bucket for checkpoints (create if doesn't exist)
resource "aws_s3_bucket" "checkpoints" {
  bucket = var.s3_checkpoint_bucket

  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Purpose = "training-checkpoints"
  }
}

# S3 bucket versioning
resource "aws_s3_bucket_versioning" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket lifecycle rules to save costs
resource "aws_s3_bucket_lifecycle_configuration" "checkpoints" {
  bucket = aws_s3_bucket.checkpoints.id

  rule {
    id     = "transition-old-checkpoints"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }

  rule {
    id     = "expire-old-heartbeats"
    status = "Enabled"

    filter {
      prefix = "heartbeats/"
    }

    expiration {
      days = 7
    }
  }
}

# Bootstrap script template
locals {
  run_id = var.run_id != "" ? var.run_id : "run_${formatdate("YYYYMMDD_hhmmss", timestamp())}"
  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    git_sha              = var.git_sha
    resume_from          = var.resume_from
    run_id               = local.run_id
    s3_checkpoint_bucket = var.s3_checkpoint_bucket
    region               = var.region
  })
}

# Spot instance request
resource "aws_spot_instance_request" "training" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.training_instance.id]
  iam_instance_profile   = aws_iam_instance_profile.training_instance.name
  
  spot_price                      = var.spot_price
  wait_for_fulfillment           = true
  instance_interruption_behavior = "terminate"
  
  user_data = local.user_data

  root_block_device {
    volume_size           = 500
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name    = "training-spot-${local.run_id}"
    Purpose = "model-training"
    RunID   = local.run_id
  }

  lifecycle {
    create_before_destroy = true
  }
}

# CloudWatch metric alarm for cost monitoring
resource "aws_cloudwatch_metric_alarm" "cost_alarm" {
  alarm_name          = "training-cost-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"  # 24 hours
  statistic           = "Maximum"
  threshold           = var.daily_budget
  alarm_description   = "Alert when daily spending exceeds budget"

  dimensions = {
    Currency = "USD"
  }
}

# Outputs
output "instance_id" {
  description = "Spot instance request ID"
  value       = aws_spot_instance_request.training.id
}

output "public_ip" {
  description = "Public IP of the instance"
  value       = aws_spot_instance_request.training.public_ip
}

output "s3_bucket" {
  description = "S3 bucket for checkpoints"
  value       = aws_s3_bucket.checkpoints.id
}

output "run_id" {
  description = "Training run ID"
  value       = local.run_id
}

output "ssh_command" {
  description = "SSH command to connect to instance"
  value       = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_spot_instance_request.training.public_ip}"
} 