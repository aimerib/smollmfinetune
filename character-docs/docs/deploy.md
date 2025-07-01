---
sidebar_position: 7
---

# Production Deployment Guide

This guide covers production deployment scenarios for the Character AI Platform, including cloud infrastructure, monitoring, scaling, and best practices.

<div style={{background: 'linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%)', padding: '2rem', borderRadius: '12px', color: 'white', marginBottom: '2rem'}}>
  <h2 style={{marginTop: 0}}>🚀 Production Ready</h2>
  <p>Deploy your Character AI Platform to production with confidence using our battle-tested configurations and monitoring setup.</p>
</div>

## Architecture Overview

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[NGINX/CloudFlare]
    end
    
    subgraph "Application Layer"
        APP1[Character Studio 1]
        APP2[Character Studio 2]
        APP3[Character Studio 3]
    end
    
    subgraph "Inference Layer"
        INF1[vLLM Server 1]
        INF2[vLLM Server 2]
    end
    
    subgraph "Storage Layer"
        DB[(PostgreSQL)]
        REDIS[(Redis Cache)]
        S3[(Model Storage)]
    end
    
    subgraph "Monitoring"
        PROM[Prometheus]
        GRAF[Grafana]
        ALERT[AlertManager]
    end
    
    LB --> APP1
    LB --> APP2
    LB --> APP3
    
    APP1 --> INF1
    APP2 --> INF1
    APP3 --> INF2
    
    APP1 --> DB
    APP2 --> DB
    APP3 --> DB
    
    APP1 --> REDIS
    APP2 --> REDIS
    APP3 --> REDIS
    
    INF1 --> S3
    INF2 --> S3
    
    PROM --> APP1
    PROM --> APP2
    PROM --> APP3
    PROM --> INF1
    PROM --> INF2
    
    GRAF --> PROM
    ALERT --> PROM
```

## Infrastructure as Code

### Terraform Configuration

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC Configuration
resource "aws_vpc" "character_ai_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "character-ai-vpc"
  }
}

# GPU Instance for Inference
resource "aws_instance" "inference_server" {
  count         = var.inference_server_count
  ami           = "ami-0c02fb55956c7d316"  # Deep Learning AMI
  instance_type = "g4dn.xlarge"           # NVIDIA T4 GPU
  
  subnet_id                   = aws_subnet.private_subnet.id
  vpc_security_group_ids      = [aws_security_group.inference_sg.id]
  associate_public_ip_address = false
  
  root_block_device {
    volume_type = "gp3"
    volume_size = 100
    encrypted   = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data_inference.sh", {
    model_s3_bucket = aws_s3_bucket.models.bucket
  }))

  tags = {
    Name = "character-ai-inference-${count.index + 1}"
    Type = "inference"
  }
}

# Application Servers
resource "aws_instance" "app_server" {
  count         = var.app_server_count
  ami           = "ami-0c02fb55956c7d316"
  instance_type = "c5.2xlarge"
  
  subnet_id                   = aws_subnet.private_subnet.id
  vpc_security_group_ids      = [aws_security_group.app_sg.id]
  associate_public_ip_address = false
  
  user_data = base64encode(templatefile("${path.module}/user_data_app.sh", {
    db_endpoint = aws_rds_instance.character_ai_db.endpoint
    redis_endpoint = aws_elasticache_cluster.character_ai_cache.cache_nodes[0].address
  }))

  tags = {
    Name = "character-ai-app-${count.index + 1}"
    Type = "application"
  }
}

# RDS Database
resource "aws_rds_instance" "character_ai_db" {
  identifier = "character-ai-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "character_ai"
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.db_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.character_ai_db_subnet_group.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "character-ai-db-final-snapshot"
  
  tags = {
    Name = "character-ai-database"
  }
}

# S3 Bucket for Models
resource "aws_s3_bucket" "models" {
  bucket = "character-ai-models-${random_string.bucket_suffix.result}"
  
  tags = {
    Name = "character-ai-models"
  }
}

resource "aws_s3_bucket_versioning" "models_versioning" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "inference_server_count" {
  description = "Number of inference servers"
  type        = number
  default     = 2
}

variable "app_server_count" {
  description = "Number of application servers"
  type        = number
  default     = 3
}

variable "db_username" {
  description = "Database username"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}
```

### Kubernetes Deployment

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: character-ai

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: character-ai-config
  namespace: character-ai
data:
  ENVIRONMENT: "production"
  STREAMLIT_SERVER_HEADLESS: "true"
  STREAMLIT_SERVER_ADDRESS: "0.0.0.0"
  STREAMLIT_SERVER_PORT: "8888"
  VLLM_GPU_MEMORY_UTILIZATION: "0.90"
  VLLM_MAX_MODEL_LEN: "4096"

---
# k8s/deployment-app.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: character-ai-app
  namespace: character-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: character-ai-app
  template:
    metadata:
      labels:
        app: character-ai-app
    spec:
      containers:
      - name: character-ai
        image: your-registry/character-ai:latest
        ports:
        - containerPort: 8888
        envFrom:
        - configMapRef:
            name: character-ai-config
        - secretRef:
            name: character-ai-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8888
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8888
          initialDelaySeconds: 30
          periodSeconds: 10

---
# k8s/deployment-inference.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: character-ai-inference
  namespace: character-ai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: character-ai-inference
  template:
    metadata:
      labels:
        app: character-ai-inference
    spec:
      nodeSelector:
        node-type: gpu
      containers:
      - name: vllm-server
        image: vllm/vllm-openai:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_NAME
          value: "PocketDoc/Dans-PersonalityEngine-V1.3.0-24b"
        - name: GPU_MEMORY_UTILIZATION
          value: "0.90"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "24Gi"
            cpu: "8"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 60

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: character-ai-app-service
  namespace: character-ai
spec:
  selector:
    app: character-ai-app
  ports:
  - port: 80
    targetPort: 8888
  type: ClusterIP

---
apiVersion: v1
kind: Service
metadata:
  name: character-ai-inference-service
  namespace: character-ai
spec:
  selector:
    app: character-ai-inference
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP

---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: character-ai-ingress
  namespace: character-ai
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: character-ai-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: character-ai-app-service
            port:
              number: 80
```

## Docker Production Setup

### Multi-Stage Production Dockerfile

```dockerfile
# Dockerfile.production
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY app/requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY app/ .
COPY narrative_engine/ /app/narrative_engine/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8888/_stcore/health || exit 1

# Expose port
EXPOSE 8888

# Start application
CMD ["streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8888", \
     "--server.headless", "true", \
     "--server.enableCORS", "false", \
     "--server.enableXsrfProtection", "true"]
```

### Docker Compose Production Stack

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  character-ai-app:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8888:8888"
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/character_ai
      - REDIS_URL=redis://redis:6379/0
      - INFERENCE_URL=http://inference:8000
    depends_on:
      - db
      - redis
      - inference
    restart: unless-stopped
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

  inference:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=PocketDoc/Dans-PersonalityEngine-V1.3.0-24b
      - GPU_MEMORY_UTILIZATION=0.90
      - MAX_MODEL_LEN=4096
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=character_ai
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - character-ai-app
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'character-ai-app'
    static_configs:
      - targets: ['character-ai-app:8888']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'vllm-inference'
    static_configs:
      - targets: ['inference:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Custom Metrics Collection

```python
# app/utils/metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import psutil
import GPUtil

# Custom metrics
CHARACTER_GENERATIONS = Counter('character_generations_total', 'Total character generations', ['character_id', 'model'])
TRAINING_DURATION = Histogram('training_duration_seconds', 'Training duration in seconds')
ACTIVE_USERS = Gauge('active_users', 'Number of active users')
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes', ['type'])

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        
    def record_generation(self, character_id: str, model: str):
        CHARACTER_GENERATIONS.labels(character_id=character_id, model=model).inc()
    
    def record_training_duration(self, duration: float):
        TRAINING_DURATION.observe(duration)
    
    def update_system_metrics(self):
        # CPU and Memory
        memory = psutil.virtual_memory()
        MEMORY_USAGE.labels(type='used').set(memory.used)
        MEMORY_USAGE.labels(type='available').set(memory.available)
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                GPU_UTILIZATION.set(gpus[0].load * 100)
        except:
            pass
    
    def start_metrics_server(self, port=8000):
        start_http_server(port)

# Usage in app.py
metrics = MetricsCollector()
metrics.start_metrics_server(8001)
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "id": null,
    "title": "Character AI Platform",
    "tags": ["character-ai"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Character Generations per Minute",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(character_generations_total[1m])",
            "legendFormat": "{{character_id}} - {{model}}"
          }
        ],
        "yAxes": [
          {
            "label": "Generations/min"
          }
        ]
      },
      {
        "id": 2,
        "title": "GPU Utilization",
        "type": "singlestat",
        "targets": [
          {
            "expr": "gpu_utilization_percent",
            "legendFormat": "GPU %"
          }
        ],
        "thresholds": "70,90"
      },
      {
        "id": 3,
        "title": "Active Users",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_users"
          }
        ]
      },
      {
        "id": 4,
        "title": "Training Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, training_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, training_duration_seconds_bucket)",
            "legendFormat": "50th percentile"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## Security Hardening

### SSL/TLS Configuration

```nginx
# nginx-ssl.conf
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location / {
        proxy_pass http://character-ai-app:8888;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Environment Security

```bash
#!/bin/bash
# security-hardening.sh

# Update system
apt-get update && apt-get upgrade -y

# Install security tools
apt-get install -y fail2ban ufw unattended-upgrades

# Configure firewall
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

# Configure fail2ban
cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log

[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
EOF

systemctl enable fail2ban
systemctl start fail2ban

# Configure automatic security updates
echo 'Unattended-Upgrade::Automatic-Reboot "false";' >> /etc/apt/apt.conf.d/50unattended-upgrades
systemctl enable unattended-upgrades

# Secure shared memory
echo 'tmpfs /run/shm tmpfs defaults,noexec,nosuid 0 0' >> /etc/fstab

# Set secure file permissions
chmod 600 /etc/ssh/sshd_config
chmod 644 /etc/passwd
chmod 600 /etc/shadow

echo "✅ Security hardening completed"
```

## Scaling and Performance

### Auto-scaling Configuration

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: character-ai-app-hpa
  namespace: character-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: character-ai-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Load Testing

```python
# scripts/load_test.py
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

async def generate_character_interaction(session, character_id, message):
    url = "http://localhost:8888/api/chat"
    payload = {
        "character_id": character_id,
        "message": message,
        "temperature": 0.7
    }
    
    start_time = time.time()
    async with session.post(url, json=payload) as response:
        result = await response.json()
        duration = time.time() - start_time
        return duration, response.status

async def run_load_test(concurrent_users=10, requests_per_user=50):
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for user in range(concurrent_users):
            for request in range(requests_per_user):
                task = generate_character_interaction(
                    session, 
                    f"test-character-{user % 5}",
                    f"Hello, this is test message {request}"
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        durations = [r[0] for r in results]
        statuses = [r[1] for r in results]
        
        print(f"Total requests: {len(results)}")
        print(f"Average response time: {sum(durations) / len(durations):.2f}s")
        print(f"95th percentile: {sorted(durations)[int(len(durations) * 0.95)]:.2f}s")
        print(f"Success rate: {statuses.count(200) / len(statuses) * 100:.1f}%")

if __name__ == "__main__":
    asyncio.run(run_load_test(concurrent_users=50, requests_per_user=100))
```

## Backup and Disaster Recovery

### Automated Backup System

```bash
#!/bin/bash
# production-backup.sh

set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
S3_BUCKET="your-backup-bucket"

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Database backup
echo "🗄️ Backing up database..."
docker exec character-ai-db pg_dump -U postgres character_ai | gzip > "$BACKUP_DIR/$DATE/database.sql.gz"

# Application data backup
echo "📁 Backing up application data..."
tar -czf "$BACKUP_DIR/$DATE/app_data.tar.gz" /app/data

# Model files backup
echo "🤖 Backing up model files..."
tar -czf "$BACKUP_DIR/$DATE/models.tar.gz" /app/models

# Upload to S3
echo "☁️ Uploading to S3..."
aws s3 sync "$BACKUP_DIR/$DATE" "s3://$S3_BUCKET/backups/$DATE/"

# Cleanup local backups (keep last 7 days)
find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \;

# Cleanup S3 backups (keep last 30 days)
aws s3 ls "s3://$S3_BUCKET/backups/" | while read -r line; do
    backup_date=$(echo $line | awk '{print $2}' | tr -d '/')
    if [[ $(date -d "$backup_date" +%s) -lt $(date -d "30 days ago" +%s) ]]; then
        aws s3 rm "s3://$S3_BUCKET/backups/$backup_date/" --recursive
    fi
done

echo "✅ Backup completed: $DATE"

# Send notification
curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"✅ Character AI backup completed: $DATE\"}" \
    "$SLACK_WEBHOOK_URL"
```

### Disaster Recovery Plan
#### 1. Assess the Situation
- Check monitoring dashboards
- Identify failed components
- Estimate recovery time
    
#### 2. Communication
- Notify stakeholders
- Update status page
- Set up incident channel
    
#### 3. Database Recovery
```bash
# Restore from latest backup
BACKUP_DATE="20241201_120000"
aws s3 cp "s3://your-backup-bucket/backups/$BACKUP_DATE/database.sql.gz" .
gunzip database.sql.gz

# Restore to new database instance
psql -h new-db-host -U postgres -d character_ai < database.sql
```
    
#### 4. Application Recovery
```bash
# Update database connection
kubectl patch deployment character-ai-app -p '{"spec":{"template":{"spec":{"containers":[{"name":"character-ai","env":[{"name":"DATABASE_URL","value":"postgresql://postgres:password@new-db-host:5432/character_ai"}]}]}}}}'

# Scale up replicas
kubectl scale deployment character-ai-app --replicas=5
```
    
#### 5. Verification
- Check application health
- Run smoke tests
- Verify data integrity

#### 6. Post-Incident
- Document lessons learned
- Update runbooks
- Improve monitoring


## Cost Optimization

### Resource Right-sizing

```python
# scripts/cost_optimizer.py
import boto3
import json
from datetime import datetime, timedelta

def analyze_instance_utilization():
    cloudwatch = boto3.client('cloudwatch')
    ec2 = boto3.client('ec2')
    
    instances = ec2.describe_instances()
    recommendations = []
    
    for reservation in instances['Reservations']:
        for instance in reservation['Instances']:
            instance_id = instance['InstanceId']
            instance_type = instance['InstanceType']
            
            # Get CPU utilization for last 7 days
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=7)
            
            cpu_metrics = cloudwatch.get_metric_statistics(
                Namespace='AWS/EC2',
                MetricName='CPUUtilization',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average']
            )
            
            if cpu_metrics['Datapoints']:
                avg_cpu = sum(dp['Average'] for dp in cpu_metrics['Datapoints']) / len(cpu_metrics['Datapoints'])
                
                if avg_cpu < 20:
                    recommendations.append({
                        'instance_id': instance_id,
                        'current_type': instance_type,
                        'avg_cpu': avg_cpu,
                        'recommendation': 'Consider downsizing or using spot instances'
                    })
    
    return recommendations

def generate_cost_report():
    ce = boto3.client('ce')
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    response = ce.get_cost_and_usage(
        TimePeriod={'Start': start_date, 'End': end_date},
        Granularity='MONTHLY',
        Metrics=['BlendedCost'],
        GroupBy=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
    )
    
    costs = {}
    for result in response['ResultsByTime']:
        for group in result['Groups']:
            service = group['Keys'][0]
            cost = float(group['Metrics']['BlendedCost']['Amount'])
            costs[service] = cost
    
    return costs

if __name__ == "__main__":
    print("🔍 Analyzing instance utilization...")
    recommendations = analyze_instance_utilization()
    
    print("💰 Generating cost report...")
    costs = generate_cost_report()
    
    print("\n📊 Cost Report (Last 30 days):")
    for service, cost in sorted(costs.items(), key=lambda x: x[1], reverse=True):
        if cost > 0:
            print(f"  {service}: ${cost:.2f}")
    
    print("\n🎯 Optimization Recommendations:")
    for rec in recommendations:
        print(f"  Instance {rec['instance_id']} ({rec['current_type']}): {rec['recommendation']} (Avg CPU: {rec['avg_cpu']:.1f}%)")
```