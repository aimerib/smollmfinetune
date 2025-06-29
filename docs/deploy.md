# Production Deployment Guide

This guide covers deploying the Character AI Training Studio in production with horizontal scaling, monitoring, and high availability.

## Quick Start (Docker Compose)

### Prerequisites
- Docker Engine 20.10+
- Docker Compose v2.0+
- GPU support (NVIDIA Container Toolkit for GPU workloads)
- 16GB+ RAM recommended
- 100GB+ storage for models and training data

### Basic Production Deployment

1. **Clone and Configure**
   ```bash
   git clone <repository-url>
   cd smollmfinetune
   cp .env.example .env
   ```

2. **Environment Configuration**
   ```bash
   # .env file
   ENVIRONMENT=production
   SENTRY_DSN=your_sentry_dsn_here
   REDIS_URL=redis://redis:6379/0
   DATABASE_URL=sqlite:///data/platform.db
   
   # Optional: Analytics
   MIXPANEL_TOKEN=your_mixpanel_token
   
   # Optional: Cost tracking
   COST_BUDGET=1000  # Monthly budget in USD
   ```

3. **Launch Production Stack**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

4. **Verify Health**
   ```bash
   curl http://localhost:8888/health
   ```

## Horizontal Scaling Architecture

### Multi-GPU Training Setup

For high-throughput training workloads, deploy multiple worker nodes:

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  # Main app (single instance)
  app:
    extends:
      file: docker-compose.prod.yml
      service: app
    deploy:
      replicas: 1

  # Load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - app

  # Multiple GPU workers
  worker-gpu-0:
    extends:
      file: docker-compose.prod.yml
      service: worker
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - WORKER_ID=gpu-0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

  worker-gpu-1:
    extends:
      file: docker-compose.prod.yml
      service: worker
    environment:
      - CUDA_VISIBLE_DEVICES=1
      - WORKER_ID=gpu-1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]

  # Redis cluster for high availability
  redis-master:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 1gb
    
  redis-replica:
    image: redis:7-alpine
    command: redis-server --replicaof redis-master 6379
    depends_on:
      - redis-master
```

### Kubernetes Deployment

For enterprise-scale deployment:

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: character-ai-studio
---
# k8s/app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: character-ai-app
  namespace: character-ai-studio
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
      - name: app
        image: character-ai-studio:latest
        ports:
        - containerPort: 8888
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: DATABASE_URL
          value: "postgresql://user:pass@postgres-service:5432/platform"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 8888
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8888
          initialDelaySeconds: 30
          periodSeconds: 10
---
# k8s/worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: character-ai-worker
  namespace: character-ai-studio
spec:
  replicas: 4
  selector:
    matchLabels:
      app: character-ai-worker
  template:
    metadata:
      labels:
        app: character-ai-worker
    spec:
      containers:
      - name: worker
        image: character-ai-studio:latest
        command: ["celery", "worker", "-A", "worker", "--loglevel=info"]
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
      nodeSelector:
        nvidia.com/gpu: "true"
```

## Resource Planning

### Minimum Requirements

| Component | CPU | RAM | Storage | GPU |
|-----------|-----|-----|---------|-----|
| App Server | 2 cores | 4GB | 20GB | Optional |
| Training Worker | 4 cores | 8GB | 100GB | RTX 3080+ |
| Redis | 1 core | 2GB | 10GB | No |
| Database | 2 cores | 4GB | 50GB | No |

### Recommended Production Setup

| Component | CPU | RAM | Storage | GPU | Replicas |
|-----------|-----|-----|---------|-----|----------|
| App Server | 4 cores | 8GB | 50GB | No | 2-3 |
| Training Worker | 8 cores | 16GB | 500GB | RTX 4090 | 2-4 |
| Redis Cluster | 2 cores | 4GB | 20GB | No | 3 |
| PostgreSQL | 4 cores | 8GB | 200GB | No | 1 (HA) |

### Performance Targets

- **Concurrent Users**: 50-100 simultaneous users
- **Training Jobs**: 10+ concurrent training runs
- **API Response Time**: <500ms 95th percentile
- **Training Throughput**: 1000+ samples/hour per GPU
- **Uptime**: 99.9% availability

## Monitoring & Observability

### Health Monitoring

The application exposes comprehensive health endpoints:

```bash
# Basic health check
curl http://localhost:8888/health

# Detailed health with metrics
curl http://localhost:8888/health?detailed=true
```

Health check response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-14T12:00:00Z",
  "uptime_seconds": 3600,
  "checks": {
    "database": {"status": "healthy", "tables_count": 12},
    "redis": {"status": "healthy", "connected_clients": 5},
    "system": {"status": "healthy", "cpu_percent": 45.2},
    "storage": {"status": "healthy", "adapters_count": 15}
  }
}
```

### Prometheus Metrics

Key metrics exposed at `/metrics`:

- `app_requests_total` - Total HTTP requests
- `app_request_duration_seconds` - Request duration histogram
- `training_jobs_active` - Active training jobs
- `training_jobs_completed_total` - Completed training jobs
- `gpu_memory_usage_bytes` - GPU memory usage
- `model_inference_duration_seconds` - Model inference latency

### Grafana Dashboard

Import the dashboard from `monitoring/grafana-dashboard.json`:

**Key Panels:**
- System Resource Usage (CPU, Memory, GPU)
- Training Job Statistics
- API Performance Metrics
- Error Rate Tracking
- Cost Analysis

### Alerting Rules

```yaml
# prometheus/alerts.yml
groups:
- name: character-ai-studio
  rules:
  - alert: HighErrorRate
    expr: rate(app_requests_total{status="error"}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      
  - alert: TrainingJobStuck
    expr: training_jobs_active > 0 and rate(training_progress_total[10m]) == 0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Training job appears stuck"
      
  - alert: HighGPUMemoryUsage
    expr: gpu_memory_usage_bytes / gpu_memory_total_bytes > 0.95
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "GPU memory usage > 95%"
```

## Security Hardening

### SSL/TLS Configuration

```nginx
# nginx/ssl.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://app:8888;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Rate Limiting

Built-in rate limiting is configured in the application:
- 60 requests/minute per user
- 10 training jobs per user per hour
- File upload limits: 100MB per file

### Access Control

Configure role-based access:
```python
# User roles and permissions
ADMIN: Full system access
CREATOR: Create/train characters, manage worlds
PLAYER: Use characters, discover worlds
```

## Backup & Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# SQLite backup
sqlite3 /data/platform.db ".backup $BACKUP_DIR/platform.db"

# Training data backup
tar -czf "$BACKUP_DIR/training_output.tar.gz" /app/training_output/

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR" s3://your-backup-bucket/$(date +%Y%m%d)/ --recursive
```

### Disaster Recovery

1. **Database Recovery**
   ```bash
   # Restore from backup
   cp /backups/latest/platform.db /data/platform.db
   ```

2. **Training Data Recovery**
   ```bash
   # Restore training outputs
   tar -xzf /backups/latest/training_output.tar.gz -C /app/
   ```

3. **Application Recovery**
   ```bash
   # Restart services
   docker-compose -f docker-compose.prod.yml down
   docker-compose -f docker-compose.prod.yml up -d
   ```

## Cost Optimization

### Resource Optimization

1. **GPU Scheduling**
   - Use spot instances for non-critical training
   - Scale workers based on queue depth
   - Implement training job prioritization

2. **Storage Optimization**
   - Implement model compression
   - Automatic cleanup of old training artifacts
   - Use S3 lifecycle policies for archival

3. **Compute Optimization**
   - Auto-scaling based on CPU/memory usage
   - Scheduled scaling for predictable workloads
   - Container resource limits

### Cost Monitoring

The application tracks:
- GPU usage costs
- API call costs (OpenAI, etc.)
- Infrastructure costs
- Storage costs

Monthly cost dashboard shows:
- Cost per user
- Cost per training job
- Cost trends and projections
- Budget alerts

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats
   
   # Restart workers if needed
   docker-compose restart worker
   ```

2. **Training Jobs Stuck**
   ```bash
   # Check worker logs
   docker-compose logs worker
   
   # Clear stuck jobs
   docker exec redis redis-cli FLUSHDB
   ```

3. **GPU Out of Memory**
   ```bash
   # Check GPU usage
   nvidia-smi
   
   # Reduce batch size in training config
   # Or restart GPU workers
   ```

### Performance Tuning

1. **Database Optimization**
   ```sql
   -- Add indexes for common queries
   CREATE INDEX idx_training_runs_status ON training_runs(status);
   CREATE INDEX idx_characters_user_id ON characters(user_id);
   ```

2. **Redis Optimization**
   ```redis
   # Configure memory limits
   CONFIG SET maxmemory 2gb
   CONFIG SET maxmemory-policy allkeys-lru
   ```

3. **App Configuration**
   ```env
   # Optimize worker concurrency
   CELERY_WORKER_CONCURRENCY=4
   
   # Tune vLLM settings
   VLLM_GPU_MEMORY_UTILIZATION=0.85
   VLLM_MAX_MODEL_LEN=4096
   ```

## Support & Maintenance

### Regular Maintenance Tasks

1. **Weekly**
   - Review error logs
   - Check disk usage
   - Validate backups

2. **Monthly**
   - Update security patches
   - Review performance metrics
   - Clean up old data

3. **Quarterly**
   - Capacity planning review
   - Security audit
   - Disaster recovery testing

### Getting Help

- **Documentation**: [Link to docs]
- **Community**: [Discord/Forum link]
- **Enterprise Support**: [Contact information]

For production issues, include:
- Health check output
- Recent logs
- Performance metrics
- Configuration details 