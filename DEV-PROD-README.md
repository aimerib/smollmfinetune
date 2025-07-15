# 🚀 Production-like Development Environment

This setup provides a development environment that **mirrors production architecture** but is optimized for local development on Mac with 32GB RAM.

## 🎯 Why This Approach?

Running your development environment as close to production as possible helps you:
- **Catch issues early** that only appear in production-like setups
- **Understand operational challenges** before they hit production
- **Test load balancing and scaling** in a realistic environment
- **Debug complex multi-service interactions** locally
- **Build confidence** in your production deployment

## 🏗️ Architecture Overview

This environment includes **all production components** but scaled down for development:

### 🎮 Frontend (React)
- **2 replicas** (vs 3 in production) for load balancing testing
- Hot reloading enabled for development
- Source maps enabled for debugging

### 🔧 Backend (FastAPI)
- **2 replicas** (vs 4 in production) with debug ports exposed
- Live code reloading with volume mounts
- Debug ports: 5678, 5679

### 🧠 Supporting Services
- **PostgreSQL** with primary/replica setup
- **Redis** cluster for caching and message queuing
- **Celery** workers for async tasks
- **Inference Engine** for character processing
- **Nginx** load balancer (simplified for dev)

### 📊 Monitoring Stack
- **Prometheus** for metrics collection
- **Grafana** for dashboards (admin/admin123)
- **Elasticsearch** for log aggregation
- **Kibana** for log visualization
- **Flower** for Celery monitoring

## 🚀 Quick Start

### Prerequisites
- Docker Desktop for Mac (4GB+ memory allocated)
- 16GB+ RAM (32GB recommended)
- macOS with Intel or Apple Silicon

### 1. Start the Environment
```bash
# Start everything
./start-dev-prod.sh

# Or just run the script without arguments (defaults to start)
./start-dev-prod.sh start
```

### 2. Access Your Services

**Main Application:**
- http://localhost - Load balanced application
- http://localhost/monitoring - Production monitoring dashboard

**Direct Service Access:**
- http://localhost:3001 - React Client 1
- http://localhost:3002 - React Client 2  
- http://localhost:8001 - Platform API 1
- http://localhost:8002 - Platform API 2
- http://localhost:8100 - Inference Engine

**Monitoring & Debugging:**
- http://localhost:9090 - Prometheus Metrics
- http://localhost:3000 - Grafana Dashboards
- http://localhost:5601 - Kibana Logs
- http://localhost:5555 - Celery Flower
- http://localhost:8080 - Debug proxy (access all services)

**Database Access:**
- localhost:5432 - PostgreSQL Primary
- localhost:5433 - PostgreSQL Replica
- localhost:6379 - Redis
- localhost:9200 - Elasticsearch

**Debug Ports:**
- localhost:5678 - Platform API 1 Debug
- localhost:5679 - Platform API 2 Debug
- localhost:5680 - Inference Engine Debug

## 🛠️ Management Commands

```bash
# Start the environment
./start-dev-prod.sh start

# Stop all services
./start-dev-prod.sh stop

# Restart everything
./start-dev-prod.sh restart

# View logs for all services
./start-dev-prod.sh logs

# View logs for specific service
./start-dev-prod.sh logs platform-api-1

# Check service status
./start-dev-prod.sh status

# Check service health
./start-dev-prod.sh health

# Clean up everything (removes volumes)
./start-dev-prod.sh cleanup

# Show help
./start-dev-prod.sh help
```

## 🔍 What You Can Test

### Load Balancing
- Make requests to http://localhost and see them distributed across API instances
- Stop one API instance and verify traffic continues flowing
- Monitor distribution in Grafana dashboards

### Database Replication
- Connect to primary (5432) and replica (5433) databases
- Write to primary, read from replica
- Test failover scenarios

### Monitoring & Alerting
- Generate load and watch metrics in Prometheus
- View real-time dashboards in Grafana
- Test alert thresholds and notifications

### WebSocket Scaling
- Open multiple browser tabs with WebSocket connections
- Monitor connection distribution across backend instances
- Test real-time features with multiple clients

### Deployment Simulation
- Use the production monitoring dashboard
- Trigger deployments through the API
- Test rollback scenarios

## 🐛 Debugging Features

### 1. Debug Ports
Each service exposes debug ports for IDE debugging:
```bash
# Attach debugger to Platform API 1
debugpy --listen 0.0.0.0:5678 --wait-for-client

# VS Code launch.json example
{
  "name": "Debug Platform API 1",
  "type": "python",
  "request": "attach",
  "host": "localhost",
  "port": 5678
}
```

### 2. Live Code Reloading
- Backend: Code changes trigger automatic reloads
- Frontend: React Fast Refresh for instant updates
- Models: Shared volume for model files

### 3. Log Aggregation
```bash
# View aggregated logs in Kibana
open http://localhost:5601

# Or use traditional docker logs
./start-dev-prod.sh logs platform-api-1
```

### 4. Direct Service Access
```bash
# Test individual services directly
curl http://localhost:8001/health  # API 1
curl http://localhost:8002/health  # API 2

# Or use the debug proxy
curl http://localhost:8080/api1/health
curl http://localhost:8080/api2/health
```

## 📊 Monitoring Your Development

### Grafana Dashboards
1. Open http://localhost:3000
2. Login: admin / admin123
3. Explore pre-configured dashboards

### Prometheus Metrics
1. Open http://localhost:9090
2. Query metrics like `http_requests_total`
3. Set up alerts for development scenarios

### Elasticsearch Logs
1. Open http://localhost:5601
2. Configure index patterns for `dreamcast-dev*`
3. Create visualizations and dashboards

## 🎯 Production Parity Features

### ✅ What Matches Production
- **Multi-service architecture** with load balancing
- **Database replication** and connection pooling
- **Monitoring stack** with metrics and logging
- **WebSocket handling** across multiple instances
- **Async task processing** with Celery
- **Container networking** and service discovery

### 🔧 Development Optimizations
- **Fewer replicas** (2 vs 3-4 in production)
- **Debug ports exposed** for IDE integration
- **Hot reloading enabled** for faster development
- **Relaxed security** for local development
- **Smaller resource limits** for laptop compatibility

## 🚨 Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check Docker resources
docker system df

# Increase Docker memory if needed (8GB+ recommended)
# Docker Desktop > Preferences > Resources > Memory
```

**Port conflicts:**
```bash
# Check what's using ports
lsof -i :8000
lsof -i :3000

# Stop conflicting services or change ports in docker-compose.dev-prod.yml
```

**Services unhealthy:**
```bash
# Check service logs
./start-dev-prod.sh logs [service-name]

# Check service status
./start-dev-prod.sh status

# Restart specific service
docker-compose -f docker-compose.dev-prod.yml restart [service-name]
```

### Performance Optimization

**For 16GB RAM systems:**
```bash
# Reduce Elasticsearch memory
# In docker-compose.dev-prod.yml:
ES_JAVA_OPTS=-Xms256m -Xmx256m
```

**For slower machines:**
```bash
# Disable some monitoring services
docker-compose -f docker-compose.dev-prod.yml up -d \
  --scale kibana=0 \
  --scale elasticsearch=0
```

## 🎮 Ready to Code!

Your production-like development environment is ready! You now have:

- ✅ **Console-quality reliability** in development
- ✅ **Production architecture** for realistic testing
- ✅ **Debug capabilities** for efficient development
- ✅ **Monitoring tools** to understand your application
- ✅ **Load balancing** to test multi-instance scenarios

Start building amazing AI character experiences with confidence! 🎭✨

---

*Questions? Check the logs, monitor the dashboards, and debug with confidence!* 