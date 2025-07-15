# 🚀 Production Deployment Demo

This guide shows you how to experience the enterprise-grade production infrastructure of the Character Creation Devkit.

## 🎯 What You'll Experience

After completing this demo, you'll have seen:
- **Real-time Production Monitoring** with console-quality dashboard
- **Auto-scaling in Action** as the system responds to load
- **A/B Testing Framework** for platform optimization
- **Zero-downtime Deployment** with automatic rollback
- **Enterprise Reliability** with comprehensive health monitoring

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose installed
- Node.js 18+ for the React client
- Python 3.11+ for the backend
- At least 8GB RAM for full production stack

### 1. Launch the Production Environment

```bash
# Clone and navigate to the project
git clone [repository-url]
cd smollmfinetune

# Start the full production stack
docker-compose -f docker-compose.prod.yml up -d

# Wait for all services to be healthy (about 2-3 minutes)
docker-compose -f docker-compose.prod.yml ps
```

**Expected Output:**
```
NAME                    COMMAND                  SERVICE             STATUS
platform-api-1         "uvicorn app.main:ap…"   platform-api       Up (healthy)
platform-api-2         "uvicorn app.main:ap…"   platform-api       Up (healthy)  
platform-api-3         "uvicorn app.main:ap…"   platform-api       Up (healthy)
platform-client-1      "nginx -g 'daemon of…"   platform-client    Up (healthy)
platform-client-2      "nginx -g 'daemon of…"   platform-client    Up (healthy)
db-primary             "docker-entrypoint.s…"   db-cluster         Up (healthy)
db-replica-1           "docker-entrypoint.s…"   db-cluster         Up (healthy)
redis-cluster-1        "redis-server --clus…"   redis-cluster      Up (healthy)
nginx-lb               "nginx -g 'daemon of…"   nginx-lb           Up (healthy)
```

### 2. Access the Production Monitoring Dashboard

```bash
# Open the production monitoring dashboard
open http://localhost:3000/monitoring

# Or manually navigate to:
# http://localhost:3000/monitoring
```

## 📊 Exploring the Monitoring Dashboard

### Real-time Platform Status

You'll see a **console-quality dashboard** with:

1. **Service Health Grid**: Color-coded status of all platform components
   - 🟢 Green: Service healthy and responsive
   - 🟡 Yellow: Service experiencing issues  
   - 🔴 Red: Service down or critical error

2. **Performance Charts**: Live updating graphs showing:
   - API response times (should be < 200ms)
   - Memory and CPU utilization
   - Active WebSocket connections
   - Voice generation metrics

3. **User Analytics Panel**: Real-time user behavior:
   - Active users on the platform
   - Character creation and interaction rates
   - Feature adoption metrics
   - Journey completion rates

### Testing Monitoring Alerts

1. **Simulate High Load**:
```bash
# Generate API load to trigger scaling
for i in {1..100}; do
  curl -X GET http://localhost:8000/health &
done
```

2. **Watch Auto-scaling**:
   - Monitor the dashboard for CPU usage spikes
   - Watch auto-scaling notifications appear
   - See new service instances being created

3. **Trigger Alert**:
```bash
# Simulate a service issue to see alerting
docker-compose -f docker-compose.prod.yml stop platform-api-1

# Watch the monitoring dashboard show:
# - Service status change to yellow/red
# - Alert notifications appear
# - Auto-healing attempts
```

4. **Recovery**:
```bash
# Restart the service to see recovery
docker-compose -f docker-compose.prod.yml start platform-api-1

# Watch the dashboard show green status return
```

## 🧪 A/B Testing Framework Demo

### 1. Create an Experiment

```bash
# Use the API to create an A/B test
curl -X POST http://localhost:8000/api/v1/ab-testing/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ui_layout_test",
    "name": "Character Builder Layout Test",
    "description": "Testing different character creation UI layouts",
    "hypothesis": "New layout will improve character creation completion rates",
    "variants": ["control", "new_layout"],
    "traffic_split": {"control": 50, "new_layout": 50},
    "success_metrics": ["creation_completion_rate", "time_to_complete"]
  }'
```

### 2. Test User Assignment

```bash
# Test consistent user assignment
for user_id in user1 user2 user3 user1 user2 user3; do
  echo "User: $user_id"
  curl -X GET "http://localhost:8000/api/v1/ab-testing/assignment/$user_id/ui_layout_test"
  echo
done
```

**Expected Output**: Same users get assigned to the same variant consistently.

### 3. Monitor Experiment

- Visit the monitoring dashboard
- Navigate to "A/B Testing" section
- See experiment status, traffic distribution, and early results

## 🔄 Zero-downtime Deployment Demo

### 1. Trigger a Rolling Deployment

```bash
# Simulate a new version deployment
curl -X POST http://localhost:8000/api/v1/deployment/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "service_name": "platform-api",
    "image": "dreamcast-platform-api",
    "tag": "v1.1.0",
    "strategy": "rolling_update",
    "replicas": 3
  }'
```

### 2. Watch the Deployment Process

In the monitoring dashboard:
1. **Deployment Tracker** shows progress
2. **Service Health** remains green throughout
3. **Performance Metrics** show no interruption
4. **Deployment Timeline** shows each step

### 3. Monitor Service Continuity

```bash
# Continuously test API during deployment
while true; do
  response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
  echo "$(date): HTTP $response"
  sleep 1
done
```

**Expected**: You should see **zero 5xx errors** during the entire deployment.

### 4. Test Automatic Rollback

```bash
# Simulate a failed deployment
curl -X POST http://localhost:8000/api/v1/deployment/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "service_name": "platform-api",
    "image": "dreamcast-platform-api",
    "tag": "broken-version",
    "strategy": "rolling_update"
  }'
```

Watch the monitoring dashboard show:
1. Deployment attempt
2. Health check failures
3. **Automatic rollback** triggered
4. Service restored to previous version

## 🎮 Multi-Character Conversation with Production Monitoring

### 1. Start a Multi-Character Conversation

```bash
# Launch the React client
./launch-client.sh

# Navigate to Multi-Character Audio Mixer
# http://localhost:3000/multi-character-mixer
```

### 2. Monitor Voice Generation Performance

In the production dashboard:
1. **Voice Quality Metrics**:
   - Generation latency (should be < 500ms)
   - Quality scores from voice evaluation
   - Character consistency metrics

2. **WebSocket Performance**:
   - Active connections
   - Message throughput
   - Real-time audio streaming metrics

3. **Resource Utilization**:
   - GPU usage for voice generation
   - Memory usage for model inference
   - Network bandwidth for audio streaming

### 3. Test Load Balancing

```bash
# Create multiple WebSocket connections
for i in {1..10}; do
  # Start multiple conversation sessions
  # (Use the React client in multiple browser tabs)
  echo "Session $i started"
done
```

Watch the monitoring dashboard show:
- **Even distribution** across backend instances
- **Stable performance** under load
- **Auto-scaling** if thresholds are exceeded

## 📈 Performance Benchmarking

### 1. API Performance Test

```bash
# Install hey for load testing (if not already installed)
# macOS: brew install hey
# Linux: wget https://github.com/rakyll/hey/releases/download/v0.1.4/hey_linux_amd64

# Test API performance
hey -n 1000 -c 10 -q 10 http://localhost:8000/health

# Expected results:
# - Average response time: < 100ms
# - 99th percentile: < 500ms
# - Zero errors under normal load
```

### 2. WebSocket Performance Test

```bash
# Test WebSocket scaling
node -e "
const WebSocket = require('ws');
const connections = [];

for (let i = 0; i < 100; i++) {
  const ws = new WebSocket('ws://localhost:8000/ws/voice');
  connections.push(ws);
}

console.log('Created 100 WebSocket connections');
setTimeout(() => {
  connections.forEach(ws => ws.close());
  console.log('Closed all connections');
}, 10000);
"
```

### 3. Database Performance Test

```bash
# Test database query performance
curl -X GET "http://localhost:8000/api/v1/monitoring/database/performance"

# Expected metrics:
# - Query time: < 50ms average
# - Connection pool: Healthy
# - Cache hit ratio: > 90%
```

## 🔒 Security & Compliance Demo

### 1. Test Rate Limiting

```bash
# Test API rate limiting (should get 429 errors)
for i in {1..150}; do
  curl -w "%{http_code}\n" -s -o /dev/null http://localhost:8000/health
done | tail -20

# Expected: See 429 (Too Many Requests) after ~100 requests
```

### 2. Test SSL/HTTPS (if configured)

```bash
# Test SSL certificate (if you have SSL configured)
curl -I https://your-domain.com/health

# Expected: Valid SSL certificate response
```

### 3. Test Data Export

```bash
# Test data export functionality
curl -X GET "http://localhost:8000/api/v1/export/user-data" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Expected: Complete data export in JSON format
```

## 🎯 Success Criteria

After completing this demo, you should have verified:

### ✅ Infrastructure Reliability
- [ ] All services start successfully and show healthy status
- [ ] Monitoring dashboard displays real-time metrics
- [ ] Auto-scaling responds to load changes
- [ ] Alerting system detects and reports issues

### ✅ Deployment Capabilities  
- [ ] Zero-downtime rolling deployments work correctly
- [ ] Automatic rollback triggers on deployment failures
- [ ] Service continuity maintained during updates
- [ ] Deployment tracking shows detailed progress

### ✅ Performance Standards
- [ ] API response times consistently under 200ms
- [ ] Voice generation latency under 500ms
- [ ] WebSocket connections handle concurrent users
- [ ] Database queries perform within SLA thresholds

### ✅ Security & Monitoring
- [ ] Rate limiting prevents abuse
- [ ] Real-time monitoring captures all metrics
- [ ] A/B testing framework manages experiments
- [ ] Data export functionality works correctly

## 🚀 Next Steps

Now that you've experienced the production infrastructure:

1. **Deploy to Your Own Environment**: Use the production configuration for your deployment
2. **Customize Monitoring**: Adapt alerts and metrics for your specific needs  
3. **Scale for Your Load**: Configure auto-scaling thresholds for your expected traffic
4. **Integrate with Your Systems**: Connect monitoring to your existing DevOps tools

## 📞 Need Help?

If you encounter any issues during the demo:

1. **Check Service Logs**:
```bash
docker-compose -f docker-compose.prod.yml logs [service-name]
```

2. **Monitor Resource Usage**:
```bash
docker stats
```

3. **Verify Network Connectivity**:
```bash
curl -v http://localhost:8000/health
```

4. **Review Monitoring Dashboard**: The dashboard often provides insights into what's causing issues.

The Character Creation Devkit is now **production-ready** with enterprise-grade infrastructure! 🎉

---

*Ready to deploy your own AI character platform? The infrastructure is battle-tested and waiting for you!* 