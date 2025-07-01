# Production Inference Engine

High-performance inference server for the Narrative-LLM with triple-head outputs, hot-swappable adapters, and mobile-first architecture.

## Overview

The Production Inference Engine is designed to serve character models at scale with optimized response times, supporting the unique requirements of persistent character interactions across mobile and web platforms.

### Key Features

- **Triple-Head Model Support**: Simultaneous generation of text, control tokens, and memory vectors
- **Hot-Swappable Adapters**: Change character models without server restart (<1s swap time)
- **Memory Integration**: Real-time memory formation and retrieval with vector search
- **Session Persistence**: Maintain character state across requests
- **Mobile Optimized**: WebSocket support, proactive agents, and efficient batching
- **Production Ready**: Health monitoring, auto-recovery, and horizontal scaling

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   Mobile Client     │     │    Web Client       │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           └─────────┬─────────────────┘
                     │ HTTPS/WebSocket
           ┌─────────▼─────────────────┐
           │   FastAPI Server          │
           │  ┌──────────────────┐    │
           │  │ Request Queue     │    │
           │  └────────┬─────────┘    │
           │           │              │
           │  ┌────────▼─────────┐    │
           │  │ Inference Engine │    │
           │  │ ┌─────────────┐ │    │
           │  │ │Triple-Head  │ │    │
           │  │ │   Model      │ │    │
           │  │ └─────────────┘ │    │
           │  └──────────────────┘    │
           └───────────┬──────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│  Adapter  │  │  Memory   │  │  Session  │
│  Manager  │  │  Service  │  │  Manager  │
└───────────┘  └───────────┘  └───────────┘
```

## Quick Start

### Starting the Server

```bash
# Basic start
python scripts/run_inference_server.py

# With vLLM optimization
python scripts/run_inference_server.py --enable-vllm

# With triple-head features
python scripts/run_inference_server.py --enable-triple-head

# Custom configuration
python scripts/run_inference_server.py \
  --port 8080 \
  --max-concurrent 20 \
  --gpu-memory-threshold 0.9
```

### Client Example

```python
import aiohttp
import asyncio

async def chat_with_character():
    async with aiohttp.ClientSession() as session:
        # Generate response
        async with session.post("http://localhost:8000/generate", json={
            "session_id": "mobile-123",
            "character_id": "alice",
            "prompt": "Hello! How are you today?",
            "max_tokens": 150
        }) as response:
            data = await response.json()
            
            print(f"Alice: {data['generation_text']}")
            print(f"Emotion: {data['control_tokens']}")
            print(f"Memory importance: {data['memory_metadata']['importance']}")

asyncio.run(chat_with_character())
```

## API Reference

### POST /generate

Generate a response from a character.

**Request:**
```json
{
  "session_id": "string",
  "character_id": "string", 
  "prompt": "string",
  "max_tokens": 150,
  "temperature": 0.8,
  "top_p": 0.9,
  "use_cache": true,
  "forced_control_tokens": ["<emotion_happy>"],
  "memory_context_ids": ["mem-123", "mem-456"]
}
```

**Response:**
```json
{
  "session_id": "string",
  "character_id": "string",
  "generation_text": "Hello! I'm doing wonderfully today!",
  "control_tokens": [
    {"token": "<emotion_happy>", "probability": 0.89}
  ],
  "memory_vector": [0.123, 0.456, ...], // 768 dimensions
  "memory_metadata": {
    "importance": 0.75,
    "emotional_valence": 0.8,
    "recency": 1.0,
    "coherence": 0.9
  },
  "inference_time_ms": 145.2,
  "tokens_generated": 12,
  "cache_hit": false
}
```

### GET /health

Check server health status.

**Response:**
```json
{
  "status": "healthy",
  "gpu_utilization": 0.65,
  "memory_usage_gb": 12.5,
  "active_requests": 3,
  "queued_requests": 0,
  "warnings": []
}
```

### GET /metrics

Get performance metrics.

**Response:**
```json
{
  "avg_inference_time_ms": 152.3,
  "p95_inference_time_ms": 198.7,
  "requests_per_minute": 234,
  "total_requests": 15234,
  "avg_queue_depth": 2.1
}
```

## Components

### Adapter Manager

Handles hot-swapping of character models:

```python
from app.inference_engine import AdapterManager

manager = AdapterManager()

# Load adapter
await manager.load_adapter("alice", "adapters/alice_v2.safetensors")

# Hot-swap to new version
await manager.hot_swap_adapter("alice", "adapters/alice_v3.safetensors")

# Rollback if needed
await manager.rollback_adapter("alice", "v2")
```

### Memory Service

Vector storage and retrieval for character memories:

```python
from app.inference_engine import MemoryService, MemoryVector

service = MemoryService()

# Store memory
memory = MemoryVector(
    session_id="session-123",
    character_id="alice",
    embedding=[0.1, 0.2, ...],  # 768 dims
    content="User said they love hiking",
    metadata={"importance": 0.8}
)
await service.store_memory(memory)

# Search memories
results = await service.search_memories(
    query_embedding=[0.15, 0.25, ...],
    session_id="session-123",
    top_k=5
)
```

### Session Manager

Persistent session state across requests:

```python
from app.inference_engine import SessionStateManager

manager = SessionStateManager()

# Create session
session_id = await manager.create_session(
    user_id="user-123",
    character_ids=["alice", "bob"],
    world_id="fantasy-world"
)

# Update character state
await manager.set_character_state(session_id, "alice", {
    "mood": "excited",
    "location": "garden",
    "talking_to": ["bob"]
})

# Get scene state
scene = await manager.get_scene_state(session_id)
```

## Mobile Integration

The engine is designed with mobile-first architecture:

### WebSocket Support

For real-time character interactions:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.on('message', (data) => {
  const event = JSON.parse(data);
  if (event.type === 'proactive_message') {
    showNotification(event.character, event.message);
  }
});
```

### Proactive Agents

Characters can initiate conversations:

```python
# Server-side proactive check
if time_since_last_interaction > 3600:  # 1 hour
    response = await engine.generate({
        "session_id": session_id,
        "character_id": "companion",
        "prompt": "",  # Empty - character initiates
        "forced_control_tokens": ["<proactive_greeting>"]
    })
    
    # Send push notification
    await send_push_notification(user_id, response)
```

### Efficient Batching

Optimized for mobile network conditions:

- Request queuing with priorities
- Attention cache for repeated contexts
- Compressed response formats
- Progressive response streaming

## Performance Tuning

### GPU Memory Management

```python
# Configure memory thresholds
config = {
    "gpu_memory_threshold": 0.85,  # Trigger cleanup at 85%
    "model_cache_size": 3,         # Max models in memory
    "cache_ttl": 300               # 5-minute attention cache
}
```

### Concurrency Settings

```python
# Optimize for your hardware
config = {
    "max_concurrent": 10,      # RTX 4090: 10-15
    "queue_max_size": 1000,    # Request queue depth
    "batch_timeout": 0.1       # 100ms batching window
}
```

### vLLM Integration

When available, enable vLLM for better performance:

```bash
python scripts/run_inference_server.py --enable-vllm
```

Benefits:
- Continuous batching
- PagedAttention for memory efficiency
- 3-5x throughput improvement

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/inference_engine ./app/inference_engine
COPY scripts/run_inference_server.py ./

EXPOSE 8000

CMD ["python", "run_inference_server.py", "--enable-vllm"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-engine
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: inference
        image: narrative-llm/inference:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
        env:
        - name: MAX_CONCURRENT
          value: "10"
        - name: ENABLE_VLLM
          value: "true"
```

### Load Balancing

Use a reverse proxy for horizontal scaling:

```nginx
upstream inference_backend {
    least_conn;
    server inference1:8000 max_fails=3 fail_timeout=30s;
    server inference2:8000 max_fails=3 fail_timeout=30s;
    server inference3:8000 max_fails=3 fail_timeout=30s;
}

server {
    location /generate {
        proxy_pass http://inference_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

## Monitoring

### Prometheus Metrics

The engine exposes metrics at `/metrics`:

```
# HELP inference_request_duration_seconds Request duration
# TYPE inference_request_duration_seconds histogram
inference_request_duration_seconds_bucket{le="0.1"} 234
inference_request_duration_seconds_bucket{le="0.2"} 456
inference_request_duration_seconds_bucket{le="0.5"} 567

# HELP gpu_memory_usage_bytes GPU memory usage
# TYPE gpu_memory_usage_bytes gauge
gpu_memory_usage_bytes 12884901888
```

### Logging

Structured logging with context:

```python
logger.info("Inference completed", extra={
    "session_id": session_id,
    "character_id": character_id,
    "inference_time_ms": 145.2,
    "tokens_generated": 150,
    "cache_hit": True
})
```

## Troubleshooting

### High Latency

1. Check GPU utilization: `GET /health`
2. Review queue depth: `GET /queue/status`
3. Clear attention cache: `POST /cache/clear`
4. Reduce concurrent requests or add more GPUs

### Memory Issues

1. Monitor with: `nvidia-smi -l 1`
2. Reduce `model_cache_size`
3. Lower `gpu_memory_threshold`
4. Enable automatic cleanup in config

### Adapter Loading Failures

1. Verify adapter compatibility with base model
2. Check file permissions
3. Ensure sufficient GPU memory
4. Review adapter metadata JSON

## Future Enhancements

- **Streaming Responses**: Server-sent events for progressive generation
- **Multi-GPU Support**: Distributed inference across multiple GPUs
- **Quantization**: INT8/INT4 quantization for mobile models
- **Edge Deployment**: ONNX export for on-device inference
- **Advanced Caching**: Redis-based distributed cache

## License

See the main project LICENSE file. 