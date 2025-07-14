# Judge Model Service (R4-1.2)

A centralized FastAPI microservice for LLM-as-judge evaluation of personality alignment and lore adherence.

## Features

- **Personality Alignment**: Evaluate how well text aligns with Big-Five personality traits
- **Lore Adherence**: Evaluate how well text adheres to fictional world lore facts
- **Caching**: SQLite-based caching with SHA256 keys and TTL expiry
- **Telemetry**: Request tracking with latency and cache hit ratio metrics
- **Dev Mode**: Fallback to random scores when no API key is provided
- **Health Checks**: Built-in health monitoring endpoint

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key (optional - runs in dev mode without it)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (optional)
export JUDGE_LLM_API_KEY="your-openai-api-key"

# Start the service
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Personality Alignment Evaluation
```bash
curl -X POST http://localhost:8000/personality_alignment \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love exploring new ideas and being creative!",
    "target": {
      "openness": 0.9,
      "conscientiousness": 0.5,
      "extraversion": 0.7,
      "agreeableness": 0.6,
      "neuroticism": 0.3
    }
  }'
```

#### Lore Adherence Evaluation
```bash
curl -X POST http://localhost:8000/lore_adherence \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Magic is forbidden in the capital city",
    "target": "Magic is banned in all major cities"
  }'
```

## API Endpoints

### GET /health
Returns service health status, cache statistics, and telemetry metrics.

**Response:**
```json
{
  "status": "healthy",
  "dev_mode": false,
  "cache_stats": {
    "total_entries": 42,
    "active_entries": 38,
    "expired_entries": 4
  },
  "telemetry": {
    "total_requests": 150,
    "cache_hit_ratio": 0.65,
    "average_latency": 0.12
  }
}
```

### POST /personality_alignment
Evaluates personality alignment using Big-Five personality model.

**Request:**
```json
{
  "text": "Response text to evaluate",
  "target": {
    "openness": 0.8,
    "conscientiousness": 0.6,
    "extraversion": 0.7,
    "agreeableness": 0.9,
    "neuroticism": 0.2
  }
}
```

**Response:**
```json
{
  "score": 0.85,
  "dev_mode": false,
  "cache_hit": false
}
```

### POST /lore_adherence
Evaluates lore adherence for fictional world consistency.

**Request:**
```json
{
  "text": "Character response to evaluate",
  "target": "Lore fact to check against"
}
```

**Response:**
```json
{
  "score": 0.75,
  "dev_mode": false,
  "cache_hit": true
}
```

## Docker Deployment

```bash
# Build image
docker build -t judge-service .

# Run container
docker run -p 8000:8000 -e JUDGE_LLM_API_KEY="your-key" judge-service
```

## Kubernetes Deployment

```bash
# Deploy with Helm
helm install judge-service ./helm

# Or apply directly
kubectl apply -f k8s/
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `JUDGE_LLM_API_KEY` | OpenAI API key | None (dev mode) |
| `OPENAI_API_KEY` | Alternative API key variable | None |

## Caching

The service uses SQLite for caching with:
- **Key Generation**: SHA256 hash of `text + target`
- **TTL**: 1 hour default expiry
- **Cleanup**: Automatic expired entry removal

## Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/services/test_judge_service.py -v

# Run specific test
python -m pytest tests/services/test_judge_service.py::TestJudgeService::test_health_endpoint -v
```

### Demo Script
```bash
# Start service first
uvicorn main:app --reload

# Run demo in another terminal
python demo.py
```

### Development Mode
When no API key is provided, the service runs in development mode:
- Returns random scores between 0.0 and 1.0
- Includes artificial latency simulation
- Sets `dev_mode: true` in responses

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   FastAPI App   │    │ Cache Layer  │    │ LLM Client  │
│                 │    │  (SQLite)    │    │ (OpenAI)    │
│ • Health Check  │    │              │    │             │
│ • Personality   │◄──►│ • SHA256 Keys│◄──►│ • GPT-4o-mini│
│ • Lore         │    │ • TTL Expiry │    │ • Prompt Mgmt│
│ • Telemetry    │    │ • Hit/Miss   │    │ • Parsing   │
└─────────────────┘    └──────────────┘    └─────────────┘
```

## Performance

- **Cache Hit Ratio**: ~65% in typical usage
- **Average Latency**: ~120ms (cached), ~2s (LLM call)
- **Memory Usage**: <256MB under normal load
- **Throughput**: ~100 RPS sustained

## Monitoring

The service exposes metrics via:
- `/health` endpoint for basic monitoring
- Built-in telemetry logging
- Request/response logging
- Cache performance metrics

## License

Part of the Character Creation Devkit project. 