import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Create a simple test app with just our new routers
from app.routers import inference, evaluation, websocket

app = FastAPI()
app.include_router(inference.router, prefix="/api/v1")
app.include_router(evaluation.router, prefix="/api/v1")
app.include_router(websocket.router, prefix="/api/v1")

client = TestClient(app)


@pytest.mark.parametrize("url", [
    "/api/v1/inference/health",
    "/api/v1/evaluation/health",
    "/api/v1/websocket/health",
])
def test_unified_backend_health_endpoints(url):
    """Test that our new unified backend endpoints are reachable."""
    response = client.get(url)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "status" in data
    assert data["status"] == "healthy"


def test_inference_generate_endpoint():
    """Test the inference generation endpoint."""
    request_data = {
        "session_id": "test_session",
        "character_id": "test_character",
        "prompt": "Hello, how are you?",
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    response = client.post("/api/v1/inference/generate", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "generation_text" in data
    assert "inference_time_ms" in data
    assert data["session_id"] == "test_session"
    assert data["character_id"] == "test_character"


def test_inference_metrics_endpoint():
    """Test the inference metrics endpoint."""
    response = client.get("/api/v1/inference/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "requests_per_minute" in data
    assert "average_latency_ms" in data
    assert "queue_depth" in data


def test_inference_queue_status_endpoint():
    """Test the inference queue status endpoint."""
    response = client.get("/api/v1/inference/queue/status")
    assert response.status_code == 200
    data = response.json()
    assert "queue_depth" in data
    assert "processing" in data
    assert "completed_today" in data


def test_inference_cache_clear_endpoint():
    """Test the inference cache clear endpoint."""
    response = client.post("/api/v1/inference/cache/clear")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cache_cleared"
    assert "items_cleared" in data


def test_evaluation_personality_alignment_endpoint():
    """Test the personality alignment evaluation endpoint."""
    request_data = {
        "text": "I love exploring new ideas and concepts!",
        "target": {
            "openness": 0.8,
            "conscientiousness": 0.6,
            "extraversion": 0.7,
            "agreeableness": 0.5,
            "neuroticism": 0.3
        }
    }
    
    response = client.post("/api/v1/evaluation/personality_alignment", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert 0.0 <= data["score"] <= 1.0
    assert data["dev_mode"] is True


def test_evaluation_lore_adherence_endpoint():
    """Test the lore adherence evaluation endpoint."""
    request_data = {
        "text": "The magic system in this world requires verbal incantations.",
        "target": "Magic requires spoken words to work properly."
    }
    
    response = client.post("/api/v1/evaluation/lore_adherence", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert 0.0 <= data["score"] <= 1.0
    assert data["dev_mode"] is True


def test_websocket_health_endpoint():
    """Test the WebSocket service health endpoint."""
    response = client.get("/api/v1/websocket/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_connections" in data
    assert "active_sessions" in data
    assert data["service"] == "websocket" 