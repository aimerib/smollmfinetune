import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.parametrize("url", [
    "/api/v1/inference/health",
    "/api/v1/evaluation/health",
    "/api/v1/websocket/health",
])
def test_health_endpoints(url):
    """Basic sanity checks that the new unified backend endpoints are reachable."""
    response = client.get(url)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "status" in data
    assert data["status"] in {"healthy", "Healthy", "Healthy", "healthy"}


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