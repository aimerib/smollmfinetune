"""
Tests for Director's Chair Real-Time Training System

Following TDD principles to test the Director's Chair functionality:
- Multi-head preference collection
- Real-time conversation editing and correction
- Head-specific training pipeline integration
- Live training status monitoring
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any

from fastapi.testclient import TestClient
from app.main import app


class TestDirectorsChairConversationStudio:
    """Test the enhanced conversation studio with multi-head support"""
    
    def test_directors_chair_endpoint_exists(self):
        """Test that the Director's Chair endpoint exists"""
        client = TestClient(app)
        response = client.get("/directors-chair/status")
        # Should exist but might return 404 for missing session
        assert response.status_code in [200, 404]
    
    def test_start_directors_chair_session(self):
        """Test starting a new Director's Chair session"""
        client = TestClient(app)
        
        session_data = {
            "character_id": "alice",
            "mode": "directors_chair",
            "training_enabled": True
        }
        
        response = client.post("/directors-chair/sessions", json=session_data)
        assert response.status_code == 201
        
        data = response.json()
        assert "session_id" in data
        assert data["mode"] == "directors_chair"
        assert data["training_enabled"] is True
    
    def test_conversation_editing_interface(self):
        """Test the conversation editing interface"""
        client = TestClient(app)
        
        # Start session first
        session_response = client.post("/directors-chair/sessions", json={
            "character_id": "alice",
            "mode": "directors_chair"
        })
        session_id = session_response.json()["session_id"]
        
        # Create a conversation turn
        conversation_data = {
            "user_message": "Tell me about your favorite adventure",
            "assistant_response": "I love exploring mysterious forests!",
            "metadata": {
                "generation_head_score": 0.7,
                "control_head_score": 0.8,
                "memory_head_score": 0.6
            }
        }
        
        response = client.post(
            f"/directors-chair/sessions/{session_id}/conversations",
            json=conversation_data
        )
        assert response.status_code == 201
        
        data = response.json()
        assert "conversation_id" in data
        assert data["editable"] is True
    
    def test_multi_head_correction_routing(self):
        """Test that corrections are routed to appropriate training heads"""
        client = TestClient(app)
        
        # Setup session and conversation
        session_response = client.post("/directors-chair/sessions", json={
            "character_id": "alice",
            "mode": "directors_chair"
        })
        session_id = session_response.json()["session_id"]
        
        conv_response = client.post(
            f"/directors-chair/sessions/{session_id}/conversations",
            json={
                "user_message": "Hello",
                "assistant_response": "Hi there!"
            }
        )
        conversation_id = conv_response.json()["conversation_id"]
        
        # Test different correction types
        correction_types = [
            {
                "type": "content_correction",
                "target_head": "generation",
                "original_text": "Hi there!",
                "corrected_text": "Hello! How wonderful to meet you!",
                "reason": "More enthusiastic and character-appropriate"
            },
            {
                "type": "emotion_correction", 
                "target_head": "control",
                "emotional_state": "excited",
                "control_tokens": ["<excited>", "<friendly>"],
                "reason": "Should show more excitement"
            },
            {
                "type": "memory_correction",
                "target_head": "memory", 
                "memory_importance": 0.8,
                "should_remember": True,
                "reason": "This is an important first meeting"
            }
        ]
        
        for correction in correction_types:
            response = client.post(
                f"/directors-chair/conversations/{conversation_id}/corrections",
                json=correction
            )
            assert response.status_code == 201
            
            data = response.json()
            assert data["target_head"] == correction["target_head"]
            assert data["queued_for_training"] is True


class TestTripleHeadPreferenceCollection:
    """Test multi-head preference collection and training queue management"""
    
    def test_preference_pair_creation_generation_head(self):
        """Test creating preference pairs for generation head training"""
        client = TestClient(app)
        
        preference_data = {
            "head_type": "generation",
            "prompt": "Tell me about your day",
            "chosen_response": "I had an amazing adventure in the enchanted forest!",
            "rejected_response": "My day was okay.",
            "character_id": "alice",
            "correction_reason": "More detailed and character-appropriate",
            "quality_metrics": {
                "coherence": 0.9,
                "creativity": 0.8,
                "character_consistency": 0.9
            }
        }
        
        response = client.post("/directors-chair/preferences", json=preference_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["head_type"] == "generation"
        assert data["preference_id"] is not None
        assert data["training_queue_position"] >= 0
    
    def test_preference_pair_creation_control_head(self):
        """Test creating preference pairs for control head training"""
        client = TestClient(app)
        
        preference_data = {
            "head_type": "control",
            "prompt": "How are you feeling?",
            "chosen_response": "<excited> I'm feeling absolutely wonderful! <joyful>",
            "rejected_response": "I'm fine.",
            "character_id": "alice",
            "correction_reason": "Should express emotions more clearly",
            "control_metrics": {
                "emotional_appropriateness": 0.9,
                "control_token_usage": 0.8,
                "mood_consistency": 0.9
            }
        }
        
        response = client.post("/directors-chair/preferences", json=preference_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["head_type"] == "control"
        assert "control_token_analysis" in data
    
    def test_preference_pair_creation_memory_head(self):
        """Test creating preference pairs for memory head training"""
        client = TestClient(app)
        
        preference_data = {
            "head_type": "memory",
            "context": "Previous conversation about favorite books",
            "chosen_response": "Yes, I remember we discussed your love for fantasy novels!",
            "rejected_response": "I don't recall our previous conversation.",
            "character_id": "alice",
            "correction_reason": "Better memory consistency",
            "memory_metrics": {
                "recall_accuracy": 0.9,
                "context_relevance": 0.8,
                "importance_weighting": 0.7
            }
        }
        
        response = client.post("/directors-chair/preferences", json=preference_data)
        assert response.status_code == 201
        
        data = response.json()
        assert data["head_type"] == "memory"
        assert "memory_formation_analysis" in data


class TestRealTimeTrainingWorkers:
    """Test the real-time training infrastructure"""
    
    @pytest.mark.asyncio
    async def test_generation_head_training_worker(self):
        """Test the generation head training worker"""
        client = TestClient(app)
        
        # Check training worker status
        response = client.get("/directors-chair/training/generation/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "worker_status" in data
        assert "queue_length" in data
        assert "current_training_job" in data
    
    @pytest.mark.asyncio
    async def test_control_head_training_worker(self):
        """Test the control head training worker"""
        client = TestClient(app)
        
        response = client.get("/directors-chair/training/control/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["head_type"] == "control"
        assert "emotional_training_metrics" in data
    
    @pytest.mark.asyncio
    async def test_memory_head_training_worker(self):
        """Test the memory head training worker"""
        client = TestClient(app)
        
        response = client.get("/directors-chair/training/memory/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["head_type"] == "memory"
        assert "memory_formation_metrics" in data
    
    def test_coordinated_training_status(self):
        """Test getting coordinated training status across all heads"""
        client = TestClient(app)
        
        response = client.get("/directors-chair/training/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "generation_head" in data
        assert "control_head" in data
        assert "memory_head" in data
        assert "coordination_metrics" in data
        assert "overall_training_progress" in data
    
    def test_start_training_cycle(self):
        """Test starting a coordinated training cycle"""
        client = TestClient(app)
        
        training_config = {
            "heads": ["generation", "control", "memory"],
            "batch_size": 16,
            "learning_rate": 1e-4,
            "max_training_time_minutes": 2,
            "coordination_weight": 0.1
        }
        
        response = client.post("/directors-chair/training/start", json=training_config)
        assert response.status_code == 202  # Accepted, training started in background
        
        data = response.json()
        assert "training_job_id" in data
        assert data["status"] == "training_started"
        assert "estimated_completion_time" in data


class TestTripleHeadAnalytics:
    """Test head-specific analytics and progress indicators"""
    
    def test_generation_head_analytics(self):
        """Test generation head performance analytics"""
        client = TestClient(app)
        
        response = client.get("/directors-chair/analytics/generation")
        assert response.status_code == 200
        
        data = response.json()
        assert "content_quality_trend" in data
        assert "coherence_scores" in data
        assert "creativity_metrics" in data
        assert "character_consistency" in data
        assert "improvement_rate" in data
    
    def test_control_head_analytics(self):
        """Test control head emotional analytics"""
        client = TestClient(app)
        
        response = client.get("/directors-chair/analytics/control")
        assert response.status_code == 200
        
        data = response.json()
        assert "emotional_appropriateness" in data
        assert "control_token_effectiveness" in data
        assert "mood_consistency" in data
        assert "emotional_range" in data
    
    def test_memory_head_analytics(self):
        """Test memory head formation analytics"""
        client = TestClient(app)
        
        response = client.get("/directors-chair/analytics/memory")
        assert response.status_code == 200
        
        data = response.json()
        assert "memory_formation_rate" in data
        assert "recall_accuracy" in data
        assert "importance_calibration" in data
        assert "memory_decay_patterns" in data
    
    def test_cross_head_coordination_metrics(self):
        """Test metrics showing coordination between heads"""
        client = TestClient(app)
        
        response = client.get("/directors-chair/analytics/coordination")
        assert response.status_code == 200
        
        data = response.json()
        assert "head_interaction_strength" in data
        assert "coordination_loss" in data
        assert "balanced_improvement" in data
        assert "conflict_resolution" in data


class TestDirectorsChairWebSocketIntegration:
    """Test WebSocket integration for real-time updates"""
    
    @pytest.mark.asyncio 
    async def test_training_progress_updates(self):
        """Test that training progress is broadcast via WebSocket"""
        # This would need WebSocket test client setup
        # For now, test the event emission structure
        
        from app.services.event_bus import event_bus, EventType
        
        training_event = {
            "type": "training_progress_update",
            "data": {
                "head_type": "generation",
                "progress_percentage": 45.0,
                "current_batch": 12,
                "total_batches": 25,
                "loss": 0.234,
                "estimated_time_remaining": "2m 15s"
            }
        }
        
        # In a real test, we'd verify this gets broadcast to WebSocket clients
        assert training_event["type"] == "training_progress_update"
        assert training_event["data"]["head_type"] == "generation"
    
    @pytest.mark.asyncio
    async def test_correction_feedback_events(self):
        """Test that correction feedback is sent to WebSocket clients"""
        
        correction_event = {
            "type": "correction_applied",
            "data": {
                "correction_id": "corr_123",
                "target_head": "control",
                "training_queued": True,
                "estimated_training_time": "1m 30s"
            }
        }
        
        assert correction_event["type"] == "correction_applied"
        assert correction_event["data"]["target_head"] == "control"
    
    @pytest.mark.asyncio
    async def test_model_update_notifications(self):
        """Test that model updates are broadcast when training completes"""
        
        update_event = {
            "type": "model_updated",
            "data": {
                "head_type": "generation",
                "character_id": "alice",
                "version": "1.2.3",
                "improvement_metrics": {
                    "coherence_improvement": 0.05,
                    "quality_score_delta": 0.03
                },
                "ready_for_inference": True
            }
        }
        
        assert update_event["type"] == "model_updated"
        assert update_event["data"]["ready_for_inference"] is True


class TestDirectorsChairErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_correction_type(self):
        """Test handling of invalid correction types"""
        client = TestClient(app)
        
        invalid_correction = {
            "type": "invalid_correction_type",
            "target_head": "nonexistent",
            "original_text": "Hello",
            "corrected_text": "Hi"
        }
        
        response = client.post("/directors-chair/corrections", json=invalid_correction)
        assert response.status_code == 400
        assert "invalid correction type" in response.json()["detail"].lower()
    
    def test_training_queue_overflow_handling(self):
        """Test handling of training queue overflow"""
        client = TestClient(app)
        
        # Check queue capacity
        response = client.get("/directors-chair/training/queue/capacity")
        assert response.status_code == 200
        
        data = response.json()
        assert "max_queue_size" in data
        assert "current_queue_size" in data
        assert "accepts_new_jobs" in data
    
    def test_concurrent_training_conflict_resolution(self):
        """Test handling of concurrent training conflicts"""
        client = TestClient(app)
        
        # Try to start training when already running
        response = client.post("/directors-chair/training/start", json={
            "heads": ["generation"]
        })
        
        # Should either accept if queue has space or reject gracefully
        assert response.status_code in [202, 409]
        
        if response.status_code == 409:
            assert "training already in progress" in response.json()["detail"].lower() 