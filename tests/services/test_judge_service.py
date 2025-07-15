"""
Tests for Judge Model Service (R4-1.2)

Tests the centralized LLM-as-judge microservice including:
- FastAPI endpoints for personality alignment and lore adherence
- SQLite caching layer with SHA256 keys
- Health check endpoint
- Real LLM integration testing
"""

import unittest
import pytest
import json
import hashlib
import sqlite3
import tempfile
import os
from fastapi.testclient import TestClient
import httpx


class TestJudgeService(unittest.TestCase):
    """Test the Judge Model microservice"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Test data
        self.test_response = "I love exploring new ideas and thinking creatively about problems!"
        self.test_big_five = {
            "openness": 0.9,
            "conscientiousness": 0.5,
            "extraversion": 0.7,
            "agreeableness": 0.6,
            "neuroticism": 0.3
        }
        self.test_lore_fact = "Magic is forbidden in the capital city"
        
        # This will fail initially since the service doesn't exist yet
        from backend.app.services.judge_service.main import app
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
    
    def test_personality_alignment_endpoint(self):
        """Test the personality alignment endpoint"""
        # Test basic endpoint functionality - GREEN phase should be minimal
        response = self.client.post("/personality_alignment", json={
            "text": self.test_response,
            "target": self.test_big_five
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("score", data)
        self.assertIsInstance(data["score"], (int, float))
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)
        # In dev mode, should return dev_mode flag
        self.assertIn("dev_mode", data)
    
    def test_lore_adherence_endpoint(self):
        """Test the lore adherence endpoint"""
        # Test basic endpoint functionality - GREEN phase should be minimal
        response = self.client.post("/lore_adherence", json={
            "text": self.test_response,
            "target": self.test_lore_fact
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("score", data)
        self.assertIsInstance(data["score"], (int, float))
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)
        # In dev mode, should return dev_mode flag
        self.assertIn("dev_mode", data)
    
    def test_cache_functionality(self):
        """Test that identical requests are cached"""
        from backend.app.services.judge_service.cache import CacheManager
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_db:
            cache = CacheManager(temp_db.name)
            
            # First call should miss cache
            cache_key = cache._generate_cache_key(self.test_response, str(self.test_big_five))
            result = cache.get(cache_key)
            self.assertIsNone(result)
            
            # Store result in cache
            cache.set(cache_key, {"score": 0.85})
            
            # Second call should hit cache
            cached_result = cache.get(cache_key)
            self.assertIsNotNone(cached_result)
            self.assertEqual(cached_result["score"], 0.85)
        
        # Clean up
        os.unlink(temp_db.name)
    
    def test_cache_key_generation(self):
        """Test SHA256 cache key generation"""
        from backend.app.services.judge_service.cache import CacheManager
        
        cache = CacheManager()
        key1 = cache._generate_cache_key("text1", "target1")
        key2 = cache._generate_cache_key("text1", "target1")  # Same inputs
        key3 = cache._generate_cache_key("text2", "target1")  # Different inputs
        
        # Same inputs should generate same key
        self.assertEqual(key1, key2)
        # Different inputs should generate different keys
        self.assertNotEqual(key1, key3)
        # Keys should be SHA256 hashes (64 characters)
        self.assertEqual(len(key1), 64)
    
    def test_cache_expiry(self):
        """Test cache expiry functionality"""
        from backend.app.services.judge_service.cache import CacheManager
        import time
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_db:
            cache = CacheManager(temp_db.name, ttl_seconds=1)  # 1 second TTL
            
            cache_key = cache._generate_cache_key("test", "target")
            cache.set(cache_key, {"score": 0.5})
            
            # Should be available immediately
            result = cache.get(cache_key)
            self.assertIsNotNone(result)
            
            # Wait for expiry
            time.sleep(1.5)
            
            # Should be expired now
            result = cache.get(cache_key)
            self.assertIsNone(result)
        
        os.unlink(temp_db.name)
    
    def test_invalid_request_format(self):
        """Test handling of invalid request formats"""
        # Missing required fields
        response = self.client.post("/personality_alignment", json={
            "text": self.test_response
            # Missing "target" field
        })
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
        
        # Wrong data types
        response = self.client.post("/lore_adherence", json={
            "text": 123,  # Should be string
            "target": self.test_lore_fact
        })
        self.assertEqual(response.status_code, 422)
    
    def test_empty_text_handling(self):
        """Test handling of empty text inputs"""
        response = self.client.post("/personality_alignment", json={
            "text": "",
            "target": self.test_big_five
        })
        self.assertEqual(response.status_code, 400)  # Bad Request
        
        data = response.json()
        self.assertIn("detail", data)
    
    def test_dev_mode_random_scores(self):
        """Test that dev mode returns random scores when API key is missing"""
        # Clear env vars to simulate missing API key
        old_key = os.environ.get("OPENAI_API_KEY")
        if old_key:
            del os.environ["OPENAI_API_KEY"]
        
        try:
            response = self.client.post("/personality_alignment", json={
                "text": self.test_response,
                "target": self.test_big_five
            })
            
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("score", data)
            self.assertIn("dev_mode", data)
            self.assertTrue(data["dev_mode"])
            
            # Score should be between 0 and 1
            self.assertGreaterEqual(data["score"], 0.0)
            self.assertLessEqual(data["score"], 1.0)
        finally:
            # Restore original key
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
    
    def test_telemetry_logging(self):
        """Test that telemetry data is logged correctly"""
        from backend.app.services.judge_service.main import telemetry_logger
        
        # Simple test - just ensure endpoint works and telemetry logger exists
        response = self.client.post("/personality_alignment", json={
            "text": self.test_response,
            "target": self.test_big_five
        })
        
        self.assertEqual(response.status_code, 200)
        # Verify telemetry logger has the required methods
        self.assertTrue(hasattr(telemetry_logger, 'log_request'))
        self.assertTrue(hasattr(telemetry_logger, 'get_metrics'))
    
    def test_prompt_template_loading(self):
        """Test that prompt templates are loaded correctly"""
        from backend.app.services.judge_service.prompts import load_personality_prompt, load_lore_prompt
        
        personality_prompt = load_personality_prompt()
        lore_prompt = load_lore_prompt()
        
        # Should contain key elements from original implementations
        self.assertIn("personality psychologist", personality_prompt.lower())
        self.assertIn("big-five", personality_prompt.lower())
        self.assertIn("lore consistency", lore_prompt.lower())
        self.assertIn("lore fact", lore_prompt.lower())


class TestJudgeServiceIntegration(unittest.TestCase):
    """Integration tests for the Judge Service"""
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_end_to_end_personality_evaluation(self):
        """Test end-to-end personality evaluation with real LLM calls"""
        # This test will initially fail - that's expected in TDD
        from backend.app.services.judge_service.main import app
        
        client = TestClient(app)
        
        # Test with high openness/extraversion response
        response = client.post("/personality_alignment", json={
            "text": "I absolutely love trying new cuisines and exploring different cultures! Meeting new people energizes me so much!",
            "target": {
                "openness": 0.9,
                "conscientiousness": 0.5,
                "extraversion": 0.8,
                "agreeableness": 0.7,
                "neuroticism": 0.2
            }
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("score", data)
        self.assertIsInstance(data["score"], (int, float))
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)
        
        # For high alignment, expect score > 0.6
        if not data.get("dev_mode", False):  # Only check if not in dev mode
            self.assertGreater(data["score"], 0.6)
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_end_to_end_lore_adherence_evaluation(self):
        """Test end-to-end lore adherence evaluation with real LLM calls"""
        from backend.app.services.judge_service.main import app
        
        client = TestClient(app)
        
        # Test with lore-compliant response
        response = client.post("/lore_adherence", json={
            "text": "I cannot use magic here because it is strictly forbidden in the capital city.",
            "target": "Magic is forbidden in the capital city of Whiterun."
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("score", data)
        self.assertIsInstance(data["score"], (int, float))
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)
        
        # For good lore adherence, expect score > 0.6
        if not data.get("dev_mode", False):  # Only check if not in dev mode
            self.assertGreater(data["score"], 0.6)
        
        # Test with lore-contradicting response
        response = client.post("/lore_adherence", json={
            "text": "Let me cast a powerful spell right here in the capital!",
            "target": "Magic is forbidden in the capital city of Whiterun."
        })
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("score", data)
        
        # For lore contradiction, expect score < 0.4
        if not data.get("dev_mode", False):  # Only check if not in dev mode
            self.assertLess(data["score"], 0.4)
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_caching_with_real_requests(self):
        """Test that caching works correctly with real LLM requests"""
        from backend.app.services.judge_service.main import app
        
        client = TestClient(app)
        
        # Make the same request twice
        request_data = {
            "text": "I enjoy quiet contemplation and reading books.",
            "target": {
                "openness": 0.7,
                "conscientiousness": 0.8,
                "extraversion": 0.3,
                "agreeableness": 0.6,
                "neuroticism": 0.2
            }
        }
        
        # First request
        response1 = client.post("/personality_alignment", json=request_data)
        self.assertEqual(response1.status_code, 200)
        data1 = response1.json()
        
        # Second identical request (should use cache)
        response2 = client.post("/personality_alignment", json=request_data)
        self.assertEqual(response2.status_code, 200)
        data2 = response2.json()
        
        # Results should be identical due to caching
        if not data1.get("dev_mode", False):  # Only check if not in dev mode
            self.assertEqual(data1["score"], data2["score"])
        
        # Both should be valid scores
        self.assertGreaterEqual(data1["score"], 0.0)
        self.assertLessEqual(data1["score"], 1.0)
        self.assertGreaterEqual(data2["score"], 0.0)
        self.assertLessEqual(data2["score"], 1.0) 