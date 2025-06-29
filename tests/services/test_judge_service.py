"""
Tests for Judge Model Service (R4-1.2)

Tests the centralized LLM-as-judge microservice including:
- FastAPI endpoints for personality alignment and lore adherence
- SQLite caching layer with SHA256 keys
- Health check endpoint
- Mock-based testing for LLM calls
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock
import json
import hashlib
import sqlite3
import tempfile
import os
from fastapi.testclient import TestClient
import httpx
import pytest


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
        from services.judge_service.main import app
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
        from services.judge_service.cache import CacheManager
        
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
        from services.judge_service.cache import CacheManager
        
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
        from services.judge_service.cache import CacheManager
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
        with patch.dict(os.environ, {}, clear=True):  # Clear all env vars
            with patch('services.judge_service.main.call_llm_judge') as mock_llm:
                # Should not be called in dev mode
                mock_llm.return_value = None
                
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
    
    def test_telemetry_logging(self):
        """Test that telemetry data is logged correctly"""
        from services.judge_service.main import telemetry_logger
        
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
        from services.judge_service.prompts import load_personality_prompt, load_lore_prompt
        
        personality_prompt = load_personality_prompt()
        lore_prompt = load_lore_prompt()
        
        # Should contain key elements from original implementations
        self.assertIn("personality psychologist", personality_prompt.lower())
        self.assertIn("big-five", personality_prompt.lower())
        self.assertIn("lore consistency", lore_prompt.lower())
        self.assertIn("lore fact", lore_prompt.lower())


class TestJudgeServiceIntegration(unittest.TestCase):
    """Integration tests for the Judge Service"""
    
    def test_end_to_end_personality_evaluation(self):
        """Test end-to-end personality evaluation with real prompts"""
        # This test will initially fail - that's expected in TDD
        from services.judge_service.main import app
        
        client = TestClient(app)
        
        # Mock the actual OpenAI/Anthropic API call
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": '{"alignment_score": 0.85}'}}]
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            response = client.post("/personality_alignment", json={
                "text": "I absolutely love trying new cuisines and exploring different cultures!",
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
            self.assertGreaterEqual(data["score"], 0.0)
            self.assertLessEqual(data["score"], 1.0) 