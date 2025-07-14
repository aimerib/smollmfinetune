"""
Tests for Streaming Inference Engine

Tests the real-time multimodal generation capabilities of the streaming inference system.
"""

import pytest
import asyncio
import torch
import numpy as np
import time
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass

from backend.app.services.inference.streaming_inference_engine import (
    StreamingInferenceEngine,
    GenerationState, 
    StreamingStep,
    KVCacheManager,
    CharacterStateManager,
    create_streaming_engine
)
from backend.app.narrative_engine.config import NarrativeLLMConfig
from backend.app.services.dataset.multimodal_dataset_pipeline import CharacterVoiceRegistry


class TestGenerationState:
    """Test generation state management"""
    
    def test_generation_state_creation(self):
        """Test creating generation state"""
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1, 1]])
        
        state = GenerationState(
            input_ids=input_ids,
            attention_mask=attention_mask,
            character_id="alice",
            character_embedding_idx=0
        )
        
        assert torch.equal(state.input_ids, input_ids)
        assert torch.equal(state.attention_mask, attention_mask)
        assert state.character_id == "alice"
        assert state.character_embedding_idx == 0
        assert state.past_key_values is None
        assert len(state.speech_frames) == 0
        assert len(state.control_history) == 0
        assert len(state.memory_updates) == 0
        assert state.total_tokens_generated == 0
        assert state.total_speech_frames == 0
    
    def test_generation_state_reset(self):
        """Test resetting generation state"""
        state = GenerationState(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
            character_id="alice"
        )
        
        # Add some data
        state.speech_frames.append(torch.randn(1, 80))
        state.control_history.append("happy")
        state.memory_updates.append({"test": "data"})
        state.total_tokens_generated = 10
        state.total_speech_frames = 5
        state.past_key_values = (torch.randn(2, 4, 8, 64),)
        
        # Reset
        state.reset()
        
        assert state.past_key_values is None
        assert len(state.speech_frames) == 0
        assert len(state.control_history) == 0
        assert len(state.memory_updates) == 0
        assert state.total_tokens_generated == 0
        assert state.total_speech_frames == 0


class TestStreamingStep:
    """Test streaming step data structure"""
    
    def test_streaming_step_creation(self):
        """Test creating streaming step"""
        step = StreamingStep(
            step_id=1,
            timestamp=time.time(),
            text_token="hello",
            control_signal="happy",
            latency_ms=15.5
        )
        
        assert step.step_id == 1
        assert step.text_token == "hello"
        assert step.control_signal == "happy"
        assert step.latency_ms == 15.5
        assert step.is_finished is False
        assert isinstance(step.confidence_scores, dict)
    
    def test_streaming_step_completion(self):
        """Test marking streaming step as finished"""
        step = StreamingStep(
            step_id=5,
            timestamp=time.time(),
            is_finished=True
        )
        
        assert step.is_finished is True
        assert step.step_id == 5


class TestKVCacheManager:
    """Test KV cache management functionality"""
    
    def test_cache_manager_initialization(self):
        """Test cache manager initializes correctly"""
        manager = KVCacheManager(max_cache_length=1024)
        
        assert manager.max_cache_length == 1024
        assert manager.cache_data is None
        assert manager.cache_length == 0
    
    def test_cache_manager_first_update(self):
        """Test first cache update"""
        manager = KVCacheManager()
        
        # Mock cache data (simplified structure)
        new_cache = (
            (torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64)),  # Layer 0 (key, value)
            (torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64)),  # Layer 1 (key, value)
        )
        
        updated_cache = manager.update_cache(new_cache, 10)
        
        assert manager.cache_data is not None
        assert manager.cache_length == 10
        assert updated_cache == new_cache
    
    def test_cache_manager_concatenation(self):
        """Test cache concatenation"""
        manager = KVCacheManager()
        
        # First cache
        cache1 = (
            (torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)),
        )
        manager.update_cache(cache1, 5)
        
        # Second cache  
        cache2 = (
            (torch.randn(1, 8, 3, 64), torch.randn(1, 8, 3, 64)),
        )
        updated_cache = manager.update_cache(cache2, 3)
        
        assert manager.cache_length == 8
        # Check concatenated dimensions
        assert updated_cache[0][0].shape[2] == 8  # 5 + 3 = 8 sequence length
        assert updated_cache[0][1].shape[2] == 8
    
    def test_cache_manager_truncation(self):
        """Test cache truncation when exceeding max length"""
        manager = KVCacheManager(max_cache_length=10)
        
        # Add initial cache that fills the limit
        large_cache = (
            (torch.randn(1, 8, 8, 64), torch.randn(1, 8, 8, 64)),
        )
        manager.update_cache(large_cache, 8)
        
        # Add more cache that would exceed limit
        new_cache = (
            (torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)),
        )
        updated_cache = manager.update_cache(new_cache, 5)
        
        # Should truncate to maintain max_cache_length
        assert manager.cache_length <= manager.max_cache_length
        assert updated_cache[0][0].shape[2] <= manager.max_cache_length
    
    def test_cache_manager_clear(self):
        """Test clearing cache"""
        manager = KVCacheManager()
        
        # Add some cache
        cache = (
            (torch.randn(1, 8, 5, 64), torch.randn(1, 8, 5, 64)),
        )
        manager.update_cache(cache, 5)
        
        assert manager.cache_length > 0
        assert manager.cache_data is not None
        
        # Clear
        manager.clear()
        
        assert manager.cache_length == 0
        assert manager.cache_data is None


class TestCharacterStateManager:
    """Test character state management"""
    
    def test_character_state_manager_initialization(self):
        """Test character state manager initializes correctly"""
        registry = CharacterVoiceRegistry()
        manager = CharacterStateManager(registry)
        
        assert manager.character_registry == registry
        assert len(manager.active_states) == 0
    
    def test_get_or_create_state_new_character(self):
        """Test creating state for new character"""
        registry = CharacterVoiceRegistry()
        manager = CharacterStateManager(registry)
        
        state = manager.get_or_create_state("alice")
        
        assert state.character_id == "alice"
        assert state.character_embedding_idx == 0  # First character gets index 0
        assert "alice" in manager.active_states
        assert registry.get_character_idx("alice") == 0
    
    def test_get_or_create_state_existing_character(self):
        """Test getting state for existing character"""
        registry = CharacterVoiceRegistry()
        registry.register_character("alice", {"voice": "friendly"})
        manager = CharacterStateManager(registry)
        
        state1 = manager.get_or_create_state("alice")
        state2 = manager.get_or_create_state("alice")
        
        assert state1 is state2  # Should return same state object
        assert state1.character_id == "alice"
    
    def test_update_character_state(self):
        """Test updating character state"""
        registry = CharacterVoiceRegistry()
        manager = CharacterStateManager(registry)
        
        state = manager.get_or_create_state("alice")
        original_tokens = state.total_tokens_generated
        
        manager.update_character_state("alice", total_tokens_generated=42)
        
        assert state.total_tokens_generated == 42
        assert state.total_tokens_generated != original_tokens
    
    def test_reset_character_state(self):
        """Test resetting character state"""
        registry = CharacterVoiceRegistry()
        manager = CharacterStateManager(registry)
        
        state = manager.get_or_create_state("alice")
        state.control_history.append("happy")
        state.total_tokens_generated = 10
        
        manager.reset_character_state("alice")
        
        assert len(state.control_history) == 0
        assert state.total_tokens_generated == 0


class TestStreamingInferenceEngine:
    """Test the main streaming inference engine"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock quad-head model"""
        model = Mock()
        model.eval.return_value = None
        
        # Mock model outputs
        mock_outputs = Mock()
        mock_outputs.logits = torch.randn(1, 10, 50000)  # (batch, seq, vocab)
        mock_outputs.text_logits = torch.randn(1, 10, 50000)
        mock_outputs.speech_logits = torch.randn(1, 10, 80, 16)  # (batch, seq, mel_bins, quantization_levels)
        mock_outputs.control_logits = torch.randn(1, 10, 10)  # (batch, seq, control_vocab)
        mock_outputs.memory_logits = torch.randn(1, 10, 512)  # (batch, seq, memory_dim)
        mock_outputs.past_key_values = None
        
        model.return_value = mock_outputs
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer"""
        tokenizer = Mock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }
        tokenizer.decode.return_value = " world"
        return tokenizer
    
    def test_engine_initialization(self, mock_model):
        """Test engine initializes correctly"""
        config = NarrativeLLMConfig()
        registry = CharacterVoiceRegistry()
        
        engine = StreamingInferenceEngine(
            model=mock_model,
            config=config,
            character_registry=registry
        )
        
        assert engine.model == mock_model
        assert engine.config == config
        assert engine.character_registry == registry
        assert isinstance(engine.character_manager, CharacterStateManager)
        assert isinstance(engine.kv_cache_manager, KVCacheManager)
        assert engine.generation_stats["total_tokens"] == 0
    
    def test_engine_load_model_fallback(self):
        """Test engine loads model when not provided"""
        config = NarrativeLLMConfig()
        
        with patch('backend.app.services.inference.streaming_inference_engine.create_quad_head_model') as mock_create:
            mock_model = Mock()
            mock_model.eval.return_value = None
            mock_create.return_value = mock_model
            
            engine = StreamingInferenceEngine(config=config)
            
            assert engine.model == mock_model
            mock_create.assert_called_once_with(config)
    
    @pytest.mark.asyncio
    async def test_stream_generate_basic(self, mock_model, mock_tokenizer):
        """Test basic streaming generation"""
        engine = StreamingInferenceEngine(
            model=mock_model,
            tokenizer=mock_tokenizer
        )
        
        steps = []
        async for step in engine.stream_generate(
            prompt="Hello",
            character_id="alice",
            max_new_tokens=3,
            temperature=0.7
        ):
            steps.append(step)
            if step.is_finished:
                break
        
        assert len(steps) > 0
        assert all(isinstance(step, StreamingStep) for step in steps)
        assert steps[-1].is_finished is True
        
        # Check that character state was created
        assert "alice" in engine.character_manager.active_states
    
    @pytest.mark.asyncio
    async def test_stream_generate_without_tokenizer(self, mock_model):
        """Test streaming generation without tokenizer"""
        engine = StreamingInferenceEngine(model=mock_model)
        
        steps = []
        async for step in engine.stream_generate(
            prompt="hello world",
            character_id="bob",
            max_new_tokens=2
        ):
            steps.append(step)
            if step.is_finished:
                break
        
        assert len(steps) > 0
        assert steps[0].text_token is not None  # Should generate token strings
        
    def test_sample_text_token_greedy(self, mock_model):
        """Test greedy text token sampling"""
        engine = StreamingInferenceEngine(model=mock_model)
        
        # Create logits with clear maximum
        logits = torch.tensor([[[1.0, 5.0, 2.0, 1.0]]])  # Token 1 has highest logit
        
        token_id, text_token, confidence = engine._sample_text_token(
            logits, temperature=1.0, top_p=1.0, do_sample=False
        )
        
        assert token_id == 1  # Should select token with highest logit
        assert text_token.startswith("<token_")  # Without tokenizer, uses placeholder
        assert 0.0 <= confidence <= 1.0
    
    def test_sample_text_token_with_temperature(self, mock_model):
        """Test text token sampling with temperature"""
        engine = StreamingInferenceEngine(model=mock_model)
        
        logits = torch.tensor([[[1.0, 2.0, 1.0, 1.0]]])
        
        # Test with high temperature (more random)
        token_id, text_token, confidence = engine._sample_text_token(
            logits, temperature=2.0, top_p=1.0, do_sample=True
        )
        
        assert token_id is not None
        assert text_token is not None
        assert 0.0 <= confidence <= 1.0
    
    def test_sample_speech_frame(self, mock_model):
        """Test speech frame sampling"""
        engine = StreamingInferenceEngine(model=mock_model)
        
        # Create speech logits (batch, seq, mel_bins, quantization_levels)
        speech_logits = torch.randn(1, 5, 80, 16)
        
        speech_frame, confidence = engine._sample_speech_frame(speech_logits, temperature=1.0)
        
        assert speech_frame is not None
        assert speech_frame.shape == (1, 80)  # (1, mel_bins)
        assert 0.0 <= confidence <= 1.0
        # Values should be normalized to [0, 1]
        assert speech_frame.min() >= 0.0
        assert speech_frame.max() <= 1.0
    
    def test_process_control_signal(self, mock_model):
        """Test control signal processing"""
        engine = StreamingInferenceEngine(model=mock_model)
        
        # Create control logits with clear maximum for "happy" (index 1)
        control_logits = torch.zeros(1, 5, 10)
        control_logits[0, -1, 1] = 10.0  # High value for "happy"
        
        control_signal, confidence = engine._process_control_signal(control_logits)
        
        assert control_signal == "happy"
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high due to clear maximum
    
    def test_process_memory_update(self, mock_model):
        """Test memory update processing"""
        engine = StreamingInferenceEngine(model=mock_model)
        
        memory_logits = torch.randn(1, 5, 512)
        
        memory_update, confidence = engine._process_memory_update(memory_logits)
        
        assert memory_update is not None
        assert "embedding" in memory_update
        assert "timestamp" in memory_update
        assert "importance" in memory_update
        assert "type" in memory_update
        assert len(memory_update["embedding"]) == 512
        assert 0.0 <= confidence <= 1.0
    
    def test_should_stop_generation(self, mock_model):
        """Test generation stopping conditions"""
        engine = StreamingInferenceEngine(model=mock_model)
        
        # Test max tokens stopping
        assert engine._should_stop_generation(None, None, 100, 100) is True
        assert engine._should_stop_generation(None, None, 50, 100) is False
        
        # Test end-of-sequence token stopping
        assert engine._should_stop_generation("</s>", None, 10, 100) is True
        assert engine._should_stop_generation("<eos>", None, 10, 100) is True
        assert engine._should_stop_generation("hello", None, 10, 100) is False
        
        # Test control signal stopping
        assert engine._should_stop_generation(None, "end_conversation", 10, 100) is True
        assert engine._should_stop_generation(None, "stop", 10, 100) is True
        assert engine._should_stop_generation(None, "happy", 10, 100) is False
    
    def test_performance_stats_tracking(self, mock_model):
        """Test performance statistics tracking"""
        engine = StreamingInferenceEngine(model=mock_model)
        
        initial_stats = engine.get_performance_stats()
        assert initial_stats["total_tokens"] == 0
        assert initial_stats["num_requests"] == 0
        
        # Simulate updating stats
        engine._update_stats(tokens_generated=10, total_time=1.0)
        
        updated_stats = engine.get_performance_stats()
        assert updated_stats["total_tokens"] == 10
        assert updated_stats["num_requests"] == 1
        assert updated_stats["avg_tokens_per_second"] > 0
    
    def test_reset_character_conversation(self, mock_model):
        """Test resetting character conversation"""
        engine = StreamingInferenceEngine(model=mock_model)
        
        # Create character state
        state = engine.character_manager.get_or_create_state("alice")
        state.control_history.append("happy")
        
        # Add cache data
        engine.kv_cache_manager.cache_data = (torch.randn(2, 4, 8, 64),)
        engine.kv_cache_manager.cache_length = 8
        
        # Reset
        engine.reset_character_conversation("alice")
        
        assert len(state.control_history) == 0
        assert engine.kv_cache_manager.cache_data is None
        assert engine.kv_cache_manager.cache_length == 0
    
    @pytest.mark.asyncio
    async def test_character_conversation_context_manager(self, mock_model):
        """Test character conversation context manager"""
        engine = StreamingInferenceEngine(model=mock_model)
        
        async with engine.character_conversation("alice") as state:
            assert state.character_id == "alice"
            assert isinstance(state, GenerationState)
        
        # State should still exist after context
        assert "alice" in engine.character_manager.active_states


class TestFactoryFunctions:
    """Test factory functions and utilities"""
    
    def test_create_streaming_engine(self):
        """Test streaming engine factory function"""
        config = NarrativeLLMConfig()
        
        with patch('backend.app.services.inference.streaming_inference_engine.create_quad_head_model') as mock_create:
            mock_model = Mock()
            mock_model.eval.return_value = None
            mock_create.return_value = mock_model
            
            engine = create_streaming_engine(config=config, device="cpu")
            
            assert isinstance(engine, StreamingInferenceEngine)
            assert engine.config == config
            assert engine.device == "cpu"


class TestIntegration:
    """Integration tests for the complete streaming system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_streaming_generation(self):
        """Test complete end-to-end streaming generation"""
        config = NarrativeLLMConfig()
        config.enable_speech_head = True
        
        with patch('backend.app.services.inference.streaming_inference_engine.create_quad_head_model') as mock_create:
            # Create comprehensive mock model
            mock_model = Mock()
            mock_model.eval.return_value = None
            
            # Mock realistic outputs
            outputs = Mock()
            outputs.text_logits = torch.randn(1, 1, 50000)
            outputs.speech_logits = torch.randn(1, 1, 80, 16) 
            outputs.control_logits = torch.randn(1, 1, 10)
            outputs.memory_logits = torch.randn(1, 1, 512)
            outputs.past_key_values = None
            
            mock_model.return_value = outputs
            mock_create.return_value = mock_model
            
            # Create engine
            engine = create_streaming_engine(config=config)
            
            # Run streaming generation
            steps = []
            async for step in engine.stream_generate(
                prompt="Hello there!",
                character_id="test_character",
                max_new_tokens=5,
                temperature=0.8
            ):
                steps.append(step)
                
                # Verify step structure
                assert isinstance(step, StreamingStep)
                assert step.step_id >= 0
                assert step.timestamp > 0
                assert step.latency_ms >= 0
                
                if step.is_finished:
                    break
            
            # Verify generation completed
            assert len(steps) > 0
            assert steps[-1].is_finished is True
            
            # Verify character state was managed
            assert "test_character" in engine.character_manager.active_states
            
            # Verify performance stats updated
            stats = engine.get_performance_stats()
            assert stats["num_requests"] == 1
            assert stats["total_tokens"] > 0
    
    @pytest.mark.asyncio
    async def test_multiple_character_conversations(self):
        """Test handling multiple character conversations simultaneously"""
        config = NarrativeLLMConfig()
        
        with patch('backend.app.services.inference.streaming_inference_engine.create_quad_head_model') as mock_create:
            mock_model = Mock()
            mock_model.eval.return_value = None
            
            # Simple mock outputs
            outputs = Mock()
            outputs.text_logits = torch.randn(1, 1, 50000)
            outputs.speech_logits = None
            outputs.control_logits = None
            outputs.memory_logits = None
            outputs.past_key_values = None
            
            mock_model.return_value = outputs
            mock_create.return_value = mock_model
            
            engine = create_streaming_engine(config=config)
            
            # Start conversations with different characters
            characters = ["alice", "bob", "charlie"]
            
            for char_id in characters:
                steps = []
                async for step in engine.stream_generate(
                    prompt=f"Hello {char_id}!",
                    character_id=char_id,
                    max_new_tokens=2
                ):
                    steps.append(step)
                    if step.is_finished:
                        break
                
                assert len(steps) > 0
            
            # Verify all character states exist
            for char_id in characters:
                assert char_id in engine.character_manager.active_states
                state = engine.character_manager.active_states[char_id]
                assert state.character_id == char_id
    
    @pytest.mark.asyncio
    async def test_error_handling_during_generation(self):
        """Test error handling during streaming generation"""
        config = NarrativeLLMConfig()
        
        with patch('backend.app.services.inference.streaming_inference_engine.create_quad_head_model') as mock_create:
            mock_model = Mock()
            mock_model.eval.return_value = None
            
            # Make model raise exception on forward pass
            mock_model.side_effect = RuntimeError("Model inference failed")
            mock_create.return_value = mock_model
            
            engine = create_streaming_engine(config=config)
            
            # Generation should handle error gracefully
            steps = []
            async for step in engine.stream_generate(
                prompt="Test prompt",
                character_id="error_test",
                max_new_tokens=5
            ):
                steps.append(step)
                if step.is_finished:
                    break
            
            # Should receive at least one step indicating completion due to error
            assert len(steps) >= 1
            assert steps[-1].is_finished is True 