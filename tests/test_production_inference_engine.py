"""
Test suite for Production Inference Engine (R4-10)

Tests the high-performance inference server with triple-head outputs,
hot-swappable adapters, and memory integration.
"""

import asyncio
import pytest
import pytest_asyncio
import time
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np

# Import the components we'll be testing (these will be created next)
# from app.inference_engine import (
#     ProductionInferenceEngine,
#     AdapterManager,
#     MemoryService,
#     SessionStateManager,
#     InferenceRequest,
#     InferenceResponse,
#     TripleHeadOutput,
#     AdapterMetadata,
#     MemoryVector
# )


@pytest_asyncio.fixture
async def inference_engine():
    """Create inference engine instance for testing"""
    from app.inference_engine import ProductionInferenceEngine
    engine = ProductionInferenceEngine()
    await engine.initialize()
    yield engine
    await engine.shutdown()


@pytest.fixture
def mock_model():
    """Create mock model with triple-head outputs"""
    from app.inference_engine import TripleHeadOutput
    model = AsyncMock()
    model.generate.return_value = TripleHeadOutput(
        generation_text="Hello, I'm your character!",
        control_tokens=[{"token": "<emotion_happy>", "probability": 0.8}],
        memory_vector=np.random.rand(768).tolist(),
        memory_metadata={
            "importance": 0.7,
            "emotional_valence": 0.5,
            "recency": 1.0,
            "coherence": 0.8
        }
    )
    return model


class TestProductionInferenceEngine:
    """Test high-performance inference server"""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test server starts with proper configuration"""
        from app.inference_engine import ProductionInferenceEngine
        engine = ProductionInferenceEngine()
        await engine.initialize()
        
        assert engine.is_ready()
        assert engine.health_check()["status"] == "healthy"
        # GPU memory check - may be 0 on CPU/MPS systems
        assert engine.gpu_memory_available() >= 0
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_triple_head_inference(self, inference_engine, mock_model):
        """Test inference with triple-head outputs"""
        from app.inference_engine import InferenceRequest, InferenceResponse
        
        # Mock the _load_model method
        with patch.object(inference_engine, '_load_model', new_callable=AsyncMock) as mock_load:
            mock_load.return_value = mock_model
            
            # Mock the process_request to return expected output
            async def mock_process(request):
                from app.inference_engine import TripleHeadOutput
                output = TripleHeadOutput(
                    generation_text="Hello, I'm your character!",
                    control_tokens=[{"token": "<emotion_happy>", "probability": 0.8}],
                    memory_vector=np.random.rand(768).tolist(),
                    memory_metadata={
                        "importance": 0.7,
                        "emotional_valence": 0.5,
                        "recency": 1.0,
                        "coherence": 0.8
                    }
                )
                return InferenceResponse.from_triple_head(
                    request, output, 150.0, False
                )
            
            with patch.object(inference_engine, '_process_request', new=mock_process):
                request = InferenceRequest(
                    session_id="test-session",
                    character_id="test-char",
                    prompt="Hello, how are you?",
                    max_tokens=150,
                    temperature=0.8
                )
                
                response = await inference_engine.generate(request)
                
                assert isinstance(response, InferenceResponse)
                assert response.generation_text == "Hello, I'm your character!"
                assert len(response.control_tokens) > 0
                assert response.control_tokens[0]["token"] == "<emotion_happy>"
                assert len(response.memory_vector) == 768
                assert response.memory_metadata["importance"] == 0.7
    
    @pytest.mark.asyncio
    async def test_concurrent_inference(self, inference_engine, mock_model):
        """Test handling multiple concurrent requests"""
        from app.inference_engine import InferenceRequest, InferenceResponse
        
        with patch.object(inference_engine, '_load_model', return_value=mock_model):
            requests = [
                InferenceRequest(
                    session_id=f"session-{i}",
                    character_id=f"char-{i}",
                    prompt=f"Test prompt {i}",
                    max_tokens=100
                )
                for i in range(10)
            ]
            
            start_time = time.time()
            responses = await asyncio.gather(*[
                inference_engine.generate(req) for req in requests
            ])
            elapsed = time.time() - start_time
            
            assert len(responses) == 10
            assert all(isinstance(r, InferenceResponse) for r in responses)
            # Should process 10 requests in under 2 seconds
            assert elapsed < 2.0
    
    @pytest.mark.asyncio
    async def test_attention_caching(self, inference_engine):
        """Test GPU memory optimization with attention caching"""
        from app.inference_engine import InferenceRequest
        
        # First request - cold cache
        request1 = InferenceRequest(
            session_id="test-session",
            character_id="test-char",
            prompt="Tell me about yourself",
            use_cache=True
        )
        
        metrics1 = await inference_engine.generate_with_metrics(request1)
        
        # Second request - should use cached attention
        request2 = InferenceRequest(
            session_id="test-session",
            character_id="test-char",
            prompt="What do you like to do?",
            use_cache=True
        )
        
        metrics2 = await inference_engine.generate_with_metrics(request2)
        
        assert metrics2["cache_hit"] == True
        assert metrics2["inference_time_ms"] < metrics1["inference_time_ms"]
    
    @pytest.mark.asyncio
    async def test_request_queueing(self, inference_engine):
        """Test request queueing and load balancing"""
        # Simulate high load
        requests = [
            InferenceRequest(
                session_id=f"session-{i}",
                character_id="test-char",
                prompt=f"Request {i}",
                priority=i % 3  # Different priorities
            )
            for i in range(50)
        ]
        
        # Submit all requests
        futures = [inference_engine.generate(req) for req in requests]
        
        # Check queue status
        queue_status = inference_engine.get_queue_status()
        assert queue_status["queued_requests"] > 0
        assert queue_status["processing_requests"] <= inference_engine.max_concurrent
        
        # Wait for all to complete
        responses = await asyncio.gather(*futures)
        assert len(responses) == 50
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, inference_engine):
        """Test health monitoring and automatic recovery"""
        # Get initial health
        health = inference_engine.health_check()
        assert health["status"] == "healthy"
        assert health["gpu_utilization"] < 0.9
        assert health["memory_usage_gb"] < 20
        
        # Simulate unhealthy state
        with patch.object(inference_engine, '_gpu_memory_usage', return_value=0.95):
            health = inference_engine.health_check()
            assert health["status"] == "warning"
            assert health["warnings"] == ["High GPU memory usage"]
        
        # Test automatic recovery
        recovery_triggered = await inference_engine.check_and_recover()
        assert recovery_triggered == True


class TestAdapterManager:
    """Test adapter management system"""
    
    @pytest.mark.asyncio
    async def test_hot_swap_adapter(self):
        """Test hot-swapping adapters without model restart"""
        manager = AdapterManager()
        
        # Load initial adapter
        adapter1_path = "adapters/character1_v1.safetensors"
        await manager.load_adapter("char1", adapter1_path)
        
        assert manager.get_active_adapter("char1") == adapter1_path
        assert manager.get_adapter_memory_usage("char1") < 500  # MB
        
        # Hot-swap to new adapter
        adapter2_path = "adapters/character1_v2.safetensors"
        swap_time = await manager.hot_swap_adapter("char1", adapter2_path)
        
        assert manager.get_active_adapter("char1") == adapter2_path
        assert swap_time < 1.0  # Under 1 second
        
        # Verify old adapter is unloaded
        assert adapter1_path not in manager.loaded_adapters
    
    @pytest.mark.asyncio
    async def test_adapter_versioning(self):
        """Test adapter versioning and rollback"""
        manager = AdapterManager()
        
        # Load multiple versions
        versions = ["v1", "v2", "v3"]
        for version in versions:
            await manager.load_adapter(
                "char1",
                f"adapters/character1_{version}.safetensors",
                version=version
            )
        
        # Check version history
        history = manager.get_version_history("char1")
        assert len(history) == 3
        assert history[-1]["version"] == "v3"
        
        # Rollback to v1
        await manager.rollback_adapter("char1", "v1")
        assert manager.get_active_version("char1") == "v1"
    
    @pytest.mark.asyncio
    async def test_multi_adapter_inference(self):
        """Test inference with multiple adapters for character ensemble"""
        manager = AdapterManager()
        
        # Load multiple character adapters
        characters = ["alice", "bob", "charlie"]
        for char in characters:
            await manager.load_adapter(char, f"adapters/{char}.safetensors")
        
        # Test ensemble inference
        results = await manager.ensemble_inference(
            prompt="What do you think about this?",
            character_ids=characters,
            aggregation="weighted"
        )
        
        assert len(results) == 3
        assert all(char in results for char in characters)
        assert all("response" in results[char] for char in characters)
    
    @pytest.mark.asyncio
    async def test_adapter_performance_monitoring(self):
        """Test adapter performance monitoring and benchmarking"""
        manager = AdapterManager()
        
        await manager.load_adapter("char1", "adapters/character1.safetensors")
        
        # Run benchmark
        benchmark = await manager.benchmark_adapter("char1", num_samples=10)
        
        assert "avg_inference_time_ms" in benchmark
        assert "tokens_per_second" in benchmark
        assert "memory_usage_mb" in benchmark
        assert benchmark["tokens_per_second"] > 100  # Reasonable throughput
    
    @pytest.mark.asyncio
    async def test_memory_head_adapter_support(self):
        """Test memory-specific adapter fine-tuning support"""
        manager = AdapterManager()
        
        # Load adapter with memory head fine-tuning
        metadata = AdapterMetadata(
            character_id="char1",
            adapter_type="triple_head",
            memory_head_trained=True,
            memory_vector_dim=768
        )
        
        await manager.load_adapter("char1", "adapters/char1_memory.safetensors", metadata)
        
        adapter_info = manager.get_adapter_info("char1")
        assert adapter_info["memory_head_trained"] == True
        assert adapter_info["memory_vector_dim"] == 768


class TestMemoryService:
    """Test memory integration architecture"""
    
    @pytest.mark.asyncio
    async def test_memory_embedding_storage(self):
        """Test memory embedding and storage"""
        service = MemoryService()
        
        memory = MemoryVector(
            session_id="test-session",
            character_id="test-char",
            embedding=np.random.rand(768).tolist(),
            content="I love exploring new places",
            timestamp=time.time(),
            metadata={
                "importance": 0.8,
                "emotional_valence": 0.7
            }
        )
        
        memory_id = await service.store_memory(memory)
        assert memory_id is not None
        
        # Retrieve memory
        retrieved = await service.get_memory(memory_id)
        assert retrieved.content == memory.content
        assert len(retrieved.embedding) == 768
    
    @pytest.mark.asyncio
    async def test_vector_search(self):
        """Test vector similarity search"""
        service = MemoryService()
        
        # Store multiple memories
        memories = [
            MemoryVector(
                session_id="test-session",
                character_id="test-char",
                embedding=np.random.rand(768).tolist(),
                content=content,
                timestamp=time.time()
            )
            for content in [
                "I love pizza",
                "My favorite food is pasta",
                "I enjoy hiking in mountains",
                "The weather is nice today"
            ]
        ]
        
        for memory in memories:
            await service.store_memory(memory)
        
        # Search for food-related memories
        query_embedding = np.random.rand(768).tolist()
        results = await service.search_memories(
            query_embedding=query_embedding,
            session_id="test-session",
            top_k=2
        )
        
        assert len(results) == 2
        assert all(hasattr(r, 'similarity_score') for r in results)
        assert results[0].similarity_score > results[1].similarity_score
    
    @pytest.mark.asyncio
    async def test_cross_attention_injection(self):
        """Test memory injection into model attention"""
        service = MemoryService()
        
        # Prepare memories for injection
        memories = await service.get_relevant_memories(
            session_id="test-session",
            character_id="test-char",
            context="Tell me about your favorite activities",
            top_k=5
        )
        
        # Format for cross-attention
        attention_context = service.format_for_attention(memories)
        
        assert "memory_embeddings" in attention_context
        assert "memory_weights" in attention_context
        assert len(attention_context["memory_embeddings"]) <= 5
    
    @pytest.mark.asyncio
    async def test_memory_quality_scoring(self):
        """Test memory quality scoring and pruning"""
        service = MemoryService()
        
        # Create memories with different quality scores
        memories = []
        for i in range(10):
            memory = MemoryVector(
                session_id="test-session",
                character_id="test-char",
                embedding=np.random.rand(768).tolist(),
                content=f"Memory {i}",
                timestamp=time.time() - i * 3600,  # Older memories
                metadata={
                    "importance": np.random.rand(),
                    "coherence": np.random.rand(),
                    "access_count": np.random.randint(0, 10)
                }
            )
            await service.store_memory(memory)
            memories.append(memory)
        
        # Score memories
        scored = await service.score_memories("test-session", "test-char")
        assert len(scored) == 10
        assert all("quality_score" in m for m in scored)
        
        # Prune low-quality memories
        pruned_count = await service.prune_memories(
            session_id="test-session",
            character_id="test-char",
            quality_threshold=0.3,
            max_memories=5
        )
        
        remaining = await service.count_memories("test-session", "test-char")
        assert remaining <= 5
    
    @pytest.mark.asyncio
    async def test_memory_head_processing(self):
        """Test real-time memory formation from model outputs"""
        service = MemoryService()
        
        # Simulate model output with memory head
        model_output = TripleHeadOutput(
            generation_text="I really enjoyed our conversation about art",
            control_tokens=[{"token": "<emotion_joy>", "probability": 0.9}],
            memory_vector=np.random.rand(768).tolist(),
            memory_metadata={
                "importance": 0.85,
                "emotional_valence": 0.8,
                "recency": 1.0,
                "coherence": 0.9
            }
        )
        
        # Process and store memory
        memory_id = await service.process_memory_head_output(
            output=model_output,
            session_id="test-session",
            character_id="test-char",
            context="User asked about favorite hobbies"
        )
        
        # Verify memory was formed correctly
        memory = await service.get_memory(memory_id)
        assert memory.content == model_output.generation_text
        assert memory.metadata["importance"] == 0.85
        assert memory.metadata["source"] == "memory_head"


class TestControlTokenProcessing:
    """Test control token processing system"""
    
    @pytest.mark.asyncio
    async def test_control_token_recognition(self):
        """Test real-time control token recognition"""
        processor = ControlTokenProcessor()
        
        output = TripleHeadOutput(
            generation_text="I'm feeling happy today! <emotion_happy>",
            control_tokens=[
                {"token": "<emotion_happy>", "probability": 0.9},
                {"token": "<action_smile>", "probability": 0.7}
            ],
            memory_vector=np.random.rand(768).tolist()
        )
        
        tokens = processor.extract_control_tokens(output)
        
        assert len(tokens) == 2
        assert tokens[0]["token"] == "<emotion_happy>"
        assert tokens[0]["probability"] == 0.9
        assert tokens[0]["type"] == "emotion"
    
    @pytest.mark.asyncio
    async def test_token_triggered_actions(self):
        """Test token-triggered action pipeline"""
        processor = ControlTokenProcessor()
        
        # Register action handlers
        smile_handler = AsyncMock()
        processor.register_action_handler("<action_smile>", smile_handler)
        
        # Process tokens
        tokens = [
            {"token": "<action_smile>", "probability": 0.8},
            {"token": "<emotion_happy>", "probability": 0.9}
        ]
        
        await processor.process_tokens(tokens, context={"session_id": "test"})
        
        # Verify handler was called
        smile_handler.assert_called_once()
        call_args = smile_handler.call_args[0][0]
        assert call_args["token"] == "<action_smile>"
        assert call_args["context"]["session_id"] == "test"
    
    @pytest.mark.asyncio
    async def test_ui_manipulation_commands(self):
        """Test UI manipulation through control tokens"""
        processor = ControlTokenProcessor()
        
        ui_commands = []
        
        async def ui_handler(event):
            ui_commands.append(event)
        
        processor.register_ui_handler(ui_handler)
        
        # Process UI control tokens
        tokens = [
            {"token": "<ui_highlight>user_message</ui_highlight>", "probability": 0.9},
            {"token": "<ui_show_image>memory_123.jpg</ui_show_image>", "probability": 0.8}
        ]
        
        await processor.process_tokens(tokens)
        
        assert len(ui_commands) == 2
        assert ui_commands[0]["action"] == "highlight"
        assert ui_commands[0]["target"] == "user_message"
        assert ui_commands[1]["action"] == "show_image"
        assert ui_commands[1]["resource"] == "memory_123.jpg"
    
    @pytest.mark.asyncio
    async def test_narrative_flow_control(self):
        """Test narrative flow control mechanisms"""
        processor = ControlTokenProcessor()
        
        flow_events = []
        
        async def flow_handler(event):
            flow_events.append(event)
        
        processor.register_flow_handler(flow_handler)
        
        # Process narrative control tokens
        tokens = [
            {"token": "<scene_change>mysterious_forest</scene_change>", "probability": 0.85},
            {"token": "<mood_shift>tense</mood_shift>", "probability": 0.9}
        ]
        
        await processor.process_tokens(tokens)
        
        assert len(flow_events) == 2
        assert flow_events[0]["type"] == "scene_change"
        assert flow_events[0]["target"] == "mysterious_forest"
        assert flow_events[1]["type"] == "mood_shift"
        assert flow_events[1]["mood"] == "tense"


class TestSessionStateManager:
    """Test session state management"""
    
    @pytest.mark.asyncio
    async def test_session_persistence(self):
        """Test persistent session state across requests"""
        manager = SessionStateManager()
        
        # Create session
        session_id = await manager.create_session(
            user_id="test-user",
            character_ids=["char1", "char2"],
            world_id="fantasy-world"
        )
        
        # Update session state
        await manager.update_state(session_id, {
            "conversation_turn": 5,
            "current_location": "tavern",
            "relationship_scores": {"char1": 0.7, "char2": 0.5}
        })
        
        # Retrieve state
        state = await manager.get_state(session_id)
        assert state["conversation_turn"] == 5
        assert state["current_location"] == "tavern"
        assert state["relationship_scores"]["char1"] == 0.7
    
    @pytest.mark.asyncio
    async def test_session_memory_management(self):
        """Test session-specific memory management"""
        manager = SessionStateManager()
        
        session_id = await manager.create_session("test-user", ["char1"])
        
        # Add memories to session
        memory_ids = []
        for i in range(5):
            memory_id = await manager.add_session_memory(
                session_id,
                character_id="char1",
                content=f"Memory {i}",
                embedding=np.random.rand(768).tolist()
            )
            memory_ids.append(memory_id)
        
        # Get session memories
        memories = await manager.get_session_memories(session_id, limit=3)
        assert len(memories) == 3
        
        # Clear old memories
        cleared = await manager.clear_old_memories(session_id, keep_recent=2)
        assert cleared == 3
    
    @pytest.mark.asyncio
    async def test_multi_character_coordination(self):
        """Test coordination between multiple characters in session"""
        manager = SessionStateManager()
        
        session_id = await manager.create_session(
            user_id="test-user",
            character_ids=["alice", "bob", "charlie"]
        )
        
        # Set character states
        await manager.set_character_state(session_id, "alice", {
            "mood": "happy",
            "location": "garden",
            "talking_to": ["bob"]
        })
        
        await manager.set_character_state(session_id, "bob", {
            "mood": "curious",
            "location": "garden",
            "talking_to": ["alice"]
        })
        
        # Get coordinated state
        scene_state = await manager.get_scene_state(session_id)
        assert len(scene_state["characters_present"]) == 2
        assert "alice" in scene_state["characters_present"]
        assert scene_state["location"] == "garden"
        assert len(scene_state["active_conversations"]) == 1
    
    @pytest.mark.asyncio
    async def test_session_analytics(self):
        """Test session analytics and optimization"""
        manager = SessionStateManager()
        
        session_id = await manager.create_session("test-user", ["char1"])
        
        # Simulate session activity
        for i in range(10):
            await manager.log_interaction(session_id, {
                "type": "message",
                "character_id": "char1",
                "response_time_ms": 150 + i * 10,
                "tokens_generated": 100 + i * 5
            })
        
        # Get analytics
        analytics = await manager.get_session_analytics(session_id)
        
        assert analytics["total_interactions"] == 10
        assert analytics["avg_response_time_ms"] < 200
        assert analytics["total_tokens_generated"] > 1000
        assert "peak_activity_time" in analytics
    
    @pytest.mark.asyncio
    async def test_clean_session_lifecycle(self):
        """Test clean session lifecycle management"""
        manager = SessionStateManager()
        
        # Create session
        session_id = await manager.create_session("test-user", ["char1"])
        assert await manager.is_session_active(session_id)
        
        # Pause session
        await manager.pause_session(session_id)
        state = await manager.get_state(session_id)
        assert state["status"] == "paused"
        
        # Resume session
        await manager.resume_session(session_id)
        assert await manager.is_session_active(session_id)
        
        # End session
        await manager.end_session(session_id)
        assert not await manager.is_session_active(session_id)
        
        # Verify cleanup
        with pytest.raises(SessionNotFoundError):
            await manager.get_state(session_id)


class TestPerformanceTargets:
    """Test performance targets are met"""
    
    @pytest.mark.asyncio
    async def test_inference_latency(self, inference_engine):
        """Test <200ms inference time target"""
        request = InferenceRequest(
            session_id="perf-test",
            character_id="test-char",
            prompt="Quick response test",
            max_tokens=50
        )
        
        times = []
        for _ in range(10):
            start = time.time()
            response = await inference_engine.generate(request)
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        assert avg_time < 200  # Average under 200ms
        assert p95_time < 300  # 95th percentile under 300ms
    
    @pytest.mark.asyncio
    async def test_concurrent_users(self, inference_engine):
        """Test 10+ concurrent users per GPU"""
        # Simulate 15 concurrent users
        users = []
        for i in range(15):
            user_session = {
                "session_id": f"user-{i}",
                "character_id": f"char-{i % 3}",  # 3 different characters
                "active": True
            }
            users.append(user_session)
        
        # Generate requests from all users
        async def user_interaction(user):
            for j in range(5):  # 5 messages per user
                request = InferenceRequest(
                    session_id=user["session_id"],
                    character_id=user["character_id"],
                    prompt=f"Message {j} from {user['session_id']}",
                    max_tokens=100
                )
                await inference_engine.generate(request)
                await asyncio.sleep(0.1)  # Simulate thinking time
        
        # Run all users concurrently
        start_time = time.time()
        await asyncio.gather(*[user_interaction(user) for user in users])
        total_time = time.time() - start_time
        
        # Should handle 15 users * 5 messages = 75 requests efficiently
        assert total_time < 10  # Should complete in under 10 seconds
        
        # Check system stayed healthy
        health = inference_engine.health_check()
        assert health["status"] in ["healthy", "warning"]
    
    @pytest.mark.asyncio
    async def test_adapter_swap_time(self):
        """Test <1 second adapter swap time"""
        manager = AdapterManager()
        
        # Pre-load adapter
        await manager.load_adapter("char1", "adapters/char1_v1.safetensors")
        
        # Time the swap
        start_time = time.time()
        await manager.hot_swap_adapter("char1", "adapters/char1_v2.safetensors")
        swap_time = time.time() - start_time
        
        assert swap_time < 1.0  # Under 1 second
    
    @pytest.mark.asyncio
    async def test_memory_formation_latency(self):
        """Test <50ms memory formation latency"""
        service = MemoryService()
        
        output = TripleHeadOutput(
            generation_text="Test memory formation",
            control_tokens=[],
            memory_vector=np.random.rand(768).tolist(),
            memory_metadata={"importance": 0.5}
        )
        
        times = []
        for _ in range(20):
            start = time.time()
            await service.process_memory_head_output(
                output=output,
                session_id="test",
                character_id="char1",
                context="test context"
            )
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
        
        avg_time = np.mean(times)
        assert avg_time < 50  # Under 50ms average
    
    @pytest.mark.asyncio
    async def test_session_memory_overhead(self):
        """Test <50MB memory overhead per session"""
        manager = SessionStateManager()
        
        # Create session with multiple characters
        session_id = await manager.create_session(
            user_id="test-user",
            character_ids=["char1", "char2", "char3"]
        )
        
        # Add reasonable amount of state
        for i in range(100):
            await manager.add_session_memory(
                session_id,
                character_id=f"char{i % 3 + 1}",
                content=f"Memory content {i}" * 10,  # ~100 chars
                embedding=np.random.rand(768).tolist()
            )
        
        # Measure memory usage
        memory_usage = await manager.get_session_memory_usage(session_id)
        
        assert memory_usage["total_mb"] < 50  # Under 50MB
        assert memory_usage["breakdown"]["embeddings_mb"] < 30  # Embeddings under 30MB
        assert memory_usage["breakdown"]["metadata_mb"] < 10  # Metadata under 10MB 