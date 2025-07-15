"""
Tests for Memory Generation Pipeline

Tests the OpenAI-based memory generation for dataset augmentation.
"""

import pytest
import asyncio
from datetime import datetime

from backend.app.narrative_engine.memory_generator import (
    MemoryGenerator,
    MemoryGenerationRequest,
    GeneratedMemory,
    MemoryBatch,
    generate_memories_for_dataset,
)
from backend.app.narrative_engine.memory_schema import Turn, MemoryAnnotation


class TestMemoryGenerator:
    """Test the MemoryGenerator class"""
    
    @pytest.fixture
    def sample_turns(self):
        """Sample conversation turns for testing"""
        return [
            Turn(sender="user", text="Hi! I'm Alex. What's your name?"),
            Turn(sender="assistant", text="Hello Alex! I'm Clara. It's nice to meet you!"),
            Turn(sender="user", text="You have such a pretty name. It suits you."),
            Turn(sender="assistant", text="Oh... thank you! That's very kind of you to say."),
        ]
    
    @pytest.fixture
    def sample_character(self):
        """Sample character profile for testing"""
        return {
            "id": "clara_001",
            "name": "Clara",
            "personality_traits": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.4,
                "agreeableness": 0.9,
                "neuroticism": 0.3,
            },
            "relationship_status": {
                "familiarity": 0.1,
            }
        }
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_generate_memories(self, sample_turns, sample_character):
        """Test basic memory generation"""
        # Arrange
        generator = MemoryGenerator(api_key="test-key")
        
        request = MemoryGenerationRequest(
            conversation_turns=sample_turns,
            character_profile=sample_character,
        )
        
        # Act
        result = await generator.generate_memories(request)
        
        # Assert
        assert isinstance(result, MemoryBatch)
        assert len(result.memories) >= 1  # Should generate at least one memory
        assert len(result.memories) <= 5  # Reasonable upper bound
        
        for memory in result.memories:
            assert isinstance(memory, GeneratedMemory)
            assert isinstance(memory.memory_content, str)
            assert len(memory.memory_content.strip()) > 10  # Meaningful content
            assert 0.0 <= memory.surprise_score <= 1.0
            assert 0.0 <= memory.emotional_valence <= 1.0
            assert 0.0 <= memory.importance <= 1.0
            assert memory.memory_type in ["episodic", "emotional", "factual", "social"]
            assert memory.importance_level in ["low", "medium", "high"]
            assert memory.valence_category in ["negative", "neutral", "positive"]
            
            # Check Method A tokens are generated
            assert isinstance(memory.method_a_tokens, list)
            assert len(memory.method_a_tokens) > 0
            assert "<memory_form>" in memory.method_a_tokens
            
            # Check Method B vector is generated
            assert memory.method_b_vector is not None
            assert len(memory.method_b_vector) == 768
            assert memory.method_b_metadata is not None
            assert "decay_rate" in memory.method_b_metadata
        
        # Check overall batch properties
        assert 0.0 <= result.overall_surprise <= 1.0
        assert isinstance(result.emotional_arc, str)
        assert len(result.emotional_arc.strip()) > 5
    
    def test_token_mapping(self):
        """Test that token mappings are correct"""
        # Arrange
        generator = MemoryGenerator(api_key="test-key")
        
        # Assert
        assert generator.importance_token_map["high"] == "<memory_importance_high>"
        assert generator.valence_token_map["positive"] == "<memory_valence_positive>"
        assert generator.type_token_map["emotional"] == "<memory_type_emotional>"
    
    def test_pseudo_embedding_generation(self):
        """Test pseudo-embedding generation for Method B"""
        # Arrange
        generator = MemoryGenerator(api_key="test-key")
        memory = GeneratedMemory(
            memory_content="Test memory",
            turn_index=0,
            surprise_score=0.5,
            emotional_valence=0.8,
            importance=0.6,
            memory_type="emotional",
            importance_level="medium",
            valence_category="positive",
            reasoning="Test"
        )
        
        # Act
        embedding = generator._generate_pseudo_embedding(memory)
        
        # Assert
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)
        
        # Check normalization
        import numpy as np
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be unit vector
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_method_a_only(self, sample_turns, sample_character):
        """Test generating only Method A tokens"""
        # Arrange
        generator = MemoryGenerator(api_key="test-key")
        
        request = MemoryGenerationRequest(
            conversation_turns=sample_turns,
            character_profile=sample_character,
            include_method_a=True,
            include_method_b=False,
        )
        
        # Act
        result = await generator.generate_memories(request)
        
        # Assert
        assert len(result.memories) > 0
        for memory in result.memories:
            assert len(memory.method_a_tokens) > 0
            assert memory.method_b_vector is None
            assert memory.method_b_metadata is None
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_method_b_only(self, sample_turns, sample_character):
        """Test generating only Method B vectors"""
        # Arrange
        generator = MemoryGenerator(api_key="test-key")
        
        request = MemoryGenerationRequest(
            conversation_turns=sample_turns,
            character_profile=sample_character,
            include_method_a=False,
            include_method_b=True,
        )
        
        # Act
        result = await generator.generate_memories(request)
        
        # Assert
        assert len(result.memories) > 0
        for memory in result.memories:
            assert len(memory.method_a_tokens) == 0
            assert memory.method_b_vector is not None
            assert len(memory.method_b_vector) == 768
            assert memory.method_b_metadata is not None
    
    def test_system_prompt_generation(self, sample_character):
        """Test system prompt includes character information"""
        # Arrange
        generator = MemoryGenerator(api_key="test-key")
        
        # Act
        prompt = generator._build_system_prompt(sample_character)
        
        # Assert
        assert "Clara" in prompt
        assert "openness" in prompt.lower()
        assert "0.8" in prompt  # Openness value
        assert "cognitive psychology" in prompt
        assert "episodic" in prompt.lower()
    
    def test_user_prompt_generation(self, sample_turns):
        """Test user prompt includes conversation"""
        # Arrange
        generator = MemoryGenerator(api_key="test-key")
        
        # Act
        prompt = generator._build_user_prompt(sample_turns)
        
        # Assert
        assert "USER: Hi! I'm Alex" in prompt
        assert "ASSISTANT: Hello Alex!" in prompt
        assert "emotional weight" in prompt
        assert "1-3 memories" in prompt
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_memory_quality_and_relevance(self, sample_turns, sample_character):
        """Test that generated memories are relevant to the conversation"""
        # Arrange
        generator = MemoryGenerator(api_key="test-key")
        
        request = MemoryGenerationRequest(
            conversation_turns=sample_turns,
            character_profile=sample_character,
        )
        
        # Act
        result = await generator.generate_memories(request)
        
        # Assert
        memories_text = " ".join([mem.memory_content.lower() for mem in result.memories])
        
        # Should reference elements from the conversation
        conversation_elements = ["alex", "clara", "name", "compliment", "pretty", "meet"]
        assert any(element in memories_text for element in conversation_elements)
        
        # Should reflect character's personality (high agreeableness = positive response to compliment)
        assert any(keyword in memories_text for keyword in ["appreciate", "kind", "thank", "happy", "pleased"])


class TestMemoryDatasetGeneration:
    """Test dataset-level memory generation"""
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_generate_memories_for_dataset(self, tmp_path):
        """Test generating memories for multiple conversations"""
        # Arrange
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm doing well, thanks!"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Tell me a joke"},
                    {"role": "assistant", "content": "Why did the scarecrow win an award?"},
                    {"role": "user", "content": "I don't know, why?"},
                    {"role": "assistant", "content": "Because he was outstanding in his field!"},
                ]
            }
        ]
        
        character = {
            "id": "test_char",
            "name": "TestBot",
            "personality_traits": {"openness": 0.7}
        }
        
        # Act
        output_path = tmp_path / "memories.json"
        result = await generate_memories_for_dataset(
            conversations, 
            character,
            output_path
        )
        
        # Assert
        assert len(result) == 2  # Should process both conversations
        assert output_path.exists()
        
        # Check each batch
        for batch in result:
            assert isinstance(batch, MemoryBatch)
            assert len(batch.memories) > 0
            assert batch.conversation_id is not None
        
        # Check saved file
        import json
        with open(output_path) as f:
            saved_data = json.load(f)
        
        assert saved_data["character_id"] == "test_char"
        assert saved_data["total_conversations"] == 2
        assert saved_data["total_memories"] >= 2  # At least one memory per conversation
        assert len(saved_data["batches"]) == 2
        
        # Check that memories are relevant to their conversations
        for i, batch_data in enumerate(saved_data["batches"]):
            memories_text = " ".join([mem["memory_content"].lower() for mem in batch_data["memories"]])
            
            if i == 0:  # First conversation about greeting
                assert any(keyword in memories_text for keyword in ["hello", "greeting", "meet"])
            else:  # Second conversation about joke
                assert any(keyword in memories_text for keyword in ["joke", "funny", "humor", "scarecrow"])
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_memory_consistency_across_conversations(self, sample_character):
        """Test that memory generation is consistent across similar conversations"""
        # Arrange
        generator = MemoryGenerator(api_key="test-key")
        
        # Similar conversations
        turns1 = [
            Turn(sender="user", text="Hi, nice to meet you!"),
            Turn(sender="assistant", text="Hello! Nice to meet you too!"),
        ]
        
        turns2 = [
            Turn(sender="user", text="Hello there, good to see you!"),
            Turn(sender="assistant", text="Hi! Good to see you as well!"),
        ]
        
        request1 = MemoryGenerationRequest(conversation_turns=turns1, character_profile=sample_character)
        request2 = MemoryGenerationRequest(conversation_turns=turns2, character_profile=sample_character)
        
        # Act
        result1 = await generator.generate_memories(request1)
        result2 = await generator.generate_memories(request2)
        
        # Assert
        # Both should generate memories
        assert len(result1.memories) > 0 and len(result2.memories) > 0
        
        # Both should have similar characteristics for similar conversations
        all_memories = result1.memories + result2.memories
        memory_types = [mem.memory_type for mem in all_memories]
        
        # Should mostly be social/episodic memories for greetings
        assert any(mem_type in ["social", "episodic"] for mem_type in memory_types) 