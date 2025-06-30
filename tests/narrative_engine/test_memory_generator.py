"""
Tests for Memory Generation Pipeline

Tests the OpenAI-based memory generation for dataset augmentation.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
from datetime import datetime

from narrative_engine.memory_generator import (
    MemoryGenerator,
    MemoryGenerationRequest,
    GeneratedMemory,
    MemoryBatch,
    generate_memories_for_dataset,
)
from narrative_engine.memory_schema import Turn, MemoryAnnotation


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
    
    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI response with structured output"""
        mock_memory = GeneratedMemory(
            memory_content="The user complimented my name, which made me feel appreciated",
            turn_index=3,
            surprise_score=0.7,
            emotional_valence=0.8,
            importance=0.7,
            memory_type="emotional",
            importance_level="high",
            valence_category="positive",
            reasoning="Receiving a personal compliment early in conversation is memorable"
        )
        
        return MemoryBatch(
            memories=[mock_memory],
            overall_surprise=0.7,
            emotional_arc="Friendly introduction leading to warm appreciation"
        )
    
    @pytest.mark.asyncio
    async def test_generate_memories(self, sample_turns, sample_character, mock_openai_response):
        """Test basic memory generation"""
        # Arrange
        generator = MemoryGenerator()
        
        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_openai_response
        
        generator.client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        request = MemoryGenerationRequest(
            conversation_turns=sample_turns,
            character_profile=sample_character,
        )
        
        # Act
        result = await generator.generate_memories(request)
        
        # Assert
        assert len(result.memories) == 1
        memory = result.memories[0]
        
        assert memory.memory_content == "The user complimented my name, which made me feel appreciated"
        assert memory.surprise_score == 0.7
        assert memory.emotional_valence == 0.8
        assert memory.importance == 0.7
        assert memory.memory_type == "emotional"
        
        # Check Method A tokens
        assert "<memory_form>" in memory.method_a_tokens
        assert "<memory_importance_high>" in memory.method_a_tokens
        assert "<memory_type_emotional>" in memory.method_a_tokens
        assert "<memory_valence_positive>" in memory.method_a_tokens
        
        # Check Method B vector
        assert memory.method_b_vector is not None
        assert len(memory.method_b_vector) == 768
        assert memory.method_b_metadata is not None
        assert "decay_rate" in memory.method_b_metadata
    
    def test_token_mapping(self):
        """Test that token mappings are correct"""
        # Arrange
        with patch('narrative_engine.memory_generator.AsyncOpenAI'):
            generator = MemoryGenerator(api_key="test-key")
        
        # Assert
        assert generator.importance_token_map["high"] == "<memory_importance_high>"
        assert generator.valence_token_map["positive"] == "<memory_valence_positive>"
        assert generator.type_token_map["emotional"] == "<memory_type_emotional>"
    
    def test_pseudo_embedding_generation(self):
        """Test pseudo-embedding generation for Method B"""
        # Arrange
        with patch('narrative_engine.memory_generator.AsyncOpenAI'):
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
    
    @pytest.mark.asyncio
    async def test_method_a_only(self, sample_turns, sample_character, mock_openai_response):
        """Test generating only Method A tokens"""
        # Arrange
        generator = MemoryGenerator()
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_openai_response
        generator.client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        request = MemoryGenerationRequest(
            conversation_turns=sample_turns,
            character_profile=sample_character,
            include_method_a=True,
            include_method_b=False,
        )
        
        # Act
        result = await generator.generate_memories(request)
        
        # Assert
        memory = result.memories[0]
        assert len(memory.method_a_tokens) > 0
        assert memory.method_b_vector is None
        assert memory.method_b_metadata is None
    
    @pytest.mark.asyncio
    async def test_method_b_only(self, sample_turns, sample_character, mock_openai_response):
        """Test generating only Method B vectors"""
        # Arrange
        generator = MemoryGenerator()
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_openai_response
        generator.client.beta.chat.completions.parse = AsyncMock(return_value=mock_response)
        
        request = MemoryGenerationRequest(
            conversation_turns=sample_turns,
            character_profile=sample_character,
            include_method_a=False,
            include_method_b=True,
        )
        
        # Act
        result = await generator.generate_memories(request)
        
        # Assert
        memory = result.memories[0]
        assert len(memory.method_a_tokens) == 0
        assert memory.method_b_vector is not None
        assert len(memory.method_b_vector) == 768
        assert memory.method_b_metadata is not None
    
    def test_system_prompt_generation(self, sample_character):
        """Test system prompt includes character information"""
        # Arrange
        with patch('narrative_engine.memory_generator.AsyncOpenAI'):
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
        with patch('narrative_engine.memory_generator.AsyncOpenAI'):
            generator = MemoryGenerator(api_key="test-key")
        
        # Act
        prompt = generator._build_user_prompt(sample_turns)
        
        # Assert
        assert "USER: Hi! I'm Alex" in prompt
        assert "ASSISTANT: Hello Alex!" in prompt
        assert "emotional weight" in prompt
        assert "1-3 memories" in prompt


class TestMemoryDatasetGeneration:
    """Test dataset-level memory generation"""
    
    @pytest.mark.asyncio
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
        
        # Mock the generator
        with patch('narrative_engine.memory_generator.MemoryGenerator') as MockGenerator:
            mock_instance = MockGenerator.return_value
            
            # Create mock responses
            mock_batch1 = MagicMock()
            mock_batch1.memories = [
                MemoryAnnotation(
                    conversation_window=[Turn(sender="user", text="Hello!")],
                    memory_content="User greeted me",
                    surprise_score=0.3,
                    emotional_valence=0.5,
                    importance=0.4,
                    memory_type="episodic",
                    character_id="test_char",
                    method_a_tokens=["<memory_form>"],
                )
            ]
            mock_batch1.conversation_id = "conv_1"
            mock_batch1.to_training_format.return_value = {"test": "data1"}
            
            mock_batch2 = MagicMock()
            mock_batch2.memories = [
                MemoryAnnotation(
                    conversation_window=[Turn(sender="user", text="Tell me a joke")],
                    memory_content="User asked for humor",
                    surprise_score=0.5,
                    emotional_valence=0.6,
                    importance=0.5,
                    memory_type="episodic",
                    character_id="test_char",
                    method_a_tokens=["<memory_form>"],
                )
            ]
            mock_batch2.conversation_id = "conv_2"
            mock_batch2.to_training_format.return_value = {"test": "data2"}
            
            mock_instance.generate_memories = AsyncMock(side_effect=[mock_batch1, mock_batch2])
            
            # Act
            output_path = tmp_path / "memories.json"
            result = await generate_memories_for_dataset(
                conversations, 
                character,
                output_path
            )
            
            # Assert
            assert len(result) == 2
            assert output_path.exists()
            
            # Check saved file
            import json
            with open(output_path) as f:
                saved_data = json.load(f)
            
            assert saved_data["character_id"] == "test_char"
            assert saved_data["total_conversations"] == 2
            assert saved_data["total_memories"] == 2
            assert len(saved_data["batches"]) == 2 