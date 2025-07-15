"""
Tests for Memory Schema Pydantic Models

Following TDD principles to validate our memory data structures.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from backend.app.narrative_engine.memory_schema import (
    Turn,
    MemoryAnnotation,
    MemoryFormationEvent,
    MemoryQuery,
    MemoryRetrievalResult,
    MemoryTrainingBatch,
    EmotionalMomentumState,
)


class TestMemoryAnnotation:
    """Test the core MemoryAnnotation schema"""
    
    def test_valid_memory_annotation(self):
        """Test creating a valid memory annotation"""
        # Arrange
        turns = [
            Turn(sender="user", text="Tell me about yourself"),
            Turn(sender="assistant", text="I'm Clara, a curious soul who loves stories"),
        ]
        
        # Act
        memory = MemoryAnnotation(
            conversation_window=turns,
            memory_content="The user asked me to introduce myself",
            surprise_score=0.3,
            emotional_valence=0.5,
            importance=0.7,
            memory_type="episodic",
            character_id="clara_001",
            familiarity_score=0.2,
            method_a_tokens=["<memory_form>", "<importance_high>"],
            method_b_vector=[0.1] * 768,  # 768-dim vector
        )
        
        # Assert
        assert memory.memory_content == "The user asked me to introduce myself"
        assert memory.surprise_score == 0.3
        assert len(memory.method_b_vector) == 768
        assert memory.decay_rate == 0.1  # Default value
    
    def test_invalid_surprise_score(self):
        """Test that surprise score must be between 0 and 1"""
        # Arrange
        turns = [Turn(sender="user", text="Hello")]
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            MemoryAnnotation(
                conversation_window=turns,
                memory_content="Test memory",
                surprise_score=1.5,  # Invalid: > 1
                emotional_valence=0,
                importance=0.5,
                memory_type="episodic",
                character_id="test_001",
            )
        
        assert "less than or equal to 1" in str(exc_info.value)
    
    def test_invalid_emotional_valence(self):
        """Test that emotional valence must be between -1 and 1"""
        # Arrange
        turns = [Turn(sender="user", text="Hello")]
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            MemoryAnnotation(
                conversation_window=turns,
                memory_content="Test memory",
                surprise_score=0.5,
                emotional_valence=-2,  # Invalid: < -1
                importance=0.5,
                memory_type="episodic",
                character_id="test_001",
            )
        
        assert "greater than or equal to -1" in str(exc_info.value)
    
    def test_invalid_token_format(self):
        """Test that Method A tokens must be in <token> format"""
        # Arrange
        turns = [Turn(sender="user", text="Hello")]
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            MemoryAnnotation(
                conversation_window=turns,
                memory_content="Test memory",
                surprise_score=0.5,
                emotional_valence=0,
                importance=0.5,
                memory_type="episodic",
                character_id="test_001",
                method_a_tokens=["memory_form", "importance_high"],  # Invalid: missing < >
            )
        
        assert "Control token must be in format <token>" in str(exc_info.value)
    
    def test_invalid_vector_dimension(self):
        """Test that Method B vector must be 768-dimensional"""
        # Arrange
        turns = [Turn(sender="user", text="Hello")]
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            MemoryAnnotation(
                conversation_window=turns,
                memory_content="Test memory",
                surprise_score=0.5,
                emotional_valence=0,
                importance=0.5,
                memory_type="episodic",
                character_id="test_001",
                method_b_vector=[0.1] * 512,  # Invalid: wrong dimension
            )
        
        assert "Memory vector must be 768-dimensional" in str(exc_info.value)


class TestMemoryFormationEvent:
    """Test the MemoryFormationEvent schema for real-time visualization"""
    
    def test_auto_color_generation(self):
        """Test that bubble color is auto-generated from emotional valence"""
        # Arrange
        memory = MemoryAnnotation(
            conversation_window=[Turn(sender="user", text="I love you")],
            memory_content="User expressed love",
            surprise_score=0.9,
            emotional_valence=0.8,  # Positive
            importance=0.9,
            memory_type="emotional",
            character_id="clara_001",
        )
        
        # Act
        event = MemoryFormationEvent(
            session_id="session_123",
            character_id="clara_001",
            memory=memory,
            bubble_color="",  # Empty to trigger auto-generation
            bubble_size=50,
        )
        
        # Assert
        # Valence 0.8 should map to greenish hue
        assert "hsl(" in event.bubble_color
        assert "108" in event.bubble_color  # Approximately 60 * (0.8 + 1) = 108
    
    def test_custom_bubble_properties(self):
        """Test setting custom visualization properties"""
        # Arrange
        memory = MemoryAnnotation(
            conversation_window=[Turn(sender="user", text="Boo!")],
            memory_content="User scared me",
            surprise_score=1.0,
            emotional_valence=-0.5,
            importance=0.8,
            memory_type="episodic",
            character_id="clara_001",
        )
        
        # Act
        event = MemoryFormationEvent(
            session_id="session_123",
            character_id="clara_001",
            memory=memory,
            bubble_color="hsl(0, 100%, 50%)",  # Red
            bubble_size=80,
            animation_duration=3.5,
        )
        
        # Assert
        assert event.bubble_color == "hsl(0, 100%, 50%)"
        assert event.bubble_size == 80
        assert event.animation_duration == 3.5


class TestMemoryQuery:
    """Test the MemoryQuery schema for retrieval operations"""
    
    def test_default_query_parameters(self):
        """Test that query has sensible defaults"""
        # Act
        query = MemoryQuery(
            session_id="session_123",
            query_text="What did we talk about earlier?",
        )
        
        # Assert
        assert query.k == 5
        assert query.recency_weight == 0.3
        assert query.importance_threshold == 0.1
        assert query.memory_types is None  # All types
    
    def test_filtered_memory_query(self):
        """Test querying specific memory types"""
        # Act
        query = MemoryQuery(
            session_id="session_123",
            query_text="What emotions have I felt?",
            memory_types=["emotional", "episodic"],
            k=10,
            recency_weight=0.1,  # Care less about recency
            importance_threshold=0.5,  # Only important memories
        )
        
        # Assert
        assert query.memory_types == ["emotional", "episodic"]
        assert query.k == 10
        assert query.importance_threshold == 0.5


class TestEmotionalMomentumState:
    """Test the EmotionalMomentumState schema for tracking emotions"""
    
    def test_visualization_data_generation(self):
        """Test generating visualization data for Director's View"""
        # Arrange
        state = EmotionalMomentumState(
            active_emotions={
                "<mood_happy_1>": 0.8,
                "<mood_nervous_1>": 0.3,
            },
            decay_states={
                "<mood_happy_1>": {
                    "strength": 0.8,
                    "decay_rate": 0.1,
                    "turns_remaining": 3,
                },
            },
            recirculation_tokens=["<mood_happy_1>"],
            surprise_history=[0.2, 0.8, 0.5, 0.9],
        )
        
        # Act
        viz_data = state.get_visualization_data()
        
        # Assert
        assert len(viz_data['emotions']) == 2
        assert viz_data['emotions'][0]['token'] == "<mood_happy_1>"
        assert viz_data['emotions'][0]['strength'] == 0.8
        assert 'hsl(45' in viz_data['emotions'][0]['color']  # Yellow for happy
        assert 0.5 < float(viz_data['average_surprise']) < 0.61  # Convert numpy float for comparison
        assert viz_data['emotional_volatility'] > 0
    
    def test_emotion_color_mapping(self):
        """Test that emotions map to appropriate colors"""
        # Arrange
        test_cases = [
            ("<mood_happy_1>", "hsl(45"),    # Yellow
            ("<mood_sad_2>", "hsl(210"),     # Blue
            ("<mood_angry_1>", "hsl(0"),     # Red
            ("<feeling_love>", "hsl(330"),   # Pink
            ("<unknown_token>", "hsl(0, 0%"), # Gray
        ]
        
        # Act & Assert
        for token, expected_color in test_cases:
            color = EmotionalMomentumState._emotion_to_color(token)
            assert expected_color in color


class TestMemoryTrainingBatch:
    """Test the MemoryTrainingBatch schema for dataset augmentation"""
    
    def test_training_format_conversion(self):
        """Test converting batch to training format"""
        # Arrange
        memories = [
            MemoryAnnotation(
                conversation_window=[Turn(sender="user", text="Hi")],
                memory_content="First meeting",
                surprise_score=0.8,
                emotional_valence=0.5,
                importance=0.9,
                memory_type="episodic",
                character_id="clara_001",
                method_a_tokens=["<memory_form>", "<importance_high>"],
                method_b_vector=[0.1] * 768,
            ),
            MemoryAnnotation(
                conversation_window=[Turn(sender="user", text="Bye")],
                memory_content="User leaving",
                surprise_score=0.3,
                emotional_valence=-0.2,
                importance=0.5,
                memory_type="episodic",
                character_id="clara_001",
                method_a_tokens=["<memory_form>", "<importance_medium>"],
                method_b_vector=[0.2] * 768,
            ),
        ]
        
        batch = MemoryTrainingBatch(
            conversation_id="conv_123",
            memories=memories,
            method_a_sequence=[
                {"position": "after_turn_1", "tokens": ["<memory_form>", "<importance_high>"]},
                {"position": "after_turn_3", "tokens": ["<memory_form>", "<importance_medium>"]},
            ],
            method_b_targets=[
                {"vector": [0.1] * 768, "metadata": {"decay_rate": 0.1}},
                {"vector": [0.2] * 768, "metadata": {"decay_rate": 0.2}},
            ],
        )
        
        # Act
        training_data = batch.to_training_format()
        
        # Assert
        assert training_data['conversation_id'] == "conv_123"
        assert training_data['memory_count'] == 2
        assert len(training_data['method_a_tokens']) == 2
        assert training_data['importance_scores'] == [0.9, 0.5]
        assert training_data['surprise_scores'] == [0.8, 0.3] 