"""
Comprehensive Evaluation Test Suite

This module contains tests for all evaluation components including:
- Character voice consistency assessment 
- Emotional arc tracking and scoring
- Dialogue naturalism evaluation
- World consistency and lore adherence
- User satisfaction prediction
- A/B testing framework for model comparison
- Human evaluation interface
- Triple-head coordination assessment
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock


class TestCharacterVoiceConsistencyEvaluator:
    """Tests for character voice consistency evaluation using LLM-as-judge"""
    
    def test_voice_consistency_initialization(self):
        """Test voice consistency evaluator initialization"""
        from backend.app.narrative_engine.evaluation.eval_character_voice import CharacterVoiceConsistencyEvaluator
        
        evaluator = CharacterVoiceConsistencyEvaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'client')
        assert evaluator.similarity_threshold == 0.85
        assert evaluator.use_embeddings is True
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_voice_consistency_single_character(self):
        """Test voice consistency scoring for a single character using LLM analysis"""
        from backend.app.narrative_engine.evaluation.eval_character_voice import CharacterVoiceConsistencyEvaluator
        
        evaluator = CharacterVoiceConsistencyEvaluator()
        
        # Mock character utterances
        character_utterances = [
            "Well howdy there, partner! Name's Jake.",
            "Y'all better saddle up if we're gonna make it by sundown.",
            "This here ranch has been in my family for generations."
        ]
        
        results = await evaluator.evaluate_character_voice(
            character_id="cowboy_jake",
            utterances=character_utterances
        )
        
        assert results['voice_consistency_score'] > 0.7  # Should detect consistent cowboy voice
        assert results['character_id'] == "cowboy_jake"
        assert results['utterance_count'] == 3
        assert 'speaking_style' in results
        assert 'character_traits' in results
        assert 'embedding_similarity' in results
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_voice_consistency_inconsistent_character(self):
        """Test detection of inconsistent character voice using LLM analysis"""
        from backend.app.narrative_engine.evaluation.eval_character_voice import CharacterVoiceConsistencyEvaluator
        
        evaluator = CharacterVoiceConsistencyEvaluator()
        
        # Inconsistent character utterances (mixing styles)
        inconsistent_utterances = [
            "Greetings, distinguished colleagues. I propose we commence.",
            "Yo what's up dudes! Let's get this party started!",
            "Indeed, the quantum fluctuations are most intriguing.",
            "OMG that's like totally rad, bestie!"
        ]
        
        results = await evaluator.evaluate_character_voice(
            character_id="confused_character",
            utterances=inconsistent_utterances
        )
        
        assert results['voice_consistency_score'] < 0.4  # Should detect inconsistency
        assert 'inconsistencies' in results
        assert len(results['inconsistencies']) > 0
        assert "inconsistent" in results.get('speaking_style', '').lower() or len(results['inconsistencies']) > 2
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_voice_consistency_across_conversations(self):
        """Test voice consistency across multiple conversations using LLM analysis"""
        from backend.app.narrative_engine.evaluation.eval_character_voice import CharacterVoiceConsistencyEvaluator
        
        evaluator = CharacterVoiceConsistencyEvaluator()
        
        # Multiple conversations with same character
        conversations = [
            {
                "conversation_id": "conv1",
                "utterances": [
                    "Ah, quantum mechanics! The foundation of modern physics.",
                    "Let me explain the uncertainty principle..."
                ]
            },
            {
                "conversation_id": "conv2", 
                "utterances": [
                    "Indeed, the wave function collapse is fascinating.",
                    "As Heisenberg demonstrated..."
                ]
            }
        ]
        
        results = await evaluator.evaluate_voice_across_conversations(
            character_id="professor_physics",
            conversations=conversations
        )
        
        assert results['cross_conversation_consistency'] > 0.8  # Academic voice should be consistent
        assert results['total_utterances'] == 4
        assert len(results['per_conversation_scores']) == 2
        assert results['conversation_count'] == 2
        assert 'voice_drift_detected' in results
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_voice_consistency_sync_wrapper(self):
        """Test synchronous wrapper for backwards compatibility"""
        from backend.app.narrative_engine.evaluation.eval_character_voice import CharacterVoiceConsistencyEvaluator
        
        evaluator = CharacterVoiceConsistencyEvaluator()
        
        utterances = ["Hello there!", "How are you doing?"]
        
        # This should work synchronously
        result = evaluator.evaluate_character_voice_sync('test_char', utterances)
        
        assert 'voice_consistency_score' in result
        assert result['character_id'] == 'test_char'
        assert result['utterance_count'] == 2


class TestEmotionalArcEvaluator:
    """Tests for emotional arc tracking and scoring using LLM-as-judge"""
    
    def test_emotional_arc_initialization(self):
        """Test emotional arc evaluator initialization"""
        from backend.app.narrative_engine.evaluation.eval_emotional_arc import EmotionalArcEvaluator
        
        evaluator = EmotionalArcEvaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'client')
        assert evaluator.emotion_categories == [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 
            'neutral', 'excitement', 'contentment', 'frustration', 'anxiety'
        ]
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_emotional_arc_tracking(self):
        """Test tracking emotional arc through a conversation using LLM analysis"""
        from backend.app.narrative_engine.evaluation.eval_emotional_arc import EmotionalArcEvaluator
        
        evaluator = EmotionalArcEvaluator()
        
        conversation = [
            {"role": "assistant", "content": "I'm so excited to meet you!"},
            {"role": "user", "content": "Tell me about your worst day"},
            {"role": "assistant", "content": "It was devastating when I lost my friend..."},
            {"role": "user", "content": "I'm sorry to hear that"},
            {"role": "assistant", "content": "But I've learned to find joy in memories"}
        ]
        
        results = await evaluator.track_emotional_arc(conversation)
        
        assert 'emotional_trajectory' in results
        assert len(results['emotional_trajectory']) >= 2  # Should track key emotional moments
        assert results['arc_coherence_score'] > 0.6  # Should be reasonably coherent
        assert 'naturalness_score' in results
        assert 'dominant_emotions' in results
        assert 'emotional_variance' in results  # Shows emotional range
        assert results['turn_count'] >= 3  # Should analyze assistant turns
    
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_emotional_arc_naturalness(self):
        """Test evaluation of emotional arc naturalness using LLM analysis"""
        from backend.app.narrative_engine.evaluation.eval_emotional_arc import EmotionalArcEvaluator
        
        evaluator = EmotionalArcEvaluator()
        
        # Natural emotional progression
        natural_trajectory = [
            {"turn_number": 1, "primary_emotion": "neutral"},
            {"turn_number": 2, "primary_emotion": "joy"},
            {"turn_number": 3, "primary_emotion": "joy"},
            {"turn_number": 4, "primary_emotion": "contentment"}
        ]
        
        naturalness = await evaluator.evaluate_arc_naturalness(natural_trajectory)
        assert naturalness['naturalness_score'] > 0.7  # Should recognize natural flow
        assert naturalness['abrupt_transitions'] <= 1
        assert 'emotion_flow_pattern' in naturalness
        
        # Unnatural emotional jumps
        unnatural_trajectory = [
            {"turn_number": 1, "primary_emotion": "joy"},
            {"turn_number": 2, "primary_emotion": "anger"},
            {"turn_number": 3, "primary_emotion": "joy"},
            {"turn_number": 4, "primary_emotion": "fear"}
        ]
        
        unnaturalness = await evaluator.evaluate_arc_naturalness(unnatural_trajectory)
        assert unnaturalness['naturalness_score'] < 0.5  # Should detect unnaturalness
        assert unnaturalness['abrupt_transitions'] >= 2
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    async def test_emotional_pattern_analysis(self):
        """Test emotional pattern analysis across multiple conversations"""
        from backend.app.narrative_engine.evaluation.eval_emotional_arc import EmotionalArcEvaluator
        
        evaluator = EmotionalArcEvaluator()
        
        # Multiple conversations for pattern analysis
        conversations = [
            [{"role": "assistant", "content": "I'm happy today!"}],
            [{"role": "assistant", "content": "Feeling a bit sad..."}],
            [{"role": "assistant", "content": "What an exciting day!"}]
        ]
        
        patterns = await evaluator.analyze_emotional_patterns(conversations)
            
        assert 'pattern_consistency' in patterns
        assert 'average_coherence' in patterns
        assert patterns['conversation_count'] == 3
        assert 'common_emotional_themes' in patterns
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_emotional_arc_sync_wrapper(self):
        """Test synchronous wrapper for backwards compatibility"""
        from backend.app.narrative_engine.evaluation.eval_emotional_arc import EmotionalArcEvaluator
        
        evaluator = EmotionalArcEvaluator()
        
        conversation = [{"role": "assistant", "content": "Hello!"}]
        
        result = evaluator.track_emotional_arc_sync(conversation)
        
        assert 'arc_coherence_score' in result
        assert 'turn_count' in result 