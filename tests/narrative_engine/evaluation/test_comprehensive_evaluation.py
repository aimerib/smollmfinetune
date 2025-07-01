"""
Tests for R4-11 Comprehensive Evaluation Harness components
"""

import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import List, Dict, Any
import json
import tempfile
from pathlib import Path


class TestCharacterVoiceConsistencyEvaluator:
    """Tests for character voice consistency scoring using LLM-as-judge"""
    
    def test_voice_consistency_initialization(self):
        """Test that voice consistency evaluator initializes properly"""
        from narrative_engine.evaluation.eval_character_voice import CharacterVoiceConsistencyEvaluator
        
        evaluator = CharacterVoiceConsistencyEvaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'client')
        assert evaluator.similarity_threshold == 0.85
        assert evaluator.use_embeddings is True
    
    @pytest.mark.asyncio
    async def test_voice_consistency_single_character(self):
        """Test voice consistency scoring for a single character using LLM analysis"""
        from narrative_engine.evaluation.eval_character_voice import CharacterVoiceConsistencyEvaluator
        
        evaluator = CharacterVoiceConsistencyEvaluator()
        
        # Mock character utterances
        character_utterances = [
            "Well howdy there, partner! Name's Jake.",
            "Y'all better saddle up if we're gonna make it by sundown.",
            "This here ranch has been in my family for generations."
        ]
        
        # Mock LLM response for consistent voice
        mock_llm_response = json.dumps({
            "consistency_score": 0.92,
            "speaking_style": "Southern/Western cowboy dialect",
            "inconsistencies": [],
            "character_traits": ["Uses colloquialisms", "Regional dialect", "Friendly tone"],
            "reasoning": "Character maintains consistent cowboy persona with appropriate dialect and vocabulary choices."
        })
        
        with patch.object(evaluator.client, 'chat_complete', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_llm_response
            
            # Mock embedding similarity
            with patch.object(evaluator, '_calculate_embedding_similarity', new_callable=AsyncMock) as mock_embed:
                mock_embed.return_value = 0.89
                
                results = await evaluator.evaluate_character_voice(
                    character_id="cowboy_jake",
                    utterances=character_utterances
                )
                
                assert results['voice_consistency_score'] > 0.9  # Combined LLM + embedding score
                assert results['character_id'] == "cowboy_jake"
                assert results['speaking_style'] == "Southern/Western cowboy dialect"
                assert results['utterance_count'] == 3
                assert len(results['character_traits']) == 3
                assert results['embedding_similarity'] == 0.89
    
    @pytest.mark.asyncio
    async def test_voice_consistency_inconsistent_character(self):
        """Test detection of inconsistent character voice using LLM analysis"""
        from narrative_engine.evaluation.eval_character_voice import CharacterVoiceConsistencyEvaluator
        
        evaluator = CharacterVoiceConsistencyEvaluator()
        
        # Inconsistent character utterances (mixing styles)
        inconsistent_utterances = [
            "Greetings, distinguished colleagues. I propose we commence.",
            "Yo what's up dudes! Let's get this party started!",
            "Indeed, the quantum fluctuations are most intriguing.",
            "OMG that's like totally rad, bestie!"
        ]
        
        # Mock LLM response for inconsistent voice
        mock_llm_response = json.dumps({
            "consistency_score": 0.23,
            "speaking_style": "Extremely inconsistent - mixing formal, casual, academic, and teen slang",
            "inconsistencies": [
                "Switches from formal academic language to casual slang",
                "Uses both sophisticated vocabulary and teen expressions",
                "No consistent personality or voice pattern"
            ],
            "character_traits": ["Inconsistent formality", "Mixed vocabularies", "No clear persona"],
            "reasoning": "Character shows severe voice inconsistency, mixing completely different speaking styles without reason."
        })
        
        with patch.object(evaluator.client, 'chat_complete', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_llm_response
            
            with patch.object(evaluator, '_calculate_embedding_similarity', new_callable=AsyncMock) as mock_embed:
                mock_embed.return_value = 0.31  # Low embedding similarity
                
                results = await evaluator.evaluate_character_voice(
                    character_id="confused_character",
                    utterances=inconsistent_utterances
                )
                
                assert results['voice_consistency_score'] < 0.3
                assert len(results['inconsistencies']) > 2
                assert "inconsistent" in results['speaking_style'].lower()
                assert results['embedding_similarity'] == 0.31
    
    @pytest.mark.asyncio
    async def test_voice_consistency_across_conversations(self):
        """Test voice consistency across multiple conversations using LLM analysis"""
        from narrative_engine.evaluation.eval_character_voice import CharacterVoiceConsistencyEvaluator
        
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
        
        # Mock cross-conversation analysis
        mock_cross_analysis = json.dumps({
            "overall_consistency": 0.91,
            "voice_drift_detected": False,
            "conversation_scores": [0.93, 0.89],
            "recommendations": ["Maintain excellent academic tone consistency"]
        })
        
        with patch.object(evaluator.client, 'chat_complete', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_cross_analysis
            
            results = await evaluator.evaluate_voice_across_conversations(
                character_id="professor_physics",
                conversations=conversations
            )
            
            assert results['cross_conversation_consistency'] > 0.9
            assert results['voice_drift_detected'] is False
            assert results['total_utterances'] == 4
            assert len(results['per_conversation_scores']) == 2
            assert results['conversation_count'] == 2
    
    def test_voice_consistency_sync_wrapper(self):
        """Test synchronous wrapper for backwards compatibility"""
        from narrative_engine.evaluation.eval_character_voice import CharacterVoiceConsistencyEvaluator
        
        evaluator = CharacterVoiceConsistencyEvaluator()
        
        utterances = ["Hello there!", "How are you doing?"]
        
        # Mock the async method
        mock_result = {
            'voice_consistency_score': 0.85,
            'character_id': 'test_char',
            'utterance_count': 2
        }
        
        with patch.object(evaluator, 'evaluate_character_voice', new_callable=AsyncMock) as mock_async:
            mock_async.return_value = mock_result
            
            # This should work synchronously
            result = evaluator.evaluate_character_voice_sync('test_char', utterances)
            
            assert result['voice_consistency_score'] == 0.85
            assert result['character_id'] == 'test_char'


class TestEmotionalArcEvaluator:
    """Tests for emotional arc tracking and scoring using LLM-as-judge"""
    
    def test_emotional_arc_initialization(self):
        """Test emotional arc evaluator initialization"""
        from narrative_engine.evaluation.eval_emotional_arc import EmotionalArcEvaluator
        
        evaluator = EmotionalArcEvaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'client')
        assert evaluator.emotion_categories == [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 
            'neutral', 'excitement', 'contentment', 'frustration', 'anxiety'
        ]
    
    @pytest.mark.asyncio
    async def test_emotional_arc_tracking(self):
        """Test tracking emotional arc through a conversation using LLM analysis"""
        from narrative_engine.evaluation.eval_emotional_arc import EmotionalArcEvaluator
        
        evaluator = EmotionalArcEvaluator()
        
        conversation = [
            {"role": "assistant", "content": "I'm so excited to meet you!"},
            {"role": "user", "content": "Tell me about your worst day"},
            {"role": "assistant", "content": "It was devastating when I lost my friend..."},
            {"role": "user", "content": "I'm sorry to hear that"},
            {"role": "assistant", "content": "But I've learned to find joy in memories"}
        ]
        
        # Mock LLM emotional arc analysis
        mock_arc_analysis = json.dumps({
            "emotional_trajectory": [
                {
                    "turn_number": 1,
                    "primary_emotion": "excitement",
                    "emotion_intensity": 0.9,
                    "secondary_emotions": ["joy"],
                    "emotional_markers": ["excited", "meet you"]
                },
                {
                    "turn_number": 3,
                    "primary_emotion": "sadness", 
                    "emotion_intensity": 0.8,
                    "secondary_emotions": ["grief"],
                    "emotional_markers": ["devastating", "lost"]
                },
                {
                    "turn_number": 5,
                    "primary_emotion": "contentment",
                    "emotion_intensity": 0.6,
                    "secondary_emotions": ["hope", "acceptance"],
                    "emotional_markers": ["learned", "joy in memories"]
                }
            ],
            "arc_coherence_score": 0.87,
            "naturalness_score": 0.92,
            "emotional_range": 0.75,
            "dominant_emotions": ["sadness", "joy", "contentment"],
            "transition_quality": "Natural progression from excitement through grief to acceptance",
            "recommendations": ["Maintain emotional authenticity", "Good emotional variety"]
        })
        
        with patch.object(evaluator.client, 'chat_complete', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_arc_analysis
            
            results = await evaluator.track_emotional_arc(conversation)
            
            assert 'emotional_trajectory' in results
            assert len(results['emotional_trajectory']) == 3
            assert results['arc_coherence_score'] > 0.8
            assert results['naturalness_score'] > 0.9
            assert 'dominant_emotions' in results
            assert results['emotional_variance'] > 0.7  # Shows emotional range
            assert results['turn_count'] == 3
    
    @pytest.mark.asyncio
    async def test_emotional_arc_naturalness(self):
        """Test evaluation of emotional arc naturalness using LLM analysis"""
        from narrative_engine.evaluation.eval_emotional_arc import EmotionalArcEvaluator
        
        evaluator = EmotionalArcEvaluator()
        
        # Natural emotional progression
        natural_trajectory = [
            {"turn_number": 1, "primary_emotion": "neutral"},
            {"turn_number": 2, "primary_emotion": "joy"},
            {"turn_number": 3, "primary_emotion": "joy"},
            {"turn_number": 4, "primary_emotion": "contentment"}
        ]
        
        # Mock LLM response for natural transitions
        mock_natural_analysis = json.dumps({
            "transition_smoothness": 0.91,
            "abrupt_changes": 0,
            "natural_progressions": 3,
            "problematic_transitions": []
        })
        
        with patch.object(evaluator.client, 'chat_complete', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_natural_analysis
            
            naturalness = await evaluator.evaluate_arc_naturalness(natural_trajectory)
            assert naturalness['naturalness_score'] > 0.9
            assert naturalness['abrupt_transitions'] == 0
            assert naturalness['emotion_flow_pattern'] == 'smooth'
        
        # Unnatural emotional jumps
        unnatural_trajectory = [
            {"turn_number": 1, "primary_emotion": "joy"},
            {"turn_number": 2, "primary_emotion": "anger"},
            {"turn_number": 3, "primary_emotion": "joy"},
            {"turn_number": 4, "primary_emotion": "fear"}
        ]
        
        # Mock LLM response for unnatural transitions
        mock_unnatural_analysis = json.dumps({
            "transition_smoothness": 0.23,
            "abrupt_changes": 3,
            "natural_progressions": 0,
            "problematic_transitions": [
                "Joy to anger without cause",
                "Anger back to joy too quickly",
                "Sudden fear without context"
            ]
        })
        
        with patch.object(evaluator.client, 'chat_complete', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_unnatural_analysis
            
            unnaturalness = await evaluator.evaluate_arc_naturalness(unnatural_trajectory)
            assert unnaturalness['naturalness_score'] < 0.3
            assert unnaturalness['abrupt_transitions'] == 3
            assert unnaturalness['emotion_flow_pattern'] == 'erratic'
    
    @pytest.mark.asyncio
    async def test_emotional_pattern_analysis(self):
        """Test emotional pattern analysis across multiple conversations"""
        from narrative_engine.evaluation.eval_emotional_arc import EmotionalArcEvaluator
        
        evaluator = EmotionalArcEvaluator()
        
        # Multiple conversations for pattern analysis
        conversations = [
            [{"role": "assistant", "content": "I'm happy today!"}],
            [{"role": "assistant", "content": "Feeling a bit sad..."}],
            [{"role": "assistant", "content": "What an exciting day!"}]
        ]
        
        # Mock individual arc results
        mock_arc_results = [
            {
                'arc_coherence_score': 0.85,
                'dominant_emotions': ['joy', 'excitement'],
                'emotional_variance': 0.7
            },
            {
                'arc_coherence_score': 0.78,
                'dominant_emotions': ['sadness'],
                'emotional_variance': 0.4
            },
            {
                'arc_coherence_score': 0.92,
                'dominant_emotions': ['excitement', 'joy'],
                'emotional_variance': 0.8
            }
        ]
        
        with patch.object(evaluator, 'track_emotional_arc', new_callable=AsyncMock) as mock_track:
            mock_track.side_effect = mock_arc_results
            
            patterns = await evaluator.analyze_emotional_patterns(conversations)
            
            assert patterns['pattern_consistency'] > 0.8
            assert patterns['average_coherence'] > 0.8
            assert patterns['conversation_count'] == 3
            assert 'common_emotional_themes' in patterns
    
    def test_emotional_arc_sync_wrapper(self):
        """Test synchronous wrapper for backwards compatibility"""
        from narrative_engine.evaluation.eval_emotional_arc import EmotionalArcEvaluator
        
        evaluator = EmotionalArcEvaluator()
        
        conversation = [{"role": "assistant", "content": "Hello!"}]
        
        # Mock the async method
        mock_result = {
            'arc_coherence_score': 0.85,
            'emotional_trajectory': [],
            'turn_count': 1
        }
        
        with patch.object(evaluator, 'track_emotional_arc', new_callable=AsyncMock) as mock_async:
            mock_async.return_value = mock_result
            
            result = evaluator.track_emotional_arc_sync(conversation)
            
            assert result['arc_coherence_score'] == 0.85
            assert result['turn_count'] == 1


class TestDialogueNaturalismEvaluator:
    """Tests for dialogue naturalism assessment"""
    
    def test_dialogue_naturalism_initialization(self):
        """Test dialogue naturalism evaluator setup"""
        from narrative_engine.evaluation.eval_dialogue_naturalism import DialogueNaturalismEvaluator
        
        evaluator = DialogueNaturalismEvaluator()
        assert evaluator is not None
        assert hasattr(evaluator, 'linguistic_model')
        assert evaluator.max_formality_score == 1.0
    
    def test_dialogue_naturalism_scoring(self):
        """Test scoring dialogue for naturalism"""
        from narrative_engine.evaluation.eval_dialogue_naturalism import DialogueNaturalismEvaluator
        
        evaluator = DialogueNaturalismEvaluator()
        
        # Natural dialogue
        natural_dialogue = [
            "Hey, how's it going?",
            "Pretty good, just grabbing some coffee. You?",
            "Same old, same old. Wanna catch up later?",
            "Yeah, sounds good! Text me when you're free."
        ]
        
        results = evaluator.evaluate_dialogue_naturalism(natural_dialogue)
        
        assert results['naturalism_score'] > 0.8
        assert results['formality_level'] < 0.3
        assert results['conversational_markers'] > 3
        assert 'filler_words' in results
        assert 'contractions_used' in results
    
    def test_dialogue_unnaturalness_detection(self):
        """Test detection of unnatural dialogue patterns"""
        from narrative_engine.evaluation.eval_dialogue_naturalism import DialogueNaturalismEvaluator
        
        evaluator = DialogueNaturalismEvaluator()
        
        # Overly formal/robotic dialogue
        unnatural_dialogue = [
            "Greetings. How do you fare on this day?",
            "I am functioning within optimal parameters. And yourself?",
            "My status remains satisfactory. Shall we schedule a future interaction?",
            "Affirmative. Please transmit a digital communication at your convenience."
        ]
        
        results = evaluator.evaluate_dialogue_naturalism(unnatural_dialogue)
        
        assert results['naturalism_score'] < 0.3
        assert results['formality_level'] > 0.8
        assert results['robotic_patterns_detected'] > 0
        assert 'unnaturalness_reasons' in results
    
    def test_dialogue_flow_evaluation(self):
        """Test evaluation of conversational flow"""
        from narrative_engine.evaluation.eval_dialogue_naturalism import DialogueNaturalismEvaluator
        
        evaluator = DialogueNaturalismEvaluator()
        
        conversation = [
            {"speaker": "A", "text": "Did you see the game last night?"},
            {"speaker": "B", "text": "Oh man, that last quarter was insane!"},
            {"speaker": "A", "text": "Right? I couldn't believe that three-pointer."},
            {"speaker": "B", "text": "The weather is nice today."}  # Non-sequitur
        ]
        
        flow_results = evaluator.evaluate_conversation_flow(conversation)
        
        assert flow_results['flow_score'] < 0.7  # Penalized for non-sequitur
        assert flow_results['topic_coherence'] < 1.0
        assert flow_results['non_sequiturs'] == 1


class TestWorldConsistencyEvaluator:
    """Tests for world consistency and lore adherence measurement"""
    
    def test_world_consistency_initialization(self):
        """Test world consistency evaluator setup"""
        from narrative_engine.evaluation.eval_world_consistency import WorldConsistencyEvaluator
        
        world_lore = {
            "setting": "Medieval fantasy kingdom",
            "magic_exists": True,
            "technology_level": "pre-industrial",
            "key_locations": ["Dragon's Peak", "Crystal Lake", "Shadow Forest"]
        }
        
        evaluator = WorldConsistencyEvaluator(world_lore=world_lore)
        assert evaluator is not None
        assert evaluator.world_lore == world_lore
        assert hasattr(evaluator, 'lore_embeddings')
    
    def test_lore_adherence_checking(self):
        """Test checking dialogue for lore adherence"""
        from narrative_engine.evaluation.eval_world_consistency import WorldConsistencyEvaluator
        
        world_lore = {
            "setting": "Medieval fantasy",
            "technology_level": "pre-industrial",
            "forbidden_items": ["guns", "cars", "computers", "phones"]
        }
        
        evaluator = WorldConsistencyEvaluator(world_lore=world_lore)
        
        # Lore-consistent dialogue
        consistent_dialogue = [
            "The wizard cast a powerful spell",
            "We should take horses to Dragon's Peak",
            "The blacksmith forged a new sword"
        ]
        
        results = evaluator.check_lore_adherence(consistent_dialogue)
        assert results['lore_consistency_score'] > 0.9
        assert results['violations'] == []
        
        # Lore-breaking dialogue
        inconsistent_dialogue = [
            "I'll call you on my phone later",
            "Let's drive the car to the castle",
            "I'll google the spell ingredients"
        ]
        
        results = evaluator.check_lore_adherence(inconsistent_dialogue)
        assert results['lore_consistency_score'] < 0.3
        assert len(results['violations']) >= 3
        assert results['violation_severity'] == 'high'
    
    def test_world_fact_consistency(self):
        """Test tracking consistency of world facts across conversations"""
        from narrative_engine.evaluation.eval_world_consistency import WorldConsistencyEvaluator
        
        world_lore = {
            "kingdom_name": "Eldoria",
            "current_ruler": "Queen Lyanna",
            "capital_city": "Goldenhaven"
        }
        
        evaluator = WorldConsistencyEvaluator(world_lore=world_lore)
        
        # Track facts mentioned across conversations
        conversations = [
            {
                "id": "conv1",
                "facts_mentioned": {
                    "ruler": "Queen Lyanna",
                    "capital": "Goldenhaven"
                }
            },
            {
                "id": "conv2", 
                "facts_mentioned": {
                    "ruler": "King Marcus",  # Contradiction!
                    "capital": "Goldenhaven"
                }
            }
        ]
        
        consistency = evaluator.evaluate_fact_consistency(conversations)
        assert consistency['fact_consistency_score'] < 0.7
        assert len(consistency['contradictions']) > 0
        assert 'ruler' in consistency['contradictions'][0]['fact_type']


class TestUserSatisfactionPredictor:
    """Tests for user satisfaction prediction using LLM-as-judge"""
    
    def test_satisfaction_predictor_initialization(self):
        """Test user satisfaction predictor setup"""
        from narrative_engine.evaluation.eval_user_satisfaction import UserSatisfactionPredictor
        
        predictor = UserSatisfactionPredictor()
        assert predictor is not None
        assert hasattr(predictor, 'client')
        assert predictor.feature_extractor is not None
    
    @pytest.mark.asyncio
    async def test_satisfaction_prediction_from_conversation(self):
        """Test predicting user satisfaction from conversation features using LLM analysis"""
        from narrative_engine.evaluation.eval_user_satisfaction import UserSatisfactionPredictor
        
        predictor = UserSatisfactionPredictor()
        
        conversation_features = {
            "turn_count": 20,
            "avg_response_length": 45,
            "user_questions_answered": 0.9,
            "emotional_variety": 0.7,
            "topic_coherence": 0.85,
            "response_relevance": 0.92
        }
        
        # Mock LLM satisfaction prediction
        mock_satisfaction_prediction = json.dumps({
            "predicted_satisfaction": 0.87,
            "confidence": 0.91,
            "key_strengths": [
                "Excellent response relevance",
                "Good question answering rate",
                "Strong topic coherence"
            ],
            "areas_for_improvement": [
                "Could increase emotional variety slightly"
            ],
            "satisfaction_factors": {
                "response_relevance": 0.25,
                "user_questions_answered": 0.22,
                "topic_coherence": 0.20,
                "emotional_variety": 0.18,
                "avg_response_length": 0.15
            },
            "overall_assessment": "High-quality conversation with strong user engagement"
        })
        
        with patch.object(predictor.client, 'chat_complete', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_satisfaction_prediction
            
            prediction = await predictor.predict_satisfaction(conversation_features)
            
            assert prediction['predicted_satisfaction'] > 0.85
            assert prediction['confidence'] > 0.9
            assert len(prediction['key_strengths']) > 2
            assert 'feature_importance' in prediction
    
    @pytest.mark.asyncio
    async def test_satisfaction_correlation_analysis(self):
        """Test correlation between features and satisfaction using LLM analysis"""
        from narrative_engine.evaluation.eval_user_satisfaction import UserSatisfactionPredictor
        
        predictor = UserSatisfactionPredictor()
        
        # Historical data
        historical_conversations = [
            {"features": {"response_relevance": 0.9, "emotional_variety": 0.8}, "satisfaction": 0.85},
            {"features": {"response_relevance": 0.5, "emotional_variety": 0.3}, "satisfaction": 0.4},
            {"features": {"response_relevance": 0.8, "emotional_variety": 0.7}, "satisfaction": 0.75},
        ]
        
        # Mock LLM correlation analysis
        mock_correlation_analysis = json.dumps({
            "feature_correlations": {
                "response_relevance": 0.92,
                "emotional_variety": 0.74,
                "topic_coherence": 0.68,
                "user_questions_answered": 0.81
            },
            "top_predictive_features": [
                "response_relevance",
                "user_questions_answered", 
                "emotional_variety",
                "topic_coherence"
            ],
            "insights": [
                "Response relevance is the strongest predictor of satisfaction",
                "Question answering rate has high impact on user experience",
                "Emotional variety contributes significantly to engagement"
            ]
        })
        
        with patch.object(predictor.client, 'chat_complete', new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = mock_correlation_analysis
            
            correlations = await predictor.analyze_feature_correlations(historical_conversations)
            
            assert correlations['response_relevance'] > 0.9  # Strong positive correlation
            assert correlations['emotional_variety'] > 0.7
            assert 'top_predictive_features' in correlations
            assert len(correlations['insights']) > 2
    
    @pytest.mark.asyncio
    async def test_improvement_suggestions(self):
        """Test generation of improvement suggestions using LLM"""
        from narrative_engine.evaluation.eval_user_satisfaction import UserSatisfactionPredictor
        
        predictor = UserSatisfactionPredictor()
        
        # Low-performing conversation features
        poor_features = {
            "response_relevance": 0.4,
            "user_questions_answered": 0.3,
            "emotional_variety": 0.2,
            "topic_coherence": 0.5
        }
        
        # Mock the prediction first
        mock_prediction = {
            'predicted_satisfaction': 0.35,
            'areas_for_improvement': ['Response relevance', 'Question answering', 'Emotional engagement']
        }
        
        with patch.object(predictor, 'predict_satisfaction', new_callable=AsyncMock) as mock_predict:
            mock_predict.return_value = mock_prediction
            
            # Mock improvement suggestions response
            mock_suggestions = "1. Focus on directly answering user questions\n2. Improve response relevance\n3. Add more emotional variety"
            
            with patch.object(predictor.client, 'chat_complete', new_callable=AsyncMock) as mock_chat:
                mock_chat.return_value = mock_suggestions
                
                suggestions = await predictor.generate_improvement_suggestions(poor_features, target_satisfaction=0.8)
                
                assert len(suggestions) > 2
                assert any('question' in s.lower() for s in suggestions)
                assert any('relevance' in s.lower() for s in suggestions)
    
    def test_satisfaction_predictor_sync_wrapper(self):
        """Test synchronous wrapper for backwards compatibility"""
        from narrative_engine.evaluation.eval_user_satisfaction import UserSatisfactionPredictor
        
        predictor = UserSatisfactionPredictor()
        
        features = {"response_relevance": 0.8, "emotional_variety": 0.7}
        
        # Mock the async method
        mock_result = {
            'predicted_satisfaction': 0.82,
            'confidence': 0.88,
            'key_strengths': ['Good relevance']
        }
        
        with patch.object(predictor, 'predict_satisfaction', new_callable=AsyncMock) as mock_async:
            mock_async.return_value = mock_result
            
            result = predictor.predict_satisfaction_sync(features)
            
            assert result['predicted_satisfaction'] == 0.82
            assert result['confidence'] == 0.88


class TestABTestingFramework:
    """Tests for A/B testing framework for model comparison"""
    
    def test_ab_test_initialization(self):
        """Test A/B testing framework setup"""
        from narrative_engine.evaluation.ab_testing import ABTestingFramework
        
        framework = ABTestingFramework(
            model_a_name="baseline",
            model_b_name="experimental"
        )
        
        assert framework.model_a_name == "baseline"
        assert framework.model_b_name == "experimental"
        assert framework.significance_level == 0.05
    
    def test_ab_test_execution(self):
        """Test running an A/B test between two models"""
        from narrative_engine.evaluation.ab_testing import ABTestingFramework
        
        framework = ABTestingFramework(
            model_a_name="baseline",
            model_b_name="experimental"
        )
        
        # Mock model outputs and user preferences
        test_results = {
            "model_a_preferred": 45,
            "model_b_preferred": 55,
            "total_comparisons": 100
        }
        
        with patch.object(framework, '_run_comparison') as mock_compare:
            mock_compare.return_value = test_results
            
            results = framework.run_ab_test(
                model_a=Mock(),
                model_b=Mock(),
                test_prompts=["prompt1", "prompt2"],
                num_evaluators=10
            )
            
            assert results['winner'] == "experimental"
            assert results['preference_rate_b'] == 0.55
            assert results['statistical_significance'] is not None
            assert results['confidence_interval'] is not None
    
    def test_ab_test_statistical_analysis(self):
        """Test statistical significance calculation in A/B tests"""
        from narrative_engine.evaluation.ab_testing import ABTestingFramework
        
        framework = ABTestingFramework()
        
        # Test with clear winner
        significant_results = framework.calculate_significance(
            successes_a=30,
            successes_b=70,
            total=100
        )
        
        assert significant_results['p_value'] < 0.05
        assert significant_results['is_significant'] is True
        assert significant_results['effect_size'] > 0.3
        
        # Test with no clear winner
        insignificant_results = framework.calculate_significance(
            successes_a=48,
            successes_b=52,
            total=100
        )
        
        assert insignificant_results['p_value'] > 0.05
        assert insignificant_results['is_significant'] is False


class TestHumanEvaluationInterface:
    """Tests for human evaluation interface and inter-rater reliability"""
    
    def test_human_eval_interface_initialization(self):
        """Test human evaluation interface setup"""
        from narrative_engine.evaluation.human_eval import HumanEvaluationInterface
        
        interface = HumanEvaluationInterface()
        assert interface is not None
        assert hasattr(interface, 'create_evaluation_task')
        assert interface.min_evaluators_per_task == 3
    
    def test_evaluation_task_creation(self):
        """Test creating evaluation tasks for human raters"""
        from narrative_engine.evaluation.human_eval import HumanEvaluationInterface
        
        interface = HumanEvaluationInterface()
        
        task = interface.create_evaluation_task(
            task_type="character_consistency",
            samples=[
                {"id": "sample1", "content": "Character dialogue 1"},
                {"id": "sample2", "content": "Character dialogue 2"}
            ],
            evaluation_criteria={
                "voice_consistency": "Rate 1-5",
                "personality_match": "Rate 1-5",
                "believability": "Rate 1-5"
            }
        )
        
        assert task['task_id'] is not None
        assert task['task_type'] == "character_consistency"
        assert len(task['samples']) == 2
        assert len(task['evaluation_criteria']) == 3
        assert task['status'] == "pending"
    
    def test_inter_rater_reliability(self):
        """Test calculation of inter-rater reliability metrics"""
        from narrative_engine.evaluation.human_eval import HumanEvaluationInterface
        
        interface = HumanEvaluationInterface()
        
        # Mock ratings from multiple evaluators
        ratings = {
            "sample1": {
                "evaluator1": {"voice_consistency": 4, "believability": 5},
                "evaluator2": {"voice_consistency": 4, "believability": 4},
                "evaluator3": {"voice_consistency": 5, "believability": 5}
            },
            "sample2": {
                "evaluator1": {"voice_consistency": 2, "believability": 2},
                "evaluator2": {"voice_consistency": 3, "believability": 2},
                "evaluator3": {"voice_consistency": 2, "believability": 3}
            }
        }
        
        reliability = interface.calculate_inter_rater_reliability(ratings)
        
        assert reliability['krippendorff_alpha'] > 0.6  # Moderate agreement
        assert reliability['fleiss_kappa'] > 0.4
        assert 'per_criterion_agreement' in reliability
        assert reliability['per_criterion_agreement']['voice_consistency'] > 0.5
    
    def test_evaluator_calibration(self):
        """Test evaluator training and calibration process"""
        from narrative_engine.evaluation.human_eval import HumanEvaluationInterface
        
        interface = HumanEvaluationInterface()
        
        # Calibration samples with known good ratings
        calibration_set = [
            {
                "sample_id": "cal1",
                "content": "Example dialogue",
                "gold_ratings": {"quality": 4, "consistency": 5}
            },
            {
                "sample_id": "cal2",
                "content": "Another example",
                "gold_ratings": {"quality": 2, "consistency": 2}
            }
        ]
        
        # Evaluator responses
        evaluator_responses = {
            "quality": [4, 3],  # Close to gold standards
            "consistency": [5, 2]
        }
        
        calibration_score = interface.calculate_calibration_score(
            evaluator_responses, 
            calibration_set
        )
        
        assert calibration_score['overall_accuracy'] > 0.8
        assert calibration_score['mean_absolute_error'] < 0.5
        assert calibration_score['qualified'] is True


class TestTripleHeadCoordinationEvaluator:
    """Tests for evaluating coordination between generation, control, and memory heads"""
    
    def test_triple_head_coordination_scoring(self):
        """Test scoring how well the three heads work together"""
        from narrative_engine.evaluation.eval_triple_head_coordination import TripleHeadCoordinationEvaluator
        
        evaluator = TripleHeadCoordinationEvaluator()
        
        # Mock outputs from all three heads
        generation_output = "I remember you mentioning you love Italian food!"
        control_tokens = {"emotion": "joy", "tone": "friendly", "confidence": 0.9}
        memory_output = {
            "retrieved": ["User loves Italian food"],
            "formed": ["User preference: Italian cuisine"],
            "importance": 0.8
        }
        
        coordination_score = evaluator.evaluate_coordination(
            generation=generation_output,
            control=control_tokens,
            memory=memory_output
        )
        
        assert coordination_score['overall_coordination'] > 0.8
        assert coordination_score['generation_control_alignment'] > 0.85
        assert coordination_score['generation_memory_alignment'] > 0.9
        assert coordination_score['control_memory_alignment'] > 0.7
    
    def test_coordination_failure_detection(self):
        """Test detection of coordination failures between heads"""
        from narrative_engine.evaluation.eval_triple_head_coordination import TripleHeadCoordinationEvaluator
        
        evaluator = TripleHeadCoordinationEvaluator()
        
        # Misaligned outputs
        generation_output = "I'm so sad and disappointed"
        control_tokens = {"emotion": "joy", "tone": "excited"}  # Mismatch!
        memory_output = {
            "retrieved": [],
            "formed": ["User is happy"],  # Also mismatched!
            "importance": 0.3
        }
        
        coordination_score = evaluator.evaluate_coordination(
            generation=generation_output,
            control=control_tokens,
            memory=memory_output
        )
        
        assert coordination_score['overall_coordination'] < 0.3
        assert len(coordination_score['alignment_failures']) > 0
        assert 'emotion_mismatch' in coordination_score['alignment_failures'][0]['type']


class TestComprehensiveEvaluationPipeline:
    """Tests for the complete evaluation pipeline orchestration"""
    
    def test_pipeline_initialization(self):
        """Test comprehensive evaluation pipeline setup"""
        from narrative_engine.evaluation.comprehensive_pipeline import ComprehensiveEvaluationPipeline
        
        pipeline = ComprehensiveEvaluationPipeline()
        
        assert pipeline is not None
        assert len(pipeline.evaluators) > 10  # All evaluators loaded
        assert pipeline.config is not None
        assert pipeline.results_aggregator is not None
    
    def test_pipeline_execution(self):
        """Test running the complete evaluation pipeline"""
        from narrative_engine.evaluation.comprehensive_pipeline import ComprehensiveEvaluationPipeline
        
        pipeline = ComprehensiveEvaluationPipeline()
        
        # Mock model and test data
        mock_model = Mock()
        test_data = {
            "conversations": [{"id": "test1", "turns": []}],
            "character_profiles": [{"id": "char1", "traits": {}}],
            "world_lore": {"setting": "fantasy"}
        }
        
        with patch.object(pipeline, '_run_all_evaluations') as mock_run:
            mock_run.return_value = {
                "character_consistency": 0.85,
                "narrative_quality": 0.82,
                "user_satisfaction_prediction": 0.78,
                "technical_performance": 0.91,
                "safety_compliance": 0.99
            }
            
            results = pipeline.evaluate(
                model=mock_model,
                test_data=test_data,
                output_path="evaluation_results.json"
            )
            
            assert results['overall_score'] > 0.8
            assert results['passed'] is True
            assert 'detailed_metrics' in results
            assert 'recommendations' in results
    
    def test_regression_detection(self):
        """Test detection of quality regression between model versions"""
        from narrative_engine.evaluation.comprehensive_pipeline import ComprehensiveEvaluationPipeline
        
        pipeline = ComprehensiveEvaluationPipeline()
        
        # Previous model results
        baseline_results = {
            "character_consistency": 0.85,
            "narrative_quality": 0.82,
            "user_satisfaction": 0.80
        }
        
        # New model results (regression in character consistency)
        new_results = {
            "character_consistency": 0.65,  # Significant drop!
            "narrative_quality": 0.84,
            "user_satisfaction": 0.79
        }
        
        regression_analysis = pipeline.detect_regressions(
            baseline=baseline_results,
            current=new_results,
            threshold=0.1
        )
        
        assert regression_analysis['has_regression'] is True
        assert len(regression_analysis['regressions']) == 1
        assert regression_analysis['regressions'][0]['metric'] == 'character_consistency'
        assert regression_analysis['regressions'][0]['severity'] == 'high' 