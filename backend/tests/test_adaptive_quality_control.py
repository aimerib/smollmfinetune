"""
Tests for Adaptive Quality Control

This module tests the adaptive quality control system that adjusts voice generation
parameters based on narrative context:
- Quality level determination from narrative context
- Performance vs quality tradeoffs  
- Real-time adaptation during conversation
- Integration with existing voice streaming

Focus on unit testing the quality control logic, not actual voice generation.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional

# Mock dependencies before importing our modules
with patch('backend.app.redis_client.get_redis_pool'), \
     patch('backend.app.database.create_tables'):
    from backend.app.services.voice.adaptive_quality_controller import AdaptiveQualityController
    from backend.app.services.voice.narrative_quality_analyzer import NarrativeQualityAnalyzer
    from backend.app.services.voice.quality_models import QualityLevel, NarrativeContext


class TestAdaptiveQualityController:
    """Test adaptive quality control for voice generation"""
    
    @pytest.fixture
    def quality_controller(self):
        """Create AdaptiveQualityController instance"""
        return AdaptiveQualityController(
            default_quality=QualityLevel.MEDIUM,
            adaptation_threshold=0.1,
            performance_weight=0.3
        )
    
    @pytest.fixture
    def narrative_analyzer(self):
        """Create NarrativeQualityAnalyzer instance"""
        return NarrativeQualityAnalyzer()
    
    @pytest.fixture
    def sample_narrative_contexts(self):
        """Sample narrative contexts for testing"""
        return {
            "casual_conversation": NarrativeContext(
                tension_level=0.2,
                emotional_intensity=0.3,
                narrative_importance=0.1,
                dialogue_type="casual",
                scene_type="conversation",
                character_focus="background"
            ),
            "dramatic_climax": NarrativeContext(
                tension_level=0.9,
                emotional_intensity=0.95,
                narrative_importance=0.9,
                dialogue_type="dramatic",
                scene_type="climax",
                character_focus="protagonist"
            ),
            "action_sequence": NarrativeContext(
                tension_level=0.8,
                emotional_intensity=0.7,
                narrative_importance=0.6,
                dialogue_type="urgent",
                scene_type="action",
                character_focus="multiple"
            ),
            "intimate_moment": NarrativeContext(
                tension_level=0.4,
                emotional_intensity=0.8,
                narrative_importance=0.7,
                dialogue_type="emotional",
                scene_type="intimate",
                character_focus="two_characters"
            )
        }
    
    def test_quality_controller_initialization(self, quality_controller):
        """Test AdaptiveQualityController initializes correctly"""
        assert quality_controller.default_quality == QualityLevel.MEDIUM
        assert quality_controller.adaptation_threshold == 0.1
        assert quality_controller.performance_weight == 0.3
        assert quality_controller.narrative_analyzer is not None
        assert quality_controller.performance_tracker is not None
    
    @pytest.mark.asyncio
    async def test_determine_quality_level_dramatic_scene(self, quality_controller, sample_narrative_contexts):
        """Test quality level determination for dramatic scenes"""
        dramatic_context = sample_narrative_contexts["dramatic_climax"]
        
        quality_decision = await quality_controller.determine_quality_level(
            narrative_context=dramatic_context,
            character_id="protagonist_001",
            current_load=0.3
        )
        
        # Dramatic scenes should get high quality
        assert quality_decision.quality_level == QualityLevel.HIGH
        assert "High narrative importance" in quality_decision.reasoning
        assert quality_decision.confidence_score > 0.8
    
    @pytest.mark.asyncio 
    async def test_determine_quality_level_casual_conversation(self, quality_controller, sample_narrative_contexts):
        """Test quality level determination for casual conversation"""
        casual_context = sample_narrative_contexts["casual_conversation"]
        
        quality_decision = await quality_controller.determine_quality_level(
            narrative_context=casual_context,
            character_id="background_npc_042",
            current_load=0.7
        )
        
        # Casual conversation should get low/medium quality, especially under load
        assert quality_decision.quality_level in [QualityLevel.LOW, QualityLevel.MEDIUM]
        assert "Low narrative importance" in quality_decision.reasoning or "background character" in quality_decision.reasoning
        assert quality_decision.performance_impact < 0.5
    
    @pytest.mark.asyncio
    async def test_adapt_to_performance_load(self, quality_controller, sample_narrative_contexts):
        """Test quality adaptation under high system load"""
        dramatic_context = sample_narrative_contexts["dramatic_climax"]
        
        # High load should reduce quality even for dramatic scenes
        quality_decision = await quality_controller.determine_quality_level(
            narrative_context=dramatic_context,
            character_id="protagonist_001", 
            current_load=0.95  # Very high load
        )
        
        # Should compromise on quality due to performance
        assert quality_decision.quality_level in [QualityLevel.MEDIUM, QualityLevel.HIGH]
        assert quality_decision.performance_consideration_applied is True
        assert "performance load" in quality_decision.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_character_importance_influence(self, quality_controller, sample_narrative_contexts):
        """Test that character importance influences quality decisions"""
        context = sample_narrative_contexts["action_sequence"]
        
        # Protagonist should get higher quality than background characters
        protagonist_decision = await quality_controller.determine_quality_level(
            narrative_context=context,
            character_id="protagonist_001",
            current_load=0.5
        )
        
        background_decision = await quality_controller.determine_quality_level(
            narrative_context=context, 
            character_id="background_guard_17",
            current_load=0.5
        )
        
        assert protagonist_decision.quality_level.value >= background_decision.quality_level.value
    
    @pytest.mark.asyncio
    async def test_quality_caching_for_similar_contexts(self, quality_controller, sample_narrative_contexts):
        """Test that identical narrative contexts are cached for performance"""
        # Use dramatic context which should have high confidence and be cacheable
        context = sample_narrative_contexts["dramatic_climax"]
        
        # First call
        decision1 = await quality_controller.determine_quality_level(
            narrative_context=context,
            character_id="protagonist_001",
            current_load=0.4
        )
        
        # Second call with identical context should use cache
        decision2 = await quality_controller.determine_quality_level(
            narrative_context=context,  # Same context
            character_id="protagonist_001",
            current_load=0.4
        )
        
        # Should have identical decisions and second should be cached
        assert decision1.quality_level == decision2.quality_level
        assert decision1.confidence_score > 0.7  # Should be high confidence
        assert decision2.cached_decision is True
    
    @pytest.mark.asyncio
    async def test_real_time_adaptation_during_conversation(self, quality_controller):
        """Test quality adaptation as conversation evolves"""
        conversation_states = [
            # Starting casual
            ("Hello there!", NarrativeContext(
                tension_level=0.1, emotional_intensity=0.2, narrative_importance=0.1,
                dialogue_type="greeting", scene_type="conversation", character_focus="background"
            )),
            # Building tension
            ("Wait, did you hear that noise?", NarrativeContext(
                tension_level=0.6, emotional_intensity=0.4, narrative_importance=0.3,
                dialogue_type="suspicious", scene_type="mystery", character_focus="protagonist"
            )),
            # High drama
            ("The building is collapsing! We need to get out NOW!", NarrativeContext(
                tension_level=0.95, emotional_intensity=0.9, narrative_importance=0.8,
                dialogue_type="urgent", scene_type="crisis", character_focus="protagonist"
            ))
        ]
        
        quality_progression = []
        for text, context in conversation_states:
            decision = await quality_controller.determine_quality_level(
                narrative_context=context,
                character_id="protagonist_001",
                current_load=0.4
            )
            quality_progression.append(decision.quality_level.value)
        
        # Quality should generally increase as tension/importance increases
        assert quality_progression[0] <= quality_progression[1] <= quality_progression[2]
    
    @pytest.mark.asyncio
    async def test_get_optimized_tts_parameters(self, quality_controller, sample_narrative_contexts):
        """Test generation of optimized TTS parameters based on quality level"""
        dramatic_context = sample_narrative_contexts["dramatic_climax"]
        
        tts_params = await quality_controller.get_optimized_tts_parameters(
            narrative_context=dramatic_context,
            character_id="protagonist_001",
            base_voice_config={"model": "orpheus", "voice_id": "protagonist_voice"}
        )
        
        # High quality should have optimized parameters
        assert tts_params.quality_level == QualityLevel.HIGH
        assert tts_params.sample_rate >= 44100  # Higher sample rate for quality
        assert tts_params.inference_steps >= 50  # More steps for better quality
        assert tts_params.temperature <= 0.7  # Lower temperature for consistency
        assert tts_params.use_advanced_features is True


class TestNarrativeQualityAnalyzer:
    """Test narrative context analysis for quality determination"""
    
    @pytest.fixture
    def analyzer(self):
        """Create NarrativeQualityAnalyzer instance"""
        return NarrativeQualityAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test NarrativeQualityAnalyzer initializes correctly"""
        assert analyzer.tension_weight > 0
        assert analyzer.emotion_weight > 0
        assert analyzer.importance_weight > 0
        assert hasattr(analyzer, 'dialogue_type_mapping')
        assert hasattr(analyzer, 'scene_type_mapping')
    
    @pytest.mark.asyncio
    async def test_analyze_text_for_narrative_context(self, analyzer):
        """Test extraction of narrative context from text"""
        dramatic_text = "NO! You can't do this! After everything we've been through!"
        casual_text = "Hey, how's the weather today?"
        
        dramatic_context = await analyzer.analyze_text_for_context(
            text=dramatic_text,
            conversation_history=["Previous urgent message"],
            character_role="protagonist"
        )
        
        casual_context = await analyzer.analyze_text_for_context(
            text=casual_text,
            conversation_history=["Hi there", "Hello"],
            character_role="background"
        )
        
        # Dramatic text should have higher scores
        assert dramatic_context.emotional_intensity > casual_context.emotional_intensity
        assert dramatic_context.tension_level > casual_context.tension_level
        assert dramatic_context.dialogue_type != casual_context.dialogue_type
    
    @pytest.mark.asyncio
    async def test_calculate_narrative_importance_score(self, analyzer):
        """Test calculation of overall narrative importance"""
        high_importance_context = NarrativeContext(
            tension_level=0.9,
            emotional_intensity=0.8,
            narrative_importance=0.9,
            dialogue_type="dramatic",
            scene_type="climax",
            character_focus="protagonist"
        )
        
        low_importance_context = NarrativeContext(
            tension_level=0.1,
            emotional_intensity=0.2,
            narrative_importance=0.1,
            dialogue_type="casual",
            scene_type="filler",
            character_focus="background"
        )
        
        high_score = await analyzer.calculate_importance_score(high_importance_context)
        low_score = await analyzer.calculate_importance_score(low_importance_context)
        
        assert high_score > 0.7
        assert low_score < 0.3
        assert high_score > low_score * 2  # Significant difference
    
    def test_dialogue_type_classification(self, analyzer):
        """Test classification of dialogue types"""
        test_cases = [
            ("Hello, how are you?", "casual"),
            ("LOOK OUT! INCOMING!", "urgent"),
            ("I... I love you.", "emotional"),
            ("The data suggests otherwise.", "informational"),
            ("You shall not pass!", "dramatic")
        ]
        
        for text, expected_type in test_cases:
            classified_type = analyzer.classify_dialogue_type(text)
            assert classified_type == expected_type
    
    def test_scene_type_detection(self, analyzer):
        """Test detection of scene types from context"""
        test_contexts = [
            (["character enters room", "normal lighting"], "conversation"),
            (["explosions in distance", "urgent music"], "action"),  
            (["two characters alone", "soft lighting"], "intimate"),
            (["all characters present", "dramatic revelation"], "climax")
        ]
        
        for context_clues, expected_scene in test_contexts:
            detected_scene = analyzer.detect_scene_type(context_clues)
            assert detected_scene == expected_scene


class TestQualityModels:
    """Test quality level models and enums"""
    
    def test_quality_level_enum(self):
        """Test QualityLevel enum values and ordering"""
        assert QualityLevel.LOW.value < QualityLevel.MEDIUM.value
        assert QualityLevel.MEDIUM.value < QualityLevel.HIGH.value
        assert QualityLevel.HIGH.value < QualityLevel.ULTRA.value
        
        # Test string representations
        assert str(QualityLevel.LOW) == "low"
        assert str(QualityLevel.HIGH) == "high"
    
    def test_narrative_context_validation(self):
        """Test NarrativeContext model validation"""
        # Valid context
        valid_context = NarrativeContext(
            tension_level=0.5,
            emotional_intensity=0.7,
            narrative_importance=0.6,
            dialogue_type="casual",
            scene_type="conversation",
            character_focus="protagonist"
        )
        
        assert valid_context.tension_level == 0.5
        assert valid_context.is_high_priority() == False  # Not dramatic enough
        
        # High priority context
        dramatic_context = NarrativeContext(
            tension_level=0.9,
            emotional_intensity=0.9,
            narrative_importance=0.8,
            dialogue_type="dramatic",
            scene_type="climax", 
            character_focus="protagonist"
        )
        
        assert dramatic_context.is_high_priority() == True
    
    def test_quality_decision_model(self):
        """Test QualityDecision model structure"""
        from backend.app.services.voice.quality_models import QualityDecision
        
        decision = QualityDecision(
            quality_level=QualityLevel.HIGH,
            reasoning="High narrative importance",
            confidence_score=0.85,
            performance_impact=0.7,
            estimated_latency_ms=1200,
            cached_decision=False,
            performance_consideration_applied=True
        )
        
        assert decision.quality_level == QualityLevel.HIGH
        assert decision.confidence_score == 0.85
        assert decision.should_use_cache() == True  # High confidence, not already cached
        assert decision.is_acceptable_latency(max_latency_ms=1500) == True
        
        # Test a decision that shouldn't be cached
        low_confidence_decision = QualityDecision(
            quality_level=QualityLevel.MEDIUM,
            reasoning="Uncertain context",
            confidence_score=0.5,  # Low confidence
            performance_impact=0.5,
            estimated_latency_ms=600,
            cached_decision=False,
            performance_consideration_applied=False
        )
        
        assert low_confidence_decision.should_use_cache() == False  # Low confidence 