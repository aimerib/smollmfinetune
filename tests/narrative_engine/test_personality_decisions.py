"""
Tests for personality-driven decision making system.

This module tests the psychological authenticity of character decision-making,
ensuring that Big Five personality traits, emotional states, relationships,
and group dynamics all influence choices in realistic ways.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import the classes we'll be implementing
from backend.app.narrative_engine.personality_decision_engine import (
    PersonalityDecisionEngine,
    DecisionContext,
    ChoiceOption,
    DecisionResult,
    PersonalityWeights
)
from backend.app.narrative_engine.emotional_decision_modifiers import EmotionalDecisionModifier
from backend.app.narrative_engine.relationship_decision_influence import RelationshipDecisionInfluence
from backend.app.narrative_engine.decision_consistency_tracker import (
    DecisionConsistencyTracker,
    DecisionRecord,
    PersonalityConsistencyReport,
    PersonalityGrowthSuggestion
)


class TestPersonalityDecisionEngine:
    """Test the core personality-driven decision making system."""
    
    @pytest.fixture
    def mock_triple_head_model(self):
        """Mock triple head model for testing."""
        model = Mock()
        model.analyze_relationship_impact = AsyncMock(return_value=Mock(impact_score=0.3))
        return model
    
    @pytest.fixture
    def mock_relationship_manager(self):
        """Mock relationship manager for testing."""
        return Mock()
    
    @pytest.fixture
    def decision_engine(self, mock_triple_head_model, mock_relationship_manager):
        """Create a decision engine for testing."""
        return PersonalityDecisionEngine(mock_triple_head_model, mock_relationship_manager)
    
    @pytest.fixture
    def high_openness_personality(self):
        """High openness personality profile."""
        return {
            'openness': 0.8,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.3
        }
    
    @pytest.fixture
    def high_conscientiousness_personality(self):
        """High conscientiousness personality profile."""
        return {
            'openness': 0.4,
            'conscientiousness': 0.9,
            'extraversion': 0.5,
            'agreeableness': 0.6,
            'neuroticism': 0.3
        }
    
    @pytest.fixture
    def novel_vs_familiar_choices(self):
        """Choice between novel and familiar options."""
        return [
            ChoiceOption(
                choice_id="novel_adventure",
                description="Explore the mysterious cave",
                predicted_outcomes=["Discovery", "Possible danger"],
                personality_alignment={
                    'openness_appeal': 0.9,
                    'conventional_appeal': 0.1,
                    'risk_level': 0.7
                },
                emotional_cost={'anxiety': 0.4},
                relationship_impact={},
                moral_weight={},
                group_acceptance=0.5
            ),
            ChoiceOption(
                choice_id="familiar_path",
                description="Take the well-known route",
                predicted_outcomes=["Safety", "Predictable outcome"],
                personality_alignment={
                    'openness_appeal': 0.2,
                    'conventional_appeal': 0.9,
                    'risk_level': 0.1
                },
                emotional_cost={'anxiety': 0.1},
                relationship_impact={},
                moral_weight={},
                group_acceptance=0.8
            )
        ]
    
    @pytest.fixture
    def duty_vs_fun_choices(self):
        """Choice between duty and personal pleasure."""
        return [
            ChoiceOption(
                choice_id="fulfill_duty",
                description="Complete the promised task",
                predicted_outcomes=["Honor kept", "Personal sacrifice"],
                personality_alignment={
                    'duty_alignment': 0.9,
                    'long_term_benefit': 0.8,
                    'immediate_gratification': 0.1
                },
                emotional_cost={'stress': 0.3},
                relationship_impact={},
                moral_weight={'honor': 0.9},
                group_acceptance=0.7
            ),
            ChoiceOption(
                choice_id="pursue_pleasure",
                description="Enjoy the festival instead",
                predicted_outcomes=["Immediate joy", "Broken promise"],
                personality_alignment={
                    'duty_alignment': 0.1,
                    'long_term_benefit': 0.2,
                    'immediate_gratification': 0.9
                },
                emotional_cost={'guilt': 0.6},
                relationship_impact={},
                moral_weight={'honor': 0.1},
                group_acceptance=0.9
            )
        ]
    
    async def test_high_openness_prefers_novel_choices(self, decision_engine, high_openness_personality, novel_vs_familiar_choices):
        """Test that high openness agents choose novel/creative options."""
        context = DecisionContext(
            agent_id="clara",
            available_choices=novel_vs_familiar_choices,
            situation_description="Standing at crossroads",
            emotional_state={'curious': 0.7, 'anxious': 0.2},
            relationship_context={},
            group_context=None,
            time_pressure=0.3,
            stakes_level=0.5,
            moral_dimensions=[]
        )
        
        # Mock the personality retrieval
        with patch.object(decision_engine, '_get_agent_personality', return_value=high_openness_personality):
            with patch.object(decision_engine, '_score_choices_by_personality') as mock_score:
                # High openness should score novel choice higher
                mock_score.return_value = {
                    'novel_adventure': 0.8,
                    'familiar_path': 0.3
                }
                
                with patch.object(decision_engine, '_apply_emotional_influence', side_effect=lambda scores, *args: scores):
                    with patch.object(decision_engine, '_apply_relationship_influence', side_effect=lambda scores, *args: scores):
                        with patch.object(decision_engine, '_generate_final_decision') as mock_final:
                            mock_final.return_value = DecisionResult(
                                chosen_option=novel_vs_familiar_choices[0],
                                confidence_score=0.8,
                                reasoning="High openness drives exploration",
                                personality_factors={'openness': 0.8}
                            )
                            
                            result = await decision_engine.make_personality_driven_decision(context)
                            
                            assert result.chosen_option.choice_id == "novel_adventure"
                            assert result.confidence_score > 0.7
                            assert "openness" in result.reasoning.lower()

    async def test_conscientiousness_drives_duty_based_decisions(self, decision_engine, high_conscientiousness_personality, duty_vs_fun_choices):
        """Test that conscientious agents prioritize duty and responsibility."""
        context = DecisionContext(
            agent_id="tom",
            available_choices=duty_vs_fun_choices,
            situation_description="Festival day with pending obligation",
            emotional_state={'conflicted': 0.6, 'tempted': 0.5},
            relationship_context={},
            group_context=None,
            time_pressure=0.7,
            stakes_level=0.8,
            moral_dimensions=['honor', 'responsibility']
        )
        
        with patch.object(decision_engine, '_get_agent_personality', return_value=high_conscientiousness_personality):
            with patch.object(decision_engine, '_score_choices_by_personality') as mock_score:
                # High conscientiousness should score duty choice higher
                mock_score.return_value = {
                    'fulfill_duty': 0.9,
                    'pursue_pleasure': 0.2
                }
                
                with patch.object(decision_engine, '_apply_emotional_influence', side_effect=lambda scores, *args: scores):
                    with patch.object(decision_engine, '_apply_relationship_influence', side_effect=lambda scores, *args: scores):
                        with patch.object(decision_engine, '_generate_final_decision') as mock_final:
                            mock_final.return_value = DecisionResult(
                                chosen_option=duty_vs_fun_choices[0],
                                confidence_score=0.9,
                                reasoning="High conscientiousness demands duty fulfillment",
                                personality_factors={'conscientiousness': 0.9}
                            )
                            
                            result = await decision_engine.make_personality_driven_decision(context)
                            
                            assert result.chosen_option.choice_id == "fulfill_duty"
                            assert result.confidence_score > 0.8
                            assert "conscientiousness" in result.reasoning.lower() or "duty" in result.reasoning.lower()

    async def test_personality_scoring_algorithm(self, decision_engine, high_openness_personality, novel_vs_familiar_choices):
        """Test the personality scoring algorithm produces expected results."""
        scores = await decision_engine._score_choices_by_personality(
            novel_vs_familiar_choices, 
            high_openness_personality,
            Mock()  # context mock
        )
        
        # High openness should prefer novel choice
        assert scores['novel_adventure'] > scores['familiar_path']
        assert scores['novel_adventure'] > 0.6  # Should be significantly positive


class TestEmotionalDecisionModification:
    """Test how emotions modify personality-based decisions."""
    
    @pytest.fixture
    def emotional_modifier(self):
        """Create emotional decision modifier for testing."""
        return EmotionalDecisionModifier()
    
    @pytest.fixture
    def base_choice_scores(self):
        """Base personality-driven choice scores."""
        return {
            'confrontational_choice': 0.3,
            'cooperative_choice': 0.7,
            'risky_choice': 0.6,
            'safe_choice': 0.4
        }
    
    async def test_anger_increases_confrontational_choices(self, emotional_modifier, base_choice_scores):
        """Test that anger makes agents more likely to choose confrontation."""
        angry_emotional_state = {'angry': 0.8, 'frustrated': 0.6}
        personality = {'agreeableness': 0.6}
        
        with patch.object(emotional_modifier, '_choice_involves_confrontation') as mock_confrontation:
            mock_confrontation.side_effect = lambda choice_id: choice_id == 'confrontational_choice'
            
            with patch.object(emotional_modifier, '_choice_involves_cooperation') as mock_cooperation:
                mock_cooperation.side_effect = lambda choice_id: choice_id == 'cooperative_choice'
                
                modified_scores = await emotional_modifier.apply_emotional_influence(
                    base_choice_scores, angry_emotional_state, personality
                )
                
                # Anger should increase confrontational choice score and decrease cooperative choice
                assert modified_scores['confrontational_choice'] > base_choice_scores['confrontational_choice']
                assert modified_scores['cooperative_choice'] < base_choice_scores['cooperative_choice']

    async def test_fear_increases_risk_aversion(self, emotional_modifier, base_choice_scores):
        """Test that fear makes agents avoid risky choices."""
        fearful_emotional_state = {'fearful': 0.7, 'anxious': 0.5}
        personality = {'neuroticism': 0.6}
        
        with patch.object(emotional_modifier, '_assess_choice_risk') as mock_risk:
            mock_risk.side_effect = lambda choice_id: 0.8 if choice_id == 'risky_choice' else 0.2
            
            modified_scores = await emotional_modifier.apply_emotional_influence(
                base_choice_scores, fearful_emotional_state, personality
            )
            
            # Fear should decrease risky choice score and increase safe choice preference
            assert modified_scores['risky_choice'] < base_choice_scores['risky_choice']
            assert modified_scores['safe_choice'] >= base_choice_scores['safe_choice']

    async def test_joy_increases_social_openness(self, emotional_modifier, base_choice_scores):
        """Test that happiness makes agents more socially open."""
        joyful_emotional_state = {'happy': 0.8, 'excited': 0.6}
        personality = {'extraversion': 0.4}  # Normally introverted
        
        social_choice_scores = {
            'social_interaction': 0.3,
            'solitary_activity': 0.7
        }
        
        with patch.object(emotional_modifier, '_choice_involves_social_interaction') as mock_social:
            mock_social.side_effect = lambda choice_id: choice_id == 'social_interaction'
            
            modified_scores = await emotional_modifier.apply_emotional_influence(
                social_choice_scores, joyful_emotional_state, personality
            )
            
            # Joy should temporarily increase social choice preference
            assert modified_scores['social_interaction'] > social_choice_scores['social_interaction']


class TestRelationshipInfluence:
    """Test how relationships affect decision making."""
    
    @pytest.fixture
    def relationship_influence(self):
        """Create relationship decision influence system."""
        mock_relationship_manager = Mock()
        return RelationshipDecisionInfluence(mock_relationship_manager)
    
    @pytest.fixture
    def high_affinity_relationship_context(self):
        """Context with high-affinity relationship."""
        return {
            'best_friend_id': {
                'status': 'Best Friend',
                'affinity': 0.9,
                'emotional_history': ['supportive', 'loyal', 'caring'],
                'memory_significance': 0.8
            }
        }
    
    async def test_high_affinity_relationships_strongly_influence_decisions(self, relationship_influence, high_affinity_relationship_context):
        """Test that close relationships have strong decision influence."""
        base_scores = {
            'help_friend': 0.4,
            'ignore_friend': 0.6
        }
        personality = {'agreeableness': 0.7}
        
        with patch.object(relationship_influence, '_analyze_choice_relationship_impact') as mock_impact:
            mock_impact.return_value = 0.6  # Positive impact for helping friend
            
            with patch.object(relationship_influence, '_calculate_relationship_importance') as mock_importance:
                mock_importance.return_value = 0.8  # High importance for best friend
                
                modified_scores = await relationship_influence.apply_relationship_influence(
                    base_scores, high_affinity_relationship_context, personality
                )
                
                # Close relationship should strongly influence toward helping
                assert modified_scores['help_friend'] > base_scores['help_friend']
                
                # Verify the relationship importance calculation was called
                mock_importance.assert_called()

    async def test_agreeable_agents_weight_relationships_more_heavily(self, relationship_influence):
        """Test that agreeable personalities prioritize relationship harmony."""
        high_agreeableness = {'agreeableness': 0.9}
        low_agreeableness = {'agreeableness': 0.2}
        
        mock_relationship = Mock()
        mock_relationship.status = "Friend"
        mock_relationship.affinity = 0.6
        mock_relationship.memory_significance = 0.5
        
        high_agree_importance = relationship_influence._calculate_relationship_importance(
            mock_relationship, high_agreeableness
        )
        low_agree_importance = relationship_influence._calculate_relationship_importance(
            mock_relationship, low_agreeableness
        )
        
        # High agreeableness should weight relationships more heavily
        assert high_agree_importance > low_agree_importance


class TestDecisionConsistencyTracker:
    """Test decision pattern consistency and personality growth tracking."""
    
    @pytest.fixture
    def consistency_tracker(self):
        """Create decision consistency tracker."""
        return DecisionConsistencyTracker()
    
    @pytest.fixture
    def consistent_decision_history(self):
        """Create a history of personality-consistent decisions."""
        decisions = []
        for i in range(10):
            decision = DecisionRecord(
                decision_id=f"decision_{i}",
                agent_id="clara",
                context=Mock(),
                chosen_option=Mock(),
                personality_scores={'openness': 0.8},
                final_score=0.8,  # Consistently high scores for openness-aligned choices
                reasoning="High openness preference",
                timestamp=datetime.now(),
                outcomes=[],
                regret_level=0.1
            )
            decisions.append(decision)
        return decisions
    
    async def test_decision_consistency_calculation(self, consistency_tracker, consistent_decision_history):
        """Test that consistent decisions receive high consistency scores."""
        consistency_tracker.decision_history["clara"] = consistent_decision_history
        
        with patch.object(consistency_tracker, '_calculate_expected_personality_score') as mock_expected:
            mock_expected.return_value = 0.8  # Expected score matches actual
            
            report = await consistency_tracker.analyze_decision_consistency("clara", recent_decisions=10)
            
            # Consistent decisions should have high consistency score
            assert report.consistency_score > 0.8
            assert len(report.concerning_patterns) == 0

    async def test_personality_growth_suggestions(self, consistency_tracker):
        """Test personality growth suggestion system."""
        # Create decision history showing potential growth
        growth_decisions = []
        for i in range(10):
            # Gradually increasing social choices despite low extraversion
            social_score = 0.3 + (i * 0.05)  # Increasing over time
            decision = DecisionRecord(
                decision_id=f"social_decision_{i}",
                agent_id="tom",
                context=Mock(),
                chosen_option=Mock(),
                personality_scores={'extraversion': social_score},
                final_score=social_score,
                reasoning="Increasingly social choices",
                timestamp=datetime.now(),
                outcomes=["positive_social_outcome"],
                regret_level=0.1
            )
            growth_decisions.append(decision)
        
        consistency_tracker.decision_history["tom"] = growth_decisions
        
        with patch.object(consistency_tracker, '_analyze_growth_patterns') as mock_growth:
            mock_growth.return_value = Mock(
                growth_opportunities=['extraversion'],
                evidence="Increasing social engagement over time",
                trait_deltas={'extraversion': 0.1}
            )
            
            suggestion = await consistency_tracker.suggest_personality_growth("tom")
            
            assert 'extraversion' in suggestion.growth_areas
            assert suggestion.suggested_trait_adjustments.get('extraversion', 0) > 0


class TestPersonalityDecisionIntegration:
    """Integration tests for the complete personality decision system."""
    
    @pytest.fixture
    def full_decision_system(self):
        """Create complete integrated decision system."""
        mock_model = Mock()
        mock_relationship_manager = Mock()
        decision_engine = PersonalityDecisionEngine(mock_model, mock_relationship_manager)
        return decision_engine
    
    async def test_complete_decision_workflow(self, full_decision_system):
        """Test full decision making process from context to choice."""
        complex_context = DecisionContext(
            agent_id="alex",
            available_choices=[
                ChoiceOption(
                    choice_id="complex_choice_a",
                    description="Multi-faceted decision A",
                    predicted_outcomes=["Outcome A1", "Outcome A2"],
                    personality_alignment={'openness_appeal': 0.7},
                    emotional_cost={'stress': 0.3},
                    relationship_impact={'friend_1': 0.5},
                    moral_weight={'honesty': 0.8},
                    group_acceptance=0.6
                ),
                ChoiceOption(
                    choice_id="complex_choice_b",
                    description="Multi-faceted decision B",
                    predicted_outcomes=["Outcome B1", "Outcome B2"],
                    personality_alignment={'conscientiousness': 0.8},
                    emotional_cost={'anxiety': 0.4},
                    relationship_impact={'friend_1': -0.2},
                    moral_weight={'duty': 0.9},
                    group_acceptance=0.8
                )
            ],
            situation_description="Complex moral and social dilemma",
            emotional_state={'conflicted': 0.7, 'determined': 0.5},
            relationship_context={'friend_1': {'affinity': 0.8}},
            group_context=Mock(),
            time_pressure=0.6,
            stakes_level=0.8,
            moral_dimensions=['honesty', 'duty']
        )
        
        # Mock all the internal methods to return realistic values
        personality = {'openness': 0.7, 'conscientiousness': 0.6}
        
        with patch.object(full_decision_system, '_get_agent_personality', return_value=personality):
            with patch.object(full_decision_system, '_score_choices_by_personality') as mock_score:
                mock_score.return_value = {'complex_choice_a': 0.6, 'complex_choice_b': 0.7}
                
                with patch.object(full_decision_system, '_apply_emotional_influence') as mock_emotional:
                    mock_emotional.return_value = {'complex_choice_a': 0.5, 'complex_choice_b': 0.8}
                    
                    with patch.object(full_decision_system, '_apply_relationship_influence') as mock_relationship:
                        mock_relationship.return_value = {'complex_choice_a': 0.7, 'complex_choice_b': 0.6}
                        
                        with patch.object(full_decision_system, '_generate_final_decision') as mock_final:
                            expected_result = DecisionResult(
                                chosen_option=complex_context.available_choices[0],
                                confidence_score=0.7,
                                reasoning="Complex decision integrating personality, emotions, and relationships",
                                personality_factors=personality
                            )
                            mock_final.return_value = expected_result
                            
                            result = await full_decision_system.make_personality_driven_decision(complex_context)
                            
                            # Verify all influence factors were considered
                            mock_score.assert_called_once()
                            mock_emotional.assert_called_once()
                            mock_relationship.assert_called_once()
                            mock_final.assert_called_once()
                            
                            assert result.chosen_option is not None
                            assert result.confidence_score > 0.5
                            assert len(result.reasoning) > 0

    async def test_group_and_relationship_interaction(self, full_decision_system):
        """Test how group pressure and relationship influence interact."""
        # Create scenario where group wants one thing, close friend wants another
        context_with_conflict = DecisionContext(
            agent_id="sara",
            available_choices=[
                ChoiceOption(
                    choice_id="please_group",
                    description="Go along with group decision",
                    predicted_outcomes=["Group harmony", "Friend disappointment"],
                    personality_alignment={},
                    emotional_cost={},
                    relationship_impact={'best_friend': -0.5},
                    moral_weight={},
                    group_acceptance=0.9
                ),
                ChoiceOption(
                    choice_id="support_friend",
                    description="Support friend against group",
                    predicted_outcomes=["Friend loyalty", "Group tension"],
                    personality_alignment={},
                    emotional_cost={},
                    relationship_impact={'best_friend': 0.7},
                    moral_weight={'loyalty': 0.8},
                    group_acceptance=0.2
                )
            ],
            situation_description="Group vs friend loyalty conflict",
            emotional_state={'torn': 0.8, 'stressed': 0.6},
            relationship_context={'best_friend': {'affinity': 0.9, 'status': 'Best Friend'}},
            group_context=Mock(peer_pressure_strength=0.7),
            time_pressure=0.5,
            stakes_level=0.9,
            moral_dimensions=['loyalty', 'conformity']
        )
        
        personality = {'agreeableness': 0.8, 'openness': 0.6}  # High agreeableness = values relationships
        
        with patch.object(full_decision_system, '_get_agent_personality', return_value=personality):
            with patch.object(full_decision_system, '_score_choices_by_personality', return_value={'please_group': 0.5, 'support_friend': 0.5}):
                with patch.object(full_decision_system, '_apply_emotional_influence', side_effect=lambda scores, *args: scores):
                    with patch.object(full_decision_system, '_apply_relationship_influence') as mock_relationship:
                        # High agreeableness should weight close relationship heavily
                        mock_relationship.return_value = {'please_group': 0.3, 'support_friend': 0.8}
                        
                        with patch.object(full_decision_system, '_apply_group_influence') as mock_group:
                            # Group influence should counter relationship influence somewhat
                            mock_group.return_value = {'please_group': 0.5, 'support_friend': 0.6}
                            
                            with patch.object(full_decision_system, '_generate_final_decision') as mock_final:
                                # Should still favor friend due to high agreeableness and relationship strength
                                mock_final.return_value = DecisionResult(
                                    chosen_option=context_with_conflict.available_choices[1],
                                    confidence_score=0.6,
                                    reasoning="High agreeableness prioritizes close relationship over group pressure",
                                    personality_factors=personality
                                )
                                
                                result = await full_decision_system.make_personality_driven_decision(context_with_conflict)
                                
                                assert result.chosen_option.choice_id == "support_friend"
                                assert "relationship" in result.reasoning.lower() or "friend" in result.reasoning.lower()


# Data structure tests to ensure proper model definitions
class TestDecisionDataStructures:
    """Test the data structures used in personality decision making."""
    
    def test_decision_context_creation(self):
        """Test DecisionContext can be created with all required fields."""
        context = DecisionContext(
            agent_id="test_agent",
            available_choices=[],
            situation_description="Test situation",
            emotional_state={},
            relationship_context={},
            group_context=None,
            time_pressure=0.5,
            stakes_level=0.7,
            moral_dimensions=[]
        )
        
        assert context.agent_id == "test_agent"
        assert context.time_pressure == 0.5
        assert context.stakes_level == 0.7
    
    def test_choice_option_creation(self):
        """Test ChoiceOption can be created with all personality alignment data."""
        choice = ChoiceOption(
            choice_id="test_choice",
            description="Test choice description",
            predicted_outcomes=["Outcome 1", "Outcome 2"],
            personality_alignment={'openness_appeal': 0.8},
            emotional_cost={'anxiety': 0.3},
            relationship_impact={'friend_1': 0.5},
            moral_weight={'honesty': 0.7},
            group_acceptance=0.6
        )
        
        assert choice.choice_id == "test_choice"
        assert choice.personality_alignment['openness_appeal'] == 0.8
        assert choice.emotional_cost['anxiety'] == 0.3 