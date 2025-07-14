"""
Tests for Agent Think Functionality

Testing the enhanced agent.think() method that returns ThinkResult
containing both action and subtext for the Iceberg Model.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime

from backend.app.narrative_engine.agent import BaseAgent, Perception, MoveToAction, SpeakToAction
from backend.app.narrative_engine.types import ThinkResult, SubtextParseError
from backend.app.narrative_engine.state_manager import StateManager


class TestAgentThink:
    """Test the enhanced think method with subtext generation"""
    
    @pytest.fixture
    def mock_narrative_model(self):
        """Create a mock narrative model for testing"""
        model = AsyncMock()
        return model
    
    @pytest.fixture
    def test_agent(self, mock_narrative_model):
        """Create a test agent with mocked model"""
        return BaseAgent(
            agent_id="test_agent",
            narrative_model=mock_narrative_model,
            goals=["Test goal"],
            personality_traits={"openness": 0.8}
        )
    
    @pytest.fixture
    def test_perception(self):
        """Create a test perception object"""
        return Perception(
            agent_id="test_agent",
            current_location="Test Location",
            nearby_agents=["other_agent"],
            recent_events=["test_event"],
            agent_state={"mood": "curious"},
            world_context={"time_of_day": "morning"}
        )
    
    
    async def test_think_returns_think_result(self, test_agent, test_perception, mock_narrative_model):
        """Test that think method returns ThinkResult object"""
        # Mock model response with valid subtext/action tags
        mock_narrative_model.generate_with_control.return_value = {
            'generated_text': """
            [SUBTEXT]
            I wonder what's happening here.
            [/SUBTEXT]
            
            [ACTION]
            I will move to the Forest.
            [/ACTION]
            """
        }
        
        result = await test_agent.think(test_perception)
        
        assert isinstance(result, ThinkResult)
        assert isinstance(result.action, MoveToAction)
        assert result.subtext == "I wonder what's happening here."
        assert result.action.target_location == "Forest"
    
    
    async def test_think_with_valid_subtext_and_action(self, test_agent, test_perception, mock_narrative_model):
        """Test parsing of valid subtext and action from model output"""
        mock_narrative_model.generate_with_control.return_value = {
            'generated_text': """
            Character considers the situation.
            
            [SUBTEXT]
            This person seems suspicious. I should be careful around them.
            [/SUBTEXT]
            
            [ACTION]
            I will speak to Clara and greet her politely.
            [/ACTION]
            """
        }
        
        result = await test_agent.think(test_perception)
        
        assert result.subtext == "This person seems suspicious. I should be careful around them."
        assert isinstance(result.action, SpeakToAction)
        assert "clara" in result.action.target_agent_id.lower()
    
    
    async def test_think_fallback_on_parse_error(self, test_agent, test_perception, mock_narrative_model):
        """Test fallback behavior when subtext parsing fails"""
        # Mock model response with malformed tags
        mock_narrative_model.generate_with_control.return_value = {
            'generated_text': "I will move to the village but no proper tags here."
        }
        
        result = await test_agent.think(test_perception)
        
        # Should fallback gracefully with empty subtext
        assert isinstance(result, ThinkResult)
        assert result.subtext == ""
        assert isinstance(result.action, MoveToAction)
    
    
    async def test_think_missing_subtext_graceful_fallback(self, test_agent, test_perception, mock_narrative_model):
        """Test graceful fallback when only ACTION tags are present"""
        mock_narrative_model.generate_with_control.return_value = {
            'generated_text': """
            [ACTION]
            I will move to the Village Square.
            [/ACTION]
            """
        }
        
        result = await test_agent.think(test_perception)
        
        assert result.subtext == ""  # Graceful fallback
        assert isinstance(result.action, MoveToAction)
        assert result.action.target_location == "Village"
    
    
    async def test_think_without_narrative_model(self):
        """Test fallback behavior when no narrative model is provided"""
        agent = BaseAgent(
            agent_id="test_agent",
            narrative_model=None,  # No model
            goals=["Test goal"]
        )
        
        perception = Perception(
            agent_id="test_agent",
            current_location="Test Location",
            nearby_agents=["other_agent"]
        )
        
        result = await agent.think(perception)
        
        assert isinstance(result, ThinkResult)
        assert result.subtext == ""  # No model means no subtext
        assert isinstance(result.action, SpeakToAction)  # Should use fallback logic
    
    
    async def test_think_model_exception_handling(self, test_agent, test_perception, mock_narrative_model):
        """Test error handling when model call fails"""
        # Mock model to raise an exception
        mock_narrative_model.generate_with_control.side_effect = Exception("Model error")
        
        result = await test_agent.think(test_perception)
        
        # Should gracefully fall back
        assert isinstance(result, ThinkResult)
        assert result.subtext == ""
        assert isinstance(result.action, SpeakToAction)  # Fallback behavior
    
    
    async def test_think_prompt_enhancement(self, test_agent, test_perception, mock_narrative_model):
        """Test that the prompt is enhanced with subtext tags"""
        mock_narrative_model.generate_with_control.return_value = {
            'generated_text': """
            [SUBTEXT]
            Test thought.
            [/SUBTEXT]
            
            [ACTION]
            I will move to the Forest.
            [/ACTION]
            """
        }
        
        await test_agent.think(test_perception)
        
        # Verify model was called with enhanced prompt
        mock_narrative_model.generate_with_control.assert_called_once()
        call_args = mock_narrative_model.generate_with_control.call_args
        
        # The enhanced prompt should contain the subtext/action tags
        prompt = call_args[1]['user_input']
        assert "[SUBTEXT]" in prompt
        assert "[ACTION]" in prompt
        assert "inner thoughts" in prompt or "private" in prompt
    
    
    async def test_think_result_validation(self, test_agent, test_perception, mock_narrative_model):
        """Test that ThinkResult validation works correctly"""
        mock_narrative_model.generate_with_control.return_value = {
            'generated_text': """
            [SUBTEXT]
            Valid subtext content.
            [/SUBTEXT]
            
            [ACTION]
            I will move to the Forest.
            [/ACTION]
            """
        }
        
        result = await test_agent.think(test_perception)
        
        assert result.validate() is True
        assert isinstance(result.timestamp, datetime)
    
    
    async def test_think_result_serialization(self, test_agent, test_perception, mock_narrative_model):
        """Test that ThinkResult can be serialized to dict"""
        mock_narrative_model.generate_with_control.return_value = {
            'generated_text': """
            [SUBTEXT]
            Serializable subtext.
            [/SUBTEXT]
            
            [ACTION]
            I will move to the Forest.
            [/ACTION]
            """
        }
        
        result = await test_agent.think(test_perception)
        result_dict = result.to_dict()
        
        assert "action" in result_dict
        assert "subtext" in result_dict
        assert "timestamp" in result_dict
        assert result_dict["subtext"] == "Serializable subtext."
    
    
    async def test_think_increased_token_limit(self, test_agent, test_perception, mock_narrative_model):
        """Test that token limit is increased for subtext + action generation"""
        mock_narrative_model.generate_with_control.return_value = {
            'generated_text': """
            [SUBTEXT]
            Test.
            [/SUBTEXT]
            
            [ACTION]
            I will move to the Forest.
            [/ACTION]
            """
        }
        
        await test_agent.think(test_perception)
        
        call_args = mock_narrative_model.generate_with_control.call_args
        max_tokens = call_args[1]['max_new_tokens']
        
        # Should be increased from original 150 to 200 for dual output
        assert max_tokens == 200 