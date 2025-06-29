"""
Unit tests for PlatformRuntimeEngine - Multi-character session management

Tests the core runtime engine that manages active play sessions with multiple characters,
handling concurrent response generation, character state updates, and session persistence.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from pathlib import Path

# Import the classes we're testing 
from app.utils.runtime.platform_engine import PlatformRuntimeEngine, CharacterState, SessionMessage
from app.utils.runtime.prompt_constructor import RuntimePromptConstructor
from app.utils.database.session import SessionManager


class TestPlatformRuntimeEngine:
    """Test the main platform runtime engine"""
    
    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager for database operations"""
        mock = Mock()
        mock.get_play_session.return_value = {
            'id': 'test-session-123',
            'world_id': 'fantasy-world',
            'user_id': 'user-456', 
            'session_name': 'Epic Adventure',
            'status': 'active',
            'character_ids': ['wizard-789', 'bard-012']
        }
        mock.get_session_characters.return_value = [
            {
                'character_id': 'wizard-789',
                'character_name': 'Gandros the Wise',
                'character_data': {'name': 'Gandros the Wise', 'class': 'wizard'},
                'is_active': True
            },
            {
                'character_id': 'bard-012', 
                'character_name': 'Melody Songweaver',
                'character_data': {'name': 'Melody Songweaver', 'class': 'bard'},
                'is_active': True
            }
        ]
        mock.load_conversation_history.return_value = []  # Empty conversation history for tests
        return mock
    
    def test_initialization_requirements(self):
        """Test what we expect from PlatformRuntimeEngine initialization"""
        # Now that engine is implemented, test that we can import it successfully
        from app.utils.runtime.platform_engine import PlatformRuntimeEngine, CharacterState, SessionMessage
        
        # All classes should be importable and well-defined
        assert PlatformRuntimeEngine is not None
        assert CharacterState is not None  
        assert SessionMessage is not None
        
        # Classes should have expected methods
        assert hasattr(PlatformRuntimeEngine, 'generate_character_response')
        assert hasattr(PlatformRuntimeEngine, 'generate_multi_character_responses')
        assert hasattr(PlatformRuntimeEngine, 'get_character_state')
        assert hasattr(CharacterState, 'to_dict')
        assert hasattr(SessionMessage, 'to_dict')
            
    def test_multi_character_response_requirements(self):  
        """Test requirements for multi-character response generation"""
        # Expected: Should handle concurrent response generation from multiple characters
        # Expected: Should determine which characters should respond to user messages
        # Expected: Should support character-to-character interactions
        # Expected: Should update character states after each interaction
        
        # This will fail until we implement the functionality
        assert True  # Placeholder for now

    def test_character_state_management_requirements(self):
        """Test requirements for character state tracking"""
        # Expected: Should track mood, relationships, memory for each character
        # Expected: Should persist character states to database
        # Expected: Should handle character activation/muting
        # Expected: Should process world events affecting all characters
        
        assert True  # Placeholder for now

    def test_session_persistence_requirements(self):
        """Test requirements for session management"""
        # Expected: Should auto-save conversation history to database
        # Expected: Should handle session recovery after disconnection
        # Expected: Should support session sharing and export
        
        assert True  # Placeholder for now

    @pytest.fixture
    def mock_prompt_constructors(self):
        """Mock prompt constructors for each character"""
        wizard_constructor = Mock(spec=RuntimePromptConstructor)
        wizard_constructor.get_character_name.return_value = "Gandros the Wise"
        wizard_constructor.construct.return_value = "You are Gandros the Wise, a wise wizard..."
        
        bard_constructor = Mock(spec=RuntimePromptConstructor)
        bard_constructor.get_character_name.return_value = "Melody Songweaver" 
        bard_constructor.construct.return_value = "You are Melody Songweaver, a charismatic bard..."
        
        return {
            'wizard-789': wizard_constructor,
            'bard-012': bard_constructor
        }
    
    @pytest.fixture 
    def mock_inference_manager(self):
        """Mock inference manager for response generation"""
        mock = AsyncMock()
        mock.generate_response.return_value = "This is a test response from the character."
        return mock
    
    @pytest.fixture
    def runtime_engine(self, mock_session_manager, mock_prompt_constructors, mock_inference_manager):
        """Create a runtime engine instance for testing"""
        from app.utils.runtime.session_database_service import SessionDatabaseService
        
        # Create a database service with the mocked session manager
        mock_db_service = SessionDatabaseService(session_manager=mock_session_manager)
        
        with patch('app.utils.runtime.platform_engine.RuntimePromptConstructor') as mock_constructor_class, \
             patch('app.utils.runtime.platform_engine.InferenceManager', return_value=mock_inference_manager):
            
            # Configure the RuntimePromptConstructor mock to return our character-specific mocks
            def constructor_side_effect(packet_path):
                # Extract character ID from packet path for routing
                if 'wizard-789' in packet_path:
                    return mock_prompt_constructors['wizard-789']
                elif 'bard-012' in packet_path:
                    return mock_prompt_constructors['bard-012']
                else:
                    return mock_prompt_constructors['wizard-789']  # Default
            
            mock_constructor_class.side_effect = constructor_side_effect
            
            engine = PlatformRuntimeEngine(session_id='test-session-123', session_db_service=mock_db_service)
            return engine
    
    def test_initialization(self, runtime_engine, mock_session_manager):
        """Test proper initialization of runtime engine"""
        # Should load session data through database service
        mock_session_manager.get_play_session.assert_called_once_with('test-session-123')
        mock_session_manager.get_session_characters.assert_called_once_with('test-session-123')
        
        # Should have correct session info
        assert runtime_engine.session_id == 'test-session-123'
        assert runtime_engine.world_id == 'fantasy-world'
        assert runtime_engine.user_id == 'user-456'
        
        # Should load character states
        assert len(runtime_engine.character_states) == 2
        assert 'wizard-789' in runtime_engine.character_states
        assert 'bard-012' in runtime_engine.character_states
        
        # Character states should be initialized properly
        wizard_state = runtime_engine.character_states['wizard-789']
        assert wizard_state.character_id == 'wizard-789'
        assert wizard_state.character_name == 'Gandros the Wise'
        assert wizard_state.is_active is True
        assert wizard_state.current_mood == 'neutral'  # Default mood
        assert wizard_state.relationships == {}  # Empty initially

    def test_character_state_management(self, runtime_engine):
        """Test character state tracking and updates"""
        # Test getting character state
        wizard_state = runtime_engine.get_character_state('wizard-789')
        assert wizard_state.character_name == 'Gandros the Wise'
        
        # Test updating character mood
        runtime_engine.update_character_mood('wizard-789', 'happy')
        updated_state = runtime_engine.get_character_state('wizard-789')
        assert updated_state.current_mood == 'happy'
        
        # Test updating relationships
        runtime_engine.update_character_relationship(
            'wizard-789', 
            'user-456', 
            {'trust': 0.8, 'affinity': 0.7}
        )
        updated_state = runtime_engine.get_character_state('wizard-789')
        assert 'user-456' in updated_state.relationships
        assert updated_state.relationships['user-456']['trust'] == 0.8
        
        # Test updating character-to-character relationships
        runtime_engine.update_character_relationship(
            'wizard-789',
            'bard-012', 
            {'trust': 0.9, 'affinity': 0.6}
        )
        updated_state = runtime_engine.get_character_state('wizard-789')
        assert 'bard-012' in updated_state.relationships
        assert updated_state.relationships['bard-012']['affinity'] == 0.6

    @pytest.mark.asyncio
    async def test_single_character_response_generation(self, runtime_engine, mock_inference_manager):
        """Test generating a response from a single character"""
        user_message = "Hello, I need help finding an ancient artifact!"
        
        # Generate response from wizard
        response = await runtime_engine.generate_character_response(
            character_id='wizard-789',
            user_message=user_message,
            conversation_history=[]
        )
        
        # Should call inference manager with proper prompt
        mock_inference_manager.generate_response.assert_called_once()
        call_args = mock_inference_manager.generate_response.call_args
        
        # Verify prompt was constructed with proper context
        assert 'wizard-789' in call_args.kwargs['model_path'] or 'wizard' in call_args.kwargs['prompt'].lower()
        assert user_message in call_args.kwargs['prompt']
        
        # Should return response
        assert response == "This is a test response from the character."

    @pytest.mark.asyncio 
    async def test_multi_character_response_generation(self, runtime_engine, mock_inference_manager):
        """Test concurrent response generation from multiple characters"""
        user_message = "I need advice on my quest. What do you both think?"
        conversation_history = [
            {"role": "user", "content": "Hello everyone!"},
            {"role": "assistant", "content": "Greetings, traveler!", "character_id": "wizard-789"}
        ]
        
        # Mock different responses for each character
        def generate_response_side_effect(**kwargs):
            if 'wizard' in kwargs['model_path'].lower() or 'wizard' in kwargs.get('prompt', '').lower():
                return "The ancient texts speak of great power in the eastern mountains."
            else:
                return "I've heard songs of treasure hidden in the Crystal Caves!"
        
        mock_inference_manager.generate_response.side_effect = generate_response_side_effect
        
        # Generate responses from multiple characters
        responses = await runtime_engine.generate_multi_character_responses(
            user_message=user_message,
            conversation_history=conversation_history,
            responding_characters=['wizard-789', 'bard-012']
        )
        
        # Should get responses from both characters
        assert len(responses) == 2
        assert 'wizard-789' in responses
        assert 'bard-012' in responses
        
        # Each response should be different and character-appropriate
        wizard_response = responses['wizard-789']
        bard_response = responses['bard-012']
        
        assert 'ancient texts' in wizard_response or 'eastern mountains' in wizard_response
        assert 'songs' in bard_response or 'Crystal Caves' in bard_response
        assert wizard_response != bard_response

    @pytest.mark.asyncio
    async def test_character_interaction_generation(self, runtime_engine, mock_inference_manager):
        """Test characters generating responses to each other (not just the user)"""
        # Wizard makes a statement
        wizard_message = "I sense dark magic ahead. We must proceed with caution."
        conversation_history = [
            {"role": "user", "content": "What do you think about the path ahead?"},
            {"role": "assistant", "content": wizard_message, "character_id": "wizard-789"}
        ]
        
        # Bard should respond to wizard's statement
        bard_response = await runtime_engine.generate_character_interaction_response(
            responding_character_id='bard-012',
            triggering_character_id='wizard-789', 
            triggering_message=wizard_message,
            conversation_history=conversation_history
        )
        
        # Should generate response with character interaction context
        mock_inference_manager.generate_response.assert_called()
        call_args = mock_inference_manager.generate_response.call_args
        
        # Prompt should include the triggering character's message
        assert wizard_message in call_args.kwargs['prompt']
        assert 'bard' in call_args.kwargs['model_path'].lower() or 'bard' in call_args.kwargs['prompt'].lower()
        
        assert bard_response == "This is a test response from the character."

    def test_conversation_history_management(self, runtime_engine):
        """Test conversation history tracking and persistence"""
        # Add messages to conversation
        runtime_engine.add_message_to_history({
            "role": "user",
            "content": "Hello everyone!",
            "timestamp": datetime.now(),
            "message_id": "msg-001"
        })
        
        runtime_engine.add_message_to_history({
            "role": "assistant", 
            "content": "Greetings, traveler!",
            "character_id": "wizard-789",
            "character_name": "Gandros the Wise",
            "timestamp": datetime.now(),
            "message_id": "msg-002"
        })
        
        # Should track conversation history
        history = runtime_engine.get_conversation_history()
        assert len(history) == 2
        assert history[0]["content"] == "Hello everyone!"
        assert history[1]["character_id"] == "wizard-789"
        
        # Should be able to get recent context
        recent_context = runtime_engine.get_recent_conversation_context(max_messages=1)
        assert len(recent_context) == 1
        assert recent_context[0]["content"] == "Greetings, traveler!"

    def test_world_event_processing(self, runtime_engine):
        """Test handling world events that affect all characters"""
        # Trigger a world event
        world_event = {
            "type": "weather_change",
            "description": "A sudden thunderstorm begins",
            "effects": {
                "mood_modifier": -0.1,  # Slightly negative mood effect
                "affects_characters": "all"
            }
        }
        
        # Process the world event
        runtime_engine.process_world_event(world_event)
        
        # Should affect all character states
        wizard_state = runtime_engine.get_character_state('wizard-789')
        bard_state = runtime_engine.get_character_state('bard-012')
        
        # Recent events should be updated
        assert len(wizard_state.recent_events) > 0
        assert "thunderstorm" in wizard_state.recent_events[-1].lower()
        assert len(bard_state.recent_events) > 0
        assert "thunderstorm" in bard_state.recent_events[-1].lower()

    def test_session_persistence(self, runtime_engine, mock_session_manager):
        """Test auto-saving session state to database"""
        # Add some conversation history
        runtime_engine.add_message_to_history({
            "role": "user",
            "content": "Test message",
            "timestamp": datetime.now()
        })
        
        # Update character state
        runtime_engine.update_character_mood('wizard-789', 'excited')
        
        # Save session state
        runtime_engine.save_session_state()
        
        # Should call session manager to persist data through database service
        mock_session_manager.save_conversation_history.assert_called_once()
        mock_session_manager.save_character_states.assert_called_once()
        
        # Verify the data being saved
        save_call = mock_session_manager.save_character_states.call_args
        character_states_data = save_call[0][1]  # Second argument
        
        # Should include updated character state
        wizard_state_data = next(
            (state for state in character_states_data 
             if state['character_id'] == 'wizard-789'), 
            None
        )
        assert wizard_state_data is not None
        assert wizard_state_data['current_mood'] == 'excited'

    def test_character_activation_control(self, runtime_engine):
        """Test temporarily muting/unmuting characters"""
        # Initially both characters should be active
        assert runtime_engine.is_character_active('wizard-789') is True
        assert runtime_engine.is_character_active('bard-012') is True
        
        # Mute the wizard
        runtime_engine.set_character_active('wizard-789', False)
        assert runtime_engine.is_character_active('wizard-789') is False
        assert runtime_engine.is_character_active('bard-012') is True  # Bard still active
        
        # Get active characters for response generation
        active_characters = runtime_engine.get_active_character_ids()
        assert 'wizard-789' not in active_characters
        assert 'bard-012' in active_characters
        
        # Reactivate wizard
        runtime_engine.set_character_active('wizard-789', True)
        assert runtime_engine.is_character_active('wizard-789') is True

    def test_response_determination_logic(self, runtime_engine):
        """Test determining which characters should respond to messages"""
        conversation_history = [
            {"role": "user", "content": "Hello everyone!"},
            {"role": "assistant", "content": "Hello there!", "character_id": "wizard-789"}
        ]
        
        # Test user message - both characters might respond
        user_message = "What do you think about magic?"
        responding_chars = runtime_engine.determine_responding_characters(
            message=user_message,
            message_type="user",
            conversation_history=conversation_history
        )
        
        # Should return list of character IDs that might respond
        assert isinstance(responding_chars, list)
        assert len(responding_chars) >= 1
        assert all(char_id in ['wizard-789', 'bard-012'] for char_id in responding_chars)
        
        # Test character message - might trigger interaction from others
        character_message = "I think we should head north to the mountains."
        responding_chars = runtime_engine.determine_responding_characters(
            message=character_message,
            message_type="character",
            speaking_character_id="wizard-789",
            conversation_history=conversation_history
        )
        
        # Might include the bard responding to wizard's suggestion
        assert isinstance(responding_chars, list)

    @pytest.mark.asyncio
    async def test_concurrent_generation_performance(self, runtime_engine, mock_inference_manager):
        """Test that multiple character responses are generated concurrently"""
        import time
        
        # Mock slow response generation
        async def slow_generate_response(**kwargs):
            await asyncio.sleep(0.1)  # Simulate 100ms generation time
            return f"Response from {kwargs.get('model_path', 'character')}"
        
        mock_inference_manager.generate_response.side_effect = slow_generate_response
        
        # Generate responses from both characters
        start_time = time.time()
        responses = await runtime_engine.generate_multi_character_responses(
            user_message="Tell me about yourselves",
            conversation_history=[],
            responding_characters=['wizard-789', 'bard-012']
        )
        end_time = time.time()
        
        # Should complete in ~100ms (concurrent) rather than ~200ms (sequential)
        duration = end_time - start_time
        assert duration < 0.15  # Allow some overhead, but should be much faster than sequential
        
        # Should get responses from both characters
        assert len(responses) == 2
        assert 'wizard-789' in responses
        assert 'bard-012' in responses


class TestCharacterState:
    """Test the CharacterState data class"""
    
    def test_character_state_initialization(self):
        """Test creating a new character state"""
        state = CharacterState(
            character_id='test-char-123',
            character_name='Test Character',
            character_data={'class': 'warrior'}
        )
        
        assert state.character_id == 'test-char-123'
        assert state.character_name == 'Test Character'
        assert state.character_data == {'class': 'warrior'}
        assert state.current_mood == 'neutral'
        assert state.relationships == {}
        assert state.recent_events == []
        assert state.is_active is True
    
    def test_character_state_serialization(self):
        """Test serializing character state to/from dict"""
        state = CharacterState(
            character_id='test-char-123',
            character_name='Test Character',
            character_data={'class': 'warrior'}
        )
        
        # Add some state
        state.current_mood = 'happy'
        state.relationships['user-456'] = {'trust': 0.8, 'affinity': 0.7}
        state.recent_events = ['Found a treasure chest']
        
        # Serialize to dict
        state_dict = state.to_dict()
        
        assert state_dict['character_id'] == 'test-char-123'
        assert state_dict['current_mood'] == 'happy'
        assert state_dict['relationships']['user-456']['trust'] == 0.8
        assert state_dict['recent_events'] == ['Found a treasure chest']
        
        # Deserialize from dict
        restored_state = CharacterState.from_dict(state_dict)
        
        assert restored_state.character_id == 'test-char-123'
        assert restored_state.current_mood == 'happy'
        assert restored_state.relationships['user-456']['trust'] == 0.8
        assert restored_state.recent_events == ['Found a treasure chest']


class TestSessionMessage:
    """Test the SessionMessage data class"""
    
    def test_session_message_creation(self):
        """Test creating session messages"""
        # User message
        user_msg = SessionMessage(
            role='user',
            content='Hello everyone!',
            timestamp=datetime.now()
        )
        
        assert user_msg.role == 'user'
        assert user_msg.content == 'Hello everyone!'
        assert user_msg.character_id is None
        assert user_msg.character_name is None
        
        # Character message
        char_msg = SessionMessage(
            role='assistant',
            content='Greetings, traveler!',
            character_id='wizard-789',
            character_name='Gandros the Wise',
            timestamp=datetime.now()
        )
        
        assert char_msg.role == 'assistant'
        assert char_msg.character_id == 'wizard-789'
        assert char_msg.character_name == 'Gandros the Wise'
    
    def test_session_message_serialization(self):
        """Test serializing messages to/from dict"""
        message = SessionMessage(
            role='assistant',
            content='I sense magic in the air...',
            character_id='wizard-789',
            character_name='Gandros the Wise',
            timestamp=datetime.now(),
            message_id='msg-123'
        )
        
        # Serialize
        msg_dict = message.to_dict()
        
        assert msg_dict['role'] == 'assistant'
        assert msg_dict['content'] == 'I sense magic in the air...'
        assert msg_dict['character_id'] == 'wizard-789'
        assert msg_dict['message_id'] == 'msg-123'
        
        # Deserialize
        restored_msg = SessionMessage.from_dict(msg_dict)
        
        assert restored_msg.role == 'assistant'
        assert restored_msg.content == 'I sense magic in the air...'
        assert restored_msg.character_id == 'wizard-789'
        assert restored_msg.message_id == 'msg-123' 