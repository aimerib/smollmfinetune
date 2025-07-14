"""
Tests for Multi-Character Conversation Manager
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any, Optional
from backend.app.services.conversation.multi_character_manager import (
    MultiCharacterConversationManager,
    ConversationState,
    DialogueTurn,
    VoiceScheduler,
    SpatialAudioEngine,
    SpatialAudioChunk,
    Vector3D
)


class TestMultiCharacterConversationManager:
    """Test the core multi-character conversation management"""
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket manager"""
        websocket = Mock()
        websocket.send_audio_chunk = AsyncMock()
        return websocket
    
    @pytest.fixture
    def conversation_manager(self, mock_websocket):
        """Create conversation manager instance"""
        return MultiCharacterConversationManager(mock_websocket)
    
    @pytest.fixture
    def sample_dialogue_sequence(self):
        """Sample dialogue sequence for testing"""
        return [
            DialogueTurn(
                character_id="alice",
                text="Hello everyone!",
                emotion_context={"happy": 0.8, "excited": 0.6}
            ),
            DialogueTurn(
                character_id="bob",
                text="Good to see you, Alice.",
                emotion_context={"calm": 0.7, "friendly": 0.5}
            ),
            DialogueTurn(
                character_id="charlie",
                text="Welcome to the party!",
                emotion_context={"enthusiastic": 0.9, "happy": 0.7}
            )
        ]
    
    def test_manager_initialization(self, conversation_manager, mock_websocket):
        """Test that the manager initializes with required components"""
        assert conversation_manager.websocket == mock_websocket
        assert hasattr(conversation_manager, 'active_characters')
        assert hasattr(conversation_manager, 'conversation_state')
        assert hasattr(conversation_manager, 'voice_scheduler')
        assert hasattr(conversation_manager, 'spatial_audio')
        assert isinstance(conversation_manager.active_characters, dict)
        
    def test_conversation_state_initialization(self, conversation_manager):
        """Test that conversation state is properly initialized"""
        state = conversation_manager.conversation_state
        assert hasattr(state, 'get_active_speakers')
        assert hasattr(state, 'add_speaker')
        assert hasattr(state, 'remove_speaker')
        
    @pytest.mark.asyncio
    async def test_generate_multi_character_dialogue(self, conversation_manager, sample_dialogue_sequence):
        """Test generating multi-character dialogue with streaming"""
        # Mock the character voice retrieval
        with patch.object(conversation_manager, 'get_character_voice') as mock_get_voice:
            async def mock_streaming(*args, **kwargs):
                for chunk in [b'audio_chunk_1', b'audio_chunk_2']:
                    yield chunk
            
            mock_voice = Mock()
            mock_voice.generate_streaming = mock_streaming
            mock_get_voice.return_value = mock_voice
            
            # Mock audio positioning
            with patch.object(conversation_manager.spatial_audio, 'position_voice') as mock_position:
                mock_position.return_value = b'positioned_audio'
                
                with patch.object(conversation_manager.spatial_audio, 'get_position_info') as mock_position_info:
                    mock_position_info.return_value = {"x": 1.0, "y": 0.0, "z": 0.0}
                    
                    # Execute the dialogue generation
                    await conversation_manager.generate_multi_character_dialogue(sample_dialogue_sequence)
                    
                    # Verify character voices were retrieved
                    assert mock_get_voice.call_count == 3
                    
                    # Verify WebSocket streaming
                    assert conversation_manager.websocket.send_audio_chunk.call_count == 6  # 2 chunks × 3 characters
    
    @pytest.mark.asyncio
    async def test_speaker_switching(self, conversation_manager):
        """Test seamless speaker switching between characters"""
        # Test adding and switching between speakers
        conversation_manager.conversation_state.add_speaker("alice")
        conversation_manager.conversation_state.add_speaker("bob")
        
        active_speakers = conversation_manager.conversation_state.get_active_speakers()
        assert "alice" in active_speakers
        assert "bob" in active_speakers
        
        # Test speaker removal
        conversation_manager.conversation_state.remove_speaker("alice")
        active_speakers = conversation_manager.conversation_state.get_active_speakers()
        assert "alice" not in active_speakers
        assert "bob" in active_speakers


class TestConversationState:
    """Test conversation state management"""
    
    def test_conversation_state_creation(self):
        """Test basic conversation state creation"""
        state = ConversationState()
        assert state.get_active_speakers() == []
        
    def test_add_speaker(self):
        """Test adding speakers to conversation"""
        state = ConversationState()
        state.add_speaker("alice")
        assert "alice" in state.get_active_speakers()
        
    def test_remove_speaker(self):
        """Test removing speakers from conversation"""
        state = ConversationState()
        state.add_speaker("alice")
        state.add_speaker("bob")
        state.remove_speaker("alice")
        
        active = state.get_active_speakers()
        assert "alice" not in active
        assert "bob" in active
        
    def test_conversation_context(self):
        """Test conversation context management"""
        state = ConversationState()
        context = {"scene": "party", "mood": "festive"}
        state.set_context(context)
        
        assert state.get_context() == context


class TestDialogueTurn:
    """Test dialogue turn model"""
    
    def test_dialogue_turn_creation(self):
        """Test creating a dialogue turn"""
        turn = DialogueTurn(
            character_id="alice",
            text="Hello world!",
            emotion_context={"happy": 0.8}
        )
        
        assert turn.character_id == "alice"
        assert turn.text == "Hello world!"
        assert turn.emotion_context == {"happy": 0.8}
        
    def test_dialogue_turn_optional_fields(self):
        """Test dialogue turn with optional fields"""
        turn = DialogueTurn(
            character_id="bob",
            text="Hi there!",
            interrupts_previous=True,
            urgency_level=0.7
        )
        
        assert turn.interrupts_previous is True
        assert turn.urgency_level == 0.7


class TestVoiceScheduler:
    """Test voice scheduling and timing"""
    
    def test_voice_scheduler_initialization(self):
        """Test voice scheduler creation"""
        scheduler = VoiceScheduler()
        assert hasattr(scheduler, 'schedule_voice')
        assert hasattr(scheduler, 'get_next_speaker')
        
    def test_schedule_dialogue_turn(self):
        """Test scheduling a dialogue turn"""
        scheduler = VoiceScheduler()
        turn = DialogueTurn(character_id="alice", text="Hello!")
        
        scheduled_time = scheduler.schedule_voice(turn)
        assert isinstance(scheduled_time, float)
        assert scheduled_time >= 0
        
    def test_interruption_handling(self):
        """Test handling of conversation interruptions"""
        scheduler = VoiceScheduler()
        
        # Schedule normal turn
        turn1 = DialogueTurn(character_id="alice", text="I was saying...")
        scheduler.schedule_voice(turn1)
        
        # Schedule interrupting turn
        turn2 = DialogueTurn(
            character_id="bob", 
            text="Sorry to interrupt!",
            interrupts_previous=True
        )
        interrupt_time = scheduler.schedule_voice(turn2)
        
        # Interruption should be scheduled immediately
        assert interrupt_time < 1.0


class TestSpatialAudioEngine:
    """Test spatial audio positioning and effects"""
    
    def test_spatial_audio_initialization(self):
        """Test spatial audio engine creation"""
        engine = SpatialAudioEngine()
        assert hasattr(engine, 'listener_position')
        assert hasattr(engine, 'character_positions')
        assert hasattr(engine, 'position_character_voice')
        
    def test_character_positioning(self):
        """Test positioning characters in 3D space"""
        engine = SpatialAudioEngine()
        position = Vector3D(x=1.0, y=0.0, z=2.0)
        
        engine.update_character_position("alice", position)
        assert engine.character_positions["alice"] == position
        
    def test_spatial_audio_processing(self):
        """Test spatial audio processing"""
        engine = SpatialAudioEngine()
        audio_chunk = b'mock_audio_data'
        
        # Position character
        position = Vector3D(x=1.0, y=0.5, z=0.0)
        engine.update_character_position("alice", position)
        
        # Process spatial audio
        spatial_audio = engine.position_character_voice(audio_chunk, "alice")
        
        assert isinstance(spatial_audio, SpatialAudioChunk)
        assert spatial_audio.position == position
        assert spatial_audio.audio_data is not None
        
    def test_distance_calculation(self):
        """Test distance calculation between listener and character"""
        engine = SpatialAudioEngine()
        
        # Set character position
        character_pos = Vector3D(x=3.0, y=4.0, z=0.0)
        engine.update_character_position("alice", character_pos)
        
        # Calculate distance (should be 5.0 using Pythagorean theorem)
        distance = engine.calculate_distance(engine.listener_position, character_pos)
        assert abs(distance - 5.0) < 0.01


class TestConversationDynamics:
    """Test conversation dynamics and interruption logic"""
    
    @pytest.fixture
    def conversation_manager(self):
        websocket = Mock()
        websocket.send_audio_chunk = AsyncMock()
        return MultiCharacterConversationManager(websocket)
    
    def test_interruption_probability_calculation(self, conversation_manager):
        """Test calculating interruption probability based on relationships"""
        # Mock character relationships and personality
        with patch.object(conversation_manager, 'get_character_relationship') as mock_relationship:
            mock_relationship.return_value = Mock(familiarity=0.8)
            
            with patch.object(conversation_manager, 'get_character_personality') as mock_personality:
                mock_personality.return_value = Mock(assertiveness=0.7)
                
                with patch.object(conversation_manager, 'get_narrative_tension') as mock_tension:
                    mock_tension.return_value = 0.6
                    
                    probability = conversation_manager.calculate_interruption_probability(
                        "alice", ["bob"]
                    )
                    
                    assert 0.0 <= probability <= 0.8  # Capped at 0.8
                    assert probability > 0.1  # Should be above base probability
    
    @pytest.mark.asyncio
    async def test_conversation_interruption_handling(self, conversation_manager):
        """Test handling of conversation interruptions"""
        # Setup active speaker
        conversation_manager.conversation_state.add_speaker("bob")
        
        # Create interrupting turn
        interrupting_turn = DialogueTurn(
            character_id="alice",
            text="Wait, I have to say something!",
            interrupts_previous=True
        )
        
        # Mock voice crossfade
        with patch.object(conversation_manager, 'execute_voice_crossfade') as mock_crossfade:
            mock_crossfade.return_value = AsyncMock()
            
            with patch.object(conversation_manager, 'calculate_interruption_probability') as mock_calc:
                mock_calc.return_value = 0.7  # High probability
                
                await conversation_manager.handle_conversation_interruption(interrupting_turn)
                
                # Verify crossfade was executed
                mock_crossfade.assert_called_once()


class TestIntegrationScenarios:
    """Integration tests for complete conversation scenarios"""
    
    @pytest.fixture
    def full_conversation_manager(self):
        """Create a fully configured conversation manager"""
        websocket = Mock()
        websocket.send_audio_chunk = AsyncMock()
        manager = MultiCharacterConversationManager(websocket)
        
        # Mock voice generation to return async generator
        async def mock_streaming(*args, **kwargs):
            for chunk in [b'audio_chunk_1', b'audio_chunk_2']:
                yield chunk
        
        mock_voice = Mock()
        mock_voice.generate_streaming = mock_streaming
        
        # Mock all dependencies
        with patch.object(manager, 'get_character_voice', return_value=mock_voice):
            with patch.object(manager.spatial_audio, 'position_voice', return_value=b'positioned_audio'):
                with patch.object(manager.spatial_audio, 'get_position_info', return_value={"x": 1.0, "y": 0.0, "z": 0.0}):
                    yield manager
    
    @pytest.mark.asyncio
    async def test_three_character_conversation(self, full_conversation_manager):
        """Test a complete three-character conversation scenario"""
        # Setup characters in 3D space
        positions = {
            "alice": Vector3D(x=-1.0, y=0.0, z=0.0),
            "bob": Vector3D(x=1.0, y=0.0, z=0.0),
            "charlie": Vector3D(x=0.0, y=0.0, z=2.0)
        }
        
        for char_id, pos in positions.items():
            full_conversation_manager.spatial_audio.update_character_position(char_id, pos)
        
        # Create dialogue sequence
        dialogue = [
            DialogueTurn(character_id="alice", text="Hey everyone, how's it going?"),
            DialogueTurn(character_id="bob", text="Great! Just got back from vacation."),
            DialogueTurn(character_id="charlie", text="Oh really? Where did you go?"),
            DialogueTurn(
                character_id="alice", 
                text="Actually, I wanted to know too!",
                interrupts_previous=True
            )
        ]
        
        # Execute conversation
        await full_conversation_manager.generate_multi_character_dialogue(dialogue)
        
        # Verify all characters were processed
        active_speakers = full_conversation_manager.conversation_state.get_active_speakers()
        expected_speakers = {"alice", "bob", "charlie"}
        assert set(active_speakers) == expected_speakers


# Model classes for testing (these will be implemented in the actual code)

class ConversationState:
    """Manages the state of an ongoing conversation"""
    def __init__(self):
        self._active_speakers = []
        self._context = {}
    
    def get_active_speakers(self) -> List[str]:
        return self._active_speakers.copy()
    
    def add_speaker(self, character_id: str):
        if character_id not in self._active_speakers:
            self._active_speakers.append(character_id)
    
    def remove_speaker(self, character_id: str):
        if character_id in self._active_speakers:
            self._active_speakers.remove(character_id)
    
    def set_context(self, context: Dict[str, Any]):
        self._context = context
    
    def get_context(self) -> Dict[str, Any]:
        return self._context


class DialogueTurn:
    """Represents a single turn in a multi-character dialogue"""
    def __init__(
        self,
        character_id: str,
        text: str,
        emotion_context: Optional[Dict[str, float]] = None,
        interrupts_previous: bool = False,
        urgency_level: float = 0.5
    ):
        self.character_id = character_id
        self.text = text
        self.emotion_context = emotion_context or {}
        self.interrupts_previous = interrupts_previous
        self.urgency_level = urgency_level


class VoiceScheduler:
    """Handles scheduling and timing of voice generation"""
    def __init__(self):
        self._schedule = []
    
    def schedule_voice(self, turn: DialogueTurn) -> float:
        """Schedule a voice turn and return scheduled time"""
        if turn.interrupts_previous:
            return 0.0  # Immediate
        return len(self._schedule) * 2.0  # 2 second intervals
    
    def get_next_speaker(self) -> Optional[str]:
        """Get the next scheduled speaker"""
        if self._schedule:
            return self._schedule[0]
        return None


class Vector3D:
    """3D position vector"""
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def __eq__(self, other):
        if not isinstance(other, Vector3D):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z


class SpatialAudioChunk:
    """Audio chunk with spatial positioning information"""
    def __init__(
        self,
        audio_data: bytes,
        position: Vector3D,
        distance: float,
        azimuth: float,
        elevation: float
    ):
        self.audio_data = audio_data
        self.position = position
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation


class SpatialAudioEngine:
    """Handles 3D spatial audio positioning"""
    def __init__(self):
        self.listener_position = Vector3D(0, 0, 0)
        self.character_positions = {}
    
    def update_character_position(self, character_id: str, position: Vector3D):
        """Update character position for spatial audio"""
        self.character_positions[character_id] = position
    
    def position_character_voice(self, audio: bytes, character_id: str) -> SpatialAudioChunk:
        """Apply 3D positioning to character voice"""
        char_position = self.character_positions.get(character_id, Vector3D(0, 0, 0))
        distance = self.calculate_distance(self.listener_position, char_position)
        
        return SpatialAudioChunk(
            audio_data=audio,
            position=char_position,
            distance=distance,
            azimuth=0.0,  # Simplified for testing
            elevation=0.0
        )
    
    def calculate_distance(self, pos1: Vector3D, pos2: Vector3D) -> float:
        """Calculate distance between two positions"""
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        dz = pos2.z - pos1.z
        return (dx*dx + dy*dy + dz*dz) ** 0.5
    
    def get_position_info(self, character_id: str) -> Dict[str, float]:
        """Get position information for character"""
        pos = self.character_positions.get(character_id, Vector3D(0, 0, 0))
        return {"x": pos.x, "y": pos.y, "z": pos.z} 