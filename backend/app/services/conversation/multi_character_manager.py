"""
Multi-Character Conversation Manager

Handles sophisticated multi-character conversations with advanced speech features,
spatial audio positioning, and dynamic conversation dynamics.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from backend.app.narrative_engine.tts_integration import TTSOrchestrator
from backend.app.services.character.voice_profile import CharacterVoiceManager

logger = logging.getLogger(__name__)


@dataclass
class Vector3D:
    """3D position vector for spatial audio"""
    x: float
    y: float
    z: float
    
    def __eq__(self, other):
        if not isinstance(other, Vector3D):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z


@dataclass
class SpatialAudioChunk:
    """Audio chunk with spatial positioning information"""
    audio_data: bytes
    position: Vector3D
    distance: float
    azimuth: float
    elevation: float


@dataclass
class DialogueTurn:
    """Represents a single turn in a multi-character dialogue"""
    character_id: str
    text: str
    emotion_context: Optional[Dict[str, float]] = None
    interrupts_previous: bool = False
    urgency_level: float = 0.5


class ConversationState:
    """Manages the state of an ongoing conversation"""
    
    def __init__(self):
        self._active_speakers: List[str] = []
        self._context: Dict[str, Any] = {}
        self._speaker_history: List[str] = []
        
    def get_active_speakers(self) -> List[str]:
        """Get list of currently active speakers"""
        return self._active_speakers.copy()
    
    def add_speaker(self, character_id: str):
        """Add a speaker to the conversation"""
        if character_id not in self._active_speakers:
            self._active_speakers.append(character_id)
            self._speaker_history.append(character_id)
    
    def remove_speaker(self, character_id: str):
        """Remove a speaker from the conversation"""
        if character_id in self._active_speakers:
            self._active_speakers.remove(character_id)
    
    def set_context(self, context: Dict[str, Any]):
        """Set conversation context"""
        self._context = context
    
    def get_context(self) -> Dict[str, Any]:
        """Get conversation context"""
        return self._context.copy()
    
    def get_speaker_history(self) -> List[str]:
        """Get the history of speakers"""
        return self._speaker_history.copy()


class VoiceScheduler:
    """Handles scheduling and timing of voice generation"""
    
    def __init__(self):
        self._schedule: List[Tuple[float, DialogueTurn]] = []
        self._current_time = 0.0
    
    def schedule_voice(self, turn: DialogueTurn) -> float:
        """Schedule a voice turn and return scheduled time"""
        if turn.interrupts_previous:
            # Immediate scheduling for interruptions
            scheduled_time = 0.0
        else:
            # Schedule after current speakers with natural pauses
            scheduled_time = self._current_time + len(self._schedule) * 2.0
        
        self._schedule.append((scheduled_time, turn))
        return scheduled_time
    
    def get_next_speaker(self) -> Optional[str]:
        """Get the next scheduled speaker"""
        if self._schedule:
            return self._schedule[0][1].character_id
        return None
    
    def advance_time(self, delta: float):
        """Advance the scheduler's internal time"""
        self._current_time += delta


class SpatialAudioEngine:
    """Handles 3D spatial audio positioning and effects"""
    
    def __init__(self):
        self.listener_position = Vector3D(0, 0, 0)
        self.character_positions: Dict[str, Vector3D] = {}
        self.hrtf_processor = None  # Would be initialized with actual HRTF processor
    
    def update_character_position(self, character_id: str, position: Vector3D):
        """Update character position for spatial audio"""
        self.character_positions[character_id] = position
        return {
            'character_id': character_id,
            'position': position,
            'distance': self.calculate_distance(self.listener_position, position)
        }
    
    def position_voice(self, audio: bytes, character_id: str) -> SpatialAudioChunk:
        """Apply 3D positioning to character voice"""
        char_position = self.character_positions.get(character_id, Vector3D(0, 0, 0))
        distance = self.calculate_distance(self.listener_position, char_position)
        azimuth = self.calculate_azimuth(self.listener_position, char_position)
        elevation = self.calculate_elevation(self.listener_position, char_position)
        
        # Apply HRTF for 3D audio (simplified for testing)
        spatial_audio = self._apply_hrtf(audio, azimuth, elevation, distance)
        
        return SpatialAudioChunk(
            audio_data=spatial_audio,
            position=char_position,
            distance=distance,
            azimuth=azimuth,
            elevation=elevation
        )
    
    def calculate_distance(self, pos1: Vector3D, pos2: Vector3D) -> float:
        """Calculate distance between two positions"""
        dx = pos2.x - pos1.x
        dy = pos2.y - pos1.y
        dz = pos2.z - pos1.z
        return (dx*dx + dy*dy + dz*dz) ** 0.5
    
    def calculate_azimuth(self, listener: Vector3D, source: Vector3D) -> float:
        """Calculate azimuth angle from listener to source"""
        import math
        dx = source.x - listener.x
        dz = source.z - listener.z
        return math.atan2(dx, dz)
    
    def calculate_elevation(self, listener: Vector3D, source: Vector3D) -> float:
        """Calculate elevation angle from listener to source"""
        import math
        dy = source.y - listener.y
        horizontal_distance = math.sqrt(
            (source.x - listener.x)**2 + (source.z - listener.z)**2
        )
        return math.atan2(dy, horizontal_distance)
    
    def _apply_hrtf(self, audio: bytes, azimuth: float, elevation: float, distance: float) -> bytes:
        """Apply Head-Related Transfer Function for 3D audio"""
        # Simplified HRTF application - in real implementation would use proper HRTF library
        return audio
    
    def get_position_info(self, character_id: str) -> Dict[str, float]:
        """Get position information for character"""
        pos = self.character_positions.get(character_id, Vector3D(0, 0, 0))
        return {"x": pos.x, "y": pos.y, "z": pos.z}


class MultiCharacterConversationManager:
    """Manages multi-character conversations with advanced speech features"""
    
    def __init__(self, platform_websocket):
        self.websocket = platform_websocket
        self.active_characters: Dict[str, Any] = {}  # character_id -> voice_model
        self.conversation_state = ConversationState()
        self.voice_scheduler = VoiceScheduler()
        self.spatial_audio = SpatialAudioEngine()
        
        # Initialize TTS and voice management
        self.tts_orchestrator = TTSOrchestrator()
        self.voice_manager = CharacterVoiceManager()
        
        logger.info("MultiCharacterConversationManager initialized")
    
    async def generate_multi_character_dialogue(self, dialogue_sequence: List[DialogueTurn]):
        """Generate multi-character conversation with advanced features"""
        logger.info(f"Generating multi-character dialogue with {len(dialogue_sequence)} turns")
        
        for turn in dialogue_sequence:
            # Add speaker to conversation state
            self.conversation_state.add_speaker(turn.character_id)
            
            # Get character voice and context
            character_voice = await self.get_character_voice(turn.character_id)
            speech_context = await self.build_speech_context(turn)
            
            # Apply conversation dynamics
            audio_params = self.calculate_audio_parameters(turn, speech_context)
            
            # Generate with streaming
            async for audio_chunk in character_voice.generate_streaming(
                text=turn.text,
                context=speech_context,
                audio_params=audio_params
            ):
                # Apply spatial positioning and effects
                positioned_audio = self.spatial_audio.position_voice(
                    audio_chunk, turn.character_id
                )
                
                # Stream to React client via WebSocket
                await self.websocket.send_audio_chunk({
                    'type': 'multi_character_audio',
                    'character_id': turn.character_id,
                    'audio_data': positioned_audio,
                    'spatial_info': self.spatial_audio.get_position_info(turn.character_id)
                })
    
    async def get_character_voice(self, character_id: str):
        """Get or create voice model for character"""
        if character_id not in self.active_characters:
            # Create mock character voice for testing
            voice_model = MockCharacterVoice(character_id)
            self.active_characters[character_id] = voice_model
        
        return self.active_characters[character_id]
    
    async def build_speech_context(self, turn: DialogueTurn) -> Dict[str, Any]:
        """Build speech context for the dialogue turn"""
        return {
            'character_id': turn.character_id,
            'emotion_context': turn.emotion_context or {},
            'conversation_state': self.conversation_state.get_context(),
            'speaker_history': self.conversation_state.get_speaker_history(),
            'interrupts_previous': turn.interrupts_previous
        }
    
    def calculate_audio_parameters(self, turn: DialogueTurn, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate audio parameters based on turn and context"""
        base_params = {
            'volume': 1.0,
            'pitch_modifier': 1.0,
            'speed_modifier': 1.0
        }
        
        # Adjust for urgency
        if turn.urgency_level > 0.7:
            base_params['speed_modifier'] = 1.2
            base_params['pitch_modifier'] = 1.1
        
        # Adjust for interruptions
        if turn.interrupts_previous:
            base_params['volume'] = 1.2
            base_params['speed_modifier'] = 1.1
        
        return base_params
    
    async def handle_conversation_interruption(self, interrupting_turn: DialogueTurn):
        """Handle natural conversation interruptions"""
        current_speakers = self.conversation_state.get_active_speakers()
        
        if current_speakers:
            # Calculate interruption probability based on relationships
            should_interrupt = self.calculate_interruption_probability(
                interrupting_turn.character_id, current_speakers
            )
            
            if should_interrupt:
                # Fade out current speakers, fade in interrupting character
                await self.execute_voice_crossfade(current_speakers, interrupting_turn)
    
    def calculate_interruption_probability(self, interrupting_char: str, current_speakers: List[str]) -> float:
        """Calculate probability of conversation interruption"""
        base_probability = 0.1
        
        # Factor in character relationships
        relationship_modifier = 0.0
        for speaker in current_speakers:
            relationship = self.get_character_relationship(interrupting_char, speaker)
            relationship_modifier += relationship.familiarity * 0.2
        
        # Factor in character personality
        personality = self.get_character_personality(interrupting_char)
        personality_modifier = personality.assertiveness * 0.3
        
        # Factor in narrative tension
        tension_modifier = self.get_narrative_tension() * 0.2
        
        probability = base_probability + relationship_modifier + personality_modifier + tension_modifier
        return min(0.8, probability)
    
    def get_character_relationship(self, char1: str, char2: str):
        """Get relationship between two characters"""
        # Mock implementation for testing
        class MockRelationship:
            def __init__(self):
                self.familiarity = 0.8
        return MockRelationship()
    
    def get_character_personality(self, character_id: str):
        """Get character personality traits"""
        # Mock implementation for testing
        class MockPersonality:
            def __init__(self):
                self.assertiveness = 0.7
        return MockPersonality()
    
    def get_narrative_tension(self) -> float:
        """Get current narrative tension level"""
        # Mock implementation for testing
        return 0.6
    
    async def execute_voice_crossfade(self, current_speakers: List[str], interrupting_turn: DialogueTurn):
        """Execute voice crossfade between speakers"""
        logger.info(f"Executing voice crossfade: {current_speakers} -> {interrupting_turn.character_id}")
        
        # Fade out current speakers
        for speaker in current_speakers:
            self.conversation_state.remove_speaker(speaker)
        
        # Fade in interrupting character
        self.conversation_state.add_speaker(interrupting_turn.character_id)
        
        # Simulate crossfade timing
        await asyncio.sleep(0.1)


class MockCharacterVoice:
    """Mock character voice for testing"""
    
    def __init__(self, character_id: str):
        self.character_id = character_id
    
    async def generate_streaming(self, text: str, context: Dict[str, Any], audio_params: Dict[str, Any]):
        """Generate streaming audio chunks"""
        # Mock streaming audio generation
        audio_chunks = [b'audio_chunk_1', b'audio_chunk_2']
        for chunk in audio_chunks:
            yield chunk
            await asyncio.sleep(0.01)  # Simulate processing time 