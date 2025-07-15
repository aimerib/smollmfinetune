"""
Voice Streaming WebSocket Router

Provides real-time voice generation and streaming via WebSocket connections.
Integrates with existing TTS system and character voice profiles.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from backend.app.narrative_engine.tts_integration import TTSOrchestrator
from backend.app.services.character.voice_profile import CharacterVoiceManager, VoiceCharacteristics
from backend.app.services.character.control_token_translator import ControlTokenTranslator
from backend.app.services.voice.phrase_cache_manager import PhraseCacheManager
from backend.app.services.voice.adaptive_quality_controller import AdaptiveQualityController
from backend.app.services.voice.quality_models import NarrativeContext, QualityLevel
from backend.app.redis_client import RedisCache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["voice-streaming"])

# Global instances (will be dependency injected in production)
_tts_orchestrator: Optional[TTSOrchestrator] = None
_voice_manager: Optional[CharacterVoiceManager] = None
_control_translator: Optional[ControlTokenTranslator] = None
_phrase_cache_manager: Optional[PhraseCacheManager] = None
_quality_controller: Optional[AdaptiveQualityController] = None


class VoiceStreamRequest(BaseModel):
    """Request model for voice streaming"""
    type: str
    text: str
    character_id: str
    emotion_context: Optional[Dict[str, Any]] = None


class ConnectionManager:
    """Manages WebSocket connections for voice streaming"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, character_id: str):
        """Accept WebSocket connection and store it"""
        await websocket.accept()
        connection_id = f"{character_id}_{id(websocket)}"
        self.active_connections[connection_id] = websocket
        logger.info(f"Voice streaming connection established: {connection_id}")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove WebSocket connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            logger.info(f"Voice streaming connection closed: {connection_id}")


# Global connection manager
connection_manager = ConnectionManager()


def get_tts_orchestrator() -> TTSOrchestrator:
    """Get TTS orchestrator instance (dependency injection point)"""
    global _tts_orchestrator
    if _tts_orchestrator is None:
        _tts_orchestrator = TTSOrchestrator()
    return _tts_orchestrator


def get_voice_manager() -> CharacterVoiceManager:
    """Get voice manager instance (dependency injection point)"""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = CharacterVoiceManager()
    return _voice_manager


def get_voice_profile(character_id: str) -> VoiceCharacteristics:
    """Get voice profile for character (dependency injection point)"""
    voice_manager = get_voice_manager()
    return voice_manager.get_or_create_voice_profile({"id": character_id, "name": character_id})


def _extract_emotion_tags(request: VoiceStreamRequest) -> list:
    """Extract emotion tags from voice stream request"""
    if not request.emotion_context:
        return []
    
    emotion_tags = []
    if "emotion" in request.emotion_context:
        emotion = request.emotion_context["emotion"]
        # Map common emotions to tags
        emotion_mapping = {
            "happy": ["<laugh>"],
            "sad": ["<sigh>"],
            "excited": ["<gasp>"],
            "tired": ["<yawn>"],
            "confused": ["<hmm>"]
        }
        emotion_tags.extend(emotion_mapping.get(emotion, []))
    
    return emotion_tags


# Voice streaming router initialized with all services integrated 


def get_control_translator() -> ControlTokenTranslator:
    """Get control token translator instance (dependency injection point)"""
    global _control_translator
    if _control_translator is None:
        _control_translator = ControlTokenTranslator()
    return _control_translator


def get_phrase_cache_manager() -> PhraseCacheManager:
    """Get phrase cache manager instance (dependency injection point)"""
    global _phrase_cache_manager
    if _phrase_cache_manager is None:
        _phrase_cache_manager = PhraseCacheManager()
    return _phrase_cache_manager


def get_quality_controller() -> AdaptiveQualityController:
    """Get adaptive quality controller instance (dependency injection point)"""
    global _quality_controller
    if _quality_controller is None:
        _quality_controller = AdaptiveQualityController()
    return _quality_controller 


@router.websocket("/stream/{character_id}")
async def voice_stream_websocket(websocket: WebSocket, character_id: str):
    """
    WebSocket endpoint for real-time voice streaming
    
    Accepts voice generation requests and streams audio chunks back in real-time.
    Maintains character voice consistency and supports emotion context.
    """
    connection_id = await connection_manager.connect(websocket, character_id)
    
    try:
        # Get dependencies
        tts_orchestrator = get_tts_orchestrator()
        voice_profile = get_voice_profile(character_id)
        phrase_cache = get_phrase_cache_manager()
        quality_controller = get_quality_controller()
        
        logger.info(f"Voice streaming session started for character: {character_id}")
        
        while True:
            # Receive voice generation request
            try:
                data = await websocket.receive_text()
                request_data = json.loads(data)
                request = VoiceStreamRequest(**request_data)
                
                logger.info(f"Processing voice request: {request.text[:50]}...")
                
                # Check enhanced phrase cache first
                cached_audio = await phrase_cache.get_cached_audio(
                    character_id=character_id,
                    text=request.text,
                    emotion_context=request.emotion_context
                )
                
                if cached_audio:
                    logger.info("Using cached audio from PhraseCacheManager")
                    await websocket.send_bytes(cached_audio)
                    continue
                
                # Create narrative context for quality control
                narrative_context = NarrativeContext(
                    tension_level=0.5,  # Default values - can be extracted from request
                    emotional_intensity=0.7 if request.emotion_context else 0.5,
                    narrative_importance=0.6,
                    dialogue_type="conversation",
                    scene_type="character_interaction",
                    character_focus=True
                )
                
                # Get optimized quality settings
                quality_decision = await quality_controller.determine_quality_level(
                    narrative_context=narrative_context,
                    character_id=character_id,
                    current_load=0.5  # Could be dynamic based on system metrics
                )
                
                logger.info(f"Quality decision: {quality_decision.quality_level} (confidence: {quality_decision.confidence_score:.2f})")
                
                # Generate audio using TTS orchestrator with quality optimization
                emotion_tags = _extract_emotion_tags(request)
                
                if hasattr(tts_orchestrator, 'synthesize_character_voice_streaming'):
                    # Use streaming method if available
                    audio_chunks = []
                    async for audio_chunk in tts_orchestrator.synthesize_character_voice_streaming(
                        text=request.text,
                        character={"id": character_id},
                        emotion_tags=emotion_tags,
                        voice_profile=voice_profile
                    ):
                        # Stream audio chunk immediately
                        await websocket.send_bytes(audio_chunk)
                        audio_chunks.append(audio_chunk)
                        
                        # Adaptive delay based on quality settings
                        delay = 0.01 if quality_decision.quality_level == QualityLevel.HIGH else 0.005
                        await asyncio.sleep(delay)
                    
                    # Cache complete audio using enhanced cache manager
                    if audio_chunks:
                        complete_audio = b''.join(audio_chunks)
                        await phrase_cache.cache_audio(
                            character_id=character_id,
                            text=request.text,
                            emotion_context=request.emotion_context or {},
                            audio_data=complete_audio
                        )
                
                else:
                    # Fallback to regular synthesis
                    mock_character = {"id": character_id}
                    
                    audio, sample_rate = await tts_orchestrator.synthesize_character_voice(
                        text=request.text,
                        character=mock_character,
                        emotion_tags=emotion_tags
                    )
                    
                    # Convert numpy array to bytes and stream
                    if hasattr(audio, 'tobytes'):
                        audio_bytes = audio.tobytes()
                    else:
                        audio_bytes = bytes(audio)
                    
                    await websocket.send_bytes(audio_bytes)
                    
                    # Cache using enhanced cache manager
                    await phrase_cache.cache_audio(
                        character_id=character_id,
                        text=request.text,
                        emotion_context=request.emotion_context or {},
                        audio_data=audio_bytes
                    )
                
                logger.info(f"Voice generation completed for: {request.text[:50]}...")
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Voice generation error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": f"Voice generation failed: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(connection_id) 