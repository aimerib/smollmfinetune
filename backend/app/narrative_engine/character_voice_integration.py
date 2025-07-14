# This file is intentionally left blank for the RED step of TDD. 

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from backend.app.services.character.voice_profile import CharacterVoiceManager, VoiceCharacteristics
from backend.app.services.character.control_token_translator import ControlTokenTranslator
from .tts_integration import MicroserviceTTSProvider

class CharacterVoiceSynthesizer:
    """Handles character-aware speech synthesis for multimodal datasets"""
    
    def __init__(self, tts_service_url: str = "http://localhost:8002"):
        self.voice_manager = CharacterVoiceManager()
        self.token_translator = ControlTokenTranslator()
        self.tts_provider = MicroserviceTTSProvider(tts_service_url)
        
    async def synthesize_character_speech(
        self, 
        text: str, 
        character: Dict[str, Any],
        narrative_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """Synthesize speech with character voice consistency"""
        
        # Get or create voice profile for character
        voice_profile = self.voice_manager.get_or_create_voice_profile(character)
        
        # Translate control tokens to TTS parameters
        tts_request = self.token_translator.translate_to_tts_request(
            text, character, voice_profile
        )
        
        # Add narrative context modulation
        if narrative_context:
            tts_request = self._apply_narrative_context(tts_request, narrative_context, voice_profile)
        
        # Synthesize speech
        audio, sample_rate = await self.tts_provider.synthesize_speech(
            tts_request["text"],
            character,
            emotion_tags=self._extract_emotion_tags(tts_request["text"])
        )
        
        # Track voice usage for consistency
        voice_metadata = {
            "voice_consistency_hash": voice_profile.consistency_hash,
            "model_used": tts_request.get("force_model", voice_profile.preferred_model),
            "emotion_intensity": tts_request["emotion_intensity"],
            "control_tokens_applied": self.token_translator.extract_control_tokens(text)[1]
        }
        
        return audio, sample_rate, voice_metadata
    
    def _apply_narrative_context(
        self, 
        tts_request: Dict[str, Any], 
        context: Dict[str, Any], 
        voice_profile: VoiceCharacteristics
    ) -> Dict[str, Any]:
        """Apply narrative context to modulate voice characteristics"""
        
        tension_level = context.get("tension", 0.5)
        scene_type = context.get("scene_type", "dialogue")
        
        # Increase emotion intensity during high tension scenes
        if tension_level > 0.7:
            current_intensity = tts_request.get("emotion_intensity", 0.5)
            tts_request["emotion_intensity"] = min(1.0, current_intensity * 1.3)
        
        # Use Orpheus for action scenes regardless of default preference
        if scene_type == "action" and voice_profile.emotional_range > 0.6:
            tts_request["force_model"] = "orpheus"
        
        return tts_request
    
    def _extract_emotion_tags(self, text: str) -> List[str]:
        """Extract emotion tags from processed text"""
        emotion_tags = []
        for tag in ["<laugh>", "<chuckle>", "<sigh>", "<gasp>", "<groan>", "<yawn>"]:
            if tag in text:
                emotion_tags.append(tag.strip("<>"))
        return emotion_tags 