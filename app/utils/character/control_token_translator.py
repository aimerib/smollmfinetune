# This file is intentionally left blank for the RED step of TDD. 

import re
from typing import Dict, Any, Tuple
from app.utils.character.voice_profile import VoiceCharacteristics

class ControlTokenTranslator:
    """Translates control tokens to TTS microservice parameters"""
    
    def __init__(self):
        # Emotion mapping to Orpheus tags
        self.emotion_to_orpheus = {
            "joy": "<laugh>",
            "happiness": "<chuckle>", 
            "sadness": "<sigh>",
            "fear": "<gasp>",
            "anger": "<groan>",
            "tired": "<yawn>",
            "surprise": "<gasp>",
            "disgust": "<groan>"
        }
        
        # Pace mapping to speaking rate modifiers
        self.pace_modifiers = {
            "very_slow": 0.3,
            "slow": 0.4,
            "normal": 0.5,
            "fast": 0.7,
            "very_fast": 0.9
        }
        
        # Tone mapping to model preferences
        self.tone_preferences = {
            "dramatic": "orpheus",
            "emotional": "orpheus", 
            "expressive": "orpheus",
            "neutral": "kokoro",
            "calm": "kokoro",
            "matter_of_fact": "kokoro"
        }
    
    def extract_control_tokens(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Extract control tokens from text and return clean text + parameters"""
        clean_text = text
        parameters = {}
        
        # Define patterns
        emotion_pattern = r'\[EMOTION:([^:]+):([0-9.]+)\]'
        pace_pattern = r'\[PACE:([^\]]+)\]'
        tone_pattern = r'\[TONE:([^\]]+)\]'

        # Extract emotion tokens
        emotion_matches = re.findall(emotion_pattern, clean_text)
        for emotion, intensity in emotion_matches:
            parameters["emotion"] = emotion.lower()
            parameters["emotion_intensity"] = float(intensity)
        clean_text = re.sub(emotion_pattern, '', clean_text)

        # Extract pace tokens
        pace_matches = re.findall(pace_pattern, clean_text)
        for pace in pace_matches:
            parameters["pace"] = pace.lower()
        clean_text = re.sub(pace_pattern, '', clean_text)
        
        # Extract tone tokens
        tone_matches = re.findall(tone_pattern, clean_text)
        for tone in tone_matches:
            parameters["tone"] = tone.lower()
        clean_text = re.sub(tone_pattern, '', clean_text)

        # Clean up extra whitespace more robustly
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        # Remove space before punctuation
        clean_text = re.sub(r'\s([?.!,])', r'\1', clean_text)
        
        return clean_text, parameters
    
    def translate_to_tts_request(
        self, 
        text: str, 
        character: Dict[str, Any],
        voice_profile: VoiceCharacteristics
    ) -> Dict[str, Any]:
        """Translate text + character + voice profile to TTS request"""
        
        # Extract control tokens
        clean_text, control_params = self.extract_control_tokens(text)
        
        # Start with base voice characteristics
        tts_request = {
            "text": clean_text,
            "character_id": character["id"],
            "emotion_intensity": voice_profile.energy_level
        }
        
        # Apply control token overrides
        if "emotion" in control_params:
            emotion = control_params["emotion"]
            intensity = control_params.get("emotion_intensity", 0.7)
            
            # Add Orpheus emotion tag if applicable
            if emotion in self.emotion_to_orpheus:
                orpheus_tag = self.emotion_to_orpheus[emotion]
                tts_request["text"] = f"{orpheus_tag} {clean_text}"
                tts_request["force_model"] = "orpheus"
            
            # Scale emotion intensity with character traits
            base_intensity = intensity * voice_profile.emotional_range
            tts_request["emotion_intensity"] = min(1.0, base_intensity)
        
        # Apply pace modifications
        if "pace" in control_params:
            pace = control_params["pace"]
            if pace in self.pace_modifiers:
                # Modify speaking rate (this would be used by TTS service)
                tts_request["speaking_rate"] = self.pace_modifiers[pace]
        
        # Apply tone preferences
        if "tone" in control_params:
            tone = control_params["tone"]
            if tone in self.tone_preferences:
                tts_request["force_model"] = self.tone_preferences[tone]
        
        # Default model selection if not overridden
        if "force_model" not in tts_request:
            tts_request["force_model"] = voice_profile.preferred_model
        
        return tts_request 