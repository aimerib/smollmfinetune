---
# R6-2 🎭 Character Voice System & Control Token Integration
Status: **PENDING** 
Ring: R6
Created: 2025-01-20
---

## Goal
Implement character-specific voice management that ensures voice consistency across conversations and integrates seamlessly with our existing control token system via the Kokoro+Orpheus TTS microservice.

## Context
We have:
- ✅ Control token system with `[EMOTION:value]`, `[PACE:speed]`, `[TONE:style]` tokens
- ✅ Character personality system with Big Five traits
- ✅ TTS microservice with Kokoro+Orpheus models (from R6-1)
- ❌ **MISSING**: Character voice consistency and control token → TTS translation

## Acceptance Criteria
- [x] Each character gets consistent voice characteristics across all generated samples
- [x] Control tokens automatically translate to appropriate TTS parameters
- [x] Character personality traits influence voice selection and modulation
- [x] Voice characteristics persist in character profiles for reuse
- [x] Emotion intensity scales based on character personality and narrative context
- [x] Graceful handling of missing or invalid control tokens
- [x] Character voice evolution system for long-term narrative consistency

## Implementation Notes

### 1. Character Voice Profile System (`app/utils/character/voice_profile.py`)
Extend character system with voice characteristics:
```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import hashlib

@dataclass
class VoiceCharacteristics:
    """Voice characteristics for a character"""
    base_pitch: float = 0.5  # 0.0 = very low, 1.0 = very high
    pitch_variance: float = 0.5  # How much pitch varies (expressiveness)
    speaking_rate: float = 0.5  # 0.0 = very slow, 1.0 = very fast
    energy_level: float = 0.5  # Overall vocal energy
    emotional_range: float = 0.5  # How expressive the character is
    preferred_model: str = "kokoro"  # "kokoro" or "orpheus"
    
    # Voice evolution tracking
    evolution_history: List[Dict] = None
    consistency_hash: str = ""
    
    def __post_init__(self):
        if self.evolution_history is None:
            self.evolution_history = []
        self._update_consistency_hash()
    
    def _update_consistency_hash(self):
        """Generate hash for voice consistency tracking"""
        voice_data = f"{self.base_pitch}{self.pitch_variance}{self.speaking_rate}"
        self.consistency_hash = hashlib.md5(voice_data.encode()).hexdigest()[:8]

class CharacterVoiceManager:
    """Manages voice profiles for characters"""
    
    def __init__(self):
        self.voice_profiles: Dict[str, VoiceCharacteristics] = {}
        
    def get_or_create_voice_profile(self, character: Dict[str, Any]) -> VoiceCharacteristics:
        """Get existing voice profile or create from personality"""
        character_id = character["id"]
        
        if character_id not in self.voice_profiles:
            self.voice_profiles[character_id] = self._generate_voice_from_personality(character)
            
        return self.voice_profiles[character_id]
    
    def _generate_voice_from_personality(self, character: Dict[str, Any]) -> VoiceCharacteristics:
        """Generate voice characteristics from Big Five personality traits"""
        personality = character.get("personality", {})
        
        # Map Big Five traits to voice characteristics
        openness = personality.get("openness", 0.5)
        conscientiousness = personality.get("conscientiousness", 0.5)
        extraversion = personality.get("extraversion", 0.5)
        agreeableness = personality.get("agreeableness", 0.5)
        neuroticism = personality.get("neuroticism", 0.5)
        
        # Voice characteristic calculations
        base_pitch = 0.3 + (0.4 * extraversion)  # Extraverts tend to speak higher
        pitch_variance = 0.2 + (0.6 * openness)  # Open people are more expressive
        speaking_rate = 0.3 + (0.4 * extraversion) + (0.3 * neuroticism)  # Fast if extraverted or anxious
        energy_level = 0.2 + (0.7 * extraversion) + (0.1 * conscientiousness)
        emotional_range = 0.3 + (0.5 * openness) + (0.2 * neuroticism)
        
        # Model selection based on expressiveness
        expressiveness = (openness + extraversion + emotional_range) / 3
        preferred_model = "orpheus" if expressiveness > 0.6 else "kokoro"
        
        return VoiceCharacteristics(
            base_pitch=base_pitch,
            pitch_variance=pitch_variance,
            speaking_rate=speaking_rate,
            energy_level=energy_level,
            emotional_range=emotional_range,
            preferred_model=preferred_model
        )
    
    def save_voice_profiles(self, filepath: str):
        """Save voice profiles to file"""
        profiles_data = {}
        for char_id, profile in self.voice_profiles.items():
            profiles_data[char_id] = {
                "base_pitch": profile.base_pitch,
                "pitch_variance": profile.pitch_variance,
                "speaking_rate": profile.speaking_rate,
                "energy_level": profile.energy_level,
                "emotional_range": profile.emotional_range,
                "preferred_model": profile.preferred_model,
                "evolution_history": profile.evolution_history,
                "consistency_hash": profile.consistency_hash
            }
        
        with open(filepath, 'w') as f:
            json.dump(profiles_data, f, indent=2)
    
    def load_voice_profiles(self, filepath: str):
        """Load voice profiles from file"""
        try:
            with open(filepath, 'r') as f:
                profiles_data = json.load(f)
            
            for char_id, data in profiles_data.items():
                self.voice_profiles[char_id] = VoiceCharacteristics(**data)
        except FileNotFoundError:
            pass  # Start with empty profiles
```

### 2. Control Token Translation (`app/utils/character/control_token_translator.py`)
Translate control tokens to TTS parameters:
```python
import re
from typing import Dict, List, Optional, Tuple, Any

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
        
        # Extract emotion tokens: [EMOTION:joy:0.8]
        emotion_pattern = r'\[EMOTION:([^:]+):([0-9.]+)\]'
        emotion_matches = re.findall(emotion_pattern, text)
        
        for emotion, intensity in emotion_matches:
            parameters["emotion"] = emotion.lower()
            parameters["emotion_intensity"] = float(intensity)
            clean_text = re.sub(emotion_pattern, '', clean_text)
        
        # Extract pace tokens: [PACE:fast]
        pace_pattern = r'\[PACE:([^\]]+)\]'
        pace_matches = re.findall(pace_pattern, text)
        
        for pace in pace_matches:
            parameters["pace"] = pace.lower()
            clean_text = re.sub(pace_pattern, '', clean_text)
        
        # Extract tone tokens: [TONE:dramatic]
        tone_pattern = r'\[TONE:([^\]]+)\]'
        tone_matches = re.findall(tone_pattern, text)
        
        for tone in tone_matches:
            parameters["tone"] = tone.lower()
            clean_text = re.sub(tone_pattern, '', clean_text)
        
        # Clean up extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
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
```

### 3. Integration with Multimodal Generator (`narrative_engine/character_voice_integration.py`)
Integrate voice system with dataset generation:
```python
from app.utils.character.voice_profile import CharacterVoiceManager, VoiceCharacteristics
from app.utils.character.control_token_translator import ControlTokenTranslator
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
    
    def save_voice_profiles(self, output_dir: str):
        """Save character voice profiles for dataset consistency"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        profile_path = os.path.join(output_dir, "character_voice_profiles.json")
        self.voice_manager.save_voice_profiles(profile_path)
```

### 4. Update Multimodal Dataset Generator (`narrative_engine/synthetic_multimodal_dataset.py`)
Replace basic speech synthesizer with character-aware version:
```python
# In MultimodalDatasetGenerator.__init__()
self.character_voice_synthesizer = CharacterVoiceSynthesizer()

# In _generate_sample() method
async def _generate_sample(self, character, narrative_type, sample_index):
    # ... existing text generation ...
    
    # Character-aware speech synthesis
    audio, sr, voice_metadata = await self.character_voice_synthesizer.synthesize_character_speech(
        text=generated_text,
        character=character,
        narrative_context={
            "type": narrative_type,
            "tension": random.uniform(0.3, 0.9),
            "scene_type": narrative_type
        }
    )
    
    # Include voice metadata in sample
    sample.voice_metadata = voice_metadata
    
    return sample
```

## Guard-rails & Gotchas
- **Voice Consistency**: Always use the same voice profile hash for a character across sessions
- **Control Token Parsing**: Handle malformed tokens gracefully without breaking synthesis
- **Model Fallback**: If Orpheus is requested but unavailable, fallback to Kokoro with warning
- **Memory Usage**: Voice profiles should be lightweight and not consume excessive memory
- **Character Evolution**: Track voice changes over time but maintain core consistency
- **Concurrent Access**: Ensure thread-safe access to voice profiles during parallel generation

## TDD Instructions
1. **Voice Profile Tests**: Test personality → voice characteristic mapping
2. **Control Token Tests**: Test all control token parsing and translation scenarios  
3. **Consistency Tests**: Verify same character gets same voice across multiple calls
4. **Integration Tests**: Test full pipeline from control tokens to TTS synthesis
5. **Fallback Tests**: Test graceful degradation when models or tokens are invalid

## Success Criteria
- ✅ Characters maintain consistent voice characteristics across all generated samples
- ✅ Control tokens `[EMOTION:joy:0.8]` correctly translate to TTS parameters
- ✅ Character personality traits appropriately influence voice selection
- ✅ Voice profiles persist and reload correctly for dataset consistency
- ✅ Narrative context appropriately modulates voice characteristics
- ✅ System handles invalid/missing control tokens gracefully
- ✅ Generated datasets include voice metadata for training validation

## References
- Control tokens: `app/utils/control_tokens.py`
- Character system: `app/utils/character/`
- TTS microservice: R6-1 implementation
- Multimodal generator: `narrative_engine/synthetic_multimodal_dataset.py`
