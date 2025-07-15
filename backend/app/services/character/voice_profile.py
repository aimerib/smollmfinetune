# This file is intentionally left blank for the RED step of TDD. 

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
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
    evolution_history: List[Dict] = field(default_factory=list)
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