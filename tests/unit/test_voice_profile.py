import pytest
from app.utils.character.voice_profile import CharacterVoiceManager, VoiceCharacteristics

@pytest.fixture
def character_voice_manager():
    return CharacterVoiceManager()

@pytest.fixture
def sample_character():
    return {
        "id": "char_123",
        "name": "Test Character",
        "personality": {
            "openness": 0.8,
            "conscientiousness": 0.4,
            "extraversion": 0.7,
            "agreeableness": 0.6,
            "neuroticism": 0.3
        }
    }

def test_generate_voice_from_personality(character_voice_manager, sample_character):
    """
    Tests that voice characteristics are generated from personality traits.
    """
    voice_profile = character_voice_manager._generate_voice_from_personality(sample_character)
    
    assert isinstance(voice_profile, VoiceCharacteristics)
    assert 0.0 <= voice_profile.base_pitch <= 1.0
    assert 0.0 <= voice_profile.pitch_variance <= 1.0
    assert 0.0 <= voice_profile.speaking_rate <= 1.0
    assert voice_profile.preferred_model in ["kokoro", "orpheus"]
    
    # Check a specific calculation based on the formula in the implementation plan
    # base_pitch = 0.3 + (0.4 * extraversion)  = 0.3 + (0.4 * 0.7) = 0.3 + 0.28 = 0.58
    assert voice_profile.base_pitch == pytest.approx(0.58)

def test_get_or_create_voice_profile_creates_new(character_voice_manager, sample_character):
    """
    Tests that a new voice profile is created if one doesn't exist.
    """
    assert len(character_voice_manager.voice_profiles) == 0
    profile = character_voice_manager.get_or_create_voice_profile(sample_character)
    assert len(character_voice_manager.voice_profiles) == 1
    assert isinstance(profile, VoiceCharacteristics)
    assert "char_123" in character_voice_manager.voice_profiles

def test_get_or_create_voice_profile_returns_existing(character_voice_manager, sample_character):
    """
    Tests that an existing voice profile is returned.
    """
    profile1 = character_voice_manager.get_or_create_voice_profile(sample_character)
    profile2 = character_voice_manager.get_or_create_voice_profile(sample_character)
    
    assert len(character_voice_manager.voice_profiles) == 1
    assert profile1 is profile2

def test_voice_consistency_hash(sample_character):
    """
    Tests that the voice consistency hash is generated and updated.
    """
    # This test doesn't fit the RED-GREEN-REFACTOR cycle well without the implementation,
    # as it depends on the hashing logic inside VoiceCharacteristics.
    # It will be implemented alongside the model.
    pass 