import pytest
from backend.app.services.character.control_token_translator import ControlTokenTranslator
from backend.app.services.character.voice_profile import VoiceCharacteristics

@pytest.fixture
def translator():
    return ControlTokenTranslator()

@pytest.fixture
def sample_character():
    return {"id": "char_456", "name": "Translator Test Character"}

@pytest.fixture
def sample_voice_profile():
    return VoiceCharacteristics(emotional_range=0.6, energy_level=0.5, preferred_model="kokoro")

def test_extract_control_tokens(translator):
    """
    Tests that control tokens are correctly extracted from a string.
    """
    text = "Hello [EMOTION:joy:0.8] world [PACE:fast]! This is a [TONE:dramatic] test."
    clean_text, params = translator.extract_control_tokens(text)
    
    assert clean_text == "Hello world! This is a test."
    assert params == {
        "emotion": "joy",
        "emotion_intensity": 0.8,
        "pace": "fast",
        "tone": "dramatic"
    }

def test_extract_no_control_tokens(translator):
    """
    Tests that text without control tokens is handled correctly.
    """
    text = "Hello world! This is a test without tokens."
    clean_text, params = translator.extract_control_tokens(text)
    assert clean_text == text
    assert params == {}

def test_translate_to_tts_request_with_emotion(translator, sample_character, sample_voice_profile):
    """
    Tests translation with an emotion token.
    """
    text = "I am so [EMOTION:joy:0.9] happy!"
    tts_request = translator.translate_to_tts_request(text, sample_character, sample_voice_profile)
    
    assert tts_request["text"] == "<laugh> I am so happy!"
    assert tts_request["force_model"] == "orpheus"
    assert tts_request["emotion_intensity"] == pytest.approx(0.9 * 0.6) # intensity * emotional_range

def test_translate_to_tts_request_with_pace(translator, sample_character, sample_voice_profile):
    """
    Tests translation with a pace token.
    """
    text = "Please go [PACE:slow]."
    tts_request = translator.translate_to_tts_request(text, sample_character, sample_voice_profile)
    
    assert tts_request["text"] == "Please go."
    assert "speaking_rate" in tts_request
    assert tts_request["speaking_rate"] == 0.4 # From pace_modifiers map

def test_translate_to_tts_request_with_tone(translator, sample_character, sample_voice_profile):
    """
    Tests translation with a tone token.
    """
    text = "He said, in a [TONE:dramatic] voice."
    tts_request = translator.translate_to_tts_request(text, sample_character, sample_voice_profile)
    
    assert tts_request["text"] == "He said, in a voice."
    assert tts_request["force_model"] == "orpheus"

def test_translate_to_tts_request_no_tokens(translator, sample_character, sample_voice_profile):
    """
    Tests translation with no control tokens, using defaults from voice profile.
    """
    text = "This is a standard line of dialogue."
    tts_request = translator.translate_to_tts_request(text, sample_character, sample_voice_profile)
    
    assert tts_request["text"] == text
    assert tts_request["force_model"] == "kokoro" # From voice profile default
    assert tts_request["emotion_intensity"] == 0.5 # From voice profile default 