import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

# Mock the modules that will be imported by the class under test
from backend.app.services.character.voice_profile import CharacterVoiceManager, VoiceCharacteristics
from backend.app.services.character.control_token_translator import ControlTokenTranslator
from backend.app.narrative_engine.tts_integration import MicroserviceTTSProvider

# Since the file doesn't exist yet, we'll define a dummy class for type hinting
# and to allow the test file to be written. This will be replaced by the
# actual implementation.
class CharacterVoiceSynthesizer:
    def __init__(self, tts_service_url="http://localhost:8002"):
        self.voice_manager = CharacterVoiceManager()
        self.token_translator = ControlTokenTranslator()
        # The real file will import MicroserviceTTSProvider from tts_integration
        # We mock it here.
        self.tts_provider = AsyncMock(spec=MicroserviceTTSProvider)

    async def synthesize_character_speech(self, text, character, narrative_context=None):
        pass # To be implemented

@pytest.fixture
def sample_character():
    return {
        "id": "char_789",
        "name": "Integration Test Character",
        "personality": {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5
        }
    }

@pytest.mark.asyncio
async def test_synthesize_character_speech_integration(sample_character):
    """
    Tests the full integration of the character voice synthesizer.
    """
    # We will patch the dependencies of CharacterVoiceSynthesizer
    with patch('backend.app.narrative_engine.character_voice_integration.CharacterVoiceManager') as MockVoiceManager, \
         patch('backend.app.narrative_engine.character_voice_integration.ControlTokenTranslator') as MockTokenTranslator, \
         patch('backend.app.narrative_engine.character_voice_integration.MicroserviceTTSProvider') as MockTTSProvider:
        
        # Configure mocks
        mock_voice_manager = MockVoiceManager.return_value
        mock_token_translator = MockTokenTranslator.return_value
        mock_tts_provider = MockTTSProvider.return_value
        mock_tts_provider.synthesize_speech = AsyncMock(return_value=(np.zeros(100), 22050))

        mock_voice_profile = VoiceCharacteristics()
        mock_voice_manager.get_or_create_voice_profile.return_value = mock_voice_profile
        
        mock_tts_request = {
            "text": "<laugh> I am happy!",
            "character_id": "char_789",
            "emotion_intensity": 0.5,
            "force_model": "orpheus"
        }
        mock_token_translator.translate_to_tts_request.return_value = mock_tts_request
        mock_token_translator.extract_control_tokens.return_value = ("I am happy!", {"emotion": "joy"})

        # Now import the class to be tested
        from backend.app.narrative_engine.character_voice_integration import CharacterVoiceSynthesizer

        # Instantiate the synthesizer
        synthesizer = CharacterVoiceSynthesizer()

        # Call the method
        text_with_token = "I am [EMOTION:joy:0.8] happy!"
        audio, sr, metadata = await synthesizer.synthesize_character_speech(
            text=text_with_token,
            character=sample_character
        )

        # Assertions
        mock_voice_manager.get_or_create_voice_profile.assert_called_once_with(sample_character)
        mock_token_translator.translate_to_tts_request.assert_called_once_with(
            text_with_token, sample_character, mock_voice_profile
        )
        mock_tts_provider.synthesize_speech.assert_awaited_once_with(
            mock_tts_request["text"],
            sample_character,
            emotion_tags=["laugh"]
        )
        
        assert isinstance(audio, np.ndarray)
        assert sr == 22050
        assert metadata["voice_consistency_hash"] == mock_voice_profile.consistency_hash
        assert metadata["model_used"] == "orpheus"
        assert "emotion" in metadata["control_tokens_applied"] 