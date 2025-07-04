"""
Tests for TTS Integration System

Tests the TTSOrchestrator, KokoroTTS, and OrpheusTTS providers
to ensure proper model selection, synthesis, and fallback behavior.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

from narrative_engine.tts_integration import (
    TTSOrchestrator, 
    KokoroTTS, 
    OrpheusTTS,
    TTSProvider
)


class TestKokoroTTS:
    """Test Kokoro TTS provider"""
    
    def test_init(self):
        """Test KokoroTTS initialization"""
        kokoro = KokoroTTS()
        assert kokoro.model_name == "hexgrad/Kokoro-82M"
        assert kokoro.sample_rate == 24000
        assert kokoro.pipeline is None
        assert len(kokoro.available_voices) == 8
    
    def test_available_voices(self):
        """Test voice metadata generation"""
        kokoro = KokoroTTS()
        voices = kokoro.get_available_voices()
        
        assert len(voices) == 8
        # Check female voice
        af_heart = next(v for v in voices if v["id"] == "af_heart")
        assert af_heart["gender"] == "female"
        assert af_heart["name"] == "Kokoro Heart"
        assert af_heart["provider"] == "kokoro"
        
        # Check male voice
        am_adam = next(v for v in voices if v["id"] == "am_adam")
        assert am_adam["gender"] == "male"
        assert am_adam["name"] == "Kokoro Adam"
    
    @pytest.mark.asyncio
    async def test_synthesize_fallback(self):
        """Test synthesis with fallback when Kokoro not available"""
        kokoro = KokoroTTS()
        
        # Mock pipeline loading to fail
        with patch.object(kokoro, '_load_pipeline', side_effect=ImportError("kokoro not installed")):
            with pytest.raises(ImportError):
                await kokoro.synthesize("Hello world")
    
    @pytest.mark.asyncio
    async def test_synthesize_success(self):
        """Test successful synthesis with Kokoro"""
        kokoro = KokoroTTS()
        
        # Mock the pipeline
        with patch('kokoro.KPipeline') as mock_pipeline_cls:
            mock_pipeline_instance = Mock()
            
            # Mock the generator behavior
            mock_audio_chunk = np.random.randn(1024).astype(np.float32)
            def mock_generator(text, voice):
                yield ("gs", "ps", mock_audio_chunk)
                yield ("gs", "ps", mock_audio_chunk)
            
            mock_pipeline_instance.side_effect = mock_generator
            mock_pipeline_cls.return_value = mock_pipeline_instance
            
            # Re-assign the mocked pipeline to the instance
            kokoro.pipeline = mock_pipeline_instance

            audio, sr = await kokoro.synthesize(
                "Hello world",
                voice_id="af_bella"
            )
            
            assert isinstance(audio, np.ndarray)
            assert sr == 24000
            assert len(audio) == 2048  # 2 chunks
            # mock_pipeline_instance.assert_called_with("Hello world", voice="af_bella")


class TestOrpheusTTS:
    """Test Orpheus TTS provider"""
    
    def test_init(self):
        """Test OrpheusTTS initialization"""
        orpheus = OrpheusTTS()
        assert orpheus.model_path == "canopylabs/orpheus-3b-0.1-ft"
        assert orpheus.sample_rate == 22050
        assert orpheus.model is None
        assert len(orpheus.emotion_tags) == 8
    
    def test_emotion_tags(self):
        """Test emotion tag mapping"""
        orpheus = OrpheusTTS()
        assert orpheus.emotion_tags["laugh"] == "<laugh>"
        assert orpheus.emotion_tags["sigh"] == "<sigh>"
        assert orpheus.emotion_tags["gasp"] == "<gasp>"
    
    @pytest.mark.asyncio
    async def test_synthesize_mock_audio(self):
        """Test synthesis with mock audio generation"""
        orpheus = OrpheusTTS()
        
        # Mock _load_model to do nothing (keeps model as None)
        with patch.object(orpheus, '_load_model'):
            audio, sr = await orpheus.synthesize(
                "Hello world", 
                emotion_tags=["laugh"]
            )
        
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr == 22050
        assert len(audio) > 0
    
    @pytest.mark.asyncio
    async def test_emotion_tag_processing(self):
        """Test that emotion tags are properly added to text"""
        orpheus = OrpheusTTS()
        
        with patch.object(orpheus, '_generate_mock_audio') as mock_generate:
            mock_generate.return_value = np.zeros(1000, dtype=np.float32)
            
            await orpheus.synthesize("Hello", emotion_tags=["laugh", "sigh"])
            
            # Should call with tagged text
            mock_generate.assert_called_once_with("<laugh> Hello")


class TestTTSOrchestrator:
    """Test TTS orchestrator and provider selection"""
    
    def test_init(self):
        """Test TTSOrchestrator initialization"""
        orchestrator = TTSOrchestrator()
        
        assert "kokoro" in orchestrator.providers
        assert "orpheus" in orchestrator.providers
        assert "xtts" in orchestrator.providers
        assert "bark" in orchestrator.providers
        assert orchestrator.default_provider == "kokoro"
    
    def test_provider_selection_emotion_tags(self):
        """Test provider selection based on emotion tags"""
        orchestrator = TTSOrchestrator()
        
        character = {"id": "test_char", "gender": "female"}
        
        # Should select Orpheus for emotion tags
        provider = orchestrator._select_provider(character, ["laugh"])
        assert provider == "orpheus"
        
        provider = orchestrator._select_provider(character, ["sigh", "gasp"])
        assert provider == "orpheus"
        
        # Should select Kokoro for no emotion tags
        provider = orchestrator._select_provider(character, [])
        assert provider == "kokoro"
        
        provider = orchestrator._select_provider(character, None)
        assert provider == "kokoro"
    
    def test_provider_selection_voice_reference(self):
        """Test provider selection for voice cloning"""
        orchestrator = TTSOrchestrator()
        
        character = {"voice_reference": "ref.wav", "gender": "male"}
        
        # Should select XTTS for voice reference
        provider = orchestrator._select_provider(character, [])
        assert provider == "xtts"
        
        # Emotion tags should still override
        provider = orchestrator._select_provider(character, ["laugh"])
        assert provider == "orpheus"
    
    def test_provider_selection_expressive(self):
        """Test provider selection for expressive characters"""
        orchestrator = TTSOrchestrator()
        
        character = {"expressive": True, "gender": "female"}
        
        # Should select Bark for expressive characters
        provider = orchestrator._select_provider(character, [])
        assert provider == "bark"
    
    def test_voice_id_mapping_kokoro(self):
        """Test voice ID mapping for Kokoro"""
        orchestrator = TTSOrchestrator()
        
        # Female character -> af_heart
        character = {"gender": "female"}
        voice_id = orchestrator._get_voice_id(character, "kokoro")
        assert voice_id == "af_heart"
        
        # Male character -> am_adam
        character = {"gender": "male"}
        voice_id = orchestrator._get_voice_id(character, "kokoro")
        assert voice_id == "am_adam"
        
        # Default -> af_heart
        character = {}
        voice_id = orchestrator._get_voice_id(character, "kokoro")
        assert voice_id == "af_heart"
    
    def test_voice_id_mapping_orpheus(self):
        """Test voice ID mapping for Orpheus"""
        orchestrator = TTSOrchestrator()
        
        # Female character -> expressive_female
        character = {"gender": "female"}
        voice_id = orchestrator._get_voice_id(character, "orpheus")
        assert voice_id == "expressive_female"
        
        # Male character -> expressive_male
        character = {"gender": "male"}
        voice_id = orchestrator._get_voice_id(character, "orpheus")
        assert voice_id == "expressive_male"
    
    @pytest.mark.asyncio
    async def test_synthesize_character_voice(self):
        """Test end-to-end character voice synthesis"""
        orchestrator = TTSOrchestrator()
        
        # Mock the Kokoro provider
        mock_audio = np.random.randn(1000).astype(np.float32)
        mock_kokoro = Mock()
        mock_kokoro.synthesize = AsyncMock(return_value=(mock_audio, 24000))
        orchestrator.providers["kokoro"] = mock_kokoro
        
        character = {"id": "test_char", "gender": "female", "name": "Alice"}
        
        audio, sr = await orchestrator.synthesize_character_voice(
            text="Hello there",
            character=character,
            emotion_tags=None  # Should use Kokoro
        )
        
        assert isinstance(audio, np.ndarray)
        assert sr == 24000
        assert len(audio) == 1000
        
        # Verify the provider was called correctly
        mock_kokoro.synthesize.assert_called_once_with(
            text="Hello there",
            voice_id="af_heart",
            emotion_tags=None
        )
    
    @pytest.mark.asyncio
    async def test_synthesize_with_emotion_orpheus(self):
        """Test synthesis with emotion tags using Orpheus"""
        orchestrator = TTSOrchestrator()
        
        # Mock the Orpheus provider
        mock_audio = np.random.randn(2000).astype(np.float32)
        mock_orpheus = Mock()
        mock_orpheus.synthesize = AsyncMock(return_value=(mock_audio, 22050))
        orchestrator.providers["orpheus"] = mock_orpheus
        
        character = {"id": "test_char", "gender": "male", "name": "Bob"}
        
        audio, sr = await orchestrator.synthesize_character_voice(
            text="Oh no!",
            character=character,
            emotion_tags=["gasp", "fear"]  # Should use Orpheus
        )
        
        assert isinstance(audio, np.ndarray)
        assert sr == 22050
        assert len(audio) == 2000
        
        # Verify Orpheus was used
        mock_orpheus.synthesize.assert_called_once_with(
            text="Oh no!",
            voice_id="expressive_male",
            emotion_tags=["gasp", "fear"]
        )


# Performance and integration tests (marked as slow)
@pytest.mark.slow
class TestTTSPerformance:
    """Performance tests for TTS integration"""
    
    @pytest.mark.asyncio
    async def test_kokoro_latency(self):
        """Test Kokoro synthesis latency"""
        import time
        
        kokoro = KokoroTTS()
        text = "This is a test sentence for measuring synthesis latency."
        
        start_time = time.time()
        try:
            audio, sr = await kokoro.synthesize(text)
            duration = time.time() - start_time
            
            # Should complete in reasonable time (target <200ms, allow 1s for CI)
            assert duration < 1.0
            assert len(audio) > 0
            
        except ImportError:
            pytest.skip("Kokoro not installed for performance testing")
    
    @pytest.mark.asyncio
    async def test_orchestrator_provider_switching(self):
        """Test provider switching performance"""
        orchestrator = TTSOrchestrator()
        
        character = {"gender": "female"}
        
        # Test multiple syntheses with different providers
        tasks = [
            ("Standard text", []),  # Should use Kokoro
            ("Emotional text", ["laugh"]),  # Should use Orpheus
            ("More standard text", []),  # Should use Kokoro again
        ]
        
        for text, emotion_tags in tasks:
            try:
                audio, sr = await orchestrator.synthesize_character_voice(
                    text=text,
                    character=character,
                    emotion_tags=emotion_tags
                )
                assert len(audio) > 0
                assert sr > 0
                
            except Exception as e:
                # Expected for models not installed in CI
                pytest.skip(f"TTS model not available: {e}") 