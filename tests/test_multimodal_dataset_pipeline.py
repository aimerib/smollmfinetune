"""
Tests for Multimodal Dataset Pipeline

Tests the comprehensive text-speech aligned dataset processing pipeline.
"""

import pytest
import numpy as np
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from backend.app.services.dataset.multimodal_dataset_pipeline import (
    MultimodalDatasetPipeline,
    CharacterVoiceRegistry,
    SpeechPreprocessor, 
    ForcedAligner,
    TextSpeechAlignment,
    MultimodalTrainingSample,
    create_multimodal_pipeline,
    create_sample_dataset_config
)


class TestCharacterVoiceRegistry:
    """Test character voice registry functionality"""
    
    def test_registry_initialization(self):
        """Test that registry initializes correctly"""
        registry = CharacterVoiceRegistry()
        
        assert registry.character_voices == {}
        assert registry.character_to_idx == {}
        assert registry.idx_to_character == {}
        assert registry._next_idx == 0
    
    def test_register_character_new(self):
        """Test registering a new character"""
        registry = CharacterVoiceRegistry()
        voice_profile = {"pitch_mean": 220.0, "voice_style": "friendly"}
        
        idx = registry.register_character("alice", voice_profile)
        
        assert idx == 0
        assert registry.character_to_idx["alice"] == 0
        assert registry.idx_to_character[0] == "alice"
        assert registry.character_voices["alice"] == voice_profile
        assert registry._next_idx == 1
    
    def test_register_character_existing(self):
        """Test registering an existing character updates voice profile"""
        registry = CharacterVoiceRegistry()
        
        # Register first time
        voice_profile1 = {"pitch_mean": 220.0}
        idx1 = registry.register_character("alice", voice_profile1)
        
        # Register again with different profile
        voice_profile2 = {"pitch_mean": 240.0, "voice_style": "excited"}
        idx2 = registry.register_character("alice", voice_profile2)
        
        assert idx1 == idx2 == 0  # Same index
        assert registry.character_voices["alice"] == voice_profile2  # Updated profile
        assert registry._next_idx == 1  # Index counter doesn't increment
    
    def test_register_multiple_characters(self):
        """Test registering multiple different characters"""
        registry = CharacterVoiceRegistry()
        
        alice_idx = registry.register_character("alice", {"pitch": 220})
        bob_idx = registry.register_character("bob", {"pitch": 150})
        charlie_idx = registry.register_character("charlie", {"pitch": 180})
        
        assert alice_idx == 0
        assert bob_idx == 1
        assert charlie_idx == 2
        assert len(registry.character_to_idx) == 3
        assert registry._next_idx == 3
    
    def test_get_character_idx_existing(self):
        """Test getting index for existing character"""
        registry = CharacterVoiceRegistry()
        registry.register_character("alice", {"pitch": 220})
        
        idx = registry.get_character_idx("alice")
        assert idx == 0
    
    def test_get_character_idx_nonexistent(self):
        """Test getting index for non-existent character returns None"""
        registry = CharacterVoiceRegistry()
        
        idx = registry.get_character_idx("nonexistent")
        assert idx is None
    
    def test_get_voice_profile_existing(self):
        """Test getting voice profile for existing character"""
        registry = CharacterVoiceRegistry()
        voice_profile = {"pitch_mean": 220.0, "voice_style": "friendly"}
        registry.register_character("alice", voice_profile)
        
        profile = registry.get_voice_profile("alice")
        assert profile == voice_profile
    
    def test_get_voice_profile_nonexistent(self):
        """Test getting voice profile for non-existent character returns None"""
        registry = CharacterVoiceRegistry()
        
        profile = registry.get_voice_profile("nonexistent")
        assert profile is None


class TestSpeechPreprocessor:
    """Test speech preprocessing functionality"""
    
    def test_preprocessor_initialization(self):
        """Test that preprocessor initializes with correct defaults"""
        preprocessor = SpeechPreprocessor()
        
        assert preprocessor.sample_rate == 22050
        assert preprocessor.n_mels == 80
        assert preprocessor.hop_length == 256
        assert preprocessor.quantization_bits == 4
        assert preprocessor.quantization_levels == 16
    
    def test_preprocessor_custom_config(self):
        """Test preprocessor with custom configuration"""
        preprocessor = SpeechPreprocessor(
            sample_rate=16000,
            n_mels=128,
            quantization_bits=8
        )
        
        assert preprocessor.sample_rate == 16000
        assert preprocessor.n_mels == 128
        assert preprocessor.quantization_bits == 8
        assert preprocessor.quantization_levels == 256
    
    @patch('librosa.feature.melspectrogram')
    @patch('librosa.power_to_db')
    def test_extract_mel_spectrogram(self, mock_power_to_db, mock_melspectrogram):
        """Test mel-spectrogram extraction"""
        preprocessor = SpeechPreprocessor()
        audio = np.random.randn(22050)  # 1 second of audio
        
        # Mock librosa functions
        mock_mel = np.random.rand(80, 100)  # 80 mel bins, 100 time frames
        mock_melspectrogram.return_value = mock_mel
        mock_log_mel = np.random.rand(80, 100)
        mock_power_to_db.return_value = mock_log_mel
        
        result = preprocessor.extract_mel_spectrogram(audio)
        
        # Check that librosa functions were called correctly
        mock_melspectrogram.assert_called_once_with(
            y=audio,
            sr=22050,
            n_mels=80,
            hop_length=256,
            win_length=1024,
            n_fft=1024
        )
        mock_power_to_db.assert_called_once()
        
        # Result should be transposed (time_frames, mel_bins)
        assert result.shape == (100, 80)
    
    def test_quantize_mel_spectrogram(self):
        """Test mel-spectrogram quantization"""
        preprocessor = SpeechPreprocessor(quantization_bits=4)
        mel_spec = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.9]])
        
        quantized = preprocessor.quantize_mel_spectrogram(mel_spec)
        
        # Should be quantized to 16 levels (0-15)
        expected = np.array([[0, 8, 15], [4, 11, 14]], dtype=np.int32)
        np.testing.assert_array_equal(quantized, expected)
        assert quantized.dtype == np.int32
    
    def test_get_frame_times(self):
        """Test frame time calculation"""
        preprocessor = SpeechPreprocessor(sample_rate=22050, hop_length=256)
        audio_length = 22050  # 1 second
        
        frame_times = preprocessor.get_frame_times(audio_length)
        
        # Should have correct number of frames
        expected_frames = (audio_length - 1024) // 256 + 1
        assert len(frame_times) == expected_frames
        
        # Times should be in seconds
        assert frame_times[0] == 0.0
        assert abs(frame_times[1] - (256 / 22050)) < 1e-6


class TestTextSpeechAlignment:
    """Test text-speech alignment functionality"""
    
    def test_alignment_creation(self):
        """Test creating text-speech alignment"""
        tokens = ["hello", "world"]
        start_times = [0.0, 0.5]
        end_times = [0.5, 1.0]
        mel_frames = np.random.rand(100, 80)
        frame_times = [i * 0.01 for i in range(100)]
        
        alignment = TextSpeechAlignment(
            text_tokens=tokens,
            token_start_times=start_times,
            token_end_times=end_times,
            mel_frames=mel_frames,
            frame_times=frame_times,
            character_id="alice",
            total_duration=1.0
        )
        
        assert alignment.text_tokens == tokens
        assert alignment.character_id == "alice"
        assert alignment.total_duration == 1.0
        assert alignment.mel_frames.shape == (100, 80)
    
    def test_get_token_frame_alignment(self):
        """Test getting token-to-frame alignment mapping"""
        alignment = TextSpeechAlignment(
            text_tokens=["hello", "world"],
            token_start_times=[0.0, 0.5],
            token_end_times=[0.5, 1.0],
            mel_frames=np.random.rand(100, 80),
            frame_times=[i * 0.01 for i in range(100)],  # 0.00, 0.01, 0.02, ..., 0.99
            character_id="alice",
            total_duration=1.0
        )
        
        token_frame_alignment = alignment.get_token_frame_alignment()
        
        assert len(token_frame_alignment) == 2  # Two tokens
        
        # First token (0.0-0.5s) should align with frames 0-49
        token_0_idx, token_0_frames = token_frame_alignment[0]
        assert token_0_idx == 0
        assert len(token_0_frames) == 50  # Frames 0-49
        
        # Second token (0.5-1.0s) should align with frames 50-99
        token_1_idx, token_1_frames = token_frame_alignment[1]
        assert token_1_idx == 1
        assert len(token_1_frames) == 50  # Frames 50-99


class TestForcedAligner:
    """Test forced alignment functionality"""
    
    def test_aligner_initialization(self):
        """Test that aligner initializes correctly"""
        aligner = ForcedAligner()
        assert aligner.alignment_model is None
    
    @pytest.mark.asyncio
    @patch('librosa.load')
    async def test_align_text_speech(self, mock_librosa_load):
        """Test text-speech alignment"""
        aligner = ForcedAligner()
        text = "hello world test"
        audio = np.random.randn(22050)  # 1 second
        sample_rate = 22050
        
        # Mock librosa functions in SpeechPreprocessor
        with patch.object(SpeechPreprocessor, 'extract_mel_spectrogram') as mock_extract:
            with patch.object(SpeechPreprocessor, 'get_frame_times') as mock_frame_times:
                mock_extract.return_value = np.random.rand(100, 80)
                mock_frame_times.return_value = [i * 0.01 for i in range(100)]
                
                alignment = await aligner.align_text_speech(text, audio, sample_rate)
                
                assert len(alignment.text_tokens) == 3  # "hello", "world", "test"
                assert len(alignment.token_start_times) == 3
                assert len(alignment.token_end_times) == 3
                assert alignment.mel_frames.shape == (100, 80)
                assert alignment.total_duration == 1.0
                
                # Check token timing distribution
                assert alignment.token_start_times[0] == 0.0
                assert abs(alignment.token_end_times[0] - (1.0 / 3)) < 1e-6
                assert abs(alignment.token_start_times[1] - (1.0 / 3)) < 1e-6


class TestMultimodalTrainingSample:
    """Test multimodal training sample data structure"""
    
    def test_sample_creation(self):
        """Test creating multimodal training sample"""
        sample = MultimodalTrainingSample(
            input_ids=torch.tensor([1, 2, 3]),
            attention_mask=torch.tensor([1, 1, 1]),
            text_labels=torch.tensor([2, 3, 4]),
            mel_frames=torch.randn(100, 80),
            speech_labels=torch.randint(0, 16, (100, 80)),
            speech_attention_mask=torch.ones(100),
            text_to_speech_alignment=torch.randn(3, 100),
            character_id="alice",
            character_embedding_idx=0,
            sample_id="alice_12345",
            duration=1.0
        )
        
        assert sample.input_ids.shape == (3,)
        assert sample.mel_frames.shape == (100, 80)
        assert sample.character_id == "alice"
        assert sample.character_embedding_idx == 0
        assert sample.duration == 1.0
    
    def test_sample_with_optional_fields(self):
        """Test sample with optional control and memory labels"""
        sample = MultimodalTrainingSample(
            input_ids=torch.tensor([1, 2, 3]),
            attention_mask=torch.tensor([1, 1, 1]),
            text_labels=torch.tensor([2, 3, 4]),
            mel_frames=torch.randn(100, 80),
            speech_labels=torch.randint(0, 16, (100, 80)),
            speech_attention_mask=torch.ones(100),
            text_to_speech_alignment=torch.randn(3, 100),
            character_id="alice",
            character_embedding_idx=0,
            control_labels=torch.tensor([1, 0, 1]),
            memory_labels=torch.tensor([0.5, 0.8, 0.2])
        )
        
        assert sample.control_labels is not None
        assert sample.memory_labels is not None
        assert sample.control_labels.shape == (3,)
        assert sample.memory_labels.shape == (3,)


class TestMultimodalDatasetPipeline:
    """Test the main multimodal dataset pipeline"""
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly"""
        pipeline = MultimodalDatasetPipeline()
        
        assert pipeline.tokenizer is None
        assert pipeline.max_text_length == 512
        assert pipeline.max_speech_length == 1000
        assert isinstance(pipeline.character_registry, CharacterVoiceRegistry)
        assert isinstance(pipeline.forced_aligner, ForcedAligner)
        assert isinstance(pipeline.speech_preprocessor, SpeechPreprocessor)
    
    def test_pipeline_custom_config(self):
        """Test pipeline with custom configuration"""
        registry = CharacterVoiceRegistry()
        mock_tokenizer = Mock()
        
        pipeline = MultimodalDatasetPipeline(
            tokenizer=mock_tokenizer,
            max_text_length=256,
            max_speech_length=500,
            character_registry=registry
        )
        
        assert pipeline.tokenizer == mock_tokenizer
        assert pipeline.max_text_length == 256
        assert pipeline.max_speech_length == 500
        assert pipeline.character_registry == registry
    
    def test_create_alignment_matrix(self):
        """Test cross-modal alignment matrix creation"""
        pipeline = MultimodalDatasetPipeline()
        
        # Mock token-frame alignment: token 0 -> frames [0, 1], token 1 -> frames [2, 3]
        token_frame_alignment = [(0, [0, 1]), (1, [2, 3])]
        text_length = 3
        speech_length = 5
        
        alignment_matrix = pipeline._create_alignment_matrix(
            token_frame_alignment, text_length, speech_length
        )
        
        assert alignment_matrix.shape == (3, 5)
        
        # Check alignment for token 0 (frames 0, 1)
        assert alignment_matrix[0, 0] == 0.5  # Normalized
        assert alignment_matrix[0, 1] == 0.5
        assert alignment_matrix[0, 2] == 0.0
        
        # Check alignment for token 1 (frames 2, 3)
        assert alignment_matrix[1, 2] == 0.5
        assert alignment_matrix[1, 3] == 0.5
        assert alignment_matrix[1, 0] == 0.0
        
        # Token 2 should have no alignment (all zeros, normalized to prevent div by zero)
        assert torch.allclose(alignment_matrix[2], torch.ones(5) / 5)  # Uniform distribution
    
    @pytest.mark.asyncio
    @patch('librosa.load')
    async def test_process_text_speech_pair_no_tokenizer(self, mock_librosa_load):
        """Test processing text-speech pair without tokenizer"""
        pipeline = MultimodalDatasetPipeline()
        
        # Mock audio loading
        mock_audio = np.random.randn(22050)
        mock_librosa_load.return_value = (mock_audio, 22050)
        
        # Mock forced alignment
        mock_alignment = TextSpeechAlignment(
            text_tokens=["hello", "world"],
            token_start_times=[0.0, 0.5],
            token_end_times=[0.5, 1.0],
            mel_frames=np.random.rand(100, 80),
            frame_times=[i * 0.01 for i in range(100)],
            character_id="alice",
            total_duration=1.0
        )
        
        with patch.object(pipeline.forced_aligner, 'align_text_speech', return_value=mock_alignment):
            sample = await pipeline.process_text_speech_pair(
                text="hello world",
                audio_path="/fake/path.wav",
                character_id="alice",
                voice_profile={"pitch": 220}
            )
            
            assert isinstance(sample, MultimodalTrainingSample)
            assert sample.character_id == "alice"
            assert sample.character_embedding_idx == 0  # First character registered
            assert sample.input_ids.shape[0] == pipeline.max_text_length
            assert sample.mel_frames.shape == (pipeline.max_speech_length, 80)
            assert sample.duration == 1.0
    
    @pytest.mark.asyncio
    @patch('librosa.load')
    async def test_process_text_speech_pair_with_tokenizer(self, mock_librosa_load):
        """Test processing text-speech pair with tokenizer"""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 0, 0]]),  # Padded to length 5
            'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])
        }
        
        pipeline = MultimodalDatasetPipeline(tokenizer=mock_tokenizer, max_text_length=5)
        
        # Mock audio loading
        mock_audio = np.random.randn(22050)
        mock_librosa_load.return_value = (mock_audio, 22050)
        
        # Mock forced alignment
        mock_alignment = TextSpeechAlignment(
            text_tokens=["hello", "world"],
            token_start_times=[0.0, 0.5],
            token_end_times=[0.5, 1.0],
            mel_frames=np.random.rand(50, 80),  # Shorter than max_speech_length
            frame_times=[i * 0.02 for i in range(50)],
            character_id="alice",
            total_duration=1.0
        )
        
        with patch.object(pipeline.forced_aligner, 'align_text_speech', return_value=mock_alignment):
            sample = await pipeline.process_text_speech_pair(
                text="hello world",
                audio_path="/fake/path.wav",
                character_id="alice"
            )
            
            assert torch.equal(sample.input_ids, torch.tensor([1, 2, 3, 0, 0]))
            assert torch.equal(sample.attention_mask, torch.tensor([1, 1, 1, 0, 0]))
            
            # Check tokenizer was called correctly
            mock_tokenizer.assert_called_once_with(
                "hello world",
                max_length=5,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
    
    @pytest.mark.asyncio
    async def test_process_dataset(self):
        """Test processing entire dataset"""
        pipeline = MultimodalDatasetPipeline()
        
        # Mock dataset config
        dataset_config = {
            "samples": [
                {
                    "text": "hello world",
                    "audio_path": "/fake/path1.wav",
                    "character_id": "alice",
                    "voice_profile": {"pitch": 220}
                },
                {
                    "text": "goodbye friend",
                    "audio_path": "/fake/path2.wav",
                    "character_id": "bob"
                }
            ]
        }
        
        # Mock sample processing
        mock_sample1 = MultimodalTrainingSample(
            input_ids=torch.tensor([1, 2, 3]),
            attention_mask=torch.tensor([1, 1, 1]),
            text_labels=torch.tensor([2, 3, 4]),
            mel_frames=torch.randn(100, 80),
            speech_labels=torch.randint(0, 16, (100, 80)),
            speech_attention_mask=torch.ones(100),
            text_to_speech_alignment=torch.randn(3, 100),
            character_id="alice",
            character_embedding_idx=0,
            duration=1.0
        )
        
        mock_sample2 = MultimodalTrainingSample(
            input_ids=torch.tensor([4, 5, 6]),
            attention_mask=torch.tensor([1, 1, 1]),
            text_labels=torch.tensor([5, 6, 7]),
            mel_frames=torch.randn(100, 80),
            speech_labels=torch.randint(0, 16, (100, 80)),
            speech_attention_mask=torch.ones(100),
            text_to_speech_alignment=torch.randn(3, 100),
            character_id="bob",
            character_embedding_idx=1,
            duration=1.5
        )
        
        with patch.object(pipeline, 'process_text_speech_pair', side_effect=[mock_sample1, mock_sample2]):
            with tempfile.TemporaryDirectory() as temp_dir:
                stats = await pipeline.process_dataset(dataset_config, temp_dir)
                
                # Check statistics
                assert stats["total_samples"] == 2
                assert stats["successful_samples"] == 2
                assert stats["failed_samples"] == 0
                assert stats["total_duration"] == 2.5
                assert set(stats["characters"]) == {"alice", "bob"}
                
                # Check files were created
                output_path = Path(temp_dir)
                assert (output_path / "batch_0000.pt").exists()
                assert (output_path / "metadata.json").exists()
                
                # Check metadata
                with open(output_path / "metadata.json") as f:
                    metadata = json.load(f)
                
                assert metadata["processing_stats"]["successful_samples"] == 2
                assert "character_registry" in metadata
                assert "pipeline_config" in metadata
    
    @pytest.mark.asyncio
    async def test_process_dataset_with_failures(self):
        """Test dataset processing handles failures gracefully"""
        pipeline = MultimodalDatasetPipeline()
        
        dataset_config = {
            "samples": [
                {
                    "text": "hello world",
                    "audio_path": "/fake/path1.wav",
                    "character_id": "alice"
                },
                {
                    "text": "this will fail",
                    "audio_path": "/bad/path.wav",
                    "character_id": "bob"
                }
            ]
        }
        
        # Mock one success, one failure
        mock_sample = MultimodalTrainingSample(
            input_ids=torch.tensor([1, 2, 3]),
            attention_mask=torch.tensor([1, 1, 1]),
            text_labels=torch.tensor([2, 3, 4]),
            mel_frames=torch.randn(100, 80),
            speech_labels=torch.randint(0, 16, (100, 80)),
            speech_attention_mask=torch.ones(100),
            text_to_speech_alignment=torch.randn(3, 100),
            character_id="alice",
            character_embedding_idx=0,
            duration=1.0
        )
        
        with patch.object(pipeline, 'process_text_speech_pair', side_effect=[mock_sample, Exception("File not found")]):
            with tempfile.TemporaryDirectory() as temp_dir:
                stats = await pipeline.process_dataset(dataset_config, temp_dir)
                
                assert stats["total_samples"] == 1  # Only successful sample
                assert stats["successful_samples"] == 1
                assert stats["failed_samples"] == 1


class TestFactoryFunctions:
    """Test factory functions and utilities"""
    
    def test_create_multimodal_pipeline(self):
        """Test pipeline factory function"""
        pipeline = create_multimodal_pipeline()
        assert isinstance(pipeline, MultimodalDatasetPipeline)
        assert pipeline.tokenizer is None
    
    def test_create_multimodal_pipeline_with_args(self):
        """Test pipeline factory with custom arguments"""
        mock_tokenizer = Mock()
        pipeline = create_multimodal_pipeline(
            tokenizer=mock_tokenizer,
            max_text_length=256
        )
        
        assert pipeline.tokenizer == mock_tokenizer
        assert pipeline.max_text_length == 256
    
    @pytest.mark.asyncio
    async def test_create_sample_dataset_config(self):
        """Test sample dataset config creation"""
        config = await create_sample_dataset_config()
        
        assert "samples" in config
        assert len(config["samples"]) == 2
        
        sample1 = config["samples"][0]
        assert "text" in sample1
        assert "audio_path" in sample1
        assert "character_id" in sample1
        assert "voice_profile" in sample1
        
        assert sample1["character_id"] == "alice"
        assert sample1["voice_profile"]["voice_style"] == "friendly"


class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    @pytest.mark.asyncio
    @patch('librosa.load')
    async def test_end_to_end_pipeline(self, mock_librosa_load):
        """Test complete end-to-end pipeline functionality"""
        pipeline = create_multimodal_pipeline(max_text_length=10, max_speech_length=50)
        
        # Mock audio loading
        mock_audio = np.random.randn(11025)  # 0.5 seconds
        mock_librosa_load.return_value = (mock_audio, 22050)
        
        # Mock mel-spectrogram extraction
        with patch.object(SpeechPreprocessor, 'extract_mel_spectrogram') as mock_extract:
            with patch.object(SpeechPreprocessor, 'get_frame_times') as mock_frame_times:
                mock_extract.return_value = np.random.rand(25, 80)  # 25 frames
                mock_frame_times.return_value = [i * 0.02 for i in range(25)]
                
                sample = await pipeline.process_text_speech_pair(
                    text="hello world test",
                    audio_path="/fake/audio.wav",
                    character_id="test_character",
                    voice_profile={"pitch": 200}
                )
                
                # Verify all components work together
                assert sample.character_id == "test_character"
                assert sample.input_ids.shape[0] == 10  # max_text_length
                assert sample.mel_frames.shape == (50, 80)  # max_speech_length, n_mels
                assert sample.text_to_speech_alignment.shape == (10, 50)
                assert sample.character_embedding_idx == 0
                
                # Verify character was registered
                char_idx = pipeline.character_registry.get_character_idx("test_character")
                assert char_idx == 0
                
                voice_profile = pipeline.character_registry.get_voice_profile("test_character")
                assert voice_profile["pitch"] == 200 