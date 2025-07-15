import pytest
import torch
import numpy as np
from backend.app.narrative_engine.config import NarrativeLLMConfig


def test_quad_head_model_imports():
    """Test that QuadHeadNarrativeLM can be imported - GREEN phase"""
    # Should now work since we created the module
    from backend.app.narrative_engine.quad_head_model import QuadHeadNarrativeLM
    assert QuadHeadNarrativeLM is not None


def test_speech_head_imports():
    """Test that SpeechHead can be imported - GREEN phase"""
    # Should now work since we created the module
    from backend.app.narrative_engine.speech_head import SpeechHead
    assert SpeechHead is not None


def test_quad_head_loss_imports():
    """Test that QuadHeadLoss can be imported - GREEN phase"""
    # Should now work since we created the loss function
    from backend.app.narrative_engine.loss import QuadHeadLoss
    assert QuadHeadLoss is not None


class TestQuadHeadArchitecture:
    """Test suite for quad-head NarrativeLM architecture"""
    
    def test_quad_head_model_instantiation(self):
        """Test that QuadHeadNarrativeLM can be instantiated - GREEN phase"""
        # Should now work since we created the class
        
        from backend.app.narrative_engine.quad_head_model import QuadHeadNarrativeLM
        
        config = NarrativeLLMConfig(
            base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            control_head_dim=128,
            num_hidden_layers=2,
            enable_speech_head=True,  # New config parameter
            speech_mel_bins=80,       # New config parameter
            speech_quantization_bits=4  # New config parameter
        )
        
        model = QuadHeadNarrativeLM(config)
        
        # Check that model has all four heads
        assert hasattr(model, 'base_model')
        assert hasattr(model, 'control_head')
        assert hasattr(model, 'memory_head') 
        assert hasattr(model, 'speech_head')  # New speech head
        assert hasattr(model, 'tokenizer')
    
    def test_speech_head_output_shape(self):
        """Test that speech head produces correct mel-spectrogram output shape - GREEN phase"""
        # Should now work since we created the SpeechHead class
        
        from backend.app.narrative_engine.speech_head import SpeechHead
        
        # Create speech head
        hidden_size = 768
        mel_bins = 80
        quantization_bits = 4
        
        speech_head = SpeechHead(
            hidden_size=hidden_size,
            mel_bins=mel_bins,
            quantization_bits=quantization_bits
        )
        
        # Test input
        batch_size = 2
        seq_len = 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Forward pass
        speech_output = speech_head(hidden_states)
        
        # Check output shape: [batch_size, seq_len, mel_bins]
        expected_shape = (batch_size, seq_len, mel_bins)
        assert speech_output.shape == expected_shape
        
        # Check output is properly quantized (4-bit = 16 discrete levels)
        unique_values = torch.unique(speech_output)
        assert len(unique_values) <= 16, "Output should be 4-bit quantized"
    
    def test_quad_head_forward_pass(self):
        """Test that quad-head model forward pass produces all four outputs - GREEN phase"""
        
        from backend.app.narrative_engine.quad_head_model import QuadHeadNarrativeLM
        
        config = NarrativeLLMConfig(
            base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            control_head_dim=64,
            num_hidden_layers=1,
            enable_speech_head=True,
            speech_mel_bins=80,
            speech_quantization_bits=4
        )
        
        model = QuadHeadNarrativeLM(config)
        
        # Create dummy inputs
        batch_size = 1
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check all four outputs exist
        assert 'text_logits' in outputs
        assert 'action_logits' in outputs  # control head
        assert 'memory_embedding' in outputs
        assert 'speech_logits' in outputs  # NEW: speech head output
        
        # Check speech output shape
        speech_logits = outputs['speech_logits']
        expected_shape = (batch_size, seq_len, 80)  # 80 mel bins
        assert speech_logits.shape == expected_shape
    
    def test_cross_modal_attention(self):
        """Test that speech head has cross-modal attention to text - GREEN phase"""
        
        from backend.app.narrative_engine.speech_head import SpeechHead
        
        speech_head = SpeechHead(
            hidden_size=768,
            mel_bins=80,
            quantization_bits=4,
            enable_cross_attention=True  # Enable text-speech attention
        )
        
        # Test inputs
        batch_size = 1
        seq_len = 10
        hidden_size = 768
        
        text_hidden = torch.randn(batch_size, seq_len, hidden_size)
        speech_hidden = torch.randn(batch_size, seq_len, hidden_size)  # Speech hidden states
        speech_frames = torch.randn(batch_size, seq_len, 80)  # Previous speech frames
        
        # Forward pass with cross-attention
        speech_output = speech_head(
            hidden_states=speech_hidden,
            text_hidden_states=text_hidden,
            speech_frames=speech_frames
        )
        
        # Check output shape and properties
        assert speech_output.shape == (batch_size, seq_len, 80)
        assert hasattr(speech_head, 'cross_attention')
    
    def test_character_voice_conditioning(self):
        """Test that speech head supports character-specific voice conditioning - GREEN phase"""
        
        from backend.app.narrative_engine.speech_head import SpeechHead
        
        speech_head = SpeechHead(
            hidden_size=768,
            mel_bins=80,
            quantization_bits=4,
            num_character_embeddings=1000  # Support up to 1000 characters
        )
        
        # Test inputs
        batch_size = 2
        seq_len = 5
        hidden_states = torch.randn(batch_size, seq_len, 768)
        character_ids = torch.tensor([1, 42])  # Different characters
        
        # Forward pass with character conditioning
        speech_output = speech_head(
            hidden_states=hidden_states,
            character_ids=character_ids
        )
        
        assert speech_output.shape == (batch_size, seq_len, 80)
        assert hasattr(speech_head, 'character_embedding')


class TestQuadHeadLoss:
    """Test suite for quad-head loss function"""
    
    def test_quad_head_loss_instantiation(self):
        """Test that QuadHeadLoss can be instantiated - GREEN phase"""
        
        from backend.app.narrative_engine.loss import QuadHeadLoss
        
        loss_fn = QuadHeadLoss(
            text_weight=1.0,
            control_weight=0.8,
            memory_weight=0.6,
            speech_weight=0.5  # NEW: speech loss weight
        )
        
        assert loss_fn.text_weight == 1.0
        assert loss_fn.control_weight == 0.8
        assert loss_fn.memory_weight == 0.6
        assert loss_fn.speech_weight == 0.5
    
    def test_quad_head_loss_calculation(self):
        """Test that QuadHeadLoss calculates losses for all four heads - GREEN phase"""
        
        from backend.app.narrative_engine.loss import QuadHeadLoss
        
        loss_fn = QuadHeadLoss(
            text_weight=1.0,
            control_weight=0.8,
            memory_weight=0.6,
            speech_weight=0.5
        )
        
        # Create dummy outputs and targets
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        mel_bins = 80
        
        # Model outputs
        text_logits = torch.randn(batch_size, seq_len, vocab_size)
        control_logits = torch.sigmoid(torch.randn(batch_size, 64))  # 64 control tokens (probabilities)
        memory_embedding = torch.randn(batch_size, 768)
        memory_metadata = torch.randn(batch_size, 4)
        speech_logits = torch.randn(batch_size, seq_len, mel_bins)  # NEW
        
        # Targets
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        control_labels = torch.randint(0, 2, (batch_size, 64)).float()
        memory_labels = torch.randn(batch_size, 772)  # 768 + 4
        speech_labels = torch.randn(batch_size, seq_len, mel_bins)  # NEW
        
        # Calculate loss
        losses = loss_fn(
            text_logits=text_logits,
            control_logits=control_logits,
            memory_embedding=memory_embedding,
            memory_metadata=memory_metadata,
            speech_logits=speech_logits,  # NEW
            labels=labels,
            control_labels=control_labels,
            memory_labels=memory_labels,
            speech_labels=speech_labels  # NEW
        )
        
        # Check all losses are calculated
        assert 'generation_loss' in losses
        assert 'control_loss' in losses
        assert 'memory_loss' in losses
        assert 'speech_loss' in losses  # NEW
        assert 'total_loss' in losses
        
        # Check loss values are reasonable
        assert losses['total_loss'].item() > 0
        assert losses['speech_loss'].item() > 0


class TestMultimodalTraining:
    """Test suite for multimodal training infrastructure"""
    
    def test_multimodal_dataset_format(self):
        """Test that multimodal dataset supports speech data - RED phase"""
        pytest.skip("Multimodal dataset format not implemented yet")
        
        # Expected dataset format for quad-head training
        sample = {
            'input_ids': torch.tensor([1, 2, 3, 4, 5]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1]),
            'labels': torch.tensor([2, 3, 4, 5, 6]),
            'control_labels': torch.zeros(64),  # Control token targets
            'memory_labels': torch.randn(772),  # Memory targets (768 + 4)
            'speech_labels': torch.randn(5, 80),  # NEW: Mel-spectrogram targets
            'character_id': torch.tensor(42),  # Character conditioning
            'speech_frames': torch.randn(5, 80)  # Previous speech context
        }
        
        # Validate sample format
        assert 'speech_labels' in sample
        assert 'character_id' in sample
        assert 'speech_frames' in sample
        assert sample['speech_labels'].shape == (5, 80)  # seq_len, mel_bins


class TestRealTimeInference:
    """Test suite for real-time speech generation"""
    
    def test_streaming_speech_generation(self):
        """Test that model supports streaming speech generation - RED phase"""
        pytest.skip("Streaming speech generation not implemented yet")
        
        from backend.app.narrative_engine.quad_head_model import QuadHeadNarrativeLM
        
        config = NarrativeLLMConfig(
            base_model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            enable_speech_head=True,
            speech_mel_bins=80,
            enable_streaming=True  # NEW: Streaming capability
        )
        
        model = QuadHeadNarrativeLM(config)
        
        # Test streaming generation
        input_ids = torch.tensor([[1, 2, 3]])
        
        # Should be able to generate speech frames incrementally
        speech_frames = []
        for step in range(5):  # Generate 5 speech frames
            output = model.generate_speech_frame(
                input_ids=input_ids,
                previous_speech_frames=torch.stack(speech_frames) if speech_frames else None
            )
            
            assert output.shape == (1, 80)  # Single frame, 80 mel bins
            speech_frames.append(output)
        
        # Check we generated 5 frames
        assert len(speech_frames) == 5 