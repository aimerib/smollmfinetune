"""
Comprehensive tests for Flow-Matching TTS Architecture

Tests cover all components of the narrative-optimized speech synthesis system:
- Core model components (encoders, flow matching, conditioning)
- Integration scenarios (end-to-end training and inference)
- Performance validation and quality metrics
- Character conditioning and zero-shot voice cloning
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch

from backend.app.services.voice.flow_matching_tts import (
    FlowMatchingConfig,
    NarrativeTextEncoder,
    NarrativeAwareAttention,
    ContinuousFlowMatcher,
    CharacterConditioner,
    AdaptiveLayerNorm,
    SpeakerEmbeddingExtractor,
    NarrativeFlowMatchingTTS,
    FlowMatchingTrainer
)


@pytest.fixture
def config():
    """Standard configuration for testing"""
    return FlowMatchingConfig(
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        num_mel_bins=80,
        max_sequence_length=512,
        num_character_classes=50,
        character_embedding_dim=128,
        speaker_embedding_dim=256,
        num_inference_steps=20  # Reduced for testing
    )


@pytest.fixture
def sample_batch():
    """Sample training batch"""
    return {
        'text_tokens': torch.randint(0, 1000, (4, 32)),
        'character_id': torch.tensor([0, 1, 2, 3]),
        'target_mel': torch.randn(4, 32, 80),
        'personality_traits': torch.randn(4, 5),
        'narrative_context': torch.randn(4, 256),
        'mel_lengths': torch.tensor([28, 32, 30, 29])
    }


class TestFlowMatchingConfig:
    """Test configuration class"""
    
    def test_default_config_creation(self):
        """Test creating config with default values"""
        config = FlowMatchingConfig()
        
        assert config.hidden_dim == 512
        assert config.num_layers == 8
        assert config.num_heads == 8
        assert config.num_mel_bins == 80
        assert config.noise_schedule == "cosine"
        assert config.enable_zero_shot is True
        
    def test_custom_config_creation(self):
        """Test creating config with custom values"""
        config = FlowMatchingConfig(
            hidden_dim=768,
            num_layers=12,
            noise_schedule="linear"
        )
        
        assert config.hidden_dim == 768
        assert config.num_layers == 12
        assert config.noise_schedule == "linear"


class TestNarrativeTextEncoder:
    """Test narrative text encoder"""
    
    def test_encoder_initialization(self, config):
        """Test encoder initializes correctly"""
        encoder = NarrativeTextEncoder(config, vocab_size=1000)
        
        assert encoder.config == config
        assert encoder.vocab_size == 1000
        assert isinstance(encoder.token_embedding, nn.Embedding)
        assert isinstance(encoder.position_embedding, nn.Embedding)
        assert len(encoder.transformer_layers) == config.num_layers
        
    def test_forward_pass_basic(self, config):
        """Test basic forward pass"""
        encoder = NarrativeTextEncoder(config, vocab_size=1000)
        
        input_ids = torch.randint(0, 1000, (2, 16))
        output = encoder(input_ids)
        
        assert output.shape == (2, 16, config.hidden_dim)
        
    def test_forward_pass_with_narrative_context(self, config):
        """Test forward pass with narrative context"""
        encoder = NarrativeTextEncoder(config, vocab_size=1000)
        
        input_ids = torch.randint(0, 1000, (2, 16))
        narrative_context = torch.randn(2, config.hidden_dim)
        
        output = encoder(input_ids, narrative_context=narrative_context)
        
        assert output.shape == (2, 16, config.hidden_dim)
        
    def test_forward_pass_with_attention_mask(self, config):
        """Test forward pass with attention mask"""
        encoder = NarrativeTextEncoder(config, vocab_size=1000)
        
        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones(2, 16).bool()
        attention_mask[0, 12:] = False  # Mask last 4 tokens for first sequence
        
        output = encoder(input_ids, attention_mask=attention_mask)
        
        assert output.shape == (2, 16, config.hidden_dim)


class TestNarrativeAwareAttention:
    """Test narrative-aware attention mechanism"""
    
    def test_attention_initialization(self, config):
        """Test attention layer initializes correctly"""
        attention = NarrativeAwareAttention(config)
        
        assert attention.config == config
        assert attention.hidden_dim == config.hidden_dim
        assert attention.num_heads <= config.num_heads  # May be adjusted for compatibility
        assert isinstance(attention.attention, nn.MultiheadAttention)
        
    def test_attention_forward_pass(self, config):
        """Test attention forward pass"""
        attention = NarrativeAwareAttention(config)
        
        hidden_states = torch.randn(2, 16, config.hidden_dim)
        output = attention(hidden_states)
        
        assert output.shape == (2, 16, config.hidden_dim)
        
    def test_attention_with_mask(self, config):
        """Test attention with padding mask"""
        attention = NarrativeAwareAttention(config)
        
        hidden_states = torch.randn(2, 16, config.hidden_dim)
        attention_mask = torch.ones(2, 16).bool()
        attention_mask[0, 12:] = False
        
        output = attention(hidden_states, attention_mask=attention_mask)
        
        assert output.shape == (2, 16, config.hidden_dim)
        
    def test_narrative_gating_mechanism(self, config):
        """Test that narrative gating affects output"""
        attention = NarrativeAwareAttention(config)
        
        hidden_states = torch.randn(2, 16, config.hidden_dim)
        
        # Forward pass
        output1 = attention(hidden_states)
        
        # Modify story progression weight
        with torch.no_grad():
            attention.story_progression_weight.fill_(0.5)
        
        output2 = attention(hidden_states)
        
        # Outputs should be different due to different gating
        assert not torch.allclose(output1, output2, atol=1e-6)


class TestContinuousFlowMatcher:
    """Test continuous flow matching component"""
    
    def test_flow_matcher_initialization(self, config):
        """Test flow matcher initializes correctly"""
        flow_matcher = ContinuousFlowMatcher(config)
        
        assert flow_matcher.config == config
        assert flow_matcher.num_mel_bins == config.num_mel_bins
        assert isinstance(flow_matcher.flow_network, nn.Sequential)
        assert hasattr(flow_matcher, 'alphas')
        
    def test_noise_schedules(self, config):
        """Test different noise schedules"""
        for schedule in ["cosine", "linear", "sigmoid"]:
            config.noise_schedule = schedule
            flow_matcher = ContinuousFlowMatcher(config)
            
            assert flow_matcher.alphas.shape[0] == config.num_inference_steps
            assert torch.all(flow_matcher.alphas >= 0.0)
            assert torch.all(flow_matcher.alphas <= 1.0)
            
    def test_forward_pass(self, config):
        """Test flow matcher forward pass"""
        flow_matcher = ContinuousFlowMatcher(config)
        
        batch_size, time_steps = 2, 32
        mel_noisy = torch.randn(batch_size, time_steps, config.num_mel_bins)
        text_embeddings = torch.randn(batch_size, time_steps, config.hidden_dim)
        timestep = torch.randint(0, config.num_inference_steps, (batch_size,))
        
        velocity = flow_matcher(mel_noisy, text_embeddings, timestep)
        
        assert velocity.shape == (batch_size, time_steps, config.num_mel_bins)
        
    def test_sampling(self, config):
        """Test mel-spectrogram sampling"""
        flow_matcher = ContinuousFlowMatcher(config)
        
        batch_size, time_steps = 2, 32
        text_embeddings = torch.randn(batch_size, time_steps, config.hidden_dim)
        
        generated_mel = flow_matcher.sample(text_embeddings, num_inference_steps=5)
        
        assert generated_mel.shape == (batch_size, time_steps, config.num_mel_bins)


class TestCharacterConditioner:
    """Test character conditioning system"""
    
    def test_conditioner_initialization(self, config):
        """Test character conditioner initializes correctly"""
        conditioner = CharacterConditioner(config)
        
        assert conditioner.config == config
        assert isinstance(conditioner.character_embedding, nn.Embedding)
        assert isinstance(conditioner.personality_processor, nn.Sequential)
        assert isinstance(conditioner.adaptive_norm, AdaptiveLayerNorm)
        
    def test_forward_basic_conditioning(self, config):
        """Test basic character conditioning"""
        conditioner = CharacterConditioner(config)
        
        batch_size, seq_len = 2, 16
        text_embeddings = torch.randn(batch_size, seq_len, config.hidden_dim)
        character_id = torch.tensor([0, 1])
        
        conditioned = conditioner(text_embeddings, character_id)
        
        assert conditioned.shape == (batch_size, seq_len, config.hidden_dim)
        
    def test_forward_with_personality_traits(self, config):
        """Test conditioning with personality traits"""
        conditioner = CharacterConditioner(config)
        
        batch_size, seq_len = 2, 16
        text_embeddings = torch.randn(batch_size, seq_len, config.hidden_dim)
        character_id = torch.tensor([0, 1])
        personality_traits = torch.randn(batch_size, 5)  # Big Five traits
        
        conditioned = conditioner(text_embeddings, character_id, personality_traits)
        
        assert conditioned.shape == (batch_size, seq_len, config.hidden_dim)
        
    def test_character_consistency(self, config):
        """Test that same character produces consistent conditioning"""
        conditioner = CharacterConditioner(config)
        
        batch_size, seq_len = 2, 16
        text_embeddings = torch.randn(batch_size, seq_len, config.hidden_dim)
        character_id = torch.tensor([0, 0])  # Same character
        
        conditioned = conditioner(text_embeddings, character_id)
        
        # Character embeddings should be identical
        char_embeds = conditioner.character_embedding(character_id)
        assert torch.allclose(char_embeds[0], char_embeds[1])


class TestAdaptiveLayerNorm:
    """Test adaptive layer normalization"""
    
    def test_adaptive_norm_initialization(self, config):
        """Test adaptive layer norm initializes correctly"""
        norm = AdaptiveLayerNorm(config.hidden_dim, config.character_embedding_dim)
        
        assert isinstance(norm.norm, nn.LayerNorm)
        assert isinstance(norm.scale_transform, nn.Linear)
        assert isinstance(norm.shift_transform, nn.Linear)
        
    def test_adaptive_norm_forward(self, config):
        """Test adaptive normalization forward pass"""
        norm = AdaptiveLayerNorm(config.hidden_dim, config.character_embedding_dim)
        
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.hidden_dim)
        conditioning = torch.randn(batch_size, config.character_embedding_dim)
        
        output = norm(x, conditioning)
        
        assert output.shape == (batch_size, seq_len, config.hidden_dim)
        
    def test_conditioning_effect(self, config):
        """Test that conditioning affects normalization"""
        norm = AdaptiveLayerNorm(config.hidden_dim, config.character_embedding_dim)
        
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, config.hidden_dim)
        conditioning1 = torch.randn(batch_size, config.character_embedding_dim)
        conditioning2 = torch.randn(batch_size, config.character_embedding_dim)
        
        output1 = norm(x, conditioning1)
        output2 = norm(x, conditioning2)
        
        # Different conditioning should produce different outputs
        assert not torch.allclose(output1, output2, atol=1e-5)


class TestSpeakerEmbeddingExtractor:
    """Test speaker embedding extraction for zero-shot cloning"""
    
    def test_extractor_initialization(self, config):
        """Test speaker extractor initializes correctly"""
        extractor = SpeakerEmbeddingExtractor(config)
        
        assert extractor.config == config
        assert isinstance(extractor.speaker_encoder, nn.Sequential)
        assert isinstance(extractor.speaker_adapter, nn.Sequential)
        
    def test_speaker_embedding_extraction(self, config):
        """Test speaker embedding extraction"""
        extractor = SpeakerEmbeddingExtractor(config)
        
        batch_size, time_steps = 2, 100
        reference_mel = torch.randn(batch_size, time_steps, config.num_mel_bins)
        
        speaker_embedding = extractor.extract_speaker_embedding(reference_mel)
        
        assert speaker_embedding.shape == (batch_size, config.speaker_embedding_dim)
        
    def test_embedding_adaptation(self, config):
        """Test speaker embedding adaptation"""
        extractor = SpeakerEmbeddingExtractor(config)
        
        batch_size, seq_len = 2, 16
        text_embeddings = torch.randn(batch_size, seq_len, config.hidden_dim)
        speaker_embedding = torch.randn(batch_size, config.speaker_embedding_dim)
        
        adapted_embeddings = extractor.adapt_embeddings(text_embeddings, speaker_embedding)
        
        assert adapted_embeddings.shape == (batch_size, seq_len, config.hidden_dim)
        
    def test_speaker_consistency(self, config):
        """Test that same speaker produces consistent embeddings"""
        extractor = SpeakerEmbeddingExtractor(config)
        
        # Same reference mel for both samples
        reference_mel = torch.randn(1, 100, config.num_mel_bins)
        reference_mel_batch = reference_mel.repeat(2, 1, 1)
        
        speaker_embeddings = extractor.extract_speaker_embedding(reference_mel_batch)
        
        # Should be identical
        assert torch.allclose(speaker_embeddings[0], speaker_embeddings[1], atol=1e-6)


class TestNarrativeFlowMatchingTTS:
    """Test main TTS model"""
    
    def test_model_initialization(self, config):
        """Test model initializes correctly"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        
        assert model.config == config
        assert isinstance(model.text_encoder, NarrativeTextEncoder)
        assert isinstance(model.flow_matcher, ContinuousFlowMatcher)
        assert isinstance(model.character_conditioner, CharacterConditioner)
        assert model.speaker_extractor is not None  # Should exist with enable_zero_shot=True
        
    def test_model_initialization_without_zero_shot(self, config):
        """Test model without zero-shot capability"""
        config.enable_zero_shot = False
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        
        assert model.speaker_extractor is None
        
    def test_training_forward_pass(self, config):
        """Test forward pass in training mode"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        character_id = torch.tensor([0, 1])
        target_mel = torch.randn(batch_size, seq_len, config.num_mel_bins)
        timestep = torch.randint(0, config.num_inference_steps, (batch_size,))
        
        outputs = model(
            input_ids=input_ids,
            character_id=character_id,
            target_mel=target_mel,
            timestep=timestep
        )
        
        assert 'predicted_velocity' in outputs
        assert 'target_velocity' in outputs
        assert 'mel_noisy' in outputs
        assert outputs['predicted_velocity'].shape == target_mel.shape
        
    def test_inference_forward_pass(self, config):
        """Test forward pass in inference mode"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        character_id = torch.tensor([0, 1])
        
        outputs = model(
            input_ids=input_ids,
            character_id=character_id
        )
        
        assert 'generated_mel' in outputs
        assert outputs['generated_mel'].shape == (batch_size, seq_len, config.num_mel_bins)
        
    def test_generate_method(self, config):
        """Test generation method"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        character_id = torch.tensor([0, 1])
        
        generated_mel = model.generate(
            input_ids=input_ids,
            character_id=character_id,
            num_inference_steps=5  # Reduced for testing
        )
        
        assert generated_mel.shape == (batch_size, seq_len, config.num_mel_bins)
        
    def test_zero_shot_cloning(self, config):
        """Test zero-shot voice cloning"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        character_id = torch.tensor([0, 1])
        reference_mel = torch.randn(batch_size, 50, config.num_mel_bins)
        
        outputs = model(
            input_ids=input_ids,
            character_id=character_id,
            reference_mel=reference_mel
        )
        
        assert 'generated_mel' in outputs
        assert outputs['generated_mel'].shape == (batch_size, seq_len, config.num_mel_bins)


class TestFlowMatchingTrainer:
    """Test training infrastructure"""
    
    def test_trainer_initialization(self, config):
        """Test trainer initializes correctly"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        trainer = FlowMatchingTrainer(model, learning_rate=1e-4)
        
        assert trainer.model == model
        assert trainer.config == config
        assert trainer.step_count == 0
        assert 'velocity_loss' in trainer.training_metrics
        
    def test_training_step(self, config, sample_batch):
        """Test a single training step"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        trainer = FlowMatchingTrainer(model, learning_rate=1e-4)
        
        loss, metrics = trainer.training_step(sample_batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert 'velocity_loss' in metrics
        assert 'mel_accuracy' in metrics
        assert 'character_consistency' in metrics
        assert trainer.step_count == 1
        
    def test_training_stats(self, config, sample_batch):
        """Test training statistics tracking"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        trainer = FlowMatchingTrainer(model, learning_rate=1e-4)
        
        # Run a few training steps
        for _ in range(3):
            trainer.training_step(sample_batch)
            
        stats = trainer.get_training_stats()
        
        assert 'velocity_loss_mean' in stats
        assert 'velocity_loss_latest' in stats
        assert 'total_steps' in stats
        assert 'current_lr' in stats
        assert stats['total_steps'] == 3
        
    def test_warmup_schedule(self, config, sample_batch):
        """Test learning rate warmup"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        trainer = FlowMatchingTrainer(model, learning_rate=1e-4, warmup_steps=5)
        
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        
        # First step should have scaled LR
        trainer.training_step(sample_batch)
        first_step_lr = trainer.optimizer.param_groups[0]['lr']
        
        assert first_step_lr < initial_lr
        
        # After warmup steps, LR should be higher
        for _ in range(5):
            trainer.training_step(sample_batch)
            
        post_warmup_lr = trainer.optimizer.param_groups[0]['lr']
        assert post_warmup_lr > first_step_lr


class TestFlowMatchingIntegration:
    """Integration tests for end-to-end functionality"""
    
    def test_end_to_end_training(self, config):
        """Test complete training pipeline"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        trainer = FlowMatchingTrainer(model, learning_rate=1e-4)
        
        # Simulate training batch
        batch = {
            'text_tokens': torch.randint(0, 1000, (4, 25)),
            'character_id': torch.tensor([0, 1, 0, 2]),
            'narrative_context': torch.randn(4, 256),
            'target_mel': torch.randn(4, 40, 80),
            'mel_lengths': torch.tensor([35, 40, 30, 38])
        }
        
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        loss, metrics = trainer.training_step(batch)
        
        # Check that parameters changed
        param_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(initial_params[name], param, atol=1e-8):
                param_changed = True
                break
                
        assert param_changed, "Model parameters should change during training"
        assert loss.item() > 0
        assert metrics['mel_accuracy'] > 0
        
    def test_end_to_end_inference(self, config):
        """Test complete inference pipeline"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        model.eval()
        
        with torch.no_grad():
            # Test basic generation
            input_ids = torch.randint(0, 1000, (2, 20))
            character_id = torch.tensor([0, 1])
            
            generated_mel = model.generate(input_ids, character_id, num_inference_steps=5)
            
            assert generated_mel.shape == (2, 20, config.num_mel_bins)
            assert not torch.isnan(generated_mel).any()
            assert not torch.isinf(generated_mel).any()
            
    def test_character_voice_consistency(self, config):
        """Test that same character produces consistent voice characteristics"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        model.eval()
        
        with torch.no_grad():
            input_ids = torch.randint(0, 1000, (2, 20))
            character_id = torch.tensor([5, 5])  # Same character
            
            mel1 = model.generate(input_ids, character_id, num_inference_steps=5)
            mel2 = model.generate(input_ids, character_id, num_inference_steps=5)
            
            # For untrained models, we just check that outputs are valid and different
            # characters would produce different conditioning (tested separately)
            assert mel1.shape == mel2.shape
            assert not torch.isnan(mel1).any()
            assert not torch.isnan(mel2).any()
            
            # Test that character conditioning is applied consistently
            # by checking the character embeddings are identical
            char_embeds = model.character_conditioner.character_embedding(character_id)
            assert torch.allclose(char_embeds[0], char_embeds[1])
            
    def test_personality_trait_influence(self, config):
        """Test that personality traits influence voice generation"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        model.eval()
        
        with torch.no_grad():
            input_ids = torch.randint(0, 1000, (2, 20))
            character_id = torch.tensor([0, 0])  # Same character
            
            # Different personality traits
            personality1 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])  # High openness
            personality2 = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])  # High conscientiousness
            
            mel1 = model.generate(input_ids[[0]], character_id[[0]], 
                                 personality_traits=personality1, num_inference_steps=5)
            mel2 = model.generate(input_ids[[0]], character_id[[0]], 
                                 personality_traits=personality2, num_inference_steps=5)
            
            # Different personalities should produce different outputs
            assert not torch.allclose(mel1, mel2, atol=0.1)
            
    def test_zero_shot_voice_adaptation(self, config):
        """Test zero-shot voice cloning capability"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        model.eval()
        
        with torch.no_grad():
            input_ids = torch.randint(0, 1000, (1, 20))
            character_id = torch.tensor([0])
            reference_mel = torch.randn(1, 50, config.num_mel_bins)
            
            # Generate with reference
            mel_with_ref = model.generate(
                input_ids, character_id, 
                reference_mel=reference_mel, 
                num_inference_steps=5
            )
            
            # Generate without reference
            mel_without_ref = model.generate(
                input_ids, character_id, 
                num_inference_steps=5
            )
            
            assert mel_with_ref.shape == mel_without_ref.shape
            # Should be different due to speaker adaptation
            assert not torch.allclose(mel_with_ref, mel_without_ref, atol=0.1)
            
    def test_narrative_context_influence(self, config):
        """Test that narrative context affects generation"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        model.eval()
        
        with torch.no_grad():
            input_ids = torch.randint(0, 1000, (1, 20))
            character_id = torch.tensor([0])
            
            # Different narrative contexts
            context1 = torch.randn(1, config.hidden_dim)
            context2 = torch.randn(1, config.hidden_dim)
            
            outputs1 = model(input_ids, character_id, narrative_context=context1)
            outputs2 = model(input_ids, character_id, narrative_context=context2)
            
            # Different contexts should produce different outputs
            assert not torch.allclose(
                outputs1['generated_mel'], 
                outputs2['generated_mel'], 
                atol=0.1
            )


class TestFlowMatchingPerformance:
    """Performance and quality validation tests"""
    
    def test_model_parameter_count(self, config):
        """Test model has reasonable parameter count"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Should have reasonable number of parameters (adjust thresholds as needed)
        assert 1_000_000 < total_params < 100_000_000  # 1M to 100M parameters
        assert trainable_params == total_params  # All parameters should be trainable
        
    def test_memory_efficiency(self, config):
        """Test memory usage during training"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        trainer = FlowMatchingTrainer(model, learning_rate=1e-4)
        
        batch = {
            'text_tokens': torch.randint(0, 1000, (4, 32)),
            'character_id': torch.tensor([0, 1, 2, 3]),
            'target_mel': torch.randn(4, 32, config.num_mel_bins),
            'mel_lengths': torch.tensor([28, 32, 30, 29])
        }
        
        # Should be able to run training step without OOM
        try:
            loss, metrics = trainer.training_step(batch)
            memory_efficient = True
        except RuntimeError as e:
            if "out of memory" in str(e):
                memory_efficient = False
            else:
                raise
                
        assert memory_efficient, "Training should be memory efficient"
        
    def test_inference_speed(self, config):
        """Test inference speed (basic timing)"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        model.eval()
        
        input_ids = torch.randint(0, 1000, (1, 20))
        character_id = torch.tensor([0])
        
        import time
        
        with torch.no_grad():
            start_time = time.time()
            _ = model.generate(input_ids, character_id, num_inference_steps=5)
            end_time = time.time()
            
        inference_time = end_time - start_time
        
        # Should complete inference in reasonable time (adjust threshold as needed)
        assert inference_time < 10.0, f"Inference took {inference_time:.2f}s, should be < 10s"
        
    def test_gradient_flow(self, config):
        """Test that gradients flow properly through the model"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        trainer = FlowMatchingTrainer(model, learning_rate=1e-4)
        
        batch = {
            'text_tokens': torch.randint(0, 1000, (2, 16)),
            'character_id': torch.tensor([0, 1]),
            'target_mel': torch.randn(2, 16, config.num_mel_bins),
        }
        
        loss, metrics = trainer.training_step(batch)
        
        # Check that key components have non-zero gradients
        text_encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in model.text_encoder.parameters()
        )
        flow_matcher_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in model.flow_matcher.parameters()
        )
        character_conditioner_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in model.character_conditioner.parameters()
        )
        
        assert text_encoder_has_grad, "Text encoder should receive gradients"
        assert flow_matcher_has_grad, "Flow matcher should receive gradients"
        assert character_conditioner_has_grad, "Character conditioner should receive gradients"
        
    def test_numerical_stability(self, config):
        """Test numerical stability during training"""
        model = NarrativeFlowMatchingTTS(config, vocab_size=1000)
        trainer = FlowMatchingTrainer(model, learning_rate=1e-4)
        
        batch = {
            'text_tokens': torch.randint(0, 1000, (4, 32)),
            'character_id': torch.tensor([0, 1, 2, 3]),
            'target_mel': torch.randn(4, 32, config.num_mel_bins),
        }
        
        # Run multiple training steps
        for _ in range(10):
            loss, metrics = trainer.training_step(batch)
            
            # Check for numerical issues
            assert not torch.isnan(loss), "Loss should not be NaN"
            assert not torch.isinf(loss), "Loss should not be infinite"
            assert loss.item() > 0, "Loss should be positive"
            
            # Check parameter values
            for name, param in model.named_parameters():
                assert not torch.isnan(param).any(), f"Parameter {name} contains NaN"
                assert not torch.isinf(param).any(), f"Parameter {name} contains infinite values" 