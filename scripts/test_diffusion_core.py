#!/usr/bin/env python3
"""
Simple test for core diffusion model functionality
Tests configuration, model creation, and forward pass without external dependencies.
"""

import sys
import logging
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from narrative_engine.diffusion_config import (
    DiffusionMultimodalConfig, 
    get_small_config, 
    get_medium_config, 
    get_large_config
)
from narrative_engine.diffusion_model import create_diffusion_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_configuration():
    """Test diffusion configuration system"""
    logger.info("🔧 Testing Configuration System...")
    
    # Test predefined configs
    configs = {
        "small": get_small_config(),
        "medium": get_medium_config(), 
        "large": get_large_config()
    }
    
    for name, config in configs.items():
        issues = config.validate()
        params = config.get_total_parameters()
        
        if issues:
            logger.info(f"❌ {name.capitalize()}: {issues}")
        else:
            logger.info(f"✅ {name.capitalize()}: {params:,} parameters")
    
    # Test serialization
    config = configs["small"]
    config_dict = config.to_dict()
    restored = DiffusionMultimodalConfig.from_dict(config_dict)
    
    if restored.guidance_scale == config.guidance_scale:
        logger.info("✅ Serialization works")
    else:
        logger.info("❌ Serialization failed")
    
    return config


def test_model_creation(config):
    """Test model creation and basic properties"""
    logger.info("\n🤖 Testing Model Creation...")
    
    try:
        model = create_diffusion_model(config)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model created: {total_params:,} parameters")
        
        # Test model components
        logger.info(f"✅ Text head: {type(model.text_head).__name__}")
        logger.info(f"✅ Speech head: {type(model.speech_head).__name__}")
        logger.info(f"✅ Control head: {type(model.control_head).__name__}")
        logger.info(f"✅ Memory head: {type(model.memory_head).__name__}")
        logger.info(f"✅ Noise scheduler: {type(model.noise_scheduler).__name__}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        raise


def test_forward_pass(model, config):
    """Test forward pass with dummy data"""
    logger.info("\n🚀 Testing Forward Pass...")
    
    batch_size = 2
    device = 'cpu'
    
    # Create test inputs matching config dimensions
    # Use smaller sequence lengths for the test to avoid shape mismatches
    text_seq_len = min(64, config.modalities.text_max_sequence_length)
    speech_frames = min(100, config.modalities.speech_max_frames)
    control_tokens = min(8, config.modalities.control_max_tokens)
    memory_vectors = min(4, config.modalities.memory_max_vectors)
    
    text_shape = (batch_size, text_seq_len, config.modalities.text_embedding_dim)
    speech_shape = (batch_size, speech_frames, config.modalities.speech_mel_bins)
    control_shape = (batch_size, control_tokens, config.modalities.control_embedding_dim)
    memory_shape = (batch_size, memory_vectors, 
                    config.modalities.memory_embedding_dim + config.modalities.memory_metadata_dim)
    
    # Generate random inputs
    noisy_text = torch.randn(text_shape)
    noisy_speech = torch.randn(speech_shape)
    noisy_control = torch.randn(control_shape)
    noisy_memory = torch.randn(memory_shape)
    
    clean_text = torch.randn(text_shape)
    clean_speech = torch.randn(speech_shape)
    clean_control = torch.randn(control_shape)
    clean_memory = torch.randn(memory_shape)
    
    timesteps = torch.randint(0, config.scheduler.num_train_timesteps, (batch_size,))
    character_ids = torch.randint(0, 100, (batch_size,))
    
    try:
        with torch.no_grad():
            output = model(
                noisy_text=noisy_text,
                noisy_speech=noisy_speech,
                noisy_control=noisy_control,
                noisy_memory=noisy_memory,
                timesteps=timesteps,
                character_ids=character_ids,
                clean_text=clean_text,
                clean_speech=clean_speech,
                clean_control=clean_control,
                clean_memory=clean_memory,
                return_loss=True
            )
        
        logger.info("✅ Forward pass successful")
        logger.info(f"   Text output: {output.text_prediction.shape}")
        logger.info(f"   Speech output: {output.speech_prediction.shape}")
        logger.info(f"   Control output: {output.control_prediction.shape}")
        logger.info(f"   Memory output: {output.memory_prediction.shape}")
        
        logger.info(f"✅ Loss computation successful")
        logger.info(f"   Total loss: {output.total_loss.item():.4f}")
        logger.info(f"   Text loss: {output.text_loss.item():.4f}")
        logger.info(f"   Speech loss: {output.speech_loss.item():.4f}")
        logger.info(f"   Control loss: {output.control_loss.item():.4f}")
        logger.info(f"   Memory loss: {output.memory_loss.item():.4f}")
        logger.info(f"   Alignment loss: {output.alignment_loss.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_noise_scheduler(model, config):
    """Test noise scheduler functionality"""
    logger.info("\n🌊 Testing Noise Scheduler...")
    
    scheduler = model.noise_scheduler
    
    # Test noise addition
    original = torch.randn(2, 10, config.modalities.text_embedding_dim)
    noise = torch.randn_like(original)
    timesteps = torch.tensor([100, 200])
    
    try:
        noisy = scheduler.add_noise(original, noise, timesteps, 'text')
        logger.info(f"✅ Noise addition: {noisy.shape}")
        
        # Test denoising step
        denoised = scheduler.step(noise, 100, noisy, 'text')
        logger.info(f"✅ Denoising step: {denoised.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Noise scheduler failed: {e}")
        return False


def show_architecture_summary(model, config):
    """Display architecture summary"""
    logger.info("\n📊 Architecture Summary:")
    logger.info("=" * 50)
    
    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Parameters: {total_params:,}")
    
    # Configuration summary
    logger.info(f"Hidden Size: {config.transformer.hidden_size}")
    logger.info(f"Layers: {config.transformer.num_layers}")
    logger.info(f"Attention Heads: {config.transformer.num_attention_heads}")
    logger.info(f"Cross-Attention Layers: {len(config.transformer.cross_attention_layers)}")
    
    # Modality dimensions
    logger.info("\nModality Dimensions:")
    logger.info(f"  Text: {config.modalities.text_max_sequence_length} × {config.modalities.text_embedding_dim}")
    logger.info(f"  Speech: {config.modalities.speech_max_frames} × {config.modalities.speech_mel_bins}")
    logger.info(f"  Control: {config.modalities.control_max_tokens} × {config.modalities.control_embedding_dim}")
    logger.info(f"  Memory: {config.modalities.memory_max_vectors} × {config.modalities.memory_embedding_dim + config.modalities.memory_metadata_dim}")
    
    # Training settings
    logger.info(f"\nGuidance Scale: {config.guidance_scale}")
    logger.info(f"Character Conditioning: {config.use_character_conditioning}")
    logger.info(f"EMA Enabled: {config.use_ema}")
    logger.info(f"Timesteps: {config.scheduler.num_train_timesteps}")


def main():
    """Main test function"""
    logger.info("🌊 Diffusion Multimodal Architecture Test")
    logger.info("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    try:
        # Test 1: Configuration
        config = test_configuration()
        success_count += 1
        
        # Test 2: Model Creation
        model = test_model_creation(config)
        success_count += 1
        
        # Test 3: Forward Pass
        if test_forward_pass(model, config):
            success_count += 1
        
        # Test 4: Noise Scheduler
        if test_noise_scheduler(model, config):
            success_count += 1
        
        # Show summary
        show_architecture_summary(model, config)
        
        # Final results
        logger.info("\n" + "=" * 50)
        logger.info(f"✅ Tests Passed: {success_count}/{total_tests}")
        
        if success_count == total_tests:
            logger.info("🎉 All tests passed! Diffusion architecture is ready!")
            logger.info("\nNext steps:")
            logger.info("1. Generate multimodal dataset:")
            logger.info("   python scripts/generate_multimodal_dataset.py")
            logger.info("2. Start training:")
            logger.info("   Use DiffusionTrainingManager with your dataset")
            return 0
        else:
            logger.info("❌ Some tests failed. Check the output above.")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 