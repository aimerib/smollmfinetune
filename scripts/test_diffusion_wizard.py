#!/usr/bin/env python3
"""
Test script for the Diffusion Training Wizard setup
Verifies that all components are properly configured and working
"""

import sys
import os
import json
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from narrative_engine.diffusion_config import (
    get_small_config, 
    get_medium_config, 
    get_large_config
)
from narrative_engine.diffusion_model import DiffusionMultimodalModel
from narrative_engine.diffusion_trainer import DiffusionTrainingManager

def test_configurations():
    """Test all predefined configurations"""
    print("🔧 Testing Diffusion Configurations...")
    
    configs = {
        'small': get_small_config(),
        'medium': get_medium_config(), 
        'large': get_large_config()
    }
    
    for name, config in configs.items():
        print(f"\n📊 {name.capitalize()} Configuration:")
        print(f"  • Parameters: ~{config.get_total_parameters() / 1e6:.1f}M")
        print(f"  • Hidden size: {config.transformer.hidden_size}")
        print(f"  • Layers: {config.transformer.num_layers}")
        print(f"  • Attention heads: {config.transformer.num_attention_heads}")
        print(f"  • Text embedding dim: {config.modalities.text_embedding_dim}")
        print(f"  • Text max sequence: {config.modalities.text_max_sequence_length}")
        
        # Validate configuration
        issues = config.validate()
        if issues:
            print(f"  ❌ Issues: {issues}")
            return False
        else:
            print(f"  ✅ Configuration valid")
    
    return True

def test_model_creation():
    """Test model creation for all configurations"""
    print("\n🧠 Testing Model Creation...")
    
    configs = {
        'small': get_small_config(),
        'medium': get_medium_config(),
        'large': get_large_config()
    }
    
    for name, config in configs.items():
        print(f"\n🔨 Creating {name} model...")
        
        try:
            model = DiffusionMultimodalModel(config)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  ✅ Model created successfully")
            print(f"  📊 Actual parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
            
            # Test forward pass with dummy data
            print(f"  🔄 Testing forward pass...")
            batch_size = 2
            
            # Create sample data that matches expected dimensions
            noisy_text = torch.randn(
                batch_size, 
                config.modalities.text_max_sequence_length, 
                config.modalities.text_embedding_dim
            )
            noisy_speech = torch.randn(
                batch_size, 
                config.modalities.speech_max_frames,
                config.modalities.speech_mel_bins
            )
            noisy_control = torch.randn(
                batch_size, 
                config.modalities.control_max_tokens, 
                config.modalities.control_embedding_dim
            )
            noisy_memory = torch.randn(
                batch_size, 
                config.modalities.memory_max_vectors, 
                config.modalities.memory_embedding_dim + config.modalities.memory_metadata_dim
            )
            character_ids = torch.randint(0, 10, (batch_size,))
            timesteps = torch.randint(0, config.scheduler.num_train_timesteps, (batch_size,))
            
            with torch.no_grad():
                output = model(
                    noisy_text=noisy_text,
                    noisy_speech=noisy_speech,
                    noisy_control=noisy_control,
                    noisy_memory=noisy_memory,
                    timesteps=timesteps,
                    character_ids=character_ids,
                    return_loss=False
                )
                print(f"  ✅ Forward pass successful")
                
                # Check output shapes
                print(f"  📐 Output shapes:")
                print(f"     text_prediction: {list(output.text_prediction.shape)}")
                print(f"     speech_prediction: {list(output.speech_prediction.shape)}")
                print(f"     control_prediction: {list(output.control_prediction.shape)}")
                print(f"     memory_prediction: {list(output.memory_prediction.shape)}")
            
        except Exception as e:
            print(f"  ❌ Model creation failed: {e}")
            return False
    
    return True

def test_backend_compatibility():
    """Test compatibility with backend API expectations"""
    print("\n🔗 Testing Backend API Compatibility...")
    
    # Test configuration serialization
    config = get_small_config()
    
    try:
        # Test serialization
        config_dict = config.to_dict()
        print(f"  ✅ Configuration serialization works")
        
        # Test deserialization
        config_restored = config.__class__.from_dict(config_dict)
        print(f"  ✅ Configuration deserialization works")
        
        # Verify they match
        if config_restored.transformer.hidden_size == config.transformer.hidden_size:
            print(f"  ✅ Serialization round-trip successful")
        else:
            print(f"  ❌ Serialization round-trip failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Backend compatibility test failed: {e}")
        return False
    
    return True

def test_wizard_expectations():
    """Test that configurations match wizard expectations"""
    print("\n🧙 Testing Wizard Component Expectations...")
    
    # Test parameter ranges match wizard options
    small_config = get_small_config()
    medium_config = get_medium_config()
    large_config = get_large_config()
    
    expected_ranges = {
        'small': (20e6, 40e6),  # 20-40M parameters
        'medium': (80e6, 150e6),  # 80-150M parameters  
        'large': (300e6, 500e6)  # 300-500M parameters
    }
    
    configs = {
        'small': small_config,
        'medium': medium_config,
        'large': large_config
    }
    
    for name, config in configs.items():
        params = config.get_total_parameters()
        min_params, max_params = expected_ranges[name]
        
        if min_params <= params <= max_params:
            print(f"  ✅ {name.capitalize()} model parameter count in expected range: {params / 1e6:.1f}M")
        else:
            print(f"  ⚠️ {name.capitalize()} model parameter count outside expected range: {params / 1e6:.1f}M")
    
    # Test learning rate ranges
    valid_lr_ranges = [5e-5, 1e-4, 2e-4, 5e-4]
    if small_config.learning_rate in valid_lr_ranges:
        print(f"  ✅ Default learning rate in wizard range: {small_config.learning_rate}")
    else:
        print(f"  ⚠️ Default learning rate not in wizard range: {small_config.learning_rate}")
    
    # Test guidance scale
    if 3.0 <= small_config.guidance_scale <= 15.0:
        print(f"  ✅ Default guidance scale in wizard range: {small_config.guidance_scale}")
    else:
        print(f"  ⚠️ Default guidance scale not in wizard range: {small_config.guidance_scale}")
    
    return True

def main():
    """Run all tests"""
    print("🌊 Diffusion Training Wizard Setup Test")
    print("=" * 50)
    
    tests = [
        ("Configuration Validation", test_configurations),
        ("Model Creation", test_model_creation),
        ("Backend Compatibility", test_backend_compatibility),
        ("Wizard Expectations", test_wizard_expectations)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Summary:")
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All tests passed! The Diffusion Training Wizard is ready to use.")
        print(f"\n🚀 Next steps:")
        print(f"  1. Start your React application")
        print(f"  2. Start your backend API server")
        print(f"  3. Navigate to /creator/diffusion-training")
        print(f"  4. Follow the wizard steps to train your first diffusion model!")
    else:
        print(f"\n💥 Some tests failed. Please review the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 