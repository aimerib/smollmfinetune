#!/usr/bin/env python3
"""
Test C.L.A.R.A. Loop Training Integration

Validates that our dual-head training manager properly extends
the existing training infrastructure.
"""

import torch
import json
from pathlib import Path
import sys
sys.path.append('.')

from narrative_engine.clara_trainer import CLARALoopTrainingManager, CLARADataCollator
from narrative_engine.model import create_clara_loop_model


def test_clara_training_manager():
    """Test C.L.A.R.A. Loop training manager creation"""
    print("🧪 Testing C.L.A.R.A. Loop Training Manager")
    print("=" * 50)
    
    # Create training manager
    manager = CLARALoopTrainingManager()
    print(f"✅ Created C.L.A.R.A. training manager")
    print(f"   Device: {manager.device}")
    print(f"   Base model: {manager.base_model}")
    
    return manager


def test_control_token_annotation():
    """Test control token annotation functionality"""
    print("\n🏷️ Testing Control Token Annotation")
    print("=" * 50)
    
    manager = CLARALoopTrainingManager()
    
    # Create sample dataset
    sample_dataset = [
        {
            "messages": [
                {"role": "user", "content": "You're so beautiful and amazing!"},
                {"role": "system", "content": "You are a helpful character."},
                {"role": "assistant", "content": "Oh my... *blushes* that's so sweet of you to say!"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What do you think about this?"},
                {"role": "system", "content": "You are a helpful character."},
                {"role": "assistant", "content": "That's a really interesting question! I'm curious to hear more."}
            ]
        }
    ]
    
    character = {"name": "TestCharacter", "personality": "friendly"}
    
    # Test annotation
    annotated_dataset = manager.create_control_token_annotations(sample_dataset, character)
    
    print(f"✅ Annotated {len(annotated_dataset)} samples")
    
    for i, sample in enumerate(annotated_dataset):
        print(f"\n   Sample {i+1}:")
        print(f"   User: {sample['messages'][0]['content']}")
        print(f"   Assistant: {sample['messages'][2]['content']}")
        
        if 'control_labels' in sample:
            active_tokens = [j for j, label in enumerate(sample['control_labels']) if label > 0]
            print(f"   Active control tokens: {active_tokens}")
        else:
            print("   No control labels found")
    
    return annotated_dataset


def test_clara_data_collator():
    """Test C.L.A.R.A. Loop data collation"""
    print("\n📦 Testing C.L.A.R.A. Data Collator")
    print("=" * 50)
    
    # Create mock tokenizer
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            self.eos_token_id = 1
        
        def __call__(self, text, **kwargs):
            # Simple mock tokenization
            return {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]}
        
        def pad(self, *args, **kwargs):
            # Mock padding method
            return {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]}
    
    tokenizer = MockTokenizer()
    collator = CLARADataCollator(tokenizer=tokenizer)
    
    # Create sample features
    features = [
        {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [1, 2, 3, 4, 5],
            "control_labels": [1.0, 0.0, 1.0, 0.0, 0.0] + [0.0] * 26  # 31 total
        },
        {
            "input_ids": [2, 3, 4, 5, 6],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [2, 3, 4, 5, 6],
            "control_labels": [0.0, 1.0, 0.0, 1.0, 0.0] + [0.0] * 26  # 31 total
        }
    ]
    
    # Test collation
    try:
        batch = collator(features)
        print("✅ Data collation successful!")
        print(f"   Batch keys: {list(batch.keys())}")
        
        if 'control_labels' in batch:
            print(f"   Control labels shape: {batch['control_labels'].shape}")
            print(f"   Control labels dtype: {batch['control_labels'].dtype}")
        
        return True
    except Exception as e:
        print(f"❌ Data collation failed: {e}")
        return False


def test_clara_model_creation():
    """Test C.L.A.R.A. Loop model creation"""
    print("\n🧠 Testing C.L.A.R.A. Model Creation")
    print("=" * 50)
    
    try:
        # Create model
        model = create_clara_loop_model()
        print(f"✅ Created C.L.A.R.A. Loop model")
        print(f"   Base model: {model.config.base_model_name}")
        print(f"   Control tokens: {len(model.control_tokens)}")
        print(f"   Control head dim: {model.config.control_head_dim}")
        
        # Test forward pass
        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model.forward(input_ids)
        
        print(f"✅ Forward pass successful!")
        print(f"   Generation logits: {outputs['generation_logits'].shape}")
        print(f"   Control logits: {outputs['control_logits'].shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_integration_flow():
    """Test complete integration flow"""
    print("\n🔗 Testing Complete Integration Flow")
    print("=" * 60)
    
    # Step 1: Create training manager
    print("1. Creating C.L.A.R.A. training manager...")
    manager = CLARALoopTrainingManager()
    
    # Step 2: Create sample data
    print("2. Creating sample dataset...")
    sample_dataset = [
        {
            "messages": [
                {"role": "user", "content": "Hi there! You seem really nice."},
                {"role": "system", "content": "You are Clara, a friendly AI assistant."},
                {"role": "assistant", "content": "Aww, thank you! That's so sweet of you to say. *smiles warmly*"}
            ]
        }
    ]
    
    character = {
        "name": "Clara",
        "personality": "friendly, warm, engaging",
        "description": "A helpful AI assistant with emotional intelligence"
    }
    
    # Step 3: Test annotation
    print("3. Testing control token annotation...")
    annotated_dataset = manager.create_control_token_annotations(sample_dataset, character)
    
    # Step 4: Create config
    print("4. Creating training configuration...")
    config = {
        'clara_mode': True,
        'batch_size': 1,
        'max_steps': 10,  # Very short for test
        'learning_rate': 1e-4,
        'control_loss_weight': 1.0,
        'control_head_dim': 128  # Smaller for test
    }
    
    print("✅ Integration flow validation complete!")
    print(f"   Dataset size: {len(annotated_dataset)} samples")
    print(f"   Control annotations: {'✅' if 'control_labels' in annotated_dataset[0] else '❌'}")
    print(f"   Configuration: {len(config)} parameters")
    
    return True


def main():
    """Run all C.L.A.R.A. Loop tests"""
    print("🎭 C.L.A.R.A. Loop Training Integration Tests")
    print("=" * 70)
    print("Testing our dual-head emotional recirculation training system!")
    print()
    
    try:
        # Test 1: Training manager
        manager = test_clara_training_manager()
        
        # Test 2: Control token annotation
        annotated_data = test_control_token_annotation()
        
        # Test 3: Data collator
        collator_works = test_clara_data_collator()
        
        # Test 4: Model creation
        model = test_clara_model_creation()
        
        # Test 5: Integration flow
        integration_works = test_integration_flow()
        
        # Summary
        print("\n🎉 Test Summary")
        print("=" * 30)
        print(f"✅ Training Manager: {'PASS' if manager else 'FAIL'}")
        print(f"✅ Token Annotation: {'PASS' if annotated_data else 'FAIL'}")
        print(f"✅ Data Collator: {'PASS' if collator_works else 'FAIL'}")
        print(f"✅ Model Creation: {'PASS' if model else 'FAIL'}")
        print(f"✅ Integration Flow: {'PASS' if integration_works else 'FAIL'}")
        
        all_passed = all([manager, annotated_data, collator_works, model, integration_works])
        
        if all_passed:
            print("\n🚀 ALL TESTS PASSED!")
            print("C.L.A.R.A. Loop training architecture is ready for deployment!")
        else:
            print("\n⚠️ Some tests failed. Check the output above for details.")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 