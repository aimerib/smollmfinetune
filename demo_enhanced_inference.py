#!/usr/bin/env python3
"""
Demo script for Enhanced InferenceManager with Adapter Hot-Swapping

This script demonstrates the new capabilities of the InferenceManager
that allow loading characters with multiple adapters and hot-swapping
between them at runtime.
"""

import sys
sys.path.append('app')

from utils.inference import InferenceManager
import logging

# Set up logging to see the details
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_enhanced_inference():
    """Demonstrate the enhanced inference manager with hot-swapping"""
    
    print("🎮 Enhanced InferenceManager Demo - Adapter Hot-Swapping")
    print("=" * 60)
    
    # Initialize the enhanced inference manager
    inference_manager = InferenceManager()
    
    # Check if NarrativeLLM is available
    from utils.inference import NARRATIVE_ENGINE_AVAILABLE
    if not NARRATIVE_ENGINE_AVAILABLE:
        print("❌ NarrativeLLM not available. Please ensure narrative engine is installed.")
        return
    
    print("✅ NarrativeLLM hot-swapping capabilities enabled!")
    print()
    
    # Find available characters
    print("🔍 Looking for characters with trained adapters...")
    
    # Let's check what characters are available
    from pathlib import Path
    adapters_dir = Path("training_output/adapters")
    available_characters = []
    
    if adapters_dir.exists():
        for char_dir in adapters_dir.iterdir():
            if char_dir.is_dir():
                adapters = inference_manager._find_all_character_adapters(char_dir.name)
                if adapters:
                    available_characters.append((char_dir.name, adapters))
    
    if not available_characters:
        print("❌ No characters with adapters found.")
        print("💡 Please train some character adapters first using the training pipeline.")
        return
    
    print(f"📦 Found {len(available_characters)} characters with adapters:")
    for char_name, adapters in available_characters:
        print(f"  • {char_name}: {list(adapters.keys())}")
    print()
    
    # Demo with the first available character
    demo_character, demo_adapters = available_characters[0]
    print(f"🎭 Using '{demo_character}' for demonstration")
    print()
    
    # Step 1: Load character with all adapters
    print("📥 Loading character with hot-swapping enabled...")
    result = inference_manager.load_character_with_hot_swap(demo_character)
    print(f"Result: {result}")
    print()
    
    # Step 2: Show character info
    print("📊 Character Information:")
    char_info = inference_manager.get_character_info(demo_character)
    for key, value in char_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Step 3: Test generation with default adapter
    print("💬 Testing generation with default adapter...")
    test_prompt = "Hello! How are you feeling today?"
    response = inference_manager.generate_with_character(
        demo_character, 
        test_prompt, 
        max_tokens=100
    )
    
    if "error" not in response:
        print(f"📝 Prompt: {test_prompt}")
        print(f"🤖 Response ({response['active_adapter']}): {response['response']}")
        if response.get('control_tokens'):
            print(f"🎛️ Control Tokens: {response['control_tokens']}")
        if response.get('emotional_state'):
            print(f"😊 Emotional State: {response['emotional_state']}")
        print()
    else:
        print(f"❌ Generation failed: {response['error']}")
        return
    
    # Step 4: Demo hot-swapping if multiple adapters available
    available_types = list(demo_adapters.keys())
    if len(available_types) > 1:
        print("🔄 Demonstrating adapter hot-swapping...")
        
        # Try switching to each available adapter
        for adapter_type in available_types:
            print(f"\n  Switching to {adapter_type} adapter...")
            switch_result = inference_manager.switch_character_adapter(demo_character, adapter_type)
            print(f"  Result: {switch_result}")
            
            if "✅" in switch_result:
                # Generate with the new adapter
                response = inference_manager.generate_with_character(
                    demo_character, 
                    test_prompt, 
                    max_tokens=50
                )
                
                if "error" not in response:
                    print(f"  🤖 Response ({adapter_type}): {response['response'][:100]}...")
                else:
                    print(f"  ❌ Generation failed: {response['error']}")
        print()
    else:
        print("ℹ️ Only one adapter available - no hot-swapping to demonstrate")
        print()
    
    # Step 5: Show all loaded characters
    print("📋 Currently loaded characters:")
    loaded_chars = inference_manager.list_loaded_characters()
    for char in loaded_chars:
        print(f"  • {char['character_name']}: {char['total_adapters']} adapters, active: {char['active_adapter']}")
    print()
    
    # Step 6: Cleanup
    print("🧹 Cleaning up...")
    unload_result = inference_manager.unload_character(demo_character)
    print(f"Result: {unload_result}")
    
    print("\n🎉 Demo completed! The enhanced InferenceManager enables:")
    print("  ✨ Loading characters with multiple adapters")
    print("  🔄 Hot-swapping between SFT, RLHF, and checkpoint adapters")
    print("  🎛️ Advanced generation with control tokens")
    print("  💾 Efficient memory management")
    print("  📊 Detailed character and adapter information")


if __name__ == "__main__":
    demo_enhanced_inference() 