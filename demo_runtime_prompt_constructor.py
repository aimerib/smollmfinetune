#!/usr/bin/env python3
"""
Demo script for RuntimePromptConstructor

This script demonstrates how to use the RuntimePromptConstructor
to create dynamic prompts for character interactions at runtime.

Run this after exporting a character runtime packet.
"""

import json
import os
from pathlib import Path

# Add the app directory to Python path for imports
import sys
sys.path.append('app')

from utils.runtime.prompt_constructor import RuntimePromptConstructor


def demo_runtime_prompt_constructor():
    """Demonstrate the RuntimePromptConstructor functionality"""
    
    print("🎮 RuntimePromptConstructor Demo")
    print("=" * 50)
    
    # Check if we have any runtime packets to demo with
    runtime_packets_dir = Path("runtime_packets")
    if not runtime_packets_dir.exists():
        print("❌ No runtime_packets directory found.")
        print("💡 Please export a character using the Model Management page first.")
        return
    
    # Find available packets
    packets = [d for d in runtime_packets_dir.iterdir() if d.is_dir()]
    if not packets:
        print("❌ No runtime packets found.")
        print("💡 Please export a character using the Model Management page first.")
        return
    
    # Use the first available packet
    packet_path = str(packets[0])
    character_name = packets[0].name
    
    print(f"📦 Using runtime packet: {character_name}")
    print(f"📁 Path: {packet_path}")
    print()
    
    try:
        # Initialize the constructor
        print("🔧 Initializing RuntimePromptConstructor...")
        constructor = RuntimePromptConstructor(packet_path)
        print(f"✅ Loaded character: {constructor.get_character_name()}")
        print(f"🎛️  Available tokens: {len(constructor.get_available_tokens())}")
        print()
        
        # Demo 1: Basic conversation
        print("💬 Demo 1: Basic Conversation")
        print("-" * 30)
        
        conversation_history = [
            {"role": "user", "content": "Hello! I'd like to get to know you better."}
        ]
        
        basic_prompt = constructor.construct(conversation_history)
        print("Prompt:")
        print(basic_prompt)
        print()
        
        # Demo 2: Dynamic mood and relationship state
        print("😊 Demo 2: Dynamic State (Happy Mood, High Trust)")
        print("-" * 50)
        
        conversation_history = [
            {"role": "user", "content": "Good morning! How are you feeling?"},
            {"role": "assistant", "content": "I'm feeling wonderful today! Thank you for asking."},
            {"role": "user", "content": "That's great to hear. What are your plans?"}
        ]
        
        dynamic_state = {
            "current_mood": "happy",
            "relationship_to_user": {
                "trust": 0.9,
                "affinity": 0.8
            },
            "recent_events": [
                "Had a great conversation about shared interests",
                "User showed genuine care about my wellbeing",
                "Feeling more connected than before"
            ]
        }
        
        dynamic_prompt = constructor.construct(conversation_history, dynamic_state)
        print("Prompt with dynamic state:")
        print(dynamic_prompt)
        print()
        
        # Demo 3: Different mood and forced tokens
        print("🤔 Demo 3: Curious Mood with Forced Tokens")
        print("-" * 40)
        
        conversation_history = [
            {"role": "user", "content": "I have something interesting to tell you"}
        ]
        
        # Get available tokens for demonstration
        available_tokens = constructor.get_available_tokens()
        mood_tokens = constructor.get_tokens_by_category("mood")
        
        forced_tokens = []
        if mood_tokens:
            forced_tokens.append(mood_tokens[0]["token"])  # Use first mood token
        
        dynamic_state = {
            "current_mood": "curious",
            "forced_control_tokens": forced_tokens,
            "relationship_to_user": {"trust": 0.6, "affinity": 0.7}
        }
        
        curious_prompt = constructor.construct(conversation_history, dynamic_state)
        print("Prompt with curiosity and forced tokens:")
        print(curious_prompt)
        print()
        
        # Demo 4: Error handling
        print("⚠️  Demo 4: Error Handling")
        print("-" * 25)
        
        try:
            # Test with invalid conversation structure
            invalid_history = [{"invalid": "structure"}]
            fallback_prompt = constructor.construct(invalid_history)
            print("Fallback prompt (handles errors gracefully):")
            print(fallback_prompt)
        except Exception as e:
            print(f"Error handled: {e}")
        
        print()
        print("🎉 Demo completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("  • Character personality injection (Big-Five traits)")
        print("  • World lore integration")
        print("  • Dynamic mood control")
        print("  • Relationship state tracking")
        print("  • Recent events memory")
        print("  • Control token injection")
        print("  • Conversation history formatting")
        print("  • Graceful error handling")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n🔍 Troubleshooting:")
        print("  • Make sure you've exported a character runtime packet")
        print("  • Check that all required files exist in the packet")
        print("  • Verify the packet format is correct")


if __name__ == "__main__":
    demo_runtime_prompt_constructor() 