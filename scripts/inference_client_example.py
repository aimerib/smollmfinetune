#!/usr/bin/env python
"""
Example client for the Production Inference Engine.

This demonstrates how mobile apps and other clients would interact
with the inference server, including:
- Basic text generation
- Session management
- Memory integration
- Control token handling
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class InferenceClient:
    """Client for the Production Inference Engine"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate(self, 
                      session_id: str,
                      character_id: str,
                      prompt: str,
                      **kwargs) -> Dict[str, Any]:
        """Generate a response from character"""
        
        payload = {
            "session_id": session_id,
            "character_id": character_id,
            "prompt": prompt,
            **kwargs
        }
        
        async with self.session.post(
            f"{self.base_url}/generate",
            json=payload
        ) as response:
            return await response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        async with self.session.get(f"{self.base_url}/metrics") as response:
            return await response.json()


async def mobile_chat_example():
    """Example: Mobile app chat interaction"""
    print("\n=== Mobile Chat Example ===\n")
    
    async with InferenceClient() as client:
        # Check server health
        health = await client.health_check()
        print(f"Server status: {health['status']}")
        print(f"GPU available: {health.get('gpu_utilization', 0) * 100:.1f}%")
        
        # Start a chat session
        session_id = "mobile-session-123"
        character_id = "alice"
        
        # Conversation turns
        messages = [
            "Hello Alice! How are you today?",
            "What's your favorite thing to do?",
            "Tell me a story about your adventures."
        ]
        
        for i, message in enumerate(messages):
            print(f"\n👤 User: {message}")
            
            # Generate response
            response = await client.generate(
                session_id=session_id,
                character_id=character_id,
                prompt=message,
                max_tokens=150,
                temperature=0.8
            )
            
            print(f"🤖 {character_id.title()}: {response['generation_text']}")
            
            # Show control tokens if any
            if response.get('control_tokens'):
                print(f"   [Control: {', '.join(t['token'] for t in response['control_tokens'])}]")
            
            # Show memory formation
            if response.get('memory_metadata'):
                importance = response['memory_metadata'].get('importance', 0)
                if importance > 0.7:
                    print(f"   💭 Important memory formed (importance: {importance:.2f})")
            
            # Brief pause between turns
            await asyncio.sleep(1)
        
        # Show metrics
        print("\n=== Performance Metrics ===")
        metrics = await client.get_metrics()
        print(f"Average response time: {metrics['avg_inference_time_ms']:.1f}ms")
        print(f"Requests processed: {metrics['total_requests']}")


async def proactive_agent_example():
    """Example: Proactive agent checking in"""
    print("\n=== Proactive Agent Example ===\n")
    
    async with InferenceClient() as client:
        session_id = "proactive-session-456"
        character_id = "companion_ai"
        
        # Simulate time-based check-in
        print("🔔 Proactive check-in triggered (3 hours since last interaction)\n")
        
        # Agent initiates conversation based on context
        response = await client.generate(
            session_id=session_id,
            character_id=character_id,
            prompt="",  # Empty prompt - agent initiates
            forced_control_tokens=["<proactive_greeting>"],
            memory_context_ids=["last_conversation", "user_preferences"],
            max_tokens=100
        )
        
        print(f"🤖 {character_id}: {response['generation_text']}")
        
        # Show emotional state
        control_tokens = response.get('control_tokens', [])
        emotions = [t for t in control_tokens if 'emotion' in t.get('token', '')]
        if emotions:
            print(f"   [Emotional state: {emotions[0]['token']}]")


async def multi_character_scene():
    """Example: Multiple characters in a scene"""
    print("\n=== Multi-Character Scene Example ===\n")
    
    async with InferenceClient() as client:
        session_id = "scene-session-789"
        
        # Characters in the scene
        characters = ["narrator", "hero", "villain"]
        
        # Scene setup
        print("📍 Location: Dark Castle Throne Room")
        print("🎭 Characters: Hero confronts the Villain\n")
        
        # Narrator sets the scene
        narrator_response = await client.generate(
            session_id=session_id,
            character_id="narrator",
            prompt="Describe the tense moment as the hero enters the throne room",
            forced_control_tokens=["<scene_change>throne_room</scene_change>"],
            max_tokens=100
        )
        
        print(f"📖 Narrator: {narrator_response['generation_text']}\n")
        
        # Hero speaks
        hero_response = await client.generate(
            session_id=session_id,
            character_id="hero",
            prompt="I've come to stop your evil plans!",
            forced_control_tokens=["<emotion_determined>"],
            max_tokens=80
        )
        
        print(f"⚔️ Hero: {hero_response['generation_text']}")
        
        # Villain responds
        villain_response = await client.generate(
            session_id=session_id,
            character_id="villain",
            prompt=hero_response['generation_text'],  # Villain responds to hero
            forced_control_tokens=["<emotion_amused>", "<action_laugh>"],
            max_tokens=100
        )
        
        print(f"👿 Villain: {villain_response['generation_text']}")
        
        # Show scene coordination
        if any('<scene_change>' in str(t) for t in villain_response.get('control_tokens', [])):
            print("\n🎬 [Scene transition triggered]")


async def main():
    """Run all examples"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     Production Inference Engine - Client Examples         ║
    ║                                                           ║
    ║  Demonstrating:                                           ║
    ║  • Mobile chat interactions                               ║
    ║  • Proactive agent check-ins                             ║
    ║  • Multi-character scenes                                 ║
    ║  • Memory formation and retrieval                        ║
    ║  • Control token processing                              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Note: Make sure the inference server is running!
    print("⚠️  Make sure the inference server is running on localhost:8000")
    print("   Run: python scripts/run_inference_server.py\n")
    
    try:
        # Run examples
        await mobile_chat_example()
        await asyncio.sleep(2)
        
        await proactive_agent_example()
        await asyncio.sleep(2)
        
        await multi_character_scene()
        
    except aiohttp.ClientError as e:
        print(f"\n❌ Error: Could not connect to server. Is it running?")
        print(f"   Details: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main()) 