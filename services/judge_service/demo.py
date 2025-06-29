#!/usr/bin/env python3
"""
Demo script for Judge Model Service

Shows how to interact with the Judge Service endpoints.
"""

import asyncio
import json
import httpx

BASE_URL = "http://localhost:8000"

async def demo_judge_service():
    """Demonstrate the Judge Model Service functionality"""
    async with httpx.AsyncClient() as client:
        print("🎯 Judge Model Service Demo")
        print("=" * 40)
        
        # Health check
        print("\n1. Health Check")
        health_response = await client.get(f"{BASE_URL}/health")
        print(f"Status: {health_response.status_code}")
        health_data = health_response.json()
        print(f"Dev Mode: {health_data['dev_mode']}")
        print(f"Cache Stats: {health_data['cache_stats']}")
        
        # Personality alignment evaluation
        print("\n2. Personality Alignment Evaluation")
        personality_request = {
            "text": "I absolutely love trying new cuisines and exploring different cultures! It's so exciting to discover something completely unexpected.",
            "target": {
                "openness": 0.9,
                "conscientiousness": 0.5,
                "extraversion": 0.8,
                "agreeableness": 0.7,
                "neuroticism": 0.2
            }
        }
        
        personality_response = await client.post(
            f"{BASE_URL}/personality_alignment", 
            json=personality_request
        )
        print(f"Status: {personality_response.status_code}")
        personality_data = personality_response.json()
        print(f"Score: {personality_data['score']:.3f}")
        print(f"Cache Hit: {personality_data['cache_hit']}")
        
        # Lore adherence evaluation
        print("\n3. Lore Adherence Evaluation")
        lore_request = {
            "text": "I've never seen magic being used in the capital city - it's strictly forbidden here.",
            "target": "Magic is forbidden in the capital city"
        }
        
        lore_response = await client.post(
            f"{BASE_URL}/lore_adherence", 
            json=lore_request
        )
        print(f"Status: {lore_response.status_code}")
        lore_data = lore_response.json()
        print(f"Score: {lore_data['score']:.3f}")
        print(f"Cache Hit: {lore_data['cache_hit']}")
        
        # Test caching by repeating the same request
        print("\n4. Testing Cache (repeat personality request)")
        personality_response_2 = await client.post(
            f"{BASE_URL}/personality_alignment", 
            json=personality_request
        )
        personality_data_2 = personality_response_2.json()
        print(f"Score: {personality_data_2['score']:.3f}")
        print(f"Cache Hit: {personality_data_2['cache_hit']}")
        
        print("\n✅ Demo complete!")

if __name__ == "__main__":
    print("Starting Judge Model Service demo...")
    print("Make sure the service is running on http://localhost:8000")
    print("You can start it with: uvicorn services.judge_service.main:app --reload")
    print()
    
    try:
        asyncio.run(demo_judge_service())
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure the Judge Service is running!") 