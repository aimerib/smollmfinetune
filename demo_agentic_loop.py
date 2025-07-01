#!/usr/bin/env python3
"""
Demo: Agentic Loop Framework

Demonstrates autonomous NPCs using the perceive -> think -> act cycle
in a living world environment.
"""

import asyncio
import logging
from datetime import datetime
from narrative_engine.state_manager import StateManager, EntityState
from narrative_engine.agent import BaseAgent, Scheduler

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_living_world():
    """
    Demonstrate a living world with autonomous NPCs.
    
    Clara and Tom will autonomously explore and interact!
    """
    print("🌟 Starting Living World Demo with Agentic Loop Framework!")
    print("=" * 60)
    
    # 1. Create the world state
    state_manager = StateManager()
    
    # Create Clara in the Village Square
    clara = EntityState(
        entity_id="npc_clara",
        entity_type="character",
        location="Village Square",
        custom_data={
            "mood": "curious",
            "personality": "friendly and adventurous",
            "goals": ["Explore the village", "Make new friends"]
        }
    )
    
    # Create Tom in the Forest
    tom = EntityState(
        entity_id="npc_tom",
        entity_type="character",
        location="Forest",
        custom_data={
            "mood": "contemplative",
            "personality": "wise and helpful",
            "goals": ["Help travelers", "Study nature"]
        }
    )
    
    state_manager.create_entity(clara)
    state_manager.create_entity(tom)
    print(f"✅ Created world with Clara in {clara.location} and Tom in {tom.location}")
    
    # 2. Create autonomous agents (no AI model for demo, using fallback behavior)
    clara_agent = BaseAgent(
        agent_id="npc_clara",
        narrative_model=None,  # Using fallback behavior
        goals=["Explore the village", "Make new friends"],
        personality_traits={"extraversion": 0.8, "openness": 0.9}
    )
    
    tom_agent = BaseAgent(
        agent_id="npc_tom", 
        narrative_model=None,  # Using fallback behavior
        goals=["Help travelers", "Study nature"],
        personality_traits={"agreeableness": 0.9, "conscientiousness": 0.8}
    )
    
    print("✅ Created autonomous agents with distinct personalities")
    
    # 3. Set up the scheduler
    scheduler = Scheduler(
        state_manager=state_manager,
        tick_rate=3.0,  # 3 seconds between ticks
        max_agents_per_tick=1  # Process one agent per tick
    )
    
    scheduler.register_agent(clara_agent)
    scheduler.register_agent(tom_agent)
    print("✅ Registered agents with scheduler (3-second tick rate)")
    
    # 4. Run the living world for a short demo
    print("\n🚀 Starting the living world! Watch the agents act autonomously...")
    print("-" * 60)
    
    # Start the scheduler
    scheduler_task = asyncio.create_task(scheduler.run())
    
    # Let the world run for 15 seconds (5 ticks)
    await asyncio.sleep(15)
    
    # Stop the scheduler
    scheduler.stop()
    await scheduler_task
    
    print("-" * 60)
    print("🏁 Demo complete! The world has been running autonomously.")
    
    # 5. Show the final world state
    print("\n📊 Final World State:")
    final_clara = state_manager.get_entity("npc_clara")
    final_tom = state_manager.get_entity("npc_tom")
    
    print(f"Clara is now in: {final_clara.location}")
    print(f"Tom is now in: {final_tom.location}")
    
    # Show recent events
    recent_events = state_manager.get_recent_events(limit=10)
    if recent_events:
        print("\n📜 Recent Events in the World:")
        for event in recent_events[-5:]:  # Last 5 events
            print(f"  • {event.timestamp.strftime('%H:%M:%S')} - {event.event_type}: {event.entity_id}")
    
    print("\n🎯 What Happened:")
    print("• Each agent autonomously perceived their environment")
    print("• They made decisions based on their personality and goals")
    print("• They executed actions that changed the world state")
    print("• The world evolved through their autonomous behavior!")
    print("\nThis is the foundation for truly living NPCs! 🌟")


async def demo_single_agent_cycle():
    """
    Demonstrate a single agent going through the perceive -> think -> act cycle.
    """
    print("\n" + "=" * 60)
    print("🔍 Single Agent Cycle Demo")
    print("=" * 60)
    
    # Set up world
    state_manager = StateManager()
    
    # Create an NPC and a player in the same location
    npc = EntityState(
        entity_id="npc_guide",
        entity_type="character",
        location="Village Entrance",
        custom_data={"mood": "welcoming", "role": "village guide"}
    )
    
    player = EntityState(
        entity_id="player_001",
        entity_type="character",
        location="Village Entrance",
        custom_data={"mood": "curious", "role": "traveler"}
    )
    
    state_manager.create_entity(npc)
    state_manager.create_entity(player)
    
    # Create agent
    guide_agent = BaseAgent(
        agent_id="npc_guide",
        narrative_model=None,
        goals=["Welcome new travelers", "Provide helpful information"],
        personality_traits={"extraversion": 0.9, "agreeableness": 0.8}
    )
    
    print("🏞️  Scene: A village guide meets a traveler at the entrance")
    print(f"📍 Location: {npc.location}")
    print(f"👥 Present: {npc.entity_id}, {player.entity_id}")
    
    # Execute one complete cycle
    print("\n🧠 Executing Perceive -> Think -> Act cycle...")
    
    # PERCEIVE
    print("\n1️⃣ PERCEIVE:")
    perception = await guide_agent.perceive(state_manager)
    print(f"   • Agent: {perception.agent_id}")
    print(f"   • Location: {perception.current_location}")
    print(f"   • Nearby: {perception.nearby_agents}")
    print(f"   • Mood: {perception.agent_state}")
    
    # THINK
    print("\n2️⃣ THINK:")
    action = await guide_agent.think(perception)
    print(f"   • Decision: {action.action_type}")
    print(f"   • Action: {action}")
    
    # ACT
    print("\n3️⃣ ACT:")
    result = await guide_agent.act(action, state_manager)
    print(f"   • Success: {result.success}")
    print(f"   • Result: {result.message}")
    if result.side_effects:
        print(f"   • Effects: {result.side_effects}")
    
    print("\n✨ The agent successfully completed one full autonomous cycle!")


async def main():
    """Run all demos"""
    print("🎮 Agentic Loop Framework Demo")
    print("Building the foundation of living worlds!")
    
    # Run the single cycle demo first
    await demo_single_agent_cycle()
    
    # Then run the living world demo
    await demo_living_world()
    
    print("\n" + "=" * 60)
    print("🏆 Demo complete! You've seen the future of autonomous NPCs!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main()) 