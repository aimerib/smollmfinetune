#!/usr/bin/env python3
"""
Character to Narrative Format Converter

This script converts character folders (character_core.json + mes_example.txt) 
into the unified DatasetSample format for the Narrative-LLM training pipeline.
"""

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

from narrative_engine.data_schema import Turn, DatasetSample


def convert_character_to_dataset_sample(char_folder: Path) -> DatasetSample:
    """
    Convert a character folder to a DatasetSample object.
    
    Args:
        char_folder: Path to character folder containing character_core.json
        
    Returns:
        DatasetSample object ready for training
        
    Raises:
        FileNotFoundError: If character_core.json doesn't exist
        ValueError: If character data is invalid
    """
    char_folder = Path(char_folder)
    
    # Load character_core.json
    core_file = char_folder / "character_core.json"
    if not core_file.exists():
        raise FileNotFoundError(f"character_core.json not found in {char_folder}")
    
    with open(core_file, 'r', encoding='utf-8') as f:
        character_core = json.load(f)
    
    # Load mes_example.txt if available
    example_file = char_folder / "mes_example.txt"
    mes_example = ""
    if example_file.exists():
        with open(example_file, 'r', encoding='utf-8') as f:
            mes_example = f.read()
    
    # Generate session ID
    session_id = f"char_{character_core['name'].replace(' ', '_').lower()}_{uuid.uuid4().hex[:8]}"
    
    # Create persona mix from character tags and traits
    persona_mix = _create_persona_mix(character_core)
    
    # Generate memory slots from character background
    memory_slots = _extract_memory_slots(character_core)
    
    # Convert mes_example to turns, or create basic turns if no example
    turns = _convert_example_to_turns(mes_example, character_core) if mes_example else _create_basic_turns(character_core)
    
    return DatasetSample(
        session_id=session_id,
        persona_mix=persona_mix,
        memory_slots=memory_slots,
        turns=turns
    )


def _create_persona_mix(character_core: Dict[str, Any]) -> Dict[str, float]:
    """Create persona mix based on character tags and personality traits."""
    persona_mix = {}
    
    # Extract primary persona from tags
    tags = character_core.get('tags', [])
    
    # Map tags to persona names
    tag_to_persona = {
        'detective': 'DetectiveNoir',
        'noir': 'DetectiveNoir', 
        'supernatural': 'CosmicHorror',
        'fantasy': 'FantasyAdventurer',
        'scifi': 'SciFiExplorer',
        'romance': 'RomanticLead',
        'comedy': 'ComedyCharacter',
        'horror': 'HorrorProtagonist'
    }
    
    # Build persona mix from tags
    for tag in tags:
        tag_lower = tag.lower()
        if tag_lower in tag_to_persona:
            persona_name = tag_to_persona[tag_lower]
            persona_mix[persona_name] = persona_mix.get(persona_name, 0) + 0.5
    
    # If no tags matched, create a default persona based on personality
    if not persona_mix:
        personality = character_core.get('personality_traits', {})
        
        # High openness + low neuroticism = Explorer
        if personality.get('openness', 0.5) > 0.7 and personality.get('neuroticism', 0.5) < 0.4:
            persona_mix['Explorer'] = 0.8
        # High conscientiousness + low extraversion = Scholar  
        elif personality.get('conscientiousness', 0.5) > 0.7 and personality.get('extraversion', 0.5) < 0.4:
            persona_mix['Scholar'] = 0.8
        # High extraversion + high agreeableness = Companion
        elif personality.get('extraversion', 0.5) > 0.7 and personality.get('agreeableness', 0.5) > 0.7:
            persona_mix['Companion'] = 0.8
        # Default balanced persona
        else:
            persona_mix['Balanced'] = 1.0
    
    # Normalize weights to sum to 1.0
    total_weight = sum(persona_mix.values())
    if total_weight > 0:
        persona_mix = {k: v / total_weight for k, v in persona_mix.items()}
    else:
        persona_mix = {'Default': 1.0}
    
    return persona_mix


def _extract_memory_slots(character_core: Dict[str, Any]) -> List[str]:
    """Extract key memories from character background information."""
    memory_slots = []
    
    # Add core character facts
    name = character_core.get('name', 'Unknown')
    memory_slots.append(f"Character name is {name}")
    
    # Add key personality insight
    personality = character_core.get('personality_traits', {})
    if personality:
        # Find the most extreme trait
        max_trait = max(personality.items(), key=lambda x: abs(x[1] - 0.5))
        trait_name, trait_value = max_trait
        if trait_value > 0.7:
            memory_slots.append(f"Character is highly {trait_name.lower()}")
        elif trait_value < 0.3:
            memory_slots.append(f"Character is low in {trait_name.lower()}")
    
    # Add primary goal
    goals = character_core.get('goals', [])
    if goals:
        memory_slots.append(f"Character's main goal: {goals[0]}")
    
    # Add key relationship
    relationships = character_core.get('relationships', [])
    if relationships:
        rel = relationships[0]
        rel_name = rel.get('name', 'Unknown')
        affinity = rel.get('affinity', 0)
        if affinity > 50:
            memory_slots.append(f"Character trusts {rel_name}")
        elif affinity < -50:
            memory_slots.append(f"Character has conflict with {rel_name}")
        else:
            memory_slots.append(f"Character knows {rel_name}")
    
    # Add scenario context
    scenario = character_core.get('scenario', '')
    if scenario:
        # Extract key context from scenario
        memory_slots.append(f"Current situation: {scenario[:100]}...")
    
    return memory_slots[:5]  # Limit to 5 memory slots


def _convert_example_to_turns(mes_example: str, character_core: Dict[str, Any]) -> List[Turn]:
    """Convert mes_example.txt content into Turn objects."""
    turns = []
    char_name = character_core.get('name', 'Assistant')
    
    # Split example into lines and parse conversation
    lines = mes_example.strip().split('\n')
    current_speaker = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this line starts a new speaker turn
        if line.startswith('User:'):
            # Save previous turn if exists
            if current_speaker and current_text:
                text = ' '.join(current_text).strip()
                if text:
                    channel = "action" if _contains_action_indicators(text) else "text"
                    turns.append(Turn(
                        sender=current_speaker,
                        text=text,
                        channel=channel,
                        action=_extract_action(text) if channel == "action" else None
                    ))
            
            # Start new user turn
            current_speaker = "user"
            current_text = [line[5:].strip()]  # Remove "User:" prefix
            
        elif line.startswith(f'{char_name}:') or line.startswith('Assistant:'):
            # Save previous turn if exists
            if current_speaker and current_text:
                text = ' '.join(current_text).strip()
                if text:
                    channel = "action" if _contains_action_indicators(text) else "text"
                    turns.append(Turn(
                        sender=current_speaker,
                        text=text,
                        channel=channel,
                        action=_extract_action(text) if channel == "action" else None
                    ))
            
            # Start new assistant turn
            current_speaker = "assistant"
            prefix = f'{char_name}:' if line.startswith(f'{char_name}:') else 'Assistant:'
            current_text = [line[len(prefix):].strip()]
            
        elif current_speaker:
            # Continue current turn
            current_text.append(line)
    
    # Add final turn
    if current_speaker and current_text:
        text = ' '.join(current_text).strip()
        if text:
            channel = "action" if _contains_action_indicators(text) else "text"
            turns.append(Turn(
                sender=current_speaker,
                text=text,
                channel=channel,
                action=_extract_action(text) if channel == "action" else None
            ))
    
    return turns


def _create_basic_turns(character_core: Dict[str, Any]) -> List[Turn]:
    """Create basic conversation turns when no mes_example is available."""
    char_name = character_core.get('name', 'Assistant')
    description = character_core.get('description', '')
    
    turns = [
        Turn(
            sender="user",
            text="Hello! Can you tell me about yourself?",
            channel="text"
        ),
        Turn(
            sender="assistant", 
            text=f"Hello! I'm {char_name}. {description}",
            channel="text"
        )
    ]
    
    # Add a goal-related exchange if goals exist
    goals = character_core.get('goals', [])
    if goals:
        turns.extend([
            Turn(
                sender="user",
                text="What are you trying to accomplish?",
                channel="text"
            ),
            Turn(
                sender="assistant",
                text=f"My main focus right now is {goals[0].lower()}. It's really important to me.",
                channel="text"
            )
        ])
    
    return turns


def _contains_action_indicators(text: str) -> bool:
    """Check if text contains indicators of structured actions."""
    # Simple heuristics for action detection
    action_indicators = [
        'search', 'find', 'look up', 'check', 'verify', 'analyze',
        'remember', 'recall', 'access', 'query', 'investigate'
    ]
    
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in action_indicators)


def _extract_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract structured action from text if possible."""
    text_lower = text.lower()
    
    # Simple action extraction patterns
    if 'search' in text_lower or 'find' in text_lower:
        return {
            "tool": "search",
            "query": text[:50] + "..." if len(text) > 50 else text
        }
    elif 'remember' in text_lower or 'recall' in text_lower:
        return {
            "tool": "memory_recall",
            "context": text[:50] + "..." if len(text) > 50 else text
        }
    elif 'check' in text_lower or 'verify' in text_lower:
        return {
            "tool": "verification",
            "target": text[:50] + "..." if len(text) > 50 else text
        }
    
    # Generic fallback action
    return {
        "tool": "generic_action",
        "description": text[:100] + "..." if len(text) > 100 else text
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python convert_to_narrative_format.py <character_folder>")
        sys.exit(1)
    
    char_folder = Path(sys.argv[1])
    try:
        dataset_sample = convert_character_to_dataset_sample(char_folder)
        print(json.dumps(dataset_sample.model_dump(), indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) 