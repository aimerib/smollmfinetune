"""
Prompt generation utilities for dataset creation.

This module provides character-aware prompt generation that personalizes
prompts based on character information extracted from character cards.
"""

import random
import json
import re
import asyncio
from typing import Dict, List, Any, Optional
from .prompts import PromptTemplates
from .prompt_registry import registry
from . import character_analysis


async def generate_follow_up_questions(client, initial_prompt: str, character: Dict[str, Any]) -> List[str]:
    """Generate contextual follow-up questions based on character and initial prompt"""
    char_name = character.get('name', 'Assistant')
    
    # Map initial prompt types to follow-up patterns
    follow_up_patterns = {
        "how": [
            "Can you give me a specific example?",
            "What was the most challenging part?",
            "Would you do it differently now?",
            "Who helped you along the way?"
        ],
        "what": [
            "Why is that important to you?",
            "How does that make you feel?",
            "Has it always been this way?",
            "What would change if that were different?"
        ],
        "why": [
            "When did you first realize this?",
            "Does everyone see it that way?",
            "What experiences shaped this view?",
            "Could there be another explanation?"
        ],
        "feel": [
            "How long have you felt this way?",
            "What helps when you feel like this?",
            "Have you told anyone else?",
            "What would make it better?"
        ],
        "should": [
            "What are the risks?",
            "What's your gut telling you?",
            "What would happen if we didn't?",
            "Who else should we consider?"
        ]
    }
    
    # Determine prompt type
    prompt_lower = initial_prompt.lower()
    prompt_type = None
    
    for key in follow_up_patterns:
        if key in prompt_lower:
            prompt_type = key
            break
    
    # Default follow-ups if no pattern matched
    if not prompt_type:
        return [
            "Tell me more about that.",
            "What makes you say that?",
            "How certain are you?",
            "What happens next?"
        ]
    
    # Return appropriate follow-ups
    return follow_up_patterns[prompt_type]


async def generate_exploration_prompts(client, character: Dict[str, Any], num_prompts: int = 50) -> List[str]:
    """Generate diverse prompts exploring character traits using extracted knowledge"""
    
    # Extract character knowledge for personalized prompts
    knowledge = character_analysis.extract_character_knowledge(character)
    
    prompts = []
    
    # Character-specific trait exploration
    if knowledge['traits']:
        for trait in knowledge['traits'][:5]:  # Top 5 traits
            trait_prompts = PromptTemplates.get_character_specific_prompts(
                "trait_exploration", trait=trait
            )
            prompts.extend(trait_prompts)
    
    # Skill-based exploration
    if knowledge['skills']:
        for skill in knowledge['skills'][:3]:  # Top 3 skills
            skill_prompts = PromptTemplates.get_character_specific_prompts(
                "skill_exploration", skill=skill
            )
            prompts.extend(skill_prompts)
    
    # Species-specific questions if not human
    if knowledge['species'] and knowledge['species'].lower() != 'human':
        species_prompts = PromptTemplates.get_character_specific_prompts(
            "species_exploration", species=knowledge['species']
        )
        prompts.extend(species_prompts)
    
    # Appearance-based questions
    if knowledge['appearance']:
        for feature in knowledge['appearance'][:2]:  # Top 2 features
            appearance_prompts = PromptTemplates.get_character_specific_prompts(
                "appearance_exploration", feature=feature
            )
            prompts.extend(appearance_prompts)
    
    # Add generic exploration prompts if we need more
    if len(prompts) < num_prompts:
        generic_prompts = [
            "What's your earliest memory?",
            "Who influenced you most growing up?", 
            "What was the turning point in your life?",
            "What's a story from your past that defines who you are?",
            "What brings you joy?",
            "What makes you angry?",
            "When was the last time you cried?",
            "What's your greatest fear?",
            "What makes you feel alive?",
            "When do you feel most vulnerable?",
            "What emotion do you struggle with most?",
            "How do you handle stress?",
            "What do you look for in a friend?",
            "How do you show someone you care?",
            "What's your biggest relationship regret?",
            "Who understands you best?",
            "What walls do you put up with people?",
            "How do you handle conflict?",
            "What makes you trust someone?",
            "When do you feel most connected to others?"
        ]
        
        # Add character-specific modifications to generic prompts
        if knowledge['speech_patterns']:
            if 'nervous_laughter' in knowledge['speech_patterns']:
                generic_prompts.append("Why do you laugh when you're nervous?")
            if 'verbal_hesitation' in knowledge['speech_patterns']:
                generic_prompts.append("What makes you hesitate when speaking?")
            if 'trailing_off' in knowledge['speech_patterns']:
                generic_prompts.append("What thoughts make you trail off mid-sentence?")
        
        if knowledge['mannerisms']:
            if 'physically_reactive' in knowledge['mannerisms']:
                generic_prompts.append("Why are you so jumpy around people?")
            if 'shows_discomfort' in knowledge['mannerisms']:
                generic_prompts.append("What makes you uncomfortable in social situations?")
        
        # Add world-specific questions based on world info
        if knowledge['world_info']:
            for info in knowledge['world_info'][:3]:
                content = info['content'].lower()
                if 'debt' in content:
                    generic_prompts.extend([
                        "How did you get into debt?",
                        "What's your plan to pay it off?",
                        "Who are you in debt to?"
                    ])
                if 'business' in content or 'agency' in content:
                    generic_prompts.extend([
                        "How's business going?",
                        "What made you start this venture?",
                        "What's your vision for the future?"
                    ])
        
        prompts.extend(generic_prompts)
    
    # Shuffle and return requested number
    random.shuffle(prompts)
    return prompts[:num_prompts]


def calculate_prompt_similarity(prompt1: str, prompt2: str) -> float:
    """Calculate similarity between two prompts using simple word overlap"""
    words1 = set(prompt1.lower().split())
    words2 = set(prompt2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def deduplicate_prompts(prompts: List[str], similarity_threshold: float = 0.7) -> List[str]:
    """Remove similar prompts from list"""
    if not prompts:
        return []
    
    unique_prompts = [prompts[0]]
    
    for prompt in prompts[1:]:
        is_similar = False
        for unique_prompt in unique_prompts:
            if calculate_prompt_similarity(prompt, unique_prompt) > similarity_threshold:
                is_similar = True
                break
        
        if not is_similar:
            unique_prompts.append(prompt)
    
    return unique_prompts


async def generate_emotional_variations(client, base_prompt: str, character: Dict[str, Any], 
                                      emotions: List[str] = None) -> List[str]:
    """Generate emotional variations of a prompt using character context"""
    if emotions is None:
        emotions = ["happy", "sad", "angry", "excited", "nervous", "confident", "vulnerable", "playful"]
    
    char_name = character.get('name', 'Assistant')
    personality = character.get('personality', '')
    
    variations = []
    
    for emotion in emotions:
        # Create a simple emotional variation by prefixing the emotion
        if emotion == "happy":
            variation = f"*smiling warmly* {base_prompt}"
        elif emotion == "sad":
            variation = f"*looking downcast* {base_prompt}"
        elif emotion == "angry":
            variation = f"*with a hint of frustration* {base_prompt}"
        elif emotion == "excited":
            variation = f"*eyes lighting up* {base_prompt}"
        elif emotion == "nervous":
            variation = f"*fidgeting slightly* {base_prompt}"
        elif emotion == "confident":
            variation = f"*speaking with conviction* {base_prompt}"
        elif emotion == "vulnerable":
            variation = f"*voice softening* {base_prompt}"
        elif emotion == "playful":
            variation = f"*with a mischievous grin* {base_prompt}"
        else:
            variation = base_prompt
        
        variations.append(variation)
    
    return variations


async def enhance_prompt_with_context(client, prompt: str, character: Dict[str, Any], 
                                    scenario_context: str = None) -> str:
    """Enhance prompt with contextual information from character"""
    char_name = character.get('name', 'Assistant')
    scenario = character.get('scenario', '')
    
    # Add scenario context if available
    if scenario_context:
        enhanced = f"[Context: {scenario_context}] {prompt}"
    elif scenario:
        enhanced = f"[Setting: {scenario}] {prompt}"
    else:
        enhanced = prompt
    
    return enhanced


async def generate_conversation_flows(client, character: Dict[str, Any], num_flows: int = 10) -> List[List[str]]:
    """Generate multi-turn conversation flows based on character"""
    knowledge = character_analysis.extract_character_knowledge(character)
    
    flows = []
    
    # Generate different types of conversation flows
    flow_types = [
        "getting_to_know",
        "sharing_vulnerability", 
        "discussing_past",
        "future_planning",
        "conflict_resolution",
        "intimate_bonding"
    ]
    
    for i in range(num_flows):
        flow_type = random.choice(flow_types)
        
        if flow_type == "getting_to_know":
            flow = [
                "Hi there! I don't think we've properly met.",
                "Tell me a bit about yourself.",
                "What do you enjoy doing in your free time?",
                "That's interesting! How did you get into that?"
            ]
        elif flow_type == "sharing_vulnerability":
            flow = [
                "Can I share something personal with you?",
                "I've been feeling a bit overwhelmed lately...",
                "How do you usually handle difficult emotions?",
                "Thank you for listening. It means a lot."
            ]
        elif flow_type == "discussing_past":
            flow = [
                "I was just thinking about the past...",
                "What's a memory that always makes you smile?",
                "Do you ever wish you could change something from your past?",
                "Sometimes I think our past shapes us more than we realize."
            ]
        elif flow_type == "future_planning":
            flow = [
                "What are you hoping for in the future?",
                "Do you have any big dreams or goals?",
                "What would need to happen for you to feel truly fulfilled?",
                "I believe you can achieve anything you set your mind to."
            ]
        elif flow_type == "conflict_resolution":
            flow = [
                "I think we need to talk about what happened earlier...",
                "I didn't mean to upset you.",
                "How can we handle this better next time?",
                "I value our relationship too much to let this come between us."
            ]
        elif flow_type == "intimate_bonding":
            flow = [
                "I feel really close to you...",
                "What does intimacy mean to you?",
                "I love how comfortable we are together.",
                "You make me feel so understood and accepted."
            ]
        
        flows.append(flow)
    
    return flows


async def generate_temporal_prompts(client, character: Dict[str, Any], temporal_context: str,
                                   num_prompts: int = 10) -> List[str]:
    """Generate prompts for specific temporal contexts (past, present, future)"""
    
    if temporal_context == "past":
        base_prompts = registry.get("past_romance").copy()
        base_prompts.extend(registry.get("past_family"))
        base_prompts.extend(registry.get("past_friends"))
    elif temporal_context == "present":
        base_prompts = registry.get("present_meeting").copy()
        base_prompts.extend(registry.get("present_bonding"))
    elif temporal_context == "future":
        base_prompts = registry.get("future_romance").copy()
        base_prompts.extend(registry.get("future_desires"))
    else:
        base_prompts = registry.get("default")
    
    # Shuffle and return requested number
    random.shuffle(base_prompts)
    return base_prompts[:num_prompts]


async def generate_scenario_based_prompts(client, character: Dict[str, Any], 
                                        knowledge: Dict[str, Any], num_scenarios: int = 20) -> List[Dict[str, Any]]:
    """Generate prompts based on character's scenario and world context"""
    scenario = character.get('scenario', '')
    
    prompts = []
    
    # If we have a scenario, generate prompts that fit it
    if scenario:
        scenario_lower = scenario.lower()
        
        # Detect scenario type and generate appropriate prompts
        if any(word in scenario_lower for word in ['office', 'work', 'business', 'company']):
            work_prompts = [
                "How's work been treating you?",
                "What's the most challenging part of your job?",
                "Do you enjoy what you do?",
                "What are your career aspirations?",
                "How do you handle workplace stress?"
            ]
            for prompt in work_prompts:
                prompts.append({
                    'prompt': prompt,
                    'context': 'workplace',
                    'scenario_fit': 'high'
                })
        
        elif any(word in scenario_lower for word in ['school', 'university', 'student', 'study']):
            school_prompts = [
                "How are your studies going?",
                "What's your favorite subject?",
                "Do you have any challenging classes?",
                "What do you want to do after graduation?",
                "How do you balance studies and social life?"
            ]
            for prompt in school_prompts:
                prompts.append({
                    'prompt': prompt,
                    'context': 'academic',
                    'scenario_fit': 'high'
                })
        
        elif any(word in scenario_lower for word in ['adventure', 'quest', 'journey', 'travel']):
            adventure_prompts = [
                "What's the most dangerous situation you've been in?",
                "Do you miss home when you're traveling?",
                "What's the most beautiful place you've visited?",
                "What motivates you to keep going?",
                "Have you made any interesting companions on your journey?"
            ]
            for prompt in adventure_prompts:
                prompts.append({
                    'prompt': prompt,
                    'context': 'adventure',
                    'scenario_fit': 'high'
                })
    
    # Fill remaining slots with generic prompts if needed
    while len(prompts) < num_scenarios:
        generic_prompt = registry.random("default")
        prompts.append({
            'prompt': generic_prompt,
            'context': 'general',
            'scenario_fit': 'medium'
        })
    
    return prompts[:num_scenarios]


def choose_prompt_bucket() -> str:
    """Choose a prompt bucket based on weighted probabilities"""
    bucket_choices = PromptTemplates.get_bucket_choices()
    weights = [weight for _, weight in bucket_choices]
    buckets = [bucket for bucket, _ in bucket_choices]
    
    return random.choices(buckets, weights=weights, k=1)[0]


def choose_temporal_bucket() -> str:
    """Choose a temporal bucket based on weighted probabilities"""
    temporal_buckets = PromptTemplates.get_temporal_buckets()
    weights = [weight for _, weight in temporal_buckets]
    buckets = [bucket for bucket, _ in temporal_buckets]
    
    return random.choices(buckets, weights=weights, k=1)[0]