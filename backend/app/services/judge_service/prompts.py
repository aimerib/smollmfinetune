"""
Prompt template management for Judge Service

Loads and manages prompt templates for personality alignment and lore adherence evaluation.
"""

import os
from pathlib import Path

# Get the directory containing this file
PROMPTS_DIR = Path(__file__).parent


def load_personality_prompt() -> str:
    """Load the personality alignment prompt template"""
    prompt_path = PROMPTS_DIR / "personality_prompt.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_lore_prompt() -> str:
    """Load the lore adherence prompt template"""
    prompt_path = PROMPTS_DIR / "lore_prompt.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def format_personality_prompt(response: str, big_five_scores: dict) -> str:
    """Format the personality prompt with actual values"""
    template = load_personality_prompt()
    return template.format(
        response=response,
        openness=big_five_scores.get('openness', 0.5),
        conscientiousness=big_five_scores.get('conscientiousness', 0.5),
        extraversion=big_five_scores.get('extraversion', 0.5),
        agreeableness=big_five_scores.get('agreeableness', 0.5),
        neuroticism=big_five_scores.get('neuroticism', 0.5)
    )


def format_lore_prompt(response: str, lore_fact: str) -> str:
    """Format the lore prompt with actual values"""
    template = load_lore_prompt()
    return template.format(
        response=response,
        lore_fact=lore_fact
    ) 