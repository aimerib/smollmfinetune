"""
Prompt Templates for Narrative Engine

Contains reusable prompt templates for different narrative engine functions.
"""

# Iceberg Model (Subtext Generation) Prompt Suffix
SUBTEXT_ACTION_PROMPT_SUFFIX = """
Based on the above, generate the character's next action and their private inner thoughts.

[SUBTEXT]
(The character's inner monologue and true feelings)
[/SUBTEXT]

[ACTION]
(The structured action the character will perform, in the specified format)
[/ACTION]
"""

def build_subtext_prompt(base_prompt: str) -> str:
    """
    Build a prompt that requests both action and subtext generation.
    
    Args:
        base_prompt: The base prompt describing the character and situation
        
    Returns:
        Enhanced prompt with subtext/action tags
    """
    return base_prompt + SUBTEXT_ACTION_PROMPT_SUFFIX 