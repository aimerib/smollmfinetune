"""
Prompt templates and collections for dataset generation.

This module centralizes all prompt templates and prompt collections used for generating
diverse, character-specific conversations. Prompts are organized by category and use
template strings that can be personalized with character information.
"""

from typing import List, Tuple
from dataclasses import dataclass
from .prompt_registry import registry


@dataclass
class PromptCollection:
    """Container for categorized prompts with weights"""
    prompts: List[str]
    weight: float = 1.0
    description: str = ""


class PromptTemplates:
    """Central repository for all prompt templates and collections"""
    
    # Base conversation templates
    TEMPLATES = [
        "short_qa",
        "narration", 
        "monologue",
        "dialogue_turn",
        "internal_thought",
        "character_response",
    ]

    # All prompt lists are now managed by PromptRegistry
    # Legacy property accessors for backward compatibility
    @property 
    def CASUAL_PROMPTS(self) -> List[str]:
        return registry.get("casual")
    
    @property
    def PERSONAL_PROMPTS(self) -> List[str]:
        return registry.get("personal")
    
    @property
    def ACTION_PROMPTS(self) -> List[str]:
        return registry.get("action")
    
    @property
    def EMOTION_PROMPTS(self) -> List[str]:
        return registry.get("emotional")
    
    @property
    def INTIMATE_PROMPTS(self) -> List[str]:
        return registry.get("intimate")
    
    @property
    def NSFW_PROMPTS(self) -> List[str]:
        return registry.get("nsfw")
    
    @property
    def INTIMATE_EMOTIONAL_PROMPTS(self) -> List[str]:
        return registry.get("intimate_emotional")
    
    @property
    def INTIMATE_PLAYFUL_PROMPTS(self) -> List[str]:
        return registry.get("intimate_playful")
    
    @property
    def INTIMATE_ROMANTIC_PROMPTS(self) -> List[str]:
        return registry.get("intimate_romantic")
    
    @property
    def PAST_ROMANCE_PROMPTS(self) -> List[str]:
        return registry.get("past_romance")
    
    @property
    def PAST_FAMILY_PROMPTS(self) -> List[str]:
        return registry.get("past_family")
    
    @property
    def PAST_FRIENDS_PROMPTS(self) -> List[str]:
        return registry.get("past_friends")
    
    @property
    def PRESENT_MEETING_PROMPTS(self) -> List[str]:
        return registry.get("present_meeting")
    
    @property
    def PRESENT_BONDING_PROMPTS(self) -> List[str]:
        return registry.get("present_bonding")
    
    @property
    def FUTURE_ROMANCE_PROMPTS(self) -> List[str]:
        return registry.get("future_romance")
    
    @property
    def FUTURE_DESIRES_PROMPTS(self) -> List[str]:
        return registry.get("future_desires")

    # Character-specific prompt templates (use placeholders)
    CHARACTER_SPECIFIC_TEMPLATES = {
        "trait_exploration": [
            "People say you're {trait}. Is that accurate?",
            "When did you first become {trait}?",
            "What made you so {trait}?",
            "Do you ever wish you weren't so {trait}?"
        ],
        "skill_exploration": [
            "Show me your {skill}.",
            "What's your greatest achievement with {skill}?",
            "Who taught you {skill}?",
            "What's the secret to mastering {skill}?"
        ],
        "species_exploration": [
            "What's it like being a {species}?",
            "Are there any misconceptions about {species}?",
            "What advantages does being a {species} give you?",
            "Do you face any challenges as a {species}?"
        ],
        "appearance_exploration": [
            "Is there a story behind your {feature}?",
            "How do people react to your {feature}?",
            "Are you self-conscious about your {feature}?",
            "What does your {feature} mean to you?"
        ]
    }

    # Default fallback prompts
    DEFAULT_QUESTIONS: List[str] = registry.get("default")

    DEFAULT_TOPICS = [
        "the nature of courage",
        "loneliness on the road", 
        "the weight of leadership",
        "how the stars guide travellers",
        "the meaning of home",
        "finding purpose in chaos",
        "the price of power",
        "love and loss",
    ]

    DEFAULT_SITUATIONS = [
        "facing an impossible challenge",
        "meeting an old enemy",
        "discovering a hidden truth", 
        "making a difficult choice",
        "losing something important",
        "facing certain death",
        "experiencing unexpected kindness",
        "confronting past mistakes",
        "finding unexpected allies",
        "dealing with betrayal",
    ]

    @classmethod
    def get_bucket_choices(cls) -> List[Tuple[str, float]]:
        """Get the bucket choices with their sampling weights"""
        return [
            ("casual", 0.20),
            ("personal", 0.15),
            ("action", 0.10),
            ("emotional", 0.10),
            ("intimate", 0.10),
            ("worldbuilding", 0.05),
            ("cnc_scene", 0.05),
            ("nsfw", 0.30),
            ("dirty_soft", 0.05),
            ("dirty_explicit", 0.05),
            ("dirty_filthy", 0.03),
        ]

    @classmethod
    def get_temporal_buckets(cls) -> List[Tuple[str, float]]:
        """Get temporal bucket distribution"""
        return [
            ("past", 0.30),      # 30% past relationships/events
            ("present", 0.50),   # 50% first meeting/getting to know  
            ("future", 0.20),    # 20% established relationship
        ]

    @classmethod
    def get_prompts_by_bucket(cls, bucket_name: str) -> List[str]:
        """Get prompts for a specific bucket - thin wrapper around PromptRegistry"""
        return registry.get(bucket_name)

    @classmethod
    def get_character_specific_prompts(cls, template_type: str, **kwargs) -> List[str]:
        """Generate character-specific prompts from templates"""
        templates = cls.CHARACTER_SPECIFIC_TEMPLATES.get(template_type, [])
        return [template.format(**kwargs) for template in templates] 