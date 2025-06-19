from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class Personality(BaseModel):
    """Big Five personality traits (0-1 scale)"""
    openness: float = Field(default=0.5, ge=0.0, le=1.0, description="Openness to experience")
    conscientiousness: float = Field(default=0.5, ge=0.0, le=1.0, description="Conscientiousness")
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0, description="Extraversion")
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0, description="Agreeableness")
    neuroticism: float = Field(default=0.5, ge=0.0, le=1.0, description="Neuroticism")


class Relationship(BaseModel):
    """Character relationship with affinity score"""
    name: str = Field(..., description="Name of the related character/entity")
    affinity: int = Field(..., ge=-100, le=100, description="Relationship affinity (-100 to 100)")


class CharacterCore(BaseModel):
    """Core character data structure for the new format"""
    name: str = Field(..., description="Character name")
    description: str = Field(..., description="Character description")
    scenario: str = Field(default="", description="Current scenario/setting")
    backstory: str = Field(default="", description="Character backstory")
    appearance: str = Field(default="", description="Physical appearance description")
    personality_traits: Personality = Field(default_factory=Personality, description="Big Five personality traits")
    goals: List[str] = Field(default_factory=list, description="Character goals and motivations")
    relationships: List[Relationship] = Field(default_factory=list, description="Character relationships")
    tags: List[str] = Field(default_factory=list, description="Character tags (genres, species, etc.)")
    imports: Dict[str, Any] = Field(default_factory=dict, description="Import metadata (source, etc.)")
    
    model_config = {
        "json_encoders": {
            # Ensure proper JSON serialization
        }
    }


async def llm_estimate_big5(description: str, mes_example: str = "") -> Personality:
    """
    Use LLM to estimate Big Five personality traits from character description and examples
    
    Args:
        description: Character description text
        mes_example: Example messages/dialogue
        
    Returns:
        Personality object with estimated Big Five scores
    """
    from ..openai_client import get_client
    
    prompt = f"""Analyze the following character information and estimate their Big Five personality traits on a scale of 0.0 to 1.0.

Character Description:
{description}

Example Dialogue:
{mes_example}

Please provide scores for each Big Five trait:

1. **Openness to Experience** (0.0 = conventional, traditional vs 1.0 = creative, curious, open to new ideas)
2. **Conscientiousness** (0.0 = disorganized, impulsive vs 1.0 = organized, disciplined, goal-oriented)  
3. **Extraversion** (0.0 = introverted, reserved vs 1.0 = extraverted, sociable, energetic)
4. **Agreeableness** (0.0 = competitive, skeptical vs 1.0 = cooperative, trusting, helpful)
5. **Neuroticism** (0.0 = calm, emotionally stable vs 1.0 = anxious, emotionally reactive)

Respond with ONLY a JSON object in this exact format:
{{
    "openness": 0.X,
    "conscientiousness": 0.X,
    "extraversion": 0.X,
    "agreeableness": 0.X,
    "neuroticism": 0.X
}}"""

    try:
        client = get_client()
        response = await client.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,  # Low temperature for consistent scoring
            stop=["}"]
        )
        
        # Add closing brace if truncated
        response_text = response.strip()
        if not response_text.endswith("}"):
            response_text += "}"
            
        # Parse JSON response
        scores = json.loads(response_text)
        
        # Validate and clamp scores to 0-1 range
        for key in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            if key in scores:
                scores[key] = max(0.0, min(1.0, float(scores[key])))
            else:
                scores[key] = 0.5  # Default fallback
        
        return Personality(**scores)
        
    except Exception as e:
        logger.warning(f"Failed to estimate Big Five traits via LLM: {e}")
        # Return default personality on failure
        return Personality() 