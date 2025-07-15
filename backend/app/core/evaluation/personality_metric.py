"""
Big-Five Personality Alignment Metric

This module provides functionality to evaluate how well a generated response
aligns with a character's Big-Five personality profile using an LLM as judge.
"""

import json
import logging
from typing import Dict
from functools import lru_cache

from ..openai_client import get_client

logger = logging.getLogger(__name__)


def _get_big_five_rubric() -> str:
    """Get the detailed rubric for Big-Five personality traits"""
    return """
**Detailed Big-Five Rubric for Dialogue Evaluation:**

1. **Openness (O)** - Openness to Experience
   - HIGH (0.7-1.0): Shows curiosity, creativity, imagination, abstract thinking, unconventional ideas, willingness to try new things, intellectual curiosity, artistic interests
   - MEDIUM (0.3-0.7): Balanced between conventional and creative, some interest in new ideas but also practical
   - LOW (0.0-0.3): Prefers routine, conventional thinking, practical/concrete focus, skeptical of new ideas, traditional values

2. **Conscientiousness (C)** - Organization and Dependability  
   - HIGH (0.7-1.0): Organized speech, follows through on topics, attention to detail, responsible language, planning ahead, disciplined responses
   - MEDIUM (0.3-0.7): Some organization but flexible, moderately reliable, balance of structure and spontaneity
   - LOW (0.0-0.3): Spontaneous, impulsive responses, topic-jumping, careless language, procrastination mentions, lack of planning

3. **Extraversion (E)** - Social Energy and Assertiveness
   - HIGH (0.7-1.0): Enthusiastic, talkative, energetic language, seeks interaction, bold statements, optimistic, uses exclamations
   - MEDIUM (0.3-0.7): Moderate social energy, friendly but not overly enthusiastic, balanced engagement
   - LOW (0.0-0.3): Reserved, quiet, prefers solitude, minimal enthusiasm, subdued responses, reflective rather than reactive

4. **Agreeableness (A)** - Compassion and Cooperation
   - HIGH (0.7-1.0): Warm, trusting, helpful, cooperative language, concern for others, polite, accommodating, empathetic
   - MEDIUM (0.3-0.7): Balanced between self and others, reasonably cooperative but can assert boundaries
   - LOW (0.0-0.3): Competitive, skeptical, blunt/direct, argumentative, self-focused, challenging others' views

5. **Neuroticism (N)** - Emotional Stability (reverse scored)
   - HIGH (0.7-1.0): Anxious language, emotional volatility, worry expressions, self-doubt, stress reactions, negative emotions
   - MEDIUM (0.3-0.7): Some emotional expression but generally balanced, occasional worry but manageable
   - LOW (0.0-0.3): Calm, secure, confident, emotionally stable, resilient, handles stress well, positive outlook
"""


def calculate_personality_alignment(response: str, big_five_scores: Dict[str, float]) -> float:
    """
    Calculate how well a response aligns with Big-Five personality scores.
    
    Args:
        response: The generated response to evaluate
        big_five_scores: Dictionary with keys 'openness', 'conscientiousness', 
                        'extraversion', 'agreeableness', 'neuroticism' and values 0-1
    
    Returns:
        float: Alignment score between 0.0 and 1.0
    
    Raises:
        ValueError: If response is empty or big_five_scores is invalid
    """
    # Input validation
    if not response or not response.strip():
        raise ValueError("Response cannot be empty")
    
    if not big_five_scores:
        raise ValueError("big_five_scores cannot be empty")
    
    # Validate score ranges
    for trait, score in big_five_scores.items():
        if trait not in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
            logger.warning(f"Unknown trait '{trait}' in big_five_scores")
        if not isinstance(score, (int, float)) or score < 0 or score > 1:
            raise ValueError(f"Score for {trait} must be between 0.0 and 1.0, got {score}")
    
    # Construct the system prompt
    system_prompt = """You are an expert personality psychologist specializing in the Big-Five personality model. 
Your task is to evaluate how well a dialogue response aligns with a specific Big-Five personality profile.
You must be precise and objective in your assessment."""
    
    # Construct the user prompt with Big-Five scores and detailed rubric
    user_prompt = f"""Please evaluate the following response based on the provided Big-Five personality profile.

{_get_big_five_rubric()}

**Target Big-Five Profile for this character:**
- Openness: {big_five_scores.get('openness', 0.5):.1f}
- Conscientiousness: {big_five_scores.get('conscientiousness', 0.5):.1f}
- Extraversion: {big_five_scores.get('extraversion', 0.5):.1f}
- Agreeableness: {big_five_scores.get('agreeableness', 0.5):.1f}
- Neuroticism: {big_five_scores.get('neuroticism', 0.5):.1f}

**Response to evaluate:**
"{response}"

Analyze how well this response demonstrates the target personality profile. Consider:
1. Does the language style match the extraversion level?
2. Does the content reflect the appropriate openness to experience?
3. Is the organization/structure consistent with the conscientiousness score?
4. Does the tone align with the agreeableness level?
5. Are emotional expressions consistent with the neuroticism score?

Provide a single alignment score from 0.0 (completely misaligned) to 1.0 (perfectly aligned).
Return ONLY a JSON object with your score.

Example:
{{"alignment_score": 0.85}}"""
    
    try:
        # Get the OpenAI client
        client = get_client()
        
        # Make the API call with caching for identical inputs
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Using a lightweight model for speed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent scoring
            max_tokens=100,
            seed=42  # For more reproducible results
        )
        
        # Parse the response
        result_text = completion.choices[0].message.content.strip()
        logger.debug(f"LLM response: {result_text}")
        
        try:
            result_json = json.loads(result_text)
            alignment_score = result_json.get('alignment_score', 0.5)
            
            # Ensure score is within bounds
            alignment_score = max(0.0, min(1.0, float(alignment_score)))
            
            logger.info(f"Personality alignment score: {alignment_score}")
            return alignment_score
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse alignment score from LLM response: {result_text}. Error: {e}")
            return 0.5  # Default score on parsing error
            
    except Exception as e:
        logger.error(f"Error calculating personality alignment: {e}")
        return 0.5  # Default score on API error


# Optional: Cache wrapper for expensive API calls
@lru_cache(maxsize=128)
def calculate_personality_alignment_cached(response: str, big_five_tuple: tuple) -> float:
    """
    Cached version of calculate_personality_alignment.
    
    Args:
        response: The generated response to evaluate
        big_five_tuple: Tuple of (openness, conscientiousness, extraversion, agreeableness, neuroticism)
    
    Returns:
        float: Alignment score between 0.0 and 1.0
    """
    # Convert tuple back to dict
    big_five_scores = {
        'openness': big_five_tuple[0],
        'conscientiousness': big_five_tuple[1],
        'extraversion': big_five_tuple[2],
        'agreeableness': big_five_tuple[3],
        'neuroticism': big_five_tuple[4]
    }
    return calculate_personality_alignment(response, big_five_scores) 