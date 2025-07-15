"""Content evaluation utilities for dataset quality assessment."""

import json
import re
from typing import Dict, List, Any, Literal
import logging

from pydantic import BaseModel

# Define CharacterAnalysis here to avoid circular imports
class CharacterAnalysis(BaseModel):
    character_consistency: float = 5.0
    emotional_authenticity: float = 5.0
    narrative_flow: float = 5.0
    creative_expression: float = 5.0
    sensual_detail: float = 5.0
    overall_score: float = 5.0

logger = logging.getLogger(__name__)


def evaluate_response_quality(response: str, character: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """Evaluate the quality of a generated response"""
    quality_metrics = {
        'length_score': 0.0,
        'character_consistency': 0.0,
        'relevance_score': 0.0,
        'diversity_score': 0.0,
        'overall_score': 0.0,
        'issues': []
    }
    
    # Length evaluation
    word_count = len(response.split())
    if word_count < 5:
        quality_metrics['length_score'] = 0.0
        quality_metrics['issues'].append("Response too short")
    elif word_count < 20:
        quality_metrics['length_score'] = 0.5
    elif word_count < 200:
        quality_metrics['length_score'] = 1.0
    else:
        quality_metrics['length_score'] = 0.8  # Slightly penalize very long responses
    
    # Character consistency check
    char_name = character.get('name', 'Assistant')
    
    # Check for character name consistency
    if char_name.lower() in response.lower():
        # Character referring to themselves in third person (usually bad)
        quality_metrics['character_consistency'] -= 0.3
        quality_metrics['issues'].append("Character refers to self in third person")
    
    # Check for prompt leakage
    if any(token in response for token in ['<|system|>', '<|user|>', '<|assistant|>', '<|endoftext|>']):
        quality_metrics['character_consistency'] = 0.0
        quality_metrics['issues'].append("Contains formatting tokens")
        
    # Check for meta-commentary
    meta_phrases = ['as an ai', 'as a language model', 'i cannot', 'i don\'t have', 'my training']
    if any(phrase in response.lower() for phrase in meta_phrases):
        quality_metrics['character_consistency'] -= 0.5
        quality_metrics['issues'].append("Contains meta-commentary")
    
    # Base character consistency score
    if not quality_metrics['issues']:
        quality_metrics['character_consistency'] = 1.0
    else:
        quality_metrics['character_consistency'] = max(0.0, quality_metrics['character_consistency'] + 1.0)
    
    # Relevance to prompt
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    
    # Check if response addresses the prompt
    common_words = prompt_words.intersection(response_words)
    if len(common_words) > 2:
        quality_metrics['relevance_score'] = min(1.0, len(common_words) / 10)
    else:
        quality_metrics['relevance_score'] = 0.3
        quality_metrics['issues'].append("Low relevance to prompt")
    
    # Diversity score (vocabulary richness)
    unique_words = len(set(response.lower().split()))
    total_words = len(response.split())
    if total_words > 0:
        quality_metrics['diversity_score'] = min(1.0, unique_words / total_words * 2)
    
    # Calculate overall score
    quality_metrics['overall_score'] = (
        quality_metrics['length_score'] * 0.2 +
        quality_metrics['character_consistency'] * 0.4 +
        quality_metrics['relevance_score'] * 0.2 +
        quality_metrics['diversity_score'] * 0.2
    )
    
    return quality_metrics

class NsfwResponse(BaseModel):
    is_nsfw: bool


async def is_nsfw_content(client, content: str) -> bool:
    """Check if a sample contains NSFW/intimate content"""

    if not content:
        return False

    prompt = f"""
    Analyze the following content for NSFW/intimate content:
    {content}

    Respond with ONLY a JSON object with a single boolean value:
    {{"is_nsfw": true}}
    """

    try:
        response = await client.generate(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.1,
            top_p=0.95,
            response_format=NsfwResponse
        )

        return response.is_nsfw
    except Exception as e:
        logger.debug(f"NSFW content check error: {e}")
        return False


class NsfwStyle(BaseModel):
    style: Literal['romantic', 'playful', 'passionate', 'emotional', 'sensual', 'hardcore', 'shy', 'dominant', 'submissive', 'kinky', 'voyeuristic', 'exhibitionistic', 'fetishistic', 'masochistic', 'sadistic']
    name: str

async def categorize_nsfw_style(client, content: str) -> str:
    """Categorize the style of NSFW content"""
    assistant_msg = content.lower()

    if not assistant_msg:
        return 'non_nsfw'

    prompt = f"""
    Categorize the style of the following content:
    {content}

    Respond with ONLY a JSON array containing objects with a single string value and the name of the character:
    [{{"style": "romantic", "name": "cricket"}}, {{"style": "playful", "name": "Sarah"}}, {{"style": "passionate", "name": "Jen"}}]

    The style should be one of the following:
    - romantic
    - playful
    - passionate
    - emotional
    - sensual
    - hardcore
    - shy
    - dominant
    - submissive
    - kinky
    - voyeuristic
    - exhibitionistic
    - fetishistic
    - masochistic
    - sadistic
    """

    try:
        response = await client.generate(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.1,
            top_p=0.95,
            response_format=[NsfwStyle]
        )

        return response
    except Exception as e:
        logger.debug(f"NSFW style categorization error: {e}")
        return [{'style': 'non_nsfw', 'name': 'non_nsfw'}]


async def evaluate_nsfw_quality(client, response: str, character: Dict[str, Any], 
                               prompt: str) -> Dict[str, float]:
    """Evaluate NSFW response quality with specialized criteria"""
    
    char_name = character.get('name', 'Assistant')
    personality = character.get('personality', '')
    description = character.get('description', '')
    
    judgment_prompt = f"""Evaluate this intimate/romantic response for quality in an adult entertainment context:

Character: {char_name}
Personality: {personality}
Description: {description}

User: {prompt}
Response: {response}

Rate on these criteria (0-10):
1. character_consistency: Does the intimate response match their established personality?
2. emotional_authenticity: Are the emotions and desires believable and well-expressed?
3. narrative_flow: Does it build tension/intimacy naturally without being abrupt?
4. consent_awareness: Is there mutual respect, enthusiasm, and clear consent?
5. creative_expression: Is it unique and character-specific rather than generic?
6. sensual_detail: Are descriptions evocative and tasteful without being crude?

Respond with ONLY a JSON object with numeric scores:
{{"character_consistency": 8, "emotional_authenticity": 9, "narrative_flow": 7, "creative_expression": 8, "sensual_detail": 7}}"""
    
    try:
        response = await client.generate(
            prompt=judgment_prompt,
            max_tokens=1000,
            temperature=0.1,
            top_p=0.95,
            response_format=CharacterAnalysis
        )
        
        # Parse the response if it's a string, otherwise return the structured response
        if isinstance(response, str):
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
            else:
                return _get_default_scores()
        else:
            # If it's already structured (Pydantic model), convert to dict
            return response.model_dump() if hasattr(response, 'model_dump') else response
        
        # # Parse the response
        # json_match = re.search(r'\{[^}]+\}', response)
        # if json_match:
        #     scores_dict = json.loads(json_match.group())
            
        #     # Calculate weighted score for NSFW content
        #     weights = {
        #         'character_consistency': 0.25,
        #         'emotional_authenticity': 0.20,
        #         'narrative_flow': 0.15,
        #         'creative_expression': 0.10,
        #         'sensual_detail': 0.10
        #     }
            
        #     overall_score = sum(
        #         scores_dict.get(key, 5) * weight 
        #         for key, weight in weights.items()
        #     )
            
        #     scores_dict['overall_score'] = overall_score
        #     return scores_dict
        # else:
        #     # Fallback scores
        #     return {
        #         'character_consistency': 5.0,
        #         'emotional_authenticity': 5.0,
        #         'narrative_flow': 5.0,
        #         'creative_expression': 5.0,
        #         'sensual_detail': 5.0,
        #         'overall_score': 5.0
        #     }
            
    except Exception as e:
        logger.debug(f"NSFW evaluation error: {e}")
        return _get_default_scores()


def _get_default_scores() -> Dict[str, float]:
    """Return default scores when evaluation fails"""
    return CharacterAnalysis().model_dump()


def analyze_temporal_distribution(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze temporal distribution of dataset samples"""
    if not dataset:
        return {}

    temporal_counts = {"past": 0, "present": 0, "future": 0, "unknown": 0}
    relationship_counts = {}
    past_relationship_samples = []
    future_relationship_samples = []

    for sample in dataset:
        # Analyze system prompt to determine temporal context
        system_content = sample.get('messages', [{}])[0].get('content', '')
        user_content = sample.get('messages', [{}])[1].get(
            'content', '') if len(sample.get('messages', [])) > 1 else ''

        # Detect temporal context from system prompt
        temporal_context = "unknown"
        if "speaking with a family member about your past" in system_content:
            temporal_context = "past"
            relationship_counts["family"] = relationship_counts.get(
                "family", 0) + 1
            past_relationship_samples.append(user_content[:100])
        elif "reminiscing with an old friend" in system_content:
            temporal_context = "past"
            relationship_counts["friend"] = relationship_counts.get(
                "friend", 0) + 1
            past_relationship_samples.append(user_content[:100])
        elif "reflecting with someone who taught" in system_content:
            temporal_context = "past"
            relationship_counts["mentor"] = relationship_counts.get(
                "mentor", 0) + 1
            past_relationship_samples.append(user_content[:100])
        elif "past romantic connection" in system_content:
            temporal_context = "past"
            relationship_counts["romance"] = relationship_counts.get(
                "romance", 0) + 1
            past_relationship_samples.append(user_content[:100])
        elif "known each other for years" in system_content:
            temporal_context = "future"
            future_relationship_samples.append(user_content[:100])
        elif "meeting the User for the first time" in system_content:
            temporal_context = "present"
        elif any(past_keyword in user_content.lower() for past_keyword in ["childhood", "growing up", "when you were young", "your father", "your mother"]):
            temporal_context = "past"
        elif any(future_keyword in user_content.lower() for future_keyword in ["after all this time", "our relationship", "years together"]):
            temporal_context = "future"
        else:
            temporal_context = "present"

        temporal_counts[temporal_context] += 1

    total_samples = sum(temporal_counts.values())
    temporal_percentages = {k: (v/total_samples)*100 if total_samples > 0 else 0
                            for k, v in temporal_counts.items()}

    return {
        'temporal_distribution': temporal_counts,
        'temporal_percentages': temporal_percentages,
        'relationship_contexts': relationship_counts,
        # First 5 examples
        'past_relationship_samples': past_relationship_samples[:5],
        # First 5 examples
        'future_relationship_samples': future_relationship_samples[:5],
        # Out of 3 temporal buckets
        'temporal_diversity_score': len([v for v in temporal_counts.values() if v > 0]) / 3 * 100
    }


def generate_prompts_from_greetings(character: Dict[str, Any], knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate prompts based on alternate greetings which show different scenarios"""
    prompts = []
    
    # Process first message
    first_mes = character.get('first_mes', '')
    if first_mes:
        # Analyze the greeting scenario
        greeting_prompts = _analyze_greeting_for_prompts(first_mes, "initial greeting")
        if greeting_prompts:
            prompts.append({
                'context': 'Initial meeting scenario',
                'prompts': greeting_prompts
            })
    
    # Process alternate greetings
    alternate_greetings = character.get('alternate_greetings', [])
    for i, greeting in enumerate(alternate_greetings[:3]):  # Process up to 3 alternates
        if greeting:
            greeting_prompts = _analyze_greeting_for_prompts(greeting, f"alternate scenario {i+1}")
            if greeting_prompts:
                # Determine context from greeting content
                context = _extract_greeting_context(greeting)
                prompts.append({
                    'context': context,
                    'prompts': greeting_prompts
                })
    
    return prompts


def _analyze_greeting_for_prompts(greeting: str, greeting_type: str) -> List[str]:
    """Analyze a greeting to generate contextual prompts"""
    prompts = []
    greeting_lower = greeting.lower()
    
    # Financial stress scenario
    if any(word in greeting_lower for word in ['money', 'gold', 'coin', 'rent', 'debt', 'pay']):
        prompts.extend([
            "How much money do you need exactly?",
            "What happened to your finances?",
            "Who are you in debt to?",
            "What's your plan to make money?"
        ])
    
    # Celebration/success scenario
    if any(word in greeting_lower for word in ['celebrate', 'success', 'did it', 'woohoo', 'toast']):
        prompts.extend([
            "What are we celebrating?",
            "How did you pull it off?",
            "What's next after this success?",
            "Who else helped make this happen?"
        ])
    
    # Danger/trouble scenario
    if any(word in greeting_lower for word in ['danger', 'trouble', 'help', 'emergency', 'problem']):
        prompts.extend([
            "What kind of trouble are you in?",
            "How can I help?",
            "Who's after you?",
            "How urgent is this?"
        ])
    
    # Business/work scenario
    if any(word in greeting_lower for word in ['job', 'work', 'contract', 'guild', 'agency', 'business']):
        prompts.extend([
            "What kind of job is it?",
            "What's the pay like?",
            "Why did you choose this line of work?",
            "Any interesting contracts lately?"
        ])
    
    # Location-specific prompts
    if 'tavern' in greeting_lower:
        prompts.extend([
            "Come here often?",
            "What's good to drink here?",
            "Know any interesting people here?"
        ])
    elif 'warehouse' in greeting_lower or 'office' in greeting_lower:
        prompts.extend([
            "How long have you had this place?",
            "Business been good?",
            "What kind of work do you do here?"
        ])
    
    # Emotional state prompts based on actions/descriptions
    if '*sigh*' in greeting or '*groan*' in greeting or 'frustrated' in greeting_lower:
        prompts.extend([
            "Rough day?",
            "What's got you so frustrated?",
            "Anything I can do to help?"
        ])
    elif '*smile*' in greeting or '*grin*' in greeting or 'excited' in greeting_lower:
        prompts.extend([
            "You seem happy!",
            "What's the good news?",
            "Share the excitement!"
        ])
    
    return prompts


def _extract_greeting_context(greeting: str) -> str:
    """Extract a descriptive context from a greeting"""
    greeting_lower = greeting.lower()
    
    # Identify the primary scenario
    if 'money' in greeting_lower or 'debt' in greeting_lower or 'rent' in greeting_lower:
        return "Financial crisis scenario"
    elif 'celebrate' in greeting_lower or 'success' in greeting_lower:
        return "Celebration scenario"
    elif 'danger' in greeting_lower or 'trouble' in greeting_lower:
        return "Danger/emergency scenario"
    elif 'tavern' in greeting_lower:
        return "Tavern meeting scenario"
    elif 'job' in greeting_lower or 'contract' in greeting_lower:
        return "Business opportunity scenario"
    elif 'visitor' in greeting_lower or 'customer' in greeting_lower:
        return "Unexpected visitor scenario"
    else:
        return "Alternative meeting scenario"