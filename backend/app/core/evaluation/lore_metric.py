"""
Lore Adherence Metric for Character AI Training

This module provides functionality to evaluate how well a generated response
adheres to established lore facts using an LLM as judge.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from functools import lru_cache

from ..openai_client import get_client

logger = logging.getLogger(__name__)


@dataclass
class LoreEvaluationResult:
    """Result of lore adherence evaluation"""
    average_score: float
    individual_scores: List[float]
    lore_facts_evaluated: int
    evaluation_details: List[Dict[str, Any]]


def calculate_lore_adherence(response: str, lore_fact: str) -> float:
    """
    Calculate how well a response adheres to a specific lore fact.
    
    Args:
        response: The generated response to evaluate
        lore_fact: The lore fact that should be respected
    
    Returns:
        float: Adherence score where:
               1.0 = correctly incorporated lore fact
               0.5 = ignored lore fact (neutral)
               0.0 = contradicted lore fact
    
    Raises:
        ValueError: If response or lore_fact is empty/None
    """
    # Input validation
    if not response or (response and not response.strip()):
        raise ValueError("Response cannot be empty")
    
    if not lore_fact or (lore_fact and not lore_fact.strip()):
        raise ValueError("Lore fact cannot be empty")
    
    if response is None or lore_fact is None:
        raise ValueError("Response and lore_fact cannot be None")
    
    # Construct the system prompt
    system_prompt = """You are an expert lore consistency evaluator for fictional worlds. 
Your task is to determine if a character response correctly incorporates, ignores, or contradicts a given lore fact.
You must be precise and objective in your assessment."""
    
    # Construct the evaluation prompt
    user_prompt = f"""Please evaluate how well this response handles the given lore fact.

**Lore Fact to Evaluate:**
"{lore_fact}"

**Character Response:**
"{response}"

**Scoring Guidelines:**
- Score 1.0 if the response correctly incorporates or acknowledges the lore fact
- Score 0.5 if the response ignores the lore fact entirely (neither confirms nor contradicts)
- Score 0.0 if the response contradicts or conflicts with the lore fact

Provide your evaluation as a JSON object with the score and reasoning.

Example:
{{"lore_score": 0.8, "reasoning": "Response correctly mentions the character as described in lore"}}"""
    
    try:
        # Get the OpenAI client
        client = get_client()
        
        # Make the API call
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cost-effective for evaluation
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Lower temperature for consistent scoring
            max_tokens=150,
            seed=42  # For reproducible results
        )
        
        # Parse the response
        result_text = completion.choices[0].message.content.strip()
        logger.debug(f"LLM lore evaluation response: {result_text}")
        
        try:
            result_json = json.loads(result_text)
            lore_score = result_json.get('lore_score', 0.5)
            
            # Ensure score is within bounds
            lore_score = max(0.0, min(1.0, float(lore_score)))
            
            logger.info(f"Lore adherence score: {lore_score}")
            return lore_score
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse lore score from LLM response: {result_text}. Error: {e}")
            return 0.5  # Default neutral score on parsing error
            
    except Exception as e:
        logger.error(f"Error calculating lore adherence: {e}")
        return 0.5  # Default neutral score on API error


def evaluate_multiple_lore_facts(response: str, lore_facts: List[str]) -> LoreEvaluationResult:
    """
    Evaluate a response against multiple lore facts.
    
    Args:
        response: The generated response to evaluate
        lore_facts: List of lore facts to check against
    
    Returns:
        LoreEvaluationResult: Comprehensive evaluation result
    """
    if not lore_facts:
        return LoreEvaluationResult(
            average_score=0.5,
            individual_scores=[],
            lore_facts_evaluated=0,
            evaluation_details=[]
        )
    
    individual_scores = []
    evaluation_details = []
    
    for fact in lore_facts:
        try:
            score = calculate_lore_adherence(response, fact)
            individual_scores.append(score)
            
            # Store evaluation details
            evaluation_details.append({
                "fact": fact,
                "score": score,
                "reasoning": f"Evaluated response adherence to: {fact[:50]}..."
            })
            
        except Exception as e:
            logger.warning(f"Failed to evaluate lore fact '{fact}': {e}")
            individual_scores.append(0.5)  # Default neutral score on error
            evaluation_details.append({
                "fact": fact,
                "score": 0.5,
                "reasoning": f"Evaluation failed: {str(e)}"
            })
    
    # Calculate average score
    average_score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.5
    
    return LoreEvaluationResult(
        average_score=average_score,
        individual_scores=individual_scores,
        lore_facts_evaluated=len(lore_facts),
        evaluation_details=evaluation_details
    )


@lru_cache(maxsize=128)
def calculate_lore_adherence_cached(response: str, lore_fact: str) -> float:
    """
    Cached version of calculate_lore_adherence for performance optimization.
    
    Args:
        response: The generated response to evaluate
        lore_fact: The lore fact that should be respected
    
    Returns:
        float: Adherence score between 0.0 and 1.0
    """
    return calculate_lore_adherence(response, lore_fact)


def get_world_lore_facts(world_context: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Extract lore facts from world context for evaluation.
    
    Args:
        world_context: World context dictionary containing lore information
    
    Returns:
        List[str]: List of lore facts to evaluate against
    """
    if not world_context:
        return []
    
    lore_facts = []
    
    # Extract from different world context structures
    if 'lore' in world_context:
        lore_data = world_context['lore']
        if isinstance(lore_data, list):
            lore_facts.extend(lore_data)
        elif isinstance(lore_data, dict):
            # Extract from various lore categories
            for category, facts in lore_data.items():
                if isinstance(facts, list):
                    lore_facts.extend(facts)
                elif isinstance(facts, str):
                    lore_facts.append(facts)
    
    # Extract from world rules/facts
    if 'world_rules' in world_context:
        rules = world_context['world_rules']
        if isinstance(rules, list):
            lore_facts.extend(rules)
        elif isinstance(rules, str):
            lore_facts.append(rules)
    
    # Extract from character world integration
    if 'world_facts' in world_context:
        facts = world_context['world_facts']
        if isinstance(facts, list):
            lore_facts.extend(facts)
    
    return lore_facts


def evaluate_response_lore_adherence(
    response: str, 
    character_context: Optional[Dict[str, Any]] = None,
    world_context: Optional[Dict[str, Any]] = None
) -> LoreEvaluationResult:
    """
    Comprehensive lore adherence evaluation for a response.
    
    Args:
        response: The generated response to evaluate
        character_context: Character-specific context
        world_context: World context containing lore facts
    
    Returns:
        LoreEvaluationResult: Complete evaluation result
    """
    # Gather all relevant lore facts
    lore_facts = []
    
    # Get world lore facts
    if world_context:
        lore_facts.extend(get_world_lore_facts(world_context))
    
    # Get character-specific lore facts
    if character_context and 'lore_facts' in character_context:
        char_lore = character_context['lore_facts']
        if isinstance(char_lore, list):
            lore_facts.extend(char_lore)
        elif isinstance(char_lore, str):
            lore_facts.append(char_lore)
    
    # If no lore facts available, return neutral evaluation
    if not lore_facts:
        logger.info("No lore facts available for evaluation")
        return LoreEvaluationResult(
            average_score=0.5,
            individual_scores=[],
            lore_facts_evaluated=0,
            evaluation_details=[]
        )
    
    # Limit to most important lore facts to avoid excessive API calls
    max_facts = 5
    if len(lore_facts) > max_facts:
        lore_facts = lore_facts[:max_facts]
        logger.info(f"Limited lore evaluation to {max_facts} most important facts")
    
    return evaluate_multiple_lore_facts(response, lore_facts) 