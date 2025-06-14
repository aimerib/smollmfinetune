"""
Enhanced Character Consistency Judge System
Specialized LLM judge for character-specific quality evaluation and progressive refinement.
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConsistencyAspect(Enum):
    """Aspects of character consistency to evaluate"""
    PERSONALITY = "personality"
    SPEECH_PATTERNS = "speech_patterns" 
    KNOWLEDGE = "knowledge"
    RELATIONSHIPS = "relationships"
    EMOTIONAL_STATE = "emotional_state"
    BEHAVIOR = "behavior"
    PHYSICAL_ACTIONS = "physical_actions"
    MORAL_ALIGNMENT = "moral_alignment"


@dataclass
class CharacterProfile:
    """Structured character profile for consistent evaluation"""
    name: str
    personality_traits: List[str]
    speech_patterns: List[str]
    knowledge_domains: List[str]
    relationship_style: str
    emotional_tendencies: List[str]
    behavioral_patterns: List[str]
    moral_compass: str
    background_elements: List[str]
    
    def to_evaluation_context(self) -> str:
        """Convert to context string for judge LLM"""
        return f"""
CHARACTER PROFILE: {self.name}

PERSONALITY TRAITS: {', '.join(self.personality_traits)}
SPEECH PATTERNS: {', '.join(self.speech_patterns)}
KNOWLEDGE DOMAINS: {', '.join(self.knowledge_domains)}
RELATIONSHIP STYLE: {self.relationship_style}
EMOTIONAL TENDENCIES: {', '.join(self.emotional_tendencies)}
BEHAVIORAL PATTERNS: {', '.join(self.behavioral_patterns)}
MORAL COMPASS: {self.moral_compass}
BACKGROUND: {'; '.join(self.background_elements)}
"""


@dataclass
class ConsistencyScore:
    """Detailed consistency scoring"""
    aspect: ConsistencyAspect
    score: float  # 0.0 to 1.0
    confidence: float  # Judge confidence in the score
    explanation: str
    evidence: List[str]  # Specific text evidence
    suggestions: List[str]  # Improvement suggestions


@dataclass
class JudgmentResult:
    """Complete judgment result for a sample"""
    sample_id: str
    overall_score: float
    aspect_scores: Dict[ConsistencyAspect, ConsistencyScore]
    quality_issues: List[str]
    strengths: List[str] 
    recommended_action: str  # "accept", "revise", "reject"
    revision_suggestions: List[str]
    character_voice_score: float
    narrative_coherence: float


class CharacterJudge:
    """Specialized character consistency judge with progressive refinement"""
    
    def __init__(self, client, character_profile: CharacterProfile):
        self.client = client
        self.character_profile = character_profile
        
        # Evaluation configuration
        self.evaluation_config = {
            'min_acceptance_score': 0.7,
            'min_voice_score': 0.6,
            'min_coherence_score': 0.6,
            'confidence_threshold': 0.8,
            'enable_progressive_refinement': True,
            'max_refinement_iterations': 3
        }
        
        # Aspect weights for overall scoring
        self.aspect_weights = {
            ConsistencyAspect.PERSONALITY: 0.25,
            ConsistencyAspect.SPEECH_PATTERNS: 0.15,
            ConsistencyAspect.KNOWLEDGE: 0.15,
            ConsistencyAspect.RELATIONSHIPS: 0.10,
            ConsistencyAspect.EMOTIONAL_STATE: 0.10,
            ConsistencyAspect.BEHAVIOR: 0.15,
            ConsistencyAspect.PHYSICAL_ACTIONS: 0.05,
            ConsistencyAspect.MORAL_ALIGNMENT: 0.05
        }
        
        # Performance tracking
        self.judgment_stats = {
            'total_judgments': 0,
            'acceptance_rate': 0.0,
            'avg_scores_by_aspect': defaultdict(list),
            'common_issues': defaultdict(int),
            'refinement_success_rate': 0.0
        }
    
    async def judge_sample(self, sample: Dict[str, Any], 
                          context: Optional[Dict[str, Any]] = None) -> JudgmentResult:
        """Judge a single sample for character consistency"""
        
        sample_id = context.get('sample_id', f"sample_{id(sample)}")
        
        # Extract conversation components
        messages = sample.get('messages', [])
        if len(messages) < 3:
            return self._create_rejection_result(sample_id, "Insufficient messages")
        
        system_msg = messages[0].get('content', '') if messages[0].get('role') == 'system' else ''
        user_msg = messages[1].get('content', '')
        assistant_msg = messages[2].get('content', '')
        
        # Evaluate each aspect
        aspect_scores = {}
        for aspect in ConsistencyAspect:
            score = await self._evaluate_aspect(aspect, user_msg, assistant_msg, system_msg, context)
            aspect_scores[aspect] = score
        
        # Calculate overall scores
        overall_score = self._calculate_overall_score(aspect_scores)
        voice_score = self._calculate_voice_score(aspect_scores)
        coherence_score = self._calculate_coherence_score(aspect_scores)
        
        # Analyze quality issues and strengths
        quality_issues = self._identify_quality_issues(aspect_scores, assistant_msg)
        strengths = self._identify_strengths(aspect_scores, assistant_msg)
        
        # Determine recommendation
        recommended_action = self._determine_action(overall_score, voice_score, coherence_score, aspect_scores)
        revision_suggestions = self._generate_revision_suggestions(aspect_scores, quality_issues)
        
        # Update statistics
        self._update_judgment_stats(overall_score, aspect_scores, recommended_action)
        
        return JudgmentResult(
            sample_id=sample_id,
            overall_score=overall_score,
            aspect_scores=aspect_scores,
            quality_issues=quality_issues,
            strengths=strengths,
            recommended_action=recommended_action,
            revision_suggestions=revision_suggestions,
            character_voice_score=voice_score,
            narrative_coherence=coherence_score
        )
    
    async def _evaluate_aspect(self, aspect: ConsistencyAspect, 
                              user_msg: str, assistant_msg: str, system_msg: str,
                              context: Optional[Dict[str, Any]]) -> ConsistencyScore:
        """Evaluate a specific aspect of character consistency"""
        
        # Build aspect-specific evaluation prompt
        evaluation_prompt = self._build_aspect_prompt(aspect, user_msg, assistant_msg, system_msg)
        
        try:
            # Use chat completion for structured evaluation
            messages = [
                {"role": "system", "content": self._get_judge_system_prompt(aspect)},
                {"role": "user", "content": evaluation_prompt}
            ]
            
            response = await self.client.chat_complete(
                messages=messages,
                max_tokens=800,
                temperature=0.1,  # Low temperature for consistent judging
                top_p=0.95
            )
            
            # Parse structured response
            return self._parse_aspect_evaluation(response, aspect)
            
        except Exception as e:
            logger.error(f"Error evaluating aspect {aspect}: {e}")
            return ConsistencyScore(
                aspect=aspect,
                score=0.5,
                confidence=0.0,
                explanation=f"Evaluation failed: {str(e)}",
                evidence=[],
                suggestions=[]
            )
    
    def _build_aspect_prompt(self, aspect: ConsistencyAspect,
                           user_msg: str, assistant_msg: str, system_msg: str) -> str:
        """Build aspect-specific evaluation prompt"""
        
        character_context = self.character_profile.to_evaluation_context()
        
        aspect_prompts = {
            ConsistencyAspect.PERSONALITY: f"""
Evaluate how well this response reflects {self.character_profile.name}'s established personality traits.

USER: {user_msg}
RESPONSE: {assistant_msg}

Focus on:
- Does the response show the expected personality traits?
- Are the emotional reactions consistent with the character?
- Does the tone match the character's established personality?
""",
            
            ConsistencyAspect.SPEECH_PATTERNS: f"""
Evaluate the speech patterns and language style in this response.

USER: {user_msg}
RESPONSE: {assistant_msg}

Focus on:
- Does the character speak in their established style?
- Are mannerisms and verbal tics present?
- Is the vocabulary and syntax appropriate for the character?
""",
            
            ConsistencyAspect.KNOWLEDGE: f"""
Evaluate whether the character demonstrates appropriate knowledge and expertise.

USER: {user_msg}
RESPONSE: {assistant_msg}

Focus on:
- Does the character show knowledge they should have?
- Do they avoid knowing things they shouldn't?
- Are their opinions and perspectives consistent with their background?
""",
            
            ConsistencyAspect.BEHAVIOR: f"""
Evaluate the behavioral consistency of the character's actions and choices.

USER: {user_msg}
RESPONSE: {assistant_msg}

Focus on:
- Are the character's actions consistent with their established behavior patterns?
- Do their choices align with their motivations and goals?
- Are their reactions proportionate and in-character?
"""
        }
        
        base_prompt = aspect_prompts.get(aspect, f"Evaluate the {aspect.value} consistency of this response.")
        
        return f"""
{character_context}

{base_prompt}

Provide your evaluation in this JSON format:
{{
    "score": 0.85,
    "confidence": 0.9,
    "explanation": "The response effectively demonstrates...",
    "evidence": ["Specific quote showing consistency", "Another example"],
    "suggestions": ["How to improve if needed"]
}}
"""
    
    def _get_judge_system_prompt(self, aspect: ConsistencyAspect) -> str:
        """Get system prompt for the judge LLM"""
        return f"""You are a specialized character consistency judge with expertise in evaluating AI roleplay responses. Your task is to assess how well a character's response maintains consistency with their established profile.

EVALUATION CRITERIA for {aspect.value}:
- Score from 0.0 (completely inconsistent) to 1.0 (perfectly consistent)
- Confidence from 0.0 (uncertain) to 1.0 (very confident)
- Provide specific evidence from the text
- Give actionable suggestions for improvement

SCORING GUIDELINES:
- 0.9-1.0: Excellent consistency, character is perfectly on-brand
- 0.7-0.8: Good consistency with minor issues
- 0.5-0.6: Moderate consistency, noticeable issues
- 0.3-0.4: Poor consistency, significant problems
- 0.0-0.2: Very poor, character is unrecognizable

Be thorough but concise. Focus on specific, actionable feedback."""
    
    def _parse_aspect_evaluation(self, response: str, aspect: ConsistencyAspect) -> ConsistencyScore:
        """Parse the structured evaluation response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())
                
                return ConsistencyScore(
                    aspect=aspect,
                    score=float(eval_data.get('score', 0.5)),
                    confidence=float(eval_data.get('confidence', 0.5)),
                    explanation=eval_data.get('explanation', ''),
                    evidence=eval_data.get('evidence', []),
                    suggestions=eval_data.get('suggestions', [])
                )
            else:
                # Fallback parsing
                return self._fallback_parse_evaluation(response, aspect)
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Failed to parse evaluation JSON for {aspect}: {e}")
            return self._fallback_parse_evaluation(response, aspect)
    
    def _fallback_parse_evaluation(self, response: str, aspect: ConsistencyAspect) -> ConsistencyScore:
        """Fallback evaluation parsing when JSON fails"""
        # Simple pattern matching for score
        score_match = re.search(r'score[:\s]*([0-9.]+)', response, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.5
        
        # Extract confidence if mentioned
        conf_match = re.search(r'confidence[:\s]*([0-9.]+)', response, re.IGNORECASE)
        confidence = float(conf_match.group(1)) if conf_match else 0.5
        
        return ConsistencyScore(
            aspect=aspect,
            score=min(1.0, max(0.0, score)),
            confidence=min(1.0, max(0.0, confidence)),
            explanation=response[:200] + "..." if len(response) > 200 else response,
            evidence=[],
            suggestions=[]
        )
    
    def _calculate_overall_score(self, aspect_scores: Dict[ConsistencyAspect, ConsistencyScore]) -> float:
        """Calculate weighted overall consistency score"""
        total_score = 0.0
        total_weight = 0.0
        
        for aspect, score_obj in aspect_scores.items():
            weight = self.aspect_weights.get(aspect, 0.1)
            # Weight by confidence to prioritize reliable scores
            adjusted_weight = weight * score_obj.confidence
            total_score += score_obj.score * adjusted_weight
            total_weight += adjusted_weight
        
        return total_score / max(total_weight, 0.1)
    
    def _calculate_voice_score(self, aspect_scores: Dict[ConsistencyAspect, ConsistencyScore]) -> float:
        """Calculate character voice consistency score"""
        voice_aspects = [
            ConsistencyAspect.PERSONALITY,
            ConsistencyAspect.SPEECH_PATTERNS,
            ConsistencyAspect.EMOTIONAL_STATE
        ]
        
        scores = [aspect_scores[aspect].score for aspect in voice_aspects if aspect in aspect_scores]
        return np.mean(scores) if scores else 0.5
    
    def _calculate_coherence_score(self, aspect_scores: Dict[ConsistencyAspect, ConsistencyScore]) -> float:
        """Calculate narrative coherence score"""
        coherence_aspects = [
            ConsistencyAspect.KNOWLEDGE,
            ConsistencyAspect.BEHAVIOR,
            ConsistencyAspect.RELATIONSHIPS
        ]
        
        scores = [aspect_scores[aspect].score for aspect in coherence_aspects if aspect in aspect_scores]
        return np.mean(scores) if scores else 0.5
    
    def _identify_quality_issues(self, aspect_scores: Dict[ConsistencyAspect, ConsistencyScore], 
                               assistant_msg: str) -> List[str]:
        """Identify specific quality issues"""
        issues = []
        
        # Low scoring aspects
        for aspect, score_obj in aspect_scores.items():
            if score_obj.score < 0.6:
                issues.append(f"Low {aspect.value} consistency ({score_obj.score:.2f}): {score_obj.explanation}")
        
        # Character name issues
        char_name = self.character_profile.name.lower()
        if char_name in assistant_msg.lower():
            # Check for third-person self-reference
            if not any(quote in assistant_msg for quote in ['"', "'", "*says*", "*thinks*"]):
                issues.append("Character refers to themselves in third person")
        
        # Length issues
        if len(assistant_msg.split()) < 8:
            issues.append("Response too short for meaningful character development")
        elif len(assistant_msg.split()) > 300:
            issues.append("Response may be excessively long")
        
        return issues
    
    def _identify_strengths(self, aspect_scores: Dict[ConsistencyAspect, ConsistencyScore],
                          assistant_msg: str) -> List[str]:
        """Identify response strengths"""
        strengths = []
        
        # High scoring aspects
        for aspect, score_obj in aspect_scores.items():
            if score_obj.score >= 0.8:
                strengths.append(f"Strong {aspect.value} consistency ({score_obj.score:.2f})")
        
        # Specific positive indicators
        if '*' in assistant_msg:
            strengths.append("Good use of action formatting")
        
        if any(emotion in assistant_msg.lower() for emotion in ['feel', 'emotion', 'heart']):
            strengths.append("Shows emotional depth")
        
        return strengths
    
    def _determine_action(self, overall_score: float, voice_score: float, 
                         coherence_score: float, aspect_scores: Dict) -> str:
        """Determine recommended action for the sample"""
        
        min_acceptance = self.evaluation_config['min_acceptance_score']
        min_voice = self.evaluation_config['min_voice_score']
        min_coherence = self.evaluation_config['min_coherence_score']
        
        # Check for critical failures
        critical_failures = sum(1 for score_obj in aspect_scores.values() if score_obj.score < 0.3)
        if critical_failures > 1:
            return "reject"
        
        # Check acceptance criteria
        if (overall_score >= min_acceptance and 
            voice_score >= min_voice and 
            coherence_score >= min_coherence):
            return "accept"
        
        # Check if revision might help
        if overall_score >= 0.5 and voice_score >= 0.4:
            return "revise"
        
        return "reject"
    
    def _generate_revision_suggestions(self, aspect_scores: Dict[ConsistencyAspect, ConsistencyScore],
                                     quality_issues: List[str]) -> List[str]:
        """Generate specific revision suggestions"""
        suggestions = []
        
        # Collect suggestions from low-scoring aspects
        for aspect, score_obj in aspect_scores.items():
            if score_obj.score < 0.7 and score_obj.suggestions:
                suggestions.extend(score_obj.suggestions)
        
        # Add general suggestions based on issues
        if any("personality" in issue.lower() for issue in quality_issues):
            suggestions.append("Strengthen personality expression with character-specific reactions")
        
        if any("speech" in issue.lower() for issue in quality_issues):
            suggestions.append("Improve speech patterns to match character's established style")
        
        if any("short" in issue.lower() for issue in quality_issues):
            suggestions.append("Expand response with more character-specific details")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _create_rejection_result(self, sample_id: str, reason: str) -> JudgmentResult:
        """Create a rejection result for invalid samples"""
        return JudgmentResult(
            sample_id=sample_id,
            overall_score=0.0,
            aspect_scores={},
            quality_issues=[reason],
            strengths=[],
            recommended_action="reject",
            revision_suggestions=[],
            character_voice_score=0.0,
            narrative_coherence=0.0
        )
    
    def _update_judgment_stats(self, overall_score: float, 
                              aspect_scores: Dict[ConsistencyAspect, ConsistencyScore],
                              action: str):
        """Update judgment statistics"""
        self.judgment_stats['total_judgments'] += 1
        
        # Update acceptance rate
        if action == "accept":
            acceptance_count = (self.judgment_stats['acceptance_rate'] * 
                              (self.judgment_stats['total_judgments'] - 1)) + 1
        else:
            acceptance_count = (self.judgment_stats['acceptance_rate'] * 
                              (self.judgment_stats['total_judgments'] - 1))
        
        self.judgment_stats['acceptance_rate'] = acceptance_count / self.judgment_stats['total_judgments']
        
        # Update aspect averages
        for aspect, score_obj in aspect_scores.items():
            self.judgment_stats['avg_scores_by_aspect'][aspect].append(score_obj.score)
            # Keep only recent history
            if len(self.judgment_stats['avg_scores_by_aspect'][aspect]) > 1000:
                self.judgment_stats['avg_scores_by_aspect'][aspect] = \
                    self.judgment_stats['avg_scores_by_aspect'][aspect][-1000:]
    
    async def judge_batch(self, samples: List[Dict[str, Any]], 
                         batch_size: int = 10) -> List[JudgmentResult]:
        """Judge a batch of samples efficiently"""
        results = []
        
        # Process in smaller batches to manage memory and API limits
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            
            # Create judgment tasks
            tasks = [
                self.judge_sample(sample, {'sample_id': f'batch_{i//batch_size}_sample_{j}'})
                for j, sample in enumerate(batch)
            ]
            
            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error judging sample {i+j}: {result}")
                    results.append(self._create_rejection_result(f'batch_{i//batch_size}_sample_{j}', 
                                                               f"Judgment error: {str(result)}"))
                else:
                    results.append(result)
        
        return results
    
    def get_judgment_statistics(self) -> Dict[str, Any]:
        """Get current judgment statistics"""
        avg_aspect_scores = {}
        for aspect, scores in self.judgment_stats['avg_scores_by_aspect'].items():
            if scores:
                avg_aspect_scores[aspect.value] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'count': len(scores)
                }
        
        return {
            'total_judgments': self.judgment_stats['total_judgments'],
            'acceptance_rate': self.judgment_stats['acceptance_rate'],
            'avg_aspect_scores': avg_aspect_scores,
            'refinement_success_rate': self.judgment_stats['refinement_success_rate']
        }


def create_character_profile_from_card(character_card: Dict[str, Any]) -> CharacterProfile:
    """Create a structured character profile from a character card"""
    
    name = character_card.get('name', 'Unknown')
    
    # Extract personality traits
    personality_text = character_card.get('personality', '')
    personality_traits = [trait.strip() for trait in personality_text.split(',') if trait.strip()]
    
    # Analyze description for additional traits
    description = character_card.get('description', '')
    
    # Simple trait extraction (in production, use more sophisticated NLP)
    trait_keywords = ['shy', 'confident', 'aggressive', 'kind', 'intelligent', 'playful', 'serious']
    for keyword in trait_keywords:
        if keyword in description.lower() and keyword not in [t.lower() for t in personality_traits]:
            personality_traits.append(keyword.title())
    
    # Extract speech patterns from examples
    mes_example = character_card.get('mes_example', '')
    speech_patterns = []
    if '*' in mes_example:
        speech_patterns.append("Uses action formatting")
    if '...' in mes_example:
        speech_patterns.append("Trailing speech")
    if '!' in mes_example:
        speech_patterns.append("Exclamatory")
    
    # Extract knowledge domains (simplified)
    knowledge_domains = []
    scenario = character_card.get('scenario', '').lower()
    if 'magic' in description.lower() or 'magic' in scenario:
        knowledge_domains.append("Magic")
    if 'business' in description.lower() or 'business' in scenario:
        knowledge_domains.append("Business")
    if 'academy' in description.lower() or 'school' in description.lower():
        knowledge_domains.append("Academia")
    
    # Default fallbacks
    if not personality_traits:
        personality_traits = ["Friendly"]
    if not speech_patterns:
        speech_patterns = ["Natural conversation"]
    if not knowledge_domains:
        knowledge_domains = ["General"]
    
    return CharacterProfile(
        name=name,
        personality_traits=personality_traits,
        speech_patterns=speech_patterns,
        knowledge_domains=knowledge_domains,
        relationship_style="Adaptive",  # Default
        emotional_tendencies=["Responsive"],  # Default
        behavioral_patterns=["Consistent"],  # Default
        moral_compass="Neutral Good",  # Default
        background_elements=[description[:100] + "..." if len(description) > 100 else description]
    ) 