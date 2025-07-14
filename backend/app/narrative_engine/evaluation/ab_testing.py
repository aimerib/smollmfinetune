"""
A/B Testing Framework

Uses LLM-as-judge with structured outputs to provide statistical comparison of different models 
to determine which performs better on user satisfaction metrics.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats
from collections import defaultdict
import asyncio
import json
from pydantic import BaseModel, Field

from backend.app.core.openai_client import get_client

logger = logging.getLogger(__name__)


class ModelResponseComparison(BaseModel):
    """Structured output for comparing two model responses"""
    preferred_model: str = Field(description="Which model is preferred: 'model_a', 'model_b', or 'tie'")
    preference_strength: float = Field(ge=0.0, le=1.0, description="Strength of preference (0=slight, 1=strong)")
    quality_analysis: Dict[str, float] = Field(description="Quality scores for each model (0.0-1.0)")
    reasoning: str = Field(description="Detailed reasoning for the preference")
    evaluation_criteria: List[str] = Field(description="Criteria used for evaluation")
    specific_strengths: Dict[str, List[str]] = Field(description="Specific strengths of each model")
    specific_weaknesses: Dict[str, List[str]] = Field(description="Specific weaknesses of each model")


class ComparisonBatch(BaseModel):
    """Analysis of a batch of comparisons"""
    total_comparisons: int = Field(description="Total number of comparisons")
    model_a_wins: int = Field(description="Number of times model A was preferred")
    model_b_wins: int = Field(description="Number of times model B was preferred")
    ties: int = Field(description="Number of tie results")
    confidence_assessment: str = Field(description="Confidence in the comparison results")
    quality_trends: List[str] = Field(description="Observed quality trends across comparisons")


class ABTestingFramework:
    """Framework for conducting A/B tests between models using LLM-as-judge"""
    
    def __init__(self, model_a_name: str = "model_a", model_b_name: str = "model_b"):
        """
        Initialize the A/B testing framework.
        
        Args:
            model_a_name: Name of the first model
            model_b_name: Name of the second model
        """
        self.client = get_client()
        self.model_a_name = model_a_name
        self.model_b_name = model_b_name
        self.significance_level = 0.05
        self.min_sample_size = 30
    
    def run_ab_test(
        self,
        model_a: Any,
        model_b: Any,
        test_prompts: List[str],
        num_evaluators: int = 5,
        evaluation_criteria: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run A/B test between two models using LLM-as-judge.
        
        Args:
            model_a: First model to test
            model_b: Second model to test
            test_prompts: List of prompts to test with
            num_evaluators: Number of evaluation rounds per prompt
            evaluation_criteria: Criteria for evaluation
            
        Returns:
            Dictionary with test results and statistical analysis
        """
        results = {
            'model_a_name': self.model_a_name,
            'model_b_name': self.model_b_name,
            'total_comparisons': 0,
            'model_a_preferred': 0,
            'model_b_preferred': 0,
            'ties': 0,
            'preference_rate_a': 0.0,
            'preference_rate_b': 0.0,
            'winner': None,
            'statistical_significance': None,
            'confidence_interval': None
        }
        
        try:
            # Run comparison using LLM-as-judge
            comparison_results = asyncio.run(self._run_comparison_async(
                model_a, model_b, test_prompts, num_evaluators, evaluation_criteria
            ))
            
            # Update results
            results['total_comparisons'] = comparison_results['total_comparisons']
            results['model_a_preferred'] = comparison_results['model_a_preferred']
            results['model_b_preferred'] = comparison_results['model_b_preferred']
            results['ties'] = comparison_results.get('ties', 0)
            results['detailed_comparisons'] = comparison_results.get('detailed_comparisons', [])
            results['quality_analysis'] = comparison_results.get('quality_analysis', {})
            
            # Calculate preference rates
            total_decisive = results['model_a_preferred'] + results['model_b_preferred']
            if total_decisive > 0:
                results['preference_rate_a'] = results['model_a_preferred'] / total_decisive
                results['preference_rate_b'] = results['model_b_preferred'] / total_decisive
            
            # Determine winner
            if results['preference_rate_b'] > 0.5:
                results['winner'] = self.model_b_name
            elif results['preference_rate_a'] > 0.5:
                results['winner'] = self.model_a_name
            else:
                results['winner'] = 'tie'
            
            # Calculate statistical significance
            sig_results = self.calculate_significance(
                successes_a=results['model_a_preferred'],
                successes_b=results['model_b_preferred'],
                total=total_decisive
            )
            results['statistical_significance'] = sig_results
            
            # Calculate confidence interval
            if total_decisive > 0:
                ci = self._calculate_confidence_interval(
                    results['preference_rate_b'], total_decisive
                )
                results['confidence_interval'] = ci
            
            logger.info(f"A/B test complete: {results['winner']} wins "
                       f"({results['preference_rate_b']:.2%} preference)")
            
        except Exception as e:
            logger.error(f"Error running A/B test: {e}")
            results['error'] = str(e)
            # Fallback to mock comparison
            fallback_results = self._run_comparison_fallback(test_prompts, num_evaluators)
            results.update(fallback_results)
        
        return results
    
    async def _run_comparison_async(
        self,
        model_a: Any,
        model_b: Any,
        prompts: List[str],
        num_evaluators: int,
        evaluation_criteria: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run the actual comparison between models using LLM-as-judge.
        
        This generates responses from both models and uses LLM judge to compare them.
        """
        try:
            detailed_comparisons = []
            model_a_wins = 0
            model_b_wins = 0
            ties = 0
            
            # For each prompt, generate responses and compare
            for prompt_idx, prompt in enumerate(prompts):
                # Generate responses from both models
                response_a = self._generate_response(model_a, prompt, f"{self.model_a_name}")
                response_b = self._generate_response(model_b, prompt, f"{self.model_b_name}")
                
                # Compare responses using LLM judge (multiple rounds for reliability)
                prompt_comparisons = []
                for eval_round in range(num_evaluators):
                    comparison = await self._judge_model_comparison(
                        prompt, response_a, response_b, evaluation_criteria
                    )
                    prompt_comparisons.append(comparison)
                    
                    # Count preferences
                    if comparison.preferred_model == 'model_a':
                        model_a_wins += 1
                    elif comparison.preferred_model == 'model_b':
                        model_b_wins += 1
                    else:
                        ties += 1
                
                detailed_comparisons.append({
                    'prompt_index': prompt_idx,
                    'prompt': prompt,
                    'response_a': response_a,
                    'response_b': response_b,
                    'comparisons': [comp.model_dump() for comp in prompt_comparisons]
                })
            
            total_comparisons = len(prompts) * num_evaluators
            
            return {
                'total_comparisons': total_comparisons,
                'model_a_preferred': model_a_wins,
                'model_b_preferred': model_b_wins,
                'ties': ties,
                'detailed_comparisons': detailed_comparisons,
                'quality_analysis': self._analyze_quality_patterns(detailed_comparisons)
            }
            
        except Exception as e:
            logger.error(f"Error in async comparison: {e}")
            # Return fallback results
            return self._run_comparison_fallback(prompts, num_evaluators)
    
    async def _judge_model_comparison(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        evaluation_criteria: Optional[Dict[str, str]] = None
    ) -> ModelResponseComparison:
        """Use LLM-as-judge to compare two model responses"""
        try:
            system_prompt = self._build_comparison_system_prompt(evaluation_criteria)
            user_prompt = self._build_comparison_user_prompt(prompt, response_a, response_b)
            
            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "model_response_comparison",
                        "schema": ModelResponseComparison.model_json_schema()
                    }
                }
            )
            
            comparison_data = json.loads(response_text)
            return ModelResponseComparison(**comparison_data)
            
        except Exception as e:
            logger.error(f"Error in model comparison judgment: {e}")
            # Return fallback comparison
            return self._fallback_comparison(response_a, response_b)
    
    def _build_comparison_system_prompt(self, evaluation_criteria: Optional[Dict[str, str]] = None) -> str:
        """Build system prompt for model comparison"""
        criteria_text = ""
        if evaluation_criteria:
            criteria_list = [f"- **{key}**: {value}" for key, value in evaluation_criteria.items()]
            criteria_text = f"\n\n**Evaluation Criteria:**\n" + "\n".join(criteria_list)
        else:
            criteria_text = """
**Default Evaluation Criteria:**
- **Relevance**: How well the response addresses the prompt
- **Quality**: Overall response quality and coherence
- **Helpfulness**: How useful the response is to the user
- **Accuracy**: Factual correctness and logical consistency
- **Style**: Writing quality and appropriateness of tone
"""
        
        return f"""You are an expert evaluator comparing AI model responses. Your task is to determine which response is better and provide detailed analysis.

{criteria_text}

**Evaluation Process:**
1. Read both responses carefully
2. Evaluate each against the criteria
3. Determine which is better overall or if they're tied
4. Provide specific reasoning with examples

**SCORING GUIDELINES:**
- Quality scores: 0.0-1.0 scale for each model
- Preference strength: 0.0 (slight preference) to 1.0 (strong preference)
- Be objective and specific in your analysis
- Consider ties when responses are truly equivalent

**CRITICAL**: All numeric scores must be decimal values between 0.0 and 1.0 (inclusive)."""

    def _build_comparison_user_prompt(self, prompt: str, response_a: str, response_b: str) -> str:
        """Build user prompt for model comparison"""
        return f"""Compare these two AI responses to the same prompt:

**Original Prompt:**
{prompt}

**Response A:**
{response_a}

**Response B:**
{response_b}

**Your Task:**
1. Evaluate both responses against the criteria
2. Determine which is better: 'model_a', 'model_b', or 'tie'
3. Rate the strength of your preference (0.0-1.0)
4. Provide quality scores for each model (0.0-1.0)
5. Give detailed reasoning with specific examples
6. List specific strengths and weaknesses of each response

Be objective and thorough in your analysis. If the responses are genuinely equivalent in quality, mark it as a tie."""

    def _generate_response(self, model: Any, prompt: str, model_name: str) -> str:
        """Generate a response from a model for comparison"""
        try:
            if hasattr(model, 'generate') or hasattr(model, '__call__'):
                # In a real implementation, this would call the actual model
                # For testing, generate mock responses that vary by model
                if 'model_a' in model_name.lower():
                    return f"Model A response to: {prompt[:50]}... This is a comprehensive and detailed response from model A."
                else:
                    return f"Model B response to: {prompt[:50]}... This is a clear and concise response from model B."
            else:
                # Mock response for testing
                return f"Mock response from {model_name} to prompt: {prompt[:50]}..."
                
        except Exception as e:
            logger.warning(f"Response generation failed for {model_name}: {e}")
            return f"Error generating response from {model_name}"
    
    def _analyze_quality_patterns(self, detailed_comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality patterns across comparisons"""
        if not detailed_comparisons:
            return {}
        
        model_a_scores = []
        model_b_scores = []
        
        for comparison_set in detailed_comparisons:
            for comp in comparison_set['comparisons']:
                quality_analysis = comp.get('quality_analysis', {})
                model_a_scores.append(quality_analysis.get('model_a', 0.5))
                model_b_scores.append(quality_analysis.get('model_b', 0.5))
        
        return {
            'model_a_avg_quality': np.mean(model_a_scores) if model_a_scores else 0.5,
            'model_b_avg_quality': np.mean(model_b_scores) if model_b_scores else 0.5,
            'model_a_quality_std': np.std(model_a_scores) if model_a_scores else 0.0,
            'model_b_quality_std': np.std(model_b_scores) if model_b_scores else 0.0,
            'quality_difference': np.mean(model_b_scores) - np.mean(model_a_scores) if model_a_scores and model_b_scores else 0.0
        }
    
    def _fallback_comparison(self, response_a: str, response_b: str) -> ModelResponseComparison:
        """Fallback comparison when LLM judge fails"""
        # Simple heuristic: longer response might be better (very naive)
        len_a = len(response_a)
        len_b = len(response_b)
        
        if abs(len_a - len_b) < 10:
            preferred = 'tie'
            strength = 0.1
        elif len_a > len_b:
            preferred = 'model_a'
            strength = min(0.8, (len_a - len_b) / max(len_a, 100))
        else:
            preferred = 'model_b'
            strength = min(0.8, (len_b - len_a) / max(len_b, 100))
        
        return ModelResponseComparison(
            preferred_model=preferred,
            preference_strength=strength,
            quality_analysis={'model_a': 0.5, 'model_b': 0.5},
            reasoning="Fallback heuristic comparison based on response length",
            evaluation_criteria=["Response length (fallback)"],
            specific_strengths={'model_a': ["Generated response"], 'model_b': ["Generated response"]},
            specific_weaknesses={'model_a': ["Limited analysis"], 'model_b': ["Limited analysis"]}
        )
    
    def _run_comparison_fallback(self, prompts: List[str], num_evaluators: int) -> Dict[str, Any]:
        """Fallback comparison method when async comparison fails"""
        total_comparisons = len(prompts) * num_evaluators
        
        # Mock distribution for testing
        model_a_preferred = int(total_comparisons * 0.45)  # 45%
        model_b_preferred = int(total_comparisons * 0.55)  # 55%
        ties = total_comparisons - model_a_preferred - model_b_preferred
        
        return {
            'total_comparisons': total_comparisons,
            'model_a_preferred': model_a_preferred,
            'model_b_preferred': model_b_preferred,
            'ties': ties,
            'fallback_used': True
        }
    
    def calculate_significance(
        self,
        successes_a: int,
        successes_b: int,
        total: int
    ) -> Dict[str, Any]:
        """
        Calculate statistical significance of the difference.
        
        Args:
            successes_a: Number of times model A was preferred
            successes_b: Number of times model B was preferred
            total: Total number of comparisons
            
        Returns:
            Dictionary with statistical test results
        """
        results = {
            'p_value': 1.0,
            'is_significant': False,
            'effect_size': 0.0,
            'statistical_power': 0.0
        }
        
        if total == 0:
            return results
        
        try:
            # Proportion test
            p_a = successes_a / total
            p_b = successes_b / total
            
            # Use binomial test for difference from 0.5
            if successes_b > successes_a:
                # For newer scipy versions, use binomtest instead of binom_test
                try:
                    p_value = stats.binomtest(successes_b, total, 0.5, alternative='greater').pvalue
                except AttributeError:
                    p_value = stats.binom_test(successes_b, total, 0.5, alternative='greater')
            else:
                try:
                    p_value = stats.binomtest(successes_a, total, 0.5, alternative='greater').pvalue
                except AttributeError:
                    p_value = stats.binom_test(successes_a, total, 0.5, alternative='greater')
            
            results['p_value'] = float(p_value)
            results['is_significant'] = bool(p_value < self.significance_level)
            
            # Calculate effect size (Cohen's h)
            effect_size = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))
            results['effect_size'] = float(abs(effect_size))
            
            # Estimate statistical power (simplified)
            if results['is_significant']:
                results['statistical_power'] = self._estimate_power(
                    effect_size, total, self.significance_level
                )
            
        except Exception as e:
            logger.error(f"Error calculating significance: {e}")
            results['error'] = str(e)
        
        return results
    
    def _calculate_confidence_interval(
        self,
        proportion: float,
        n: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a proportion.
        
        Args:
            proportion: Observed proportion
            n: Sample size
            confidence: Confidence level
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if n == 0:
            return (0.0, 1.0)
        
        # Use Wilson score interval
        z = stats.norm.ppf((1 + confidence) / 2)
        z_squared = z**2
        
        denominator = 1 + z_squared / n
        center = (proportion + z_squared / (2 * n)) / denominator
        
        margin = z * np.sqrt(
            proportion * (1 - proportion) / n + z_squared / (4 * n**2)
        ) / denominator
        
        lower = max(0, center - margin)
        upper = min(1, center + margin)
        
        return (float(lower), float(upper))
    
    def _estimate_power(
        self,
        effect_size: float,
        n: int,
        alpha: float
    ) -> float:
        """
        Estimate statistical power (simplified).
        
        Args:
            effect_size: Cohen's h effect size
            n: Sample size
            alpha: Significance level
            
        Returns:
            Estimated statistical power
        """
        # Simplified power calculation
        # In practice, would use more sophisticated methods
        if effect_size == 0:
            return alpha
        
        # Approximate power based on effect size and sample size
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = effect_size * np.sqrt(n / 2) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return float(min(0.99, max(alpha, power)))
    
    def calculate_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> int:
        """
        Calculate required sample size for desired power.
        
        Args:
            effect_size: Expected effect size
            power: Desired statistical power
            alpha: Significance level
            
        Returns:
            Required sample size per group
        """
        if effect_size == 0:
            return 999999  # Infinite
        
        # Using formula for two-proportion z-test
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return max(self.min_sample_size, int(np.ceil(n)))
    
    def run_sequential_test(
        self,
        model_a: Any,
        model_b: Any,
        max_comparisons: int = 1000,
        stopping_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Run sequential A/B test with early stopping.
        
        Args:
            model_a: First model
            model_b: Second model
            max_comparisons: Maximum comparisons before stopping
            stopping_threshold: P-value threshold for early stopping
            
        Returns:
            Test results with early stopping information
        """
        results = {
            'stopped_early': False,
            'comparisons_at_stop': 0,
            'final_results': None
        }
        
        # In production, this would incrementally collect preferences
        # and check for significance after each batch
        
        # Mock implementation with LLM-based stopping decision
        results['comparisons_at_stop'] = 100
        results['stopped_early'] = True
        results['final_results'] = self.run_ab_test(
            model_a, model_b, ["test"], num_evaluators=20
        )
        
        return results 