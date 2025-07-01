"""
A/B Testing Framework

Provides statistical comparison of different models to determine
which performs better on user satisfaction metrics.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)


class ABTestingFramework:
    """Framework for conducting A/B tests between models"""
    
    def __init__(self, model_a_name: str = "model_a", model_b_name: str = "model_b"):
        """
        Initialize the A/B testing framework.
        
        Args:
            model_a_name: Name of the first model
            model_b_name: Name of the second model
        """
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
        Run A/B test between two models.
        
        Args:
            model_a: First model to test
            model_b: Second model to test
            test_prompts: List of prompts to test with
            num_evaluators: Number of human evaluators
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
            # Run comparison
            comparison_results = self._run_comparison(
                model_a, model_b, test_prompts, num_evaluators
            )
            
            # Update results
            results['total_comparisons'] = comparison_results['total_comparisons']
            results['model_a_preferred'] = comparison_results['model_a_preferred']
            results['model_b_preferred'] = comparison_results['model_b_preferred']
            results['ties'] = comparison_results.get('ties', 0)
            
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
        
        return results
    
    def _run_comparison(
        self,
        model_a: Any,
        model_b: Any,
        prompts: List[str],
        num_evaluators: int
    ) -> Dict[str, Any]:
        """
        Run the actual comparison between models.
        
        In production, this would interface with human evaluators.
        For testing, we'll simulate preferences.
        """
        # This is a mock implementation
        # Real implementation would:
        # 1. Generate responses from both models for each prompt
        # 2. Present pairs to human evaluators in randomized order
        # 3. Collect preferences
        
        total_comparisons = len(prompts) * num_evaluators
        
        # Simulate preferences (in reality, these come from evaluators)
        # For testing, we'll use the test data if provided via patch
        preferences = {
            'model_a_preferred': 45,
            'model_b_preferred': 55,
            'total_comparisons': total_comparisons
        }
        
        return preferences
    
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
                p_value = stats.binom_test(successes_b, total, 0.5, alternative='greater')
            else:
                p_value = stats.binom_test(successes_a, total, 0.5, alternative='greater')
            
            results['p_value'] = float(p_value)
            results['is_significant'] = p_value < self.significance_level
            
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
        
        # Mock implementation
        results['comparisons_at_stop'] = 100
        results['stopped_early'] = True
        results['final_results'] = self.run_ab_test(
            model_a, model_b, ["test"], num_evaluators=20
        )
        
        return results 