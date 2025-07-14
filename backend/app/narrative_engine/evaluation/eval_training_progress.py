"""
Training Progress Evaluation

Tracks loss curves and convergence metrics to ensure training is progressing properly.
Helps catch issues like exploding gradients or stuck training early.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class TrainingProgressEvaluator:
    """Evaluates training progress and convergence"""
    
    def __init__(self):
        self.min_improvement_threshold = 0.01  # 1% improvement
        self.patience_steps = 50  # Steps to wait for improvement
        
    def evaluate(self, training_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Evaluate training progress based on loss history.
        
        Args:
            training_history: Dictionary containing loss curves and steps
                Expected keys: 'loss', 'steps', optionally 'generation_loss', 'control_loss'
                
        Returns:
            Dictionary with progress metrics
        """
        results = {
            'loss_decreasing': False,
            'convergence_rate': 0.0,
            'final_loss': None,
            'loss_variance': 0.0,
            'is_converged': False,
            'has_exploded': False,
            'smoothed_loss_curve': [],
            'warnings': []
        }
        
        try:
            # Extract loss history
            if 'loss' not in training_history:
                results['warnings'].append('No loss history found')
                return results
                
            losses = training_history['loss']
            steps = training_history.get('steps', list(range(len(losses))))
            
            if len(losses) < 2:
                results['warnings'].append('Insufficient loss history')
                return results
                
            # Basic metrics
            results['final_loss'] = losses[-1]
            results['loss_variance'] = float(np.std(losses))
            
            # Check if loss is decreasing overall
            results['loss_decreasing'] = self._is_decreasing(losses)
            
            # Calculate convergence rate
            results['convergence_rate'] = self._calculate_convergence_rate(losses, steps)
            
            # Check for convergence
            results['is_converged'] = self._check_convergence(losses)
            
            # Check for exploding loss
            results['has_exploded'] = self._check_explosion(losses)
            
            # Smooth loss curve for visualization
            results['smoothed_loss_curve'] = self._smooth_curve(losses)
            
            # Component losses if available
            if 'generation_loss' in training_history:
                gen_results = self._analyze_component_loss(
                    training_history['generation_loss'], 
                    'generation'
                )
                results['generation_loss_analysis'] = gen_results
                
            if 'control_loss' in training_history:
                control_results = self._analyze_component_loss(
                    training_history['control_loss'],
                    'control'
                )
                results['control_loss_analysis'] = control_results
                
            # Add warnings based on analysis
            if results['has_exploded']:
                results['warnings'].append('Loss has exploded during training')
            elif not results['loss_decreasing']:
                results['warnings'].append('Loss is not decreasing')
            elif results['convergence_rate'] < 0.001:
                results['warnings'].append('Very slow convergence')
                
            logger.info(f"Training progress: Decreasing={results['loss_decreasing']}, "
                       f"Rate={results['convergence_rate']:.4f}, Final={results['final_loss']:.4f}")
                       
        except Exception as e:
            logger.error(f"Error evaluating training progress: {e}")
            results['error'] = str(e)
            
        return results
    
    def _is_decreasing(self, losses: List[float], window: int = 10) -> bool:
        """Check if loss is decreasing over time using windowed comparison"""
        if len(losses) < window * 2:
            # Simple check for short sequences
            return losses[-1] < losses[0]
            
        # Compare average of last window vs first window
        early_avg = np.mean(losses[:window])
        late_avg = np.mean(losses[-window:])
        
        return late_avg < early_avg * (1 - self.min_improvement_threshold)
    
    def _calculate_convergence_rate(self, losses: List[float], steps: List[int]) -> float:
        """Calculate the rate of convergence (negative slope of loss curve)"""
        if len(losses) < 2:
            return 0.0
            
        try:
            # Fit linear regression to log loss (for exponential decay)
            log_losses = np.log(np.maximum(losses, 1e-10))  # Avoid log(0)
            coeffs = np.polyfit(steps, log_losses, 1)
            
            # Negative slope is convergence rate
            convergence_rate = -coeffs[0]
            return max(0.0, convergence_rate)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"Failed to calculate convergence rate: {e}")
            return 0.0
    
    def _check_convergence(self, losses: List[float]) -> bool:
        """Check if training has converged (loss plateaued)"""
        if len(losses) < self.patience_steps:
            return False
            
        # Check if recent losses are stable
        recent_losses = losses[-self.patience_steps:]
        recent_std = np.std(recent_losses)
        recent_mean = np.mean(recent_losses)
        
        # Converged if variance is very small relative to mean
        relative_variance = recent_std / (recent_mean + 1e-10)
        return relative_variance < 0.01
    
    def _check_explosion(self, losses: List[float]) -> bool:
        """Check if loss has exploded (NaN, Inf, or very large)"""
        if not losses:
            return False
            
        # Check for NaN or Inf
        if any(np.isnan(loss) or np.isinf(loss) for loss in losses):
            return True
            
        # Check for sudden large increase
        if len(losses) > 1:
            max_increase = max(losses[i] / losses[i-1] 
                             for i in range(1, len(losses))
                             if losses[i-1] > 0)
            if max_increase > 100:  # 100x increase
                return True
                
        # Check absolute magnitude
        if max(losses) > 1e6:
            return True
            
        return False
    
    def _smooth_curve(self, values: List[float], window: int = 5) -> List[float]:
        """Apply moving average smoothing to curve"""
        if len(values) < window:
            return values
            
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(np.mean(values[start:end]))
            
        return smoothed
    
    def _analyze_component_loss(self, losses: List[float], name: str) -> Dict[str, Any]:
        """Analyze a component loss (generation or control)"""
        return {
            'name': name,
            'final_value': losses[-1] if losses else None,
            'is_decreasing': self._is_decreasing(losses),
            'relative_contribution': None,  # Would need total loss to calculate
            'stability': 1.0 - min(1.0, np.std(losses[-10:]) if len(losses) > 10 else 1.0)
        } 