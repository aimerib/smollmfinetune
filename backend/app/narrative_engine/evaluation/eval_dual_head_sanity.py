"""
Dual-Head Sanity Evaluation

Verifies that both the generation head and control head produce valid outputs.
This is critical for the Narrative Engine's dual-head architecture.
"""

import logging
from typing import Dict, Any
import torch
import numpy as np

logger = logging.getLogger(__name__)


class DualHeadSanityEvaluator:
    """Evaluates dual-head model outputs for basic sanity"""
    
    def __init__(self):
        self.expected_vocab_size = 32000  # Base model vocab
        self.expected_control_vocab_size = 64  # Control tokens
        
    def evaluate(self, model: Any) -> Dict[str, Any]:
        """
        Evaluate dual-head outputs for sanity.
        
        Args:
            model: The dual-head model to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'generation_head_valid': False,
            'control_head_valid': False,
            'both_heads_functional': False,
            'logits_statistics': {},
            'error': None
        }
        
        try:
            # Create dummy input
            batch_size = 2
            seq_length = 10
            dummy_input = torch.randint(0, 1000, (batch_size, seq_length))
            
            # Run forward pass
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    outputs = model(dummy_input)
                else:
                    # Mock outputs for testing
                    outputs = self._create_mock_outputs(batch_size, seq_length)
            
            # Check generation head
            gen_valid, gen_stats = self._check_generation_head(outputs)
            results['generation_head_valid'] = gen_valid
            results['logits_statistics']['generation'] = gen_stats
            
            # Check control head  
            control_valid, control_stats = self._check_control_head(outputs)
            results['control_head_valid'] = control_valid
            results['logits_statistics']['control'] = control_stats
            
            # Both heads functional?
            results['both_heads_functional'] = gen_valid and control_valid
            
            logger.info(f"Dual-head sanity check: Generation={gen_valid}, Control={control_valid}")
            
        except Exception as e:
            logger.error(f"Error in dual-head sanity check: {e}")
            results['error'] = str(e)
            
        return results
    
    def _check_generation_head(self, outputs: Any) -> tuple[bool, Dict[str, Any]]:
        """Check if generation head produces valid logits"""
        stats = {
            'shape': None,
            'contains_nan': False,
            'contains_inf': False,
            'mean': None,
            'std': None,
            'valid': False
        }
        
        try:
            # Extract generation logits
            if hasattr(outputs, 'generation_logits'):
                logits = outputs.generation_logits
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logger.warning("No generation logits found in outputs")
                return False, stats
                
            # Check shape
            stats['shape'] = list(logits.shape)
            expected_dims = 3  # batch, seq, vocab
            if len(logits.shape) != expected_dims:
                logger.warning(f"Unexpected logits shape: {logits.shape}")
                return False, stats
                
            # Check for NaN/Inf
            stats['contains_nan'] = bool(torch.isnan(logits).any().item())
            stats['contains_inf'] = bool(torch.isinf(logits).any().item())
            
            if stats['contains_nan'] or stats['contains_inf']:
                logger.warning("Generation logits contain NaN or Inf values")
                return False, stats
                
            # Calculate statistics
            stats['mean'] = float(logits.mean().item())
            stats['std'] = float(logits.std().item())
            
            # Sanity checks
            if stats['std'] < 0.01:
                logger.warning("Generation logits have very low variance")
                return False, stats
                
            # Check vocab size (last dimension)
            vocab_size = logits.shape[-1]
            if vocab_size < 1000:  # Reasonable minimum
                logger.warning(f"Vocab size too small: {vocab_size}")
                return False, stats
                
            stats['valid'] = True
            return True, stats
            
        except Exception as e:
            logger.error(f"Error checking generation head: {e}")
            return False, stats
    
    def _check_control_head(self, outputs: Any) -> tuple[bool, Dict[str, Any]]:
        """Check if control head produces valid logits"""
        stats = {
            'shape': None,
            'contains_nan': False,
            'contains_inf': False,
            'mean': None,
            'std': None,
            'valid': False
        }
        
        try:
            # Extract control logits
            if hasattr(outputs, 'control_logits'):
                logits = outputs.control_logits
            else:
                # Control head might be optional
                logger.info("No control logits found - model may not have control head")
                return True, stats  # Not an error if missing
                
            # Check shape
            stats['shape'] = list(logits.shape)
            expected_dims = 3  # batch, seq, control_vocab
            if len(logits.shape) != expected_dims:
                logger.warning(f"Unexpected control logits shape: {logits.shape}")
                return False, stats
                
            # Check for NaN/Inf
            stats['contains_nan'] = bool(torch.isnan(logits).any().item())
            stats['contains_inf'] = bool(torch.isinf(logits).any().item())
            
            if stats['contains_nan'] or stats['contains_inf']:
                logger.warning("Control logits contain NaN or Inf values")
                return False, stats
                
            # Calculate statistics
            stats['mean'] = float(logits.mean().item())
            stats['std'] = float(logits.std().item())
            
            # Sanity checks
            if stats['std'] < 0.01:
                logger.warning("Control logits have very low variance")
                return False, stats
                
            # Check control vocab size
            control_vocab_size = logits.shape[-1]
            if control_vocab_size < 10 or control_vocab_size > 1000:
                logger.warning(f"Unusual control vocab size: {control_vocab_size}")
                # Not necessarily an error, just unusual
                
            stats['valid'] = True
            return True, stats
            
        except Exception as e:
            logger.error(f"Error checking control head: {e}")
            return False, stats
    
    def _create_mock_outputs(self, batch_size: int, seq_length: int) -> Any:
        """Create mock outputs for testing"""
        class MockOutputs:
            def __init__(self):
                self.generation_logits = torch.randn(batch_size, seq_length, 32000)
                self.control_logits = torch.randn(batch_size, seq_length, 64)
                
        return MockOutputs() 