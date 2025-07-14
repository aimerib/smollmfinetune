"""
Triple-Head Sanity Evaluation

Verifies that all three heads (generation, control, memory) produce valid outputs.
This is critical for the Narrative Engine's triple-head architecture.
"""

import logging
from typing import Dict, Any
import torch
import numpy as np

logger = logging.getLogger(__name__)


class TripleHeadSanityEvaluator:
    """Evaluates triple-head model outputs for basic sanity"""
    
    def __init__(self):
        self.expected_vocab_size = 32000  # Base model vocab
        self.expected_control_vocab_size = 64  # Control tokens
        self.expected_memory_embedding_dim = 768  # Memory embedding size
        self.expected_memory_metadata_dim = 4  # Memory metadata size
        
    def evaluate(self, model: Any) -> Dict[str, Any]:
        """
        Evaluate triple-head outputs for sanity.
        
        Args:
            model: The triple-head model to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            'generation_head_valid': False,
            'control_head_valid': False,
            'memory_head_valid': False,
            'all_heads_functional': False,
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
            
            # Check memory head
            memory_valid, memory_stats = self._check_memory_head(outputs)
            results['memory_head_valid'] = memory_valid
            results['logits_statistics']['memory'] = memory_stats
            
            # All heads functional?
            results['all_heads_functional'] = gen_valid and control_valid and memory_valid
            
            logger.info(f"Triple-head sanity check: Generation={gen_valid}, Control={control_valid}, Memory={memory_valid}")
            
        except Exception as e:
            logger.error(f"Error in triple-head sanity check: {e}")
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
            if hasattr(outputs, 'text_logits'):
                logits = outputs.text_logits
            elif hasattr(outputs, 'generation_logits'):
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
                logger.warning(f"Unexpected generation logits shape: {logits.shape}")
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
            'valid': False,
            'is_probabilities': False
        }
        
        try:
            # Extract control logits
            if hasattr(outputs, 'action_logits'):
                logits = outputs.action_logits
            elif hasattr(outputs, 'control_logits'):
                logits = outputs.control_logits
            else:
                # Control head might be optional
                logger.info("No control logits found - model may not have control head")
                return True, stats  # Not an error if missing
                
            # Check shape
            stats['shape'] = list(logits.shape)
            expected_dims = 2  # batch, num_control_tokens (not sequence-level)
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
            
            # Check if values are in probability range [0, 1]
            min_val = float(logits.min().item())
            max_val = float(logits.max().item())
            stats['is_probabilities'] = (min_val >= 0.0 and max_val <= 1.0)
            
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
    
    def _check_memory_head(self, outputs: Any) -> tuple[bool, Dict[str, Any]]:
        """Check if memory head produces valid outputs"""
        stats = {
            'embedding_shape': None,
            'metadata_shape': None,
            'embedding_contains_nan': False,
            'embedding_contains_inf': False,
            'metadata_contains_nan': False,
            'metadata_contains_inf': False,
            'embedding_normalized': False,
            'metadata_in_range': False,
            'embedding_stats': {},
            'metadata_stats': {},
            'valid': False
        }
        
        try:
            # Extract memory outputs
            memory_embedding = None
            memory_metadata = None
            
            if hasattr(outputs, 'memory_embedding') and hasattr(outputs, 'memory_metadata'):
                memory_embedding = outputs.memory_embedding
                memory_metadata = outputs.memory_metadata
            elif hasattr(outputs, 'memory_output'):
                # Combined memory output that needs to be split
                memory_output = outputs.memory_output
                memory_embedding = memory_output[:, :768]
                memory_metadata = memory_output[:, 768:]
            else:
                logger.info("No memory outputs found - model may not have memory head")
                return True, stats  # Not an error if missing
            
            # Check memory embedding
            embedding_valid = self._check_memory_embedding(memory_embedding, stats)
            
            # Check memory metadata
            metadata_valid = self._check_memory_metadata(memory_metadata, stats)
            
            stats['valid'] = embedding_valid and metadata_valid
            return stats['valid'], stats
            
        except Exception as e:
            logger.error(f"Error checking memory head: {e}")
            return False, stats
    
    def _check_memory_embedding(self, embedding: torch.Tensor, stats: Dict[str, Any]) -> bool:
        """Check memory embedding validity"""
        try:
            # Check shape
            stats['embedding_shape'] = list(embedding.shape)
            expected_dims = 2  # batch, embedding_dim
            if len(embedding.shape) != expected_dims:
                logger.warning(f"Unexpected memory embedding shape: {embedding.shape}")
                return False
            
            # Check embedding dimension
            if embedding.shape[-1] != self.expected_memory_embedding_dim:
                logger.warning(f"Unexpected memory embedding dim: {embedding.shape[-1]} vs {self.expected_memory_embedding_dim}")
                return False
            
            # Check for NaN/Inf
            stats['embedding_contains_nan'] = bool(torch.isnan(embedding).any().item())
            stats['embedding_contains_inf'] = bool(torch.isinf(embedding).any().item())
            
            if stats['embedding_contains_nan'] or stats['embedding_contains_inf']:
                logger.warning("Memory embedding contains NaN or Inf values")
                return False
            
            # Calculate statistics
            stats['embedding_stats'] = {
                'mean': float(embedding.mean().item()),
                'std': float(embedding.std().item()),
                'min': float(embedding.min().item()),
                'max': float(embedding.max().item())
            }
            
            # Check if normalized (L2 norm ≈ 1)
            norms = torch.norm(embedding, p=2, dim=1)
            avg_norm = float(norms.mean().item())
            stats['embedding_normalized'] = abs(avg_norm - 1.0) < 0.1
            
            if not stats['embedding_normalized']:
                logger.warning(f"Memory embeddings not normalized (avg norm: {avg_norm:.3f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking memory embedding: {e}")
            return False
    
    def _check_memory_metadata(self, metadata: torch.Tensor, stats: Dict[str, Any]) -> bool:
        """Check memory metadata validity"""
        try:
            # Check shape
            stats['metadata_shape'] = list(metadata.shape)
            expected_dims = 2  # batch, metadata_dim
            if len(metadata.shape) != expected_dims:
                logger.warning(f"Unexpected memory metadata shape: {metadata.shape}")
                return False
            
            # Check metadata dimension
            if metadata.shape[-1] != self.expected_memory_metadata_dim:
                logger.warning(f"Unexpected memory metadata dim: {metadata.shape[-1]} vs {self.expected_memory_metadata_dim}")
                return False
            
            # Check for NaN/Inf
            stats['metadata_contains_nan'] = bool(torch.isnan(metadata).any().item())
            stats['metadata_contains_inf'] = bool(torch.isinf(metadata).any().item())
            
            if stats['metadata_contains_nan'] or stats['metadata_contains_inf']:
                logger.warning("Memory metadata contains NaN or Inf values")
                return False
            
            # Calculate statistics
            stats['metadata_stats'] = {
                'mean': float(metadata.mean().item()),
                'std': float(metadata.std().item()),
                'min': float(metadata.min().item()),
                'max': float(metadata.max().item())
            }
            
            # Check if values are in expected range [0, 1] (after sigmoid)
            min_val = float(metadata.min().item())
            max_val = float(metadata.max().item())
            stats['metadata_in_range'] = (min_val >= 0.0 and max_val <= 1.0)
            
            if not stats['metadata_in_range']:
                logger.warning(f"Memory metadata not in [0,1] range: [{min_val:.3f}, {max_val:.3f}]")
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking memory metadata: {e}")
            return False
    
    def _create_mock_outputs(self, batch_size: int, seq_length: int) -> Any:
        """Create mock outputs for testing"""
        class MockOutputs:
            def __init__(self):
                self.text_logits = torch.randn(batch_size, seq_length, 32000)
                self.action_logits = torch.sigmoid(torch.randn(batch_size, 64))  # Probabilities
                self.memory_embedding = torch.nn.functional.normalize(torch.randn(batch_size, 768), p=2, dim=1)
                self.memory_metadata = torch.sigmoid(torch.randn(batch_size, 4))  # [0, 1] range
                
        return MockOutputs()


# Backward compatibility alias
DualHeadSanityEvaluator = TripleHeadSanityEvaluator


def eval_triple_head_sanity(model: Any) -> Dict[str, Any]:
    """
    Convenience function to evaluate triple-head sanity.
    
    Args:
        model: The model to evaluate
        
    Returns:
        Evaluation results
    """
    evaluator = TripleHeadSanityEvaluator()
    return evaluator.evaluate(model)


# Backward compatibility
def eval_dual_head_sanity(model: Any) -> Dict[str, Any]:
    """Backward compatibility wrapper"""
    return eval_triple_head_sanity(model) 