"""
Evaluation harness for Narrative Engine

This module provides evaluation functions for testing model quality,
training progress, and architectural correctness.
"""

from datetime import datetime
from .eval_basic_generation import BasicGenerationEvaluator
from .eval_triple_head_sanity import TripleHeadSanityEvaluator, eval_triple_head_sanity
from .eval_dual_head_sanity import DualHeadSanityEvaluator  # Backward compatibility
from .eval_training_progress import TrainingProgressEvaluator
from .eval_json_correctness import JSONCorrectnessEvaluator
from .eval_coherence import CoherenceEvaluator
from .eval_latency import LatencyEvaluator
from .eval_memory_consistency import MemoryConsistencyEvaluator
from .safety_layer import SafetyLayer
import logging
from typing import Dict, Any, Optional, List
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Create evaluator instances
eval_basic_generation = BasicGenerationEvaluator()
eval_triple_head_sanity = TripleHeadSanityEvaluator()
eval_training_progress = TrainingProgressEvaluator()
eval_json_correctness = JSONCorrectnessEvaluator()
eval_coherence = CoherenceEvaluator()
eval_latency = LatencyEvaluator()
eval_memory_consistency = MemoryConsistencyEvaluator()

# Backward compatibility
eval_dual_head_sanity = eval_triple_head_sanity


def run_evaluation_suite(
    checkpoint_path: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    training_history: Optional[Dict[str, List[float]]] = None,
    output_json: Optional[str] = None,
    log_to_wandb: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation suite on a model checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        training_history: Training metrics history
        output_json: Path to save results JSON
        log_to_wandb: Whether to log to Weights & Biases
        
    Returns:
        Dictionary with all evaluation results
    """
    
    logger.info(f"🧪 Starting evaluation suite for {checkpoint_path}")
    
    results = {
        'checkpoint_path': checkpoint_path,
        'passed': False,
        'evaluations': {}
    }
    start_time = datetime.now()
    
    try:
        # Load model if not provided
        if model is None:
            logger.info("Loading model from checkpoint...")
            # Import here to avoid circular imports
            from ..model import create_narrative_model
            model = create_narrative_model()
            # TODO: Load actual checkpoint weights
        
        # Load tokenizer if not provided
        if tokenizer is None and hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        
        # 1. Basic generation evaluation
        logger.info("Running basic generation evaluation...")
        try:
            gen_results = eval_basic_generation.evaluate(model, tokenizer)
            results['evaluations']['basic_generation'] = gen_results
        except Exception as e:
            logger.error(f"Basic generation evaluation failed: {e}")
            results['evaluations']['basic_generation'] = {'error': str(e)}
        
        # 2. Triple-head sanity check
        logger.info("Running triple-head sanity evaluation...")
        try:
            sanity_results = eval_triple_head_sanity.evaluate(model)
            results['evaluations']['triple_head_sanity'] = sanity_results
        except Exception as e:
            logger.error(f"Triple-head sanity evaluation failed: {e}")
            results['evaluations']['triple_head_sanity'] = {'error': str(e)}
        
        # 3. Training progress evaluation
        if training_history:
            logger.info("Running training progress evaluation...")
            try:
                progress_results = eval_training_progress.evaluate(
                    model, training_history, checkpoint_path
                )
                results['evaluations']['training_progress'] = progress_results
            except Exception as e:
                logger.error(f"Training progress evaluation failed: {e}")
                results['evaluations']['training_progress'] = {'error': str(e)}
        
        # Determine if evaluation passed
        results['passed'] = _determine_evaluation_success(results['evaluations'])
        results['timestamp'] = datetime.now().isoformat()
        results['duration'] = (datetime.now() - start_time).total_seconds()

        # Save results
        if output_json:
            logger.info(f"Saving evaluation results to {output_json}")
            with open(output_json, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Log to wandb if requested
        if log_to_wandb:
            try:
                _log_to_wandb(results)
            except Exception as e:
                logger.warning(f"Failed to log to wandb: {e}")
        
        logger.info(f"✅ Evaluation suite completed. Passed: {results['passed']}")

        
    except Exception as e:
        logger.error(f"❌ Evaluation suite failed: {e}")
        results['error'] = str(e)
        results['failure_reason'] = str(e)
        results['passed'] = False
        results['timestamp'] = datetime.now().isoformat()
        results['duration'] = (datetime.now() - start_time).total_seconds()
    
    return results


def _determine_evaluation_success(evaluations: Dict[str, Any]) -> bool:
    """
    Determine if evaluation suite passed based on individual results.
    
    Args:
        evaluations: Dictionary of evaluation results
        
    Returns:
        True if all critical evaluations passed
    """
    
    # Check basic generation
    basic_gen = evaluations.get('basic_generation', {})
    if basic_gen.get('error'):
        logger.warning("Basic generation evaluation had errors")
        return False
    
    generation_success_rate = basic_gen.get('generation_success_rate', 0)
    if generation_success_rate < 0.8:  # 80% success rate required
        logger.warning(f"Low generation success rate: {generation_success_rate}")
        return False
    
    # Check triple-head sanity
    sanity = evaluations.get('triple_head_sanity', {})
    if sanity.get('error'):
        logger.warning("Triple-head sanity evaluation had errors")
        return False
    
    all_heads_functional = sanity.get('all_heads_functional', False)
    if not all_heads_functional:
        logger.warning("Not all heads are functional")
        return False
    
    # Check training progress (if available)
    progress = evaluations.get('training_progress', {})
    if progress and not progress.get('error'):
        loss_decreasing = progress.get('loss_decreasing', False)
        if not loss_decreasing:
            logger.warning("Training loss is not decreasing")
            return False
    
    return True


def _log_to_wandb(results: Dict[str, Any]) -> None:
    """Log evaluation results to Weights & Biases"""
    try:
        import wandb
        
        # Extract key metrics
        metrics = {
            'eval/checkpoint_path': results['checkpoint_path'],
            'eval/passed': results['passed']
        }
        
        # Basic generation metrics
        basic_gen = results.get('evaluations', {}).get('basic_generation', {})
        if basic_gen and not basic_gen.get('error'):
            metrics.update({
                'eval/generation_success_rate': basic_gen.get('generation_success_rate', 0),
                'eval/avg_generation_length': basic_gen.get('avg_generation_length', 0),
                'eval/perplexity': basic_gen.get('perplexity', float('inf'))
            })
        
        # Triple-head sanity metrics
        sanity = results.get('evaluations', {}).get('triple_head_sanity', {})
        if sanity and not sanity.get('error'):
            metrics.update({
                'eval/all_heads_functional': sanity.get('all_heads_functional', False),
                'eval/generation_head_valid': sanity.get('generation_head_valid', False),
                'eval/control_head_valid': sanity.get('control_head_valid', False),
                'eval/memory_head_valid': sanity.get('memory_head_valid', False)
            })
        
        # Training progress metrics
        progress = results.get('evaluations', {}).get('training_progress', {})
        if progress and not progress.get('error'):
            metrics.update({
                'eval/loss_decreasing': progress.get('loss_decreasing', False),
                'eval/final_loss': progress.get('final_loss', float('inf')),
                'eval/loss_improvement': progress.get('loss_improvement', 0)
            })
        
        # Log to wandb
        wandb.log(metrics)
        logger.info("✅ Logged evaluation results to wandb")
        
    except ImportError:
        logger.warning("wandb not available for logging")
    except Exception as e:
        logger.error(f"Failed to log to wandb: {e}")


# Backward compatibility exports
__all__ = [
    'run_evaluation_suite',
    'eval_basic_generation',
    'eval_triple_head_sanity',
    'eval_dual_head_sanity',  # Backward compatibility
    'eval_training_progress',
    'eval_json_correctness',
    'eval_coherence',
    'eval_latency',
    'eval_memory_consistency',
    'BasicGenerationEvaluator',
    'TripleHeadSanityEvaluator',
    'DualHeadSanityEvaluator',  # Backward compatibility
    'TrainingProgressEvaluator',
    'JSONCorrectnessEvaluator',
    'CoherenceEvaluator', 
    'LatencyEvaluator',
    'MemoryConsistencyEvaluator',
    'SafetyLayer'
] 