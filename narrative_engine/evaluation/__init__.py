"""
Narrative Engine Evaluation Harness

Provides lightweight evaluation infrastructure for testing model checkpoints
during training. Focuses on catching obvious failures rather than nuanced quality.
"""

from .eval_basic_generation import BasicGenerationEvaluator
from .eval_dual_head_sanity import DualHeadSanityEvaluator  
from .eval_training_progress import TrainingProgressEvaluator

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

logger = logging.getLogger(__name__)

# Make evaluators available at package level
eval_basic_generation = BasicGenerationEvaluator
eval_dual_head_sanity = DualHeadSanityEvaluator
eval_training_progress = TrainingProgressEvaluator


def run_evaluation_suite(
    checkpoint_path: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    training_history: Optional[Dict[str, List[float]]] = None,
    output_json: Optional[str] = None,
    log_to_wandb: bool = False
) -> Dict[str, Any]:
    """
    Run the complete evaluation suite on a checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model: Loaded model (if None, will be loaded from checkpoint)
        tokenizer: Tokenizer instance (if None, will be loaded)
        training_history: Training metrics history 
        output_json: Path to save JSON results
        log_to_wandb: Whether to log results to wandb
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info(f"Starting evaluation of checkpoint: {checkpoint_path}")
    start_time = time.time()
    
    results = {
        'checkpoint_path': checkpoint_path,
        'timestamp': time.time(),
        'passed': True,
        'error': None
    }
    
    try:
        # Load model if not provided
        if model is None:
            logger.info("Loading model from checkpoint...")
            # In production, would load actual model
            # For now, using mock behavior
            model = _load_model_from_checkpoint(checkpoint_path)
            
        if tokenizer is None:
            logger.info("Loading tokenizer...")
            tokenizer = _load_tokenizer()
            
        # Put model in eval mode
        if hasattr(model, 'eval'):
            model.eval()
        
        # Run basic generation evaluation
        logger.info("Running basic generation evaluation...")
        gen_evaluator = BasicGenerationEvaluator()
        gen_results = gen_evaluator.evaluate(model, tokenizer)
        results['basic_generation'] = gen_results
        
        # Run dual-head sanity check
        logger.info("Running dual-head sanity evaluation...")
        dual_evaluator = DualHeadSanityEvaluator()
        dual_results = dual_evaluator.evaluate(model)
        results['dual_head_sanity'] = dual_results
        
        # Run training progress evaluation
        if training_history:
            logger.info("Running training progress evaluation...")
            progress_evaluator = TrainingProgressEvaluator()
            progress_results = progress_evaluator.evaluate(training_history)
            results['training_progress'] = progress_results
        
        # Determine pass/fail based on thresholds
        if gen_results.get('generation_success_rate', 0) < 0.5:
            results['passed'] = False
            results['failure_reason'] = 'Low generation success rate'
            
        if not dual_results.get('both_heads_functional', False):
            results['passed'] = False
            results['failure_reason'] = 'Dual heads not functional'
            
        if training_history and not progress_results.get('loss_decreasing', False):
            results['passed'] = False  
            results['failure_reason'] = 'Loss not decreasing'
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        results['passed'] = False
        results['error'] = str(e)
        
    # Calculate duration
    results['duration'] = time.time() - start_time
    logger.info(f"Evaluation completed in {results['duration']:.2f}s")
    
    # Log to wandb if requested
    if log_to_wandb:
        _log_to_wandb(results)
        
    # Save to JSON if requested
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_json}")
        
    return results


def _load_model_from_checkpoint(checkpoint_path: str):
    """Load model from checkpoint (placeholder for actual implementation)"""
    # In production, this would load the actual model
    # For now, return a mock
    logger.info(f"Would load model from {checkpoint_path}")
    return None


def _load_tokenizer():
    """Load tokenizer (placeholder for actual implementation)"""
    # In production, this would load the actual tokenizer
    logger.info("Would load tokenizer")
    return None


def _log_to_wandb(results: Dict[str, Any]):
    """Log evaluation results to wandb"""
    try:
        import wandb
        
        # Flatten results for wandb logging
        wandb_data = {
            'eval/basic_generation_score': results.get('basic_generation', {}).get('coherence_score', 0),
            'eval/generation_success_rate': results.get('basic_generation', {}).get('generation_success_rate', 0),
            'eval/dual_head_sanity': results.get('dual_head_sanity', {}).get('both_heads_functional', False),
            'eval/training_progress': results.get('training_progress', {}).get('loss_decreasing', False),
            'eval/passed': results['passed'],
            'eval/duration': results['duration']
        }
        
        wandb.log(wandb_data)
        logger.info("Results logged to wandb")
        
    except ImportError:
        logger.warning("wandb not available, skipping logging")
    except Exception as e:
        logger.error(f"Failed to log to wandb: {e}") 