#!/usr/bin/env python3
"""
Narrative Engine Basic Evaluation Script

Runs lightweight evaluation suite against model checkpoints to catch obvious failures.
Designed to run after each checkpoint during training (<5 min runtime).

Usage:
    python scripts/run_basic_evaluation.py --checkpoint-path ./checkpoint-100 --output-json eval_results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add narrative engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from narrative_engine.evaluation import run_evaluation_suite
from narrative_engine.model import NarrativeLLM, create_narrative_model
from narrative_engine.config import NarrativeLLMConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path) -> tuple[Any, Any]:
    """
    Load model and tokenizer from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        # Create model with default config
        config = NarrativeLLMConfig()
        model = create_narrative_model(config)
        
        # Load checkpoint weights if they exist
        checkpoint_file = checkpoint_path / "pytorch_model.bin"
        if checkpoint_file.exists():
            import torch
            state_dict = torch.load(checkpoint_file, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded checkpoint weights")
        else:
            logger.warning(f"No checkpoint file found at {checkpoint_file}")
            
        return model, model.tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        # Return None to use mock mode
        return None, None


def load_training_history(checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load training history from checkpoint metadata.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Training history dict or None
    """
    history_file = checkpoint_path / "training_history.json"
    if history_file.exists():
        try:
            with open(history_file) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load training history: {e}")
            
    # Try to load from wandb logs or other sources
    wandb_file = checkpoint_path / "wandb_history.json"
    if wandb_file.exists():
        try:
            with open(wandb_file) as f:
                data = json.load(f)
                # Convert wandb format to our format
                return {
                    'loss': data.get('train/loss', []),
                    'steps': data.get('train/global_step', []),
                    'generation_loss': data.get('train/generation_loss', []),
                    'control_loss': data.get('train/control_loss', [])
                }
        except Exception as e:
            logger.warning(f"Failed to load wandb history: {e}")
            
    return None


def main():
    parser = argparse.ArgumentParser(description="Narrative Engine Basic Evaluation")
    parser.add_argument(
        "--checkpoint-path", 
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--output-json", 
        required=True,
        help="Path to save evaluation results JSON"
    )
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Log results to wandb"
    )
    parser.add_argument(
        "--mock-mode",
        action="store_true", 
        help="Run in mock mode for testing"
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.5,
        help="Threshold for failing evaluation (generation success rate)"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists() and not args.mock_mode:
        logger.error(f"Checkpoint path does not exist: {checkpoint_path}")
        return 1
        
    try:
        # Load model and tokenizer
        model, tokenizer = None, None
        if not args.mock_mode:
            model, tokenizer = load_checkpoint(checkpoint_path)
            
        # Load training history
        training_history = load_training_history(checkpoint_path)
        
        # Run evaluation suite
        logger.info("Starting evaluation suite...")
        start_time = time.time()
        
        results = run_evaluation_suite(
            checkpoint_path=str(checkpoint_path),
            model=model,
            tokenizer=tokenizer,
            training_history=training_history,
            output_json=args.output_json,
            log_to_wandb=args.log_to_wandb
        )
        
        duration = time.time() - start_time
        
        # Print summary
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Checkpoint: {checkpoint_path}")
        logger.info(f"Duration: {duration:.2f}s")
        logger.info(f"Overall: {'PASSED' if results['passed'] else 'FAILED'}")
        
        if 'basic_generation' in results:
            gen_results = results['basic_generation']
            logger.info(f"Generation Success Rate: {gen_results.get('generation_success_rate', 0):.1%}")
            logger.info(f"Coherence Score: {gen_results.get('coherence_score', 0):.3f}")
            
        if 'dual_head_sanity' in results:
            dual_results = results['dual_head_sanity']
            logger.info(f"Dual Heads Functional: {dual_results.get('both_heads_functional', False)}")
            
        if 'training_progress' in results:
            progress = results['training_progress']
            logger.info(f"Loss Decreasing: {progress.get('loss_decreasing', False)}")
            logger.info(f"Final Loss: {progress.get('final_loss', 'N/A')}")
            
        if results.get('warnings'):
            logger.warning("Warnings:")
            for warning in results['warnings']:
                logger.warning(f"  - {warning}")
                
        # Return exit code based on pass/fail
        if results['passed']:
            logger.info("✅ Evaluation PASSED - checkpoint is good")
            return 0
        else:
            logger.error("❌ Evaluation FAILED - checkpoint should be rejected")
            if 'failure_reason' in results:
                logger.error(f"Reason: {results['failure_reason']}")
            return 1
            
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Still try to save partial results
        error_results = {
            'checkpoint_path': str(checkpoint_path),
            'passed': False,
            'error': str(e),
            'timestamp': time.time()
        }
        
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(error_results, f, indent=2)
            
        return 1


if __name__ == "__main__":
    sys.exit(main()) 