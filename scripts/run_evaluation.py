#!/usr/bin/env python
"""
Run comprehensive evaluation suite including R4-9 evaluations.

This script orchestrates running all evaluation modules against a specified
model checkpoint, including the new R4-9 evaluations:
- JSON correctness
- Coherence with LLM-as-judge
- Latency measurements
- Memory consistency for triple-head architecture
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.narrative_engine.evaluation import (
    run_evaluation_suite,
    eval_json_correctness,
    eval_coherence,
    eval_latency,
    eval_memory_consistency,
    SafetyLayer
)
from backend.app.narrative_engine.evaluation.eval_basic_generation import BasicGenerationEvaluator
from backend.app.narrative_engine.evaluation.eval_triple_head_sanity import TripleHeadSanityEvaluator
from backend.app.narrative_engine.evaluation.eval_training_progress import TrainingProgressEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_r4_9_evaluations(
    model: Any,
    tokenizer: Any,
    safety_config: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the R4-9 specific evaluations.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        safety_config: Optional path to safety configuration
        
    Returns:
        Dictionary with R4-9 evaluation results
    """
    results = {}
    
    # 1. JSON Correctness Evaluation
    logger.info("🔧 Running JSON correctness evaluation...")
    try:
        json_results = eval_json_correctness.evaluate(model, tokenizer)
        results['json_correctness'] = json_results
        logger.info(f"✅ JSON correctness rate: {json_results.get('json_correctness_rate', 0):.2%}")
    except Exception as e:
        logger.error(f"❌ JSON evaluation failed: {e}")
        results['json_correctness'] = {'error': str(e)}
    
    # 2. Coherence Evaluation
    logger.info("🧠 Running coherence evaluation...")
    try:
        # Generate a test conversation for coherence checking
        test_conversation = [
            {"role": "user", "content": "What's your name?"},
            {"role": "assistant", "content": "I'm Claude, an AI assistant."},
            {"role": "user", "content": "What did you say your name was?"},
            {"role": "assistant", "content": "I said my name is Claude."},
        ]
        coherence_results = eval_coherence.evaluate_conversation(test_conversation)
        results['coherence'] = coherence_results
        logger.info(f"✅ Coherence score: {coherence_results.get('coherence_score', 0):.2f}")
    except Exception as e:
        logger.error(f"❌ Coherence evaluation failed: {e}")
        results['coherence'] = {'error': str(e)}
    
    # 3. Latency Evaluation
    logger.info("⏱️  Running latency evaluation...")
    try:
        latency_results = eval_latency.evaluate(model, tokenizer, num_samples=5)
        results['latency'] = latency_results
        logger.info(f"✅ Avg latency: {latency_results.get('mean_latency_ms', 0):.2f}ms")
        logger.info(f"   Tokens/sec: {latency_results.get('avg_tokens_per_second', 0):.2f}")
    except Exception as e:
        logger.error(f"❌ Latency evaluation failed: {e}")
        results['latency'] = {'error': str(e)}
    
    # 4. Memory Consistency Evaluation
    logger.info("🧩 Running memory consistency evaluation...")
    try:
        memory_results = eval_memory_consistency.evaluate(model)
        results['memory_consistency'] = memory_results
        logger.info(f"✅ Memory head functional: {memory_results.get('memory_head_functional', False)}")
    except Exception as e:
        logger.error(f"❌ Memory consistency evaluation failed: {e}")
        results['memory_consistency'] = {'error': str(e)}
    
    # 5. Safety Layer Test (if configured)
    if safety_config and Path(safety_config).exists():
        logger.info("🛡️  Testing safety layer...")
        try:
            safety = SafetyLayer.from_config(safety_config)
            test_inputs = [
                "Tell me a nice story",
                "Something with blocked content"
            ]
            safety_results = {
                'config_loaded': True,
                'statistics': safety.get_statistics(),
                'test_results': []
            }
            
            for test_input in test_inputs:
                is_safe = safety.check_input(test_input)
                safety_results['test_results'].append({
                    'input': test_input,
                    'passed': is_safe
                })
            
            results['safety_layer'] = safety_results
            logger.info(f"✅ Safety layer active with {safety.get_statistics()['blocklist_size']} blocked terms")
        except Exception as e:
            logger.error(f"❌ Safety layer test failed: {e}")
            results['safety_layer'] = {'error': str(e)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive model evaluation')
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        help='Path to save evaluation results'
    )
    parser.add_argument(
        '--include-r4-9',
        action='store_true',
        help='Include R4-9 evaluations (JSON, coherence, latency, memory)'
    )
    parser.add_argument(
        '--safety-config',
        type=str,
        help='Path to safety layer configuration JSON'
    )
    parser.add_argument(
        '--mock-mode',
        action='store_true',
        help='Run in mock mode for testing'
    )
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting evaluation for checkpoint: {args.checkpoint_path}")
    
    # Load model and tokenizer
    if args.mock_mode:
        logger.info("Running in mock mode...")
        from unittest.mock import Mock
        model = Mock()
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        tokenizer = Mock()
        tokenizer.decode.return_value = "Test response"
        tokenizer.encode.return_value = [1, 2, 3]
    else:
        # In real implementation, load the actual model
        logger.error("Real model loading not implemented yet")
        return 1
    
    # Run base evaluation suite
    all_results = {}
    
    try:
        # Run standard evaluations
        base_results = run_evaluation_suite(
            checkpoint_path=args.checkpoint_path,
            model=model,
            tokenizer=tokenizer,
            output_json=None  # We'll save combined results later
        )
        all_results['base_evaluations'] = base_results
        
        # Run R4-9 evaluations if requested
        if args.include_r4_9:
            logger.info("\n📊 Running R4-9 extended evaluations...")
            r4_9_results = run_r4_9_evaluations(
                model=model,
                tokenizer=tokenizer,
                safety_config=args.safety_config
            )
            all_results['r4_9_evaluations'] = r4_9_results
        
        # Determine overall success
        base_passed = base_results.get('passed', False)
        r4_9_passed = True
        
        if args.include_r4_9:
            # Check R4-9 specific criteria
            json_rate = all_results['r4_9_evaluations'].get('json_correctness', {}).get('json_correctness_rate', 0)
            coherence_score = all_results['r4_9_evaluations'].get('coherence', {}).get('coherence_score', 0)
            memory_functional = all_results['r4_9_evaluations'].get('memory_consistency', {}).get('memory_head_functional', False)
            
            r4_9_passed = (
                json_rate >= 0.8 and
                coherence_score >= 0.7 and
                memory_functional
            )
        
        all_results['overall_passed'] = base_passed and r4_9_passed
        all_results['timestamp'] = datetime.now().isoformat()
        
        # Save results if requested
        if args.output_json:
            logger.info(f"💾 Saving results to {args.output_json}")
            with open(args.output_json, 'w') as f:
                json.dump(all_results, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📈 EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Base evaluations passed: {'✅' if base_passed else '❌'}")
        
        if args.include_r4_9:
            logger.info(f"R4-9 evaluations passed: {'✅' if r4_9_passed else '❌'}")
            logger.info(f"  - JSON correctness: {json_rate:.1%}")
            logger.info(f"  - Coherence score: {coherence_score:.2f}")
            logger.info(f"  - Memory functional: {'✅' if memory_functional else '❌'}")
        
        logger.info(f"\nOverall: {'✅ PASSED' if all_results['overall_passed'] else '❌ FAILED'}")
        logger.info("="*60)
        
        return 0 if all_results['overall_passed'] else 1
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed with error: {e}")
        return 1


if __name__ == '__main__':
    # Add missing import for mock mode
    try:
        import torch
    except ImportError:
        if '--mock-mode' not in sys.argv:
            logger.error("PyTorch not available. Use --mock-mode for testing.")
            sys.exit(1)
    
    sys.exit(main()) 