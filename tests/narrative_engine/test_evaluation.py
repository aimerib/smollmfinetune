"""
Tests for Narrative Engine evaluation harness
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

# Test that evaluation module exists and has expected structure
def test_evaluation_module_structure():
    """Test that the evaluation module has the expected structure"""
    from backend.app.narrative_engine.evaluation import (
        eval_basic_generation,
        eval_dual_head_sanity,
        eval_training_progress,
        run_evaluation_suite
    )
    
    # Check that modules exist
    assert eval_basic_generation is not None
    assert eval_dual_head_sanity is not None
    assert eval_training_progress is not None
    assert run_evaluation_suite is not None

@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.evaluation
def test_basic_generation_evaluation():
    """Test basic text generation evaluation"""
    from backend.app.narrative_engine.evaluation.eval_basic_generation import BasicGenerationEvaluator
    
    evaluator = BasicGenerationEvaluator()
    
    # Mock model that generates coherent text
    mock_model = Mock()
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    
    mock_tokenizer = Mock()
    mock_tokenizer.decode.return_value = "This is a coherent response to the prompt."
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.pad_token_id = 0
    
    # Test prompts
    test_prompts = [
        "Tell me a story about a dragon",
        "What is your favorite color?",
        "Describe a sunny day"
    ]
    
    results = evaluator.evaluate(mock_model, mock_tokenizer, test_prompts)
    
    # Check results structure
    assert 'coherence_score' in results
    assert 'generation_success_rate' in results
    assert 'avg_response_length' in results
    assert 'individual_scores' in results
    
    # Check that scores are in valid range
    assert 0.0 <= results['coherence_score'] <= 1.0
    assert 0.0 <= results['generation_success_rate'] <= 1.0
    assert results['avg_response_length'] > 0


def test_dual_head_sanity_evaluation():
    """Test dual-head output validation"""
    from backend.app.narrative_engine.evaluation.eval_dual_head_sanity import DualHeadSanityEvaluator
    
    evaluator = DualHeadSanityEvaluator()
    
    # Mock model with dual outputs
    mock_model = Mock()
    mock_outputs = Mock()
    mock_outputs.generation_logits = torch.randn(1, 10, 32000)  # batch, seq, vocab
    mock_outputs.control_logits = torch.randn(1, 10, 64)  # batch, seq, control_vocab
    mock_model.return_value = mock_outputs
    
    results = evaluator.evaluate(mock_model)
    
    # Check results
    assert 'generation_head_valid' in results
    assert 'control_head_valid' in results
    assert 'logits_statistics' in results
    assert 'both_heads_functional' in results
    
    # Both heads should be valid for our mock
    assert results['generation_head_valid'] is True
    assert results['control_head_valid'] is True
    assert results['both_heads_functional'] is True


def test_training_progress_evaluation():
    """Test training progress tracking"""
    from backend.app.narrative_engine.evaluation.eval_training_progress import TrainingProgressEvaluator
    
    evaluator = TrainingProgressEvaluator()
    
    # Mock training history
    mock_history = {
        'loss': [2.5, 2.0, 1.8, 1.5, 1.3],
        'generation_loss': [2.0, 1.8, 1.6, 1.4, 1.2],
        'control_loss': [0.5, 0.2, 0.2, 0.1, 0.1],
        'steps': [0, 100, 200, 300, 400]
    }
    
    results = evaluator.evaluate(mock_history)
    
    # Check results
    assert 'loss_decreasing' in results
    assert 'convergence_rate' in results
    assert 'final_loss' in results
    assert 'loss_variance' in results
    
    # Loss should be decreasing for our mock data
    assert results['loss_decreasing'] is True
    assert results['final_loss'] == 1.3


def test_evaluation_suite_integration():
    """Test the main evaluation suite runner"""
    from backend.app.narrative_engine.evaluation import run_evaluation_suite
    
    # Mock model and tokenizer
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    
    # Mock dual head outputs
    mock_outputs = Mock()
    mock_outputs.generation_logits = torch.randn(1, 10, 32000)
    mock_outputs.control_logits = torch.randn(1, 10, 64)
    mock_model.return_value = mock_outputs
    
    mock_tokenizer = Mock()
    mock_tokenizer.decode.return_value = "Test response"
    mock_tokenizer.encode.return_value = [1, 2, 3]
    
    # Mock checkpoint path
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint-100"
        checkpoint_path.mkdir()
        
        # Mock training history
        history = {
            'loss': [2.0, 1.5, 1.0],
            'steps': [0, 50, 100]
        }
        
        results = run_evaluation_suite(
            checkpoint_path=str(checkpoint_path),
            model=mock_model,
            tokenizer=mock_tokenizer,
            training_history=history
        )
        
        # Check overall results structure
        assert 'timestamp' in results
        assert 'checkpoint_path' in results
        assert 'basic_generation' in results['evaluations']
        assert 'triple_head_sanity' in results['evaluations']
        assert 'training_progress' in results['evaluations']
        assert 'passed' in results
        assert 'duration' in results


def test_wandb_logging_integration():
    """Test that evaluation results are logged to wandb"""
    from backend.app.narrative_engine.evaluation import run_evaluation_suite
    
    with patch('wandb.log') as mock_wandb_log:
        # Mock components
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Run evaluation
        results = run_evaluation_suite(
            checkpoint_path="/path/to/checkpoint",
            model=mock_model,
            tokenizer=mock_tokenizer,
            log_to_wandb=True
        )
        
        # Check wandb was called
        assert mock_wandb_log.called
        
        # Check logged data structure
        logged_data = mock_wandb_log.call_args[0][0]
        assert 'eval/generation_success_rate' in logged_data
        assert 'eval/avg_generation_length' in logged_data
        assert 'eval/perplexity' in logged_data
        assert 'eval/all_heads_functional' in logged_data
        assert 'eval/generation_head_valid' in logged_data
        assert 'eval/control_head_valid' in logged_data
        assert 'eval/memory_head_valid' in logged_data



def test_json_output_format():
    """Test that results are saved in correct JSON format"""
    from backend.app.narrative_engine.evaluation import run_evaluation_suite
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "eval_results.json"
        
        # Mock components
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Run evaluation with JSON output
        results = run_evaluation_suite(
            checkpoint_path="/path/to/checkpoint",
            model=mock_model,
            tokenizer=mock_tokenizer,
            output_json=str(output_path)
        )
        
        # Check JSON file was created
        assert output_path.exists()
        
        # Load and validate JSON
        with open(output_path) as f:
            loaded_results = json.load(f)
        
        assert loaded_results['checkpoint_path'] == "/path/to/checkpoint"
        assert 'timestamp' in loaded_results
        assert 'basic_generation' in loaded_results['evaluations']


def test_evaluation_failure_handling():
    """Test that evaluation handles failures gracefully"""
    from backend.app.narrative_engine.evaluation import run_evaluation_suite
    
    # Mock model that raises exception
    mock_model = Mock()
    mock_model.generate.side_effect = RuntimeError("CUDA out of memory")
    
    mock_tokenizer = Mock()
    
    results = run_evaluation_suite(
        checkpoint_path="/path/to/checkpoint",
        model=mock_model,
        tokenizer=mock_tokenizer
    )
    
    # Should not crash, but mark as failed
    assert results['passed'] is False
    # The evaluation should fail due to 0% generation success rate
    assert results.get('basic_generation', {}).get('generation_success_rate', 0.0) == 0.0
    # Or have a failure reason
    assert 'failure_reason' in results or results.get('basic_generation', {}).get('generation_success_rate', 0.0) < 0.5


def test_checkpoint_blocking_criteria():
    """Test that bad checkpoints are properly identified"""
    from backend.app.narrative_engine.evaluation import run_evaluation_suite
    
    # Mock a model that generates garbage
    mock_model = Mock()
    mock_model.generate.return_value = torch.tensor([[0, 0, 0, 0, 0]])  # All padding
    
    mock_tokenizer = Mock()
    mock_tokenizer.decode.return_value = ""  # Empty generation
    
    results = run_evaluation_suite(
        checkpoint_path="/path/to/checkpoint",
        model=mock_model,
        tokenizer=mock_tokenizer
    )
    
    # Should fail due to bad generation
    assert results['passed'] is False
    assert results['evaluations']['basic_generation']['generation_success_rate'] == 0.0


@pytest.mark.slow
@pytest.mark.llm
@pytest.mark.integration
def test_evaluation_script_cli():
    """Test the command-line script interface"""
    import subprocess
    import sys
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "results.json"
        
        # Run the script (would need actual checkpoint for real test)
        result = subprocess.run([
            sys.executable,
            "scripts/run_basic_evaluation.py",
            "--checkpoint-path", "/mock/checkpoint",
            "--output-json", str(output_json),
            "--mock-mode"  # Flag for testing without real model
        ], capture_output=True, text=True)
        
        # In mock mode, should succeed
        assert result.returncode == 0
        assert output_json.exists() 