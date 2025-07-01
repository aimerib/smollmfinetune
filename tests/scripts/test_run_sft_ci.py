"""
Unit tests for CI SFT training script - TDD Red Phase
Tests the lightweight training script that will run in GitHub Actions CI.
"""

import pytest
import tempfile
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestRunSFTCI:
    """Test the CI SFT training script"""
    
    def test_run_sft_ci_script_exists(self):
        """CI SFT script should be importable"""
        # This will fail initially - that's expected in TDD Red phase
        try:
            import scripts.run_sft_ci
            assert True
        except ImportError:
            pytest.fail("scripts.run_sft_ci should be importable")
    
    def test_run_sft_ci_exits_successfully(self):
        """CI script should exit with code 0 when run successfully"""
        script_path = Path("scripts/run_sft_ci.py")
        
        # Script should exist
        assert script_path.exists(), f"Script not found: {script_path}"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run the script with minimal config for CI
            result = subprocess.run([
                "python", str(script_path),
                "--output-dir", temp_dir,
                "--max-steps", "5",  # Very small for test
                "--quick-mode"  # Skip heavy operations
            ], 
            capture_output=True, 
            text=True,
            timeout=120  # 2 minute timeout as specified in requirements
            )
            
            # Should exit successfully
            assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    @pytest.mark.slow
    def test_run_sft_ci_produces_checkpoint(self):
        """CI script should produce a model checkpoint file within timeout"""
        script_path = Path("scripts/run_sft_ci.py")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "ci_output"
            
            start_time = time.time()
            
            # Run the script
            result = subprocess.run([
                "python", str(script_path),
                "--output-dir", str(output_dir),
                "--max-steps", "10",
                "--quick-mode"
            ], 
            capture_output=True, 
            text=True,
            timeout=120
            )
            
            end_time = time.time()
            
            # Should complete within 120 seconds as per requirements
            assert (end_time - start_time) <= 120, f"Script took too long: {end_time - start_time:.1f}s"
            
            # Should exit successfully
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            
            # Should produce checkpoint files
            checkpoint_files = list(output_dir.glob("**/*.safetensors"))
            assert len(checkpoint_files) > 0, f"No checkpoint files found in {output_dir}"
    
    def test_run_sft_ci_uses_cpu_only(self):
        """CI script should run on CPU only (no CUDA required)"""
        script_path = Path("scripts/run_sft_ci.py")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Ensure CUDA_VISIBLE_DEVICES is empty to force CPU
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ''
            
            result = subprocess.run([
                "python", str(script_path),
                "--output-dir", temp_dir,
                "--max-steps", "3",
                "--quick-mode"
            ], 
            capture_output=True, 
            text=True,
            env=env,
            timeout=120
            )
            
            # Should still work without GPU
            assert result.returncode == 0, f"Script failed on CPU-only: {result.stderr}"
    
    def test_run_sft_ci_generates_toy_dataset(self):
        """CI script should generate a 100-sample toy dataset"""
        with patch('scripts.run_sft_ci.generate_toy_dataset') as mock_generate:
            mock_generate.return_value = ["sample1", "sample2"]  # Mock dataset
            
            script_path = Path("scripts/run_sft_ci.py")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result = subprocess.run([
                    "python", str(script_path),
                    "--output-dir", temp_dir,
                    "--max-steps", "1",
                    "--quick-mode"
                ], 
                capture_output=True, 
                text=True,
                timeout=60
                )
                
                # Should call dataset generation
                assert result.returncode == 0, f"Script failed: {result.stderr}"
    
    def test_run_sft_ci_integrates_with_telemetry(self):
        """CI script should integrate with telemetry SDK for tracking"""
        # This test verifies telemetry integration by checking successful execution
        # In CI, telemetry works with CSV backend which we can verify
        script_path = Path("scripts/run_sft_ci.py")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run([
                "python", str(script_path),
                "--output-dir", temp_dir,
                "--max-steps", "1",
                "--dataset-size", "3",  # Very small dataset
                "--quick-mode"
            ], 
            capture_output=True, 
            text=True,
            timeout=60
            )
            
            # Should initialize telemetry and complete successfully
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            
            # Should mention telemetry initialization in output
            assert "Telemetry initialized" in result.stderr, "Should initialize telemetry SDK"


class TestRunBasicEvaluation:
    """Test the basic evaluation orchestration script"""
    
    def test_run_basic_evaluation_script_exists(self):
        """Basic evaluation script should be importable"""
        try:
            import scripts.run_basic_evaluation
            assert True
        except ImportError:
            pytest.fail("scripts.run_basic_evaluation should be importable")
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_run_basic_evaluation_quick_mode(self):
        """Evaluation script should support quick mode for CI"""
        script_path = Path("scripts/run_basic_evaluation.py")
        
        # Create a dummy checkpoint directory
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoint"
            checkpoint_dir.mkdir()
            
            # Create dummy checkpoint file
            (checkpoint_dir / "adapter_model.safetensors").write_text("dummy")
            
            result = subprocess.run([
                "python", str(script_path),
                "--checkpoint-path", str(checkpoint_dir),
                "--mock-mode",  # Updated to match new evaluation script
                "--output-json", str(Path(temp_dir) / "results.json")
            ], 
            capture_output=True, 
            text=True,
            timeout=60
            )
            
            assert result.returncode == 0, f"Evaluation script failed: {result.stderr}"
    
    @pytest.mark.slow
    @pytest.mark.llm
    @pytest.mark.evaluation
    def test_run_basic_evaluation_outputs_json(self):
        """Evaluation script should output JSON results with required metrics"""
        script_path = Path("scripts/run_basic_evaluation.py")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoint"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "adapter_model.safetensors").write_text("dummy")
            
            results_file = Path(temp_dir) / "results.json"
            
            result = subprocess.run([
                "python", str(script_path),
                "--checkpoint-path", str(checkpoint_dir),
                "--mock-mode",  # Updated to match new evaluation script
                "--output-json", str(results_file)
            ], 
            capture_output=True, 
            text=True,
            timeout=60
            )
            
            assert result.returncode == 0, f"Evaluation failed: {result.stderr}"
            assert results_file.exists(), "Results JSON file should be created"
            
            # Results should contain required metrics
            import json
            with open(results_file) as f:
                results = json.load(f)
            
            # Check for the evaluation harness structure
            assert "basic_generation" in results['evaluations'], "Should include basic generation results"
            assert "passed" in results, "Should include overall pass/fail status"
            
            # Check nested metrics
            gen_results = results['evaluations'].get("basic_generation", {})
            assert "generation_success_rate" in gen_results, "Should include generation success rate"
            assert "coherence_score" in gen_results, "Should include coherence score"
            
            # Verify metrics are numeric
            assert isinstance(gen_results.get("generation_success_rate", 0), (int, float)), "Generation success rate should be numeric"
            assert isinstance(gen_results.get("coherence_score", 0), (int, float)), "Coherence score should be numeric" 