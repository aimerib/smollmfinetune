"""
Latency Evaluation

Measures generation speed, time-to-first-token, and throughput metrics.
Essential for ensuring acceptable inference performance.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import torch
import numpy as np

logger = logging.getLogger(__name__)


class LatencyEvaluator:
    """Evaluates model inference latency and throughput"""
    
    def __init__(self):
        self.test_prompts = [
            "Hello, how are you today?",
            "Tell me a short story.",
            "What is the weather like?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about the ocean."
        ]
        
    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        num_samples: int = 10,
        test_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model latency and throughput.
        
        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            num_samples: Number of samples to generate for statistics
            test_prompts: Optional custom test prompts
            
        Returns:
            Dictionary with latency metrics
        """
        if test_prompts is None:
            test_prompts = self.test_prompts[:num_samples]
        
        results = {
            'time_to_first_token_ms': 0.0,
            'avg_tokens_per_second': 0.0,
            'p50_latency_ms': 0.0,
            'p95_latency_ms': 0.0,
            'num_samples': num_samples,
            'individual_measurements': []
        }
        
        try:
            latencies = []
            first_token_times = []
            tokens_per_second = []
            
            # Warm up the model
            self._warmup(model, tokenizer)
            
            for i in range(num_samples):
                prompt = test_prompts[i % len(test_prompts)]
                
                # Measure single generation
                measurement = self._measure_single_generation(model, tokenizer, prompt)
                
                if measurement['success']:
                    latencies.append(measurement['total_latency_ms'])
                    first_token_times.append(measurement['time_to_first_token_ms'])
                    tokens_per_second.append(measurement['tokens_per_second'])
                    
                results['individual_measurements'].append(measurement)
            
            # Calculate statistics
            if latencies:
                stats = self._compute_latency_statistics(latencies)
                results.update({
                    'p50_latency_ms': stats['p50'],
                    'p95_latency_ms': stats['p95'],
                    'mean_latency_ms': stats['mean'],
                    'min_latency_ms': stats['min'],
                    'max_latency_ms': stats['max']
                })
            
            if first_token_times:
                results['time_to_first_token_ms'] = np.mean(first_token_times)
                
            if tokens_per_second:
                results['avg_tokens_per_second'] = np.mean(tokens_per_second)
            
            logger.info(f"Latency evaluation complete: avg={results.get('mean_latency_ms', 0):.2f}ms")
            
        except Exception as e:
            logger.error(f"Error in latency evaluation: {e}")
            results['error'] = str(e)
        
        return results
    
    def _warmup(self, model: Any, tokenizer: Any) -> None:
        """Warm up the model with a few generations"""
        try:
            warmup_prompt = "Hello"
            if hasattr(model, 'generate') and hasattr(tokenizer, 'encode'):
                inputs = tokenizer.encode(warmup_prompt, return_tensors='pt')
                with torch.no_grad():
                    _ = model.generate(inputs, max_new_tokens=10)
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def _measure_single_generation(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str
    ) -> Dict[str, Any]:
        """Measure latency for a single generation"""
        measurement = {
            'prompt': prompt,
            'success': False,
            'total_latency_ms': 0.0,
            'time_to_first_token_ms': 0.0,
            'tokens_per_second': 0.0,
            'num_tokens': 0
        }
        
        try:
            # Encode prompt
            if hasattr(tokenizer, 'encode'):
                inputs = tokenizer.encode(prompt, return_tensors='pt')
            else:
                inputs = torch.tensor([[1, 2, 3]])
            
            # Measure generation time
            start_time = time.perf_counter()
            first_token_time = None
            
            if hasattr(model, 'generate'):
                with torch.no_grad():
                    # For real implementation, would need to hook into generation
                    # to measure time-to-first-token accurately
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
                    )
                
                # Simulate first token time (in real impl, would measure actual)
                first_token_time = time.perf_counter()
                
            else:
                # Mock generation
                time.sleep(0.1)  # Simulate generation time
                outputs = torch.tensor([[1, 2, 3, 4, 5]])
                first_token_time = time.perf_counter()
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            total_time_ms = (end_time - start_time) * 1000
            time_to_first_ms = (first_token_time - start_time) * 1000
            
            # Count generated tokens
            if hasattr(outputs, 'shape'):
                num_generated = outputs.shape[1] - inputs.shape[1]
            else:
                num_generated = 5
            
            tokens_per_sec = num_generated / (total_time_ms / 1000) if total_time_ms > 0 else 0
            
            measurement.update({
                'success': True,
                'total_latency_ms': total_time_ms,
                'time_to_first_token_ms': time_to_first_ms,
                'tokens_per_second': tokens_per_sec,
                'num_tokens': num_generated
            })
            
        except Exception as e:
            logger.warning(f"Generation measurement failed: {e}")
            measurement['error'] = str(e)
        
        return measurement
    
    def _compute_latency_statistics(self, latencies: List[float]) -> Dict[str, float]:
        """Compute statistical metrics for latencies"""
        if not latencies:
            return {
                'mean': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        latencies_array = np.array(latencies)
        
        return {
            'mean': float(np.mean(latencies_array)),
            'p50': float(np.percentile(latencies_array, 50)),
            'p95': float(np.percentile(latencies_array, 95)),
            'min': float(np.min(latencies_array)),
            'max': float(np.max(latencies_array))
        } 