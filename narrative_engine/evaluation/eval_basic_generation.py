"""
Basic Generation Evaluation

Tests that the model can generate coherent text in response to prompts.
Focuses on catching obvious failures like empty outputs or repetitive text.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np

logger = logging.getLogger(__name__)


class BasicGenerationEvaluator:
    """Evaluates basic text generation capabilities"""
    
    def __init__(self):
        self.test_prompts = [
            "Tell me a story about a brave knight.",
            "What is your favorite color and why?",
            "Describe a peaceful day in the countryside.",
            "Explain how to make a simple sandwich.",
            "Write a haiku about the ocean.",
        ]
        
    def evaluate(
        self, 
        model: Any, 
        tokenizer: Any,
        test_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate basic generation capabilities.
        
        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            test_prompts: Optional custom test prompts
            
        Returns:
            Dictionary with evaluation metrics
        """
        if test_prompts is None:
            test_prompts = self.test_prompts
            
        results = {
            'coherence_score': 0.0,
            'generation_success_rate': 0.0,
            'avg_response_length': 0.0,
            'individual_scores': []
        }
        
        try:
            successful_generations = 0
            total_length = 0
            coherence_scores = []
            
            for prompt in test_prompts:
                # Generate response
                response = self._generate_response(model, tokenizer, prompt)
                
                # Evaluate response
                eval_result = self._evaluate_single_response(response, prompt)
                results['individual_scores'].append(eval_result)
                
                if eval_result['success']:
                    successful_generations += 1
                    total_length += eval_result['length']
                    coherence_scores.append(eval_result['coherence'])
                    
            # Calculate aggregate metrics
            num_prompts = len(test_prompts)
            results['generation_success_rate'] = successful_generations / num_prompts if num_prompts > 0 else 0
            results['avg_response_length'] = total_length / successful_generations if successful_generations > 0 else 0
            results['coherence_score'] = np.mean(coherence_scores) if coherence_scores else 0.0
            
            logger.info(f"Basic generation evaluation complete: {successful_generations}/{num_prompts} successful")
            
        except Exception as e:
            logger.error(f"Error in basic generation evaluation: {e}")
            results['error'] = str(e)
            
        return results
    
    def _generate_response(self, model: Any, tokenizer: Any, prompt: str) -> str:
        """Generate a response for a single prompt"""
        try:
            # This is simplified - actual implementation would handle batching, device placement, etc.
            if hasattr(model, 'generate'):
                # Encode prompt
                inputs = tokenizer.encode(prompt, return_tensors='pt') if hasattr(tokenizer, 'encode') else torch.tensor([[1, 2, 3]])
                
                # Generate with reasonable parameters
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
                    )
                
                # Decode response
                if hasattr(tokenizer, 'decode'):
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Remove the prompt from response
                    response = response[len(prompt):].strip()
                else:
                    response = "This is a coherent response to the prompt."
                    
                return response
            else:
                # Mock response for testing
                return "This is a coherent response to the prompt."
                
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""
    
    def _evaluate_single_response(self, response: str, prompt: str) -> Dict[str, Any]:
        """Evaluate a single generated response"""
        result = {
            'prompt': prompt,
            'response': response,
            'success': False,
            'length': 0,
            'coherence': 0.0,
            'issues': []
        }
        
        # Check if response is empty
        if not response or len(response.strip()) == 0:
            result['issues'].append('Empty response')
            return result
            
        # Check response length
        result['length'] = len(response.split())
        if result['length'] < 3:
            result['issues'].append('Response too short')
            return result
            
        # Check for repetition
        repetition_score = self._check_repetition(response)
        if repetition_score > 0.5:
            result['issues'].append('High repetition detected')
            result['coherence'] = 1.0 - repetition_score
        else:
            result['coherence'] = 1.0 - repetition_score
            
        # Check for common failure patterns
        failure_patterns = [
            '�',  # Unicode errors
            '<|endoftext|>',  # Exposed special tokens
            'None',  # Null responses
            'Error:',  # Error messages
        ]
        
        for pattern in failure_patterns:
            if pattern in response:
                result['issues'].append(f'Contains failure pattern: {pattern}')
                result['coherence'] *= 0.5
                
        # Mark as successful if no critical issues
        if result['length'] >= 3 and result['coherence'] > 0.3:
            result['success'] = True
            
        return result
    
    def _check_repetition(self, text: str) -> float:
        """
        Check for repetitive patterns in text.
        Returns a score from 0 (no repetition) to 1 (highly repetitive).
        """
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
            
        # Check for repeated sequences
        repetition_count = 0
        for i in range(len(words) - 3):
            sequence = tuple(words[i:i+3])
            # Count how many times this 3-word sequence appears
            occurrences = sum(1 for j in range(len(words) - 2) 
                            if tuple(words[j:j+3]) == sequence)
            if occurrences > 2:
                repetition_count += occurrences - 1
                
        # Normalize by text length
        repetition_score = repetition_count / len(words)
        return min(repetition_score, 1.0) 