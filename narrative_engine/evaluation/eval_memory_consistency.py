"""
Memory Consistency Evaluation

Validates memory head outputs for the triple-head architecture.
Ensures embeddings are properly normalized and metadata is valid.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np

logger = logging.getLogger(__name__)


class MemoryConsistencyEvaluator:
    """Evaluates memory head consistency and functionality"""
    
    def __init__(self):
        self.embedding_dim = 768  # Expected embedding dimension
        self.memory_types = ['identity', 'preference', 'event', 'relationship']
        self.epsilon = 1e-6  # For numerical stability checks
        
    def evaluate(
        self,
        model: Any,
        test_inputs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate memory head outputs for consistency.
        
        Args:
            model: The model with memory head to evaluate
            test_inputs: Optional test inputs to generate memories
            
        Returns:
            Dictionary with memory consistency metrics
        """
        results = {
            'embeddings_normalized': False,
            'metadata_valid': False,
            'memory_head_functional': False,
            'embedding_statistics': {},
            'metadata_statistics': {}
        }
        
        try:
            # Get memory outputs from model
            if test_inputs is None:
                test_inputs = self._get_default_test_inputs()
            
            # Generate outputs with memory head
            memory_outputs = self._get_memory_outputs(model, test_inputs)
            
            if memory_outputs is None:
                results['error'] = 'Failed to get memory outputs'
                return results
            
            # Check embedding normalization
            embeddings = memory_outputs.get('memory_embeddings')
            if embeddings is not None:
                norm_check = self._check_embedding_normalization(embeddings)
                results['embeddings_normalized'] = norm_check['normalized']
                results['embedding_statistics'] = norm_check['statistics']
            
            # Check metadata validity
            metadata = memory_outputs.get('memory_metadata')
            if metadata is not None:
                metadata_check = self._check_metadata_validity(metadata)
                results['metadata_valid'] = metadata_check['valid']
                results['metadata_statistics'] = metadata_check['statistics']
            
            # Overall functionality check
            results['memory_head_functional'] = (
                results['embeddings_normalized'] and 
                results['metadata_valid']
            )
            
            logger.info(f"Memory consistency evaluation: functional={results['memory_head_functional']}")
            
        except Exception as e:
            logger.error(f"Error in memory consistency evaluation: {e}")
            results['error'] = str(e)
        
        return results
    
    def evaluate_memory_formation(
        self,
        model: Any,
        conversation: List[Dict[str, str]],
        expected_memories: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate memory formation accuracy from conversations.
        
        Args:
            model: The model to evaluate
            conversation: Conversation to extract memories from
            expected_memories: Optional expected memories for comparison
            
        Returns:
            Dictionary with formation accuracy metrics
        """
        results = {
            'formation_accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'extracted_memories': []
        }
        
        try:
            # Extract memories from conversation
            extracted = self._extract_memories_from_conversation(model, conversation)
            results['extracted_memories'] = extracted
            
            if expected_memories:
                # Calculate precision and recall
                metrics = self._calculate_memory_metrics(extracted, expected_memories)
                results.update(metrics)
            else:
                # Just check that memories were formed
                results['formation_accuracy'] = 1.0 if len(extracted) > 0 else 0.0
            
            logger.info(f"Memory formation evaluation: {len(extracted)} memories extracted")
            
        except Exception as e:
            logger.error(f"Error in memory formation evaluation: {e}")
            results['error'] = str(e)
        
        return results
    
    def _get_default_test_inputs(self) -> List[str]:
        """Get default test inputs for memory generation"""
        return [
            "My name is Alice and I love reading books.",
            "I visited Paris last summer and it was amazing.",
            "My best friend is Bob, we've known each other for 10 years.",
            "I prefer coffee over tea in the morning.",
            "Yesterday I learned how to play chess."
        ]
    
    def _get_memory_outputs(self, model: Any, inputs: List[str]) -> Optional[Dict[str, Any]]:
        """Get memory outputs from the model"""
        try:
            # In real implementation, this would process inputs through model
            # For now, check if model has expected output structure
            
            # Create mock inputs
            if hasattr(model, '__call__'):
                # Call model to get outputs
                mock_outputs = model()
                
                # Extract memory-related outputs
                if hasattr(mock_outputs, 'memory_embeddings'):
                    return {
                        'memory_embeddings': mock_outputs.memory_embeddings,
                        'memory_metadata': mock_outputs.memory_metadata
                    }
            
            # Return None if no memory outputs found
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get memory outputs: {e}")
            return None
    
    def _check_embedding_normalization(self, embeddings: torch.Tensor) -> Dict[str, Any]:
        """Check if embeddings are properly normalized"""
        result = {
            'normalized': False,
            'statistics': {}
        }
        
        try:
            # Convert to numpy for easier computation
            if hasattr(embeddings, 'numpy'):
                emb_array = embeddings.detach().cpu().numpy()
            else:
                emb_array = embeddings
            
            # Compute norms
            norms = np.linalg.norm(emb_array, axis=-1)
            
            # Check if normalized (norm should be ~1.0)
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            min_norm = np.min(norms)
            max_norm = np.max(norms)
            
            # Consider normalized if all norms are close to 1.0
            result['normalized'] = (
                abs(mean_norm - 1.0) < 0.01 and
                std_norm < 0.01 and
                min_norm > 0.99 and
                max_norm < 1.01
            )
            
            result['statistics'] = {
                'mean_norm': float(mean_norm),
                'std_norm': float(std_norm),
                'min_norm': float(min_norm),
                'max_norm': float(max_norm),
                'shape': list(embeddings.shape)
            }
            
        except Exception as e:
            logger.warning(f"Error checking embedding normalization: {e}")
            result['error'] = str(e)
        
        return result
    
    def _check_metadata_validity(self, metadata: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Check if memory metadata is valid"""
        result = {
            'valid': False,
            'statistics': {}
        }
        
        try:
            # Check importance scores
            if 'importance_scores' in metadata:
                scores = metadata['importance_scores']
                if hasattr(scores, 'numpy'):
                    scores_array = scores.detach().cpu().numpy()
                else:
                    scores_array = scores
                
                # Importance should be between 0 and 1
                scores_valid = (
                    np.all(scores_array >= 0) and 
                    np.all(scores_array <= 1)
                )
                
                result['statistics']['importance'] = {
                    'mean': float(np.mean(scores_array)),
                    'std': float(np.std(scores_array)),
                    'min': float(np.min(scores_array)),
                    'max': float(np.max(scores_array))
                }
            else:
                scores_valid = False
            
            # Check memory types
            if 'memory_types' in metadata:
                types = metadata['memory_types']
                if hasattr(types, 'numpy'):
                    types_array = types.detach().cpu().numpy()
                else:
                    types_array = types
                
                # Types should be valid indices
                max_type = len(self.memory_types) - 1
                types_valid = (
                    np.all(types_array >= 0) and 
                    np.all(types_array <= max_type)
                )
                
                # Count distribution
                unique, counts = np.unique(types_array, return_counts=True)
                type_dist = {int(t): int(c) for t, c in zip(unique, counts)}
                result['statistics']['type_distribution'] = type_dist
            else:
                types_valid = False
            
            result['valid'] = scores_valid and types_valid
            
        except Exception as e:
            logger.warning(f"Error checking metadata validity: {e}")
            result['error'] = str(e)
        
        return result
    
    def _extract_memories_from_conversation(
        self,
        model: Any,
        conversation: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Extract memories from a conversation"""
        # This is a simplified implementation
        # In reality, would use the model's memory extraction capabilities
        
        memories = []
        
        for turn in conversation:
            if turn.get('role') == 'user':
                content = turn.get('content', '').lower()
                
                # Simple heuristics for memory extraction
                if 'my name is' in content:
                    memories.append({
                        'content': f"User's name is {content.split('my name is')[1].split()[0]}",
                        'importance': 0.9,
                        'type': 'identity'
                    })
                
                if 'love' in content or 'favorite' in content:
                    memories.append({
                        'content': content,
                        'importance': 0.7,
                        'type': 'preference'
                    })
                
                if 'friend' in content:
                    memories.append({
                        'content': content,
                        'importance': 0.8,
                        'type': 'relationship'
                    })
        
        return memories
    
    def _calculate_memory_metrics(
        self,
        extracted: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate precision and recall for memory extraction"""
        if not expected:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'formation_accuracy': 0.0
            }
        
        # Simple matching based on content similarity
        matches = 0
        for ext_mem in extracted:
            for exp_mem in expected:
                # Check if memories are similar (simplified)
                if (ext_mem.get('type') == exp_mem.get('type') and
                    any(word in ext_mem.get('content', '').lower() 
                        for word in exp_mem.get('content', '').lower().split())):
                    matches += 1
                    break
        
        precision = matches / len(extracted) if extracted else 0.0
        recall = matches / len(expected) if expected else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'formation_accuracy': f1
        } 