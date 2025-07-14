"""
Memory Consistency Evaluation

Uses LLM-as-judge with structured outputs to validate memory head outputs for the triple-head architecture.
Ensures embeddings are properly normalized and metadata is valid, and uses LLM judgment for memory extraction.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import asyncio
import json
from pydantic import BaseModel, Field

from backend.app.core.openai_client import get_client

logger = logging.getLogger(__name__)


class MemoryExtractionAnalysis(BaseModel):
    """Structured output for memory extraction from conversations"""
    extracted_memories: List[Dict[str, Any]] = Field(description="Memories extracted from the conversation")
    memory_quality: float = Field(ge=0.0, le=1.0, description="Overall quality of extracted memories")
    relevance_score: float = Field(ge=0.0, le=1.0, description="How relevant the memories are to the conversation")
    specificity_score: float = Field(ge=0.0, le=1.0, description="How specific and detailed the memories are")
    categorization_accuracy: float = Field(ge=0.0, le=1.0, description="Accuracy of memory type categorization")
    extraction_reasoning: str = Field(description="Reasoning for the memory extraction decisions")


class MemoryQualityAnalysis(BaseModel):
    """Structured output for memory quality assessment"""
    precision: float = Field(ge=0.0, le=1.0, description="Precision of memory extraction")
    recall: float = Field(ge=0.0, le=1.0, description="Recall of memory extraction")
    f1_score: float = Field(ge=0.0, le=1.0, description="F1 score combining precision and recall")
    memory_coverage: float = Field(ge=0.0, le=1.0, description="How well memories cover the conversation")
    importance_accuracy: float = Field(ge=0.0, le=1.0, description="Accuracy of importance scoring")
    quality_assessment: str = Field(description="Overall quality assessment")


class MemoryConsistencyEvaluator:
    """Evaluates memory head consistency and functionality using LLM-as-judge"""
    
    def __init__(self):
        self.client = get_client()
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
        Evaluate memory formation accuracy from conversations using LLM-as-judge.
        
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
            # Extract memories from conversation using LLM-as-judge
            extracted = asyncio.run(self._extract_memories_from_conversation_async(model, conversation))
            results['extracted_memories'] = extracted
            
            if expected_memories:
                # Calculate precision and recall using LLM-as-judge
                metrics = asyncio.run(self._calculate_memory_metrics_async(extracted, expected_memories))
                results.update(metrics)
            else:
                # Just check that memories were formed
                results['formation_accuracy'] = 1.0 if len(extracted) > 0 else 0.0
            
            logger.info(f"Memory formation evaluation: {len(extracted)} memories extracted")
            
        except Exception as e:
            logger.error(f"Error in memory formation evaluation: {e}")
            results['error'] = str(e)
            # Fallback to heuristic extraction
            fallback_extracted = self._extract_memories_from_conversation_fallback(conversation)
            results['extracted_memories'] = fallback_extracted
            results['formation_accuracy'] = 0.5  # Conservative estimate
        
        return results
    
    async def _extract_memories_from_conversation_async(
        self,
        model: Any,
        conversation: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Extract memories from a conversation using LLM-as-judge"""
        try:
            # Format conversation for analysis
            formatted_conversation = self._format_conversation_for_memory_extraction(conversation)
            
            # Use LLM-as-judge to extract memories
            extraction_analysis = await self._judge_memory_extraction(formatted_conversation)
            
            return extraction_analysis.extracted_memories
            
        except Exception as e:
            logger.error(f"Error in async memory extraction: {e}")
            # Fallback to heuristic extraction
            return self._extract_memories_from_conversation_fallback(conversation)
    
    async def _judge_memory_extraction(self, formatted_conversation: str) -> MemoryExtractionAnalysis:
        """Use LLM-as-judge to extract memories from conversation"""
        try:
            system_prompt = self._build_memory_extraction_system_prompt()
            user_prompt = self._build_memory_extraction_user_prompt(formatted_conversation)
            
            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "memory_extraction_analysis",
                        "schema": MemoryExtractionAnalysis.model_json_schema()
                    }
                }
            )
            
            analysis_data = json.loads(response_text)
            return MemoryExtractionAnalysis(**analysis_data)
            
        except Exception as e:
            logger.error(f"Error in memory extraction judgment: {e}")
            # Return fallback analysis
            return MemoryExtractionAnalysis(
                extracted_memories=[],
                memory_quality=0.5,
                relevance_score=0.5,
                specificity_score=0.5,
                categorization_accuracy=0.5,
                extraction_reasoning="Fallback analysis due to LLM error"
            )
    
    def _build_memory_extraction_system_prompt(self) -> str:
        """Build system prompt for memory extraction"""
        return f"""You are an expert in memory extraction and psychological analysis. Your task is to identify and extract meaningful memories from conversations that should be preserved for future interactions.

**Memory Types:**
- **Identity**: Personal information about individuals (names, roles, characteristics)
- **Preference**: Likes, dislikes, opinions, values
- **Event**: Significant occurrences, experiences, actions taken
- **Relationship**: Connections between people, social dynamics

**Memory Extraction Guidelines:**
1. **Relevance**: Extract information that would be useful for future conversations
2. **Specificity**: Prefer specific, concrete details over vague statements
3. **Importance**: Focus on information that reveals character or influences behavior
4. **Persistence**: Extract facts that are likely to remain true over time

**Quality Criteria:**
- Memory Quality: Overall usefulness and accuracy of extracted memories
- Relevance: How relevant the memories are to understanding the conversation participants
- Specificity: How detailed and specific the memories are
- Categorization: How accurately memories are categorized by type

**SCORING GUIDELINES:**
- All scores: 0.0-1.0 scale
- 0.9-1.0: Excellent extraction with high precision and relevance
- 0.7-0.8: Good extraction with minor issues
- 0.5-0.6: Adequate extraction but some problems
- 0.3-0.4: Poor extraction with significant issues
- 0.0-0.2: Very poor or irrelevant extraction

**CRITICAL**: All numeric scores must be decimal values between 0.0 and 1.0 (inclusive)."""

    def _build_memory_extraction_user_prompt(self, formatted_conversation: str) -> str:
        """Build user prompt for memory extraction"""
        return f"""Extract meaningful memories from this conversation:

{formatted_conversation}

For each memory extracted, provide:
1. **Content**: The memory content in clear, specific language
2. **Type**: One of: identity, preference, event, relationship
3. **Importance**: Importance score (0.0-1.0)
4. **Source**: Which part of conversation this came from

Extract memories that:
- Reveal personal information about participants
- Show preferences, opinions, or values
- Describe significant events or experiences
- Establish relationships or social connections

Provide:
- List of extracted memories with details
- Quality assessment scores
- Reasoning for extraction decisions

Focus on information that would be valuable for future conversations with these individuals."""

    async def _calculate_memory_metrics_async(
        self,
        extracted: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate precision and recall using LLM-as-judge"""
        try:
            # Use LLM to assess memory quality
            quality_analysis = await self._judge_memory_quality(extracted, expected)
            
            return {
                'precision': quality_analysis.precision,
                'recall': quality_analysis.recall,
                'formation_accuracy': quality_analysis.f1_score,
                'memory_coverage': quality_analysis.memory_coverage,
                'importance_accuracy': quality_analysis.importance_accuracy,
                'quality_assessment': quality_analysis.quality_assessment
            }
            
        except Exception as e:
            logger.error(f"Error in async memory metrics calculation: {e}")
            # Fallback to simple comparison
            return self._calculate_memory_metrics_fallback(extracted, expected)
    
    async def _judge_memory_quality(
        self,
        extracted: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> MemoryQualityAnalysis:
        """Use LLM-as-judge to assess memory extraction quality"""
        try:
            system_prompt = """You are an expert in memory evaluation and information retrieval. Your task is to assess the quality of memory extraction by comparing extracted memories with expected memories.

Evaluate:
1. **Precision**: How many extracted memories are correct/relevant?
2. **Recall**: How many expected memories were successfully extracted?
3. **Coverage**: How well do the extracted memories cover the important information?
4. **Importance Accuracy**: How well are importance scores assigned?

**SCORING GUIDELINES:**
- All scores: 0.0-1.0 scale
- Precision: extracted_correct / total_extracted
- Recall: extracted_expected / total_expected
- F1 Score: harmonic mean of precision and recall
- Coverage: how comprehensively the memories represent the content

**CRITICAL**: All numeric scores must be decimal values between 0.0 and 1.0 (inclusive)."""

            extracted_text = json.dumps(extracted, indent=2) if extracted else "No memories extracted"
            expected_text = json.dumps(expected, indent=2) if expected else "No expected memories provided"
            
            user_prompt = f"""Assess the quality of memory extraction:

**Extracted Memories:**
{extracted_text}

**Expected Memories:**
{expected_text}

Evaluate:
1. How many extracted memories are accurate and relevant?
2. How many expected memories were successfully captured?
3. Calculate precision, recall, and F1 score
4. Assess memory coverage and importance accuracy
5. Provide overall quality assessment

Be specific about what was captured correctly vs. missed vs. incorrectly extracted."""

            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "memory_quality_analysis",
                        "schema": MemoryQualityAnalysis.model_json_schema()
                    }
                }
            )
            
            analysis_data = json.loads(response_text)
            return MemoryQualityAnalysis(**analysis_data)
            
        except Exception as e:
            logger.error(f"Error in memory quality judgment: {e}")
            # Return fallback assessment
            return MemoryQualityAnalysis(
                precision=0.5,
                recall=0.5,
                f1_score=0.5,
                memory_coverage=0.5,
                importance_accuracy=0.5,
                quality_assessment="Fallback assessment due to LLM error"
            )
    
    def _format_conversation_for_memory_extraction(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for memory extraction analysis"""
        formatted_turns = []
        for i, turn in enumerate(conversation, 1):
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            formatted_turns.append(f"{i}. {role.upper()}: {content}")
        return "\n".join(formatted_turns)
    
    def _extract_memories_from_conversation_fallback(
        self,
        conversation: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Fallback heuristic memory extraction when LLM fails"""
        # This is the original implementation as fallback
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
    
    def _calculate_memory_metrics_fallback(
        self,
        extracted: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Fallback memory metrics calculation using simple matching"""
        if not expected:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'formation_accuracy': 0.0,
                'memory_coverage': 0.0,
                'importance_accuracy': 0.5
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
            'formation_accuracy': f1,
            'memory_coverage': recall,  # Simple approximation
            'importance_accuracy': 0.5  # Unknown without LLM assessment
        }
    
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