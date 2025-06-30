"""
Coherence Evaluation

Uses an LLM-as-judge to check for contradictions and coherence in conversations.
Essential for ensuring character consistency and logical responses.
"""

import logging
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


class CoherenceEvaluator:
    """Evaluates conversation coherence using LLM-as-judge"""
    
    def __init__(self, judge_model: Optional[Any] = None):
        self.judge_model = judge_model
        self.max_conversation_length = 100  # Max turns to evaluate at once
        
    def evaluate_conversation(
        self,
        conversation: List[Dict[str, str]],
        character_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a conversation for coherence and contradictions.
        
        Args:
            conversation: List of conversation turns with role and content
            character_context: Optional character information for consistency checking
            
        Returns:
            Dictionary with coherence metrics
        """
        results = {
            'coherence_score': 0.0,
            'contradictions': [],
            'reasoning': '',
            'turn_count': len(conversation)
        }
        
        try:
            # Handle long conversations by chunking or summarizing
            if len(conversation) > self.max_conversation_length:
                # Process in chunks and aggregate results
                chunk_results = []
                chunk_size = 50
                
                for i in range(0, len(conversation), chunk_size):
                    chunk = conversation[i:i + chunk_size]
                    chunk_result = self._evaluate_chunk(chunk, character_context)
                    chunk_results.append(chunk_result)
                
                # Aggregate results
                results = self._aggregate_chunk_results(chunk_results)
            else:
                # Evaluate entire conversation
                judge_result = self._call_judge_llm(conversation, character_context)
                results.update(judge_result)
            
            logger.info(f"Coherence evaluation complete: score={results['coherence_score']}")
            
        except Exception as e:
            logger.error(f"Error in coherence evaluation: {e}")
            results['error'] = str(e)
        
        return results
    
    def _call_judge_llm(
        self,
        conversation: List[Dict[str, str]],
        character_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call the judge LLM to evaluate coherence.
        
        Args:
            conversation: Conversation to evaluate
            character_context: Optional character information
            
        Returns:
            Judge evaluation results
        """
        # Format conversation for judge
        formatted_conv = self._format_conversation(conversation)
        
        # Create judge prompt
        prompt = f"""Evaluate the following conversation for coherence and contradictions.

Conversation:
{formatted_conv}

Evaluate for:
1. Internal contradictions (character says conflicting things)
2. Logical consistency
3. Character consistency (if character info provided)
4. Factual consistency

Respond with JSON containing:
- coherence_score: float between 0 and 1 (1 = perfectly coherent)
- contradictions: list of specific contradictions found
- reasoning: brief explanation of the evaluation
"""
        
        if character_context:
            prompt += f"\n\nCharacter Context:\n{json.dumps(character_context, indent=2)}"
        
        # In real implementation, this would call the judge model
        # For now, return mock response based on conversation analysis
        
        # Simple heuristic: check for obvious contradictions
        contradictions = self._detect_simple_contradictions(conversation)
        
        if contradictions:
            return {
                'coherence_score': max(0.2, 1.0 - (len(contradictions) * 0.3)),
                'contradictions': contradictions,
                'reasoning': f"Found {len(contradictions)} contradictions in the conversation"
            }
        else:
            return {
                'coherence_score': 0.95,
                'contradictions': [],
                'reasoning': "Conversation appears coherent with no obvious contradictions"
            }
    
    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for display"""
        formatted = []
        for turn in conversation:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            formatted.append(f"{role.upper()}: {content}")
        return "\n".join(formatted)
    
    def _detect_simple_contradictions(self, conversation: List[Dict[str, str]]) -> List[str]:
        """
        Detect simple contradictions in conversation.
        This is a placeholder for more sophisticated analysis.
        """
        contradictions = []
        
        # Check for name contradictions (simple example)
        assistant_names = []
        for turn in conversation:
            if turn.get('role') == 'assistant':
                content = turn.get('content', '').lower()
                if 'my name is' in content:
                    # Extract name (simplified)
                    parts = content.split('my name is')
                    if len(parts) > 1:
                        name = parts[1].split('.')[0].split(',')[0].strip()
                        if name and name not in assistant_names:
                            if assistant_names:
                                contradictions.append(
                                    f"Character claimed name is {assistant_names[0]}, then {name}"
                                )
                            assistant_names.append(name)
        
        return contradictions
    
    def _evaluate_chunk(
        self,
        chunk: List[Dict[str, str]],
        character_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a chunk of conversation"""
        return self._call_judge_llm(chunk, character_context)
    
    def _aggregate_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple chunks"""
        if not chunk_results:
            return {
                'coherence_score': 0.0,
                'contradictions': [],
                'reasoning': 'No chunks to evaluate'
            }
        
        # Average coherence scores
        total_score = sum(r.get('coherence_score', 0) for r in chunk_results)
        avg_score = total_score / len(chunk_results)
        
        # Collect all contradictions
        all_contradictions = []
        for r in chunk_results:
            all_contradictions.extend(r.get('contradictions', []))
        
        # Combine reasoning
        reasonings = [r.get('reasoning', '') for r in chunk_results if r.get('reasoning')]
        combined_reasoning = ' | '.join(reasonings) if reasonings else 'Aggregated from chunks'
        
        return {
            'coherence_score': avg_score,
            'contradictions': all_contradictions,
            'reasoning': combined_reasoning,
            'chunk_count': len(chunk_results)
        } 