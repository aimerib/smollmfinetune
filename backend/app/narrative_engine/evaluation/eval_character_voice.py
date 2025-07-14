"""
Character Voice Consistency Evaluation

Uses LLM-as-judge with structured outputs and embedding similarity 
to ensure characters maintain consistent voice and speaking patterns.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from pydantic import BaseModel, Field

from backend.app.core.openai_client import get_client

logger = logging.getLogger(__name__)


class VoiceConsistencyAnalysis(BaseModel):
    """Structured output for voice consistency analysis"""
    consistency_score: float = Field(
        ge=0.0, le=1.0, 
        description="Overall voice consistency score from 0-1"
    )
    speaking_style: str = Field(description="Identified speaking style/pattern")
    inconsistencies: List[str] = Field(
        description="List of specific inconsistencies found"
    )
    character_traits: List[str] = Field(
        description="Key character voice traits identified"
    )
    reasoning: str = Field(description="Explanation of the consistency assessment")


class CrossConversationAnalysis(BaseModel):
    """Analysis across multiple conversations"""
    overall_consistency: float = Field(ge=0.0, le=1.0)
    voice_drift_detected: bool = Field(description="Whether voice drift was detected")
    conversation_scores: List[float] = Field(description="Individual conversation scores")
    recommendations: List[str] = Field(description="Improvement recommendations")


class CharacterVoiceConsistencyEvaluator:
    """Evaluates character voice consistency using LLM-as-judge and embeddings"""
    
    def __init__(self, use_embeddings: bool = True):
        """
        Initialize the voice consistency evaluator.
        
        Args:
            use_embeddings: Whether to use embedding similarity analysis
        """
        self.client = get_client()
        self.use_embeddings = use_embeddings
        self.similarity_threshold = 0.85
    
    async def evaluate_character_voice(
        self,
        character_id: str,
        utterances: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate voice consistency for a single character using LLM-as-judge.
        
        Args:
            character_id: Unique identifier for the character
            utterances: List of character utterances to analyze
            
        Returns:
            Dictionary with voice consistency metrics
        """
        if len(utterances) < 2:
            logger.warning(f"Not enough utterances for character {character_id}")
            return {
                'character_id': character_id,
                'utterance_count': len(utterances),
                'voice_consistency_score': 1.0,
                'error': 'Insufficient utterances for analysis'
            }
        
        try:
            # Prepare samples for LLM analysis (limit to avoid token limits)
            sample_utterances = utterances[:10] if len(utterances) > 10 else utterances
            
            # Create prompt for LLM judge
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(character_id, sample_utterances)
            
            # Get structured analysis from LLM
            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.3,  # Lower temperature for consistent analysis
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "voice_consistency_analysis",
                        "schema": VoiceConsistencyAnalysis.model_json_schema()
                    }
                }
            )
            
            # Parse structured response
            import json
            analysis_data = json.loads(response_text)
            analysis = VoiceConsistencyAnalysis(**analysis_data)
            
            # Add embedding-based similarity if enabled
            embedding_score = None
            if self.use_embeddings:
                embedding_score = await self._calculate_embedding_similarity(sample_utterances)
            
            # Combine LLM analysis with embedding similarity
            final_score = analysis.consistency_score
            if embedding_score is not None:
                # Weight LLM judgment more heavily (70%) than embeddings (30%)
                final_score = 0.7 * analysis.consistency_score + 0.3 * embedding_score
            
            return {
                'character_id': character_id,
                'utterance_count': len(utterances),
                'voice_consistency_score': final_score,
                'llm_analysis': analysis.model_dump(),
                'embedding_similarity': embedding_score,
                'speaking_style': analysis.speaking_style,
                'inconsistencies': analysis.inconsistencies,
                'character_traits': analysis.character_traits,
                'reasoning': analysis.reasoning
            }
            
        except Exception as e:
            logger.error(f"Error evaluating voice consistency: {e}")
            return {
                'character_id': character_id,
                'utterance_count': len(utterances),
                'voice_consistency_score': 0.0,
                'error': str(e)
            }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for voice consistency analysis"""
        return """You are an expert in character voice analysis and creative writing. Your task is to evaluate the consistency of a character's voice and speaking patterns across multiple utterances.

Analyze the character's:
- Vocabulary choices and formality level
- Sentence structure and rhythm  
- Personality markers in speech
- Emotional expression patterns
- Cultural or regional speech markers
- Consistency of character traits

Rate consistency from 0.0 (completely inconsistent) to 1.0 (perfectly consistent).

Provide your analysis in the requested JSON format."""

    def _build_user_prompt(self, character_id: str, utterances: List[str]) -> str:
        """Build user prompt with character utterances"""
        formatted_utterances = []
        for i, utterance in enumerate(utterances, 1):
            formatted_utterances.append(f"{i}. \"{utterance}\"")
        
        utterance_text = "\n".join(formatted_utterances)
        
        return f"""Analyze the voice consistency for character "{character_id}" across these utterances:

{utterance_text}

Evaluate:
1. How consistent is the speaking style and vocabulary?
2. Are personality traits reflected consistently in speech?
3. Does the character maintain the same "voice" throughout?
4. What specific inconsistencies, if any, do you notice?

Provide a consistency score and detailed analysis."""

    async def _calculate_embedding_similarity(self, utterances: List[str]) -> float:
        """
        Calculate embedding-based similarity score.
        In production, this would use a real embedding model.
        For now, return a mock score based on text similarity.
        """
        # Mock implementation - in production, use sentence-transformers or OpenAI embeddings
        # This would be something like:
        # embeddings = await self.embedding_model.encode(utterances)
        # similarities = cosine_similarity_matrix(embeddings)
        # return np.mean(similarities[np.triu_indices_from(similarities, k=1)])
        
        # Simple heuristic based on text overlap
        if len(utterances) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(utterances)):
            for j in range(i + 1, len(utterances)):
                # Basic word overlap similarity
                words1 = set(utterances[i].lower().split())
                words2 = set(utterances[j].lower().split())
                
                if not words1 or not words2:
                    continue
                    
                overlap = len(words1 & words2)
                union = len(words1 | words2)
                similarity = overlap / union if union > 0 else 0
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.5
    
    async def evaluate_voice_across_conversations(
        self,
        character_id: str,
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate voice consistency across multiple conversations using LLM analysis.
        
        Args:
            character_id: Unique identifier for the character
            conversations: List of conversation dictionaries with utterances
            
        Returns:
            Dictionary with cross-conversation consistency metrics
        """
        try:
            # Extract utterances from each conversation
            conversation_utterances = []
            for i, conv in enumerate(conversations):
                utterances = conv.get('utterances', [])
                if utterances:
                    conversation_utterances.append({
                        'conversation_id': conv.get('conversation_id', f'conv_{i}'),
                        'utterances': utterances[:5]  # Limit per conversation
                    })
            
            if not conversation_utterances:
                return {
                    'character_id': character_id,
                    'cross_conversation_consistency': 0.0,
                    'error': 'No utterances found in conversations'
                }
            
            # Build prompt for cross-conversation analysis
            system_prompt = """You are analyzing character voice consistency across multiple conversations. Evaluate whether the character maintains the same voice, personality, and speaking patterns across different contexts and time periods."""
            
            user_prompt = self._build_cross_conversation_prompt(character_id, conversation_utterances)
            
            # Get LLM analysis
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
                        "name": "cross_conversation_analysis",
                        "schema": CrossConversationAnalysis.model_json_schema()
                    }
                }
            )
            
            # Parse response
            import json
            analysis_data = json.loads(response_text)
            analysis = CrossConversationAnalysis(**analysis_data)
            
            return {
                'character_id': character_id,
                'cross_conversation_consistency': analysis.overall_consistency,
                'voice_drift_detected': analysis.voice_drift_detected,
                'conversation_count': len(conversations),
                'total_utterances': sum(len(conv.get('utterances', [])) for conv in conversations),
                'per_conversation_scores': analysis.conversation_scores,
                'recommendations': analysis.recommendations,
                'llm_analysis': analysis.model_dump()
            }
            
        except Exception as e:
            logger.error(f"Error in cross-conversation analysis: {e}")
            return {
                'character_id': character_id,
                'cross_conversation_consistency': 0.0,
                'error': str(e)
            }
    
    def _build_cross_conversation_prompt(
        self, 
        character_id: str, 
        conversation_utterances: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for cross-conversation analysis"""
        formatted_conversations = []
        
        for i, conv_data in enumerate(conversation_utterances):
            conv_id = conv_data['conversation_id']
            utterances = conv_data['utterances']
            
            utterance_list = []
            for j, utterance in enumerate(utterances, 1):
                utterance_list.append(f"  {j}. \"{utterance}\"")
            
            formatted_conversations.append(
                f"Conversation {i+1} ({conv_id}):\n" + "\n".join(utterance_list)
            )
        
        conversations_text = "\n\n".join(formatted_conversations)
        
        return f"""Analyze voice consistency for character "{character_id}" across these conversations:

{conversations_text}

Evaluate:
1. Does the character maintain consistent voice across conversations?
2. Are there any signs of voice drift or inconsistency?
3. Rate each conversation's internal consistency
4. What recommendations would improve consistency?

Provide scores and analysis in the requested format."""

    # Synchronous wrapper for backward compatibility
    def evaluate_character_voice_sync(self, character_id: str, utterances: List[str]) -> Dict[str, Any]:
        """Synchronous wrapper for the async method"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.evaluate_character_voice(character_id, utterances))
        finally:
            loop.close() 