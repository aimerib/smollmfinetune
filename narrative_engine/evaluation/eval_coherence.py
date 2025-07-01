"""
Coherence Evaluation

Uses an LLM-as-judge with structured outputs to check for contradictions and coherence in conversations.
Essential for ensuring character consistency and logical responses.
"""

import logging
from typing import List, Dict, Any, Optional
import json
import asyncio
from pydantic import BaseModel, Field

from app.utils.openai_client import get_client

logger = logging.getLogger(__name__)


class ContradictionAnalysis(BaseModel):
    """Structured output for contradiction detection"""
    contradictions_found: List[str] = Field(description="Specific contradictions identified")
    contradiction_severity: str = Field(description="Overall severity: none, low, medium, high")
    logical_consistency: float = Field(ge=0.0, le=1.0, description="Overall logical consistency score")
    factual_consistency: float = Field(ge=0.0, le=1.0, description="Factual consistency score")
    character_consistency: float = Field(ge=0.0, le=1.0, description="Character consistency score")
    reasoning: str = Field(description="Detailed reasoning for the assessment")


class CoherenceAssessment(BaseModel):
    """Complete coherence evaluation"""
    coherence_score: float = Field(ge=0.0, le=1.0, description="Overall coherence score")
    contradiction_analysis: ContradictionAnalysis = Field(description="Detailed contradiction analysis")
    flow_quality: float = Field(ge=0.0, le=1.0, description="Conversation flow quality")
    topic_consistency: float = Field(ge=0.0, le=1.0, description="Topic consistency throughout")
    improvement_suggestions: List[str] = Field(description="Specific suggestions for improvement")


class CoherenceEvaluator:
    """Evaluates conversation coherence using LLM-as-judge"""
    
    def __init__(self, judge_model: Optional[Any] = None):
        self.client = get_client()
        self.max_conversation_length = 100  # Max turns to evaluate at once
        
    def evaluate_conversation(
        self,
        conversation: List[Dict[str, str]],
        character_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a conversation for coherence and contradictions using LLM-as-judge.
        
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
            # Handle long conversations by chunking
            if len(conversation) > self.max_conversation_length:
                # Process in chunks and aggregate results
                chunk_results = []
                chunk_size = 50
                
                for i in range(0, len(conversation), chunk_size):
                    chunk = conversation[i:i + chunk_size]
                    chunk_result = asyncio.run(self._evaluate_chunk_async(chunk, character_context))
                    chunk_results.append(chunk_result)
                
                # Aggregate results
                results = self._aggregate_chunk_results(chunk_results)
            else:
                # Evaluate entire conversation
                judge_result = asyncio.run(self._call_judge_llm_async(conversation, character_context))
                results.update(judge_result)
            
            logger.info(f"Coherence evaluation complete: score={results['coherence_score']}")
            
        except Exception as e:
            logger.error(f"Error in coherence evaluation: {e}")
            results['error'] = str(e)
            # Fallback to heuristic analysis
            fallback_result = self._fallback_analysis(conversation)
            results.update(fallback_result)
        
        return results
    
    async def _call_judge_llm_async(
        self,
        conversation: List[Dict[str, str]],
        character_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call the judge LLM to evaluate coherence using structured output.
        
        Args:
            conversation: Conversation to evaluate
            character_context: Optional character information
            
        Returns:
            Judge evaluation results
        """
        try:
            # Format conversation for judge
            formatted_conv = self._format_conversation(conversation)
            
            # Build system prompt
            system_prompt = self._build_coherence_system_prompt()
            
            # Build user prompt
            user_prompt = self._build_coherence_user_prompt(formatted_conv, character_context)
            
            # Get structured analysis from LLM
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
                        "name": "coherence_assessment",
                        "schema": CoherenceAssessment.model_json_schema()
                    }
                }
            )
            
            # Parse structured response
            assessment_data = json.loads(response_text)
            assessment = CoherenceAssessment(**assessment_data)
            
            return {
                'coherence_score': assessment.coherence_score,
                'contradictions': assessment.contradiction_analysis.contradictions_found,
                'reasoning': assessment.contradiction_analysis.reasoning,
                'flow_quality': assessment.flow_quality,
                'topic_consistency': assessment.topic_consistency,
                'contradiction_severity': assessment.contradiction_analysis.contradiction_severity,
                'logical_consistency': assessment.contradiction_analysis.logical_consistency,
                'factual_consistency': assessment.contradiction_analysis.factual_consistency,
                'character_consistency': assessment.contradiction_analysis.character_consistency,
                'improvement_suggestions': assessment.improvement_suggestions,
                'llm_analysis': assessment.model_dump()
            }
            
        except Exception as e:
            logger.error(f"Error calling judge LLM for coherence: {e}")
            # Return fallback analysis
            return self._fallback_analysis(conversation)
    
    def _build_coherence_system_prompt(self) -> str:
        """Build system prompt for coherence evaluation"""
        return """You are an expert in conversation analysis and logical reasoning. Your task is to evaluate conversation coherence by identifying contradictions, assessing logical flow, and ensuring consistency.

Analyze the conversation for:
1. **Internal Contradictions**: Character says conflicting things
2. **Logical Consistency**: Responses follow logically from previous statements
3. **Character Consistency**: Character maintains consistent traits and knowledge
4. **Factual Consistency**: Facts mentioned remain consistent throughout
5. **Conversation Flow**: Natural progression of topics and responses
6. **Topic Consistency**: How well the conversation stays coherent to its themes

**SCORING GUIDELINES:**
- 0.9-1.0: Perfectly coherent with no contradictions
- 0.7-0.8: Mostly coherent with minor inconsistencies
- 0.5-0.6: Moderately coherent but noticeable issues
- 0.3-0.4: Poor coherence with significant problems
- 0.0-0.2: Incoherent with major contradictions

**CRITICAL**: All numeric scores must be decimal values between 0.0 and 1.0 (inclusive).

Provide your analysis in the requested JSON format with specific examples."""

    def _build_coherence_user_prompt(
        self, 
        formatted_conv: str, 
        character_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build user prompt for coherence evaluation"""
        prompt = f"""Evaluate this conversation for coherence and contradictions:

{formatted_conv}

Analyze for:
1. **Contradictions**: Any statements that conflict with each other
2. **Logical Flow**: Whether responses follow logically
3. **Character Consistency**: Does the character maintain consistent traits?
4. **Topic Coherence**: Does the conversation flow naturally between topics?

"""
        
        if character_context:
            prompt += f"""
**Character Context:**
{json.dumps(character_context, indent=2)}

Also check if the character's responses are consistent with their established traits and background.
"""
        
        prompt += """
Provide a comprehensive analysis with:
- Overall coherence score (0.0 to 1.0)
- Specific contradictions found (if any)
- Assessment of logical, factual, and character consistency
- Suggestions for improvement

Be specific in identifying contradictions with examples from the conversation."""
        
        return prompt
    
    async def _evaluate_chunk_async(
        self,
        chunk: List[Dict[str, str]],
        character_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a chunk of conversation asynchronously"""
        return await self._call_judge_llm_async(chunk, character_context)
    
    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation for display"""
        formatted = []
        for i, turn in enumerate(conversation, 1):
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            formatted.append(f"{i}. {role.upper()}: {content}")
        return "\n".join(formatted)
    
    def _fallback_analysis(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Fallback heuristic analysis when LLM calls fail"""
        # Simple heuristic: check for obvious contradictions
        contradictions = self._detect_simple_contradictions(conversation)
        
        if contradictions:
            return {
                'coherence_score': max(0.2, 1.0 - (len(contradictions) * 0.3)),
                'contradictions': contradictions,
                'reasoning': f"Found {len(contradictions)} contradictions using fallback analysis",
                'flow_quality': 0.5,
                'topic_consistency': 0.6,
                'error': 'Used fallback analysis due to LLM failure'
            }
        else:
            return {
                'coherence_score': 0.75,  # Conservative estimate
                'contradictions': [],
                'reasoning': "No obvious contradictions found using fallback analysis",
                'flow_quality': 0.7,
                'topic_consistency': 0.8,
                'error': 'Used fallback analysis due to LLM failure'
            }
    
    def _detect_simple_contradictions(self, conversation: List[Dict[str, str]]) -> List[str]:
        """
        Detect simple contradictions in conversation using heuristics.
        This is a fallback for when LLM calls fail.
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
                                    f"Character claimed name is '{assistant_names[0]}', then '{name}'"
                                )
                            assistant_names.append(name)
        
        return contradictions
    
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
        
        # Average other metrics
        flow_scores = [r.get('flow_quality', 0.5) for r in chunk_results if 'flow_quality' in r]
        topic_scores = [r.get('topic_consistency', 0.5) for r in chunk_results if 'topic_consistency' in r]
        
        avg_flow = sum(flow_scores) / len(flow_scores) if flow_scores else 0.5
        avg_topic = sum(topic_scores) / len(topic_scores) if topic_scores else 0.5
        
        # Combine reasoning
        reasonings = [r.get('reasoning', '') for r in chunk_results if r.get('reasoning')]
        combined_reasoning = ' | '.join(reasonings) if reasonings else 'Aggregated from chunks'
        
        return {
            'coherence_score': avg_score,
            'contradictions': all_contradictions,
            'reasoning': combined_reasoning,
            'flow_quality': avg_flow,
            'topic_consistency': avg_topic,
            'chunk_count': len(chunk_results)
        }
    
    # Synchronous wrapper for backward compatibility
    def call_judge_llm_sync(
        self,
        conversation: List[Dict[str, str]],
        character_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for the async LLM judge method"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._call_judge_llm_async(conversation, character_context))
        finally:
            loop.close() 