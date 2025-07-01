"""
Dialogue Naturalism Evaluation

Uses LLM-as-judge with structured outputs to assess the naturalness 
of dialogue, including conversational flow, linguistic markers, 
and human-like speaking patterns.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from pydantic import BaseModel, Field

from app.utils.openai_client import get_client

logger = logging.getLogger(__name__)


class LinguisticAnalysis(BaseModel):
    """Analysis of linguistic patterns in dialogue"""
    contraction_usage: float = Field(ge=0.0, le=1.0, description="Appropriate use of contractions")
    colloquial_expressions: float = Field(ge=0.0, le=1.0, description="Natural colloquial language use")
    sentence_variety: float = Field(ge=0.0, le=1.0, description="Variety in sentence structure")
    vocabulary_appropriateness: float = Field(ge=0.0, le=1.0, description="Vocabulary matches character/context")
    speech_patterns: List[str] = Field(description="Identified speech patterns")


class ConversationalFlow(BaseModel):
    """Analysis of conversation flow and coherence"""
    topic_transitions: float = Field(ge=0.0, le=1.0, description="Smoothness of topic transitions")
    response_relevance: float = Field(ge=0.0, le=1.0, description="Relevance of responses to previous messages")
    turn_taking: float = Field(ge=0.0, le=1.0, description="Natural turn-taking patterns")
    conversation_coherence: float = Field(ge=0.0, le=1.0, description="Overall conversation coherence")
    flow_issues: List[str] = Field(description="Specific flow problems identified")


class NaturalnessAssessment(BaseModel):
    """Complete naturalness assessment"""
    overall_naturalness: float = Field(ge=0.0, le=1.0, description="Overall dialogue naturalism score")
    linguistic_analysis: LinguisticAnalysis
    conversational_flow: ConversationalFlow
    artificial_patterns: List[str] = Field(description="Detected artificial or robotic patterns")
    human_like_qualities: List[str] = Field(description="Human-like qualities in the dialogue")
    improvement_suggestions: List[str] = Field(description="Specific suggestions for improvement")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the assessment")


class DialogueNaturalnessEvaluator:
    """Evaluates dialogue naturalism using LLM-as-judge"""
    
    def __init__(self):
        """Initialize the dialogue naturalism evaluator"""
        self.client = get_client()
    
    async def evaluate_dialogue_naturalism(
        self,
        dialogue: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate the naturalism of dialogue using LLM analysis.
        
        Args:
            dialogue: List of dialogue turns with role and content
            
        Returns:
            Dictionary with naturalism scores and analysis
        """
        if not dialogue:
            return {
                'naturalism_score': 0.0,
                'error': 'No dialogue provided'
            }
        
        try:
            # Extract assistant responses for analysis
            assistant_turns = [
                turn for turn in dialogue 
                if turn.get('role') == 'assistant' and turn.get('content')
            ]
            
            if not assistant_turns:
                return {
                    'naturalism_score': 0.0,
                    'error': 'No assistant turns found'
                }
            
            # Build prompt for LLM analysis
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(dialogue)
            
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
                        "name": "naturalness_assessment",
                        "schema": NaturalnessAssessment.model_json_schema()
                    }
                }
            )
            
            # Parse structured response
            import json
            assessment_data = json.loads(response_text)
            assessment = NaturalnessAssessment(**assessment_data)
            
            return {
                'naturalism_score': assessment.overall_naturalness,
                'confidence': assessment.confidence,
                'linguistic_analysis': assessment.linguistic_analysis.model_dump(),
                'conversational_flow': assessment.conversational_flow.model_dump(),
                'artificial_patterns': assessment.artificial_patterns,
                'human_like_qualities': assessment.human_like_qualities,
                'improvement_suggestions': assessment.improvement_suggestions,
                'llm_analysis': assessment.model_dump(),
                'turn_count': len(assistant_turns)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating dialogue naturalism: {e}")
            return {
                'naturalism_score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for dialogue naturalism analysis"""
        return """You are an expert in linguistics, conversation analysis, and natural language understanding. Your task is to evaluate how natural and human-like a dialogue appears.

Analyze the dialogue for:

**Linguistic Patterns:**
- Use of contractions (I'll, don't, won't) vs. formal constructions
- Colloquial expressions and informal language
- Sentence structure variety (simple, complex, fragments)
- Vocabulary appropriateness for context and character
- Natural speech patterns and rhythms

**Conversational Flow:**
- Smooth topic transitions
- Relevant responses that build on previous messages
- Natural turn-taking without abrupt changes
- Overall conversation coherence and logic

**Naturalness Indicators:**
- Human-like imperfections (hesitations, corrections, informal language)
- Emotional expressiveness appropriate to context
- Conversational markers (well, you know, actually, I mean)
- Natural response patterns

**Artificial Patterns to Detect:**
- Overly formal or robotic language
- Repetitive sentence structures
- Unnatural politeness or formality
- Generic responses that could apply to any context
- Lack of personality or emotional variation

Rate overall naturalness from 0.0 (completely artificial) to 1.0 (indistinguishable from human dialogue).

Provide detailed analysis in the requested JSON format."""

    def _build_user_prompt(self, dialogue: List[Dict[str, Any]]) -> str:
        """Build user prompt with dialogue content"""
        formatted_dialogue = []
        
        for i, turn in enumerate(dialogue):
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            formatted_dialogue.append(f"{role.capitalize()}: {content}")
        
        dialogue_text = "\n\n".join(formatted_dialogue)
        
        return f"""Analyze this dialogue for naturalism and human-like qualities:

{dialogue_text}

Evaluate:
1. How natural and human-like does this dialogue sound?
2. What linguistic patterns make it feel natural or artificial?
3. How well does the conversation flow between turns?
4. Are there any robotic or unnatural patterns?
5. What human-like qualities are present?
6. What specific improvements would make it more natural?

Provide a comprehensive naturalism assessment with scores and detailed analysis."""

    async def detect_artificial_patterns(
        self,
        text_samples: List[str]
    ) -> Dict[str, Any]:
        """
        Detect artificial patterns in text samples using LLM analysis.
        
        Args:
            text_samples: List of text samples to analyze
            
        Returns:
            Dictionary with detected patterns and scores
        """
        if not text_samples:
            return {
                'artificial_score': 0.0,
                'patterns_detected': [],
                'error': 'No text samples provided'
            }
        
        try:
            # Limit samples to avoid token limits
            sample_texts = text_samples[:8] if len(text_samples) > 8 else text_samples
            
            # Build prompt for pattern detection
            system_prompt = """You are an expert at detecting artificial or robotic patterns in text. Identify specific markers that indicate text was generated by an AI rather than written by a human."""
            
            user_prompt = self._build_pattern_detection_prompt(sample_texts)
            
            # Get LLM analysis
            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.3
            )
            
            # Parse patterns from response
            artificial_patterns = self._parse_pattern_response(response_text)
            
            # Calculate artificial score based on patterns found
            artificial_score = min(1.0, len(artificial_patterns) * 0.15)
            
            return {
                'artificial_score': artificial_score,
                'patterns_detected': artificial_patterns,
                'sample_count': len(sample_texts),
                'analysis_text': response_text
            }
            
        except Exception as e:
            logger.error(f"Error detecting artificial patterns: {e}")
            return {
                'artificial_score': 0.0,
                'patterns_detected': [],
                'error': str(e)
            }
    
    def _build_pattern_detection_prompt(self, samples: List[str]) -> str:
        """Build prompt for artificial pattern detection"""
        formatted_samples = []
        for i, sample in enumerate(samples, 1):
            formatted_samples.append(f"{i}. \"{sample}\"")
        
        samples_text = "\n".join(formatted_samples)
        
        return f"""Analyze these text samples for artificial or robotic patterns:

{samples_text}

Look for:
- Overly formal language where informal would be natural
- Repetitive sentence structures or phrases
- Generic responses that lack specificity
- Unnatural politeness or constant helpfulness
- Lack of contractions where they would be natural
- Robotic transitions between topics
- Missing emotional nuance or personality
- AI-typical phrases or constructions

List specific artificial patterns you detect, if any."""

    def _parse_pattern_response(self, response_text: str) -> List[str]:
        """Parse artificial patterns from LLM response"""
        patterns = []
        
        # Look for bullet points, numbered lists, or line items
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or 
                        line.startswith('*') or line[0].isdigit()):
                # Clean up formatting
                pattern = line.lstrip('-•*0123456789. ').strip()
                if pattern and len(pattern) > 10:  # Filter out very short items
                    patterns.append(pattern)
        
        # If no structured patterns found, look for sentences that might be patterns
        if not patterns:
            sentences = response_text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and any(keyword in sentence.lower() for keyword in 
                                            ['pattern', 'artificial', 'robotic', 'unnatural', 'generic']):
                    patterns.append(sentence)
        
        return patterns[:10]  # Limit to top 10 patterns
    
    async def analyze_conversation_flow(
        self,
        conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze conversation flow and coherence using LLM.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Dictionary with flow analysis
        """
        if len(conversation) < 2:
            return {
                'flow_score': 1.0,
                'coherence_score': 1.0,
                'topic_transitions': [],
                'error': 'Insufficient turns for flow analysis'
            }
        
        try:
            # Build prompt for flow analysis
            system_prompt = """You are analyzing conversation flow and coherence. Focus on how well turns connect to each other, how smoothly topics transition, and how natural the overall conversation feels."""
            
            user_prompt = self._build_flow_analysis_prompt(conversation)
            
            # Get structured analysis
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
                        "name": "conversational_flow",
                        "schema": ConversationalFlow.model_json_schema()
                    }
                }
            )
            
            # Parse response
            import json
            flow_data = json.loads(response_text)
            flow_analysis = ConversationalFlow(**flow_data)
            
            return {
                'flow_score': flow_analysis.turn_taking,
                'coherence_score': flow_analysis.conversation_coherence,
                'topic_transitions': flow_analysis.topic_transitions,
                'response_relevance': flow_analysis.response_relevance,
                'flow_issues': flow_analysis.flow_issues,
                'llm_analysis': flow_analysis.model_dump()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing conversation flow: {e}")
            return {
                'flow_score': 0.0,
                'coherence_score': 0.0,
                'error': str(e)
            }
    
    def _build_flow_analysis_prompt(self, conversation: List[Dict[str, Any]]) -> str:
        """Build prompt for conversation flow analysis"""
        formatted_turns = []
        
        for i, turn in enumerate(conversation):
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            formatted_turns.append(f"Turn {i+1} ({role}): {content}")
        
        conversation_text = "\n\n".join(formatted_turns)
        
        return f"""Analyze the flow and coherence of this conversation:

{conversation_text}

Evaluate:
1. How smoothly do topics transition between turns?
2. How relevant are responses to the previous messages?
3. How natural is the turn-taking pattern?
4. What is the overall conversation coherence?
5. Are there any specific flow issues or problems?

Rate each aspect from 0.0 to 1.0 and identify specific issues."""

    # Synchronous wrapper for backward compatibility
    def evaluate_dialogue_naturalism_sync(self, dialogue: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synchronous wrapper for the async method"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.evaluate_dialogue_naturalism(dialogue))
        finally:
            loop.close() 