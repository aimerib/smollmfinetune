"""
Emotional Arc Evaluation

Uses LLM-as-judge with structured outputs to track and evaluate 
emotional progression throughout conversations, ensuring natural 
and coherent emotional transitions.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from pydantic import BaseModel, Field

from backend.app.core.openai_client import get_client

logger = logging.getLogger(__name__)


class EmotionalState(BaseModel):
    """Represents an emotional state at a specific point"""
    turn_number: int = Field(description="Turn number in conversation")
    primary_emotion: str = Field(description="Primary emotion detected")
    emotion_intensity: float = Field(ge=0.0, le=1.0, description="Intensity of the emotion")
    secondary_emotions: List[str] = Field(description="Secondary emotions present")
    emotional_markers: List[str] = Field(description="Specific words/phrases indicating emotion")


class EmotionalArcAnalysis(BaseModel):
    """Complete emotional arc analysis"""
    emotional_trajectory: List[EmotionalState] = Field(description="Sequence of emotional states")
    arc_coherence_score: float = Field(ge=0.0, le=1.0, description="How coherent the emotional progression is")
    naturalness_score: float = Field(ge=0.0, le=1.0, description="How natural the emotional transitions are")
    emotional_range: float = Field(ge=0.0, le=1.0, description="Breadth of emotions expressed")
    dominant_emotions: List[str] = Field(description="Most prevalent emotions in the arc")
    transition_quality: str = Field(description="Overall quality of emotional transitions")
    recommendations: List[str] = Field(description="Suggestions for improvement")


class EmotionalTransitionAnalysis(BaseModel):
    """Analysis of specific emotional transitions"""
    transition_smoothness: float = Field(ge=0.0, le=1.0)
    abrupt_changes: int = Field(description="Number of abrupt emotional changes")
    natural_progressions: int = Field(description="Number of natural progressions")
    problematic_transitions: List[str] = Field(description="Specific transitions that seem unnatural")


class EmotionalArcEvaluator:
    """Evaluates emotional arc tracking and scoring using LLM-as-judge"""
    
    def __init__(self):
        """Initialize the emotional arc evaluator"""
        self.client = get_client()
        self.emotion_categories = [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 
            'neutral', 'excitement', 'contentment', 'frustration', 'anxiety'
        ]
    
    async def track_emotional_arc(
        self,
        conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Track emotional arc through a conversation using LLM analysis.
        
        Args:
            conversation: List of conversation turns with role and content
            
        Returns:
            Dictionary with emotional trajectory and metrics
        """
        if not conversation:
            return {
                'emotional_trajectory': [],
                'arc_coherence_score': 0.0,
                'error': 'No conversation provided'
            }
        
        try:
            # Extract assistant turns for analysis
            assistant_turns = []
            for i, turn in enumerate(conversation):
                if turn.get('role') == 'assistant' and turn.get('content'):
                    assistant_turns.append({
                        'turn_number': i + 1,
                        'content': turn['content']
                    })
            
            if not assistant_turns:
                return {
                    'emotional_trajectory': [],
                    'arc_coherence_score': 0.0,
                    'error': 'No assistant turns found'
                }
            
            # Build prompt for LLM analysis
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(assistant_turns)
            
            # Get structured analysis from LLM
            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "emotional_arc_analysis",
                        "schema": EmotionalArcAnalysis.model_json_schema()
                    }
                }
            )
            
            # Parse structured response
            import json
            analysis_data = json.loads(response_text)
            analysis = EmotionalArcAnalysis(**analysis_data)
            
            # Convert to compatible format
            return {
                'emotional_trajectory': [state.model_dump() for state in analysis.emotional_trajectory],
                'arc_coherence_score': analysis.arc_coherence_score,
                'naturalness_score': analysis.naturalness_score,
                'emotional_variance': analysis.emotional_range,
                'dominant_emotions': analysis.dominant_emotions,
                'transition_quality': analysis.transition_quality,
                'recommendations': analysis.recommendations,
                'llm_analysis': analysis.model_dump(),
                'turn_count': len(assistant_turns)
            }
            
        except Exception as e:
            logger.error(f"Error tracking emotional arc: {e}")
            return {
                'emotional_trajectory': [],
                'arc_coherence_score': 0.0,
                'error': str(e)
            }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for emotional arc analysis"""
        return f"""You are an expert in emotional intelligence and narrative analysis. Your task is to analyze the emotional arc of a character through a conversation, tracking how their emotions develop and change over time.

Analyze each turn for:
- Primary emotion expressed
- Intensity of that emotion (0.0 to 1.0)
- Secondary emotions present
- Specific words or phrases that indicate emotion
- Overall emotional progression and coherence

Available emotion categories: {', '.join(self.emotion_categories)}

Rate:
- Arc coherence: How well the emotional progression makes sense (0.0 to 1.0)
- Naturalness: How realistic the emotional transitions are (0.0 to 1.0) 
- Emotional range: Breadth of emotions expressed (0.0 to 1.0)

Provide your analysis in the requested JSON format."""

    def _build_user_prompt(self, assistant_turns: List[Dict[str, Any]]) -> str:
        """Build user prompt with conversation turns"""
        formatted_turns = []
        for turn in assistant_turns:
            formatted_turns.append(f"Turn {turn['turn_number']}: \"{turn['content']}\"")
        
        turns_text = "\n".join(formatted_turns)
        
        return f"""Analyze the emotional arc in this conversation:

{turns_text}

Track:
1. The emotional state in each turn
2. How emotions transition between turns
3. Overall coherence and naturalness of the emotional progression
4. The range and variety of emotions expressed

Provide a detailed analysis with scores and recommendations."""

    async def evaluate_arc_naturalness(
        self,
        trajectory: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate naturalness of emotional progression using LLM analysis.
        
        Args:
            trajectory: Emotional trajectory with turn numbers and emotions
            
        Returns:
            Dictionary with naturalness metrics
        """
        if len(trajectory) < 2:
            return {
                'naturalness_score': 1.0,
                'abrupt_transitions': 0,
                'emotion_flow_pattern': 'insufficient_data'
            }
        
        try:
            # Build prompt for transition analysis
            system_prompt = """You are analyzing the naturalness of emotional transitions in a character's dialogue. Evaluate whether the emotional changes feel realistic and appropriate for human psychology."""
            
            user_prompt = self._build_transition_prompt(trajectory)
            
            # Get LLM analysis
            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "emotional_transition_analysis",
                        "schema": EmotionalTransitionAnalysis.model_json_schema()
                    }
                }
            )
            
            # Parse response
            import json
            analysis_data = json.loads(response_text)
            analysis = EmotionalTransitionAnalysis(**analysis_data)
            
            # Determine flow pattern
            if analysis.abrupt_changes == 0:
                flow_pattern = 'smooth'
            elif analysis.abrupt_changes <= 1:
                flow_pattern = 'mostly_smooth'
            else:
                flow_pattern = 'erratic'
            
            return {
                'naturalness_score': analysis.transition_smoothness,
                'abrupt_transitions': analysis.abrupt_changes,
                'natural_progressions': analysis.natural_progressions,
                'emotion_flow_pattern': flow_pattern,
                'problematic_transitions': analysis.problematic_transitions,
                'llm_analysis': analysis.model_dump()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating arc naturalness: {e}")
            return {
                'naturalness_score': 0.0,
                'abrupt_transitions': 0,
                'emotion_flow_pattern': 'error',
                'error': str(e)
            }
    
    def _build_transition_prompt(self, trajectory: List[Dict[str, Any]]) -> str:
        """Build prompt for analyzing emotional transitions"""
        transitions = []
        
        for i in range(1, len(trajectory)):
            prev_state = trajectory[i-1]
            curr_state = trajectory[i]
            
            prev_emotion = prev_state.get('primary_emotion', 'unknown')
            curr_emotion = curr_state.get('primary_emotion', 'unknown')
            
            transitions.append(
                f"Turn {prev_state.get('turn_number', i)}: {prev_emotion} → "
                f"Turn {curr_state.get('turn_number', i+1)}: {curr_emotion}"
            )
        
        transitions_text = "\n".join(transitions)
        
        return f"""Analyze these emotional transitions for naturalness:

{transitions_text}

Evaluate:
1. How smooth and natural are these emotional transitions?
2. Are there any abrupt or unrealistic emotional jumps?
3. How many transitions feel natural vs. forced?
4. Which specific transitions seem problematic?

Rate the overall transition smoothness from 0.0 (very unnatural) to 1.0 (completely natural)."""

    async def analyze_emotional_patterns(
        self,
        multiple_conversations: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Analyze emotional patterns across multiple conversations.
        
        Args:
            multiple_conversations: List of conversations to analyze
            
        Returns:
            Dictionary with pattern analysis
        """
        try:
            conversation_arcs = []
            
            # Analyze each conversation
            for i, conversation in enumerate(multiple_conversations[:5]):  # Limit to 5 conversations
                arc_result = await self.track_emotional_arc(conversation)
                if 'error' not in arc_result:
                    conversation_arcs.append({
                        'conversation_id': i,
                        'arc_coherence': arc_result['arc_coherence_score'],
                        'dominant_emotions': arc_result['dominant_emotions'],
                        'emotional_range': arc_result.get('emotional_variance', 0.0)
                    })
            
            if not conversation_arcs:
                return {
                    'pattern_consistency': 0.0,
                    'error': 'No valid conversation arcs found'
                }
            
            # Calculate pattern metrics
            coherence_scores = [arc['arc_coherence'] for arc in conversation_arcs]
            emotional_ranges = [arc['emotional_range'] for arc in conversation_arcs]
            
            # Analyze consistency of emotional patterns
            pattern_consistency = 1.0 - np.std(coherence_scores) if len(coherence_scores) > 1 else 1.0
            
            # Find common emotional themes
            all_emotions = []
            for arc in conversation_arcs:
                all_emotions.extend(arc['dominant_emotions'])
            
            from collections import Counter
            emotion_frequency = Counter(all_emotions)
            common_emotions = [emotion for emotion, count in emotion_frequency.most_common(3)]
            
            return {
                'pattern_consistency': float(pattern_consistency),
                'average_coherence': float(np.mean(coherence_scores)),
                'average_emotional_range': float(np.mean(emotional_ranges)),
                'common_emotional_themes': common_emotions,
                'conversation_count': len(conversation_arcs),
                'coherence_variance': float(np.var(coherence_scores))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing emotional patterns: {e}")
            return {
                'pattern_consistency': 0.0,
                'error': str(e)
            }

    # Synchronous wrapper for backward compatibility
    def track_emotional_arc_sync(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synchronous wrapper for the async method"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.track_emotional_arc(conversation))
        finally:
            loop.close() 