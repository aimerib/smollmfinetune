"""
Triple Head Coordination Evaluation

Evaluates how well the generation, control, and memory heads
work together to produce coherent outputs.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import re

logger = logging.getLogger(__name__)


class TripleHeadCoordinationEvaluator:
    """Evaluates coordination between generation, control, and memory heads"""
    
    def __init__(self):
        """Initialize the triple head coordination evaluator"""
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'joyful', 'pleased', 'delighted', 'cheerful'],
            'sadness': ['sad', 'disappointed', 'upset', 'unhappy', 'depressed', 'melancholy'],
            'anger': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated'],
            'fear': ['scared', 'afraid', 'frightened', 'anxious', 'worried', 'terrified'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened'],
            'neutral': ['okay', 'fine', 'alright', 'normal']
        }
        
        self.tone_indicators = {
            'friendly': ['friend', 'warm', 'kind', 'nice', 'welcome'],
            'formal': ['sir', 'madam', 'formally', 'respectfully'],
            'casual': ['hey', 'yeah', 'cool', 'awesome', 'chill'],
            'excited': ['!', 'wow', 'amazing', 'fantastic', 'incredible']
        }
    
    def evaluate_coordination(
        self,
        generation: str,
        control: Dict[str, Any],
        memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate coordination between all three heads.
        
        Args:
            generation: Generated text output
            control: Control head outputs (emotion, tone, etc.)
            memory: Memory head outputs (retrieved/formed memories)
            
        Returns:
            Dictionary with coordination metrics
        """
        results = {
            'overall_coordination': 0.0,
            'generation_control_alignment': 0.0,
            'generation_memory_alignment': 0.0,
            'control_memory_alignment': 0.0,
            'alignment_failures': []
        }
        
        try:
            # Evaluate generation-control alignment
            gen_control = self._evaluate_generation_control_alignment(generation, control)
            results['generation_control_alignment'] = gen_control['score']
            if gen_control['failures']:
                results['alignment_failures'].extend(gen_control['failures'])
            
            # Evaluate generation-memory alignment
            gen_memory = self._evaluate_generation_memory_alignment(generation, memory)
            results['generation_memory_alignment'] = gen_memory['score']
            if gen_memory['failures']:
                results['alignment_failures'].extend(gen_memory['failures'])
            
            # Evaluate control-memory alignment
            control_memory = self._evaluate_control_memory_alignment(control, memory)
            results['control_memory_alignment'] = control_memory['score']
            if control_memory['failures']:
                results['alignment_failures'].extend(control_memory['failures'])
            
            # Calculate overall coordination
            scores = [
                results['generation_control_alignment'],
                results['generation_memory_alignment'],
                results['control_memory_alignment']
            ]
            results['overall_coordination'] = float(np.mean(scores))
            
            logger.info(f"Triple head coordination score: {results['overall_coordination']:.3f}")
            
        except Exception as e:
            logger.error(f"Error evaluating coordination: {e}")
            results['error'] = str(e)
        
        return results
    
    def _evaluate_generation_control_alignment(
        self,
        generation: str,
        control: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if generated text matches control tokens.
        
        Args:
            generation: Generated text
            control: Control head outputs
            
        Returns:
            Alignment score and failures
        """
        results = {'score': 0.0, 'failures': []}
        
        if not control:
            results['score'] = 1.0  # No control tokens to match
            return results
        
        generation_lower = generation.lower()
        alignment_scores = []
        
        # Check emotion alignment
        if 'emotion' in control:
            target_emotion = control['emotion']
            emotion_score = self._check_emotion_in_text(generation_lower, target_emotion)
            alignment_scores.append(emotion_score)
            
            if emotion_score < 0.3:
                results['failures'].append({
                    'type': 'emotion_mismatch',
                    'expected': target_emotion,
                    'text_snippet': generation[:50] + '...'
                })
        
        # Check tone alignment
        if 'tone' in control:
            target_tone = control['tone']
            tone_score = self._check_tone_in_text(generation_lower, target_tone)
            alignment_scores.append(tone_score)
            
            if tone_score < 0.3:
                results['failures'].append({
                    'type': 'tone_mismatch',
                    'expected': target_tone,
                    'text_snippet': generation[:50] + '...'
                })
        
        # Check confidence alignment
        if 'confidence' in control:
            confidence = control['confidence']
            # High confidence should correlate with assertive language
            if confidence > 0.8:
                assertive_score = self._check_assertiveness(generation)
                alignment_scores.append(assertive_score)
            elif confidence < 0.3:
                uncertain_score = self._check_uncertainty(generation)
                alignment_scores.append(uncertain_score)
            else:
                alignment_scores.append(0.7)  # Neutral confidence is usually OK
        
        results['score'] = float(np.mean(alignment_scores)) if alignment_scores else 1.0
        
        return results
    
    def _evaluate_generation_memory_alignment(
        self,
        generation: str,
        memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if generated text properly uses retrieved memories.
        
        Args:
            generation: Generated text
            memory: Memory head outputs
            
        Returns:
            Alignment score and failures
        """
        results = {'score': 0.0, 'failures': []}
        
        if not memory:
            results['score'] = 1.0
            return results
        
        generation_lower = generation.lower()
        alignment_scores = []
        
        # Check if retrieved memories are referenced
        retrieved = memory.get('retrieved', [])
        if retrieved:
            referenced_count = 0
            for mem in retrieved:
                # Simple check: does the generation reference the memory content?
                mem_keywords = self._extract_keywords(mem)
                if any(keyword in generation_lower for keyword in mem_keywords):
                    referenced_count += 1
            
            reference_rate = referenced_count / len(retrieved) if retrieved else 0
            alignment_scores.append(reference_rate)
            
            if reference_rate < 0.5:
                results['failures'].append({
                    'type': 'memory_not_used',
                    'retrieved_memories': len(retrieved),
                    'referenced': referenced_count
                })
        
        # Check if formed memories are appropriate
        formed = memory.get('formed', [])
        if formed:
            appropriate_count = 0
            for mem in formed:
                # Check if the formed memory relates to the generation
                if self._is_memory_appropriate(mem, generation):
                    appropriate_count += 1
            
            appropriateness_rate = appropriate_count / len(formed) if formed else 0
            alignment_scores.append(appropriateness_rate)
            
            if appropriateness_rate < 0.7:
                results['failures'].append({
                    'type': 'inappropriate_memory_formation',
                    'formed_count': len(formed),
                    'appropriate': appropriate_count
                })
        
        results['score'] = float(np.mean(alignment_scores)) if alignment_scores else 1.0
        
        return results
    
    def _evaluate_control_memory_alignment(
        self,
        control: Dict[str, Any],
        memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if control tokens and memory operations are consistent.
        
        Args:
            control: Control head outputs
            memory: Memory head outputs
            
        Returns:
            Alignment score and failures
        """
        results = {'score': 0.0, 'failures': []}
        
        if not control or not memory:
            results['score'] = 1.0
            return results
        
        alignment_scores = []
        
        # Check importance alignment
        control_confidence = control.get('confidence', 0.5)
        memory_importance = memory.get('importance', 0.5)
        
        # High confidence should correlate with high importance memories
        importance_diff = abs(control_confidence - memory_importance)
        importance_alignment = 1.0 - importance_diff
        alignment_scores.append(importance_alignment)
        
        if importance_diff > 0.5:
            results['failures'].append({
                'type': 'confidence_importance_mismatch',
                'control_confidence': control_confidence,
                'memory_importance': memory_importance
            })
        
        # Check emotion consistency in formed memories
        if 'emotion' in control and 'formed' in memory:
            target_emotion = control['emotion']
            formed_memories = memory['formed']
            
            emotion_consistent = 0
            for mem in formed_memories:
                if self._memory_matches_emotion(mem, target_emotion):
                    emotion_consistent += 1
            
            consistency_rate = emotion_consistent / len(formed_memories) if formed_memories else 1.0
            alignment_scores.append(consistency_rate)
            
            if consistency_rate < 0.5:
                results['failures'].append({
                    'type': 'memory_emotion_inconsistency',
                    'expected_emotion': target_emotion,
                    'consistent_memories': emotion_consistent,
                    'total_memories': len(formed_memories)
                })
        
        results['score'] = float(np.mean(alignment_scores)) if alignment_scores else 1.0
        
        return results
    
    def _check_emotion_in_text(self, text: str, emotion: str) -> float:
        """Check if text expresses the target emotion"""
        if emotion not in self.emotion_keywords:
            return 0.5  # Unknown emotion
        
        keywords = self.emotion_keywords[emotion]
        matches = sum(1 for keyword in keywords if keyword in text)
        
        # Also check for contradicting emotions
        contradictions = 0
        for other_emotion, other_keywords in self.emotion_keywords.items():
            if other_emotion != emotion and other_emotion != 'neutral':
                contradictions += sum(1 for keyword in other_keywords if keyword in text)
        
        if contradictions > matches:
            return 0.2  # Text expresses conflicting emotion
        
        return min(1.0, matches / 2)  # Normalize to 0-1
    
    def _check_tone_in_text(self, text: str, tone: str) -> float:
        """Check if text matches the target tone"""
        if tone not in self.tone_indicators:
            return 0.5  # Unknown tone
        
        indicators = self.tone_indicators[tone]
        matches = sum(1 for indicator in indicators if indicator in text)
        
        return min(1.0, matches / 2)  # Normalize to 0-1
    
    def _check_assertiveness(self, text: str) -> float:
        """Check if text shows assertiveness (for high confidence)"""
        assertive_patterns = [
            r'\bcertainly\b', r'\bdefinitely\b', r'\babsolutely\b',
            r'\bclearly\b', r'\bobviously\b', r'\bI know\b'
        ]
        
        matches = sum(1 for pattern in assertive_patterns if re.search(pattern, text, re.I))
        return min(1.0, matches / 2)
    
    def _check_uncertainty(self, text: str) -> float:
        """Check if text shows uncertainty (for low confidence)"""
        uncertain_patterns = [
            r'\bmaybe\b', r'\bperhaps\b', r'\bpossibly\b',
            r'\bI think\b', r'\bI guess\b', r'\bnot sure\b',
            r'\bmight\b', r'\bcould be\b'
        ]
        
        matches = sum(1 for pattern in uncertain_patterns if re.search(pattern, text, re.I))
        return min(1.0, matches / 2)
    
    def _extract_keywords(self, memory_text: str) -> List[str]:
        """Extract key words from memory text"""
        # Simple keyword extraction
        words = memory_text.lower().split()
        # Filter out common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'}
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return keywords[:5]  # Top 5 keywords
    
    def _is_memory_appropriate(self, memory: str, generation: str) -> bool:
        """Check if a formed memory is appropriate given the generation"""
        # Simple heuristic: memory should contain some keywords from generation
        gen_keywords = set(self._extract_keywords(generation))
        mem_keywords = set(self._extract_keywords(memory))
        
        overlap = len(gen_keywords & mem_keywords)
        return overlap >= 2  # At least 2 common keywords
    
    def _memory_matches_emotion(self, memory: str, emotion: str) -> bool:
        """Check if memory content matches the target emotion"""
        if emotion not in self.emotion_keywords:
            return True  # Can't verify unknown emotion
        
        memory_lower = memory.lower()
        emotion_keywords = self.emotion_keywords[emotion]
        
        # Check if memory contains emotion-related words
        return any(keyword in memory_lower for keyword in emotion_keywords) 