"""
🔍 Failure Mode Detection for R3-3 Analysis

This module implements algorithms to automatically detect different types of
character AI failures in conversations, providing scientific metrics for
R4 architecture requirements.
"""

import re
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from backend.app.core.database.session import session_scope
from backend.app.core.database.models import ConversationMessage, Character, PlaySession

logger = logging.getLogger(__name__)


@dataclass
class FailureDetection:
    """Represents a detected failure in conversation"""
    failure_type: str
    confidence: float
    message_id: str
    session_id: int
    turn_number: int
    evidence: Dict[str, Any]
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str


class FailureDetector:
    """Detects various failure modes in character conversations"""
    
    def __init__(self):
        """Initialize the failure detector with thresholds and patterns"""
        
        # Detection thresholds
        self.thresholds = {
            'repetition_similarity': 0.85,  # Jaccard similarity threshold
            'memory_window': 10,  # How many turns to remember
            'personality_deviation': 0.3,  # Deviation from expected traits
            'meta_commentary_confidence': 0.8,
            'lore_contradiction_confidence': 0.7
        }
        
        # Patterns for meta-commentary detection
        self.meta_patterns = [
            r'\bAI\b',
            r'\bassistant\b',
            r'\bprogramm',
            r'\balgorithm',
            r'\btraining\b',
            r'\bmodel\b',
            r'\breal person\b',
            r'\bnot (actually|really) (a )?human\b',
            r'\bdata\b.*\bset\b',
            r'\bhelp.*harmless\b'
        ]
        
        # Personality trait indicators (simplified NLP)
        self.personality_indicators = {
            'openness': {
                'high': ['creative', 'imaginative', 'curious', 'artistic', 'novel', 'explore'],
                'low': ['practical', 'conventional', 'traditional', 'familiar']
            },
            'extraversion': {
                'high': ['social', 'outgoing', 'energetic', 'talkative', 'party', 'people'],
                'low': ['quiet', 'reserved', 'alone', 'solitude', 'introvert']
            },
            'agreeableness': {
                'high': ['kind', 'helpful', 'cooperative', 'trust', 'nice', 'pleasant'],
                'low': ['skeptical', 'critical', 'argue', 'compete', 'disagree']
            },
            'conscientiousness': {
                'high': ['organized', 'plan', 'schedule', 'discipline', 'goal', 'careful'],
                'low': ['spontaneous', 'flexible', 'random', 'improvise']
            },
            'neuroticism': {
                'high': ['anxious', 'worry', 'stress', 'nervous', 'emotional', 'upset'],
                'low': ['calm', 'stable', 'relaxed', 'confident', 'peaceful']
            }
        }
        
        # Lore contradiction patterns (simplified)
        self.lore_contradiction_patterns = [
            r'\bdragon\b',
            r'\bmagic\b',
            r'\bwizard\b',
            r'\b\d{4}\b.*\bborn\b',  # Year mentions
            r'\bcentur(y|ies)\b.*\b(ago|old)\b'
        ]
    
    def analyze_conversation_messages(self, session_id: int) -> List[FailureDetection]:
        """Analyze all messages in a conversation session for failures"""
        logger.info(f"Analyzing conversation session {session_id}")
        
        with session_scope() as session:
            messages = session.query(ConversationMessage).filter_by(
                session_id=session_id
            ).order_by(ConversationMessage.sequence_number).all()
            
            if not messages:
                logger.warning(f"No messages found for session {session_id}")
                return []
            
            # Get character info for personality analysis
            character_messages = [msg for msg in messages if msg.character_id]
            character = None
            if character_messages:
                character = session.query(Character).get(character_messages[0].character_id)
            
            failures = []
            
            # Analyze different failure modes
            failures.extend(self._detect_repetition_failures(messages))
            failures.extend(self._detect_memory_failures(messages))
            failures.extend(self._detect_meta_commentary(messages))
            failures.extend(self._detect_lore_contradictions(messages))
            
            if character:
                failures.extend(self._detect_personality_drift(messages, character))
                failures.extend(self._detect_goal_inconsistency(messages, character))
            
            failures.extend(self._detect_emotional_inconsistency(messages))
            
            logger.info(f"Detected {len(failures)} failures in session {session_id}")
            return failures
    
    def _detect_repetition_failures(self, messages: List[ConversationMessage]) -> List[FailureDetection]:
        """Detect when character repeats the same response"""
        failures = []
        character_messages = [msg for msg in messages if msg.role == 'assistant']
        
        for i, msg in enumerate(character_messages):
            for j, other_msg in enumerate(character_messages[i+1:], i+1):
                # Check for exact or near-exact repetition
                similarity = self._calculate_text_similarity(msg.content, other_msg.content)
                
                if similarity > self.thresholds['repetition_similarity']:
                    failures.append(FailureDetection(
                        failure_type='repetition',
                        confidence=similarity,
                        message_id=other_msg.message_id,
                        session_id=msg.session_id,
                        turn_number=other_msg.sequence_number // 2,
                        evidence={
                            'original_message': msg.content,
                            'repeated_message': other_msg.content,
                            'similarity_score': similarity,
                            'turns_apart': (other_msg.sequence_number - msg.sequence_number) // 2
                        },
                        severity='high' if similarity > 0.95 else 'medium',
                        description=f"Character repeated similar content (similarity: {similarity:.2f})"
                    ))
        
        return failures
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using Jaccard similarity"""
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_memory_failures(self, messages: List[ConversationMessage]) -> List[FailureDetection]:
        """Detect when character forgets previously mentioned information"""
        failures = []
        
        # Look for explicit memory failure indicators
        memory_failure_phrases = [
            "what were we talking about",
            "i can't remember",
            "i forgot",
            "remind me",
            "what did you say",
            "i don't recall"
        ]
        
        for msg in messages:
            if msg.role == 'assistant':
                content_lower = msg.content.lower()
                for phrase in memory_failure_phrases:
                    if phrase in content_lower:
                        failures.append(FailureDetection(
                            failure_type='memory_failure',
                            confidence=0.9,
                            message_id=msg.message_id,
                            session_id=msg.session_id,
                            turn_number=msg.sequence_number // 2,
                            evidence={
                                'trigger_phrase': phrase,
                                'message_content': msg.content
                            },
                            severity='high',
                            description=f"Character explicitly indicated memory failure: '{phrase}'"
                        ))
                        break
        
        return failures
    
    def _detect_meta_commentary(self, messages: List[ConversationMessage]) -> List[FailureDetection]:
        """Detect when character breaks character with AI/meta commentary"""
        failures = []
        
        for msg in messages:
            if msg.role == 'assistant':
                content_lower = msg.content.lower()
                
                for pattern in self.meta_patterns:
                    matches = re.findall(pattern, content_lower, re.IGNORECASE)
                    if matches:
                        confidence = min(0.9, 0.6 + len(matches) * 0.1)
                        
                        failures.append(FailureDetection(
                            failure_type='meta_commentary',
                            confidence=confidence,
                            message_id=msg.message_id,
                            session_id=msg.session_id,
                            turn_number=msg.sequence_number // 2,
                            evidence={
                                'matched_patterns': matches,
                                'pattern_used': pattern,
                                'message_content': msg.content
                            },
                            severity='critical',
                            description=f"Character made meta-commentary about being AI: {matches}"
                        ))
        
        return failures
    
    def _detect_personality_drift(self, messages: List[ConversationMessage], 
                                character: Character) -> List[FailureDetection]:
        """Detect when character acts inconsistent with defined personality"""
        failures = []
        
        # Get character's Big Five traits
        traits = {
            'openness': character.openness,
            'extraversion': character.extraversion,
            'agreeableness': character.agreeableness,
            'conscientiousness': character.conscientiousness,
            'neuroticism': character.neuroticism
        }
        
        for msg in messages:
            if msg.role == 'assistant':
                detected_traits = self._analyze_personality_indicators(msg.content)
                
                for trait, expected_level in traits.items():
                    if trait in detected_traits:
                        detected_level = detected_traits[trait]
                        deviation = abs(expected_level - detected_level)
                        
                        if deviation > self.thresholds['personality_deviation']:
                            failures.append(FailureDetection(
                                failure_type='personality_drift',
                                confidence=min(0.9, deviation * 2),
                                message_id=msg.message_id,
                                session_id=msg.session_id,
                                turn_number=msg.sequence_number // 2,
                                evidence={
                                    'trait': trait,
                                    'expected_level': expected_level,
                                    'detected_level': detected_level,
                                    'deviation': deviation,
                                    'message_content': msg.content
                                },
                                severity='high' if deviation > 0.5 else 'medium',
                                description=f"{trait.title()} drift: expected {expected_level:.2f}, detected {detected_level:.2f}"
                            ))
        
        return failures
    
    def _analyze_personality_indicators(self, text: str) -> Dict[str, float]:
        """Analyze text for personality trait indicators"""
        text_lower = text.lower()
        traits = {}
        
        for trait, indicators in self.personality_indicators.items():
            high_count = sum(1 for word in indicators['high'] if word in text_lower)
            low_count = sum(1 for word in indicators['low'] if word in text_lower)
            
            total_indicators = high_count + low_count
            if total_indicators > 0:
                # Simple scoring: more high indicators = higher trait level
                traits[trait] = high_count / total_indicators
        
        return traits
    
    def _detect_lore_contradictions(self, messages: List[ConversationMessage]) -> List[FailureDetection]:
        """Detect when character contradicts established world lore"""
        failures = []
        
        for msg in messages:
            if msg.role == 'assistant':
                content_lower = msg.content.lower()
                
                for pattern in self.lore_contradiction_patterns:
                    matches = re.findall(pattern, content_lower, re.IGNORECASE)
                    if matches:
                        failures.append(FailureDetection(
                            failure_type='lore_contradiction',
                            confidence=self.thresholds['lore_contradiction_confidence'],
                            message_id=msg.message_id,
                            session_id=msg.session_id,
                            turn_number=msg.sequence_number // 2,
                            evidence={
                                'contradiction_indicators': matches,
                                'pattern': pattern,
                                'message_content': msg.content
                            },
                            severity='high',
                            description=f"Potential lore contradiction detected: {matches}"
                        ))
        
        return failures
    
    def _detect_goal_inconsistency(self, messages: List[ConversationMessage], 
                                 character: Character) -> List[FailureDetection]:
        """Detect when character contradicts their stated goals"""
        failures = []
        
        # Look for explicit goal contradictions (simplified)
        contradiction_patterns = [
            r"(want|dream|goal).*but.*actually.*",
            r"i (want|love).*wait.*i (hate|don't)",
            r"my goal.*no wait.*"
        ]
        
        for msg in messages:
            if msg.role == 'assistant':
                content_lower = msg.content.lower()
                
                for pattern in contradiction_patterns:
                    if re.search(pattern, content_lower):
                        failures.append(FailureDetection(
                            failure_type='goal_inconsistency',
                            confidence=0.8,
                            message_id=msg.message_id,
                            session_id=msg.session_id,
                            turn_number=msg.sequence_number // 2,
                            evidence={
                                'contradiction_pattern': pattern,
                                'message_content': msg.content
                            },
                            severity='medium',
                            description="Character contradicted their own goals or desires"
                        ))
        
        return failures
    
    def _detect_emotional_inconsistency(self, messages: List[ConversationMessage]) -> List[FailureDetection]:
        """Detect when character's emotional state is inconsistent"""
        failures = []
        
        # Simplified emotional state detection
        positive_emotions = ['happy', 'excited', 'wonderful', 'great', 'love', 'joy']
        negative_emotions = ['sad', 'upset', 'angry', 'frustrated', 'hate', 'terrible']
        
        emotional_states = []
        
        for msg in messages:
            if msg.role == 'assistant':
                content_lower = msg.content.lower()
                
                pos_count = sum(1 for word in positive_emotions if word in content_lower)
                neg_count = sum(1 for word in negative_emotions if word in content_lower)
                
                if pos_count > 0 or neg_count > 0:
                    emotional_states.append({
                        'message': msg,
                        'positive': pos_count,
                        'negative': neg_count,
                        'dominant': 'positive' if pos_count > neg_count else 'negative'
                    })
        
        # Look for rapid emotional switches
        for i in range(1, len(emotional_states)):
            prev_state = emotional_states[i-1]
            curr_state = emotional_states[i]
            
            if (prev_state['dominant'] != curr_state['dominant'] and 
                curr_state['message'].sequence_number - prev_state['message'].sequence_number <= 4):
                
                failures.append(FailureDetection(
                    failure_type='emotional_inconsistency',
                    confidence=0.7,
                    message_id=curr_state['message'].message_id,
                    session_id=curr_state['message'].session_id,
                    turn_number=curr_state['message'].sequence_number // 2,
                    evidence={
                        'previous_emotion': prev_state['dominant'],
                        'current_emotion': curr_state['dominant'],
                        'turns_between': (curr_state['message'].sequence_number - prev_state['message'].sequence_number) // 2
                    },
                    severity='medium',
                    description=f"Rapid emotional switch from {prev_state['dominant']} to {curr_state['dominant']}"
                ))
        
        return failures
    
    def generate_failure_report(self, session_ids: List[int] = None) -> Dict[str, Any]:
        """Generate a comprehensive failure analysis report"""
        logger.info("Generating failure analysis report")
        
        if session_ids is None:
            with session_scope() as session:
                session_ids = [s.id for s in session.query(PlaySession).all()]
        
        all_failures = []
        session_summaries = {}
        
        for session_id in session_ids:
            failures = self.analyze_conversation_messages(session_id)
            all_failures.extend(failures)
            
            session_summaries[session_id] = {
                'total_failures': len(failures),
                'failure_types': Counter(f.failure_type for f in failures),
                'severity_distribution': Counter(f.severity for f in failures),
                'average_confidence': np.mean([f.confidence for f in failures]) if failures else 0.0
            }
        
        # Overall statistics
        failure_types = Counter(f.failure_type for f in all_failures)
        severity_counts = Counter(f.severity for f in all_failures)
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_sessions_analyzed': len(session_ids),
            'total_failures_detected': len(all_failures),
            'failure_type_distribution': dict(failure_types),
            'severity_distribution': dict(severity_counts),
            'session_summaries': session_summaries,
            'failure_rate_by_type': {
                failure_type: count / len(session_ids) 
                for failure_type, count in failure_types.items()
            },
            'critical_issues': len([f for f in all_failures if f.severity == 'critical']),
            'high_confidence_failures': len([f for f in all_failures if f.confidence > 0.8]),
            'detailed_failures': [
                {
                    'type': f.failure_type,
                    'confidence': f.confidence,
                    'severity': f.severity,
                    'session_id': f.session_id,
                    'turn': f.turn_number,
                    'description': f.description,
                    'evidence': f.evidence
                } for f in all_failures
            ]
        }
        
        return report


# Convenience functions for quick analysis
def analyze_session(session_id: int) -> List[FailureDetection]:
    """Quick function to analyze a single session"""
    detector = FailureDetector()
    return detector.analyze_conversation_messages(session_id)


def generate_quick_report() -> Dict[str, Any]:
    """Quick function to generate a failure report for all sessions"""
    detector = FailureDetector()
    return detector.generate_failure_report()


if __name__ == "__main__":
    # Demo the failure detector
    detector = FailureDetector()
    
    print("🔍 Failure Detection Demo")
    print("=" * 40)
    
    # Analyze all sessions
    report = detector.generate_failure_report()
    
    print(f"📊 Total sessions analyzed: {report['total_sessions_analyzed']}")
    print(f"🚨 Total failures detected: {report['total_failures_detected']}")
    print(f"⚠️  Critical issues: {report['critical_issues']}")
    
    print(f"\n📈 Failure types:")
    for failure_type, count in report['failure_type_distribution'].items():
        print(f"  {failure_type}: {count}")
    
    print(f"\n🎯 High-confidence failures: {report['high_confidence_failures']}")
    
    if report['detailed_failures']:
        print(f"\n🔬 Sample failure:")
        failure = report['detailed_failures'][0]
        print(f"  Type: {failure['type']} (confidence: {failure['confidence']:.2f})")
        print(f"  Description: {failure['description']}") 