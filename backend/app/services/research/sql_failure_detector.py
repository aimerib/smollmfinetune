"""
🔍 SQL-Based Failure Mode Detection for R3-3 Analysis

This module implements failure detection algorithms using direct SQL queries
to avoid ORM schema issues and focus on the core research.
"""

import sqlite3
import re
import json
from typing import Dict, List, Any, Tuple
from collections import Counter
from datetime import datetime
from dataclasses import dataclass


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


class SQLFailureDetector:
    """Detects failure modes using direct SQL queries"""
    
    def __init__(self, db_path: str = "platform.db"):
        """Initialize the detector with database path"""
        self.db_path = db_path
        
        # Detection thresholds
        self.thresholds = {
            'repetition_similarity': 0.85,
            'meta_commentary_confidence': 0.8,
        }
        
        # Meta-commentary patterns
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
            r'\bhelp.*harmless\b',
            r'\bartificial intelligence\b'
        ]
    
    def get_conversation_messages(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all messages for a conversation session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM conversation_messages 
                WHERE session_id = ? 
                ORDER BY sequence_number
            """, (session_id,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def analyze_conversation_session(self, session_id: int) -> List[FailureDetection]:
        """Analyze a single conversation session for failures"""
        messages = self.get_conversation_messages(session_id)
        
        if not messages:
            return []
        
        failures = []
        
        # Analyze different failure modes
        failures.extend(self._detect_repetition_failures(messages, session_id))
        failures.extend(self._detect_memory_failures(messages, session_id))
        failures.extend(self._detect_meta_commentary(messages, session_id))
        failures.extend(self._detect_personality_drift(messages, session_id))
        failures.extend(self._detect_goal_inconsistency(messages, session_id))
        failures.extend(self._detect_emotional_inconsistency(messages, session_id))
        
        return failures
    
    def _detect_repetition_failures(self, messages: List[Dict[str, Any]], session_id: int) -> List[FailureDetection]:
        """Detect when character repeats the same response"""
        failures = []
        character_messages = [msg for msg in messages if msg['role'] == 'assistant']
        
        for i, msg in enumerate(character_messages):
            for j, other_msg in enumerate(character_messages[i+1:], i+1):
                # Calculate text similarity
                similarity = self._calculate_text_similarity(msg['content'], other_msg['content'])
                
                if similarity > self.thresholds['repetition_similarity']:
                    failures.append(FailureDetection(
                        failure_type='repetition',
                        confidence=similarity,
                        message_id=other_msg['message_id'],
                        session_id=session_id,
                        turn_number=other_msg['sequence_number'] // 2,
                        evidence={
                            'original_message': msg['content'],
                            'repeated_message': other_msg['content'],
                            'similarity_score': similarity,
                            'turns_apart': (other_msg['sequence_number'] - msg['sequence_number']) // 2
                        },
                        severity='high' if similarity > 0.95 else 'medium',
                        description=f"Character repeated similar content (similarity: {similarity:.2f})"
                    ))
        
        return failures
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two text strings"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_memory_failures(self, messages: List[Dict[str, Any]], session_id: int) -> List[FailureDetection]:
        """Detect when character explicitly indicates memory failure"""
        failures = []
        
        memory_failure_phrases = [
            "what were we talking about",
            "i can't remember",
            "i cant remember",
            "i forgot",
            "remind me",
            "what did you say",
            "i don't recall",
            "i dont recall"
        ]
        
        for msg in messages:
            if msg['role'] == 'assistant':
                content_lower = msg['content'].lower()
                
                for phrase in memory_failure_phrases:
                    if phrase in content_lower:
                        failures.append(FailureDetection(
                            failure_type='memory_failure',
                            confidence=0.9,
                            message_id=msg['message_id'],
                            session_id=session_id,
                            turn_number=msg['sequence_number'] // 2,
                            evidence={
                                'trigger_phrase': phrase,
                                'message_content': msg['content']
                            },
                            severity='high',
                            description=f"Character explicitly indicated memory failure: '{phrase}'"
                        ))
                        break
        
        return failures
    
    def _detect_meta_commentary(self, messages: List[Dict[str, Any]], session_id: int) -> List[FailureDetection]:
        """Detect when character breaks character with AI/meta commentary"""
        failures = []
        
        for msg in messages:
            if msg['role'] == 'assistant':
                content_lower = msg['content'].lower()
                
                for pattern in self.meta_patterns:
                    matches = re.findall(pattern, content_lower, re.IGNORECASE)
                    if matches:
                        confidence = min(0.9, 0.6 + len(matches) * 0.1)
                        
                        failures.append(FailureDetection(
                            failure_type='meta_commentary',
                            confidence=confidence,
                            message_id=msg['message_id'],
                            session_id=session_id,
                            turn_number=msg['sequence_number'] // 2,
                            evidence={
                                'matched_patterns': matches,
                                'pattern_used': pattern,
                                'message_content': msg['content']
                            },
                            severity='critical',
                            description=f"Character made meta-commentary about being AI: {matches}"
                        ))
        
        return failures
    
    def _detect_personality_drift(self, messages: List[Dict[str, Any]], session_id: int) -> List[FailureDetection]:
        """Detect when character acts inconsistent with expected personality traits"""
        failures = []
        
        # Get character personality data
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.openness, c.conscientiousness, c.extraversion, c.agreeableness, c.neuroticism
                FROM characters c
                JOIN conversation_messages cm ON c.id = cm.character_id
                WHERE cm.session_id = ?
                LIMIT 1
            """, (session_id,))
            
            result = cursor.fetchone()
            if not result:
                return failures
            
            expected_traits = {
                'openness': result[0],
                'conscientiousness': result[1], 
                'extraversion': result[2],
                'agreeableness': result[3],
                'neuroticism': result[4]
            }
        
        # Personality indicators
        personality_keywords = {
            'openness_high': ['creative', 'imaginative', 'curious', 'artistic', 'novel', 'explore', 'abstract'],
            'openness_low': ['practical', 'conventional', 'traditional', 'familiar', 'routine'],
            'extraversion_high': ['social', 'outgoing', 'energetic', 'talkative', 'party', 'people', 'crowd'],
            'extraversion_low': ['quiet', 'reserved', 'alone', 'solitude', 'introvert', 'shy'],
            'agreeableness_high': ['kind', 'helpful', 'cooperative', 'trust', 'nice', 'pleasant', 'friendly'],
            'agreeableness_low': ['skeptical', 'critical', 'argue', 'compete', 'disagree', 'stubborn'],
            'conscientiousness_high': ['organized', 'plan', 'schedule', 'discipline', 'goal', 'careful', 'precise'],
            'conscientiousness_low': ['spontaneous', 'flexible', 'random', 'improvise', 'messy'],
            'neuroticism_high': ['anxious', 'worry', 'stress', 'nervous', 'emotional', 'upset', 'fear'],
            'neuroticism_low': ['calm', 'stable', 'relaxed', 'confident', 'peaceful', 'secure']
        }
        
        for msg in messages:
            if msg['role'] == 'assistant':
                content_lower = msg['content'].lower()
                
                # Analyze each trait
                for trait in ['openness', 'extraversion', 'agreeableness', 'conscientiousness', 'neuroticism']:
                    expected_level = expected_traits[trait]
                    
                    high_indicators = sum(1 for word in personality_keywords[f'{trait}_high'] if word in content_lower)
                    low_indicators = sum(1 for word in personality_keywords[f'{trait}_low'] if word in content_lower)
                    
                    if high_indicators + low_indicators > 0:
                        detected_level = high_indicators / (high_indicators + low_indicators)
                        deviation = abs(expected_level - detected_level)
                        
                        # Significant deviation threshold
                        if deviation > 0.4 and (high_indicators + low_indicators) >= 2:
                            failures.append(FailureDetection(
                                failure_type='personality_drift',
                                confidence=min(0.9, deviation * 1.5),
                                message_id=msg['message_id'],
                                session_id=session_id,
                                turn_number=msg['sequence_number'] // 2,
                                evidence={
                                    'trait': trait,
                                    'expected_level': expected_level,
                                    'detected_level': detected_level,
                                    'deviation': deviation,
                                    'high_indicators': high_indicators,
                                    'low_indicators': low_indicators,
                                    'message_content': msg['content'][:100]
                                },
                                severity='high' if deviation > 0.6 else 'medium',
                                description=f"{trait.title()} drift: expected {expected_level:.2f}, detected {detected_level:.2f} (deviation: {deviation:.2f})"
                            ))
        
        return failures
    
    def _detect_goal_inconsistency(self, messages: List[Dict[str, Any]], session_id: int) -> List[FailureDetection]:
        """Detect when character contradicts their stated goals or desires"""
        failures = []
        
        # Look for explicit goal contradictions
        contradiction_patterns = [
            r"(want|dream|goal|desire|hope).*but.*actually.*",
            r"i (want|love|dream).*wait.*i (hate|don't|dislike)",
            r"my goal.*no wait.*",
            r"i (always|really) wanted.*actually.*never",
            r"(love|enjoy).*but.*i (hate|can't stand)"
        ]
        
        for msg in messages:
            if msg['role'] == 'assistant':
                content_lower = msg['content'].lower()
                
                for pattern in contradiction_patterns:
                    if re.search(pattern, content_lower):
                        failures.append(FailureDetection(
                            failure_type='goal_inconsistency',
                            confidence=0.8,
                            message_id=msg['message_id'],
                            session_id=session_id,
                            turn_number=msg['sequence_number'] // 2,
                            evidence={
                                'contradiction_pattern': pattern,
                                'message_content': msg['content']
                            },
                            severity='medium',
                            description="Character contradicted their own goals or desires within the same statement"
                        ))
        
        return failures
    
    def _detect_emotional_inconsistency(self, messages: List[Dict[str, Any]], session_id: int) -> List[FailureDetection]:
        """Detect rapid or illogical emotional state changes"""
        failures = []
        
        # Emotional indicators
        positive_emotions = ['happy', 'excited', 'wonderful', 'great', 'love', 'joy', 'amazing', 'fantastic', 'delighted']
        negative_emotions = ['sad', 'upset', 'angry', 'frustrated', 'hate', 'terrible', 'awful', 'disappointed', 'furious']
        
        emotional_states = []
        
        # Track emotional states through conversation
        for msg in messages:
            if msg['role'] == 'assistant':
                content_lower = msg['content'].lower()
                
                pos_count = sum(1 for word in positive_emotions if word in content_lower)
                neg_count = sum(1 for word in negative_emotions if word in content_lower)
                
                if pos_count > 0 or neg_count > 0:
                    dominant = 'positive' if pos_count > neg_count else 'negative'
                    emotional_states.append({
                        'message': msg,
                        'positive': pos_count,
                        'negative': neg_count,
                        'dominant': dominant,
                        'intensity': pos_count + neg_count
                    })
        
        # Look for rapid emotional switches
        for i in range(1, len(emotional_states)):
            prev_state = emotional_states[i-1]
            curr_state = emotional_states[i]
            
            # Check for emotional whiplash (quick switches with high intensity)
            if (prev_state['dominant'] != curr_state['dominant'] and 
                curr_state['message']['sequence_number'] - prev_state['message']['sequence_number'] <= 4 and
                prev_state['intensity'] >= 2 and curr_state['intensity'] >= 2):
                
                failures.append(FailureDetection(
                    failure_type='emotional_inconsistency',
                    confidence=0.7,
                    message_id=curr_state['message']['message_id'],
                    session_id=session_id,
                    turn_number=curr_state['message']['sequence_number'] // 2,
                    evidence={
                        'previous_emotion': prev_state['dominant'],
                        'current_emotion': curr_state['dominant'],
                        'turns_between': (curr_state['message']['sequence_number'] - prev_state['message']['sequence_number']) // 2,
                        'prev_intensity': prev_state['intensity'],
                        'curr_intensity': curr_state['intensity']
                    },
                    severity='medium',
                    description=f"Rapid emotional switch from {prev_state['dominant']} to {curr_state['dominant']} in {(curr_state['message']['sequence_number'] - prev_state['message']['sequence_number']) // 2} turns"
                ))
        
        return failures
    
    def generate_failure_report(self, session_ids: List[int] = None) -> Dict[str, Any]:
        """Generate a comprehensive failure analysis report"""
        
        if session_ids is None:
            # Get all session IDs from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM play_sessions")
                session_ids = [row[0] for row in cursor.fetchall()]
        
        all_failures = []
        session_summaries = {}
        
        for session_id in session_ids:
            failures = self.analyze_conversation_session(session_id)
            all_failures.extend(failures)
            
            session_summaries[session_id] = {
                'total_failures': len(failures),
                'failure_types': Counter(f.failure_type for f in failures),
                'severity_distribution': Counter(f.severity for f in failures),
                'average_confidence': sum(f.confidence for f in failures) / len(failures) if failures else 0.0
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
            'session_summaries': {str(k): v for k, v in session_summaries.items()},
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
    
    def get_session_metadata(self, session_id: int) -> Dict[str, Any]:
        """Get metadata about a session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT ps.*, c.name as character_name, c.openness, c.conscientiousness, 
                       c.extraversion, c.agreeableness, c.neuroticism
                FROM play_sessions ps
                LEFT JOIN conversation_messages cm ON ps.id = cm.session_id
                LEFT JOIN characters c ON cm.character_id = c.id
                WHERE ps.id = ?
                LIMIT 1
            """, (session_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else {}


# Convenience functions
def analyze_session(session_id: int, db_path: str = "platform.db") -> List[FailureDetection]:
    """Quick function to analyze a single session"""
    detector = SQLFailureDetector(db_path)
    return detector.analyze_conversation_session(session_id)


def generate_report(db_path: str = "platform.db") -> Dict[str, Any]:
    """Quick function to generate a complete failure report"""
    detector = SQLFailureDetector(db_path)
    return detector.generate_failure_report()


if __name__ == "__main__":
    # Demo the failure detector
    detector = SQLFailureDetector()
    
    print("🔍 SQL-Based Failure Detection Demo")
    print("=" * 50)
    
    # Generate report
    report = detector.generate_failure_report()
    
    print(f"📊 Sessions analyzed: {report['total_sessions_analyzed']}")
    print(f"🚨 Total failures detected: {report['total_failures_detected']}")
    print(f"⚠️  Critical issues: {report['critical_issues']}")
    
    print(f"\n📈 Failure types:")
    for failure_type, count in report['failure_type_distribution'].items():
        print(f"  {failure_type}: {count}")
    
    print(f"\n🎯 High-confidence failures: {report['high_confidence_failures']}")
    
    if report['detailed_failures']:
        print(f"\n🔬 Sample failures:")
        for failure in report['detailed_failures'][:3]:  # Show first 3
            print(f"  • {failure['type'].upper()} (confidence: {failure['confidence']:.2f})")
            print(f"    {failure['description']}") 