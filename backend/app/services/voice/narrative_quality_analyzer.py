"""
Narrative Quality Analyzer

Analyzes narrative context from text and conversation to determine
appropriate voice generation quality levels. Uses LLM-based analysis
and rule-based heuristics.
"""

import re
import asyncio
from typing import Dict, Any, List, Optional
from backend.app.services.voice.quality_models import NarrativeContext, QualityLevel


class NarrativeQualityAnalyzer:
    """Analyzes narrative context for quality determination"""
    
    def __init__(self):
        """Initialize the analyzer with weights and mappings"""
        # Weights for importance calculation
        self.tension_weight = 0.4
        self.emotion_weight = 0.3
        self.importance_weight = 0.3
        
        # Dialogue type mappings
        self.dialogue_type_mapping = {
            "casual": ["hello", "hey", "how are you", "weather", "nice day"],
            "urgent": ["look out", "hurry", "quick", "now", "emergency", "help"],
            "emotional": ["love", "sorry", "please", "forgive", "heart", "feel"],
            "dramatic": ["never", "betrayed", "destiny", "shall not", "final"],
            "informational": ["data", "suggests", "according", "research", "evidence"]
        }
        
        # Scene type mappings  
        self.scene_type_mapping = {
            "conversation": ["enters room", "normal lighting", "sitting", "talking"],
            "action": ["explosions", "running", "fighting", "urgent music", "chaos"],
            "intimate": ["alone", "soft lighting", "whisper", "close", "private"],
            "climax": ["revelation", "dramatic", "all characters", "final moment"]
        }
        
        # Emotional intensity indicators
        self.high_emotion_patterns = [
            r'[A-Z]{2,}',  # ALL CAPS
            r'!{2,}',      # Multiple exclamation marks
            r'\?{2,}',     # Multiple question marks  
            r'\.{3,}',     # Ellipsis indicating trailing off
            r'[*]',        # Asterisk actions
        ]
        
        # Tension level indicators
        self.high_tension_words = [
            "danger", "threat", "enemy", "attack", "emergency", "crisis",
            "betrayal", "death", "kill", "destroy", "escape", "trapped"
        ]
    
    async def analyze_text_for_context(
        self,
        text: str,
        conversation_history: List[str],
        character_role: str
    ) -> NarrativeContext:
        """
        Analyze text to extract narrative context
        
        Args:
            text: The text to analyze
            conversation_history: Previous messages for context
            character_role: Role of the character speaking
            
        Returns:
            NarrativeContext with analyzed metrics
        """
        # Calculate emotional intensity
        emotional_intensity = self._calculate_emotional_intensity(text)
        
        # Calculate tension level
        tension_level = self._calculate_tension_level(text, conversation_history)
        
        # Determine narrative importance based on character role and content
        narrative_importance = self._calculate_narrative_importance(text, character_role)
        
        # Classify dialogue type
        dialogue_type = self.classify_dialogue_type(text)
        
        # Detect scene type from context
        scene_type = self.detect_scene_type(conversation_history + [text])
        
        # Determine character focus
        character_focus = self._determine_character_focus(character_role)
        
        return NarrativeContext(
            tension_level=tension_level,
            emotional_intensity=emotional_intensity,
            narrative_importance=narrative_importance,
            dialogue_type=dialogue_type,
            scene_type=scene_type,
            character_focus=character_focus
        )
    
    async def calculate_importance_score(self, context: NarrativeContext) -> float:
        """Calculate overall narrative importance score"""
        weighted_score = (
            context.tension_level * self.tension_weight +
            context.emotional_intensity * self.emotion_weight +
            context.narrative_importance * self.importance_weight
        )
        
        # Boost for high-priority dialogue and scene types
        if context.dialogue_type in ["dramatic", "urgent", "emotional"]:
            weighted_score += 0.1
        
        if context.scene_type in ["climax", "crisis", "dramatic_revelation"]:
            weighted_score += 0.15
        
        if context.character_focus in ["protagonist", "main_character"]:
            weighted_score += 0.1
        
        return min(1.0, weighted_score)
    
    def classify_dialogue_type(self, text: str) -> str:
        """Classify the type of dialogue"""
        text_lower = text.lower()
        
        # Check for patterns in dialogue types
        for dialogue_type, keywords in self.dialogue_type_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                return dialogue_type
        
        # Additional pattern matching
        if any(re.search(pattern, text) for pattern in self.high_emotion_patterns):
            return "emotional"
        
        if len(text) > 50 and text.count('!') == 0:
            return "informational"
        
        return "casual"  # Default
    
    def detect_scene_type(self, context_clues: List[str]) -> str:
        """Detect scene type from context clues"""
        all_context = " ".join(context_clues).lower()
        
        # Check for scene type indicators
        for scene_type, indicators in self.scene_type_mapping.items():
            if any(indicator in all_context for indicator in indicators):
                return scene_type
        
        # Default scene type detection
        if any(word in all_context for word in ["fight", "battle", "run", "chase"]):
            return "action"
        elif any(word in all_context for word in ["quiet", "alone", "intimate"]):
            return "intimate"
        elif any(word in all_context for word in ["final", "end", "climax", "revelation"]):
            return "climax"
        
        return "conversation"  # Default
    
    def _calculate_emotional_intensity(self, text: str) -> float:
        """Calculate emotional intensity from text patterns"""
        intensity = 0.0
        
        # Check for emotional intensity patterns
        for pattern in self.high_emotion_patterns:
            matches = len(re.findall(pattern, text))
            intensity += matches * 0.2
        
        # Check for emotional words
        emotional_words = ["love", "hate", "angry", "sad", "joy", "fear", "excited", "devastated"]
        for word in emotional_words:
            if word in text.lower():
                intensity += 0.15
        
        # Length and punctuation intensity
        if len(text) > 100:
            intensity += 0.1
        
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            intensity += min(0.3, exclamation_count * 0.1)
        
        return min(1.0, intensity)
    
    def _calculate_tension_level(self, text: str, conversation_history: List[str]) -> float:
        """Calculate tension level from content and context"""
        tension = 0.0
        text_lower = text.lower()
        
        # Check for high tension words
        for word in self.high_tension_words:
            if word in text_lower:
                tension += 0.15
        
        # Check conversation history for building tension
        recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        history_text = " ".join(recent_history).lower()
        
        if any(word in history_text for word in ["danger", "threat", "problem", "urgent"]):
            tension += 0.2
        
        # Pattern-based tension detection
        if re.search(r'[A-Z]{3,}', text):  # ALL CAPS words
            tension += 0.25
        
        if text.count('!') >= 2:
            tension += 0.2
        
        # Question patterns indicating uncertainty/tension
        if '?' in text and any(word in text_lower for word in ["what", "why", "how", "where"]):
            tension += 0.1
        
        return min(1.0, tension)
    
    def _calculate_narrative_importance(self, text: str, character_role: str) -> float:
        """Calculate narrative importance based on content and character"""
        importance = 0.0
        
        # Base importance from character role
        role_importance = {
            "protagonist": 0.8,
            "main": 0.7,
            "supporting": 0.4,
            "background": 0.1
        }
        
        for role, score in role_importance.items():
            if role in character_role.lower():
                importance += score
                break
        else:
            importance += 0.3  # Default for unknown roles
        
        # Content-based importance
        important_words = [
            "discover", "reveal", "secret", "truth", "plan", "mission",
            "destiny", "prophecy", "key", "answer", "solution"
        ]
        
        text_lower = text.lower()
        for word in important_words:
            if word in text_lower:
                importance += 0.1
        
        # Length and complexity can indicate importance
        if len(text) > 200:
            importance += 0.15
        
        return min(1.0, importance)
    
    def _determine_character_focus(self, character_role: str) -> str:
        """Determine character focus category"""
        role_lower = character_role.lower()
        
        if "protagonist" in role_lower or "main" in role_lower:
            return "protagonist"
        elif "background" in role_lower or "npc" in role_lower:
            return "background"
        elif "two" in role_lower or "pair" in role_lower:
            return "two_characters"
        elif "multiple" in role_lower or "group" in role_lower:
            return "multiple"
        else:
            return "supporting" 