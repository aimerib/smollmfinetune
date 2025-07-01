"""
World Consistency Evaluation

Measures adherence to world lore and consistency of world facts
across conversations and character interactions.
"""

import logging
from typing import List, Dict, Any, Optional, Set
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class WorldConsistencyEvaluator:
    """Evaluates world consistency and lore adherence"""
    
    def __init__(self, world_lore: Optional[Dict[str, Any]] = None):
        """
        Initialize the world consistency evaluator.
        
        Args:
            world_lore: Dictionary containing world facts, rules, and constraints
        """
        self.world_lore = world_lore or {}
        self.lore_embeddings = None  # Would be initialized with world knowledge embeddings
        self._initialize_lore_constraints()
    
    def _initialize_lore_constraints(self):
        """Extract and process lore constraints from world data"""
        self.forbidden_items = set(self.world_lore.get('forbidden_items', []))
        self.technology_level = self.world_lore.get('technology_level', 'modern')
        self.setting = self.world_lore.get('setting', 'contemporary')
        self.magic_exists = self.world_lore.get('magic_exists', False)
        self.key_locations = set(self.world_lore.get('key_locations', []))
        
        # Build technology constraints based on level
        self.tech_constraints = self._build_tech_constraints()
    
    def _build_tech_constraints(self) -> Set[str]:
        """Build technology constraints based on world's tech level"""
        constraints = set()
        
        if self.technology_level == 'pre-industrial':
            constraints.update([
                'computer', 'phone', 'car', 'television', 'radio',
                'internet', 'electricity', 'gun', 'airplane', 'train'
            ])
        elif self.technology_level == 'industrial':
            constraints.update([
                'computer', 'phone', 'internet', 'television',
                'airplane', 'satellite', 'smartphone'
            ])
        elif self.technology_level == 'modern':
            constraints.update(['quantum computer', 'teleporter', 'time machine'])
        
        return constraints
    
    def check_lore_adherence(
        self,
        dialogue: List[str]
    ) -> Dict[str, Any]:
        """
        Check dialogue for adherence to world lore.
        
        Args:
            dialogue: List of dialogue utterances to check
            
        Returns:
            Dictionary with lore consistency metrics
        """
        results = {
            'lore_consistency_score': 1.0,
            'violations': [],
            'violation_severity': 'none'
        }
        
        if not dialogue:
            return results
        
        try:
            violations = []
            
            for i, utterance in enumerate(dialogue):
                utterance_lower = utterance.lower()
                
                # Check for forbidden items
                for item in self.forbidden_items:
                    if item.lower() in utterance_lower:
                        violations.append({
                            'type': 'forbidden_item',
                            'item': item,
                            'utterance_index': i,
                            'text': utterance[:100] + '...' if len(utterance) > 100 else utterance
                        })
                
                # Check technology constraints
                for tech in self.tech_constraints:
                    if tech in utterance_lower:
                        violations.append({
                            'type': 'technology_violation',
                            'technology': tech,
                            'utterance_index': i,
                            'text': utterance[:100] + '...' if len(utterance) > 100 else utterance
                        })
                
                # Check magic consistency
                if not self.magic_exists:
                    magic_words = ['spell', 'magic', 'wizard', 'sorcerer', 'enchant']
                    for magic_word in magic_words:
                        if magic_word in utterance_lower:
                            violations.append({
                                'type': 'magic_violation',
                                'word': magic_word,
                                'utterance_index': i,
                                'text': utterance[:100] + '...' if len(utterance) > 100 else utterance
                            })
            
            results['violations'] = violations
            
            # Calculate consistency score
            violation_count = len(violations)
            total_utterances = len(dialogue)
            
            if violation_count == 0:
                results['lore_consistency_score'] = 1.0
            else:
                # Deduct score based on violations
                penalty = min(violation_count * 0.2, 0.9)  # Max 90% penalty
                results['lore_consistency_score'] = max(0.1, 1.0 - penalty)
            
            # Determine severity
            if violation_count == 0:
                results['violation_severity'] = 'none'
            elif violation_count <= 2:
                results['violation_severity'] = 'low'
            elif violation_count <= 5:
                results['violation_severity'] = 'medium'
            else:
                results['violation_severity'] = 'high'
            
            logger.info(f"Lore consistency check complete: {violation_count} violations found")
            
        except Exception as e:
            logger.error(f"Error checking lore adherence: {e}")
            results['error'] = str(e)
        
        return results
    
    def evaluate_fact_consistency(
        self,
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate consistency of world facts across multiple conversations.
        
        Args:
            conversations: List of conversations with mentioned facts
            
        Returns:
            Dictionary with fact consistency metrics
        """
        results = {
            'fact_consistency_score': 1.0,
            'contradictions': [],
            'fact_mentions': defaultdict(list)
        }
        
        if not conversations:
            return results
        
        try:
            # Collect all fact mentions
            fact_tracker = defaultdict(list)
            
            for conv in conversations:
                conv_id = conv.get('id', 'unknown')
                facts = conv.get('facts_mentioned', {})
                
                for fact_type, fact_value in facts.items():
                    fact_tracker[fact_type].append({
                        'conversation_id': conv_id,
                        'value': fact_value
                    })
            
            # Check for contradictions
            contradictions = []
            for fact_type, mentions in fact_tracker.items():
                unique_values = set(m['value'] for m in mentions)
                
                if len(unique_values) > 1:
                    contradictions.append({
                        'fact_type': fact_type,
                        'conflicting_values': list(unique_values),
                        'conversations': [m['conversation_id'] for m in mentions]
                    })
            
            results['contradictions'] = contradictions
            results['fact_mentions'] = dict(fact_tracker)
            
            # Calculate consistency score
            if contradictions:
                # Penalize based on number and severity of contradictions
                penalty = len(contradictions) * 0.15
                results['fact_consistency_score'] = max(0.1, 1.0 - penalty)
            else:
                results['fact_consistency_score'] = 1.0
            
            logger.info(f"Fact consistency evaluation: {len(contradictions)} contradictions found")
            
        except Exception as e:
            logger.error(f"Error evaluating fact consistency: {e}")
            results['error'] = str(e)
        
        return results
    
    def validate_location_references(
        self,
        dialogue: List[str]
    ) -> Dict[str, Any]:
        """
        Validate that location references match known world locations.
        
        Args:
            dialogue: List of dialogue utterances
            
        Returns:
            Dictionary with location validation results
        """
        results = {
            'valid_locations': [],
            'unknown_locations': [],
            'location_consistency_score': 1.0
        }
        
        if not self.key_locations:
            logger.info("No key locations defined for world")
            return results
        
        try:
            mentioned_locations = []
            
            # Simple location extraction (would be more sophisticated in production)
            for utterance in dialogue:
                # Look for capitalized words that might be locations
                words = utterance.split()
                for i, word in enumerate(words):
                    if word[0].isupper() and len(word) > 2:
                        # Check if it's a known location
                        if word in self.key_locations:
                            results['valid_locations'].append(word)
                        else:
                            # Check for multi-word locations
                            if i < len(words) - 1:
                                two_word = f"{word} {words[i+1]}"
                                if two_word in self.key_locations:
                                    results['valid_locations'].append(two_word)
                                elif words[i+1][0].isupper():
                                    # Potential unknown location
                                    potential_location = two_word
                                    if potential_location not in results['unknown_locations']:
                                        results['unknown_locations'].append(potential_location)
            
            # Calculate score based on valid vs unknown locations
            total_locations = len(results['valid_locations']) + len(results['unknown_locations'])
            if total_locations > 0:
                valid_ratio = len(results['valid_locations']) / total_locations
                results['location_consistency_score'] = valid_ratio
            
        except Exception as e:
            logger.error(f"Error validating location references: {e}")
            results['error'] = str(e)
        
        return results 