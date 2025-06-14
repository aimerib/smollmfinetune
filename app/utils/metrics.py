import re
import logging
from typing import Dict, List, Any, Optional
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)


class CharacterConsistencyMetrics:
    """Evaluates character consistency and training quality metrics"""
    
    def __init__(self):
        self.personality_keywords = {
            'shy': ['nervous', 'blush', 'hesitant', 'quietly', 'timid', 'bashful'],
            'confident': ['bold', 'sure', 'certain', 'assertive', 'strong'],
            'playful': ['tease', 'laugh', 'giggle', 'fun', 'mischievous', 'playful'],
            'serious': ['focused', 'stern', 'grave', 'solemn', 'intense'],
            'kind': ['gentle', 'caring', 'sweet', 'warm', 'compassionate'],
            'aggressive': ['fierce', 'angry', 'hostile', 'violent', 'forceful'],
            'intelligent': ['analyze', 'consider', 'think', 'reason', 'understand'],
            'emotional': ['feel', 'heart', 'tears', 'passionate', 'sensitive']
        }
    
    def evaluate_character_consistency(self, sample: Dict[str, Any], character: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate how well a response matches character traits"""
        if 'messages' not in sample or len(sample['messages']) < 3:
            return self._get_zero_scores()
        
        response = sample['messages'][2]['content']
        user_prompt = sample['messages'][1]['content']
        
        scores = {}
        
        # 1. Name consistency (character shouldn't refer to themselves in third person)
        scores['name_consistency'] = self._check_name_consistency(response, character)
        
        # 2. Personality trait alignment
        scores['personality_alignment'] = self._check_personality_alignment(response, character)
        
        # 3. Speech pattern consistency
        scores['speech_pattern'] = self._check_speech_patterns(response, character)
        
        # 4. Response quality (length, coherence)
        scores['response_quality'] = self._check_response_quality(response, user_prompt)
        
        # 5. Meta-commentary detection (should be zero)
        scores['meta_commentary'] = self._check_meta_commentary(response)
        
        # 6. Character voice consistency
        scores['voice_consistency'] = self._check_voice_consistency(response, character)
        
        # Overall score (weighted average)
        weights = {
            'name_consistency': 0.25,
            'personality_alignment': 0.20,
            'speech_pattern': 0.15,
            'response_quality': 0.15,
            'meta_commentary': 0.10,  # Penalty
            'voice_consistency': 0.15
        }
        
        overall = sum(scores[key] * weights[key] for key in weights.keys())
        scores['overall_consistency'] = overall
        
        return scores
    
    def _check_name_consistency(self, response: str, character: Dict[str, Any]) -> float:
        """Check if character refers to themselves correctly"""
        char_name = character.get('name', '').lower()
        if not char_name:
            return 1.0
        
        response_lower = response.lower()
        
        # Penalty for third-person self-reference
        if char_name in response_lower:
            # Check if it's in dialogue (acceptable) vs narration (problematic)
            lines = response.split('\n')
            for line in lines:
                if char_name in line.lower() and not line.strip().startswith('"'):
                    return 0.0  # Likely third-person narration
        
        return 1.0
    
    def _check_personality_alignment(self, response: str, character: Dict[str, Any]) -> float:
        """Check if response aligns with character personality"""
        personality = character.get('personality', '').lower()
        if not personality:
            return 0.7  # Neutral score if no personality defined
        
        response_lower = response.lower()
        score = 0.0
        trait_count = 0
        
        # Check for personality traits
        for trait, keywords in self.personality_keywords.items():
            if trait in personality:
                trait_count += 1
                if any(keyword in response_lower for keyword in keywords):
                    score += 1.0
                elif any(opposite in response_lower for opposite in self._get_opposite_traits(trait)):
                    score -= 0.5  # Penalty for contradictory traits
        
        if trait_count == 0:
            return 0.7
        
        return max(0.0, min(1.0, score / trait_count))
    
    def _check_speech_patterns(self, response: str, character: Dict[str, Any]) -> float:
        """Check if speech patterns match character examples"""
        mes_example = character.get('mes_example', '')
        if not mes_example:
            return 0.7  # Neutral if no examples
        
        # Extract patterns from examples
        example_patterns = self._extract_speech_patterns(mes_example)
        response_patterns = self._extract_speech_patterns(response)
        
        # Calculate similarity
        if not example_patterns:
            return 0.7
        
        matches = sum(1 for pattern in response_patterns if pattern in example_patterns)
        return min(1.0, matches / len(example_patterns) + 0.3)  # Base score + bonus
    
    def _check_response_quality(self, response: str, user_prompt: str) -> float:
        """Evaluate response quality"""
        word_count = len(response.split())
        
        # Length scoring
        if word_count < 5:
            length_score = 0.0
        elif word_count < 10:
            length_score = 0.3
        elif word_count < 50:
            length_score = 1.0
        elif word_count < 200:
            length_score = 0.9
        else:
            length_score = 0.7  # Penalize overly long responses
        
        # Relevance scoring (simple keyword overlap)
        prompt_words = set(user_prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words.intersection(response_words))
        relevance_score = min(1.0, overlap / max(3, len(prompt_words) * 0.3))
        
        return (length_score * 0.6 + relevance_score * 0.4)
    
    def _check_meta_commentary(self, response: str) -> float:
        """Check for meta-commentary (returns penalty, 0 = good, 1 = bad)"""
        meta_phrases = [
            'as an ai', 'as a language model', 'i cannot', 'i\'m not able',
            'my training', 'as an assistant', 'i don\'t have', 'i\'m an ai'
        ]
        
        response_lower = response.lower()
        for phrase in meta_phrases:
            if phrase in response_lower:
                return 1.0  # Full penalty
        
        return 0.0  # No penalty
    
    def _check_voice_consistency(self, response: str, character: Dict[str, Any]) -> float:
        """Check overall voice and tone consistency"""
        # Check for action formatting consistency
        action_score = 0.7  # Default
        if '*' in response:
            # Good: uses action formatting
            action_score = 1.0
        
        # Check for emotional expression
        emotion_words = ['feel', 'emotion', 'heart', 'soul', 'passion', 'love', 'hate', 'fear', 'joy']
        has_emotion = any(word in response.lower() for word in emotion_words)
        emotion_score = 1.0 if has_emotion else 0.6
        
        return (action_score * 0.5 + emotion_score * 0.5)
    
    def _extract_speech_patterns(self, text: str) -> List[str]:
        """Extract speech patterns from text"""
        patterns = []
        
        # Check for specific patterns
        if '...' in text:
            patterns.append('trailing_off')
        if '!' in text and text.count('!') >= 2:
            patterns.append('excited_speech')
        if '?' in text and text.count('?') >= 2:
            patterns.append('questioning')
        if re.search(r'\*[^*]+\*', text):
            patterns.append('action_text')
        if any(word in text.lower() for word in ['um', 'uh', 'er']):
            patterns.append('hesitation')
        
        return patterns
    
    def _get_opposite_traits(self, trait: str) -> List[str]:
        """Get opposite personality traits for contradiction detection"""
        opposites = {
            'shy': ['bold', 'confident', 'assertive'],
            'confident': ['shy', 'nervous', 'hesitant'],
            'playful': ['serious', 'stern', 'grave'],
            'serious': ['playful', 'lighthearted', 'carefree'],
            'kind': ['cruel', 'harsh', 'mean'],
            'aggressive': ['gentle', 'peaceful', 'calm']
        }
        return opposites.get(trait, [])
    
    def _get_zero_scores(self) -> Dict[str, float]:
        """Return zero scores for invalid samples"""
        return {
            'name_consistency': 0.0,
            'personality_alignment': 0.0,
            'speech_pattern': 0.0,
            'response_quality': 0.0,
            'meta_commentary': 1.0,  # Full penalty
            'voice_consistency': 0.0,
            'overall_consistency': 0.0
        }
    
    def aggregate_metrics(self, all_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across multiple samples"""
        if not all_scores:
            return {}
        
        aggregated = {}
        for key in all_scores[0].keys():
            values = [scores[key] for scores in all_scores if key in scores]
            if values:
                aggregated[f'avg_{key}'] = sum(values) / len(values)
                aggregated[f'min_{key}'] = min(values)
                aggregated[f'max_{key}'] = max(values)
        
        return aggregated


class TrainingQualityTracker:
    """Tracks training quality indicators beyond just loss"""
    
    def __init__(self):
        self.character_metrics = CharacterConsistencyMetrics()
        self.history = defaultdict(list)
        self.warning_thresholds = {
            'loss_plateau_steps': 50,
            'min_consistency_score': 0.3,
            'max_meta_commentary': 0.1,
            'loss_increase_threshold': 0.5
        }
    
    def add_training_step(self, step: int, loss: float, sample_evaluations: List[Dict[str, float]] = None):
        """Add a training step with loss and optional sample evaluations"""
        self.history['step'].append(step)
        self.history['loss'].append(loss)
        
        if sample_evaluations:
            avg_consistency = sum(eval['overall_consistency'] for eval in sample_evaluations) / len(sample_evaluations)
            avg_meta = sum(eval['meta_commentary'] for eval in sample_evaluations) / len(sample_evaluations)
            
            self.history['consistency'].append(avg_consistency)
            self.history['meta_commentary'].append(avg_meta)
    
    def get_training_health(self) -> Dict[str, Any]:
        """Get overall training health indicators"""
        if len(self.history['loss']) < 10:
            return {'status': 'insufficient_data', 'warnings': []}
        
        warnings = []
        recommendations = []
        
        # Check for loss plateau
        recent_losses = self.history['loss'][-20:]
        if len(recent_losses) >= 10:
            loss_variance = torch.var(torch.tensor(recent_losses)).item()
            if loss_variance < 0.001:  # Very low variance = plateau
                warnings.append("Loss has plateaued - consider adjusting learning rate")
                recommendations.append("reduce_lr")
        
        # Check for loss explosion
        if len(self.history['loss']) >= 2:
            last_loss = self.history['loss'][-1]
            prev_loss = self.history['loss'][-2]
            if last_loss > prev_loss * 2:
                warnings.append("Loss is increasing rapidly - possible instability")
                recommendations.append("reduce_lr_immediately")
        
        # Check character consistency
        if self.history['consistency']:
            recent_consistency = self.history['consistency'][-10:]
            avg_consistency = sum(recent_consistency) / len(recent_consistency)
            if avg_consistency < self.warning_thresholds['min_consistency_score']:
                warnings.append("Character consistency is low - check dataset quality")
                recommendations.append("review_dataset")
        
        # Check meta-commentary
        if self.history['meta_commentary']:
            recent_meta = self.history['meta_commentary'][-10:]
            avg_meta = sum(recent_meta) / len(recent_meta)
            if avg_meta > self.warning_thresholds['max_meta_commentary']:
                warnings.append("Model generating meta-commentary - possible training issues")
                recommendations.append("check_dataset_format")
        
        # Overall status
        if not warnings:
            status = 'healthy'
        elif len(warnings) == 1:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'warnings': warnings,
            'recommendations': recommendations,
            'current_loss': self.history['loss'][-1] if self.history['loss'] else 0,
            'avg_consistency': sum(self.history['consistency'][-10:]) / min(10, len(self.history['consistency'])) if self.history['consistency'] else 0,
            'steps_trained': len(self.history['loss'])
        } 