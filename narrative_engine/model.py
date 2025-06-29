# narrative_engine/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType
from typing import Dict, Any, List, Optional, Tuple
import json
from pathlib import Path
import logging

from .config import NarrativeLLMConfig

logger = logging.getLogger(__name__)


class EmotionalMomentumTracker:
    """Tracks emotional state persistence with surprise-weighted decay"""
    
    def __init__(self, config: NarrativeLLMConfig):
        self.config = config
        self.emotional_state = {}  # token -> {strength, decay_rate, turns_remaining}
        self.surprise_detector = SurpriseDetector()
    
    def update_state(self, control_tokens: List[str], token_metadata: Dict[str, Dict], 
                    user_input: str, previous_context: str) -> Dict[str, float]:
        """Update emotional momentum with new tokens and surprise weighting"""
        
        # Calculate surprise factor for this turn
        surprise_score = self.surprise_detector.calculate_surprise(
            user_input, previous_context
        )
        
        # Update existing emotional states (decay)
        for token in list(self.emotional_state.keys()):
            state = self.emotional_state[token]
            state['strength'] *= (1 - state['decay_rate'])
            state['turns_remaining'] -= 1
            
            if state['strength'] < 0.01 or state['turns_remaining'] <= 0:
                del self.emotional_state[token]
        
        # Add new emotional states
        for token in control_tokens:
            if token in token_metadata:
                metadata = token_metadata[token]
                
                # Calculate surprise-weighted persistence
                base_decay = metadata.get('base_decay_rate', 0.3)
                surprise_mult = metadata.get('surprise_multiplier', 1.0)
                
                # Higher surprise = lower decay rate = longer persistence
                adjusted_decay = base_decay / (1 + surprise_score * surprise_mult)
                
                self.emotional_state[token] = {
                    'strength': 1.0,
                    'decay_rate': adjusted_decay,
                    'turns_remaining': self.config.decay_steps,
                    'surprise_factor': surprise_score
                }
        
        # Return current emotional strengths for context injection
        return {token: state['strength'] for token, state in self.emotional_state.items()}
    
    def get_recirculation_context(self) -> List[str]:
        """Get tokens to inject into next turn's context"""
        # Sort by emotional strength and return top tokens
        active_tokens = [(token, state['strength']) for token, state in self.emotional_state.items()]
        active_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 5 most emotionally significant tokens
        return [token for token, strength in active_tokens[:5] if strength > 0.1]


class SurpriseDetector:
    """Detects unexpected patterns in user input"""
    
    def __init__(self):
        self.previous_patterns = []
        self.pattern_memory_size = 10
    
    def calculate_surprise(self, user_input: str, previous_context: str) -> float:
        """Calculate surprise score (0-1) based on input unexpectedness"""
        
        # Simple heuristics for surprise detection
        surprise_indicators = [
            # Sentiment shifts
            self._detect_sentiment_shift(user_input, previous_context),
            # Unexpected compliments/praise
            self._detect_unexpected_praise(user_input),
            # Topic changes
            self._detect_topic_shift(user_input, previous_context),
            # Emotional intensity changes
            self._detect_intensity_change(user_input, previous_context)
        ]
        
        # Combine indicators (weighted average)
        surprise_score = sum(surprise_indicators) / len(surprise_indicators)
        
        # Update pattern memory
        self.previous_patterns.append(user_input.lower())
        if len(self.previous_patterns) > self.pattern_memory_size:
            self.previous_patterns.pop(0)
        
        return min(surprise_score, 1.0)
    
    def _detect_sentiment_shift(self, current: str, previous: str) -> float:
        """Detect sudden sentiment changes"""
        # Simplified sentiment analysis
        positive_words = ['love', 'amazing', 'wonderful', 'beautiful', 'perfect', 'incredible']
        negative_words = ['hate', 'awful', 'terrible', 'horrible', 'sad', 'angry']
        
        def get_sentiment(text):
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            return pos_count - neg_count
        
        current_sentiment = get_sentiment(current)
        previous_sentiment = get_sentiment(previous)
        
        # High surprise if sentiment flips dramatically
        if abs(current_sentiment - previous_sentiment) >= 2:
            return 0.8
        elif abs(current_sentiment - previous_sentiment) == 1:
            return 0.4
        return 0.0
    
    def _detect_unexpected_praise(self, user_input: str) -> float:
        """Detect compliments that might be surprising"""
        praise_words = ['beautiful', 'smart', 'amazing', 'incredible', 'perfect', 'wonderful']
        input_lower = user_input.lower()
        
        praise_count = sum(1 for word in praise_words if word in input_lower)
        return min(praise_count * 0.3, 0.9)
    
    def _detect_topic_shift(self, current: str, previous: str) -> float:
        """Detect sudden topic changes"""
        # Simple keyword overlap approach
        current_words = set(current.lower().split())
        previous_words = set(previous.lower().split())
        
        if len(previous_words) == 0:
            return 0.0
        
        overlap = len(current_words & previous_words) / len(previous_words)
        
        # Low overlap = high topic shift = higher surprise
        return max(0.0, (1.0 - overlap) * 0.6)
    
    def _detect_intensity_change(self, current: str, previous: str) -> float:
        """Detect changes in emotional intensity"""
        intensity_markers = ['!', '?', 'very', 'so', 'really', 'extremely', 'totally']
        
        def get_intensity(text):
            return sum(1 for marker in intensity_markers if marker in text.lower())
        
        current_intensity = get_intensity(current)
        previous_intensity = get_intensity(previous)
        
        intensity_change = abs(current_intensity - previous_intensity)
        return min(intensity_change * 0.2, 0.5)


class NarrativeLLM(nn.Module):
    """
    Narrative-LLM implementation with SmolLM2 backbone and C.L.A.R.A. Loop features.
    
    Features:
    - Dual-head architecture (generation + control)
    - Emotional momentum tracking with surprise weighting
    - Recirculation mechanism for turn-to-turn emotional state
    - Living interface integration
    - Session-aware conversational context
    """
    
    def __init__(self, config: 'NarrativeLLMConfig', control_tokens_path: Optional[str] = None):
        super().__init__()
        self.config = config
        
        # Load base SmolLM2 model
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.hidden_size = self.base_model.config.hidden_size
        
        # Load control tokens vocabulary
        self.control_tokens = self._load_control_tokens(control_tokens_path)
        self.control_token_to_id = {token['token']: i for i, token in enumerate(self.control_tokens)}
        self.control_id_to_token = {i: token['token'] for i, token in enumerate(self.control_tokens)}
        
        # Control head - specialized for emotional/cognitive tokens
        self.control_head = nn.Sequential(
            nn.Linear(self.hidden_size, config.control_head_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.control_head_dim, len(self.control_tokens)),
            nn.Sigmoid()  # Multi-label classification (multiple emotions possible)
        )
        
        # Recirculation layers - inject emotional context
        self.recirculation_embedding = nn.Embedding(
            len(self.control_tokens), 
            self.hidden_size // config.recirculation_layers
        )
        
        # Emotional momentum tracker
        self.momentum_tracker = EmotionalMomentumTracker(config)
        
        # Cache for token metadata
        self.token_metadata = {token['token']: token for token in self.control_tokens}
        
        logger.info(f"Initialized C.L.A.R.A. Loop with {len(self.control_tokens)} control tokens")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        session_id: Optional[torch.Tensor] = None,
        external_memory_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        control_labels: Optional[torch.Tensor] = None,
        recirculation_tokens: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual heads and recirculation.
        
        Args:
            input_ids: Token IDs for input text
            attention_mask: Attention mask
            session_id: Session identifier for stateful context
            external_memory_states: External memory for cross-attention
            labels: Target tokens for generation head
            control_labels: Target control tokens (multi-hot encoded)
            recirculation_tokens: Control tokens from previous turn
            
        Returns:
            Dictionary with text_logits, action_logits, and losses
        """
        
        # Inject recirculation context if provided
        if recirculation_tokens:
            input_ids = self._inject_recirculation_context(input_ids, recirculation_tokens)
        
        # Base model forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get final hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Generation head (standard language modeling)
        generation_logits = outputs.logits
        
        # Control head (emotional/cognitive state)
        # Use the last token's hidden state for control prediction
        last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        control_logits = self.control_head(last_hidden)  # [batch_size, num_control_tokens]
        
        # Calculate losses
        losses = {}
        
        if labels is not None:
            # Generation loss (standard next-token prediction)
            generation_loss = F.cross_entropy(
                generation_logits.view(-1, generation_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            losses['generation_loss'] = generation_loss
        
        if control_labels is not None:
            # Control loss (multi-label binary classification)
            control_loss = F.binary_cross_entropy(
                control_logits,
                control_labels.float()
            )
            losses['control_loss'] = control_loss
        
        # Combined loss
        if losses:
            total_loss = losses.get('generation_loss', 0) + losses.get('control_loss', 0)
            losses['total_loss'] = total_loss
        
        return {
            'text_logits': generation_logits,  # Standard language modeling output
            'action_logits': control_logits,   # Control token output (C.L.A.R.A. Loop)
            'generation_logits': generation_logits,  # Legacy alias
            'control_logits': control_logits,        # Legacy alias
            'hidden_states': hidden_states,
            'losses': losses
        }
    
    def generate_with_control(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        user_input: str = "",
        previous_context: str = "",
        recirculation_tokens: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response with C.L.A.R.A. loop control token emission.
        
        Returns:
            Dictionary with generated_text, control_tokens, and emotional_state
        """
        
        # Inject recirculation context
        if recirculation_tokens:
            input_ids = self._inject_recirculation_context(input_ids, recirculation_tokens)
        
        # Generate response
        with torch.no_grad():
            # Forward pass to get control tokens
            outputs = self.forward(input_ids, attention_mask)
            control_probs = outputs['control_logits'].squeeze(0)  # [num_control_tokens]
            
            # Extract active control tokens (above threshold)
            active_control_tokens = []
            for i, prob in enumerate(control_probs):
                if prob > 0.5:  # Threshold for active tokens
                    token = self.control_id_to_token[i]
                    active_control_tokens.append(token)
            
            # Generate text using base model
            generated_outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                generated_outputs[0][input_ids.shape[-1]:],
                skip_special_tokens=True
            )
        
        # Update emotional momentum
        emotional_state = self.momentum_tracker.update_state(
            active_control_tokens, 
            self.token_metadata,
            user_input,
            previous_context
        )
        
        # Get recirculation tokens for next turn
        next_recirculation = self.momentum_tracker.get_recirculation_context()
        
        return {
            'generated_text': generated_text,
            'control_tokens': active_control_tokens,
            'emotional_state': emotional_state,
            'next_recirculation': next_recirculation,
            'surprise_score': self.momentum_tracker.surprise_detector.calculate_surprise(
                user_input, previous_context
            )
        }
    
    def _inject_recirculation_context(self, input_ids: torch.Tensor, recirculation_tokens: List[str]) -> torch.Tensor:
        """Inject emotional context from previous turn"""
        
        # Convert token strings to IDs
        token_ids = []
        for token in recirculation_tokens:
            if token in self.control_token_to_id:
                # Get embedding for control token
                control_id = self.control_token_to_id[token]
                # For now, we'll add a special prefix to indicate recirculation
                # In a full implementation, this would be handled by the tokenizer
                pass
        
        # For this spike, we'll inject as text tokens
        # In production, we'd have special recirculation embeddings
        return input_ids  # TODO: Implement proper recirculation injection
    
    def _load_control_tokens(self, tokens_path: Optional[str]) -> List[Dict[str, Any]]:
        """Load control tokens from JSON file"""
        if tokens_path is None:
            tokens_path = "content/worlds/Default World/tokens.json"
        
        tokens_file = Path(tokens_path)
        if not tokens_file.exists():
            logger.warning(f"Control tokens file not found: {tokens_path}")
            return []
        
        try:
            with open(tokens_file, 'r') as f:
                tokens = json.load(f)
            
            logger.info(f"Loaded {len(tokens)} control tokens from {tokens_path}")
            return tokens
            
        except Exception as e:
            logger.error(f"Failed to load control tokens: {e}")
            return []
    
    def get_control_token_metadata(self, token: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific control token"""
        return self.token_metadata.get(token)
    
    def get_emotional_state_summary(self) -> Dict[str, Any]:
        """Get current emotional momentum state for debugging/visualization"""
        return {
            'active_emotions': self.momentum_tracker.emotional_state,
            'recirculation_context': self.momentum_tracker.get_recirculation_context(),
            'surprise_patterns': self.momentum_tracker.surprise_detector.previous_patterns[-3:]
        }


def create_narrative_model(
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    control_tokens_path: Optional[str] = None,
    **config_kwargs
) -> NarrativeLLM:
    """Factory function to create Narrative-LLM model"""
    
    config = NarrativeLLMConfig(
        base_model_name=base_model_name,
        **config_kwargs
    )
    
    model = NarrativeLLM(config, control_tokens_path)
    
    return model


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = create_narrative_model()
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 20))
    outputs = model.forward(input_ids)
    
    print(f"Generation logits shape: {outputs['generation_logits'].shape}")
    print(f"Control logits shape: {outputs['control_logits'].shape}")
    print(f"Number of control tokens: {len(model.control_tokens)}") 