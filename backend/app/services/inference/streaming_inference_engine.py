"""
Streaming Inference Engine for Real-Time Multimodal Generation

This module provides high-performance streaming inference for the QuadHeadNarrativeLM model:
- Real-time token-by-token text generation
- Synchronized frame-by-frame speech synthesis  
- Dynamic control signal processing
- Incremental memory updates
- Optimized KV caching and state management
- Character-consistent voice conditioning
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import asynccontextmanager

from backend.app.narrative_engine.quad_head_model import QuadHeadNarrativeLM, create_quad_head_model
from backend.app.narrative_engine.config import NarrativeLLMConfig
from backend.app.services.dataset.multimodal_dataset_pipeline import CharacterVoiceRegistry

logger = logging.getLogger(__name__)


@dataclass
class GenerationState:
    """Tracks the current state of streaming generation"""
    # Text generation state
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    past_key_values: Optional[Tuple] = None
    position_ids: Optional[torch.Tensor] = None
    
    # Speech generation state
    speech_frames: List[torch.Tensor] = field(default_factory=list)
    speech_attention_mask: torch.Tensor = None
    
    # Control and memory state
    control_history: List[str] = field(default_factory=list)
    memory_updates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Character conditioning
    character_id: str = ""
    character_embedding_idx: int = 0
    
    # Generation metadata
    generation_start_time: float = 0.0
    total_tokens_generated: int = 0
    total_speech_frames: int = 0
    
    def reset(self):
        """Reset state for new conversation"""
        self.past_key_values = None
        self.position_ids = None
        self.speech_frames.clear()
        self.control_history.clear()
        self.memory_updates.clear()
        self.total_tokens_generated = 0
        self.total_speech_frames = 0


@dataclass 
class StreamingStep:
    """Single step in streaming generation"""
    step_id: int
    timestamp: float
    
    # Text output
    text_token: Optional[str] = None
    text_logits: Optional[torch.Tensor] = None
    
    # Speech output  
    speech_frame: Optional[torch.Tensor] = None
    speech_logits: Optional[torch.Tensor] = None
    
    # Control output
    control_signal: Optional[str] = None
    control_logits: Optional[torch.Tensor] = None
    
    # Memory output
    memory_update: Optional[Dict[str, Any]] = None
    memory_logits: Optional[torch.Tensor] = None
    
    # Metadata
    is_finished: bool = False
    latency_ms: float = 0.0
    confidence_scores: Dict[str, float] = field(default_factory=dict)


class KVCacheManager:
    """Manages key-value cache for efficient streaming inference"""
    
    def __init__(self, max_cache_length: int = 2048):
        self.max_cache_length = max_cache_length
        self.cache_data: Optional[Tuple] = None
        self.cache_length = 0
        
    def update_cache(self, new_kv: Tuple, num_new_tokens: int) -> Tuple:
        """Update cache with new key-value pairs"""
        if self.cache_data is None:
            self.cache_data = new_kv
            self.cache_length = num_new_tokens
            return new_kv
            
        # Truncate cache if it exceeds max length
        if self.cache_length + num_new_tokens > self.max_cache_length:
            truncate_amount = (self.cache_length + num_new_tokens) - self.max_cache_length
            self.cache_data = self._truncate_cache(self.cache_data, truncate_amount)
            self.cache_length -= truncate_amount
            
        # Concatenate new cache
        updated_cache = self._concatenate_cache(self.cache_data, new_kv)
        self.cache_data = updated_cache
        self.cache_length += num_new_tokens
        
        return updated_cache
    
    def _truncate_cache(self, cache: Tuple, truncate_amount: int) -> Tuple:
        """Truncate cache from the beginning"""
        truncated = []
        for layer_cache in cache:
            # Each layer has (key, value) tensors
            key, value = layer_cache
            # Truncate from sequence dimension (usually dim 2)
            truncated_key = key[:, :, truncate_amount:, :]
            truncated_value = value[:, :, truncate_amount:, :]
            truncated.append((truncated_key, truncated_value))
        return tuple(truncated)
    
    def _concatenate_cache(self, cache1: Tuple, cache2: Tuple) -> Tuple:
        """Concatenate two caches"""
        concatenated = []
        for layer1, layer2 in zip(cache1, cache2):
            key1, value1 = layer1
            key2, value2 = layer2
            # Concatenate along sequence dimension
            concat_key = torch.cat([key1, key2], dim=2)
            concat_value = torch.cat([value1, value2], dim=2)
            concatenated.append((concat_key, concat_value))
        return tuple(concatenated)
    
    def clear(self):
        """Clear the cache"""
        self.cache_data = None
        self.cache_length = 0


class CharacterStateManager:
    """Manages character-specific state and voice conditioning"""
    
    def __init__(self, character_registry: CharacterVoiceRegistry):
        self.character_registry = character_registry
        self.active_states: Dict[str, GenerationState] = {}
        
    def get_or_create_state(self, character_id: str, tokenizer=None) -> GenerationState:
        """Get existing state or create new one for character"""
        if character_id not in self.active_states:
            # Get character embedding index
            char_idx = self.character_registry.get_character_idx(character_id)
            if char_idx is None:
                char_idx = self.character_registry.register_character(character_id, {})
            
            # Create initial state
            state = GenerationState(
                input_ids=torch.tensor([]),
                attention_mask=torch.tensor([]),
                character_id=character_id,
                character_embedding_idx=char_idx,
                generation_start_time=time.time()
            )
            self.active_states[character_id] = state
            
        return self.active_states[character_id]
    
    def update_character_state(self, character_id: str, **updates):
        """Update character state with new information"""
        if character_id in self.active_states:
            state = self.active_states[character_id]
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
    
    def reset_character_state(self, character_id: str):
        """Reset character state for new conversation"""
        if character_id in self.active_states:
            self.active_states[character_id].reset()


class StreamingInferenceEngine:
    """High-performance streaming inference engine for quad-head model"""
    
    def __init__(self,
                 model: Optional[QuadHeadNarrativeLM] = None,
                 config: Optional[NarrativeLLMConfig] = None,
                 tokenizer=None,
                 character_registry: Optional[CharacterVoiceRegistry] = None,
                 device: str = "cpu"):
        
        self.model = model
        self.config = config or NarrativeLLMConfig()
        self.tokenizer = tokenizer
        self.device = device
        
        # State management
        self.character_registry = character_registry or CharacterVoiceRegistry()
        self.character_manager = CharacterStateManager(self.character_registry)
        self.kv_cache_manager = KVCacheManager(max_cache_length=2048)
        
        # Performance tracking
        self.generation_stats = {
            "total_tokens": 0,
            "total_speech_frames": 0,
            "total_latency_ms": 0.0,
            "num_requests": 0,
            "avg_tokens_per_second": 0.0
        }
        
        # Load model if not provided
        if self.model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the quad-head model"""
        try:
            self.model = create_quad_head_model(self.config)
            if torch.cuda.is_available() and self.device != "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Quad-head model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def stream_generate(self,
                            prompt: str,
                            character_id: str,
                            max_new_tokens: int = 100,
                            temperature: float = 0.7,
                            top_p: float = 0.9,
                            do_sample: bool = True,
                            speech_temperature: float = 0.8,
                            **kwargs) -> AsyncGenerator[StreamingStep, None]:
        """
        Stream generate multimodal output token by token and frame by frame
        """
        
        start_time = time.time()
        generation_state = self.character_manager.get_or_create_state(character_id, self.tokenizer)
        
        # Prepare initial input
        if self.tokenizer:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
        else:
            # Simple tokenization for testing
            tokens = prompt.split()
            input_ids = torch.tensor([[hash(token) % 50000 for token in tokens]]).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
        
        # Update generation state
        generation_state.input_ids = input_ids
        generation_state.attention_mask = attention_mask
        generation_state.generation_start_time = start_time
        
        # Character conditioning
        character_embeddings = None
        if hasattr(self.model, 'character_embeddings'):
            char_idx = torch.tensor([generation_state.character_embedding_idx]).to(self.device)
            character_embeddings = char_idx
        
        step_id = 0
        tokens_generated = 0
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                step_start = time.time()
                
                # Prepare model inputs
                model_inputs = {
                    "input_ids": generation_state.input_ids,
                    "attention_mask": generation_state.attention_mask,
                }
                
                # Only add past_key_values if it exists
                if generation_state.past_key_values is not None:
                    model_inputs["past_key_values"] = generation_state.past_key_values
                
                # Only add character embeddings if model supports it
                if character_embeddings is not None:
                    model_inputs["character_ids"] = character_embeddings
                
                # Forward pass through quad-head model
                try:
                    outputs = self.model(**model_inputs)
                    
                    # Extract outputs from different heads
                    text_logits = getattr(outputs, "text_logits", None) or getattr(outputs, "logits", None)
                    speech_logits = getattr(outputs, "speech_logits", None)
                    control_logits = getattr(outputs, "control_logits", None) 
                    memory_logits = getattr(outputs, "memory_logits", None)
                    
                    # Update KV cache
                    if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                        generation_state.past_key_values = self.kv_cache_manager.update_cache(
                            outputs.past_key_values, 1
                        )
                
                except Exception as e:
                    logger.error(f"Model forward pass failed: {e}")
                    step_latency = (time.time() - step_start) * 1000
                    
                    # Update stats even on failure
                    total_time = time.time() - start_time
                    self._update_stats(tokens_generated, total_time)
                    
                    yield StreamingStep(
                        step_id=step_id,
                        timestamp=time.time(),
                        is_finished=True,
                        latency_ms=step_latency
                    )
                    break
                
                # Sample next text token
                next_token_id, text_token, text_confidence = self._sample_text_token(
                    text_logits, temperature, top_p, do_sample
                )
                
                # Generate speech frame if speech head is available
                speech_frame, speech_confidence = None, 0.0
                if speech_logits is not None:
                    speech_frame, speech_confidence = self._sample_speech_frame(
                        speech_logits, speech_temperature
                    )
                
                # Process control signals
                control_signal, control_confidence = self._process_control_signal(control_logits)
                
                # Process memory updates
                memory_update, memory_confidence = self._process_memory_update(memory_logits)
                
                # Update generation state
                if next_token_id is not None:
                    new_token_tensor = torch.tensor([[next_token_id]]).to(self.device)
                    generation_state.input_ids = torch.cat([generation_state.input_ids, new_token_tensor], dim=1)
                    generation_state.attention_mask = torch.cat([
                        generation_state.attention_mask, 
                        torch.ones((1, 1)).to(self.device)
                    ], dim=1)
                    tokens_generated += 1
                    generation_state.total_tokens_generated += 1
                
                if speech_frame is not None:
                    generation_state.speech_frames.append(speech_frame)
                    generation_state.total_speech_frames += 1
                
                if control_signal:
                    generation_state.control_history.append(control_signal)
                
                if memory_update:
                    generation_state.memory_updates.append(memory_update)
                
                # Calculate step latency
                step_latency = (time.time() - step_start) * 1000
                
                # Create streaming step
                step = StreamingStep(
                    step_id=step_id,
                    timestamp=time.time(),
                    text_token=text_token,
                    text_logits=text_logits,
                    speech_frame=speech_frame,
                    speech_logits=speech_logits,
                    control_signal=control_signal,
                    control_logits=control_logits,
                    memory_update=memory_update,
                    memory_logits=memory_logits,
                    latency_ms=step_latency,
                    confidence_scores={
                        "text": text_confidence,
                        "speech": speech_confidence,
                        "control": control_confidence,
                        "memory": memory_confidence
                    }
                )
                
                # Check for stopping conditions
                if self._should_stop_generation(text_token, control_signal, tokens_generated, max_new_tokens):
                    step.is_finished = True
                
                yield step
                step_id += 1
                
                if step.is_finished:
                    break
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
        
        # Update performance stats
        total_time = time.time() - start_time
        self._update_stats(tokens_generated, total_time)
        
    def _sample_text_token(self, 
                          logits: torch.Tensor, 
                          temperature: float, 
                          top_p: float, 
                          do_sample: bool) -> Tuple[Optional[int], Optional[str], float]:
        """Sample next text token from logits"""
        if logits is None:
            return None, None, 0.0
            
        # Get last token logits
        next_token_logits = logits[0, -1, :]
        
        if do_sample:
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            confidence = probs[next_token_id].item()
        else:
            # Greedy sampling
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            probs = F.softmax(next_token_logits, dim=-1)
            confidence = probs[next_token_id].item()
        
        next_token_id = next_token_id.item()
        
        # Convert to text if tokenizer available
        if self.tokenizer:
            try:
                text_token = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            except:
                text_token = f"<token_{next_token_id}>"
        else:
            text_token = f"<token_{next_token_id}>"
        
        return next_token_id, text_token, confidence
    
    def _sample_speech_frame(self, speech_logits: torch.Tensor, temperature: float) -> Tuple[Optional[torch.Tensor], float]:
        """Sample speech frame from speech head logits"""
        if speech_logits is None:
            return None, 0.0
        
        # Apply temperature and sample
        if temperature != 1.0:
            speech_logits = speech_logits / temperature
        
        # For quantized speech, sample from categorical distribution
        # Get last time step: (batch, seq, mel_bins, quantization_levels) -> (batch, mel_bins, quantization_levels)
        last_speech_logits = speech_logits[0, -1, :, :]  # (mel_bins, quantization_levels)
        
        if temperature != 1.0:
            last_speech_logits = last_speech_logits / temperature
        
        probs = F.softmax(last_speech_logits, dim=-1)
        speech_indices = torch.multinomial(probs, num_samples=1)  # (mel_bins, 1)
        
        # Convert back to mel-spectrogram frame
        speech_frame = speech_indices.squeeze(-1).float() / (2**4 - 1)  # Normalize from quantized values
        speech_frame = speech_frame.unsqueeze(0)  # (1, mel_bins)
        
        # Calculate average confidence
        confidence = probs.max(dim=-1)[0].mean().item()
        
        return speech_frame, confidence
    
    def _process_control_signal(self, control_logits: torch.Tensor) -> Tuple[Optional[str], float]:
        """Process control head output into control signals"""
        if control_logits is None:
            return None, 0.0
        
        # Sample control token
        probs = F.softmax(control_logits[0, -1, :], dim=-1)
        control_token_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[control_token_id].item()
        
        # Map to control signal (simplified mapping)
        control_signals = {
            0: "neutral",
            1: "happy", 
            2: "sad",
            3: "angry",
            4: "surprised",
            5: "thinking"
        }
        
        control_signal = control_signals.get(control_token_id % 6, "neutral")
        
        return control_signal, confidence
    
    def _process_memory_update(self, memory_logits: torch.Tensor) -> Tuple[Optional[Dict[str, Any]], float]:
        """Process memory head output into memory updates"""
        if memory_logits is None:
            return None, 0.0
        
        # Simplified memory processing - in practice this would be more sophisticated
        memory_embedding = memory_logits[0, -1, :].detach().cpu().numpy()
        confidence = torch.sigmoid(memory_logits).mean().item()
        
        memory_update = {
            "embedding": memory_embedding.tolist(),
            "timestamp": time.time(),
            "importance": confidence,
            "type": "conversation_memory"
        }
        
        return memory_update, confidence
    
    def _should_stop_generation(self, 
                               text_token: Optional[str], 
                               control_signal: Optional[str], 
                               tokens_generated: int, 
                               max_new_tokens: int) -> bool:
        """Determine if generation should stop"""
        
        # Stop if max tokens reached
        if tokens_generated >= max_new_tokens:
            return True
        
        # Stop on end-of-sequence token
        if text_token and (text_token in ["</s>", "<eos>", "<|endoftext|>"]):
            return True
        
        # Stop on specific control signals
        if control_signal in ["end_conversation", "stop"]:
            return True
        
        return False
    
    def _update_stats(self, tokens_generated: int, total_time: float):
        """Update performance statistics"""
        self.generation_stats["total_tokens"] += tokens_generated
        self.generation_stats["total_latency_ms"] += total_time * 1000
        self.generation_stats["num_requests"] += 1
        
        if total_time > 0:
            self.generation_stats["avg_tokens_per_second"] = (
                self.generation_stats["total_tokens"] / 
                (self.generation_stats["total_latency_ms"] / 1000)
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.generation_stats.copy()
    
    def reset_character_conversation(self, character_id: str):
        """Reset conversation state for a character"""
        self.character_manager.reset_character_state(character_id)
        self.kv_cache_manager.clear()
    
    @asynccontextmanager
    async def character_conversation(self, character_id: str):
        """Context manager for character conversations"""
        try:
            state = self.character_manager.get_or_create_state(character_id)
            yield state
        finally:
            # Optional cleanup - could implement conversation archiving here
            pass


def create_streaming_engine(config: Optional[NarrativeLLMConfig] = None, **kwargs) -> StreamingInferenceEngine:
    """Factory function to create streaming inference engine"""
    return StreamingInferenceEngine(config=config, **kwargs)


# Example usage and testing
async def demo_streaming_generation():
    """Demonstrate streaming inference capabilities"""
    print("🚀 Initializing Streaming Inference Engine...")
    
    # Create engine
    config = NarrativeLLMConfig()
    config.enable_speech_head = True
    engine = create_streaming_engine(config=config)
    
    print("✅ Engine ready! Starting streaming generation...")
    
    # Stream generate with character
    character_id = "alice"
    prompt = "Hello! How are you feeling today?"
    
    async for step in engine.stream_generate(
        prompt=prompt,
        character_id=character_id,
        max_new_tokens=50,
        temperature=0.7
    ):
        print(f"Step {step.step_id}: {step.text_token} | "
              f"Control: {step.control_signal} | "
              f"Latency: {step.latency_ms:.1f}ms")
        
        if step.is_finished:
            print("🎉 Generation complete!")
            break
    
    # Show performance stats
    stats = engine.get_performance_stats()
    print(f"📊 Performance: {stats['avg_tokens_per_second']:.1f} tokens/sec")


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_streaming_generation()) 