"""
Memory Generation Pipeline

Uses OpenAI's structured output to generate memory annotations for 
training data augmentation. Supports both Method A (tokens) and Method B (vectors).
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime
import numpy as np

from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import tiktoken

from .memory_schema import (
    MemoryAnnotation,
    Turn,
    MemoryTrainingBatch,
)

logger = logging.getLogger(__name__)


class MemoryGenerationRequest(BaseModel):
    """Request structure for memory generation"""
    conversation_turns: List[Turn]
    character_profile: Dict[str, Any]
    max_memories: int = Field(default=3, description="Maximum memories to generate per conversation")
    include_method_a: bool = Field(default=True, description="Generate tokens for Method A")
    include_method_b: bool = Field(default=True, description="Generate vectors for Method B")


class GeneratedMemory(BaseModel):
    """Structured output from OpenAI for a single memory"""
    memory_content: str = Field(description="The memory to be stored (1-3 sentences)")
    turn_index: int = Field(description="Which turn (0-indexed) triggered this memory")
    
    # Core characteristics
    surprise_score: float = Field(ge=0, le=1, description="How surprising (0=expected, 1=shocking)")
    emotional_valence: float = Field(ge=-1, le=1, description="Emotional tone (-1=negative, 1=positive)")
    importance: float = Field(ge=0, le=1, description="Overall importance")
    memory_type: str = Field(description="Type: episodic, semantic, emotional, or procedural")
    
    # Method A token selection
    importance_level: str = Field(description="low, medium, or high")
    valence_category: str = Field(description="positive, negative, or neutral")
    
    # Additional context
    reasoning: str = Field(description="Brief explanation of why this memory formed")


class MemoryBatch(BaseModel):
    """Structured output containing multiple memories"""
    memories: List[GeneratedMemory]
    overall_surprise: float = Field(description="Average surprise across the conversation")
    emotional_arc: str = Field(description="Brief description of emotional journey")


class MemoryGenerator:
    """Generates synthetic memories for training data augmentation"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the memory generator.
        
        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Token mappings for Method A
        self.importance_token_map = {
            "low": "<memory_importance_low>",
            "medium": "<memory_importance_medium>",
            "high": "<memory_importance_high>",
        }
        
        self.valence_token_map = {
            "positive": "<memory_valence_positive>",
            "negative": "<memory_valence_negative>",
            "neutral": "<memory_valence_neutral>",
        }
        
        self.type_token_map = {
            "episodic": "<memory_type_episodic>",
            "semantic": "<memory_type_semantic>",
            "emotional": "<memory_type_emotional>",
            "procedural": "<memory_type_procedural>",
        }
    
    async def generate_memories(
        self, 
        request: MemoryGenerationRequest
    ) -> MemoryTrainingBatch:
        """
        Generate memories for a conversation using OpenAI structured output.
        
        Args:
            request: Generation request with conversation and character info
            
        Returns:
            MemoryTrainingBatch with annotations for both methods
        """
        
        # Build the prompt
        system_prompt = self._build_system_prompt(request.character_profile)
        user_prompt = self._build_user_prompt(request.conversation_turns)
        
        try:
            # Call OpenAI with structured output
            response = await self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=MemoryBatch,
                temperature=0.7,
                max_tokens=1000,
            )
            
            # Extract the structured response
            memory_batch = response.choices[0].message.parsed
            
            # Convert to our internal format
            return self._convert_to_training_batch(
                memory_batch, 
                request.conversation_turns,
                request.character_profile,
                request.include_method_a,
                request.include_method_b,
            )
            
        except Exception as e:
            logger.error(f"Failed to generate memories: {e}")
            raise
    
    def _build_system_prompt(self, character_profile: Dict[str, Any]) -> str:
        """Build the system prompt for memory generation"""
        
        character_name = character_profile.get("name", "Character")
        personality = character_profile.get("personality_traits", {})
        
        return f"""You are an expert in cognitive psychology and memory formation for AI characters.

You are analyzing conversations to identify what memories {character_name} would form.

Character Profile:
- Name: {character_name}
- Personality Traits: {json.dumps(personality, indent=2)}

Memory Formation Guidelines:
1. SURPRISE: Unexpected events create stronger memories. Consider what would surprise this character.
2. EMOTION: Emotionally charged moments are memorable. Both positive and negative.
3. RELEVANCE: Things related to the character's goals, fears, or interests.
4. FIRSTS: First experiences are often memorable.
5. PATTERNS: Repeated behaviors might form procedural memories.

Memory Types:
- Episodic: Specific events ("When the user complimented my laugh")
- Semantic: Facts learned ("The user prefers tea over coffee")
- Emotional: Feelings tied to experiences ("I felt warm when they said that")
- Procedural: How to interact ("They respond well to gentle humor")

Be selective - not every turn creates a memory. Focus on significant moments."""
    
    def _build_user_prompt(self, turns: List[Turn]) -> str:
        """Build the user prompt with conversation context"""
        
        conversation_text = "\n".join([
            f"{turn.sender.upper()}: {turn.text}"
            for turn in turns
        ])
        
        return f"""Analyze this conversation and identify what memories the character would form.

Conversation:
{conversation_text}

Generate 1-3 memories that the character would realistically form from this interaction.
Consider the emotional weight, surprise factor, and long-term significance of each moment."""
    
    def _convert_to_training_batch(
        self,
        memory_batch: MemoryBatch,
        turns: List[Turn],
        character_profile: Dict[str, Any],
        include_method_a: bool,
        include_method_b: bool,
    ) -> MemoryTrainingBatch:
        """Convert OpenAI response to our training format"""
        
        memories = []
        method_a_sequences = []
        method_b_targets = []
        
        # Calculate familiarity score from character profile
        familiarity = character_profile.get("relationship_status", {}).get("familiarity", 0.0)
        
        for gen_memory in memory_batch.memories:
            # Build Method A tokens
            method_a_tokens = ["<memory_form>"]
            
            if include_method_a:
                # Add importance token
                method_a_tokens.append(
                    self.importance_token_map[gen_memory.importance_level]
                )
                
                # Add type token
                method_a_tokens.append(
                    self.type_token_map[gen_memory.memory_type]
                )
                
                # Add valence token
                method_a_tokens.append(
                    self.valence_token_map[gen_memory.valence_category]
                )
            
            # Generate Method B vector (simplified - in production use proper embedding model)
            method_b_vector = None
            method_b_metadata = None
            
            if include_method_b:
                # Create a pseudo-embedding based on memory characteristics
                # In production, use a proper embedding model
                method_b_vector = self._generate_pseudo_embedding(gen_memory)
                
                # Calculate decay rate based on surprise and importance
                base_decay = 0.3
                adjusted_decay = base_decay / (1 + gen_memory.surprise_score * 2.0)
                
                method_b_metadata = {
                    "decay_rate": adjusted_decay,
                    "persistence_factor": gen_memory.importance,
                    "emotional_weight": abs(gen_memory.emotional_valence),
                }
            
            # Get the context window (last 6 turns before memory formation)
            turn_idx = gen_memory.turn_index
            context_start = max(0, turn_idx - 5)
            context_window = turns[context_start:turn_idx + 1]
            
            # Create MemoryAnnotation
            memory = MemoryAnnotation(
                conversation_window=context_window,
                memory_content=gen_memory.memory_content,
                surprise_score=gen_memory.surprise_score,
                emotional_valence=gen_memory.emotional_valence,
                importance=gen_memory.importance,
                memory_type=gen_memory.memory_type,
                character_id=character_profile.get("id", "unknown"),
                familiarity_score=familiarity,
                method_a_tokens=method_a_tokens if include_method_a else [],
                method_b_vector=method_b_vector,
                method_b_metadata=method_b_metadata,
                decay_rate=adjusted_decay if include_method_b else 0.1,
                formation_strength=1.0,
            )
            
            memories.append(memory)
            
            # Add to method-specific sequences
            if include_method_a:
                method_a_sequences.append({
                    "position": f"after_turn_{turn_idx}",
                    "tokens": method_a_tokens,
                })
            
            if include_method_b:
                method_b_targets.append({
                    "vector": method_b_vector,
                    "metadata": method_b_metadata,
                })
        
        # Create the training batch
        return MemoryTrainingBatch(
            conversation_id=f"conv_{datetime.now().timestamp()}",
            memories=memories,
            method_a_sequence=method_a_sequences,
            method_b_targets=method_b_targets,
        )
    
    def _generate_pseudo_embedding(self, memory: GeneratedMemory) -> List[float]:
        """
        Generate a pseudo-embedding for Method B.
        In production, use a real embedding model.
        """
        np.random.seed(hash(memory.memory_content) % 2**32)
        
        # Create base random vector
        base_vector = np.random.randn(768)
        
        # Bias the vector based on memory characteristics
        # This is purely illustrative - real embeddings would be semantic
        
        # Emotional dimension (first 256 dims)
        emotional_bias = memory.emotional_valence * 0.5
        base_vector[:256] += emotional_bias
        
        # Importance dimension (middle 256 dims)
        importance_bias = memory.importance * 0.3
        base_vector[256:512] += importance_bias
        
        # Type dimension (last 256 dims)
        type_biases = {
            "episodic": 0.3,
            "semantic": -0.2,
            "emotional": 0.5,
            "procedural": -0.3,
        }
        type_bias = type_biases.get(memory.memory_type, 0)
        base_vector[512:] += type_bias
        
        # Normalize to unit length
        norm = np.linalg.norm(base_vector)
        if norm > 0:
            base_vector = base_vector / norm
        
        return base_vector.tolist()


async def generate_memories_for_dataset(
    conversations: List[Dict[str, Any]],
    character_profile: Dict[str, Any],
    output_path: Optional[Path] = None,
) -> List[MemoryTrainingBatch]:
    """
    Generate memories for an entire dataset of conversations.
    
    Args:
        conversations: List of conversation dictionaries
        character_profile: Character profile for context
        output_path: Optional path to save the generated memories
        
    Returns:
        List of MemoryTrainingBatch objects
    """
    generator = MemoryGenerator()
    training_batches = []
    
    for i, conv in enumerate(conversations):
        try:
            # Convert conversation to Turn objects
            turns = [
                Turn(
                    sender=msg["role"], 
                    text=msg["content"],
                    timestamp=datetime.now()
                )
                for msg in conv.get("messages", [])
            ]
            
            # Skip very short conversations
            if len(turns) < 3:
                continue
            
            # Create request
            request = MemoryGenerationRequest(
                conversation_turns=turns,
                character_profile=character_profile,
                max_memories=3,
            )
            
            # Generate memories
            batch = await generator.generate_memories(request)
            training_batches.append(batch)
            
            logger.info(f"Generated {len(batch.memories)} memories for conversation {i+1}/{len(conversations)}")
            
        except Exception as e:
            logger.error(f"Failed to process conversation {i}: {e}")
            continue
    
    # Save if requested
    if output_path:
        output_data = {
            "character_id": character_profile.get("id"),
            "generation_timestamp": datetime.now().isoformat(),
            "total_conversations": len(conversations),
            "total_memories": sum(len(b.memories) for b in training_batches),
            "batches": [
                {
                    "conversation_id": batch.conversation_id,
                    "memories": [m.model_dump() for m in batch.memories],
                    "training_format": batch.to_training_format(),
                }
                for batch in training_batches
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Saved memory dataset to {output_path}")
    
    return training_batches


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Example conversation
        turns = [
            Turn(sender="user", text="Hi! I'm Alex. What's your name?"),
            Turn(sender="assistant", text="Hello Alex! I'm Clara. It's nice to meet you!"),
            Turn(sender="user", text="You have such a pretty name. It suits you."),
            Turn(sender="assistant", text="Oh... thank you! That's very kind of you to say."),
        ]
        
        # Example character profile
        character = {
            "id": "clara_001",
            "name": "Clara",
            "personality_traits": {
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.4,
                "agreeableness": 0.9,
                "neuroticism": 0.3,
            },
            "relationship_status": {
                "familiarity": 0.1,  # Just met
            }
        }
        
        # Generate memories
        generator = MemoryGenerator()
        request = MemoryGenerationRequest(
            conversation_turns=turns,
            character_profile=character,
        )
        
        batch = await generator.generate_memories(request)
        
        print(f"Generated {len(batch.memories)} memories:")
        for i, memory in enumerate(batch.memories):
            print(f"\n{i+1}. {memory.memory_content}")
            print(f"   Type: {memory.memory_type}, Importance: {memory.importance:.2f}")
            print(f"   Surprise: {memory.surprise_score:.2f}, Valence: {memory.emotional_valence:.2f}")
            print(f"   Tokens: {memory.method_a_tokens}")
    
    asyncio.run(main()) 