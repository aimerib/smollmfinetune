"""
Memory Service for vector storage and retrieval.

Handles memory embeddings, similarity search, and memory management
for the triple-head model's memory outputs.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryVector:
    """Memory vector with metadata"""
    session_id: str
    character_id: str
    embedding: List[float]
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_id: Optional[str] = None
    
    def __post_init__(self):
        if self.memory_id is None:
            import uuid
            self.memory_id = str(uuid.uuid4())


@dataclass 
class MemorySearchResult:
    """Result from memory search"""
    memory: MemoryVector
    similarity_score: float
    relevance_metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryService:
    """
    Memory integration service for the inference engine.
    
    Features:
    - Vector storage and retrieval
    - Similarity search with multiple algorithms
    - Memory quality scoring and pruning
    - Cross-attention formatting
    - Real-time memory formation from model outputs
    """
    
    def __init__(self, storage_backend: str = "memory",
                 vector_dim: int = 768):
        """
        Initialize memory service.
        
        Args:
            storage_backend: Storage backend type (memory, faiss, qdrant)
            vector_dim: Dimension of memory vectors
        """
        self.storage_backend = storage_backend
        self.vector_dim = vector_dim
        
        # In-memory storage for now
        self.memories: Dict[str, MemoryVector] = {}
        self.session_memories: Dict[str, List[str]] = {}  # session_id -> memory_ids
        
        logger.info(f"MemoryService initialized with {storage_backend} backend")
    
    async def store_memory(self, memory: MemoryVector) -> str:
        """
        Store a memory vector.
        
        Args:
            memory: Memory to store
            
        Returns:
            Memory ID
        """
        # Validate vector dimension
        if len(memory.embedding) != self.vector_dim:
            raise ValueError(f"Expected {self.vector_dim} dimensions, got {len(memory.embedding)}")
        
        # Store memory
        self.memories[memory.memory_id] = memory
        
        # Track by session
        session_key = f"{memory.session_id}:{memory.character_id}"
        if session_key not in self.session_memories:
            self.session_memories[session_key] = []
        self.session_memories[session_key].append(memory.memory_id)
        
        logger.debug(f"Stored memory {memory.memory_id} for session {memory.session_id}")
        
        return memory.memory_id
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryVector]:
        """Get a specific memory by ID"""
        return self.memories.get(memory_id)
    
    async def search_memories(self, query_embedding: List[float],
                            session_id: str,
                            character_id: Optional[str] = None,
                            top_k: int = 5) -> List[MemorySearchResult]:
        """
        Search for similar memories using vector similarity.
        
        Args:
            query_embedding: Query vector
            session_id: Session to search within
            character_id: Optional character filter
            top_k: Number of results to return
            
        Returns:
            Top-k similar memories with scores
        """
        # Get relevant memory IDs
        session_key = f"{session_id}:{character_id}" if character_id else session_id
        memory_ids = []
        
        for key in self.session_memories:
            if key.startswith(session_key):
                memory_ids.extend(self.session_memories[key])
        
        if not memory_ids:
            return []
        
        # Calculate similarities
        query_vec = np.array(query_embedding)
        results = []
        
        for memory_id in memory_ids:
            memory = self.memories.get(memory_id)
            if not memory:
                continue
            
            # Cosine similarity
            memory_vec = np.array(memory.embedding)
            similarity = np.dot(query_vec, memory_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(memory_vec)
            )
            
            results.append(MemorySearchResult(
                memory=memory,
                similarity_score=float(similarity)
            ))
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    async def get_relevant_memories(self, session_id: str, character_id: str,
                                  context: str, top_k: int = 5) -> List[MemoryVector]:
        """
        Get relevant memories based on context.
        
        Args:
            session_id: Session ID
            character_id: Character ID
            context: Text context to match
            top_k: Number of memories to return
            
        Returns:
            Relevant memories
        """
        # For now, return recent memories
        # In production, would use embedding of context
        session_key = f"{session_id}:{character_id}"
        memory_ids = self.session_memories.get(session_key, [])
        
        memories = []
        for memory_id in memory_ids[-top_k:]:
            memory = self.memories.get(memory_id)
            if memory:
                memories.append(memory)
        
        return memories
    
    def format_for_attention(self, memories: List[MemoryVector]) -> Dict[str, Any]:
        """
        Format memories for cross-attention injection.
        
        Args:
            memories: Memories to format
            
        Returns:
            Formatted attention context
        """
        if not memories:
            return {
                "memory_embeddings": [],
                "memory_weights": [],
                "memory_mask": []
            }
        
        # Stack embeddings
        embeddings = np.array([m.embedding for m in memories])
        
        # Calculate importance weights
        weights = []
        for memory in memories:
            importance = memory.metadata.get("importance", 0.5)
            recency = memory.metadata.get("recency", 0.5)
            weight = importance * 0.7 + recency * 0.3
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        return {
            "memory_embeddings": embeddings.tolist(),
            "memory_weights": weights.tolist(),
            "memory_mask": [1] * len(memories)  # All memories are valid
        }
    
    async def score_memories(self, session_id: str, character_id: str) -> List[Dict[str, Any]]:
        """
        Score memories for quality assessment.
        
        Args:
            session_id: Session ID
            character_id: Character ID
            
        Returns:
            Scored memories
        """
        session_key = f"{session_id}:{character_id}"
        memory_ids = self.session_memories.get(session_key, [])
        
        scored_memories = []
        
        for memory_id in memory_ids:
            memory = self.memories.get(memory_id)
            if not memory:
                continue
            
            # Calculate quality score
            importance = memory.metadata.get("importance", 0.5)
            coherence = memory.metadata.get("coherence", 0.5)
            access_count = memory.metadata.get("access_count", 0)
            age = time.time() - memory.timestamp
            
            # Scoring formula
            recency_score = 1.0 / (1.0 + age / 3600)  # Decay over hours
            access_score = min(1.0, access_count / 10)  # Cap at 10 accesses
            
            quality_score = (
                importance * 0.4 +
                coherence * 0.3 +
                recency_score * 0.2 +
                access_score * 0.1
            )
            
            scored_memories.append({
                "memory_id": memory_id,
                "content": memory.content,
                "quality_score": quality_score,
                "components": {
                    "importance": importance,
                    "coherence": coherence,
                    "recency": recency_score,
                    "access": access_score
                }
            })
        
        return scored_memories
    
    async def prune_memories(self, session_id: str, character_id: str,
                           quality_threshold: float = 0.3,
                           max_memories: int = 100) -> int:
        """
        Prune low-quality memories.
        
        Args:
            session_id: Session ID
            character_id: Character ID
            quality_threshold: Minimum quality score to keep
            max_memories: Maximum memories to retain
            
        Returns:
            Number of memories pruned
        """
        # Score memories
        scored = await self.score_memories(session_id, character_id)
        
        # Sort by quality
        scored.sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Determine which to keep
        to_keep = []
        for i, memory_info in enumerate(scored):
            if i < max_memories and memory_info["quality_score"] >= quality_threshold:
                to_keep.append(memory_info["memory_id"])
        
        # Prune
        session_key = f"{session_id}:{character_id}"
        original_count = len(self.session_memories.get(session_key, []))
        
        # Remove pruned memories
        pruned_count = 0
        if session_key in self.session_memories:
            for memory_id in list(self.session_memories[session_key]):
                if memory_id not in to_keep:
                    self.memories.pop(memory_id, None)
                    self.session_memories[session_key].remove(memory_id)
                    pruned_count += 1
        
        logger.info(f"Pruned {pruned_count} memories for {session_key}")
        
        return pruned_count
    
    async def process_memory_head_output(self, output: Any,
                                       session_id: str,
                                       character_id: str,
                                       context: str) -> str:
        """
        Process memory head output from triple-head model.
        
        Args:
            output: Triple-head model output
            session_id: Session ID
            character_id: Character ID
            context: Generation context
            
        Returns:
            Memory ID
        """
        # Create memory from output
        memory = MemoryVector(
            session_id=session_id,
            character_id=character_id,
            embedding=output.memory_vector,
            content=output.generation_text,
            metadata={
                **output.memory_metadata,
                "source": "memory_head",
                "context": context
            }
        )
        
        # Store and return ID
        return await self.store_memory(memory)
    
    async def count_memories(self, session_id: str, character_id: str) -> int:
        """Count memories for a session/character"""
        session_key = f"{session_id}:{character_id}"
        return len(self.session_memories.get(session_key, [])) 