"""
Relationship Manager for Narrative Engine

Manages character relationships, affinity tracking, and social dynamics
"""

from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import json
import logging


logger = logging.getLogger(__name__)


@dataclass
class EnhancedRelationship:
    """Enhanced relationship tracking with emotional history"""
    source_id: str
    target_id: str
    affinity: float = 0.0  # -1.0 to 1.0
    status: str = "Acquaintance"  # Friend, Rival, Enemy, Lover, etc.
    emotional_history: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_significance: float = 0.0
    interaction_count: int = 0
    last_interaction: datetime = field(default_factory=datetime.utcnow)
    relationship_metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "affinity": self.affinity,
            "status": self.status,
            "emotional_history": list(self.emotional_history),
            "memory_significance": self.memory_significance,
            "interaction_count": self.interaction_count,
            "last_interaction": self.last_interaction.isoformat(),
            "relationship_metadata": self.relationship_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EnhancedRelationship':
        """Create from dictionary"""
        rel = cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            affinity=data.get("affinity", 0.0),
            status=data.get("status", "Acquaintance"),
            memory_significance=data.get("memory_significance", 0.0),
            interaction_count=data.get("interaction_count", 0),
            relationship_metadata=data.get("relationship_metadata", {})
        )
        
        # Handle emotional history
        history = data.get("emotional_history", [])
        rel.emotional_history.extend(history)
        
        # Handle last interaction
        if "last_interaction" in data:
            rel.last_interaction = datetime.fromisoformat(data["last_interaction"])
            
        return rel


class RelationshipManager:
    """Manages all character relationships in the narrative"""
    
    def __init__(self):
        self.relationships: Dict[str, EnhancedRelationship] = {}
        self._relationship_cache: Dict[str, Set[str]] = {}  # Character -> Related characters
        
    def _get_relationship_key(self, source_id: str, target_id: str) -> str:
        """Generate consistent key for relationship lookup"""
        # Always use alphabetical order for bidirectional relationships
        return f"{min(source_id, target_id)}_{max(source_id, target_id)}"
    
    async def get_or_create_relationship(
        self, 
        source_id: str, 
        target_id: str
    ) -> EnhancedRelationship:
        """Get existing relationship or create new one"""
        key = self._get_relationship_key(source_id, target_id)
        
        if key not in self.relationships:
            self.relationships[key] = EnhancedRelationship(
                source_id=source_id,
                target_id=target_id
            )
            
            # Update cache
            if source_id not in self._relationship_cache:
                self._relationship_cache[source_id] = set()
            if target_id not in self._relationship_cache:
                self._relationship_cache[target_id] = set()
                
            self._relationship_cache[source_id].add(target_id)
            self._relationship_cache[target_id].add(source_id)
            
        return self.relationships[key]
    
    def get_relationship(
        self, 
        source_id: str, 
        target_id: str
    ) -> Optional[EnhancedRelationship]:
        """Get relationship between two characters"""
        key = self._get_relationship_key(source_id, target_id)
        return self.relationships.get(key)
    
    async def get_all_relationships(self) -> Dict[str, EnhancedRelationship]:
        """Get all relationships"""
        return self.relationships.copy()
    
    async def get_character_relationships(
        self, 
        character_id: str
    ) -> List[EnhancedRelationship]:
        """Get all relationships for a specific character"""
        relationships = []
        
        for rel in self.relationships.values():
            if character_id in (rel.source_id, rel.target_id):
                relationships.append(rel)
                
        return relationships
    
    async def update_relationship(
        self,
        source_id: str,
        target_id: str,
        affinity_change: float = 0.0,
        emotion: Optional[str] = None,
        memory_significance: Optional[float] = None
    ) -> EnhancedRelationship:
        """Update relationship based on interaction"""
        relationship = await self.get_or_create_relationship(source_id, target_id)
        
        # Update affinity
        old_affinity = relationship.affinity
        relationship.affinity = max(-1.0, min(1.0, relationship.affinity + affinity_change))
        
        # Update status based on affinity thresholds
        if relationship.affinity > 0.7:
            relationship.status = "Close Friend"
        elif relationship.affinity > 0.3:
            relationship.status = "Friend"
        elif relationship.affinity > -0.3:
            relationship.status = "Acquaintance"
        elif relationship.affinity > -0.7:
            relationship.status = "Rival"
        else:
            relationship.status = "Enemy"
            
        # Add emotion to history
        if emotion:
            relationship.emotional_history.append(emotion)
            
        # Update memory significance
        if memory_significance is not None:
            relationship.memory_significance = max(
                relationship.memory_significance,
                memory_significance
            )
            
        # Update interaction tracking
        relationship.interaction_count += 1
        relationship.last_interaction = datetime.utcnow()
        
        logger.info(
            f"Relationship updated: {source_id} -> {target_id}, "
            f"affinity: {old_affinity:.2f} -> {relationship.affinity:.2f}, "
            f"status: {relationship.status}"
        )
        
        return relationship
    
    async def save_relationship(self, relationship: EnhancedRelationship):
        """Save updated relationship"""
        key = self._get_relationship_key(
            relationship.source_id, 
            relationship.target_id
        )
        self.relationships[key] = relationship
        
    def calculate_social_influence(
        self, 
        character_id: str
    ) -> float:
        """Calculate character's social influence based on relationships"""
        related_chars = self._relationship_cache.get(character_id, set())
        
        if not related_chars:
            return 0.0
            
        # Sum positive affinities
        total_influence = 0.0
        
        for other_id in related_chars:
            rel = self.relationships.get(
                self._get_relationship_key(character_id, other_id)
            )
            if rel and rel.affinity > 0:
                total_influence += rel.affinity
                
        return total_influence
    
    def find_mutual_connections(
        self, 
        char1: str, 
        char2: str
    ) -> Set[str]:
        """Find characters connected to both inputs"""
        connections1 = self._relationship_cache.get(char1, set())
        connections2 = self._relationship_cache.get(char2, set())
        
        return connections1.intersection(connections2)
    
    def get_relationship_network_stats(self) -> Dict[str, Any]:
        """Get statistics about the relationship network"""
        if not self.relationships:
            return {
                "total_relationships": 0,
                "average_affinity": 0.0,
                "most_connected": None,
                "network_density": 0.0
            }
            
        # Calculate stats
        total_affinity = sum(rel.affinity for rel in self.relationships.values())
        avg_affinity = total_affinity / len(self.relationships)
        
        # Find most connected
        connection_counts = {}
        for char_id, connections in self._relationship_cache.items():
            connection_counts[char_id] = len(connections)
            
        most_connected = max(
            connection_counts.items(), 
            key=lambda x: x[1]
        ) if connection_counts else (None, 0)
        
        # Calculate network density
        unique_characters = len(self._relationship_cache)
        max_possible_relationships = (unique_characters * (unique_characters - 1)) / 2
        density = (
            len(self.relationships) / max_possible_relationships 
            if max_possible_relationships > 0 else 0
        )
        
        return {
            "total_relationships": len(self.relationships),
            "average_affinity": avg_affinity,
            "most_connected": most_connected[0],
            "most_connected_count": most_connected[1],
            "network_density": density,
            "unique_characters": unique_characters
        }
    
    def save_to_file(self, filepath: str):
        """Save relationships to JSON file"""
        data = {
            "relationships": {
                key: rel.to_dict() 
                for key, rel in self.relationships.items()
            },
            "metadata": {
                "saved_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_from_file(self, filepath: str):
        """Load relationships from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.relationships.clear()
            self._relationship_cache.clear()
            
            for key, rel_data in data.get("relationships", {}).items():
                rel = EnhancedRelationship.from_dict(rel_data)
                self.relationships[key] = rel
                
                # Rebuild cache
                if rel.source_id not in self._relationship_cache:
                    self._relationship_cache[rel.source_id] = set()
                if rel.target_id not in self._relationship_cache:
                    self._relationship_cache[rel.target_id] = set()
                    
                self._relationship_cache[rel.source_id].add(rel.target_id)
                self._relationship_cache[rel.target_id].add(rel.source_id)
                
            logger.info(f"Loaded {len(self.relationships)} relationships from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading relationships: {e}")
            raise 