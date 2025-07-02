"""
Directors View Relationship Service

Manages relationship data, events, and state for the Director's View visualization
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass, asdict

from narrative_engine.state_manager import StateManager
from narrative_engine.relationship_manager import RelationshipManager, EnhancedRelationship


logger = logging.getLogger(__name__)


@dataclass
class RelationshipNode:
    """Node data for relationship graph visualization"""
    id: str
    name: str
    personality: Dict[str, float]
    emotional_state: Dict[str, float]
    position: Dict[str, float]
    size: float  # Based on total relationship count
    
    
@dataclass
class RelationshipEdge:
    """Edge data for relationship connections"""
    source: str
    target: str
    affinity: float
    status: str
    emotional_history: List[str]
    memory_significance: float
    interaction_count: int
    last_interaction: str


class DirectorsViewRelationshipService:
    """Service for managing relationship visualization in Director's View"""
    
    def __init__(self, state_manager: StateManager, relationship_manager: RelationshipManager):
        self.state_manager = state_manager
        self.relationship_manager = relationship_manager
        self._relationship_cache: Dict[str, EnhancedRelationship] = {}
        self._node_positions: Dict[str, Dict[str, float]] = {}
        
    async def get_relationship_graph_data(self) -> Dict[str, Any]:
        """Get complete relationship graph data for visualization"""
        try:
            # Get all active entities
            entities = await self.state_manager.get_active_entities()
            
            # Build nodes
            nodes = []
            for entity_id, entity_state in entities.items():
                node = RelationshipNode(
                    id=entity_id,
                    name=entity_state.character_name,
                    personality=entity_state.personality_traits,
                    emotional_state=entity_state.emotional_state,
                    position=self._get_or_calculate_position(entity_id),
                    size=self._calculate_node_size(entity_id)
                )
                nodes.append(asdict(node))
            
            # Build edges
            edges = []
            relationships = await self.relationship_manager.get_all_relationships()
            
            for rel_key, relationship in relationships.items():
                edge = RelationshipEdge(
                    source=relationship.source_id,
                    target=relationship.target_id,
                    affinity=relationship.affinity,
                    status=relationship.status,
                    emotional_history=list(relationship.emotional_history)[-5:],  # Last 5
                    memory_significance=relationship.memory_significance,
                    interaction_count=relationship.interaction_count,
                    last_interaction=relationship.last_interaction.isoformat()
                )
                edges.append(asdict(edge))
                
            return {
                "nodes": nodes,
                "edges": edges,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting relationship graph data: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    async def handle_relationship_update(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process relationship update events"""
        try:
            source_id = event_data.get("speaker_id")
            target_id = event_data.get("target_id")
            
            if not source_id or not target_id:
                return {"error": "Missing source or target ID"}
                
            # Get or create relationship
            relationship = await self.relationship_manager.get_or_create_relationship(
                source_id, target_id
            )
            
            # Update based on interaction type
            interaction_type = event_data.get("interaction_type", "conversation")
            affinity_change = event_data.get("affinity_change", 0.0)
            
            relationship.affinity += affinity_change
            relationship.interaction_count += 1
            relationship.last_interaction = datetime.utcnow()
            
            # Update emotional history
            emotions = event_data.get("emotional_impact", [])
            relationship.emotional_history.extend(emotions)
            
            # Update memory significance
            memory_sig = event_data.get("memory_significance", 0.0)
            relationship.memory_significance = max(
                relationship.memory_significance, 
                memory_sig
            )
            
            # Save updated relationship
            await self.relationship_manager.save_relationship(relationship)
            
            # Return formatted event for WebSocket broadcast
            return {
                "event_type": "relationship_update",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "relationship_manager",
                "speaker_id": source_id,
                "target_id": target_id,
                "interaction_type": interaction_type,
                "affinity_change": affinity_change,
                "emotional_impact": emotions,
                "memory_significance": memory_sig,
                "narrative_context": event_data.get("narrative_context", "")
            }
            
        except Exception as e:
            logger.error(f"Error handling relationship update: {e}")
            return {"error": str(e)}
    
    def _get_or_calculate_position(self, entity_id: str) -> Dict[str, float]:
        """Get cached position or calculate new one using force layout"""
        if entity_id not in self._node_positions:
            # Simple circular layout for initial positions
            import math
            num_entities = len(self._node_positions)
            angle = (2 * math.pi * num_entities) / max(8, num_entities + 1)
            radius = 300
            self._node_positions[entity_id] = {
                "x": 400 + radius * math.cos(angle),
                "y": 300 + radius * math.sin(angle)
            }
        return self._node_positions[entity_id]
    
    def _calculate_node_size(self, entity_id: str) -> float:
        """Calculate node size based on relationship count"""
        try:
            # Count relationships where this entity is involved
            count = sum(
                1 for rel in self.relationship_manager.relationships.values()
                if entity_id in (rel.source_id, rel.target_id)
            )
            # Base size 30, +10 per relationship, max 100
            return min(30 + (count * 10), 100)
        except:
            return 50  # Default size
    
    async def get_relationship_timeline(
        self, 
        entity_ids: List[str], 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get timeline of relationship events between entities"""
        try:
            # This would integrate with the event log system
            # For now, return mock data structure
            events = []
            
            # In real implementation, query event store
            # filtered by entity IDs and relationship events
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting relationship timeline: {e}")
            return []
    
    async def get_social_metrics(self) -> Dict[str, Any]:
        """Calculate social ecosystem metrics"""
        try:
            relationships = await self.relationship_manager.get_all_relationships()
            
            # Calculate metrics
            total_relationships = len(relationships)
            positive_relationships = sum(
                1 for r in relationships.values() if r.affinity > 0
            )
            negative_relationships = sum(
                1 for r in relationships.values() if r.affinity < 0
            )
            
            # Average values
            avg_affinity = (
                sum(r.affinity for r in relationships.values()) / total_relationships
                if total_relationships > 0 else 0
            )
            
            avg_interaction_count = (
                sum(r.interaction_count for r in relationships.values()) / total_relationships
                if total_relationships > 0 else 0
            )
            
            # Find most connected entities
            connection_counts = {}
            for rel in relationships.values():
                connection_counts[rel.source_id] = connection_counts.get(rel.source_id, 0) + 1
                connection_counts[rel.target_id] = connection_counts.get(rel.target_id, 0) + 1
            
            most_connected = sorted(
                connection_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            return {
                "total_relationships": total_relationships,
                "positive_relationships": positive_relationships,
                "negative_relationships": negative_relationships,
                "neutral_relationships": total_relationships - positive_relationships - negative_relationships,
                "average_affinity": round(avg_affinity, 2),
                "average_interactions": round(avg_interaction_count, 1),
                "most_connected_entities": [
                    {"entity_id": eid, "connections": count}
                    for eid, count in most_connected
                ],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating social metrics: {e}")
            return {"error": str(e)} 