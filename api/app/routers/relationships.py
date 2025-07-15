"""
Relationships Router - Director's View Relationship Visualization

Provides endpoints for retrieving and managing character relationships
in the Director's View visualization.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/relationships", tags=["relationships"])

# Mock data for demo purposes
mock_relationships = {
    "nodes": [
        {
            "id": "clara_001",
            "name": "Clara",
            "personality": {
                "openness": 0.8,
                "conscientiousness": 0.6,
                "extraversion": 0.5,
                "agreeableness": 0.8,
                "neuroticism": 0.4
            },
            "emotional_state": {
                "curious": 0.8,
                "hopeful": 0.6
            },
            "position": {"x": 200, "y": 150},
            "size": 60
        },
        {
            "id": "elder_001",
            "name": "Elder",
            "personality": {
                "openness": 0.5,
                "conscientiousness": 0.9,
                "extraversion": 0.3,
                "agreeableness": 0.7,
                "neuroticism": 0.2
            },
            "emotional_state": {
                "wise": 0.9,
                "contemplative": 0.7
            },
            "position": {"x": 500, "y": 300},
            "size": 55
        },
        {
            "id": "merchant_001",
            "name": "Merchant",
            "personality": {
                "openness": 0.7,
                "conscientiousness": 0.8,
                "extraversion": 0.8,
                "agreeableness": 0.6,
                "neuroticism": 0.3
            },
            "emotional_state": {
                "cheerful": 0.8,
                "ambitious": 0.5
            },
            "position": {"x": 350, "y": 450},
            "size": 50
        }
    ],
    "edges": [
        {
            "source": "clara_001",
            "target": "elder_001",
            "affinity": 0.7,
            "status": "Mentor",
            "emotional_history": ["grateful", "trusting", "curious", "respectful"],
            "memory_significance": 0.9,
            "interaction_count": 25,
            "last_interaction": "2024-01-20T10:30:00Z"
        },
        {
            "source": "clara_001",
            "target": "merchant_001",
            "affinity": 0.4,
            "status": "Acquaintance",
            "emotional_history": ["neutral", "curious", "cautious"],
            "memory_significance": 0.5,
            "interaction_count": 10,
            "last_interaction": "2024-01-19T15:45:00Z"
        },
        {
            "source": "elder_001",
            "target": "merchant_001",
            "affinity": -0.2,
            "status": "Skeptical",
            "emotional_history": ["disapproving", "cautious", "observant"],
            "memory_significance": 0.6,
            "interaction_count": 8,
            "last_interaction": "2024-01-18T09:00:00Z"
        }
    ]
}

# Pydantic Models
class RelationshipNode(BaseModel):
    id: str
    name: str
    personality: Dict[str, float]
    emotional_state: Dict[str, float]
    position: Dict[str, float]
    size: float

class RelationshipEdge(BaseModel):
    source: str
    target: str
    affinity: float
    status: str
    emotional_history: List[str]
    memory_significance: float
    interaction_count: int
    last_interaction: str

class RelationshipGraphResponse(BaseModel):
    nodes: List[RelationshipNode]
    edges: List[RelationshipEdge]
    social_metrics: Dict[str, Any]

class RelationshipHistoryEvent(BaseModel):
    timestamp: str
    speaker_id: str
    target_id: str
    interaction_type: str
    affinity_change: float
    emotional_impact: List[str]
    memory_significance: float
    narrative_context: Optional[str] = None

class RelationshipUpdateRequest(BaseModel):
    source_id: str
    target_id: str
    affinity_delta: float
    emotion: Optional[str] = None
    interaction_type: str = "conversation"

# Endpoints
@router.get("/graph", response_model=RelationshipGraphResponse)
async def get_relationship_graph():
    """Get the complete relationship graph for visualization"""
    
    # Calculate social metrics
    total_relationships = len(mock_relationships["edges"]) * 2  # Bidirectional
    average_affinity = sum(edge["affinity"] for edge in mock_relationships["edges"]) / len(mock_relationships["edges"])
    strong_bonds = sum(1 for edge in mock_relationships["edges"] if edge["affinity"] > 0.7)
    conflicts = sum(1 for edge in mock_relationships["edges"] if edge["affinity"] < -0.3)
    
    social_metrics = {
        "total_relationships": total_relationships,
        "average_affinity": round(average_affinity, 2),
        "strong_bonds": strong_bonds,
        "conflicts": conflicts,
        "relationship_density": round(total_relationships / len(mock_relationships["nodes"]), 2),
        "social_clusters": [
            {
                "id": "cluster_1",
                "members": ["clara_001", "elder_001"],
                "cohesion": 0.8
            }
        ],
        "recent_changes": [
            {"pair": ["clara_001", "elder_001"], "change": 0.1, "timestamp": "2024-01-20T10:30:00Z"}
        ]
    }
    
    return RelationshipGraphResponse(
        nodes=mock_relationships["nodes"],
        edges=mock_relationships["edges"],
        social_metrics=social_metrics
    )

@router.get("/history/{character_id}")
async def get_relationship_history(
    character_id: str,
    target_id: Optional[str] = None,
    limit: int = 50
):
    """Get relationship history events for a character"""
    
    # Mock history events
    history_events = [
        {
            "timestamp": "2024-01-20T10:30:00Z",
            "speaker_id": character_id,
            "target_id": target_id or "elder_001",
            "interaction_type": "conversation",
            "affinity_change": 0.1,
            "emotional_impact": ["happy", "grateful"],
            "memory_significance": 0.8,
            "narrative_context": f"{character_id} helped {target_id} with a difficult task"
        },
        {
            "timestamp": "2024-01-20T09:15:00Z",
            "speaker_id": target_id or "elder_001",
            "target_id": character_id,
            "interaction_type": "advice",
            "affinity_change": 0.05,
            "emotional_impact": ["wise", "caring"],
            "memory_significance": 0.7,
            "narrative_context": f"{target_id} confided in {character_id} about their worries"
        }
    ]
    
    return {"events": history_events[:limit]}

@router.post("/update")
async def update_relationship(request: RelationshipUpdateRequest):
    """Update a relationship affinity or status"""
    
    # Find and update the relationship
    for edge in mock_relationships["edges"]:
        if edge["source"] == request.source_id and edge["target"] == request.target_id:
            edge["affinity"] = min(1.0, max(-1.0, edge["affinity"] + request.affinity_delta))
            edge["last_interaction"] = datetime.utcnow().isoformat()
            edge["interaction_count"] += 1
            
            if request.emotion:
                edge["emotional_history"].append(request.emotion)
                if len(edge["emotional_history"]) > 10:
                    edge["emotional_history"].pop(0)
            
            # Update status based on new affinity
            if edge["affinity"] > 0.7:
                edge["status"] = "Close Friend"
            elif edge["affinity"] > 0.3:
                edge["status"] = "Friend"
            elif edge["affinity"] > -0.3:
                edge["status"] = "Acquaintance"
            elif edge["affinity"] > -0.7:
                edge["status"] = "Rival"
            else:
                edge["status"] = "Enemy"
            
            return {
                "updated": True,
                "new_affinity": edge["affinity"],
                "new_status": edge["status"]
            }
    
    raise HTTPException(status_code=404, detail="Relationship not found")

@router.get("/metrics")
async def get_relationship_metrics():
    """Get overall social ecosystem metrics"""
    
    nodes = mock_relationships["nodes"]
    edges = mock_relationships["edges"]
    
    # Calculate various metrics
    metrics = {
        "network_metrics": {
            "total_characters": len(nodes),
            "total_relationships": len(edges),
            "average_connections_per_character": round(len(edges) * 2 / len(nodes), 2),
            "network_density": round(len(edges) / (len(nodes) * (len(nodes) - 1) / 2), 2)
        },
        "affinity_distribution": {
            "strong_positive": sum(1 for e in edges if e["affinity"] > 0.7),
            "positive": sum(1 for e in edges if 0.3 < e["affinity"] <= 0.7),
            "neutral": sum(1 for e in edges if -0.3 <= e["affinity"] <= 0.3),
            "negative": sum(1 for e in edges if -0.7 <= e["affinity"] < -0.3),
            "strong_negative": sum(1 for e in edges if e["affinity"] <= -0.7)
        },
        "emotional_climate": {
            "dominant_emotions": ["curious", "hopeful", "wise"],
            "emotional_volatility": 0.3,
            "collective_mood": "contemplative"
        },
        "interaction_patterns": {
            "average_interactions_per_relationship": round(
                sum(e["interaction_count"] for e in edges) / len(edges), 1
            ),
            "most_active_pair": ["clara_001", "elder_001"],
            "least_active_pair": ["elder_001", "merchant_001"]
        }
    }
    
    return metrics

@router.get("/recommendations/{character_id}")
async def get_relationship_recommendations(character_id: str):
    """Get recommendations for improving relationships"""
    
    recommendations = {
        "character_id": character_id,
        "suggestions": [
            {
                "target_id": "merchant_001",
                "current_affinity": 0.4,
                "recommendation": "Share a common interest to strengthen bond",
                "potential_affinity_gain": 0.2
            },
            {
                "target_id": "elder_001",
                "current_affinity": 0.7,
                "recommendation": "Express gratitude for their guidance",
                "potential_affinity_gain": 0.1
            }
        ],
        "relationship_goals": [
            "Build trust with the merchant",
            "Maintain strong bond with elder"
        ]
    }
    
    return recommendations

# WebSocket handler would be in a separate file but related to this
@router.get("/ws-info")
async def get_websocket_info():
    """Get WebSocket connection info for real-time updates"""
    return {
        "endpoint": "/ws/relationships",
        "supported_events": [
            "relationship_update",
            "affinity_change",
            "new_interaction",
            "emotional_shift"
        ]
    } 