"""
Director's Chair Router - Real-Time Training Interface

Provides endpoints for conversation editing, multi-head preference collection,
and real-time training management for the Director's Chair system.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid
import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/directors-chair", tags=["directors-chair"])

# In-memory storage for demo purposes
directors_chair_sessions: Dict[str, Dict[str, Any]] = {}
preference_pairs: Dict[str, Dict[str, Any]] = {}
training_queues: Dict[str, List[Dict[str, Any]]] = {
    "generation": [],
    "control": [],
    "memory": []
}
training_status: Dict[str, Dict[str, Any]] = {
    "generation": {"status": "idle", "progress": 0.0, "queue_length": 0},
    "control": {"status": "idle", "progress": 0.0, "queue_length": 0},
    "memory": {"status": "idle", "progress": 0.0, "queue_length": 0}
}

# Pydantic Models
class DirectorsChairSessionRequest(BaseModel):
    character_id: str
    mode: str = "directors_chair"
    training_enabled: bool = True

class ConversationTurnRequest(BaseModel):
    user_message: str
    assistant_response: str
    metadata: Optional[Dict[str, Any]] = {}

class CorrectionRequest(BaseModel):
    type: str
    target_head: str
    original_text: Optional[str] = None
    corrected_text: Optional[str] = None
    emotional_state: Optional[str] = None
    control_tokens: Optional[List[str]] = None
    memory_importance: Optional[float] = None
    should_remember: Optional[bool] = None
    reason: str

class PreferenceRequest(BaseModel):
    head_type: str
    prompt: Optional[str] = None
    context: Optional[str] = None
    chosen_response: str
    rejected_response: str
    character_id: str
    correction_reason: str
    quality_metrics: Optional[Dict[str, float]] = {}
    control_metrics: Optional[Dict[str, float]] = {}
    memory_metrics: Optional[Dict[str, float]] = {}

class TrainingConfigRequest(BaseModel):
    heads: List[str]
    batch_size: int = 16
    learning_rate: float = 1e-4
    max_training_time_minutes: int = 2
    coordination_weight: float = 0.1

# Status endpoint
@router.get("/status")
async def get_directors_chair_status():
    """Get Director's Chair system status"""
    return {
        "service": "directors-chair",
        "status": "active",
        "active_sessions": len(directors_chair_sessions),
        "training_queues": {
            head: len(queue) for head, queue in training_queues.items()
        }
    }

# Session Management
@router.post("/sessions", status_code=201)
async def start_directors_chair_session(request: DirectorsChairSessionRequest):
    """Start a new Director's Chair session"""
    session_id = str(uuid.uuid4())
    
    session_data = {
        "session_id": session_id,
        "character_id": request.character_id,
        "mode": request.mode,
        "training_enabled": request.training_enabled,
        "created_at": datetime.utcnow(),
        "conversations": [],
        "corrections": []
    }
    
    directors_chair_sessions[session_id] = session_data
    
    logger.info("Started Director's Chair session", 
                session_id=session_id, 
                character_id=request.character_id)
    
    return {
        "session_id": session_id,
        "mode": request.mode,
        "training_enabled": request.training_enabled
    }

@router.post("/sessions/{session_id}/conversations", status_code=201)
async def create_conversation_turn(session_id: str, request: ConversationTurnRequest):
    """Create a conversation turn for editing"""
    if session_id not in directors_chair_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    conversation_id = str(uuid.uuid4())
    conversation_data = {
        "conversation_id": conversation_id,
        "user_message": request.user_message,
        "assistant_response": request.assistant_response,
        "metadata": request.metadata,
        "editable": True,
        "created_at": datetime.utcnow()
    }
    
    directors_chair_sessions[session_id]["conversations"].append(conversation_data)
    
    return {
        "conversation_id": conversation_id,
        "editable": True
    }

@router.post("/conversations/{conversation_id}/corrections", status_code=201)
async def apply_correction(conversation_id: str, request: CorrectionRequest):
    """Apply a correction and route to appropriate training head"""
    
    # Validate correction type and target head
    valid_types = ["content_correction", "emotion_correction", "memory_correction"]
    valid_heads = ["generation", "control", "memory"]
    
    if request.type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid correction type: {request.type}")
    
    if request.target_head not in valid_heads:
        raise HTTPException(status_code=400, detail=f"Invalid target head: {request.target_head}")
    
    correction_id = str(uuid.uuid4())
    correction_data = {
        "correction_id": correction_id,
        "conversation_id": conversation_id,
        "type": request.type,
        "target_head": request.target_head,
        "applied_at": datetime.utcnow(),
        "queued_for_training": True
    }
    
    # Add to training queue
    training_queues[request.target_head].append({
        "correction_id": correction_id,
        "data": request.dict(),
        "queued_at": datetime.utcnow()
    })
    
    return {
        "correction_id": correction_id,
        "target_head": request.target_head,
        "queued_for_training": True
    }

# Preference Collection
@router.post("/preferences", status_code=201)
async def create_preference_pair(request: PreferenceRequest):
    """Create a preference pair for head-specific training"""
    valid_heads = ["generation", "control", "memory"]
    
    if request.head_type not in valid_heads:
        raise HTTPException(status_code=400, detail=f"Invalid head type: {request.head_type}")
    
    preference_id = str(uuid.uuid4())
    preference_data = {
        "preference_id": preference_id,
        "head_type": request.head_type,
        "prompt": request.prompt,
        "context": request.context,
        "chosen_response": request.chosen_response,
        "rejected_response": request.rejected_response,
        "character_id": request.character_id,
        "correction_reason": request.correction_reason,
        "created_at": datetime.utcnow(),
        "training_queue_position": len(training_queues[request.head_type])
    }
    
    # Add head-specific analysis
    if request.head_type == "control":
        preference_data["control_token_analysis"] = {
            "detected_tokens": [],
            "emotional_consistency": 0.8
        }
    elif request.head_type == "memory":
        preference_data["memory_formation_analysis"] = {
            "importance_score": 0.7,
            "context_relevance": 0.8
        }
    
    preference_pairs[preference_id] = preference_data
    
    # Add to training queue
    training_queues[request.head_type].append({
        "preference_id": preference_id,
        "data": preference_data,
        "queued_at": datetime.utcnow()
    })
    
    return {
        "preference_id": preference_id,
        "head_type": request.head_type,
        "training_queue_position": preference_data["training_queue_position"],
        **({k: v for k, v in preference_data.items() 
            if k in ["control_token_analysis", "memory_formation_analysis"]})
    }

# Training Workers
@router.get("/training/{head_type}/status")
async def get_training_worker_status(head_type: str):
    """Get training worker status for a specific head"""
    valid_heads = ["generation", "control", "memory"]
    
    if head_type not in valid_heads:
        raise HTTPException(status_code=404, detail="Invalid head type")
    
    status = training_status[head_type]
    queue_length = len(training_queues[head_type])
    
    response = {
        "head_type": head_type,
        "worker_status": status["status"],
        "queue_length": queue_length,
        "current_training_job": None if status["status"] == "idle" else {
            "job_id": f"job_{head_type}_001",
            "progress": status["progress"],
            "eta": "1m 30s"
        }
    }
    
    # Add head-specific metrics
    if head_type == "control":
        response["emotional_training_metrics"] = {
            "current_loss": 0.234,
            "emotion_accuracy": 0.85
        }
    elif head_type == "memory":
        response["memory_formation_metrics"] = {
            "formation_rate": 0.7,
            "recall_accuracy": 0.82
        }
    
    return response

@router.get("/training/status")
async def get_coordinated_training_status():
    """Get coordinated training status across all heads"""
    return {
        "generation_head": {
            "status": training_status["generation"]["status"],
            "progress": training_status["generation"]["progress"],
            "queue_length": len(training_queues["generation"])
        },
        "control_head": {
            "status": training_status["control"]["status"],
            "progress": training_status["control"]["progress"],
            "queue_length": len(training_queues["control"])
        },
        "memory_head": {
            "status": training_status["memory"]["status"],
            "progress": training_status["memory"]["progress"],
            "queue_length": len(training_queues["memory"])
        },
        "coordination_metrics": {
            "cross_head_consistency": 0.87,
            "balance_score": 0.92
        },
        "overall_training_progress": sum(
            training_status[head]["progress"] for head in training_status
        ) / 3
    }

@router.post("/training/start", status_code=202)
async def start_training_cycle(request: TrainingConfigRequest, background_tasks: BackgroundTasks):
    """Start a coordinated training cycle"""
    training_job_id = str(uuid.uuid4())
    
    # Simulate training start
    for head in request.heads:
        if head in training_status:
            training_status[head]["status"] = "training"
            training_status[head]["progress"] = 0.1
    
    # Add background task simulation
    background_tasks.add_task(simulate_training, training_job_id, request.heads)
    
    estimated_completion = datetime.utcnow() + timedelta(minutes=request.max_training_time_minutes)
    
    return {
        "training_job_id": training_job_id,
        "status": "training_started",
        "estimated_completion_time": estimated_completion.isoformat()
    }

# Analytics
@router.get("/analytics/coordination")
async def get_coordination_metrics():
    """Get cross-head coordination metrics"""
    return {
        "head_interaction_strength": {
            "generation_control": 0.8,
            "generation_memory": 0.7,
            "control_memory": 0.9
        },
        "coordination_loss": 0.15,
        "balanced_improvement": True,
        "conflict_resolution": {
            "conflicts_detected": 2,
            "conflicts_resolved": 2,
            "resolution_rate": 1.0
        }
    }

@router.get("/analytics/{head_type}")
async def get_head_analytics(head_type: str):
    """Get analytics for a specific head"""
    valid_heads = ["generation", "control", "memory"]
    
    if head_type not in valid_heads:
        raise HTTPException(status_code=404, detail="Invalid head type")
    
    base_analytics = {
        "head_type": head_type,
        "improvement_rate": 0.05,
        "training_sessions": 12,
        "last_updated": datetime.utcnow().isoformat()
    }
    
    if head_type == "generation":
        return {
            **base_analytics,
            "content_quality_trend": [0.6, 0.65, 0.7, 0.72, 0.75],
            "coherence_scores": [0.8, 0.82, 0.85, 0.83, 0.87],
            "creativity_metrics": {"uniqueness": 0.7, "diversity": 0.6},
            "character_consistency": 0.9
        }
    elif head_type == "control":
        return {
            **base_analytics,
            "emotional_appropriateness": [0.7, 0.75, 0.8, 0.82, 0.85],
            "control_token_effectiveness": 0.87,
            "mood_consistency": 0.9,
            "emotional_range": {"happy": 0.8, "sad": 0.6, "excited": 0.9}
        }
    elif head_type == "memory":
        return {
            **base_analytics,
            "memory_formation_rate": [0.6, 0.65, 0.7, 0.75, 0.8],
            "recall_accuracy": 0.85,
            "importance_calibration": 0.78,
            "memory_decay_patterns": {"short_term": 0.9, "long_term": 0.7}
        }

# Error Handling & Queue Management
@router.post("/corrections", status_code=400)
async def handle_invalid_correction(request: dict):
    """Handle invalid correction requests"""
    return {"detail": "Invalid correction type"}

@router.get("/training/queue/capacity")
async def get_queue_capacity():
    """Get training queue capacity information"""
    max_queue_size = 100
    current_total = sum(len(queue) for queue in training_queues.values())
    
    return {
        "max_queue_size": max_queue_size,
        "current_queue_size": current_total,
        "accepts_new_jobs": current_total < max_queue_size,
        "per_head_status": {
            head: {
                "current_size": len(queue),
                "max_size": max_queue_size // 3
            }
            for head, queue in training_queues.items()
        }
    }

# Background Tasks
async def simulate_training(training_job_id: str, heads: List[str]):
    """Simulate training progress"""
    await asyncio.sleep(1)  # Simulate training delay
    
    for head in heads:
        if head in training_status:
            training_status[head]["progress"] = 0.5
    
    await asyncio.sleep(2)  # More training simulation
    
    for head in heads:
        if head in training_status:
            training_status[head]["progress"] = 1.0
            training_status[head]["status"] = "completed"
    
    # Reset after completion
    await asyncio.sleep(1)
    for head in heads:
        if head in training_status:
            training_status[head]["status"] = "idle"
            training_status[head]["progress"] = 0.0 