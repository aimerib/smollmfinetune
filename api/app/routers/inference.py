"""
Inference Router for Character Chat

Provides REST endpoints for character interaction and conversation.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()

router = APIRouter()

# In-memory storage for demo purposes
# In production, this would use a proper database
sessions: Dict[str, Dict[str, Any]] = {}
session_histories: Dict[str, List[Dict[str, Any]]] = {}

class SessionStartRequest(BaseModel):
    character_id: str
    platform: str = "web"
    device_info: Optional[Dict[str, Any]] = None

class InferenceRequest(BaseModel):
    session_id: str
    character_id: str
    prompt: str
    stream: bool = False
    include_emotions: bool = True
    memory_context_window: int = 10

class InferenceResponse(BaseModel):
    generation_text: str
    control_tokens: List[Dict[str, Any]] = []
    memory_vector: List[float] = []
    memory_metadata: Dict[str, Any] = {}
    session_id: str
    latency_ms: int = 100

class SessionState(BaseModel):
    session_id: str
    character_id: str
    message_count: int
    emotional_state: Dict[str, Any]
    memory_stats: Dict[str, Any]

# Character personalities for demo responses
character_personalities = {
    "alice": {
        "name": "Alice",
        "traits": ["curious", "adventurous", "optimistic"],
        "speaking_style": "enthusiastic and inquisitive",
        "emoji": "🧚‍♀️"
    },
    "max": {
        "name": "Max", 
        "traits": ["analytical", "precise", "logical"],
        "speaking_style": "methodical and technical",
        "emoji": "🤖"
    },
    "luna": {
        "name": "Luna",
        "traits": ["mystical", "intuitive", "calm"],
        "speaking_style": "ethereal and thoughtful", 
        "emoji": "🌙"
    }
}

def generate_character_response(character_id: str, user_message: str) -> str:
    """Generate a character response based on personality"""
    character = character_personalities.get(character_id, character_personalities["alice"])
    
    # Simple response generation for demo
    responses = {
        "alice": [
            f"Oh wow, that's fascinating! {character['emoji']} I'm always excited to explore new ideas. What else would you like to discover together?",
            f"That reminds me of an adventure I had once! {character['emoji']} Tell me, what draws you to that topic?",
            f"How exciting! {character['emoji']} I love learning about different perspectives. What's your experience with this?"
        ],
        "max": [
            f"Analyzing your input... {character['emoji']} That's an interesting data point. Let me process the implications systematically.",
            f"From a technical standpoint {character['emoji']}, there are several variables to consider. Shall I elaborate on the methodology?",
            f"Computing response... {character['emoji']} Your query suggests multiple optimization pathways. Which approach interests you most?"
        ],
        "luna": [
            f"I sense deep meaning in your words... {character['emoji']} The universe often speaks through such connections. What feelings does this evoke?",
            f"In the quiet spaces between thoughts {character['emoji']}, wisdom emerges. Your question touches something profound.",
            f"The moonlight reveals many truths {character['emoji']}... I feel this conversation flows like a gentle stream. Where shall it carry us?"
        ]
    }
    
    import random
    return random.choice(responses.get(character_id, responses["alice"]))

@router.post("/sessions", response_model=Dict[str, str])
async def start_session(request: SessionStartRequest):
    """Start a new chat session"""
    session_id = str(uuid.uuid4())
    
    sessions[session_id] = {
        "session_id": session_id,
        "character_id": request.character_id,
        "platform": request.platform,
        "device_info": request.device_info,
        "created_at": datetime.utcnow(),
        "message_count": 0,
        "emotional_state": {
            "current": "neutral",
            "history": []
        },
        "memory_stats": {
            "total_memories": 0,
            "important_memories": 0,
            "last_memory_timestamp": None
        }
    }
    
    session_histories[session_id] = []
    
    logger.info("Started chat session", session_id=session_id, character_id=request.character_id)
    
    return {"session_id": session_id}

@router.post("/inference", response_model=InferenceResponse)
async def generate_response(request: InferenceRequest, background_tasks: BackgroundTasks):
    """Generate character response to user message"""
    
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[request.session_id]
    
    # Generate character response
    response_text = generate_character_response(request.character_id, request.prompt)
    
    # Update session state
    session["message_count"] += 1
    
    # Add to conversation history
    if request.session_id not in session_histories:
        session_histories[request.session_id] = []
    
    session_histories[request.session_id].extend([
        {
            "role": "user",
            "content": request.prompt,
            "timestamp": datetime.utcnow().isoformat()
        },
        {
            "role": "assistant", 
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "emotion": "engaged"
        }
    ])
    
    # Update emotional state
    emotions = ["curious", "engaged", "thoughtful", "excited", "calm"]
    import random
    current_emotion = random.choice(emotions)
    
    session["emotional_state"]["current"] = current_emotion
    session["emotional_state"]["history"].append({
        "emotion": current_emotion,
        "timestamp": datetime.utcnow().isoformat(),
        "confidence": 0.8
    })
    
    logger.info("Generated response", 
                session_id=request.session_id, 
                character_id=request.character_id,
                message_count=session["message_count"])
    
    return InferenceResponse(
        generation_text=response_text,
        control_tokens=[],
        memory_vector=[],
        memory_metadata={
            "importance": 0.7,
            "emotional_valence": 0.6,
            "context_relevance": 0.8,
            "decay_rate": 0.1
        },
        session_id=request.session_id
    )

@router.get("/sessions/{session_id}/state", response_model=SessionState)
async def get_session_state(session_id: str):
    """Get current session state"""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    return SessionState(
        session_id=session_id,
        character_id=session["character_id"],
        message_count=session["message_count"],
        emotional_state=session["emotional_state"],
        memory_stats=session["memory_stats"]
    )

@router.post("/sessions/{session_id}/end")
async def end_session(session_id: str):
    """End a chat session"""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clean up session data
    del sessions[session_id]
    if session_id in session_histories:
        del session_histories[session_id]
    
    logger.info("Ended chat session", session_id=session_id)
    
    return {"status": "session_ended"}

@router.get("/characters/{character_id}/packet")
async def get_character_packet(character_id: str):
    """Get character packet (mock implementation)"""
    
    if character_id not in character_personalities:
        raise HTTPException(status_code=404, detail="Character not found")
    
    character = character_personalities[character_id]
    
    # Return character info as JSON
    packet = {
        "character_id": character_id,
        "name": character["name"],
        "traits": character["traits"],
        "speaking_style": character["speaking_style"],
        "emoji": character["emoji"],
        "version": "1.0.0",
        "created_at": datetime.utcnow().isoformat()
    }
    
    return packet

@router.post("/sessions/{session_id}/proactive")
async def toggle_proactive_mode(session_id: str, request: Dict[str, Any]):
    """Toggle proactive mode for session"""
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    enabled = request.get("enabled", False)
    
    sessions[session_id]["proactive_mode"] = {
        "enabled": enabled,
        "check_interval_minutes": request.get("check_interval_minutes", 30),
        "notification_preferences": request.get("notification_preferences", {})
    }
    
    logger.info("Toggled proactive mode", 
                session_id=session_id, 
                enabled=enabled)
    
    return {"status": "proactive_mode_updated", "enabled": enabled} 