"""
Multi-Character Conversation FastAPI Router

Provides REST and WebSocket endpoints for managing multi-character conversations
with real-time audio streaming and spatial positioning.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from pydantic import BaseModel

from backend.app.services.conversation.multi_character_manager import (
    MultiCharacterConversationManager,
    DialogueTurn,
    Vector3D,
    ConversationState
)
from backend.app.websocket.manager import WebSocketManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/multi-character", tags=["multi-character-conversation"])

# Global WebSocket manager for multi-character conversations
websocket_manager = WebSocketManager()

# Global conversation managers per session
conversation_managers: Dict[str, MultiCharacterConversationManager] = {}


class ConversationRequest(BaseModel):
    """Request model for starting a conversation"""
    session_id: str
    character_ids: List[str]
    initial_context: Optional[Dict[str, Any]] = None


class DialogueTurnRequest(BaseModel):
    """Request model for a dialogue turn"""
    character_id: str
    text: str
    emotion_context: Optional[Dict[str, float]] = None
    interrupts_previous: bool = False
    urgency_level: float = 0.5


class SpatialPositionRequest(BaseModel):
    """Request model for updating character spatial position"""
    character_id: str
    position: Dict[str, float]  # x, y, z coordinates


class ConversationStatusResponse(BaseModel):
    """Response model for conversation status"""
    session_id: str
    active_characters: List[str]
    conversation_context: Dict[str, Any]
    spatial_positions: Dict[str, Dict[str, float]]


@router.post("/conversations", response_model=ConversationStatusResponse)
async def create_conversation(request: ConversationRequest):
    """Create a new multi-character conversation session"""
    try:
        # Create WebSocket manager for this session
        session_websocket = WebSocketManager()
        
        # Create conversation manager
        manager = MultiCharacterConversationManager(session_websocket)
        conversation_managers[request.session_id] = manager
        
        # Initialize conversation state
        if request.initial_context:
            manager.conversation_state.set_context(request.initial_context)
        
        # Add characters to conversation
        for character_id in request.character_ids:
            manager.conversation_state.add_speaker(character_id)
        
        logger.info(f"Created multi-character conversation session: {request.session_id}")
        
        return ConversationStatusResponse(
            session_id=request.session_id,
            active_characters=manager.conversation_state.get_active_speakers(),
            conversation_context=manager.conversation_state.get_context(),
            spatial_positions={}
        )
        
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{session_id}", response_model=ConversationStatusResponse)
async def get_conversation_status(session_id: str):
    """Get the status of a conversation session"""
    if session_id not in conversation_managers:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    
    manager = conversation_managers[session_id]
    
    # Get spatial positions
    spatial_positions = {}
    for character_id in manager.conversation_state.get_active_speakers():
        position_info = manager.spatial_audio.get_position_info(character_id)
        spatial_positions[character_id] = position_info
    
    return ConversationStatusResponse(
        session_id=session_id,
        active_characters=manager.conversation_state.get_active_speakers(),
        conversation_context=manager.conversation_state.get_context(),
        spatial_positions=spatial_positions
    )


@router.post("/conversations/{session_id}/dialogue")
async def add_dialogue_turn(session_id: str, request: DialogueTurnRequest):
    """Add a dialogue turn to the conversation"""
    if session_id not in conversation_managers:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    
    manager = conversation_managers[session_id]
    
    try:
        # Create dialogue turn
        turn = DialogueTurn(
            character_id=request.character_id,
            text=request.text,
            emotion_context=request.emotion_context,
            interrupts_previous=request.interrupts_previous,
            urgency_level=request.urgency_level
        )
        
        # Handle interruption if needed
        if request.interrupts_previous:
            await manager.handle_conversation_interruption(turn)
        
        # Generate dialogue
        await manager.generate_multi_character_dialogue([turn])
        
        return {"status": "success", "message": "Dialogue turn processed"}
        
    except Exception as e:
        logger.error(f"Failed to process dialogue turn: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{session_id}/spatial-position")
async def update_spatial_position(session_id: str, request: SpatialPositionRequest):
    """Update the spatial position of a character"""
    if session_id not in conversation_managers:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    
    manager = conversation_managers[session_id]
    
    try:
        # Update character position
        position = Vector3D(
            x=request.position["x"],
            y=request.position["y"],
            z=request.position["z"]
        )
        
        result = manager.spatial_audio.update_character_position(
            request.character_id, position
        )
        
        return {
            "status": "success",
            "character_id": request.character_id,
            "position": request.position,
            "distance": result["distance"]
        }
        
    except Exception as e:
        logger.error(f"Failed to update spatial position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{session_id}")
async def end_conversation(session_id: str):
    """End a conversation session"""
    if session_id not in conversation_managers:
        raise HTTPException(status_code=404, detail="Conversation session not found")
    
    # Clean up the conversation manager
    del conversation_managers[session_id]
    
    logger.info(f"Ended conversation session: {session_id}")
    return {"status": "success", "message": "Conversation ended"}


@router.websocket("/conversations/{session_id}/stream")
async def conversation_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time multi-character conversation streaming
    
    Handles:
    - Real-time audio streaming
    - Spatial audio updates
    - Conversation state changes
    - Speaker transitions
    """
    await websocket.accept()
    
    try:
        # Get or create conversation manager
        if session_id not in conversation_managers:
            # Create a new conversation manager for this session
            session_websocket = WebSocketConnectionWrapper(websocket)
            manager = MultiCharacterConversationManager(session_websocket)
            conversation_managers[session_id] = manager
        else:
            manager = conversation_managers[session_id]
            # Update websocket connection
            manager.websocket = WebSocketConnectionWrapper(websocket)
        
        logger.info(f"WebSocket connected for conversation session: {session_id}")
        
        # Send initial conversation state
        await websocket.send_json({
            "type": "conversation_state",
            "data": {
                "session_id": session_id,
                "active_characters": manager.conversation_state.get_active_speakers(),
                "context": manager.conversation_state.get_context()
            }
        })
        
        while True:
            # Receive messages from client
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                message_type = data.get("type")
                
                if message_type == "dialogue_turn":
                    # Handle dialogue turn request
                    turn_data = data.get("data", {})
                    turn = DialogueTurn(
                        character_id=turn_data["character_id"],
                        text=turn_data["text"],
                        emotion_context=turn_data.get("emotion_context"),
                        interrupts_previous=turn_data.get("interrupts_previous", False),
                        urgency_level=turn_data.get("urgency_level", 0.5)
                    )
                    
                    # Process dialogue turn
                    if turn.interrupts_previous:
                        await manager.handle_conversation_interruption(turn)
                    
                    await manager.generate_multi_character_dialogue([turn])
                
                elif message_type == "spatial_position":
                    # Handle spatial position update
                    pos_data = data.get("data", {})
                    position = Vector3D(
                        x=pos_data["position"]["x"],
                        y=pos_data["position"]["y"],
                        z=pos_data["position"]["z"]
                    )
                    
                    result = manager.spatial_audio.update_character_position(
                        pos_data["character_id"], position
                    )
                    
                    # Send position update confirmation
                    await websocket.send_json({
                        "type": "spatial_position_updated",
                        "data": result
                    })
                
                elif message_type == "conversation_control":
                    # Handle conversation control (pause, resume, etc.)
                    control_data = data.get("data", {})
                    action = control_data.get("action")
                    
                    if action == "pause":
                        # Pause conversation (implementation depends on requirements)
                        await websocket.send_json({
                            "type": "conversation_paused",
                            "data": {"session_id": session_id}
                        })
                    elif action == "resume":
                        # Resume conversation
                        await websocket.send_json({
                            "type": "conversation_resumed",
                            "data": {"session_id": session_id}
                        })
                
                else:
                    logger.warning(f"Unknown message type: {message_type}")
                
            except json.JSONDecodeError:
                logger.error("Invalid JSON received from WebSocket")
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Invalid JSON format"}
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "data": {"message": str(e)}
        })
    finally:
        # Clean up on disconnect
        if session_id in conversation_managers:
            logger.info(f"Cleaning up conversation session: {session_id}")


class WebSocketConnectionWrapper:
    """Wrapper to make WebSocket compatible with conversation manager interface"""
    
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
    
    async def send_audio_chunk(self, data: Dict[str, Any]):
        """Send audio chunk via WebSocket"""
        await self.websocket.send_json({
            "type": "audio_chunk",
            "data": data
        })


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(conversation_managers),
        "service": "multi-character-conversation"
    } 