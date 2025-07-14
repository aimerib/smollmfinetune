"""
Quad-Head Model Streaming WebSocket Router

Provides real-time multimodal generation via WebSocket connections.
Integrates with QuadHeadNarrativeLM for native speech synthesis.
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel, Field
import torch

from backend.app.narrative_engine.quad_head_model import (
    create_quad_head_model, 
    QuadHeadNarrativeLM
)
from backend.app.narrative_engine.config import NarrativeLLMConfig
from backend.app.services.character.voice_profile import CharacterVoiceManager
from backend.app.redis_client import RedisCache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/quad-head", tags=["quad-head-streaming"])

# Global model instance (will be dependency injected in production)
_quad_head_model: Optional[QuadHeadNarrativeLM] = None
_voice_manager: Optional[CharacterVoiceManager] = None

class QuadHeadStreamRequest(BaseModel):
    """Request model for quad-head streaming"""
    type: str = Field(..., description="Request type: 'generate_multimodal'")
    text: str = Field(..., description="Input text for generation")
    character_id: str = Field(..., description="Character identifier")
    max_length: int = Field(default=100, description="Maximum generation length")
    temperature: float = Field(default=0.8, description="Generation temperature")
    speech_temperature: float = Field(default=0.7, description="Speech generation temperature")
    emotion_context: Optional[Dict[str, Any]] = Field(default=None, description="Emotional context")
    force_speech: bool = Field(default=True, description="Force speech generation")
    streaming: bool = Field(default=True, description="Enable streaming output")

class QuadHeadStreamResponse(BaseModel):
    """Response model for quad-head streaming"""
    type: str
    text_token: Optional[str] = None
    speech_frame: Optional[List[float]] = None
    control_signal: Optional[str] = None
    memory_update: Optional[Dict[str, Any]] = None
    timestamp: float
    finished: bool = False
    error: Optional[str] = None

# WebSocket connection manager
class QuadHeadConnectionManager:
    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}
        self.character_sessions: Dict[str, str] = {}  # character_id -> connection_id
    
    async def connect(self, websocket: WebSocket, character_id: str) -> str:
        """Connect websocket and return connection ID"""
        await websocket.accept()
        connection_id = f"quad_head_{character_id}_{id(websocket)}"
        self.connections[connection_id] = websocket
        self.character_sessions[character_id] = connection_id
        logger.info(f"Quad-head WebSocket connected: {connection_id}")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Disconnect websocket"""
        if connection_id in self.connections:
            del self.connections[connection_id]
            # Remove character session mapping
            for char_id, conn_id in list(self.character_sessions.items()):
                if conn_id == connection_id:
                    del self.character_sessions[char_id]
                    break
            logger.info(f"Quad-head WebSocket disconnected: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, message: dict):
        """Send message to specific connection"""
        if connection_id in self.connections:
            try:
                await self.connections[connection_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                self.disconnect(connection_id)

connection_manager = QuadHeadConnectionManager()

def get_quad_head_model() -> QuadHeadNarrativeLM:
    """Get or create quad-head model instance"""
    global _quad_head_model
    if _quad_head_model is None:
        config = NarrativeLLMConfig()
        config.enable_speech_head = True
        config.speech_mel_bins = 80
        config.speech_quantization_bits = 4
        _quad_head_model = create_quad_head_model(config)
        logger.info("Quad-head model initialized for streaming")
    return _quad_head_model

def get_voice_manager() -> CharacterVoiceManager:
    """Get voice manager instance"""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = CharacterVoiceManager()
    return _voice_manager

def mel_to_audio_chunk(mel_frames: torch.Tensor, sample_rate: int = 22050) -> bytes:
    """Convert mel-spectrogram frames to audio bytes"""
    try:
        # Simple mel-to-audio conversion (placeholder)
        # In production, this would use a vocoder like HiFiGAN
        
        # Convert mel to linear spectrogram (rough approximation)
        mel_np = mel_frames.detach().cpu().numpy()
        
        # Generate audio waveform from mel (simplified)
        # This is a placeholder - real implementation would use proper vocoder
        audio_frames = np.random.normal(0, 0.1, size=(len(mel_np) * 256,))
        audio_frames = np.clip(audio_frames, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        audio_int16 = (audio_frames * 32767).astype(np.int16)
        return audio_int16.tobytes()
        
    except Exception as e:
        logger.error(f"Error converting mel to audio: {e}")
        return b""

async def stream_multimodal_generation(
    model: QuadHeadNarrativeLM,
    request: QuadHeadStreamRequest,
    connection_id: str
):
    """Stream multimodal generation from quad-head model"""
    try:
        # Prepare input
        input_text = f"<character:{request.character_id}> {request.text}"
        
        # Tokenize input (placeholder - would use actual tokenizer)
        input_ids = torch.randint(0, 1000, (1, 20))  # Mock tokenization
        
        # Character voice conditioning
        character_embedding = torch.randn(1, model.config.hidden_size)  # Mock character embedding
        
        # Generate step by step
        current_ids = input_ids
        generated_text = ""
        
        for step in range(request.max_length):
            # Forward pass through quad-head model
            with torch.no_grad():
                outputs = model(current_ids, character_embedding=character_embedding)
                
                text_logits = outputs.get('text_logits')
                speech_logits = outputs.get('speech_logits') 
                control_logits = outputs.get('control_logits')
                memory_updates = outputs.get('memory_updates')
            
            # Sample next text token
            if text_logits is not None:
                next_token_logits = text_logits[0, -1, :] / request.temperature
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
                
                # Convert token to text (placeholder)
                token_text = f"token_{next_token.item()}"
                generated_text += token_text + " "
            
            # Generate speech frame if requested
            speech_frame = None
            if request.force_speech and speech_logits is not None:
                # Sample speech mel-spectrogram frame
                speech_frame_logits = speech_logits[0, -1, :] / request.speech_temperature
                speech_frame_quantized = torch.multinomial(torch.softmax(speech_frame_logits, dim=-1), 1)
                
                # Dequantize to mel values (4-bit to float)
                speech_frame = (speech_frame_quantized.float() / 15.0 - 0.5) * 2.0  # [-1, 1] range
                speech_frame = speech_frame.tolist()
            
            # Extract control signals
            control_signal = None
            if control_logits is not None:
                control_probs = torch.softmax(control_logits[0, -1, :], dim=-1)
                top_control = torch.argmax(control_probs)
                control_signal = f"control_{top_control.item()}"
            
            # Prepare stream response
            response = QuadHeadStreamResponse(
                type="generation_step",
                text_token=token_text if text_logits is not None else None,
                speech_frame=speech_frame,
                control_signal=control_signal,
                memory_update={"step": step} if memory_updates is not None else None,
                timestamp=asyncio.get_event_loop().time(),
                finished=(step >= request.max_length - 1)
            )
            
            # Send to client
            await connection_manager.send_to_connection(connection_id, response.model_dump())
            
            # Small delay for realistic streaming
            await asyncio.sleep(0.05)
            
            # Check for early stopping
            if generated_text.endswith("</s>") or step >= request.max_length - 1:
                break
        
        # Send completion message
        completion_response = QuadHeadStreamResponse(
            type="generation_complete",
            timestamp=asyncio.get_event_loop().time(),
            finished=True
        )
        await connection_manager.send_to_connection(connection_id, completion_response.model_dump())
        
    except Exception as e:
        logger.error(f"Error in multimodal generation: {e}")
        error_response = QuadHeadStreamResponse(
            type="error",
            error=str(e),
            timestamp=asyncio.get_event_loop().time(),
            finished=True
        )
        await connection_manager.send_to_connection(connection_id, error_response.model_dump())

@router.websocket("/stream/{character_id}")
async def quad_head_stream_websocket(websocket: WebSocket, character_id: str):
    """
    WebSocket endpoint for real-time quad-head multimodal generation
    
    Accepts generation requests and streams back:
    - Text tokens as they're generated
    - Speech mel-spectrogram frames in real-time
    - Control signals for narrative control
    - Memory updates for persistent context
    """
    connection_id = await connection_manager.connect(websocket, character_id)
    
    try:
        # Get model and dependencies
        model = get_quad_head_model()
        voice_manager = get_voice_manager()
        
        logger.info(f"Quad-head streaming session started for character: {character_id}")
        
        # Send welcome message
        welcome_msg = {
            "type": "connection_established",
            "message": f"Connected to quad-head streaming for character {character_id}",
            "capabilities": ["text_generation", "speech_synthesis", "control_signals", "memory_updates"],
            "timestamp": asyncio.get_event_loop().time()
        }
        await websocket.send_text(json.dumps(welcome_msg))
        
        while True:
            # Receive generation request
            try:
                data = await websocket.receive_text()
                request_data = json.loads(data)
                request = QuadHeadStreamRequest(**request_data)
                
                logger.info(f"Processing quad-head request: {request.text[:50]}...")
                
                if request.type == "generate_multimodal":
                    # Stream multimodal generation
                    await stream_multimodal_generation(model, request, connection_id)
                    
                elif request.type == "ping":
                    # Handle ping for connection keepalive
                    pong_response = {
                        "type": "pong",
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await websocket.send_text(json.dumps(pong_response))
                    
                else:
                    # Unknown request type
                    error_response = {
                        "type": "error",
                        "message": f"Unknown request type: {request.type}",
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await websocket.send_text(json.dumps(error_response))
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": asyncio.get_event_loop().time()
                }))
            except Exception as e:
                logger.error(f"Quad-head generation error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": f"Generation failed: {str(e)}",
                    "timestamp": asyncio.get_event_loop().time()
                }))
                
    except WebSocketDisconnect:
        logger.info(f"Quad-head WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Quad-head WebSocket error: {e}")
    finally:
        connection_manager.disconnect(connection_id)

@router.get("/stream/status")
async def streaming_status():
    """Get current streaming status and connected clients"""
    return {
        "status": "active",
        "connected_clients": len(connection_manager.connections),
        "character_sessions": len(connection_manager.character_sessions),
        "model_loaded": _quad_head_model is not None
    } 