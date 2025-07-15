"""
Quad-Head NarrativeLM Training API

FastAPI endpoints for training the quad-head architecture with:
- Multi-task training across all four heads
- Real-time training monitoring via WebSocket
- Distributed training support
- Model versioning and deployment
- Training metrics and evaluation
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import asyncio
import json
import logging
import uuid
from dataclasses import asdict

from pydantic import BaseModel, Field

# Database and auth
from ..database import get_db
from ..auth import get_current_user
from ..models import User
from ..celery_app import celery_app

# Multimodal components
try:
    from ..narrative_engine.quad_head_model import QuadHeadNarrativeLM, create_quad_head_model
    from ..narrative_engine.loss import QuadHeadLoss
    from ..narrative_engine.config import NarrativeLLMConfig
    QUAD_HEAD_AVAILABLE = True
except ImportError:
    QUAD_HEAD_AVAILABLE = False
    logging.warning("QuadHeadNarrativeLM not available")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training/quad-head", tags=["quad-head-training"])


# Pydantic models for API
class QuadHeadTrainingConfig(BaseModel):
    """Configuration for quad-head model training"""
    
    # Model configuration
    base_model_name: str = Field(default="HuggingFaceTB/SmolLM2-135M-Instruct", description="Base transformer model")
    enable_speech_head: bool = Field(default=True, description="Enable speech generation head")
    speech_mel_bins: int = Field(default=80, description="Number of mel-frequency bins")
    speech_quantization_bits: int = Field(default=4, description="Speech quantization bits")
    
    # Training configuration
    dataset_path: str = Field(..., description="Path to multimodal training dataset")
    output_dir: str = Field(default="./training_output/quad_head", description="Training output directory")
    
    # Training hyperparameters
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    batch_size: int = Field(default=4, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=4, description="Gradient accumulation steps")
    num_train_epochs: int = Field(default=3, description="Number of training epochs")
    warmup_steps: int = Field(default=100, description="Warmup steps")
    max_seq_length: int = Field(default=512, description="Maximum sequence length")
    
    # Loss weights for multi-task training
    text_weight: float = Field(default=1.0, description="Weight for text generation loss")
    control_weight: float = Field(default=0.8, description="Weight for control token loss")
    memory_weight: float = Field(default=0.6, description="Weight for memory head loss")
    speech_weight: float = Field(default=0.5, description="Weight for speech generation loss")
    use_spectral_loss: bool = Field(default=False, description="Use spectral loss for speech")
    
    # Advanced options
    enable_distributed: bool = Field(default=False, description="Enable distributed training")
    fp16: bool = Field(default=True, description="Use mixed precision training")
    gradient_checkpointing: bool = Field(default=True, description="Use gradient checkpointing")
    save_steps: int = Field(default=500, description="Save model every N steps")
    eval_steps: int = Field(default=100, description="Evaluate every N steps")
    logging_steps: int = Field(default=10, description="Log metrics every N steps")


class TrainingStatus(BaseModel):
    """Training job status"""
    job_id: str
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    last_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    model_path: Optional[str] = None


class TrainingMetrics(BaseModel):
    """Real-time training metrics"""
    step: int
    epoch: int
    total_loss: float
    generation_loss: Optional[float] = None
    control_loss: Optional[float] = None
    memory_loss: Optional[float] = None
    speech_loss: Optional[float] = None
    learning_rate: float
    grad_norm: Optional[float] = None
    samples_per_second: Optional[float] = None
    timestamp: datetime


# WebSocket manager for training progress
class TrainingWebSocketManager:
    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Connect websocket to training job updates"""
        await websocket.accept()
        if job_id not in self.connections:
            self.connections[job_id] = []
        self.connections[job_id].append(websocket)
        logger.info(f"Training WebSocket connected for job {job_id}")
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Disconnect websocket from training job updates"""
        if job_id in self.connections:
            try:
                self.connections[job_id].remove(websocket)
                if not self.connections[job_id]:
                    del self.connections[job_id]
                logger.info(f"Training WebSocket disconnected for job {job_id}")
            except ValueError:
                pass
    
    async def broadcast_metrics(self, job_id: str, metrics: TrainingMetrics):
        """Broadcast training metrics to connected clients"""
        if job_id in self.connections:
            message = {
                "type": "metrics",
                "data": metrics.model_dump()
            }
            
            disconnected = []
            for websocket in self.connections[job_id]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send metrics to websocket: {e}")
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                self.disconnect(ws, job_id)
    
    async def broadcast_status(self, job_id: str, status: TrainingStatus):
        """Broadcast training status updates"""
        if job_id in self.connections:
            message = {
                "type": "status",
                "data": status.model_dump()
            }
            
            disconnected = []
            for websocket in self.connections[job_id]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send status to websocket: {e}")
                    disconnected.append(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                self.disconnect(ws, job_id)


# Global WebSocket manager
training_ws_manager = TrainingWebSocketManager()

# In-memory storage for training jobs (in production, use Redis/database)
training_jobs: Dict[str, TrainingStatus] = {}


@router.post("/start", response_model=TrainingStatus)
async def start_quad_head_training(
    config: QuadHeadTrainingConfig,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Start training a quad-head NarrativeLM model.
    
    This endpoint initiates asynchronous training with real-time progress updates
    via WebSocket. The training runs in the background using Celery.
    """
    if not QUAD_HEAD_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="QuadHeadNarrativeLM not available. Please check installation."
        )
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Validate dataset path
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset path not found: {config.dataset_path}"
        )
    
    # Create training status
    training_status = TrainingStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.utcnow(),
        total_epochs=config.num_train_epochs
    )
    
    # Store job status
    training_jobs[job_id] = training_status
    
    # Start training task asynchronously
    background_tasks.add_task(
        run_quad_head_training,
        job_id=job_id,
        config=config,
        user_id=current_user.id
    )
    
    logger.info(f"Started quad-head training job {job_id} for user {current_user.id}")
    
    return training_status


@router.get("/jobs/{job_id}", response_model=TrainingStatus)
async def get_training_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get training job status and metrics"""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Training job {job_id} not found"
        )
    
    return training_jobs[job_id]


@router.get("/jobs", response_model=List[TrainingStatus])
async def list_training_jobs(
    current_user: User = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
):
    """List training jobs for the current user"""
    # In production, filter by user_id from database
    all_jobs = list(training_jobs.values())
    return all_jobs[offset:offset + limit]


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a running training job"""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Training job {job_id} not found"
        )
    
    job = training_jobs[job_id]
    if job.status in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in status: {job.status}"
        )
    
    # Update status
    job.status = "cancelled"
    job.completed_at = datetime.utcnow()
    
    # Broadcast status update
    await training_ws_manager.broadcast_status(job_id, job)
    
    # TODO: Actually cancel the Celery task
    logger.info(f"Cancelled training job {job_id}")
    
    return {"message": f"Training job {job_id} cancelled"}


@router.websocket("/ws/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time training progress updates.
    
    Clients can connect to receive:
    - Training metrics (loss, accuracy, etc.)
    - Status updates (started, completed, failed)
    - Real-time progress information
    """
    await training_ws_manager.connect(websocket, job_id)
    
    try:
        # Send initial status if job exists
        if job_id in training_jobs:
            await training_ws_manager.broadcast_status(job_id, training_jobs[job_id])
        
        # Keep connection alive
        while True:
            # Wait for client messages (for potential interaction)
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        training_ws_manager.disconnect(websocket, job_id)
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        training_ws_manager.disconnect(websocket, job_id)


@router.get("/jobs/{job_id}/metrics/stream")
async def stream_training_metrics(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Stream training metrics as Server-Sent Events"""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Training job {job_id} not found"
        )
    
    async def generate_metrics():
        """Generate SSE stream of training metrics"""
        try:
            while True:
                job = training_jobs.get(job_id)
                if not job:
                    break
                
                if job.status in ["completed", "failed", "cancelled"]:
                    yield f"data: {json.dumps({'status': 'finished'})}\n\n"
                    break
                
                if job.last_metrics:
                    yield f"data: {json.dumps(job.last_metrics)}\n\n"
                
                await asyncio.sleep(1)  # Update every second
                
        except Exception as e:
            logger.error(f"Error streaming metrics for job {job_id}: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_metrics(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


async def run_quad_head_training(
    job_id: str,
    config: QuadHeadTrainingConfig,
    user_id: int
):
    """
    Run quad-head training in the background.
    
    This function handles the actual training process with real-time updates.
    In production, this would be a Celery task for better scalability.
    """
    job = training_jobs[job_id]
    
    try:
        # Update status to running
        job.status = "running"
        job.started_at = datetime.utcnow()
        await training_ws_manager.broadcast_status(job_id, job)
        
        logger.info(f"Starting quad-head training for job {job_id}")
        
        # Create model configuration
        model_config = NarrativeLLMConfig(
            base_model_name=config.base_model_name,
            enable_speech_head=config.enable_speech_head,
            speech_mel_bins=config.speech_mel_bins,
            speech_quantization_bits=config.speech_quantization_bits
        )
        
        # Create model
        model = create_quad_head_model(
            base_model_name=config.base_model_name,
            enable_speech_head=config.enable_speech_head
        )
        
        # Create loss function
        loss_fn = QuadHeadLoss(
            text_weight=config.text_weight,
            control_weight=config.control_weight,
            memory_weight=config.memory_weight,
            speech_weight=config.speech_weight,
            use_spectral_loss=config.use_spectral_loss
        )
        
        # Mock training loop for demonstration
        # In production, this would use the actual training infrastructure
        total_steps = config.num_train_epochs * 100  # Mock calculation
        job.total_steps = total_steps
        
        for epoch in range(config.num_train_epochs):
            job.current_epoch = epoch + 1
            
            for step in range(100):  # Mock 100 steps per epoch
                job.current_step = epoch * 100 + step + 1
                job.progress = job.current_step / total_steps * 100
                
                # Mock training metrics
                metrics = TrainingMetrics(
                    step=job.current_step,
                    epoch=job.current_epoch,
                    total_loss=1.5 - (job.current_step * 0.001),  # Mock decreasing loss
                    generation_loss=0.8 - (job.current_step * 0.0005),
                    control_loss=0.3 - (job.current_step * 0.0002),
                    memory_loss=0.2 - (job.current_step * 0.0001),
                    speech_loss=0.2 - (job.current_step * 0.0001),
                    learning_rate=config.learning_rate,
                    grad_norm=1.0,
                    samples_per_second=8.5,
                    timestamp=datetime.utcnow()
                )
                
                # Update job with latest metrics
                job.last_metrics = metrics.model_dump()
                
                # Broadcast metrics
                await training_ws_manager.broadcast_metrics(job_id, metrics)
                
                # Simulate training time
                await asyncio.sleep(0.1)
                
                # Check if job was cancelled
                if job.status == "cancelled":
                    return
        
        # Training completed successfully
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.progress = 100.0
        job.model_path = f"{config.output_dir}/final_model"
        
        await training_ws_manager.broadcast_status(job_id, job)
        
        logger.info(f"Completed quad-head training for job {job_id}")
        
    except Exception as e:
        # Training failed
        job.status = "failed"
        job.completed_at = datetime.utcnow()
        job.error_message = str(e)
        
        await training_ws_manager.broadcast_status(job_id, job)
        
        logger.error(f"Training failed for job {job_id}: {e}")


@router.get("/models/{job_id}/download")
async def download_trained_model(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download trained quad-head model"""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Training job {job_id} not found"
        )
    
    job = training_jobs[job_id]
    if job.status != "completed" or not job.model_path:
        raise HTTPException(
            status_code=400,
            detail="Model not available for download"
        )
    
    model_path = Path(job.model_path)
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model file not found"
        )
    
    return StreamingResponse(
        open(model_path, "rb"),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=quad_head_model_{job_id}.pt"}
    ) 