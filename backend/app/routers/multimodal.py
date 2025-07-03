"""Multimodal dataset generation API endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
import json
import asyncio
import logging
import shutil

from ..database import get_db
from ..auth import get_current_user
from ..models import User, MultimodalDataset
from ..schemas import (
    MultimodalGenerationConfig, 
    MultimodalDatasetResponse, 
    MultimodalProgressUpdate
)
from ..redis_client import redis_client
from ..celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/multimodal", tags=["multimodal"])

# WebSocket manager for real-time progress updates
class MultimodalWebSocketManager:
    def __init__(self):
        self.connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        """Connect websocket to job updates"""
        await websocket.accept()
        if job_id not in self.connections:
            self.connections[job_id] = []
        self.connections[job_id].append(websocket)
        logger.info(f"WebSocket connected for job {job_id}")
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        """Disconnect websocket from job updates"""
        if job_id in self.connections:
            try:
                self.connections[job_id].remove(websocket)
                if not self.connections[job_id]:
                    del self.connections[job_id]
                logger.info(f"WebSocket disconnected for job {job_id}")
            except ValueError:
                pass  # Websocket already removed
    
    async def broadcast_progress(self, job_id: str, update: dict):
        """Broadcast progress update to all connected websockets for this job"""
        if job_id in self.connections:
            disconnected = []
            for connection in self.connections[job_id]:
                try:
                    await connection.send_json(update)
                except Exception as e:
                    logger.warning(f"Failed to send update to websocket: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected websockets
            for conn in disconnected:
                self.disconnect(conn, job_id)

multimodal_manager = MultimodalWebSocketManager()


def check_disk_space(required_gb: float = 10.0) -> bool:
    """Check if there's enough disk space for generation"""
    try:
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        return free_gb >= required_gb
    except Exception:
        return True  # If we can't check, assume it's fine


def estimate_dataset_warnings(config: MultimodalGenerationConfig) -> List[str]:
    """Estimate warnings for large dataset generation"""
    warnings = []
    
    # Memory warnings for large datasets
    if config.sampleCount > 50000:
        warnings.append("Large dataset may require significant memory and processing time")
    
    # Real TTS warnings
    if not config.useMockTTS:
        warnings.append("Real TTS generation will significantly increase processing time")
        if config.sampleCount > 10000:
            warnings.append("Large dataset with real TTS may take several hours to complete")
    
    return warnings


@router.post("/generate", response_model=MultimodalDatasetResponse)
async def start_multimodal_generation(
    config: MultimodalGenerationConfig,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start multimodal dataset generation"""
    
    # Check disk space
    if not check_disk_space():
        raise HTTPException(
            status_code=507,
            detail="Insufficient disk space for dataset generation"
        )
    
    # Generate warnings
    warnings = estimate_dataset_warnings(config)
    
    # Create multimodal dataset record
    multimodal_dataset = MultimodalDataset(
        name=config.name,
        config=config.dict(),
        status="pending",
        progress=0.0,
        samples_generated=0,
        total_samples=config.sampleCount,
        tts_provider=config.ttsProvider,
        character_count=config.characterCount,
        narrative_types=config.narrativeTypes
    )
    
    db.add(multimodal_dataset)
    db.commit()
    db.refresh(multimodal_dataset)
    
    try:
        # Start Celery task for multimodal generation
        task = celery_app.send_task(
            "multimodal_generation.generate_dataset",
            args=[multimodal_dataset.id, config.dict()]
        )
        
        # Store task ID
        multimodal_dataset.celery_task_id = task.id
        db.commit()
        
        # Store task ID in Redis for quick lookup
        await redis_client.setex(
            f"multimodal:task:{multimodal_dataset.id}",
            3600,  # 1 hour
            task.id
        )
        
        logger.info(f"Started multimodal generation job {multimodal_dataset.id} with task {task.id}")
        
        return MultimodalDatasetResponse(
            id=multimodal_dataset.id,
            name=multimodal_dataset.name,
            status=multimodal_dataset.status,
            progress=multimodal_dataset.progress,
            config=multimodal_dataset.config,
            samplesGenerated=multimodal_dataset.samples_generated,
            totalSamples=multimodal_dataset.total_samples,
            currentStep=multimodal_dataset.current_step,
            outputPath=multimodal_dataset.output_path,
            errorMessage=multimodal_dataset.error_message,
            createdAt=multimodal_dataset.created_at,
            updatedAt=multimodal_dataset.updated_at,
            warnings=warnings if warnings else None
        )
        
    except Exception as e:
        # Update status to failed if task creation fails
        multimodal_dataset.status = "failed"
        multimodal_dataset.error_message = f"TTS service unavailable: {str(e)}"
        db.commit()
        
        raise HTTPException(
            status_code=503,
            detail=f"TTS service unavailable: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=MultimodalDatasetResponse)
async def get_multimodal_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get multimodal dataset job details"""
    
    # Check cache first
    cached = await redis_client.get(f"multimodal:job:{job_id}")
    if cached:
        return MultimodalDatasetResponse.parse_raw(cached)
    
    # Query database
    job = db.query(MultimodalDataset).filter(
        MultimodalDataset.id == job_id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Multimodal job not found"
        )
    
    response = MultimodalDatasetResponse(
        id=job.id,
        name=job.name,
        status=job.status,
        progress=job.progress,
        config=job.config,
        samplesGenerated=job.samples_generated,
        totalSamples=job.total_samples,
        currentStep=job.current_step,
        outputPath=job.output_path,
        errorMessage=job.error_message,
        createdAt=job.created_at,
        updatedAt=job.updated_at
    )
    
    # Cache for 30 seconds
    await redis_client.setex(
        f"multimodal:job:{job_id}",
        30,
        response.json()
    )
    
    return response


@router.get("/jobs/{job_id}/progress", response_model=MultimodalProgressUpdate)
async def get_generation_progress(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get generation progress for a specific job"""
    
    # Verify job exists and user has access
    job = db.query(MultimodalDataset).filter(
        MultimodalDataset.id == job_id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Multimodal job not found"
        )
    
    return MultimodalProgressUpdate(
        progress=job.progress,
        currentStep=job.current_step or "Initializing...",
        samplesGenerated=job.samples_generated,
        totalSamples=job.total_samples
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_generation(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel a running multimodal generation job"""
    
    # Get job
    job = db.query(MultimodalDataset).filter(
        MultimodalDataset.id == job_id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Multimodal job not found"
        )
    
    # Check if job can be cancelled
    if job.status in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status}"
        )
    
    # Cancel Celery task if it exists
    if job.celery_task_id:
        try:
            celery_app.control.revoke(job.celery_task_id, terminate=True)
            logger.info(f"Cancelled Celery task {job.celery_task_id} for job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to cancel Celery task: {e}")
    
    # Update job status
    job.status = "cancelled"
    job.current_step = "Cancelled by user"
    db.commit()
    
    # Clear cache
    await redis_client.delete(f"multimodal:job:{job_id}")
    
    # Broadcast cancellation to websockets
    await multimodal_manager.broadcast_progress(job_id, {
        "status": "cancelled",
        "currentStep": "Cancelled by user"
    })
    
    return {"status": "cancelled", "message": "Job cancellation initiated"}


@router.get("/jobs/{job_id}/download")
async def download_dataset(
    job_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download completed multimodal dataset"""
    
    # Get job
    job = db.query(MultimodalDataset).filter(
        MultimodalDataset.id == job_id
    ).first()
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Multimodal job not found"
        )
    
    # Check if job is completed
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset generation not completed"
        )
    
    # Check if output file exists
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset file not found"
        )
    
    # Return file download
    return FileResponse(
        path=job.output_path,
        filename=f"multimodal_dataset_{job_id}.zip",
        media_type="application/zip"
    )


@router.websocket("/ws/{job_id}")
async def multimodal_websocket(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time multimodal generation progress updates"""
    await multimodal_manager.connect(websocket, job_id)
    
    try:
        while True:
            # Keep connection alive and wait for messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        multimodal_manager.disconnect(websocket, job_id)
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        multimodal_manager.disconnect(websocket, job_id)


# Utility function to update job progress (called by Celery tasks)
async def update_job_progress(
    job_id: str, 
    progress: float, 
    message: str, 
    samples_done: int,
    db_session: Optional[Session] = None
):
    """Update job progress in database and broadcast via WebSocket"""
    
    # Update database
    if db_session:
        job = db_session.query(MultimodalDataset).filter(
            MultimodalDataset.id == job_id
        ).first()
        
        if job:
            job.progress = progress
            job.current_step = message
            job.samples_generated = samples_done
            db_session.commit()
    
    # Broadcast to WebSocket connections
    await multimodal_manager.broadcast_progress(job_id, {
        "progress": progress,
        "currentStep": message,
        "samplesGenerated": samples_done
    })
    
    logger.info(f"Job {job_id} progress: {progress:.1%} - {message}") 