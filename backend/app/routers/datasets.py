from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime
import json
import asyncio

from ..database import get_db
from ..auth import get_current_user
from ..models import User, Dataset, Character, World, Project
from ..schemas import (
    DatasetBase, DatasetCreate, DatasetResponse, DatasetListResponse,
    DatasetGenerationParams
)
from ..redis_client import redis_client
from ..celery_app import celery_app

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# WebSocket manager for real-time updates
class DatasetConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, dataset_id: str):
        await websocket.accept()
        if dataset_id not in self.active_connections:
            self.active_connections[dataset_id] = []
        self.active_connections[dataset_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, dataset_id: str):
        if dataset_id in self.active_connections:
            self.active_connections[dataset_id].remove(websocket)
            if not self.active_connections[dataset_id]:
                del self.active_connections[dataset_id]
    
    async def send_update(self, dataset_id: str, update: dict):
        if dataset_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[dataset_id]:
                try:
                    await connection.send_json(update)
                except:
                    disconnected.append(connection)
            
            # Clean up disconnected websockets
            for conn in disconnected:
                self.disconnect(conn, dataset_id)

manager = DatasetConnectionManager()

@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    dataset: DatasetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new dataset for a character"""
    # Verify character ownership
    character = db.query(Character).join(World).join(Project).filter(
        Character.id == dataset.character_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found or unauthorized"
        )
    
    # Create dataset
    db_dataset = Dataset(
        character_id=dataset.character_id,
        name=dataset.name,
        description=dataset.description,
        generation_params=dataset.generation_params.dict() if dataset.generation_params else {}
    )
    
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    
    # Clear cache
    await redis_client.delete(f"datasets:character:{dataset.character_id}")
    
    return db_dataset

@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific dataset"""
    # Check cache first
    cached = await redis_client.get(f"dataset:{dataset_id}")
    if cached:
        return json.loads(cached)
    
    # Query database
    dataset = db.query(Dataset).join(Character).join(World).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Cache for 2 minutes
    await redis_client.setex(
        f"dataset:{dataset_id}",
        120,
        json.dumps({
            "id": dataset.id,
            "character_id": dataset.character_id,
            "name": dataset.name,
            "description": dataset.description,
            "conversation_count": dataset.conversation_count,
            "total_messages": dataset.total_messages,
            "file_path": dataset.file_path,
            "format": dataset.format,
            "generation_params": dataset.generation_params,
            "status": dataset.status,
            "error_message": dataset.error_message,
            "created_at": dataset.created_at.isoformat(),
            "updated_at": dataset.updated_at.isoformat()
        })
    )
    
    return dataset

@router.get("/character/{character_id}", response_model=List[DatasetListResponse])
async def list_character_datasets(
    character_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all datasets for a character"""
    # Verify character ownership
    character = db.query(Character).join(World).join(Project).filter(
        Character.id == character_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found or unauthorized"
        )
    
    # Check cache
    cached = await redis_client.get(f"datasets:character:{character_id}")
    if cached:
        return json.loads(cached)
    
    # Query datasets
    datasets = db.query(Dataset).filter(
        Dataset.character_id == character_id
    ).order_by(Dataset.created_at.desc()).all()
    
    # Format response
    result = [{
        "id": dataset.id,
        "name": dataset.name,
        "conversation_count": dataset.conversation_count,
        "status": dataset.status,
        "created_at": dataset.created_at.isoformat()
    } for dataset in datasets]
    
    # Cache for 30 seconds
    await redis_client.setex(
        f"datasets:character:{character_id}",
        30,
        json.dumps(result)
    )
    
    return result

@router.post("/{dataset_id}/generate")
async def start_generation(
    dataset_id: str,
    params: DatasetGenerationParams,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start dataset generation"""
    # Get dataset with authorization check
    dataset = db.query(Dataset).join(Character).join(World).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    if dataset.status == "generating":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Generation already in progress"
        )
    
    # Update status
    dataset.status = "generating"
    dataset.generation_params = params.dict()
    db.commit()
    
    # Start Celery task
    task = celery_app.send_task(
        "dataset_generation",
        args=[dataset_id, params.dict()]
    )
    
    # Store task ID in Redis
    await redis_client.setex(
        f"dataset:task:{dataset_id}",
        3600,  # 1 hour
        task.id
    )
    
    # Clear cache
    await redis_client.delete(f"dataset:{dataset_id}")
    await redis_client.delete(f"datasets:character:{dataset.character_id}")
    
    return {
        "message": "Generation started",
        "task_id": task.id,
        "dataset_id": dataset_id
    }

@router.post("/{dataset_id}/stop")
async def stop_generation(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Stop dataset generation"""
    # Get dataset with authorization check
    dataset = db.query(Dataset).join(Character).join(World).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Get task ID from Redis
    task_id = await redis_client.get(f"dataset:task:{dataset_id}")
    if task_id:
        # Cancel Celery task
        celery_app.control.revoke(task_id, terminate=True)
    
    # Update status
    dataset.status = "cancelled"
    db.commit()
    
    # Clear cache
    await redis_client.delete(f"dataset:{dataset_id}")
    await redis_client.delete(f"datasets:character:{dataset.character_id}")
    
    # Notify WebSocket clients
    await manager.send_update(dataset_id, {
        "type": "status",
        "status": "cancelled"
    })
    
    return {"message": "Generation stopped"}

@router.get("/{dataset_id}/progress")
async def get_generation_progress(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get generation progress"""
    # Verify ownership
    dataset = db.query(Dataset).join(Character).join(World).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Get progress from Redis
    progress_data = await redis_client.get(f"dataset:progress:{dataset_id}")
    if progress_data:
        progress = json.loads(progress_data)
    else:
        progress = {
            "current": 0,
            "total": 0,
            "percentage": 0,
            "status": dataset.status,
            "current_topic": None
        }
    
    return progress

@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a dataset"""
    # Get dataset with authorization check
    dataset = db.query(Dataset).join(Character).join(World).join(Project).filter(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    character_id = dataset.character_id
    
    # Delete dataset
    db.delete(dataset)
    db.commit()
    
    # Clear cache
    await redis_client.delete(f"dataset:{dataset_id}")
    await redis_client.delete(f"datasets:character:{character_id}")
    
    return {"message": "Dataset deleted successfully"}

@router.websocket("/ws/{dataset_id}")
async def dataset_websocket(
    websocket: WebSocket,
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time generation updates"""
    await manager.connect(websocket, dataset_id)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
            
            # Check for updates in Redis
            progress_data = await redis_client.get(f"dataset:progress:{dataset_id}")
            if progress_data:
                progress = json.loads(progress_data)
                await websocket.send_json({
                    "type": "progress",
                    **progress
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, dataset_id)

# Helper function to update progress (called from Celery tasks)
async def update_dataset_progress(
    dataset_id: str,
    current: int,
    total: int,
    current_topic: Optional[str] = None
):
    """Update dataset generation progress"""
    progress = {
        "current": current,
        "total": total,
        "percentage": (current / total * 100) if total > 0 else 0,
        "current_topic": current_topic
    }
    
    # Store in Redis
    await redis_client.setex(
        f"dataset:progress:{dataset_id}",
        300,  # 5 minutes
        json.dumps(progress)
    )
    
    # Send WebSocket update
    await manager.send_update(dataset_id, {
        "type": "progress",
        **progress
    }) 