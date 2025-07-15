"""
FastAPI router for Multimodal Studio production features.

This router provides endpoints for:
- Advanced job management with batch operations
- Dataset quality validation and reporting
- Multi-format export with progress tracking
- Performance monitoring and optimization
- User preferences and workspace management
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import asyncio
import json
import uuid
from pathlib import Path

from ..auth import get_current_user
from ..models import User
from ..services.multimodal_studio.job_manager import MultimodalJobManager
from ..services.multimodal_studio.quality_validator import DatasetQualityValidator
from ..services.multimodal_studio.export_manager import DatasetExportManager
from ..services.multimodal_studio.performance_monitor import StudioPerformanceMonitor
from ..services.multimodal_studio.preferences_manager import UserPreferencesManager
from ..websocket.manager import WebSocketManager

router = APIRouter(prefix="/api/multimodal-studio", tags=["multimodal-studio"])

# WebSocket manager for real-time updates
ws_manager = WebSocketManager()

# Service instances
job_manager = MultimodalJobManager()
quality_validator = DatasetQualityValidator()
export_manager = DatasetExportManager()
performance_monitor = StudioPerformanceMonitor()
preferences_manager = UserPreferencesManager()

# Pydantic models for request/response
from pydantic import BaseModel, Field
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "queued"
    GENERATING = "generating"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobAction(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    DELETE = "delete"
    PRIORITIZE = "prioritize"

class ExportFormat(str, Enum):
    HUGGINGFACE = "huggingface"
    JSONL = "jsonl"
    PYTORCH = "pytorch"
    CUSTOM = "custom"

class MultimodalJob(BaseModel):
    id: str
    user_id: str
    dataset_id: str
    status: JobStatus
    progress: float = 0.0
    total_steps: int = 0
    current_step: int = 0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    configuration: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    priority: int = 5  # 1 (highest) to 10 (lowest)

class BatchJobRequest(BaseModel):
    action: JobAction
    job_ids: List[str]
    options: Dict[str, Any] = {}

class QualityMetrics(BaseModel):
    dataset_id: str
    overall_score: float
    consistency_score: float
    diversity_score: float
    quality_score: float
    character_adherence: float
    dialogue_quality: float
    issues_count: int
    suggestions_count: int
    timestamp: datetime

class ValidationReport(BaseModel):
    id: str
    dataset_id: str
    metrics: QualityMetrics
    issues: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    generated_at: datetime

class ExportConfig(BaseModel):
    id: str
    name: str
    format: ExportFormat
    options: Dict[str, Any]
    created_at: datetime

class ExportJob(BaseModel):
    id: str
    dataset_id: str
    config_id: str
    user_id: str
    status: JobStatus
    progress: float = 0.0
    file_path: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class UserPreferences(BaseModel):
    user_id: str
    theme: str = "light"
    auto_save_interval: int = 30
    notifications_enabled: bool = True
    keyboard_shortcuts_enabled: bool = True
    default_export_format: ExportFormat = ExportFormat.JSONL
    workspace_layout: Dict[str, Any] = {}
    updated_at: datetime

class WorkspaceConfig(BaseModel):
    id: str
    name: str
    user_id: str
    configuration: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

# Job Management Endpoints

@router.get("/jobs", response_model=List[MultimodalJob])
async def get_user_jobs(
    user: User = Depends(get_current_user),
    status: Optional[JobStatus] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get all jobs for the current user with optional filtering."""
    return await job_manager.get_user_jobs(
        user_id=user.id,
        status=status.value if status else None,
        limit=limit,
        offset=offset
    )

@router.get("/jobs/{job_id}", response_model=MultimodalJob)
async def get_job(
    job_id: str,
    user: User = Depends(get_current_user)
):
    """Get specific job details."""
    job = await job_manager.get_job(job_id, user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.post("/jobs/batch-action")
async def batch_job_action(
    request: BatchJobRequest,
    user: User = Depends(get_current_user)
):
    """Execute batch operations on multiple jobs."""
    results = await job_manager.batch_operation(
        action=request.action.value,
        job_ids=request.job_ids,
        user_id=user.id,
        options=request.options
    )
    
    # Broadcast updates via WebSocket
    for result in results:
        if result['success']:
            await ws_manager.broadcast_to_user(
                user_id=user.id,
                message={
                    'type': 'job_action_complete',
                    'job_id': result['job_id'],
                    'action': request.action.value,
                    'status': result.get('new_status')
                }
            )
    
    return {"results": results}

@router.post("/jobs/{job_id}/priority")
async def update_job_priority(
    job_id: str,
    priority: int = Field(..., ge=1, le=10),
    user: User = Depends(get_current_user)
):
    """Update job priority (1 = highest, 10 = lowest)."""
    success = await job_manager.update_priority(job_id, priority, user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found")
    
    await ws_manager.broadcast_to_user(
        user_id=user.id,
        message={
            'type': 'job_priority_updated',
            'job_id': job_id,
            'priority': priority
        }
    )
    
    return {"success": True, "priority": priority}

@router.get("/jobs/queue")
async def get_job_queue(user: User = Depends(get_current_user)):
    """Get current job queue status."""
    queue_info = await job_manager.get_queue_status(user.id)
    return queue_info

# Quality Validation Endpoints

@router.post("/datasets/{dataset_id}/validate", response_model=ValidationReport)
async def validate_dataset_quality(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
):
    """Start comprehensive dataset quality validation."""
    # Start validation in background
    report_id = str(uuid.uuid4())
    
    background_tasks.add_task(
        quality_validator.validate_dataset_async,
        dataset_id=dataset_id,
        user_id=user.id,
        report_id=report_id
    )
    
    return {"report_id": report_id, "status": "started"}

@router.get("/datasets/{dataset_id}/quality-reports", response_model=List[ValidationReport])
async def get_quality_reports(
    dataset_id: str,
    user: User = Depends(get_current_user),
    limit: int = 10
):
    """Get quality validation reports for a dataset."""
    return await quality_validator.get_reports(dataset_id, user.id, limit)

@router.get("/quality-reports/{report_id}", response_model=ValidationReport)
async def get_quality_report(
    report_id: str,
    user: User = Depends(get_current_user)
):
    """Get specific quality validation report."""
    report = await quality_validator.get_report(report_id, user.id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

@router.post("/datasets/{dataset_id}/quality-suggestions/apply")
async def apply_quality_suggestions(
    dataset_id: str,
    suggestion_ids: List[str],
    user: User = Depends(get_current_user)
):
    """Apply quality improvement suggestions to dataset."""
    results = await quality_validator.apply_suggestions(
        dataset_id=dataset_id,
        suggestion_ids=suggestion_ids,
        user_id=user.id
    )
    return {"applied": results}

# Export Management Endpoints

@router.get("/export-configs", response_model=List[ExportConfig])
async def get_export_configs(user: User = Depends(get_current_user)):
    """Get all export configurations for user."""
    return await export_manager.get_user_configs(user.id)

@router.post("/export-configs", response_model=ExportConfig)
async def create_export_config(
    name: str,
    format: ExportFormat,
    options: Dict[str, Any],
    user: User = Depends(get_current_user)
):
    """Create new export configuration."""
    return await export_manager.create_config(
        user_id=user.id,
        name=name,
        format=format.value,
        options=options
    )

@router.post("/datasets/{dataset_id}/export")
async def start_dataset_export(
    dataset_id: str,
    config_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
):
    """Start dataset export with specified configuration."""
    export_job = await export_manager.start_export(
        dataset_id=dataset_id,
        config_id=config_id,
        user_id=user.id
    )
    
    # Start export in background
    background_tasks.add_task(
        export_manager.process_export,
        export_job.id
    )
    
    return export_job

@router.get("/exports", response_model=List[ExportJob])
async def get_export_jobs(
    user: User = Depends(get_current_user),
    status: Optional[JobStatus] = None,
    limit: int = 20
):
    """Get export jobs for user."""
    return await export_manager.get_user_exports(
        user_id=user.id,
        status=status.value if status else None,
        limit=limit
    )

@router.get("/exports/{export_id}/download")
async def download_export(
    export_id: str,
    user: User = Depends(get_current_user)
):
    """Download completed export file."""
    export_job = await export_manager.get_export(export_id, user.id)
    if not export_job:
        raise HTTPException(status_code=404, detail="Export not found")
    
    if export_job.status != JobStatus.COMPLETED or not export_job.file_path:
        raise HTTPException(status_code=400, detail="Export not ready for download")
    
    file_path = Path(export_job.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")
    
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type='application/octet-stream'
    )

@router.post("/exports/batch")
async def batch_export_datasets(
    dataset_ids: List[str],
    config_id: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
):
    """Start batch export of multiple datasets."""
    export_jobs = []
    
    for dataset_id in dataset_ids:
        export_job = await export_manager.start_export(
            dataset_id=dataset_id,
            config_id=config_id,
            user_id=user.id
        )
        export_jobs.append(export_job)
        
        # Start each export in background
        background_tasks.add_task(
            export_manager.process_export,
            export_job.id
        )
    
    return {"export_jobs": export_jobs}

# Performance Monitoring Endpoints

@router.get("/performance/metrics")
async def get_performance_metrics(user: User = Depends(get_current_user)):
    """Get current performance metrics."""
    return await performance_monitor.collect_studio_metrics(user.id)

@router.get("/performance/bottlenecks")
async def detect_bottlenecks(user: User = Depends(get_current_user)):
    """Detect current performance bottlenecks."""
    metrics = await performance_monitor.collect_studio_metrics(user.id)
    bottlenecks = await performance_monitor.detect_bottlenecks(metrics)
    return {"bottlenecks": bottlenecks}

@router.get("/performance/optimization-suggestions")
async def get_optimization_suggestions(user: User = Depends(get_current_user)):
    """Get AI-powered optimization suggestions."""
    suggestions = await performance_monitor.generate_optimization_suggestions(user.id)
    return {"suggestions": suggestions}

@router.get("/performance/history")
async def get_performance_history(
    user: User = Depends(get_current_user),
    hours: int = 24
):
    """Get performance metrics history."""
    since = datetime.utcnow() - timedelta(hours=hours)
    return await performance_monitor.get_metrics_history(user.id, since)

# User Preferences & Workspace Endpoints

@router.get("/preferences", response_model=UserPreferences)
async def get_user_preferences(user: User = Depends(get_current_user)):
    """Get user preferences."""
    prefs = await preferences_manager.get_preferences(user.id)
    return prefs or UserPreferences(user_id=user.id, updated_at=datetime.utcnow())

@router.put("/preferences", response_model=UserPreferences)
async def update_user_preferences(
    preferences: UserPreferences,
    user: User = Depends(get_current_user)
):
    """Update user preferences."""
    preferences.user_id = user.id
    preferences.updated_at = datetime.utcnow()
    return await preferences_manager.save_preferences(preferences)

@router.get("/workspaces", response_model=List[WorkspaceConfig])
async def get_workspaces(user: User = Depends(get_current_user)):
    """Get user workspace configurations."""
    return await preferences_manager.get_workspaces(user.id)

@router.post("/workspaces", response_model=WorkspaceConfig)
async def create_workspace(
    name: str,
    configuration: Dict[str, Any],
    user: User = Depends(get_current_user)
):
    """Create new workspace configuration."""
    return await preferences_manager.create_workspace(
        user_id=user.id,
        name=name,
        configuration=configuration
    )

@router.put("/workspaces/{workspace_id}", response_model=WorkspaceConfig)
async def update_workspace(
    workspace_id: str,
    name: str,
    configuration: Dict[str, Any],
    user: User = Depends(get_current_user)
):
    """Update workspace configuration."""
    return await preferences_manager.update_workspace(
        workspace_id=workspace_id,
        user_id=user.id,
        name=name,
        configuration=configuration
    )

@router.delete("/workspaces/{workspace_id}")
async def delete_workspace(
    workspace_id: str,
    user: User = Depends(get_current_user)
):
    """Delete workspace configuration."""
    success = await preferences_manager.delete_workspace(workspace_id, user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return {"success": True}

# WebSocket Endpoints

@router.websocket("/ws/jobs")
async def job_websocket(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time job updates."""
    await ws_manager.connect(websocket, user_id)
    try:
        while True:
            # Send periodic job queue updates
            queue_status = await job_manager.get_queue_status(user_id)
            await websocket.send_json({
                'type': 'queue_update',
                'data': queue_status
            })
            
            # Send job progress updates
            active_jobs = await job_manager.get_active_jobs(user_id)
            for job in active_jobs:
                await websocket.send_json({
                    'type': 'job_progress',
                    'job_id': job.id,
                    'progress': job.progress,
                    'current_step': job.current_step,
                    'total_steps': job.total_steps
                })
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, user_id)

@router.websocket("/ws/performance")
async def performance_websocket(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time performance monitoring."""
    await ws_manager.connect(websocket, user_id)
    try:
        while True:
            # Send performance metrics
            metrics = await performance_monitor.collect_studio_metrics(user_id)
            await websocket.send_json({
                'type': 'performance_metrics',
                'data': metrics
            })
            
            # Check for bottlenecks
            bottlenecks = await performance_monitor.detect_bottlenecks(metrics)
            if bottlenecks:
                await websocket.send_json({
                    'type': 'bottlenecks_detected',
                    'data': bottlenecks
                })
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, user_id)

@router.websocket("/ws/quality")
async def quality_websocket(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time quality monitoring."""
    await ws_manager.connect(websocket, user_id)
    try:
        while True:
            # Send quality updates for active validations
            active_validations = await quality_validator.get_active_validations(user_id)
            for validation in active_validations:
                await websocket.send_json({
                    'type': 'validation_progress',
                    'validation_id': validation.id,
                    'progress': validation.progress,
                    'status': validation.status
                })
            
            await asyncio.sleep(3)  # Update every 3 seconds
            
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, user_id) 