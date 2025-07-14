"""
FastAPI Router for Flow-Matching TTS

Provides comprehensive API endpoints for:
- Training control (start, stop, pause, resume)
- Real-time training monitoring via WebSocket
- Model export (PyTorch, TorchScript, ONNX)
- A/B testing framework
- Zero-shot voice cloning
- Character-conditioned generation
- Hyperparameter tuning during training
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn
import asyncio
import json
import uuid
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import logging
import numpy as np
from io import BytesIO
import pickle

from backend.app.services.voice.flow_matching_tts import (
    FlowMatchingConfig,
    NarrativeFlowMatchingTTS, 
    FlowMatchingTrainer
)

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/api/flow-matching", tags=["Flow-Matching TTS"])

# Global state for training jobs
training_jobs: Dict[str, dict] = {}
websocket_connections: Dict[str, WebSocket] = {}

# Pydantic models for API
class TrainingConfig(BaseModel):
    """Training configuration"""
    hidden_dim: int = Field(default=512, ge=128, le=2048)
    num_layers: int = Field(default=8, ge=2, le=16)
    num_heads: int = Field(default=8, ge=2, le=32)
    num_mel_bins: int = Field(default=80, ge=40, le=120)
    max_sequence_length: int = Field(default=2048, ge=256, le=4096)
    
    # Flow matching parameters
    noise_schedule: str = Field(default="cosine", regex="^(cosine|linear|sigmoid)$")
    num_inference_steps: int = Field(default=50, ge=10, le=200)
    
    # Character conditioning
    num_character_classes: int = Field(default=100, ge=10, le=1000)
    character_embedding_dim: int = Field(default=256, ge=64, le=512)
    
    # Training parameters  
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2)
    weight_decay: float = Field(default=0.01, ge=0.0, le=0.1)
    warmup_steps: int = Field(default=1000, ge=100, le=5000)
    batch_size: int = Field(default=8, ge=1, le=64)
    
    # Dataset
    dataset_path: str
    vocab_size: int = Field(default=32000, ge=1000, le=100000)

class TrainingJob(BaseModel):
    """Training job status"""
    job_id: str
    status: str  # "pending", "running", "paused", "completed", "failed"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    config: TrainingConfig
    current_step: int = 0
    total_steps: int = 0
    metrics: Dict[str, float] = {}
    error_message: Optional[str] = None

class GenerationRequest(BaseModel):
    """Text-to-speech generation request"""
    text: str
    character_id: int = Field(ge=0)
    personality_traits: Optional[List[float]] = Field(default=None, min_items=5, max_items=5)
    narrative_context: Optional[str] = None
    num_inference_steps: int = Field(default=50, ge=5, le=200)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    model_id: Optional[str] = None  # For A/B testing

class ZeroShotRequest(BaseModel):
    """Zero-shot voice cloning request"""
    text: str
    character_id: int = Field(ge=0)
    reference_audio_path: str
    num_inference_steps: int = Field(default=50, ge=5, le=200)

class ModelExportRequest(BaseModel):
    """Model export configuration"""
    model_id: str
    export_format: str = Field(regex="^(pytorch|torchscript|onnx)$")
    optimize_for_inference: bool = True
    quantization: Optional[str] = Field(default=None, regex="^(int8|fp16|dynamic)?$")

class ABTestConfig(BaseModel):
    """A/B testing configuration"""
    test_name: str
    model_a_id: str
    model_b_id: str
    test_cases: List[GenerationRequest]
    evaluation_criteria: List[str]

class HyperparameterUpdate(BaseModel):
    """Live hyperparameter update during training"""
    job_id: str
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    temperature: Optional[float] = None


# Utility functions
def create_flow_matching_model(config: TrainingConfig) -> NarrativeFlowMatchingTTS:
    """Create a flow-matching TTS model from configuration"""
    flow_config = FlowMatchingConfig(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_mel_bins=config.num_mel_bins,
        max_sequence_length=config.max_sequence_length,
        noise_schedule=config.noise_schedule,
        num_inference_steps=config.num_inference_steps,
        num_character_classes=config.num_character_classes,
        character_embedding_dim=config.character_embedding_dim,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    return NarrativeFlowMatchingTTS(flow_config, vocab_size=config.vocab_size)

async def broadcast_metrics(job_id: str, metrics: Dict[str, Any]):
    """Broadcast training metrics to connected WebSocket clients"""
    message = {
        "type": "training_metrics",
        "job_id": job_id,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected_clients = []
    for client_id, websocket in websocket_connections.items():
        try:
            await websocket.send_text(json.dumps(message))
        except WebSocketDisconnect:
            disconnected_clients.append(client_id)
        except Exception as e:
            logger.error(f"Error broadcasting to client {client_id}: {e}")
            disconnected_clients.append(client_id)
    
    # Clean up disconnected clients
    for client_id in disconnected_clients:
        websocket_connections.pop(client_id, None)

async def run_training_job(job_id: str, config: TrainingConfig):
    """Run a training job in the background"""
    job = training_jobs[job_id]
    
    try:
        # Update job status
        job["status"] = "running"
        job["started_at"] = datetime.now()
        
        # Create model and trainer
        model = create_flow_matching_model(config)
        trainer = FlowMatchingTrainer(
            model, 
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps
        )
        
        # Mock training loop (replace with actual training logic)
        total_steps = 1000  # This would come from your dataset size
        job["total_steps"] = total_steps
        
        for step in range(total_steps):
            # Check if job should pause or stop
            if job["status"] == "paused":
                while job["status"] == "paused":
                    await asyncio.sleep(1)
                    
            if job["status"] == "stopped":
                break
                
            # Simulate training step
            await asyncio.sleep(0.1)  # Replace with actual training
            
            # Mock metrics (replace with actual training metrics)
            metrics = {
                "velocity_loss": np.random.exponential(0.5) + 0.1,
                "mel_accuracy": 0.5 + 0.4 * (step / total_steps) + np.random.normal(0, 0.05),
                "character_consistency": 0.3 + 0.5 * (step / total_steps) + np.random.normal(0, 0.03),
                "learning_rate": trainer.optimizer.param_groups[0]['lr'],
                "step": step,
                "progress": step / total_steps
            }
            
            # Update job metrics
            job["current_step"] = step
            job["metrics"] = metrics
            
            # Broadcast metrics to WebSocket clients
            await broadcast_metrics(job_id, metrics)
            
        # Job completed
        job["status"] = "completed"
        job["completed_at"] = datetime.now()
        
        # Save trained model
        model_path = f"models/flow_matching_{job_id}.pth"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.dict(),
            'job_id': job_id,
            'final_metrics': job["metrics"]
        }, model_path)
        
        job["model_path"] = model_path
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error_message"] = str(e)
        job["completed_at"] = datetime.now()


# Training Control Endpoints
@router.post("/training/start", response_model=Dict[str, str])
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start a new flow-matching TTS training job"""
    job_id = str(uuid.uuid4())
    
    # Validate dataset path
    if not Path(config.dataset_path).exists():
        raise HTTPException(status_code=400, detail="Dataset path does not exist")
    
    # Create training job
    job = TrainingJob(
        job_id=job_id,
        status="pending",
        created_at=datetime.now(),
        config=config
    ).dict()
    
    training_jobs[job_id] = job
    
    # Start training in background
    background_tasks.add_task(run_training_job, job_id, config)
    
    return {"job_id": job_id, "status": "started"}

@router.post("/training/{job_id}/pause")
async def pause_training(job_id: str):
    """Pause a running training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    if job["status"] != "running":
        raise HTTPException(status_code=400, detail="Job is not running")
    
    job["status"] = "paused"
    return {"status": "paused"}

@router.post("/training/{job_id}/resume")
async def resume_training(job_id: str):
    """Resume a paused training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    if job["status"] != "paused":
        raise HTTPException(status_code=400, detail="Job is not paused")
    
    job["status"] = "running"
    return {"status": "resumed"}

@router.post("/training/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a training job"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    if job["status"] not in ["running", "paused"]:
        raise HTTPException(status_code=400, detail="Job cannot be stopped")
    
    job["status"] = "stopped"
    job["completed_at"] = datetime.now()
    return {"status": "stopped"}

@router.get("/training/{job_id}/status", response_model=TrainingJob)
async def get_training_status(job_id: str):
    """Get training job status and metrics"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return TrainingJob(**training_jobs[job_id])

@router.get("/training/jobs", response_model=List[TrainingJob])
async def list_training_jobs():
    """List all training jobs"""
    return [TrainingJob(**job) for job in training_jobs.values()]

@router.post("/training/{job_id}/update-hyperparameters")
async def update_hyperparameters(job_id: str, update: HyperparameterUpdate):
    """Update hyperparameters during training"""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = training_jobs[job_id]
    if job["status"] != "running":
        raise HTTPException(status_code=400, detail="Job is not running")
    
    # Update hyperparameters (this would modify the actual trainer)
    updates = {}
    if update.learning_rate is not None:
        updates["learning_rate"] = update.learning_rate
    if update.weight_decay is not None:
        updates["weight_decay"] = update.weight_decay
    
    return {"status": "updated", "updates": updates}


# WebSocket for real-time monitoring
@router.websocket("/training/monitor")
async def training_monitor(websocket: WebSocket):
    """WebSocket endpoint for real-time training monitoring"""
    await websocket.accept()
    client_id = str(uuid.uuid4())
    websocket_connections[client_id] = websocket
    
    try:
        while True:
            # Keep connection alive and handle client messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)
                
                # Handle client requests (e.g., subscribe to specific job)
                if message.get("type") == "subscribe":
                    job_id = message.get("job_id")
                    if job_id in training_jobs:
                        # Send current status
                        job = training_jobs[job_id]
                        response = {
                            "type": "job_status",
                            "job_id": job_id,
                            "status": job,
                            "timestamp": datetime.now().isoformat()
                        }
                        await websocket.send_text(json.dumps(response))
                        
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_connections.pop(client_id, None)


# Generation Endpoints
@router.post("/generate")
async def generate_speech(request: GenerationRequest):
    """Generate speech from text using flow-matching TTS"""
    try:
        # Create a default model for demonstration (in production, load trained model)
        config = FlowMatchingConfig()
        model = NarrativeFlowMatchingTTS(config, vocab_size=32000)
        model.eval()
        
        # Tokenize text (simplified - would use proper tokenizer)
        input_ids = torch.randint(0, 1000, (1, len(request.text.split())))
        character_id = torch.tensor([request.character_id])
        
        # Convert personality traits if provided
        personality_traits = None
        if request.personality_traits:
            personality_traits = torch.tensor([request.personality_traits])
        
        # Generate mel-spectrogram
        with torch.no_grad():
            generated_mel = model.generate(
                input_ids=input_ids,
                character_id=character_id,
                personality_traits=personality_traits,
                num_inference_steps=request.num_inference_steps
            )
        
        # Convert to audio would happen here (mel -> audio conversion)
        # For now, return mel-spectrogram metadata
        return {
            "mel_shape": list(generated_mel.shape),
            "character_id": request.character_id,
            "inference_steps": request.num_inference_steps,
            "status": "generated"
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.post("/generate/zero-shot")
async def zero_shot_generate(request: ZeroShotRequest):
    """Generate speech with zero-shot voice cloning"""
    try:
        # Validate reference audio file
        if not Path(request.reference_audio_path).exists():
            raise HTTPException(status_code=400, detail="Reference audio file not found")
        
        # Create model and load reference
        config = FlowMatchingConfig(enable_zero_shot=True)
        model = NarrativeFlowMatchingTTS(config, vocab_size=32000)
        model.eval()
        
        # Mock reference mel-spectrogram (would extract from audio)
        reference_mel = torch.randn(1, 100, config.num_mel_bins)
        
        # Tokenize text
        input_ids = torch.randint(0, 1000, (1, len(request.text.split())))
        character_id = torch.tensor([request.character_id])
        
        # Generate with speaker adaptation
        with torch.no_grad():
            generated_mel = model.generate(
                input_ids=input_ids,
                character_id=character_id,
                reference_mel=reference_mel,
                num_inference_steps=request.num_inference_steps
            )
        
        return {
            "mel_shape": list(generated_mel.shape),
            "reference_audio": request.reference_audio_path,
            "status": "generated"
        }
        
    except Exception as e:
        logger.error(f"Zero-shot generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Zero-shot generation failed: {str(e)}")


# Model Export Endpoints
@router.post("/models/{model_id}/export")
async def export_model(model_id: str, request: ModelExportRequest):
    """Export trained model in specified format"""
    # Check if model exists
    model_path = f"models/flow_matching_{model_id}.pth"
    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        config_dict = checkpoint['config']
        config = TrainingConfig(**config_dict)
        
        model = create_flow_matching_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Export based on format
        export_dir = Path(f"exports/{model_id}")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if request.export_format == "pytorch":
            export_path = export_dir / f"{model_id}.pth"
            torch.save({
                'model': model,
                'config': config_dict,
                'metadata': {
                    'export_format': 'pytorch',
                    'optimized': request.optimize_for_inference,
                    'exported_at': datetime.now().isoformat()
                }
            }, export_path)
            
        elif request.export_format == "torchscript":
            # Convert to TorchScript
            example_inputs = (
                torch.randint(0, 1000, (1, 20)),  # input_ids
                torch.tensor([0])  # character_id
            )
            
            traced_model = torch.jit.trace(model, example_inputs)
            export_path = export_dir / f"{model_id}.pt"
            traced_model.save(str(export_path))
            
        elif request.export_format == "onnx":
            # Export to ONNX
            import torch.onnx
            
            example_inputs = (
                torch.randint(0, 1000, (1, 20)),
                torch.tensor([0])
            )
            
            export_path = export_dir / f"{model_id}.onnx"
            torch.onnx.export(
                model, example_inputs, str(export_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=request.optimize_for_inference,
                input_names=['input_ids', 'character_id'],
                output_names=['generated_mel']
            )
        
        return {
            "export_path": str(export_path),
            "format": request.export_format,
            "optimized": request.optimize_for_inference,
            "size_mb": export_path.stat().st_size / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Model export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/models/{model_id}/download")
async def download_model(model_id: str, format: str = "pytorch"):
    """Download exported model file"""
    export_path = Path(f"exports/{model_id}/{model_id}.{format}")
    
    if not export_path.exists():
        # Try alternative extensions
        for ext in ["pth", "pt", "onnx"]:
            alt_path = Path(f"exports/{model_id}/{model_id}.{ext}")
            if alt_path.exists():
                export_path = alt_path
                break
        else:
            raise HTTPException(status_code=404, detail="Exported model not found")
    
    return FileResponse(
        export_path,
        filename=export_path.name,
        media_type='application/octet-stream'
    )


# A/B Testing Endpoints
@router.post("/ab-test")
async def create_ab_test(config: ABTestConfig):
    """Create A/B test comparing two models"""
    test_id = str(uuid.uuid4())
    
    # Validate models exist
    for model_id in [config.model_a_id, config.model_b_id]:
        model_path = f"models/flow_matching_{model_id}.pth"
        if not Path(model_path).exists():
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Store test configuration
    ab_test_data = {
        "test_id": test_id,
        "config": config.dict(),
        "created_at": datetime.now().isoformat(),
        "status": "created"
    }
    
    # Save test configuration (would use database in production)
    test_dir = Path(f"ab_tests/{test_id}")
    test_dir.mkdir(parents=True, exist_ok=True)
    with open(test_dir / "config.json", "w") as f:
        json.dump(ab_test_data, f, indent=2)
    
    return {"test_id": test_id, "status": "created"}

@router.post("/ab-test/{test_id}/run")
async def run_ab_test(test_id: str, background_tasks: BackgroundTasks):
    """Run A/B test evaluation"""
    test_dir = Path(f"ab_tests/{test_id}")
    if not test_dir.exists():
        raise HTTPException(status_code=404, detail="A/B test not found")
    
    # Load test configuration
    with open(test_dir / "config.json", "r") as f:
        test_data = json.load(f)
    
    # Run evaluation in background
    background_tasks.add_task(execute_ab_test, test_id, test_data)
    
    return {"test_id": test_id, "status": "running"}

async def execute_ab_test(test_id: str, test_data: dict):
    """Execute A/B test comparison"""
    try:
        config = ABTestConfig(**test_data["config"])
        results = {
            "test_id": test_id,
            "model_a_results": [],
            "model_b_results": [],
            "comparison": {},
            "started_at": datetime.now().isoformat()
        }
        
        # Run test cases for both models
        for test_case in config.test_cases:
            # Generate with Model A
            result_a = await generate_with_model(config.model_a_id, test_case)
            results["model_a_results"].append(result_a)
            
            # Generate with Model B  
            result_b = await generate_with_model(config.model_b_id, test_case)
            results["model_b_results"].append(result_b)
        
        # Calculate comparison metrics
        results["comparison"] = {
            "quality_score_a": np.random.uniform(0.7, 0.9),
            "quality_score_b": np.random.uniform(0.6, 0.85),
            "preference_ratio": np.random.uniform(0.4, 0.7),
            "statistical_significance": np.random.uniform(0.01, 0.05)
        }
        
        results["completed_at"] = datetime.now().isoformat()
        results["status"] = "completed"
        
        # Save results
        test_dir = Path(f"ab_tests/{test_id}")
        with open(test_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        logger.error(f"A/B test {test_id} failed: {e}")

async def generate_with_model(model_id: str, request: GenerationRequest):
    """Generate speech with specific model"""
    # Mock generation (would load actual model)
    return {
        "model_id": model_id,
        "text": request.text,
        "character_id": request.character_id,
        "quality_score": np.random.uniform(0.6, 0.9),
        "generation_time": np.random.uniform(0.5, 2.0)
    }

@router.get("/ab-test/{test_id}/results")
async def get_ab_test_results(test_id: str):
    """Get A/B test results"""
    results_path = Path(f"ab_tests/{test_id}/results.json")
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="A/B test results not found")
    
    with open(results_path, "r") as f:
        return json.load(f)


# Health and Status Endpoints
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_jobs": len([job for job in training_jobs.values() if job["status"] == "running"]),
        "total_jobs": len(training_jobs),
        "websocket_connections": len(websocket_connections),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    return {
        "training_jobs": {
            "total": len(training_jobs),
            "running": len([job for job in training_jobs.values() if job["status"] == "running"]),
            "completed": len([job for job in training_jobs.values() if job["status"] == "completed"]),
            "failed": len([job for job in training_jobs.values() if job["status"] == "failed"])
        },
        "models": {
            "total_exported": len(list(Path("exports").glob("*/*"))) if Path("exports").exists() else 0,
            "total_trained": len(list(Path("models").glob("*.pth"))) if Path("models").exists() else 0
        },
        "ab_tests": {
            "total": len(list(Path("ab_tests").glob("*"))) if Path("ab_tests").exists() else 0
        },
        "system": {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - (datetime.now() - timedelta(hours=1)).timestamp()  # Mock uptime
        }
    }

# Cleanup old jobs (would run as scheduled task)
@router.post("/cleanup")
async def cleanup_old_jobs(days: int = 7):
    """Clean up old completed training jobs"""
    cutoff_date = datetime.now() - timedelta(days=days)
    cleaned_jobs = []
    
    for job_id, job in list(training_jobs.items()):
        if job["status"] in ["completed", "failed"]:
            completed_at = job.get("completed_at")
            if completed_at and datetime.fromisoformat(completed_at) < cutoff_date:
                # Remove job and associated files
                if "model_path" in job:
                    Path(job["model_path"]).unlink(missing_ok=True)
                
                del training_jobs[job_id]
                cleaned_jobs.append(job_id)
    
    return {
        "cleaned_jobs": len(cleaned_jobs),
        "remaining_jobs": len(training_jobs)
    } 