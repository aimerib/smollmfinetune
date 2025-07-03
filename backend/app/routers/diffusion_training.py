from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import asyncio
import json
import os
from datetime import datetime
import uuid
import logging

from narrative_engine.diffusion_config import (
    DiffusionMultimodalConfig, 
    get_small_config, 
    get_medium_config, 
    get_large_config
)
from narrative_engine.diffusion_model import DiffusionMultimodalModel
from narrative_engine.diffusion_trainer import DiffusionTrainingManager
from narrative_engine.synthetic_multimodal_dataset import SyntheticMultimodalDatasetGenerator
from app.utils.character.manager import CharacterManager
from app.utils.world.manager import WorldManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/diffusion", tags=["diffusion"])

# Global training manager instance
training_manager: Optional[DiffusionTrainingManager] = None
training_status = {"status": "idle"}
training_metrics = {}

# Request/Response Models
class DatasetGenerationRequest(BaseModel):
    num_characters: int = Field(..., ge=1, le=1000)
    conversations_per_character: int = Field(..., ge=1, le=1000)
    multimodal_ratio: float = Field(..., ge=0.0, le=1.0)
    world_name: Optional[str] = "default"
    
class TrainingConfig(BaseModel):
    model_size: Literal['small', 'medium', 'large']
    num_characters: int = Field(..., ge=1, le=1000)
    conversations_per_character: int = Field(..., ge=1, le=1000)
    multimodal_ratio: float = Field(..., ge=0.0, le=1.0)
    batch_size: int = Field(..., ge=1, le=64)
    learning_rate: float = Field(..., ge=1e-6, le=1e-2)
    guidance_scale: float = Field(..., ge=1.0, le=20.0)
    num_inference_steps: int = Field(..., ge=10, le=1000)
    max_steps: int = Field(..., ge=100, le=1000000)
    save_steps: int = Field(..., ge=10, le=10000)
    eval_steps: int = Field(..., ge=10, le=10000)
    use_ema: bool = True
    cross_modal_weight: float = Field(..., ge=0.0, le=1.0)

class ValidationCheck(BaseModel):
    name: str
    description: str
    passed: bool
    details: Optional[str] = None

class ValidationResults(BaseModel):
    success: bool
    checks: List[ValidationCheck]
    issues: List[str]
    recommendations: List[str]
    model_parameters: int
    memory_requirements: str

class TrainingStatus(BaseModel):
    status: Literal['idle', 'training', 'paused', 'completed', 'error']
    phase: Optional[str] = None
    current_step: Optional[int] = None
    max_steps: Optional[int] = None
    start_time: Optional[str] = None
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None

class TrainingMetrics(BaseModel):
    current_step: int
    total_loss: float
    text_loss: float
    speech_loss: float
    control_loss: float
    memory_loss: float
    alignment_loss: float
    learning_rate: float
    gradient_norm: float
    samples_per_second: float
    gpu_memory_usage: float
    timestamp: str

class ExportRequest(BaseModel):
    checkpoint_id: str
    format: Literal['onnx', 'tensorrt', 'cartridge']

# Dataset Generation Endpoints
@router.post("/dataset/generate")
async def generate_dataset(
    request: DatasetGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate synthetic multimodal dataset for diffusion training"""
    try:
        # Get available characters and worlds
        character_manager = CharacterManager()
        world_manager = WorldManager()
        
        # Load world context if specified
        world_context = None
        if request.world_name and request.world_name != "default":
            try:
                world_context = world_manager.load_world(request.world_name)
            except Exception as e:
                logger.warning(f"Could not load world '{request.world_name}': {e}")
        
        # Get available characters
        characters = character_manager.list_characters()
        if len(characters) < request.num_characters:
            raise HTTPException(
                status_code=400,
                detail=f"Only {len(characters)} characters available, need {request.num_characters}"
            )
        
        # Select characters for dataset
        selected_characters = characters[:request.num_characters]
        
        # Initialize dataset generator
        dataset_generator = SyntheticMultimodalDatasetGenerator()
        
        # Generate dataset in background
        dataset_id = str(uuid.uuid4())
        background_tasks.add_task(
            _generate_dataset_background,
            dataset_generator,
            selected_characters,
            request,
            world_context,
            dataset_id
        )
        
        return {
            "dataset_id": dataset_id,
            "status": "generating",
            "estimated_samples": request.num_characters * request.conversations_per_character,
            "estimated_time_hours": (request.num_characters * request.conversations_per_character) / 100
        }
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _generate_dataset_background(
    generator: SyntheticMultimodalDatasetGenerator,
    characters: List[Dict],
    request: DatasetGenerationRequest,
    world_context: Optional[Dict],
    dataset_id: str
):
    """Background task for dataset generation"""
    try:
        # Create output directory
        output_dir = f"datasets/diffusion_multimodal_{dataset_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate dataset
        dataset = await generator.generate_dataset(
            characters=characters,
            num_conversations_per_character=request.conversations_per_character,
            multimodal_ratio=request.multimodal_ratio,
            world_context=world_context,
            output_dir=output_dir
        )
        
        # Save metadata
        metadata = {
            "dataset_id": dataset_id,
            "num_characters": request.num_characters,
            "conversations_per_character": request.conversations_per_character,
            "multimodal_ratio": request.multimodal_ratio,
            "total_samples": len(dataset),
            "generated_at": datetime.now().isoformat(),
            "output_dir": output_dir
        }
        
        with open(f"{output_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Dataset {dataset_id} generated successfully with {len(dataset)} samples")
        
    except Exception as e:
        logger.error(f"Background dataset generation failed: {e}")

# Model Validation
@router.post("/model/validate", response_model=ValidationResults)
async def validate_model():
    """Validate diffusion model architecture and configuration"""
    try:
        # Start with small config for validation
        config = get_small_config()
        checks = []
        issues = []
        recommendations = []
        
        # Check 1: Configuration validation
        try:
            config.validate()
            checks.append(ValidationCheck(
                name="config_validation",
                description="Configuration parameters are valid",
                passed=True
            ))
        except Exception as e:
            checks.append(ValidationCheck(
                name="config_validation",
                description="Configuration validation failed",
                passed=False,
                details=str(e)
            ))
            issues.append(f"Configuration error: {e}")
        
        # Check 2: Model creation
        try:
            model = DiffusionMultimodalModel(config)
            model_params = sum(p.numel() for p in model.parameters())
            checks.append(ValidationCheck(
                name="model_creation",
                description="Model architecture created successfully",
                passed=True,
                details=f"Model has {model_params:,} parameters"
            ))
        except Exception as e:
            checks.append(ValidationCheck(
                name="model_creation",
                description="Model creation failed",
                passed=False,
                details=str(e)
            ))
            issues.append(f"Model creation error: {e}")
            model_params = 0
        
        # Check 3: Forward pass test
        try:
            if 'model' in locals():
                import torch
                batch_size = 2
                sample_data = {
                    'text_embeddings': torch.randn(batch_size, config.modalities.text_max_sequence_length, config.modalities.text_embedding_dim),
                    'speech_mel': torch.randn(batch_size, config.modalities.speech_mel_channels, config.modalities.speech_max_time_steps),
                    'control_tokens': torch.randn(batch_size, config.modalities.control_max_tokens, config.modalities.control_embedding_dim),
                    'memory_vectors': torch.randn(batch_size, config.modalities.memory_max_entries, config.modalities.memory_vector_dim),
                    'character_id': torch.randint(0, 10, (batch_size,))
                }
                
                with torch.no_grad():
                    _ = model(sample_data, torch.randint(0, config.scheduler.num_train_timesteps, (batch_size,)))
                
                checks.append(ValidationCheck(
                    name="forward_pass",
                    description="Forward pass completed successfully",
                    passed=True
                ))
            else:
                raise Exception("Model not available for testing")
                
        except Exception as e:
            checks.append(ValidationCheck(
                name="forward_pass",
                description="Forward pass failed",
                passed=False,
                details=str(e)
            ))
            issues.append(f"Forward pass error: {e}")
        
        # Check 4: Memory requirements
        memory_gb = model_params * 4 / (1024**3)  # Rough estimate (float32)
        if memory_gb > 24:
            recommendations.append("Consider using gradient checkpointing or model parallelism for large models")
        if memory_gb > 80:
            issues.append("Model may not fit on single GPU")
        
        memory_requirements = f"~{memory_gb:.1f}GB GPU memory"
        
        # Check 5: Dataset availability
        dataset_dirs = [d for d in os.listdir("datasets") if "diffusion_multimodal" in d] if os.path.exists("datasets") else []
        if dataset_dirs:
            checks.append(ValidationCheck(
                name="dataset_availability",
                description=f"Found {len(dataset_dirs)} diffusion datasets",
                passed=True
            ))
        else:
            checks.append(ValidationCheck(
                name="dataset_availability",
                description="No diffusion datasets found",
                passed=False
            ))
            recommendations.append("Generate a multimodal dataset before training")
        
        success = all(check.passed for check in checks if check.name != "dataset_availability")
        
        return ValidationResults(
            success=success,
            checks=checks,
            issues=issues,
            recommendations=recommendations,
            model_parameters=model_params,
            memory_requirements=memory_requirements
        )
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Training Control Endpoints
@router.post("/training/start")
async def start_training(
    config_request: TrainingConfig,
    background_tasks: BackgroundTasks
):
    """Start diffusion model training"""
    global training_manager, training_status
    
    try:
        if training_status["status"] == "training":
            raise HTTPException(status_code=400, detail="Training already in progress")
        
        # Get model configuration
        if config_request.model_size == "small":
            model_config = get_small_config()
        elif config_request.model_size == "medium":
            model_config = get_medium_config()
        else:
            model_config = get_large_config()
        
        # Update config with request parameters
        model_config.training.learning_rate = config_request.learning_rate
        model_config.training.train_batch_size = config_request.batch_size
        model_config.training.max_train_steps = config_request.max_steps
        model_config.training.save_steps = config_request.save_steps
        model_config.training.eval_steps = config_request.eval_steps
        model_config.training.use_ema = config_request.use_ema
        model_config.loss.cross_modal_weight = config_request.cross_modal_weight
        model_config.scheduler.guidance_scale = config_request.guidance_scale
        model_config.scheduler.num_inference_steps = config_request.num_inference_steps
        
        # Find latest dataset
        dataset_dirs = [d for d in os.listdir("datasets") if "diffusion_multimodal" in d] if os.path.exists("datasets") else []
        if not dataset_dirs:
            raise HTTPException(status_code=400, detail="No datasets found. Generate a dataset first.")
        
        latest_dataset = f"datasets/{sorted(dataset_dirs)[-1]}"
        
        # Initialize training manager
        training_manager = DiffusionTrainingManager(
            config=model_config,
            dataset_path=latest_dataset,
            output_dir=f"training_output/diffusion_{config_request.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Start training in background
        training_status = {
            "status": "training",
            "phase": config_request.model_size,
            "current_step": 0,
            "max_steps": config_request.max_steps,
            "start_time": datetime.now().isoformat()
        }
        
        background_tasks.add_task(_run_training_background)
        
        return {
            "message": "Training started",
            "phase": config_request.model_size,
            "max_steps": config_request.max_steps,
            "dataset": latest_dataset
        }
        
    except Exception as e:
        logger.error(f"Training start failed: {e}")
        training_status = {"status": "error", "error_message": str(e)}
        raise HTTPException(status_code=500, detail=str(e))

async def _run_training_background():
    """Background task for training execution"""
    global training_manager, training_status, training_metrics
    
    try:
        if training_manager is None:
            raise Exception("Training manager not initialized")
        
        # Run training
        await training_manager.train()
        
        training_status["status"] = "completed"
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_status = {"status": "error", "error_message": str(e)}

@router.post("/training/pause")
async def pause_training():
    """Pause ongoing training"""
    global training_manager, training_status
    
    try:
        if training_status["status"] != "training":
            raise HTTPException(status_code=400, detail="No training in progress")
        
        if training_manager:
            training_manager.pause()
        
        training_status["status"] = "paused"
        return {"message": "Training paused"}
        
    except Exception as e:
        logger.error(f"Training pause failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/resume")
async def resume_training():
    """Resume paused training"""
    global training_manager, training_status
    
    try:
        if training_status["status"] != "paused":
            raise HTTPException(status_code=400, detail="No paused training to resume")
        
        if training_manager:
            training_manager.resume()
        
        training_status["status"] = "training"
        return {"message": "Training resumed"}
        
    except Exception as e:
        logger.error(f"Training resume failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training/stop")
async def stop_training():
    """Stop ongoing training"""
    global training_manager, training_status
    
    try:
        if training_status["status"] not in ["training", "paused"]:
            raise HTTPException(status_code=400, detail="No training to stop")
        
        if training_manager:
            training_manager.stop()
        
        training_status = {"status": "idle"}
        training_manager = None
        
        return {"message": "Training stopped"}
        
    except Exception as e:
        logger.error(f"Training stop failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    return TrainingStatus(**training_status)

@router.get("/training/metrics", response_model=TrainingMetrics)
async def get_training_metrics():
    """Get current training metrics"""
    global training_manager, training_metrics
    
    if training_manager and training_status["status"] == "training":
        # Get latest metrics from training manager
        latest_metrics = training_manager.get_metrics()
        if latest_metrics:
            training_metrics = latest_metrics
    
    if not training_metrics:
        # Return dummy metrics if no training active
        training_metrics = {
            "current_step": 0,
            "total_loss": 0.0,
            "text_loss": 0.0,
            "speech_loss": 0.0,
            "control_loss": 0.0,
            "memory_loss": 0.0,
            "alignment_loss": 0.0,
            "learning_rate": 0.0,
            "gradient_norm": 0.0,
            "samples_per_second": 0.0,
            "gpu_memory_usage": 0.0,
            "timestamp": datetime.now().isoformat()
        }
    
    return TrainingMetrics(**training_metrics)

@router.get("/training/history")
async def get_training_history():
    """Get training run history"""
    try:
        history = []
        training_output_dir = "training_output"
        
        if os.path.exists(training_output_dir):
            for run_dir in sorted(os.listdir(training_output_dir)):
                if "diffusion_" in run_dir:
                    run_path = os.path.join(training_output_dir, run_dir)
                    metadata_path = os.path.join(run_path, "training_metadata.json")
                    
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        history.append(metadata)
        
        return {"runs": history}
        
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/checkpoints")
async def get_model_checkpoints():
    """Get available model checkpoints"""
    try:
        checkpoints = []
        training_output_dir = "training_output"
        
        if os.path.exists(training_output_dir):
            for run_dir in os.listdir(training_output_dir):
                if "diffusion_" in run_dir:
                    run_path = os.path.join(training_output_dir, run_dir)
                    
                    # Look for checkpoint files
                    for file in os.listdir(run_path):
                        if file.endswith(".pt") or file.endswith(".pth"):
                            checkpoint_path = os.path.join(run_path, file)
                            checkpoints.append({
                                "id": f"{run_dir}/{file}",
                                "path": checkpoint_path,
                                "size_mb": os.path.getsize(checkpoint_path) / (1024*1024),
                                "created_at": datetime.fromtimestamp(
                                    os.path.getctime(checkpoint_path)
                                ).isoformat()
                            })
        
        return {"checkpoints": sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)}
        
    except Exception as e:
        logger.error(f"Failed to get checkpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/export")
async def export_model(request: ExportRequest):
    """Export model checkpoint to specified format"""
    try:
        checkpoint_path = f"training_output/{request.checkpoint_id}"
        
        if not os.path.exists(checkpoint_path):
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        
        export_id = str(uuid.uuid4())
        output_dir = f"exports/{request.format}_{export_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        if request.format == "cartridge":
            # Export as runtime cartridge
            cartridge_path = await _export_cartridge(checkpoint_path, output_dir)
            return {
                "export_id": export_id,
                "format": request.format,
                "output_path": cartridge_path,
                "message": "Model exported as runtime cartridge"
            }
        else:
            # TODO: Implement ONNX/TensorRT export
            raise HTTPException(status_code=501, detail=f"{request.format} export not implemented yet")
        
    except Exception as e:
        logger.error(f"Model export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _export_cartridge(checkpoint_path: str, output_dir: str) -> str:
    """Export model as runtime cartridge"""
    # This would integrate with your existing cartridge export system
    cartridge_path = os.path.join(output_dir, "character_cartridge.zip")
    
    # For now, create a placeholder
    import zipfile
    with zipfile.ZipFile(cartridge_path, 'w') as zf:
        zf.writestr("model_checkpoint.pt", "# Diffusion model checkpoint placeholder")
        zf.writestr("config.json", json.dumps({"type": "diffusion_multimodal", "version": "1.0"}))
        zf.writestr("README.md", "# Diffusion Multimodal Character Cartridge")
    
    return cartridge_path 