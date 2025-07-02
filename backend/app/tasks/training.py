from celery import Task
from app.celery_app import celery_app
from app.redis_client import TrainingStatusTracker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models import TrainingJob, Character, Dataset
import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TrainingTask(Task):
    """Custom task class for model training."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        job_id = args[0] if args else kwargs.get('job_id')
        asyncio.run(self._update_failed_status(job_id, str(exc)))
    
    async def _update_failed_status(self, job_id: str, error_message: str):
        """Update training job status to failed."""
        AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with AsyncSessionLocal() as session:
            job = await session.get(TrainingJob, job_id)
            if job:
                job.status = "failed"
                job.error_message = error_message
                job.completed_at = datetime.utcnow()
                await session.commit()

@celery_app.task(bind=True, base=TrainingTask, name='app.tasks.training.train_character_model')
def train_character_model(self, job_id: str):
    """Train a character model."""
    
    # Run async function in sync context
    return asyncio.run(_train_model(self, job_id))

async def _train_model(task, job_id: str):
    """Async model training logic."""
    
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as session:
        # Get training job
        job = await session.get(TrainingJob, job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        
        # Get character and dataset
        character = await session.get(Character, job.character_id)
        if not character:
            raise ValueError(f"Character {job.character_id} not found")
        
        dataset = None
        if job.dataset_id:
            dataset = await session.get(Dataset, job.dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {job.dataset_id} not found")
        
        # Update job status
        job.status = "running"
        job.started_at = datetime.utcnow()
        job.celery_task_id = task.request.id
        await session.commit()
        
        # Update Redis status
        await TrainingStatusTracker.update_status(
            job_id,
            "running",
            0.0,
            "Starting model training"
        )
        
        try:
            # Prepare training parameters
            training_params = job.training_params or {}
            
            # Based on job type, run appropriate training
            if job.job_type == "sft":
                result = await _run_sft_training(
                    task, job, character, dataset, training_params
                )
            elif job.job_type == "dpo":
                result = await _run_dpo_training(
                    task, job, character, training_params
                )
            elif job.job_type == "grpo":
                result = await _run_grpo_training(
                    task, job, character, training_params
                )
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            # Update job with results
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            job.adapter_path = result.get("adapter_path")
            job.training_loss = result.get("training_loss")
            job.validation_loss = result.get("validation_loss")
            job.metrics = result.get("metrics", {})
            
            # Update character with trained model
            if job.adapter_path:
                character.is_trained = True
                character.adapter_path = job.adapter_path
                character.model_version = result.get("model_version", "v1")
            
            await session.commit()
            
            # Final status update
            await TrainingStatusTracker.update_status(
                job_id,
                "completed",
                100.0,
                "Training completed successfully",
                metrics=job.metrics
            )
            
            return {
                "job_id": job_id,
                "status": "completed",
                "adapter_path": job.adapter_path,
                "metrics": job.metrics
            }
            
        except Exception as e:
            # Update status to failed
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await session.commit()
            
            await TrainingStatusTracker.update_status(
                job_id,
                "failed",
                job.progress,
                f"Training failed: {str(e)}"
            )
            
            raise

async def _run_sft_training(task, job, character, dataset, params):
    """Run supervised fine-tuning."""
    
    # Simulated training - replace with actual training logic
    for epoch in range(params.get("num_epochs", 3)):
        progress = (epoch + 1) / params.get("num_epochs", 3) * 100
        
        await TrainingStatusTracker.update_status(
            job.id,
            "running",
            progress,
            f"Training epoch {epoch + 1}/{params.get('num_epochs', 3)}",
            metrics={
                "epoch": epoch + 1,
                "loss": 2.5 - (0.5 * epoch)  # Simulated decreasing loss
            }
        )
        
        task.update_state(
            state='PROGRESS',
            meta={
                'current': progress,
                'total': 100,
                'status': f"Epoch {epoch + 1}"
            }
        )
        
        # Simulate training time
        await asyncio.sleep(2)
    
    # Return results
    adapter_path = f"{settings.TRAINING_OUTPUT_DIR}/character_{character.id}/adapter"
    
    return {
        "adapter_path": adapter_path,
        "training_loss": 1.5,
        "validation_loss": 1.8,
        "model_version": "sft-v1",
        "metrics": {
            "epochs": params.get("num_epochs", 3),
            "final_loss": 1.5,
            "perplexity": 4.48
        }
    }

async def _run_dpo_training(task, job, character, params):
    """Run Direct Preference Optimization training."""
    
    # Simulated DPO training
    steps = params.get("num_steps", 1000)
    
    for step in range(0, steps, 100):
        progress = (step / steps) * 100
        
        await TrainingStatusTracker.update_status(
            job.id,
            "running",
            progress,
            f"DPO training step {step}/{steps}",
            metrics={
                "step": step,
                "reward_margin": 0.1 + (step / steps) * 0.5
            }
        )
        
        task.update_state(
            state='PROGRESS',
            meta={
                'current': progress,
                'total': 100,
                'status': f"Step {step}"
            }
        )
        
        await asyncio.sleep(1)
    
    adapter_path = f"{settings.TRAINING_OUTPUT_DIR}/character_{character.id}/dpo_adapter"
    
    return {
        "adapter_path": adapter_path,
        "training_loss": 0.8,
        "validation_loss": 0.9,
        "model_version": "dpo-v1",
        "metrics": {
            "steps": steps,
            "final_reward_margin": 0.6,
            "accuracy": 0.85
        }
    }

async def _run_grpo_training(task, job, character, params):
    """Run Group Relative Policy Optimization training."""
    
    # Simulated GRPO training
    iterations = params.get("num_iterations", 500)
    
    for i in range(0, iterations, 50):
        progress = (i / iterations) * 100
        
        await TrainingStatusTracker.update_status(
            job.id,
            "running",
            progress,
            f"GRPO training iteration {i}/{iterations}",
            metrics={
                "iteration": i,
                "group_reward": 0.2 + (i / iterations) * 0.3
            }
        )
        
        task.update_state(
            state='PROGRESS',
            meta={
                'current': progress,
                'total': 100,
                'status': f"Iteration {i}"
            }
        )
        
        await asyncio.sleep(1)
    
    adapter_path = f"{settings.TRAINING_OUTPUT_DIR}/character_{character.id}/grpo_adapter"
    
    return {
        "adapter_path": adapter_path,
        "training_loss": 0.6,
        "validation_loss": 0.7,
        "model_version": "grpo-v1",
        "metrics": {
            "iterations": iterations,
            "final_group_reward": 0.5,
            "consistency_score": 0.9
        }
    } 