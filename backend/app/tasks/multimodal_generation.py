"""Multimodal dataset generation Celery tasks"""

from celery import Task
from app.celery_app import celery_app
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models import MultimodalDataset
import asyncio
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path to import from main app
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import narrative engine
from .narrative_engine.synthetic_multimodal_dataset import (
    MultimodalDatasetGenerator, 
    SyntheticGenerationConfig
)

logger = logging.getLogger(__name__)

# Import WebSocket manager from router
from app.routers.multimodal import multimodal_manager


class MultimodalGenerationTask(Task):
    """Custom task class for multimodal dataset generation."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        job_id = args[0] if args else kwargs.get('job_id')
        asyncio.run(self._update_failed_status(job_id, str(exc)))
    
    async def _update_failed_status(self, job_id: str, error_message: str):
        """Update multimodal dataset status to failed."""
        AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with AsyncSessionLocal() as session:
            job = await session.get(MultimodalDataset, job_id)
            if job:
                job.status = "failed"
                job.error_message = error_message
                job.current_step = "Failed"
                await session.commit()
                
                # Broadcast failure to WebSocket connections
                await multimodal_manager.broadcast_progress(job_id, {
                    "status": "failed",
                    "error": error_message,
                    "currentStep": "Generation failed"
                })


@celery_app.task(bind=True, base=MultimodalGenerationTask, name='multimodal_generation.generate_dataset')
def generate_multimodal_dataset(self, job_id: str, config: dict):
    """Generate multimodal dataset using narrative engine."""
    
    return asyncio.run(_generate_multimodal_dataset(self, job_id, config))


async def _generate_multimodal_dataset(task, job_id: str, config: dict):
    """Async multimodal dataset generation logic."""
    
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as session:
        # Get multimodal dataset job
        job = await session.get(MultimodalDataset, job_id)
        if not job:
            raise ValueError(f"Multimodal dataset job {job_id} not found")
        
        # Update status to generating
        job.status = "generating"
        job.current_step = "Initializing generation pipeline..."
        await session.commit()
        
        # Broadcast initial status
        await multimodal_manager.broadcast_progress(job_id, {
            "status": "generating",
            "progress": 0.0,
            "currentStep": "Initializing generation pipeline...",
            "samplesGenerated": 0
        })
        
        try:
            # Create narrative engine configuration
            output_dir = Path(f"multimodal_datasets/{job_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            generation_config = SyntheticGenerationConfig(
                num_samples=config["sampleCount"],
                num_characters=config["characterCount"],
                narrative_types=config["narrativeTypes"],
                tts_model="kokoro" if config["useMockTTS"] else config.get("ttsProvider", "orpheus"),
                output_dir=output_dir,
                save_audio=True,  # Always save audio for validation
                save_mel_images=False  # Skip images for performance
            )
            
            # Create generator
            generator = MultimodalDatasetGenerator(generation_config)
            
            # Progress callback for real-time updates
            async def progress_callback(progress: float, message: str, samples_done: int):
                """Update progress in database and broadcast to WebSocket connections"""
                
                # Update database
                job.progress = progress
                job.current_step = message
                job.samples_generated = samples_done
                await session.commit()
                
                # Update Celery task state
                task.update_state(
                    state='PROGRESS',
                    meta={
                        'progress': progress, 
                        'message': message, 
                        'samples': samples_done,
                        'total': config["sampleCount"]
                    }
                )
                
                # Broadcast to WebSocket connections
                await multimodal_manager.broadcast_progress(job_id, {
                    "status": "generating",
                    "progress": progress,
                    "currentStep": message,
                    "samplesGenerated": samples_done,
                    "totalSamples": config["sampleCount"]
                })
                
                logger.info(f"Job {job_id}: {progress:.1f}% - {message} ({samples_done}/{config['sampleCount']})")
            
            # Enhanced dataset generation with progress tracking
            samples = await _generate_dataset_with_progress(
                generator, 
                generation_config, 
                progress_callback
            )
            
            # Update job with completion results
            job.status = "completed"
            job.progress = 100.0
            job.current_step = "Complete"
            job.samples_generated = len(samples)
            job.output_path = str(output_dir)
            
            await session.commit()
            
            # Final broadcast
            await multimodal_manager.broadcast_progress(job_id, {
                "status": "completed",
                "progress": 100.0,
                "currentStep": "Generation complete",
                "samplesGenerated": len(samples),
                "totalSamples": config["sampleCount"],
                "outputPath": str(output_dir)
            })
            
            logger.info(f"Multimodal dataset generation completed for job {job_id}: {len(samples)} samples")
            
            return {
                "job_id": job_id,
                "status": "completed",
                "samples_generated": len(samples),
                "output_path": str(output_dir),
                "config": config
            }
            
        except Exception as e:
            # Update status to failed
            job.status = "failed"
            job.error_message = str(e)
            job.current_step = f"Failed: {str(e)}"
            await session.commit()
            
            # Broadcast failure
            await multimodal_manager.broadcast_progress(job_id, {
                "status": "failed",
                "error": str(e),
                "currentStep": f"Generation failed: {str(e)}"
            })
            
            logger.error(f"Multimodal dataset generation failed for job {job_id}: {str(e)}")
            raise


async def _generate_dataset_with_progress(
    generator: MultimodalDatasetGenerator,
    config: SyntheticGenerationConfig,
    progress_callback
) -> list:
    """Generate dataset with enhanced progress tracking"""
    
    logger.info(f"Starting multimodal dataset generation: {config.num_samples} samples")
    
    # Character generation phase (5% of total progress)
    await progress_callback(1.0, "Generating diverse characters...", 0)
    characters = await generator._generate_characters()
    await progress_callback(5.0, f"Generated {len(characters)} characters", 0)
    
    # Sample generation phase (90% of total progress)
    samples = []
    samples_per_progress_update = max(1, config.num_samples // 100)  # Update every 1%
    
    for i in range(config.num_samples):
        try:
            # Select random character and narrative type
            import random
            character = random.choice(characters)
            narrative_type = random.choice(config.narrative_types)
            
            # Update progress every few samples
            if i % samples_per_progress_update == 0 or i == config.num_samples - 1:
                progress = 5.0 + (i / config.num_samples) * 90.0  # 5% base + 90% for samples
                await progress_callback(
                    progress,
                    f"Generating speech for character: {character['name']}",
                    i
                )
            
            # Generate sample
            sample = await generator._generate_sample(
                character=character,
                narrative_type=narrative_type,
                sample_index=i
            )
            
            if sample:
                samples.append(sample)
                
        except Exception as e:
            logger.error(f"Error generating sample {i}: {e}")
            continue
    
    await progress_callback(95.0, "Saving dataset to disk...", len(samples))
    
    # Save dataset (5% of total progress)
    generator._save_dataset(samples)
    
    await progress_callback(100.0, "Generation complete!", len(samples))
    
    logger.info(f"Generated {len(samples)} valid samples out of {config.num_samples} requested")
    return samples 