from celery import Task
from app.celery_app import celery_app
from app.redis_client import TrainingStatusTracker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models import Dataset, Character
import asyncio
import sys
import os

# Add parent directory to path to import from main app
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from main app utils
from app.utils.dataset.multi_turn_dataset_generator import MultiTurnDatasetGenerator
from app.utils.character.character import CharacterManager
from app.utils.world import WorldManager

class DatasetGenerationTask(Task):
    """Custom task class for dataset generation."""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        dataset_id = args[0] if args else kwargs.get('dataset_id')
        asyncio.run(self._update_failed_status(dataset_id, str(exc)))
    
    async def _update_failed_status(self, dataset_id: str, error_message: str):
        """Update dataset status to failed."""
        AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with AsyncSessionLocal() as session:
            dataset = await session.get(Dataset, dataset_id)
            if dataset:
                dataset.status = "failed"
                dataset.error_message = error_message
                await session.commit()

@celery_app.task(bind=True, base=DatasetGenerationTask, name='app.tasks.dataset_generation.generate_character_dataset')
def generate_character_dataset(self, dataset_id: str, generation_params: dict):
    """Generate synthetic conversations for a character."""
    
    # Run async function in sync context
    return asyncio.run(_generate_dataset(self, dataset_id, generation_params))

async def _generate_dataset(task, dataset_id: str, generation_params: dict):
    """Async dataset generation logic."""
    
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as session:
        # Get dataset and character
        dataset = await session.get(Dataset, dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        character = await session.get(Character, dataset.character_id)
        if not character:
            raise ValueError(f"Character {dataset.character_id} not found")
        
        # Update status to generating
        dataset.status = "generating"
        await session.commit()
        
        # Update Redis status
        await TrainingStatusTracker.update_status(
            dataset_id,
            "generating",
            0.0,
            "Starting dataset generation"
        )
        
        try:
            # Initialize managers
            character_manager = CharacterManager()
            world_manager = WorldManager()
            
            # Load character and world data
            # Note: This is simplified - you'll need to adapt based on your actual implementation
            character_data = {
                "name": character.name,
                "description": character.description,
                "personality": {
                    "openness": character.openness,
                    "conscientiousness": character.conscientiousness,
                    "extraversion": character.extraversion,
                    "agreeableness": character.agreeableness,
                    "neuroticism": character.neuroticism
                },
                "backstory": character.backstory,
                "goals": character.goals,
                "relationships": character.relationships,
                "traits": character.traits or {},
                "voice_style": character.voice_style
            }
            
            # Generate dataset
            generator = MultiTurnDatasetGenerator()
            
            # Progress callback
            async def progress_callback(progress: float, message: str):
                await TrainingStatusTracker.update_status(
                    dataset_id,
                    "generating",
                    progress,
                    message
                )
                
                # Update Celery task state
                task.update_state(
                    state='PROGRESS',
                    meta={'current': progress, 'total': 100, 'status': message}
                )
            
            # Generate conversations
            output_path = f"datasets/{dataset_id}.jsonl"
            os.makedirs("datasets", exist_ok=True)
            
            result = await generator.generate_dataset_async(
                character_data=character_data,
                num_conversations=generation_params.get("num_conversations", 100),
                output_path=output_path,
                progress_callback=progress_callback
            )
            
            # Update dataset with results
            dataset.status = "completed"
            dataset.file_path = output_path
            dataset.conversation_count = result.get("conversation_count", 0)
            dataset.total_messages = result.get("total_messages", 0)
            
            await session.commit()
            
            # Final status update
            await TrainingStatusTracker.update_status(
                dataset_id,
                "completed",
                100.0,
                "Dataset generation completed",
                metrics={
                    "conversations": dataset.conversation_count,
                    "messages": dataset.total_messages
                }
            )
            
            return {
                "dataset_id": dataset_id,
                "status": "completed",
                "file_path": output_path,
                "conversations": dataset.conversation_count,
                "messages": dataset.total_messages
            }
            
        except Exception as e:
            # Update status to failed
            dataset.status = "failed"
            dataset.error_message = str(e)
            await session.commit()
            
            await TrainingStatusTracker.update_status(
                dataset_id,
                "failed",
                0.0,
                f"Dataset generation failed: {str(e)}"
            )
            
            raise 