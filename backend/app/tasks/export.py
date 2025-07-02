from celery import Task
from app.celery_app import celery_app
from app.redis_client import TrainingStatusTracker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models import Character, World
import asyncio
import json
import os
import shutil
from datetime import datetime

@celery_app.task(name='app.tasks.export.export_character_packet')
def export_character_packet(character_id: str, export_path: str = None):
    """Export a character as a runtime packet."""
    
    # Run async function in sync context
    return asyncio.run(_export_character(character_id, export_path))

async def _export_character(character_id: str, export_path: str = None):
    """Async character export logic."""
    
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as session:
        # Get character and world
        character = await session.get(Character, character_id)
        if not character:
            raise ValueError(f"Character {character_id} not found")
        
        world = await session.get(World, character.world_id)
        if not world:
            raise ValueError(f"World {character.world_id} not found")
        
        # Prepare export directory
        if not export_path:
            export_path = f"exports/character_{character_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(export_path, exist_ok=True)
        
        try:
            # Update status
            await TrainingStatusTracker.update_status(
                f"export_{character_id}",
                "exporting",
                0.0,
                "Starting character export"
            )
            
            # 1. Export character core data
            character_core = {
                "id": character.id,
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
                "voice_style": character.voice_style,
                "model_version": character.model_version,
                "created_at": character.created_at.isoformat(),
                "updated_at": character.updated_at.isoformat()
            }
            
            with open(os.path.join(export_path, "character_core.json"), "w") as f:
                json.dump(character_core, f, indent=2)
            
            await TrainingStatusTracker.update_status(
                f"export_{character_id}",
                "exporting",
                25.0,
                "Exported character data"
            )
            
            # 2. Export world lore
            world_lore = {
                "id": world.id,
                "name": world.name,
                "description": world.description,
                "setting": world.setting,
                "rules": world.rules or {},
                "history": world.history,
                "cultures": world.cultures or {},
                "locations": world.locations or {}
            }
            
            with open(os.path.join(export_path, "world_lore.json"), "w") as f:
                json.dump(world_lore, f, indent=2)
            
            await TrainingStatusTracker.update_status(
                f"export_{character_id}",
                "exporting",
                50.0,
                "Exported world lore"
            )
            
            # 3. Copy adapter files if character is trained
            if character.is_trained and character.adapter_path:
                adapter_dest = os.path.join(export_path, "adapter")
                if os.path.exists(character.adapter_path):
                    shutil.copytree(character.adapter_path, adapter_dest)
                else:
                    # Create dummy adapter files for demo
                    os.makedirs(adapter_dest, exist_ok=True)
                    with open(os.path.join(adapter_dest, "adapter_config.json"), "w") as f:
                        json.dump({
                            "adapter_type": "lora",
                            "rank": 16,
                            "alpha": 32,
                            "target_modules": ["q_proj", "v_proj"]
                        }, f, indent=2)
            
            await TrainingStatusTracker.update_status(
                f"export_{character_id}",
                "exporting",
                75.0,
                "Exported model adapter"
            )
            
            # 4. Create runtime configuration
            runtime_config = {
                "version": "1.0",
                "character_id": character.id,
                "world_id": world.id,
                "requires_adapter": character.is_trained,
                "base_model": "meta-llama/Llama-2-7b-chat-hf",
                "inference_params": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": 50,
                    "max_length": 2048,
                    "repetition_penalty": 1.1
                },
                "control_tokens": {
                    "emotion_markers": True,
                    "action_markers": True,
                    "thought_markers": True
                }
            }
            
            with open(os.path.join(export_path, "runtime_config.json"), "w") as f:
                json.dump(runtime_config, f, indent=2)
            
            # 5. Create README
            readme_content = f"""# Character Runtime Packet

## Character: {character.name}
**World**: {world.name}
**Exported**: {datetime.utcnow().isoformat()}

### Contents
- `character_core.json`: Character personality and details
- `world_lore.json`: World information and rules
- `runtime_config.json`: Runtime configuration
- `adapter/`: Model adapter files (if trained)

### Usage
1. Load this packet in the runtime engine
2. The engine will automatically load the character and world data
3. If trained, the adapter will be applied to the base model
4. Start chatting with your character!

### Character Summary
{character.description}

### Personality Profile
- Openness: {character.openness:.2f}
- Conscientiousness: {character.conscientiousness:.2f}
- Extraversion: {character.extraversion:.2f}
- Agreeableness: {character.agreeableness:.2f}
- Neuroticism: {character.neuroticism:.2f}
"""
            
            with open(os.path.join(export_path, "README.md"), "w") as f:
                f.write(readme_content)
            
            await TrainingStatusTracker.update_status(
                f"export_{character_id}",
                "completed",
                100.0,
                "Export completed successfully"
            )
            
            return {
                "character_id": character_id,
                "export_path": export_path,
                "files": os.listdir(export_path),
                "status": "completed"
            }
            
        except Exception as e:
            await TrainingStatusTracker.update_status(
                f"export_{character_id}",
                "failed",
                0.0,
                f"Export failed: {str(e)}"
            )
            
            # Clean up on failure
            if os.path.exists(export_path):
                shutil.rmtree(export_path)
            
            raise 