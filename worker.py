"""
🔄 Celery Worker for Async Training & Data Collection

This module defines the Celery application and tasks for:
- Asynchronous model training (R3-2)  
- Conversation data collection pipeline (R3-2.5)
- Background processing for character creation platform

Architecture:
- Training tasks run in GPU-enabled workers
- Data collection runs continuously in CPU workers  
- Database serves as the single source of truth
- Redis provides task queuing and result backend
"""

import os
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

import celery
from celery import Celery
from celery.signals import task_prerun, task_postrun

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get configuration from environment
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery app
app = Celery('character_training_platform')

# Configure Celery
app.conf.update(
    broker_url=REDIS_URL,
    result_backend=REDIS_URL,
    
    # Task routing and execution
    task_routes={
        'worker.run_training': {'queue': 'training'},
        'worker.collect_conversation_data': {'queue': 'data_collection'},
        'worker.process_conversation_batch': {'queue': 'data_processing'},
    },
    
    # Optimization settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for GPU-intensive training
    task_acks_late=True,
    worker_disable_rate_limits=True,
    
    # Task timeouts
    task_time_limit=7200,  # 2 hours max for training
    task_soft_time_limit=6900,  # 1h 55m soft limit
    
    # Result settings
    result_expires=3600,  # Results expire after 1 hour
)

# Import utilities after app configuration
try:
    from app.utils.database.session import session_scope
    from app.utils.database.models import TrainingRun, User, Character, ConversationLog
    from app.utils.training import TrainingManager
    from app.utils.character.character import CharacterManager
    from app.utils.dataset.manager import DatasetManager
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure to run worker from project root with proper PYTHONPATH")
    raise


@app.task(bind=True, name='worker.run_training')
def run_training(self, training_run_id: int) -> Dict[str, Any]:
    """
    Execute model training asynchronously.
    
    Args:
        training_run_id: Database ID of the TrainingRun record
        
    Returns:
        Dict with training results and status
    """
    logger.info(f"🚀 Starting async training for run ID: {training_run_id}")
    
    try:
        # Update task ID in database  
        with session_scope() as session:
            training_run = session.query(TrainingRun).get(training_run_id)
            if not training_run:
                raise ValueError(f"Training run {training_run_id} not found")
            
            # Update status to processing
            training_run.status = 'processing'
            training_run.started_at = datetime.now(timezone.utc)
            session.commit()
            
            # Load character and configuration
            character = training_run.character
            config_json = training_run.config_json or {}
            
            logger.info(f"📋 Training character: {character.name}")
            logger.info(f"⚙️ Base model: {training_run.base_model}")
            logger.info(f"🔧 Method: {training_run.training_method}")
        
        # Initialize training manager with production settings
        # Check if contamination MoE training is requested
        use_contamination_moe = config_json.get('use_contamination_moe', False)
        
        if use_contamination_moe:
            logger.info("🔥 Initializing Contamination-Isolation MoE training!")
            from narrative_engine.contamination_moe_trainer import create_contamination_moe_training_manager
            training_manager = create_contamination_moe_training_manager(
                base_model=training_run.base_model,
                force_gpu=True
            )
        else:
            training_manager = TrainingManager(
                base_model=training_run.base_model,
                force_gpu=True  # Workers should use GPU
            )
        
        # Load character data for training
        character_data = {
            'name': character.name,
            'description': character.description,
            'personality': character.personality_json,
            'scenario': character.scenario,
            'first_mes': character.first_message,
        }
        
        # Load dataset for the character
        dataset_manager = DatasetManager()
        character_manager = CharacterManager()
        
        # Generate or load dataset
        logger.info("📊 Loading training dataset...")
        dataset = dataset_manager.load_character_dataset(character.name)
        
        if not dataset:
            logger.info("🔄 No existing dataset found, generating new one...")
            # Generate dataset if none exists
            dataset = dataset_manager.generate_dataset(character_data, config_json.get('dataset_size', 100))
        
        logger.info(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Configure advanced training features from config
        advanced_config = config_json.get('advanced_training_config', {})
        training_manager.configure_advanced_features(advanced_config)
        
        # Start training process
        logger.info("🏁 Starting training process...")
        
        if use_contamination_moe:
            logger.info("🔥 LAUNCHING CONTAMINATION WARFARE TRAINING!")
            training_manager.start_contamination_warfare_training(character_data, dataset, config_json)
        else:
            training_manager.start_training(character_data, dataset, config_json)
        
        # Monitor training progress
        final_metrics = {}
        while training_manager.is_training:
            # Get current metrics
            current_metrics = training_manager.get_metrics()
            
            # Update database with progress
            with session_scope() as session:
                training_run = session.query(TrainingRun).get(training_run_id)
                if training_run:
                    training_run.metrics_json = current_metrics
                    session.commit()
            
            # Store final metrics when training completes
            if 'training_complete' in current_metrics.get('type', ''):
                final_metrics = current_metrics
                break
            
            # Check for task cancellation
            if self.request.called_directly:
                # Regular task execution, continue monitoring
                import time
                time.sleep(5)  # Check every 5 seconds
            else:
                # Can check for revocation in distributed setup
                break
        
        # Training completed - update database
        with session_scope() as session:
            training_run = session.query(TrainingRun).get(training_run_id)
            if training_run:
                if 'error' in final_metrics.get('type', ''):
                    training_run.status = 'failed'
                    training_run.metrics_json = {'error': final_metrics.get('message', 'Training failed')}
                else:
                    training_run.status = 'completed'
                    training_run.completed_at = datetime.now(timezone.utc)
                    
                    # Save training artifacts
                    if 'output_dir' in final_metrics:
                        training_run.sft_adapter_path = final_metrics['output_dir']
                    if 'rlhf_adapter_path' in final_metrics:
                        training_run.rlhf_adapter_path = final_metrics['rlhf_adapter_path']
                    
                    # Save final metrics
                    training_run.metrics_json = final_metrics
                    training_run.final_loss = final_metrics.get('final_loss')
                    training_run.total_steps = final_metrics.get('final_step')
                
                session.commit()
        
        logger.info(f"✅ Training completed successfully for run {training_run_id}")
        return {
            'status': 'completed',
            'training_run_id': training_run_id,
            'metrics': final_metrics
        }
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        error_traceback = traceback.format_exc()
        logger.error(f"❌ {error_msg}")
        logger.error(f"🔍 Traceback: {error_traceback}")
        
        # Update database with error
        try:
            with session_scope() as session:
                training_run = session.query(TrainingRun).get(training_run_id)
                if training_run:
                    training_run.status = 'failed'
                    training_run.metrics_json = {
                        'error': error_msg,
                        'traceback': error_traceback,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    session.commit()
        except Exception as db_error:
            logger.error(f"Failed to update database with error: {db_error}")
        
        return {
            'status': 'failed',
            'training_run_id': training_run_id,
            'error': error_msg
        }


@app.task(bind=True, name='worker.collect_conversation_data')  
def collect_conversation_data(self, user_id: int, character_id: int, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and store conversation data for model training.
    
    Args:
        user_id: ID of the user
        character_id: ID of the character
        conversation_data: Raw conversation data to process
        
    Returns:
        Dict with processing results
    """
    logger.info(f"📝 Processing conversation data for user {user_id}, character {character_id}")
    
    try:
        # Validate and sanitize conversation data
        processed_data = validate_conversation_data(conversation_data)
        
        if not processed_data:
            return {'status': 'skipped', 'reason': 'invalid_data'}
        
        # Check user consent for data collection
        with session_scope() as session:
            user = session.query(User).get(user_id)
            character = session.query(Character).get(character_id)
            
            if not user or not character:
                return {'status': 'failed', 'reason': 'invalid_ids'}
            
            # Check if user has opted in to data collection
            user_profile = user.profile_json or {}
            if not user_profile.get('data_collection_consent', False):
                return {'status': 'skipped', 'reason': 'no_consent'}
            
            # Create conversation log entry
            conversation_log = ConversationLog(
                user_id=user_id,
                character_id=character_id,
                conversation_data=processed_data,
                quality_score=calculate_conversation_quality(processed_data),
                created_at=datetime.now(timezone.utc)
            )
            
            session.add(conversation_log)
            session.commit()
            
            logger.info(f"✅ Conversation data stored for user {user_id}")
            
            return {
                'status': 'success',
                'conversation_id': conversation_log.id,
                'quality_score': conversation_log.quality_score
            }
            
    except Exception as e:
        logger.error(f"❌ Failed to process conversation data: {e}")
        return {'status': 'failed', 'error': str(e)}


@app.task(bind=True, name='worker.process_conversation_batch')
def process_conversation_batch(self, batch_size: int = 100) -> Dict[str, Any]:
    """
    Process a batch of conversations for dataset preparation.
    
    Args:
        batch_size: Number of conversations to process
        
    Returns:
        Dict with batch processing results
    """
    logger.info(f"🔄 Processing conversation batch (size: {batch_size})")
    
    try:
        processed_count = 0
        
        with session_scope() as session:
            # Get unprocessed conversations
            conversations = session.query(ConversationLog).filter(
                ConversationLog.processed_for_training == False,
                ConversationLog.quality_score >= 3.0  # Minimum quality threshold
            ).limit(batch_size).all()
            
            for conversation in conversations:
                # Apply data transformations for training
                training_sample = prepare_conversation_for_training(conversation.conversation_data)
                
                if training_sample:
                    # Save to training dataset
                    character_name = conversation.character.name
                    save_training_sample(character_name, training_sample)
                    
                    # Mark as processed
                    conversation.processed_for_training = True
                    processed_count += 1
            
            session.commit()
        
        logger.info(f"✅ Processed {processed_count} conversations for training")
        return {
            'status': 'success',
            'processed_count': processed_count
        }
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {e}")
        return {'status': 'failed', 'error': str(e)}


# Helper functions for data processing
def validate_conversation_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and sanitize conversation data."""
    required_fields = ['messages', 'timestamp']
    
    if not all(field in data for field in required_fields):
        return None
    
    # Basic PII scrubbing
    cleaned_data = scrub_pii(data)
    
    # Validate message structure
    if not isinstance(cleaned_data.get('messages'), list):
        return None
    
    return cleaned_data


def scrub_pii(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove personally identifiable information from conversation data."""
    import re
    
    # Create a deep copy to avoid modifying original
    import copy
    cleaned_data = copy.deepcopy(data)
    
    # PII patterns to remove/replace
    pii_patterns = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
        (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]'),  # Phone numbers
        (r'\b\d{1,5}\s[\w\s]{1,}\s(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b', '[ADDRESS]'),  # Addresses
    ]
    
    # Apply PII scrubbing to message content
    if 'messages' in cleaned_data:
        for message in cleaned_data['messages']:
            if 'content' in message:
                content = message['content']
                for pattern, replacement in pii_patterns:
                    content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                message['content'] = content
    
    return cleaned_data


def calculate_conversation_quality(data: Dict[str, Any]) -> float:
    """Calculate quality score for conversation data."""
    score = 5.0  # Start with perfect score
    
    messages = data.get('messages', [])
    if not messages:
        return 0.0
    
    # Penalize very short conversations
    if len(messages) < 4:
        score -= 1.0
    
    # Check for very short messages
    short_messages = sum(1 for msg in messages if len(msg.get('content', '')) < 10)
    if short_messages > len(messages) * 0.5:
        score -= 1.0
    
    # Check for engagement indicators
    total_length = sum(len(msg.get('content', '')) for msg in messages)
    avg_length = total_length / len(messages) if messages else 0
    
    if avg_length < 20:
        score -= 0.5
    elif avg_length > 100:
        score += 0.5
    
    return max(0.0, min(5.0, score))


def prepare_conversation_for_training(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert conversation data to training format."""
    messages = data.get('messages', [])
    if not messages:
        return None
    
    # Convert to training format (adjust based on your training needs)
    training_sample = {
        'messages': messages,
        'source': 'user_conversation',
        'quality_score': calculate_conversation_quality(data),
        'timestamp': data.get('timestamp')
    }
    
    return training_sample


def save_training_sample(character_name: str, sample: Dict[str, Any]) -> None:
    """Save a training sample to character's dataset."""
    # Create character dataset directory
    dataset_dir = Path(f"content/worlds/Default World/characters/{character_name}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Append to conversation dataset file
    dataset_file = dataset_dir / "user_conversations.jsonl"
    
    with open(dataset_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')


# Task lifecycle hooks
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Called before task execution."""
    logger.info(f"🏁 Starting task: {task.name} (ID: {task_id})")


@task_postrun.connect  
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Called after task execution."""
    logger.info(f"✅ Completed task: {task.name} (ID: {task_id}) - State: {state}")


if __name__ == '__main__':
    # For debugging - run worker directly
    app.start() 