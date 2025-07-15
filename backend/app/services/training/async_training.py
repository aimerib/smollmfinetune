"""
🚀 Async Training Service

This service handles:
- Training job queuing via Celery
- Status monitoring and progress tracking
- Database record management
- Data collection integration

Works with worker.py to provide seamless async training experience.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path

try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

from backend.app.core.database.session import session_scope
from backend.app.core.database.models import TrainingRun, Character, User

logger = logging.getLogger(__name__)

class AsyncTrainingService:
    """Service for managing async training jobs"""
    
    def __init__(self):
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.celery_app = None
        
        if CELERY_AVAILABLE:
            self._setup_celery()
    
    def _setup_celery(self):
        """Setup Celery connection for task queuing."""
        try:
            self.celery_app = Celery('character_training_platform')
            self.celery_app.conf.update(
                broker_url=self.redis_url,
                result_backend=self.redis_url,
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
            )
            logger.info("✅ Celery connection established")
        except Exception as e:
            logger.error(f"❌ Failed to setup Celery: {e}")
            self.celery_app = None
    
    def start_training(self, character_data: Dict[str, Any], dataset: List[Dict[str, Any]], 
                      config: Dict[str, Any], user_id: int) -> Optional[int]:
        """
        Queue a training job and return the training run ID.
        
        Args:
            character_data: Character information
            dataset: Training dataset
            config: Training configuration
            user_id: ID of the user starting training
            
        Returns:
            Training run ID if successful, None if failed
        """
        try:
            # Create training run record in database
            with session_scope() as session:
                # Find or create character in database
                character_name = character_data.get('name', 'Unknown')
                character = session.query(Character).filter_by(
                    name=character_name,
                    owner_id=user_id
                ).first()
                
                if not character:
                    # Create basic character record for training
                    character = Character(
                        name=character_name,
                        owner_id=user_id,
                        world_id=1,  # Default world for now
                        description=character_data.get('description', ''),
                        scenario=character_data.get('scenario', ''),
                        first_message=character_data.get('first_mes', ''),
                        personality_json=character_data.get('personality', {}),
                        core_data_json=character_data
                    )
                    session.add(character)
                    session.flush()  # Get the ID
                
                # Create training run record
                training_run = TrainingRun(
                    character_id=character.id,
                    owner_id=user_id,
                    base_model=config.get('base_model', 'HuggingFaceTB/SmolLM2-135M-Instruct'),
                    training_method=config.get('finetune_method', 'lora'),
                    status='queued',
                    output_directory=f"training_output/{character_name}",
                    config_json=config,
                    dataset_size=len(dataset),
                    created_at=datetime.now(timezone.utc)
                )
                
                session.add(training_run)
                session.commit()
                
                training_run_id = training_run.id
            
            # Queue the training task if Celery is available
            if self.celery_app:
                # Send task to worker
                task = self.celery_app.send_task(
                    'worker.run_training',
                    args=[training_run_id],
                    queue='training'
                )
                
                logger.info(f"🚀 Training job queued: run_id={training_run_id}, task_id={task.id}")
                
                # Update database with task ID
                with session_scope() as session:
                    training_run = session.query(TrainingRun).get(training_run_id)
                    if training_run:
                        if not training_run.config_json:
                            training_run.config_json = {}
                        training_run.config_json['celery_task_id'] = task.id
                        session.commit()
                
                return training_run_id
            else:
                # Fallback: mark as failed if no Celery
                with session_scope() as session:
                    training_run = session.query(TrainingRun).get(training_run_id)
                    if training_run:
                        training_run.status = 'failed'
                        training_run.metrics_json = {'error': 'Celery not available'}
                        session.commit()
                
                logger.error("❌ Celery not available, training job marked as failed")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to start training: {e}")
            return None
    
    def get_training_status(self, training_run_id: int) -> Dict[str, Any]:
        """
        Get current status of a training run.
        
        Args:
            training_run_id: ID of the training run
            
        Returns:
            Dict with training status and metrics
        """
        try:
            with session_scope() as session:
                training_run = session.query(TrainingRun).get(training_run_id)
                
                if not training_run:
                    return {'status': 'not_found', 'error': 'Training run not found'}
                
                # Build status response
                status_data = {
                    'training_run_id': training_run_id,
                    'status': training_run.status,
                    'character_name': training_run.character.name,
                    'base_model': training_run.base_model,
                    'training_method': training_run.training_method,
                    'created_at': training_run.created_at.isoformat() if training_run.created_at else None,
                    'started_at': training_run.started_at.isoformat() if training_run.started_at else None,
                    'completed_at': training_run.completed_at.isoformat() if training_run.completed_at else None,
                    'dataset_size': training_run.dataset_size,
                    'total_steps': training_run.total_steps,
                    'final_loss': training_run.final_loss,
                    'sft_adapter_path': training_run.sft_adapter_path,
                    'rlhf_adapter_path': training_run.rlhf_adapter_path,
                    'metrics': training_run.metrics_json or {},
                    'config': training_run.config_json or {}
                }
                
                # Add task status if available
                if self.celery_app and training_run.config_json:
                    task_id = training_run.config_json.get('celery_task_id')
                    if task_id:
                        try:
                            task_result = self.celery_app.AsyncResult(task_id)
                            status_data['task_status'] = task_result.status
                            status_data['task_info'] = task_result.info
                        except Exception as e:
                            logger.warning(f"Failed to get task status: {e}")
                
                return status_data
                
        except Exception as e:
            logger.error(f"❌ Failed to get training status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_user_training_runs(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent training runs for a user.
        
        Args:
            user_id: ID of the user
            limit: Maximum number of runs to return
            
        Returns:
            List of training run status dictionaries
        """
        try:
            with session_scope() as session:
                training_runs = session.query(TrainingRun).filter_by(
                    owner_id=user_id
                ).order_by(TrainingRun.created_at.desc()).limit(limit).all()
                
                return [self.get_training_status(run.id) for run in training_runs]
                
        except Exception as e:
            logger.error(f"❌ Failed to get user training runs: {e}")
            return []
    
    def cancel_training(self, training_run_id: int, user_id: int) -> bool:
        """
        Cancel a queued or running training job.
        
        Args:
            training_run_id: ID of the training run
            user_id: ID of the user (for authorization)
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            with session_scope() as session:
                training_run = session.query(TrainingRun).filter_by(
                    id=training_run_id,
                    owner_id=user_id
                ).first()
                
                if not training_run:
                    return False
                
                # Can only cancel queued or processing jobs
                if training_run.status not in ['queued', 'processing']:
                    return False
                
                # Cancel Celery task if available
                if self.celery_app and training_run.config_json:
                    task_id = training_run.config_json.get('celery_task_id')
                    if task_id:
                        try:
                            self.celery_app.control.revoke(task_id, terminate=True)
                            logger.info(f"🛑 Cancelled Celery task: {task_id}")
                        except Exception as e:
                            logger.warning(f"Failed to cancel Celery task: {e}")
                
                # Update database status
                training_run.status = 'cancelled'
                training_run.metrics_json = {
                    'cancelled_at': datetime.now(timezone.utc).isoformat(),
                    'cancelled_by_user': True
                }
                session.commit()
                
                logger.info(f"✅ Training run {training_run_id} cancelled")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to cancel training: {e}")
            return False


class DataCollectionService:
    """Service for collecting conversation data for training"""
    
    def __init__(self):
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.celery_app = None
        
        if CELERY_AVAILABLE:
            self._setup_celery()
    
    def _setup_celery(self):
        """Setup Celery connection for data collection tasks."""
        try:
            self.celery_app = Celery('character_training_platform')
            self.celery_app.conf.update(
                broker_url=self.redis_url,
                result_backend=self.redis_url,
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
            )
        except Exception as e:
            logger.error(f"❌ Failed to setup Celery for data collection: {e}")
            self.celery_app = None
    
    def collect_conversation(self, user_id: int, character_id: int, 
                           messages: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> bool:
        """
        Queue conversation data for collection and processing.
        
        Args:
            user_id: ID of the user
            character_id: ID of the character
            messages: List of conversation messages
            metadata: Additional metadata
            
        Returns:
            True if queued successfully, False otherwise
        """
        try:
            # Prepare conversation data
            conversation_data = {
                'messages': messages,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            # Queue data collection task if Celery is available
            if self.celery_app:
                task = self.celery_app.send_task(
                    'worker.collect_conversation_data',
                    args=[user_id, character_id, conversation_data],
                    queue='data_collection'
                )
                
                logger.info(f"📝 Conversation data queued for collection: task_id={task.id}")
                return True
            else:
                logger.warning("⚠️ Celery not available, conversation data not collected")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to queue conversation data: {e}")
            return False
    
    def process_conversation_batch(self, batch_size: int = 100) -> bool:
        """
        Trigger batch processing of collected conversations.
        
        Args:
            batch_size: Number of conversations to process
            
        Returns:
            True if queued successfully, False otherwise
        """
        try:
            if self.celery_app:
                task = self.celery_app.send_task(
                    'worker.process_conversation_batch',
                    args=[batch_size],
                    queue='data_processing'
                )
                
                logger.info(f"🔄 Conversation batch processing queued: task_id={task.id}")
                return True
            else:
                logger.warning("⚠️ Celery not available, batch processing not queued")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to queue batch processing: {e}")
            return False


# Global service instances
async_training_service = AsyncTrainingService()
data_collection_service = DataCollectionService() 