"""
Advanced Job Manager for Multimodal Studio

Provides comprehensive job management capabilities including:
- Job queue management with priority handling
- Batch operations on multiple jobs
- Real-time progress tracking
- Error recovery and job lifecycle management
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class JobStatus(str, Enum):
    QUEUED = "queued"
    GENERATING = "generating"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(int, Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 5
    LOW = 8
    BACKGROUND = 10

class MultimodalJob:
    """Represents a multimodal dataset generation job"""
    
    def __init__(
        self,
        id: str,
        user_id: str,
        dataset_id: str,
        configuration: Dict[str, Any],
        priority: int = JobPriority.NORMAL
    ):
        self.id = id
        self.user_id = user_id
        self.dataset_id = dataset_id
        self.status = JobStatus.QUEUED
        self.progress = 0.0
        self.total_steps = 0
        self.current_step = 0
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.error_message = None
        self.configuration = configuration
        self.metrics = {}
        self.priority = priority
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'dataset_id': self.dataset_id,
            'status': self.status.value,
            'progress': self.progress,
            'total_steps': self.total_steps,
            'current_step': self.current_step,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'configuration': self.configuration,
            'metrics': self.metrics,
            'priority': self.priority
        }

class JobQueue:
    """Priority-based job queue with advanced management features"""
    
    def __init__(self):
        self.jobs: Dict[str, MultimodalJob] = {}
        self.user_queues: Dict[str, List[str]] = {}  # user_id -> [job_ids]
        self.active_jobs: Dict[str, MultimodalJob] = {}
        self.completed_jobs: Dict[str, MultimodalJob] = {}
        self._lock = asyncio.Lock()
        
    async def add_job(self, job: MultimodalJob) -> None:
        """Add job to queue with priority ordering"""
        async with self._lock:
            self.jobs[job.id] = job
            
            if job.user_id not in self.user_queues:
                self.user_queues[job.user_id] = []
            
            # Insert job in priority order (lower priority number = higher priority)
            user_queue = self.user_queues[job.user_id]
            inserted = False
            
            for i, existing_job_id in enumerate(user_queue):
                existing_job = self.jobs[existing_job_id]
                if job.priority < existing_job.priority:
                    user_queue.insert(i, job.id)
                    inserted = True
                    break
            
            if not inserted:
                user_queue.append(job.id)
                
    async def get_next_job(self, user_id: str) -> Optional[MultimodalJob]:
        """Get next job for user respecting priority"""
        async with self._lock:
            if user_id not in self.user_queues or not self.user_queues[user_id]:
                return None
                
            # Find highest priority queued job
            for job_id in self.user_queues[user_id]:
                job = self.jobs.get(job_id)
                if job and job.status == JobStatus.QUEUED:
                    return job
                    
            return None
    
    async def move_to_active(self, job_id: str) -> bool:
        """Move job from queue to active status"""
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
                
            job.status = JobStatus.GENERATING
            job.started_at = datetime.utcnow()
            self.active_jobs[job_id] = job
            return True
    
    async def complete_job(self, job_id: str, success: bool = True, error_message: str = None) -> bool:
        """Complete job and move to completed status"""
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
                
            job.completed_at = datetime.utcnow()
            job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
            job.progress = 100.0 if success else job.progress
            
            if error_message:
                job.error_message = error_message
            
            # Move from active to completed
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            
            self.completed_jobs[job_id] = job
            
            # Remove from user queue
            if job.user_id in self.user_queues:
                try:
                    self.user_queues[job.user_id].remove(job_id)
                except ValueError:
                    pass  # Job already removed
                    
            return True
    
    async def get_queue_status(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive queue status for user"""
        async with self._lock:
            user_jobs = self.user_queues.get(user_id, [])
            
            queued_jobs = []
            active_jobs = []
            
            for job_id in user_jobs:
                job = self.jobs.get(job_id)
                if job:
                    if job.status == JobStatus.QUEUED:
                        queued_jobs.append(job.to_dict())
                    elif job.status == JobStatus.GENERATING:
                        active_jobs.append(job.to_dict())
            
            return {
                'queued_count': len(queued_jobs),
                'active_count': len(active_jobs),
                'queued_jobs': queued_jobs,
                'active_jobs': active_jobs,
                'estimated_wait_time': await self._estimate_wait_time(user_id)
            }
    
    async def _estimate_wait_time(self, user_id: str) -> int:
        """Estimate wait time in minutes for next job"""
        # Simple estimation based on active jobs and queue position
        active_count = len([j for j in self.active_jobs.values() if j.user_id == user_id])
        queued_count = len([j for j_id in self.user_queues.get(user_id, []) 
                           if self.jobs.get(j_id, {}).status == JobStatus.QUEUED])
        
        # Assume average job takes 15 minutes
        estimated_minutes = (active_count * 15) + (queued_count * 15)
        return max(0, estimated_minutes)

class MultimodalJobManager:
    """Advanced job manager with comprehensive capabilities"""
    
    def __init__(self):
        self.queue = JobQueue()
        self.job_storage_path = Path("data/jobs")
        self.job_storage_path.mkdir(parents=True, exist_ok=True)
        self._background_tasks: Dict[str, asyncio.Task] = {}
        
    async def create_job(
        self,
        user_id: str,
        dataset_id: str,
        configuration: Dict[str, Any],
        priority: int = JobPriority.NORMAL
    ) -> MultimodalJob:
        """Create new multimodal generation job"""
        job_id = str(uuid.uuid4())
        
        job = MultimodalJob(
            id=job_id,
            user_id=user_id,
            dataset_id=dataset_id,
            configuration=configuration,
            priority=priority
        )
        
        await self.queue.add_job(job)
        await self._save_job(job)
        
        logger.info(f"Created job {job_id} for user {user_id}")
        return job
    
    async def get_user_jobs(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get jobs for user with filtering"""
        all_jobs = []
        
        # Get jobs from queue
        for job_id in self.queue.user_queues.get(user_id, []):
            job = self.queue.jobs.get(job_id)
            if job and (not status or job.status.value == status):
                all_jobs.append(job.to_dict())
        
        # Get completed jobs
        for job in self.queue.completed_jobs.values():
            if job.user_id == user_id and (not status or job.status.value == status):
                all_jobs.append(job.to_dict())
        
        # Sort by created_at descending
        all_jobs.sort(key=lambda x: x['created_at'], reverse=True)
        
        return all_jobs[offset:offset + limit]
    
    async def get_job(self, job_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get specific job if user has access"""
        job = (self.queue.jobs.get(job_id) or 
               self.queue.active_jobs.get(job_id) or 
               self.queue.completed_jobs.get(job_id))
        
        if job and job.user_id == user_id:
            return job.to_dict()
        return None
    
    async def batch_operation(
        self,
        action: str,
        job_ids: List[str],
        user_id: str,
        options: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute batch operations on multiple jobs"""
        results = []
        options = options or {}
        
        for job_id in job_ids:
            try:
                if action == 'pause':
                    result = await self._pause_job(job_id, user_id)
                elif action == 'resume':
                    result = await self._resume_job(job_id, user_id)
                elif action == 'cancel':
                    result = await self._cancel_job(job_id, user_id)
                elif action == 'delete':
                    result = await self._delete_job(job_id, user_id)
                elif action == 'prioritize':
                    new_priority = options.get('priority', JobPriority.HIGH)
                    result = await self.update_priority(job_id, new_priority, user_id)
                else:
                    result = {'success': False, 'error': f'Unknown action: {action}'}
                
                results.append({'job_id': job_id, **result})
                
            except Exception as e:
                logger.error(f"Batch operation {action} failed for job {job_id}: {e}")
                results.append({
                    'job_id': job_id,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def update_priority(self, job_id: str, priority: int, user_id: str) -> Dict[str, Any]:
        """Update job priority"""
        job = self.queue.jobs.get(job_id)
        if not job or job.user_id != user_id:
            return {'success': False, 'error': 'Job not found'}
        
        if job.status != JobStatus.QUEUED:
            return {'success': False, 'error': 'Can only change priority of queued jobs'}
        
        old_priority = job.priority
        job.priority = priority
        
        # Re-sort queue
        await self.queue.add_job(job)  # This will re-insert in correct position
        
        await self._save_job(job)
        
        logger.info(f"Updated job {job_id} priority from {old_priority} to {priority}")
        return {
            'success': True,
            'old_priority': old_priority,
            'new_priority': priority
        }
    
    async def get_queue_status(self, user_id: str) -> Dict[str, Any]:
        """Get queue status for user"""
        return await self.queue.get_queue_status(user_id)
    
    async def get_active_jobs(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active jobs for user"""
        active_jobs = []
        for job in self.queue.active_jobs.values():
            if job.user_id == user_id:
                active_jobs.append(job.to_dict())
        return active_jobs
    
    async def start_job_processing(self, job_id: str) -> bool:
        """Start processing a job"""
        job = self.queue.jobs.get(job_id)
        if not job or job.status != JobStatus.QUEUED:
            return False
        
        await self.queue.move_to_active(job_id)
        
        # Start background processing
        task = asyncio.create_task(self._process_job(job))
        self._background_tasks[job_id] = task
        
        return True
    
    async def update_job_progress(
        self,
        job_id: str,
        progress: float,
        current_step: int,
        total_steps: int,
        metrics: Dict[str, Any] = None
    ) -> bool:
        """Update job progress"""
        job = self.queue.active_jobs.get(job_id)
        if not job:
            return False
        
        job.progress = min(100.0, max(0.0, progress))
        job.current_step = current_step
        job.total_steps = total_steps
        
        if metrics:
            job.metrics.update(metrics)
        
        await self._save_job(job)
        return True
    
    async def _pause_job(self, job_id: str, user_id: str) -> Dict[str, Any]:
        """Pause a running job"""
        job = self.queue.active_jobs.get(job_id)
        if not job or job.user_id != user_id:
            return {'success': False, 'error': 'Job not found or not active'}
        
        job.status = JobStatus.PAUSED
        await self._save_job(job)
        
        # Cancel background task
        if job_id in self._background_tasks:
            self._background_tasks[job_id].cancel()
            del self._background_tasks[job_id]
        
        return {'success': True, 'new_status': JobStatus.PAUSED.value}
    
    async def _resume_job(self, job_id: str, user_id: str) -> Dict[str, Any]:
        """Resume a paused job"""
        job = self.queue.jobs.get(job_id)
        if not job or job.user_id != user_id or job.status != JobStatus.PAUSED:
            return {'success': False, 'error': 'Job not found or not paused'}
        
        job.status = JobStatus.GENERATING
        await self.queue.move_to_active(job_id)
        
        # Restart background processing
        task = asyncio.create_task(self._process_job(job))
        self._background_tasks[job_id] = task
        
        return {'success': True, 'new_status': JobStatus.GENERATING.value}
    
    async def _cancel_job(self, job_id: str, user_id: str) -> Dict[str, Any]:
        """Cancel a job"""
        job = (self.queue.jobs.get(job_id) or 
               self.queue.active_jobs.get(job_id))
        
        if not job or job.user_id != user_id:
            return {'success': False, 'error': 'Job not found'}
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return {'success': False, 'error': 'Job already finished'}
        
        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        
        # Cancel background task
        if job_id in self._background_tasks:
            self._background_tasks[job_id].cancel()
            del self._background_tasks[job_id]
        
        await self.queue.complete_job(job_id, success=False)
        await self._save_job(job)
        
        return {'success': True, 'new_status': JobStatus.CANCELLED.value}
    
    async def _delete_job(self, job_id: str, user_id: str) -> Dict[str, Any]:
        """Delete a job (only if completed/failed/cancelled)"""
        job = (self.queue.jobs.get(job_id) or 
               self.queue.completed_jobs.get(job_id))
        
        if not job or job.user_id != user_id:
            return {'success': False, 'error': 'Job not found'}
        
        if job.status in [JobStatus.QUEUED, JobStatus.GENERATING, JobStatus.PAUSED]:
            return {'success': False, 'error': 'Cannot delete active job. Cancel first.'}
        
        # Remove from all collections
        self.queue.jobs.pop(job_id, None)
        self.queue.completed_jobs.pop(job_id, None)
        
        # Remove from user queue
        if job.user_id in self.queue.user_queues:
            try:
                self.queue.user_queues[job.user_id].remove(job_id)
            except ValueError:
                pass
        
        # Delete job file
        job_file = self.job_storage_path / f"{job_id}.json"
        if job_file.exists():
            job_file.unlink()
        
        return {'success': True}
    
    async def _process_job(self, job: MultimodalJob) -> None:
        """Background job processing simulation"""
        try:
            # This is a placeholder for actual job processing
            # In real implementation, this would:
            # 1. Load dataset configuration
            # 2. Generate multimodal dataset
            # 3. Update progress periodically
            # 4. Save results
            
            total_steps = 10
            job.total_steps = total_steps
            
            for step in range(total_steps):
                if job.status == JobStatus.CANCELLED:
                    break
                
                # Simulate processing
                await asyncio.sleep(2)
                
                progress = ((step + 1) / total_steps) * 100
                await self.update_job_progress(
                    job.id,
                    progress=progress,
                    current_step=step + 1,
                    total_steps=total_steps,
                    metrics={'processing_step': f'Step {step + 1}'}
                )
            
            if job.status != JobStatus.CANCELLED:
                await self.queue.complete_job(job.id, success=True)
                
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            await self.queue.complete_job(job.id, success=False, error_message=str(e))
        
        finally:
            # Clean up background task reference
            if job.id in self._background_tasks:
                del self._background_tasks[job.id]
    
    async def _save_job(self, job: MultimodalJob) -> None:
        """Save job to persistent storage"""
        job_file = self.job_storage_path / f"{job.id}.json"
        try:
            with open(job_file, 'w') as f:
                json.dump(job.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save job {job.id}: {e}")
    
    async def _load_job(self, job_id: str) -> Optional[MultimodalJob]:
        """Load job from persistent storage"""
        job_file = self.job_storage_path / f"{job_id}.json"
        if not job_file.exists():
            return None
        
        try:
            with open(job_file, 'r') as f:
                data = json.load(f)
            
            job = MultimodalJob(
                id=data['id'],
                user_id=data['user_id'],
                dataset_id=data['dataset_id'],
                configuration=data['configuration'],
                priority=data['priority']
            )
            
            # Restore state
            job.status = JobStatus(data['status'])
            job.progress = data['progress']
            job.total_steps = data['total_steps']
            job.current_step = data['current_step']
            job.created_at = datetime.fromisoformat(data['created_at'])
            
            if data['started_at']:
                job.started_at = datetime.fromisoformat(data['started_at'])
            if data['completed_at']:
                job.completed_at = datetime.fromisoformat(data['completed_at'])
            
            job.error_message = data.get('error_message')
            job.metrics = data.get('metrics', {})
            
            return job
            
        except Exception as e:
            logger.error(f"Failed to load job {job_id}: {e}")
            return None 