from celery import Celery
from backend.app.config import settings

# Create Celery app
celery_app = Celery(
    'character_devkit',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['backend.app.tasks']
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

# Task routing
celery_app.conf.task_routes = {
    'backend.app.tasks.dataset_generation.*': {'queue': 'dataset'},
    'backend.app.tasks.training.*': {'queue': 'training'},
    'backend.app.tasks.export.*': {'queue': 'export'},
} 