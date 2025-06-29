"""
Training Telemetry SDK

Lightweight package for experiment tracking with auto-capture of git commits,
environment variables, and pip dependencies. Supports multiple backends:
SQLite, WandB, and CSV fallback.

Usage:
    import telemetry_sdk
    
    run_id = telemetry_sdk.init("my_experiment", {"lr": 0.001, "batch_size": 32})
    
    for step in range(100):
        loss = train_step()
        telemetry_sdk.log({"loss": loss, "step": step})
"""

import atexit
import logging
import functools
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from .backends import SQLiteBackend, WandBBackend, CSVBackend

logger = logging.getLogger(__name__)

# Global state
_current_run_id: Optional[str] = None
_active_backends: List[Any] = []
_metrics_buffer: List[Dict[str, Any]] = []


def init(run_name: str, cfg: Dict[str, Any], 
         backends: List[str] = ['sqlite', 'wandb'],
         dataset_paths: Optional[List[str]] = None,
         **backend_kwargs) -> str:
    """
    Initialize telemetry tracking for an experiment run.
    
    Args:
        run_name: Human-readable name for this run
        cfg: Configuration dictionary to log
        backends: List of backend names to use ['sqlite', 'wandb', 'csv']
        dataset_paths: Optional list of dataset paths to attach to this run
        **backend_kwargs: Backend-specific configuration
        
    Returns:
        run_id: Unique identifier for this run
    """
    global _current_run_id, _active_backends, _metrics_buffer
    
    # Clear any previous state
    _active_backends = []
    _metrics_buffer = []
    
    # Capture environment metadata
    metadata = _capture_environment_metadata()
    
    # Initialize backends
    run_ids = []
    
    for backend_name in backends:
        try:
            if backend_name == 'sqlite':
                backend = SQLiteBackend(backend_kwargs.get('sqlite_path', 'telemetry.db'))
            elif backend_name == 'wandb':
                backend = WandBBackend()
            elif backend_name == 'csv':
                backend = CSVBackend(backend_kwargs.get('csv_path', 'telemetry_metrics.csv'))
            else:
                logger.warning(f"Unknown backend: {backend_name}")
                continue
                
            run_id = backend.init_run(run_name, cfg, metadata)
            _active_backends.append(backend)
            run_ids.append(run_id)
            
        except Exception as e:
            logger.error(f"Failed to initialize {backend_name} backend: {e}")
            # Fallback to CSV if other backends fail
            if backend_name != 'csv' and 'csv' not in backends:
                try:
                    csv_backend = CSVBackend(backend_kwargs.get('csv_path', 'telemetry_fallback.csv'))
                    fallback_run_id = csv_backend.init_run(run_name, cfg, metadata)
                    _active_backends.append(csv_backend)
                    run_ids.append(fallback_run_id)
                    logger.info("Fallback to CSV backend activated")
                except Exception as fallback_error:
                    logger.error(f"Fallback CSV backend also failed: {fallback_error}")
    
    # Use the first successful run_id as the primary
    _current_run_id = run_ids[0] if run_ids else f"{run_name}_no_backend"
    
    # Register cleanup on exit
    atexit.register(_flush_metrics_on_exit)
    
    # Attach dataset manifests if provided
    if dataset_paths:
        try:
            attach_datasets(dataset_paths)
        except Exception as e:
            logger.warning(f"Failed to attach datasets to telemetry: {e}")
    
    logger.info(f"Telemetry initialized for run: {_current_run_id}")
    return _current_run_id


def log(metrics: Dict[str, Union[float, int, str]], step: Optional[int] = None) -> None:
    """
    Log metrics for the current run.
    
    Args:
        metrics: Dictionary of metric name -> value
        step: Optional step number (auto-incremented if not provided)
    """
    global _current_run_id, _active_backends, _metrics_buffer
    
    if not _current_run_id:
        logger.warning("No active run - call init() first")
        return
    
    # Auto-increment step if not provided
    if step is None:
        step = len(_metrics_buffer)
    
    # Buffer metrics
    metrics_entry = {
        'step': step,
        'metrics': metrics.copy(),
        'run_id': _current_run_id
    }
    _metrics_buffer.append(metrics_entry)
    
    # Log to all active backends
    for backend in _active_backends:
        try:
            backend.log_metrics(_current_run_id, metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log to backend {backend.__class__.__name__}: {e}")


def attach_datasets(dataset_paths: List[str]) -> None:
    """
    Attach dataset manifests to the current telemetry run.
    
    Args:
        dataset_paths: List of dataset paths to attach to current run
    """
    try:
        from ..dataset_versioning import get_dataset_manifest_for_run
        
        manifests = get_dataset_manifest_for_run(dataset_paths)
        
        # Log dataset information to telemetry
        log({
            "datasets": manifests,
            "dataset_count": len(manifests),
            "dataset_attachment_timestamp": _get_current_timestamp()
        })
        
        logger.info(f"Attached {len(manifests)} dataset manifests to telemetry")
        
    except ImportError as e:
        logger.warning(f"Dataset versioning not available: {e}")
    except Exception as e:
        logger.error(f"Failed to attach datasets to telemetry: {e}")
        raise


def capture_cfg(func):
    """
    Decorator to automatically capture and log function configuration.
    Useful for dataclass configs passed to training functions.
    
    Usage:
        @capture_cfg
        def train_model(config: TrainingConfig):
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract dataclass configs from arguments
        for arg in args:
            if hasattr(arg, '__dataclass_fields__'):
                # It's a dataclass - log its fields
                config_dict = {
                    field.name: getattr(arg, field.name) 
                    for field in arg.__dataclass_fields__.values()
                }
                log({"config_captured": config_dict})
                break
        
        return func(*args, **kwargs)
    
    return wrapper


def _get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _capture_environment_metadata() -> Dict[str, Any]:
    """Capture git commit, pip freeze, and RUN_ environment variables."""
    import subprocess
    import os
    import sys
    
    metadata = {}
    
    # Capture git commit
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        metadata['git_sha'] = git_sha
    except Exception as e:
        logger.warning(f"Could not capture git commit: {e}")
        metadata['git_sha'] = "unknown"
    
    # Capture pip freeze
    try:
        pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        metadata['pip_freeze'] = pip_freeze
    except Exception as e:
        logger.warning(f"Could not capture pip freeze: {e}")
        metadata['pip_freeze'] = "unknown"
    
    # Capture RUN_ environment variables
    run_env_vars = {k: v for k, v in os.environ.items() if k.startswith('RUN_')}
    metadata['run_env_vars'] = run_env_vars
    
    return metadata


def _flush_metrics_on_exit():
    """Flush any buffered metrics on process exit."""
    if _metrics_buffer:
        logger.info(f"Flushing {len(_metrics_buffer)} buffered metrics on exit")
        # Final flush is handled by individual backends 