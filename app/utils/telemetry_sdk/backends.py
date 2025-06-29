"""
Telemetry SDK Backends

Backend implementations for logging experiment data to different destinations:
- SQLiteBackend: Local SQLite database
- WandBBackend: Weights & Biases cloud logging
- CSVBackend: Simple CSV file fallback
"""

import json
import csv
import sqlite3
import subprocess
import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod

# Import wandb at module level for testing/mocking purposes
try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


class TelemetryBackend(ABC):
    """Abstract base class for telemetry backends"""
    
    @abstractmethod
    def init_run(self, run_name: str, config: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Initialize a new run and return run_id"""
        pass
    
    @abstractmethod
    def log_metrics(self, run_id: str, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics for a run"""
        pass


class SQLiteBackend(TelemetryBackend):
    """SQLite backend for local experiment tracking"""
    
    def __init__(self, db_path: str = 'telemetry.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Runs table - stores run configuration and metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    git_sha TEXT,
                    config_json TEXT NOT NULL,
                    notes TEXT,
                    pip_freeze TEXT,
                    run_env_vars TEXT
                )
            """)
            
            # Metrics table - stores individual metric values
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs (id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_step ON metrics(run_id, step)")
            
            conn.commit()
    
    def init_run(self, run_name: str, config: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Initialize a new run in SQLite"""
        run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO runs (id, started_at, git_sha, config_json, notes, pip_freeze, run_env_vars)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                timestamp,
                metadata.get('git_sha', 'unknown'),
                json.dumps(config),
                f"Run: {run_name}",
                metadata.get('pip_freeze', ''),
                json.dumps(metadata.get('run_env_vars', {}))
            ))
            
            conn.commit()
        
        logger.info(f"SQLite run initialized: {run_id}")
        return run_id
    
    def log_metrics(self, run_id: str, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to SQLite"""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert each metric as a separate row
            for key, value in metrics.items():
                try:
                    # Convert value to float if possible
                    numeric_value = float(value) if isinstance(value, (int, float)) else 0.0
                    
                    cursor.execute("""
                        INSERT INTO metrics (run_id, step, key, value, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (run_id, step, key, numeric_value, timestamp))
                    
                except (ValueError, TypeError):
                    # Skip non-numeric values for now
                    logger.debug(f"Skipping non-numeric metric: {key}={value}")
                    continue
            
            conn.commit()


class WandBBackend(TelemetryBackend):
    """Weights & Biases backend for cloud experiment tracking"""
    
    def __init__(self):
        self.wandb = None
        self.run = None
        self._init_wandb()
    
    def _init_wandb(self):
        """Initialize wandb if available and not disabled"""
        # Check if wandb is disabled
        if os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1', 'yes'):
            logger.info("WandB disabled via WANDB_DISABLED environment variable")
            return
        
        if wandb is None:
            logger.warning("WandB not available - install with: pip install wandb")
            self.wandb = None
        else:
            self.wandb = wandb
    
    def init_run(self, run_name: str, config: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Initialize a new WandB run"""
        if not self.wandb:
            return "wandb_disabled"
        
        try:
            # Combine config with metadata for wandb
            full_config = {**config, **metadata}
            
            self.run = self.wandb.init(
                project="smollm-finetune",
                name=run_name,
                config=full_config,
                reinit=True
            )
            
            run_id = self.run.id
            logger.info(f"WandB run initialized: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to initialize WandB run: {e}")
            return "wandb_failed"
    
    def log_metrics(self, run_id: str, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to WandB"""
        if not self.wandb or not self.run:
            return
        
        try:
            # Add step to metrics
            wandb_metrics = {**metrics, "step": step}
            self.wandb.log(wandb_metrics)
            
        except Exception as e:
            logger.error(f"Failed to log metrics to WandB: {e}")


class CSVBackend(TelemetryBackend):
    """CSV backend for simple file-based logging"""
    
    def __init__(self, csv_path: str = 'telemetry_metrics.csv'):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialized = False
    
    def init_run(self, run_name: str, config: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Initialize CSV logging"""
        run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Write header if file doesn't exist
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['run_id', 'timestamp', 'step', 'metric_key', 'metric_value', 'config_json'])
        
        # Write run initialization row
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                run_id,
                datetime.now(timezone.utc).isoformat(),
                -1,  # Special step for initialization
                'run_init',
                json.dumps(config),
                json.dumps(metadata)
            ])
        
        self.initialized = True
        logger.info(f"CSV backend initialized: {run_id}")
        return run_id
    
    def log_metrics(self, run_id: str, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to CSV"""
        if not self.initialized:
            logger.warning("CSV backend not initialized")
            return
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write each metric as a separate row
            for key, value in metrics.items():
                writer.writerow([run_id, timestamp, step, key, value, '']) 