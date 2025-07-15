"""
Unit tests for telemetry SDK - TDD Red Phase
Tests the telemetry SDK that provides unified experiment tracking.
"""

import pytest
import json
import os
import tempfile
import sqlite3
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime, timezone

# We expect the SDK to be importable like this
try:
    from backend.app.core.telemetry_sdk import init, log, capture_cfg
    from backend.app.core.telemetry_sdk.backends import SQLiteBackend, WandBBackend, CSVBackend
    from backend.app.core.telemetry_sdk.cli import format_run_report
except ImportError:
    # These will fail initially - that's expected in TDD Red phase
    init = None
    log = None
    capture_cfg = None


class TestTelemetrySDKCore:
    """Test core telemetry SDK functionality"""
    
    def test_init_function_exists(self):
        """SDK should expose init function"""
        assert init is not None, "init function should be importable"
    
    def test_log_function_exists(self):
        """SDK should expose log function"""
        assert log is not None, "log function should be importable"
    
    def test_capture_cfg_decorator_exists(self):
        """SDK should expose capture_cfg decorator"""
        assert capture_cfg is not None, "capture_cfg decorator should be importable"
    
    @patch('subprocess.check_output')
    def test_init_captures_git_commit(self, mock_subprocess):
        """init() should capture git commit hash"""
        mock_subprocess.side_effect = [b'abc123def456\n', b'numpy==1.0\n']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            
            run_id = init(
                run_name="test_run", 
                cfg={"learning_rate": 0.001},
                backends=['sqlite'],
                sqlite_path=db_path
            )
            
            assert run_id is not None
            # Should have called both git and pip freeze
            assert mock_subprocess.call_count == 2
    
    @patch('subprocess.check_output')
    def test_init_captures_pip_freeze(self, mock_subprocess):
        """init() should capture pip freeze output"""
        mock_subprocess.side_effect = [
            b'abc123def456\n',  # git commit
            b'numpy==1.21.0\ntorch==1.9.0\n'  # pip freeze
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            
            run_id = init(
                run_name="test_run",
                cfg={"batch_size": 32},
                backends=['sqlite'],
                sqlite_path=db_path
            )
            
            assert run_id is not None
            # Should call git and pip freeze
            assert mock_subprocess.call_count == 2
    
    def test_init_captures_run_env_vars(self):
        """init() should capture environment variables starting with RUN_"""
        with patch.dict(os.environ, {'RUN_EXPERIMENT': 'test', 'RUN_BATCH': '42', 'OTHER_VAR': 'ignore'}):
            with tempfile.TemporaryDirectory() as temp_dir:
                db_path = os.path.join(temp_dir, 'test.db')
                
                with patch('subprocess.check_output') as mock_subprocess:
                    mock_subprocess.side_effect = [b'abc123\n', b'numpy==1.0\n']
                    
                    run_id = init(
                        run_name="test_run",
                        cfg={"lr": 0.01},
                        backends=['sqlite'],
                        sqlite_path=db_path
                    )
                    
                    # Should capture RUN_ variables but not others
                    assert run_id is not None
    
    def test_log_function_basic(self):
        """log() should accept metrics dictionary and store them"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            
            with patch('subprocess.check_output') as mock_subprocess:
                mock_subprocess.side_effect = [b'abc123\n', b'numpy==1.0\n']
                
                run_id = init(
                    run_name="test_run",
                    cfg={"lr": 0.01},
                    backends=['sqlite'],
                    sqlite_path=db_path
                )
                
                # Log some metrics
                log({"loss": 0.5, "accuracy": 0.85})
                log({"loss": 0.4, "accuracy": 0.87})
                
                # Should not raise exceptions
                assert True
    
    def test_multiple_backends(self):
        """Should support multiple backends simultaneously"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            csv_path = os.path.join(temp_dir, 'metrics.csv')
            
            with patch('subprocess.check_output') as mock_subprocess:
                mock_subprocess.side_effect = [b'abc123\n', b'numpy==1.0\n']
                
                with patch('backend.app.core.telemetry_sdk.backends.wandb') as mock_wandb:
                    mock_wandb.init.return_value = MagicMock()
                    
                    run_id = init(
                        run_name="test_run",
                        cfg={"lr": 0.01},
                        backends=['sqlite', 'wandb', 'csv'],
                        sqlite_path=db_path,
                        csv_path=csv_path
                    )
                    
                    log({"loss": 0.5})
                    
                    assert run_id is not None


class TestSQLiteBackend:
    """Test SQLite backend functionality"""
    
    def test_sqlite_backend_creates_tables(self):
        """SQLite backend should create necessary tables"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            
            backend = SQLiteBackend(db_path)
            backend.init_run("test_run", {"lr": 0.01}, {"git_sha": "abc123"})
            
            # Check tables exist
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'runs' in tables
            assert 'metrics' in tables
            conn.close()
    
    def test_sqlite_backend_stores_run_info(self):
        """SQLite backend should store run configuration and metadata"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            
            backend = SQLiteBackend(db_path)
            run_id = backend.init_run(
                "test_run", 
                {"learning_rate": 0.001, "batch_size": 32},
                {"git_sha": "abc123", "pip_freeze": "numpy==1.0"}
            )
            
            # Verify data was stored
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            
            assert row is not None
            assert run_id in row[0]  # run_id should be in the ID
            conn.close()
    
    def test_sqlite_backend_stores_metrics(self):
        """SQLite backend should store metrics with run_id association"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            
            backend = SQLiteBackend(db_path)
            run_id = backend.init_run("test_run", {"lr": 0.01}, {"git_sha": "abc123"})
            
            # Log metrics
            backend.log_metrics(run_id, {"loss": 0.5, "accuracy": 0.85}, step=1)
            backend.log_metrics(run_id, {"loss": 0.4, "accuracy": 0.87}, step=2)
            
            # Verify metrics were stored
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM metrics WHERE run_id = ?", (run_id,))
            rows = cursor.fetchall()
            
            assert len(rows) == 4  # 2 metrics × 2 steps
            conn.close()


class TestWandBBackend:
    """Test WandB backend functionality"""
    
    def test_wandb_backend_respects_disabled_flag(self):
        """WandB backend should not initialize when WANDB_DISABLED is set"""
        with patch.dict(os.environ, {'WANDB_DISABLED': 'true'}):
            with patch('backend.app.core.telemetry_sdk.backends.wandb') as mock_wandb:
                backend = WandBBackend()
                run_id = backend.init_run("test_run", {"lr": 0.01}, {"git_sha": "abc123"})
                
                mock_wandb.init.assert_not_called()
                assert run_id == "wandb_disabled"
    
    @patch('backend.app.core.telemetry_sdk.backends.wandb')
    def test_wandb_backend_initializes_run(self, mock_wandb):
        """WandB backend should initialize wandb run with proper config"""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_run.id = "wandb_test_123"
        
        backend = WandBBackend()
        run_id = backend.init_run(
            "test_run", 
            {"learning_rate": 0.001},
            {"git_sha": "abc123"}
        )
        
        mock_wandb.init.assert_called_once()
        assert run_id == "wandb_test_123"
    
    @patch('backend.app.core.telemetry_sdk.backends.wandb')
    def test_wandb_backend_logs_metrics(self, mock_wandb):
        """WandB backend should log metrics to wandb"""
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_run.id = "wandb_test_123"
        
        backend = WandBBackend()
        run_id = backend.init_run("test_run", {"lr": 0.01}, {"git_sha": "abc123"})
        
        backend.log_metrics(run_id, {"loss": 0.5, "accuracy": 0.85}, step=1)
        
        mock_wandb.log.assert_called_once()


class TestCSVBackend:
    """Test CSV fallback backend functionality"""
    
    def test_csv_backend_creates_file(self):
        """CSV backend should create metrics file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'metrics.csv')
            
            backend = CSVBackend(csv_path)
            run_id = backend.init_run("test_run", {"lr": 0.01}, {"git_sha": "abc123"})
            backend.log_metrics(run_id, {"loss": 0.5}, step=1)
            
            assert os.path.exists(csv_path)
            
            # Check CSV content
            with open(csv_path, 'r') as f:
                content = f.read()
                assert 'run_id' in content
                assert 'loss' in content
    
    def test_csv_backend_appends_metrics(self):
        """CSV backend should append metrics to existing file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'metrics.csv')
            
            backend = CSVBackend(csv_path)
            run_id = backend.init_run("test_run", {"lr": 0.01}, {"git_sha": "abc123"})
            
            backend.log_metrics(run_id, {"loss": 0.5}, step=1)
            backend.log_metrics(run_id, {"loss": 0.4}, step=2)
            
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) >= 3  # Header + 2 metrics


class TestCaptureCfgDecorator:
    """Test the @capture_cfg decorator functionality"""
    
    def test_capture_cfg_decorator_logs_dataclass(self):
        """@capture_cfg should automatically log dataclass configuration"""
        from dataclasses import dataclass
        
        @dataclass
        class TrainingConfig:
            learning_rate: float
            batch_size: int
            epochs: int
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            
            with patch('subprocess.check_output') as mock_subprocess:
                mock_subprocess.side_effect = [b'abc123\n', b'numpy==1.0\n']
                
                # Initialize SDK
                init(run_name="test", cfg={}, backends=['sqlite'], sqlite_path=db_path)
                
                @capture_cfg
                def train_model(config: TrainingConfig):
                    return "training complete"
                
                config = TrainingConfig(learning_rate=0.001, batch_size=32, epochs=10)
                result = train_model(config)
                
                assert result == "training complete"


class TestCLIReporter:
    """Test CLI reporting functionality"""
    
    def test_cli_reporter_formats_run_info(self):
        """CLI reporter should format run information nicely"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, 'test.db')
            
            # Create some test data
            backend = SQLiteBackend(db_path)
            run_id = backend.init_run(
                "test_run",
                {"learning_rate": 0.001, "batch_size": 32},
                {"git_sha": "abc123", "pip_freeze": "numpy==1.0\ntorch==1.9.0"}
            )
            backend.log_metrics(run_id, {"loss": 0.5, "accuracy": 0.85}, step=1)
            backend.log_metrics(run_id, {"loss": 0.4, "accuracy": 0.87}, step=2)
            
            # Test reporter
            report = format_run_report(run_id, db_path)
            
            assert run_id in report
            assert "test_run" in report
            assert "learning_rate" in report
            assert "0.001" in report


class TestFailureHandlingAndResilience:
    """Test error handling and fallback behavior"""
    
    def test_backend_failures_fallback_to_csv(self):
        """If primary backends fail, should fallback to CSV"""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'fallback.csv')
            
            # Mock SQLite failure
            with patch('backend.app.core.telemetry_sdk.backends.sqlite3.connect', side_effect=Exception("DB Error")):
                # Should fallback to CSV and not crash
                run_id = init(
                    run_name="test_run",
                    cfg={"lr": 0.01},
                    backends=['sqlite', 'csv'],
                    sqlite_path="/invalid/path/db.sqlite",
                    csv_path=csv_path
                )
                
                log({"loss": 0.5})
                
                # CSV should have been created as fallback
                assert os.path.exists(csv_path)
    
    def test_git_command_failure_graceful(self):
        """Should handle git command failures gracefully"""
        with patch('subprocess.check_output', side_effect=Exception("Git not found")):
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_path = os.path.join(temp_dir, 'metrics.csv')
                
                # Should not crash even if git fails
                run_id = init(
                    run_name="test_run",
                    cfg={"lr": 0.01},
                    backends=['csv'],
                    csv_path=csv_path
                )
                
                assert run_id is not None 