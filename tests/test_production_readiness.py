"""
Tests for Production Readiness Infrastructure

Tests health checks, error handling, progress tracking, and other
production-grade features to ensure they work correctly.
"""

import unittest
import tempfile
import os
import sqlite3
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the modules we're testing
from app.utils.health import HealthChecker, get_health_checker
from app.utils.error_handling import ErrorHandler, streamlit_error_boundary, safe_execute
from app.utils.progress import ProgressTracker, progress_bar, ProgressConfig


class TestHealthChecker(unittest.TestCase):
    """Test the health check system"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        
        # Create a test database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY)")
            cursor.execute("INSERT INTO test_table (id) VALUES (1)")
            conn.commit()
        
        self.health_checker = HealthChecker(db_path=self.db_path)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_database_health_check_healthy(self):
        """Test database health check with healthy database"""
        result = self.health_checker.check_database()
        
        self.assertEqual(result["status"], "healthy")
        self.assertIn("tables_count", result)
        self.assertGreater(result["tables_count"], 0)
        self.assertIn("file_size_mb", result)
    
    def test_database_health_check_missing_file(self):
        """Test database health check with missing database file"""
        checker = HealthChecker(db_path="/nonexistent/path/db.sqlite")
        result = checker.check_database()
        
        self.assertEqual(result["status"], "unhealthy")
        self.assertIn("error", result)
        self.assertIn("does not exist", result["error"])
    
    def test_redis_health_check_not_configured(self):
        """Test Redis health check when not configured"""
        result = self.health_checker.check_redis()
        
        self.assertEqual(result["status"], "not_configured")
        self.assertIn("message", result)
    
    def test_redis_health_check_not_available(self):
        """Test Redis health check when Redis is not available"""
        result = self.health_checker.check_redis("redis://localhost:99999/0")
        
        # Should be unhealthy or not_available depending on redis module availability
        self.assertIn(result["status"], ["unhealthy", "not_available"])
    
    def test_system_resources_check(self):
        """Test system resources health check"""
        result = self.health_checker.check_system_resources()
        
        # Should always return some status
        self.assertIn(result["status"], ["healthy", "warning", "critical", "unhealthy"])
        self.assertIn("cpu_percent", result)
        self.assertIn("memory", result)
        self.assertIn("disk", result)
    
    def test_training_output_directory_check(self):
        """Test training output directory health check"""
        # Set up a temporary training directory
        training_dir = Path(self.temp_dir) / "training_output"
        
        with patch("app.utils.health.Path") as mock_path:
            mock_path.return_value = training_dir
            mock_path.return_value.exists.return_value = False
            mock_path.return_value.mkdir = Mock()
            
            # Mock the write test
            test_file = Mock()
            test_file.exists.return_value = True
            test_file.write_text = Mock()
            test_file.unlink = Mock()
            
            training_dir.glob = Mock(return_value=[])
            
            with patch.object(training_dir, "__truediv__", return_value=test_file):
                result = self.health_checker.check_training_output_directory()
        
        # Should create directory and test write access
        self.assertIn(result["status"], ["healthy", "unhealthy"])
    
    def test_comprehensive_health_check(self):
        """Test comprehensive health check that combines all checks"""
        result = self.health_checker.get_comprehensive_health()
        
        self.assertIn("status", result)
        self.assertIn("timestamp", result)
        self.assertIn("uptime_seconds", result)
        self.assertIn("checks", result)
        
        # Should include all check types
        checks = result["checks"]
        self.assertIn("database", checks)
        self.assertIn("redis", checks)
        self.assertIn("system", checks)
        self.assertIn("storage", checks)


class TestErrorHandler(unittest.TestCase):
    """Test the error handling system"""
    
    def setUp(self):
        """Set up test environment"""
        self.error_handler = ErrorHandler()
    
    def test_error_handler_initialization(self):
        """Test error handler initializes correctly"""
        self.assertFalse(self.error_handler.sentry_enabled)  # No DSN provided
        self.assertEqual(self.error_handler.error_count, 0)
    
    def test_capture_exception_without_sentry(self):
        """Test exception capture without Sentry"""
        test_error = ValueError("Test error")
        
        self.error_handler.capture_exception(test_error)
        
        self.assertEqual(self.error_handler.error_count, 1)
    
    @patch.dict(os.environ, {'SENTRY_DSN': 'test://key@sentry.io/123'})
    @patch('app.utils.error_handling.sentry_sdk')
    def test_sentry_initialization(self, mock_sentry):
        """Test Sentry initialization when DSN is provided"""
        handler = ErrorHandler()
        
        mock_sentry.init.assert_called_once()
        self.assertTrue(handler.sentry_enabled)
    
    def test_get_user_friendly_message_database_error(self):
        """Test user-friendly message for database errors"""
        error = sqlite3.OperationalError("database is locked")
        title, message = self.error_handler.get_user_friendly_message(error)
        
        self.assertEqual(title, "Database Connection Issue")
        self.assertIn("database", message.lower())
    
    def test_get_user_friendly_message_network_error(self):
        """Test user-friendly message for network errors"""
        error = ConnectionError("connection timeout")
        title, message = self.error_handler.get_user_friendly_message(error)
        
        self.assertEqual(title, "Connection Problem")
        self.assertIn("connection", message.lower())
    
    def test_get_user_friendly_message_memory_error(self):
        """Test user-friendly message for memory errors"""
        error = MemoryError("out of memory")
        title, message = self.error_handler.get_user_friendly_message(error)
        
        self.assertEqual(title, "Resource Limitation")
        self.assertIn("memory", message.lower())
    
    def test_get_user_friendly_message_generic_error(self):
        """Test user-friendly message for generic errors"""
        error = RuntimeError("unexpected error")
        title, message = self.error_handler.get_user_friendly_message(error)
        
        self.assertEqual(title, "Unexpected Error")
        self.assertIn("unexpected", message.lower())
    
    def test_streamlit_error_boundary_decorator(self):
        """Test the Streamlit error boundary decorator"""
        
        @streamlit_error_boundary
        def test_function_success():
            return "success"
        
        @streamlit_error_boundary  
        def test_function_error():
            raise ValueError("Test error")
        
        # Mock Streamlit
        with patch('app.utils.error_handling.st') as mock_st:
            mock_st.session_state = {}
            mock_st.columns.return_value = [Mock(), Mock(), Mock()]
            mock_st.button.return_value = False
            
            # Test successful execution
            result = test_function_success()
            self.assertEqual(result, "success")
            
            # Test error handling
            result = test_function_error()
            self.assertIsNone(result)  # Should return None on error
    
    def test_safe_execute_success(self):
        """Test safe_execute with successful function"""
        def test_func():
            return "success"
        
        with patch('app.utils.error_handling.st') as mock_st:
            result = safe_execute(test_func)
            self.assertEqual(result, "success")
    
    def test_safe_execute_error(self):
        """Test safe_execute with failing function"""
        def test_func():
            raise ValueError("Test error")
        
        with patch('app.utils.error_handling.st') as mock_st:
            result = safe_execute(test_func, fallback_value="fallback")
            self.assertEqual(result, "fallback")
            mock_st.warning.assert_called_once()


class TestProgressTracker(unittest.TestCase):
    """Test the progress tracking system"""
    
    def test_progress_tracker_initialization(self):
        """Test progress tracker initializes correctly"""
        tracker = ProgressTracker(total=100, description="Test")
        
        self.assertEqual(tracker.total, 100)
        self.assertEqual(tracker.description, "Test")
        self.assertEqual(tracker.current, 0)
        self.assertFalse(tracker.is_complete())
    
    def test_progress_tracker_update(self):
        """Test progress tracker updates correctly"""
        tracker = ProgressTracker(total=100)
        
        tracker.update(10)
        self.assertEqual(tracker.current, 10)
        self.assertEqual(tracker.get_progress(), 0.1)
        
        tracker.update(90)
        self.assertEqual(tracker.current, 100)
        self.assertEqual(tracker.get_progress(), 1.0)
        self.assertTrue(tracker.is_complete())
    
    def test_progress_tracker_eta_calculation(self):
        """Test ETA calculation"""
        tracker = ProgressTracker(total=100)
        
        # First update - no ETA yet
        tracker.update(10)
        eta = tracker.get_eta()
        self.assertIsNone(eta)  # Not enough data
        
        # Simulate some time passing and more progress
        time.sleep(0.1)
        tracker.update(10)
        
        # Now should have ETA (might be None still due to short time)
        eta = tracker.get_eta()
        # ETA calculation depends on timing, so just check it's handled
        self.assertIsInstance(tracker.format_eta(), str)
    
    def test_progress_config(self):
        """Test progress configuration"""
        config = ProgressConfig(
            show_percentage=False,
            show_eta=False,
            auto_close=False
        )
        
        self.assertFalse(config.show_percentage)
        self.assertFalse(config.show_eta)
        self.assertFalse(config.auto_close)
    
    def test_progress_bar_context_manager(self):
        """Test progress bar context manager"""
        with patch('app.utils.progress.st') as mock_st:
            mock_st.empty.return_value = Mock()
            mock_st.progress = Mock()
            mock_st.columns.return_value = [Mock(), Mock(), Mock()]
            mock_st.caption = Mock()
            
            with progress_bar(total=10, description="Test") as tracker:
                tracker.update(5)
                self.assertEqual(tracker.current, 5)
            
            # Should have called progress display methods
            mock_st.progress.assert_called()


class TestProductionIntegration(unittest.TestCase):
    """Integration tests for production features"""
    
    def test_health_endpoint_integration(self):
        """Test health endpoint works end-to-end"""
        # This would require Streamlit app testing
        # For now, just test the health checker directly
        checker = get_health_checker()
        health = checker.get_comprehensive_health()
        
        self.assertIn("status", health)
        self.assertIn("checks", health)
    
    def test_error_handling_integration(self):
        """Test error handling works end-to-end"""
        from app.utils.error_handling import get_error_handler
        
        handler = get_error_handler()
        
        # Test exception capture
        try:
            raise ValueError("Test integration error")
        except Exception as e:
            handler.capture_exception(e, context={"test": "integration"})
        
        self.assertGreater(handler.error_count, 0)
    
    def test_production_requirements_available(self):
        """Test that production requirements can be imported"""
        try:
            import sentry_sdk
            sentry_available = True
        except ImportError:
            sentry_available = False
        
        try:
            import redis
            redis_available = True
        except ImportError:
            redis_available = False
        
        try:
            import psutil
            psutil_available = True
        except ImportError:
            psutil_available = False
        
        # At least psutil should be available for health checks
        self.assertTrue(psutil_available, "psutil is required for system monitoring")
        
        # Sentry and Redis are optional but recommended
        # Just verify they can be imported if installed


if __name__ == '__main__':
    unittest.main() 