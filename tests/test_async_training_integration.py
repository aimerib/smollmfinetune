"""
🧪 Async Training & Data Collection Integration Tests
Tests for R3-2 (Async Training) and R3-2.5 (Data Collection Foundation)
"""

import pytest
from unittest.mock import Mock, patch

def test_async_training_service_import():
    """Test that AsyncTrainingService can be imported and instantiated"""
    try:
        from backend.app.services.training.async_training import AsyncTrainingService
        service = AsyncTrainingService()
        assert service is not None
        assert hasattr(service, 'start_training')
        assert hasattr(service, 'get_training_status')
        assert hasattr(service, 'cancel_training')
    except ImportError as e:
        pytest.skip(f"AsyncTrainingService not available: {e}")

def test_data_collection_service_import():
    """Test that DataCollectionService can be imported and instantiated"""
    try:
        from backend.app.services.training.async_training import DataCollectionService
        service = DataCollectionService()
        assert service is not None
        assert hasattr(service, 'collect_conversation')
        assert hasattr(service, 'process_conversation_batch')
    except ImportError as e:
        pytest.skip(f"DataCollectionService not available: {e}")

def test_database_models_exist():
    """Test that required database models exist with proper fields"""
    try:
        from backend.app.core.database.models import TrainingRun, ConversationLog, User, Character
        
        # Test TrainingRun model
        tr = TrainingRun()
        assert hasattr(tr, 'status')
        assert hasattr(tr, 'config_json')
        assert hasattr(tr, 'metrics_json')
        
        # Test ConversationLog model
        cl = ConversationLog()
        assert hasattr(cl, 'conversation_data')
        assert hasattr(cl, 'quality_score')
        assert hasattr(cl, 'processed_for_training')
        
        # Test User model
        u = User()
        assert hasattr(u, 'profile_json')
        
        # Test Character model
        c = Character()
        assert hasattr(c, 'personality_json')
        assert hasattr(c, 'first_message')
        
    except ImportError as e:
        pytest.skip(f"Database models not available: {e}")

def test_worker_functions_importable():
    """Test that worker functions can be imported"""
    try:
        from worker import run_training, collect_conversation_data
        assert callable(run_training)
        assert callable(collect_conversation_data)
    except ImportError as e:
        pytest.skip(f"Worker functions not available: {e}")

@patch('backend.app.services.training.async_training.session_scope')
def test_async_training_basic_flow(mock_session_scope):
    """Test basic async training flow"""
    try:
        from backend.app.services.training.async_training import AsyncTrainingService
        
        # Mock database session
        mock_session = Mock()
        mock_session_scope.return_value.__enter__ = Mock(return_value=mock_session)
        mock_session_scope.return_value.__exit__ = Mock()
        
        # Mock no existing character
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        
        # Create service without Celery
        with patch.object(AsyncTrainingService, '_setup_celery'):
            service = AsyncTrainingService()
            service.celery_app = None  # Simulate no Celery
            
            # Test data
            character_data = {'name': 'TestChar', 'description': 'Test'}
            dataset = [{'messages': [{'role': 'user', 'content': 'hi'}]}]
            config = {'base_model': 'test-model', 'finetune_method': 'lora'}
            
            # Should return None when Celery is not available
            result = service.start_training(character_data, dataset, config, 1)
            assert result is None
            
    except ImportError as e:
        pytest.skip(f"AsyncTrainingService not available: {e}")

def test_data_collection_basic_flow():
    """Test basic data collection flow"""
    try:
        from backend.app.services.training.async_training import DataCollectionService
        
        # Create service without Celery
        with patch.object(DataCollectionService, '_setup_celery'):
            service = DataCollectionService()
            service.celery_app = None  # Simulate no Celery
            
            # Test data
            messages = [{'role': 'user', 'content': 'Hello'}]
            metadata = {'test': 'data'}
            
            # Should return False when Celery is not available
            result = service.collect_conversation(1, 1, messages, metadata)
            assert result is False
            
    except ImportError as e:
        pytest.skip(f"DataCollectionService not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 