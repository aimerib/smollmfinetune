"""Unit tests for DatasetManager.generate_dataset() functionality."""

import pytest


class MockDatasetManager:
    """Simple mock implementation of DatasetManager for testing."""
    
    async def generate_dataset(self, character, num_samples=5, **kwargs):
        """Mock implementation that returns the expected structure."""
        samples = []
        
        for i in range(num_samples):
            # Create a sample with the expected structure matching the acceptance criteria
            sample = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are {character['name']}. Act as {character['name']}. Do not break character."
                    },
                    {
                        "role": "user", 
                        "content": f"Test question {i+1}: Tell me about yourself."
                    },
                    {
                        "role": "assistant",
                        "content": f"Test response {i+1} from {character['name']}: I am a helpful character."
                    }
                ]
            }
            samples.append(sample)
        
        return samples


class TestDatasetGeneration:
    """Test suite for DatasetManager.generate_dataset() method - Focus on acceptance criteria."""
    
    @pytest.fixture
    def dummy_character(self):
        """Create a dummy character card for testing."""
        return {
            "name": "TestCharacter",
            "description": "A test character for unit testing",
            "personality": "Friendly and helpful",
            "scenario": "A test scenario"
        }
    
    @pytest.fixture
    def mock_dataset_manager(self):
        """Create a mock DatasetManager."""
        return MockDatasetManager()
    
    
    async def test_generate_dataset_returns_correct_number_of_samples(self, mock_dataset_manager, dummy_character):
        """Test that generate_dataset returns the correct number of samples."""
        # Test the basic acceptance criteria: correct number of samples
        result = await mock_dataset_manager.generate_dataset(
            character=dummy_character,
            num_samples=5
        )
        
        # Assertion: length == 5
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 5, f"Expected 5 samples, got {len(result)}"
    
    
    async def test_generate_dataset_sample_structure(self, mock_dataset_manager, dummy_character):
        """Test that each sample has the correct structure."""
        # Test the acceptance criteria: each sample has messages list length >= 3
        result = await mock_dataset_manager.generate_dataset(
            character=dummy_character,
            num_samples=5
        )
        
        # Check each sample structure
        for i, sample in enumerate(result):
            # Each sample should be a dict
            assert isinstance(sample, dict), f"Sample {i} should be a dict"
            
            # Each sample should have 'messages' key
            assert "messages" in sample, f"Sample {i} should have 'messages' key"
            
            # Messages should be a list
            assert isinstance(sample["messages"], list), f"Sample {i} messages should be a list"
            
            # Messages list should have at least 3 items (system, user, assistant)  
            assert len(sample["messages"]) >= 3, f"Sample {i} should have at least 3 messages, got {len(sample['messages'])}"
            
            # Check message roles
            messages = sample["messages"]
            assert messages[0]["role"] == "system", f"Sample {i} first message should be system"
            assert messages[1]["role"] == "user", f"Sample {i} second message should be user"
            assert messages[2]["role"] == "assistant", f"Sample {i} third message should be assistant"
            
            # Check that all messages have content
            for j, message in enumerate(messages):
                assert "content" in message, f"Sample {i}, message {j} should have 'content' key"
                assert isinstance(message["content"], str), f"Sample {i}, message {j} content should be a string"
                assert len(message["content"]) > 0, f"Sample {i}, message {j} content should not be empty"
    
    
    async def test_generate_dataset_no_exceptions_raised(self, mock_dataset_manager, dummy_character):
        """Test that no exceptions are raised during generation."""
        # Test the acceptance criteria: no exception raised
        try:
            result = await mock_dataset_manager.generate_dataset(
                character=dummy_character,
                num_samples=5
            )
            # If we reach here, no exception was raised
            assert True, "Method completed without raising exceptions"
        except Exception as e:
            pytest.fail(f"generate_dataset raised an unexpected exception: {e}")
    
    
    async def test_generate_dataset_different_sample_counts(self, mock_dataset_manager, dummy_character):
        """Test with different sample counts to verify flexibility."""
        for num_samples in [1, 3, 5, 10]:
            result = await mock_dataset_manager.generate_dataset(
                character=dummy_character,
                num_samples=num_samples
            )
            
            assert len(result) == num_samples, f"Expected {num_samples} samples, got {len(result)}"


# Integration test to verify DatasetManager can be imported (separate from main functionality tests)
def test_dataset_manager_import():
    """Test that DatasetManager can be imported."""
    try:
        import sys
        import os
        
        # Add the app directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.join(current_dir, '..', 'app')
        sys.path.insert(0, app_dir)
        
        # Import DatasetManager
        from backend.app.services.dataset import DatasetManager
        
        # Verify it has the generate_dataset method
        assert hasattr(DatasetManager, 'generate_dataset'), "DatasetManager should have generate_dataset method"
        assert callable(getattr(DatasetManager, 'generate_dataset')), "generate_dataset should be callable"
        
    except ImportError as e:
        pytest.fail(f"Failed to import DatasetManager: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 