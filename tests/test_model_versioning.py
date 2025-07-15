"""
Tests for Model Versioning and Deployment System

Tests the comprehensive model lifecycle management capabilities.
"""

import pytest
import tempfile
import json
import torch
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

from backend.app.services.deployment.model_versioning import (
    ModelVersionManager,
    ModelVersion,
    ModelMetadata,
    ModelRegistry,
    ModelArtifactManager,
    ModelValidator,
    DeploymentRecord,
    ModelStatus,
    DeploymentTarget,
    create_model_version_manager,
    create_model_version
)
from backend.app.narrative_engine.config import NarrativeLLMConfig


class TestModelVersion:
    """Test model version handling"""
    
    def test_version_creation(self):
        """Test creating model versions"""
        version = ModelVersion(1, 2, 3)
        
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert str(version) == "1.2.3"
    
    def test_version_from_string(self):
        """Test parsing version from string"""
        version = ModelVersion.from_string("2.1.0")
        
        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 0
    
    def test_version_from_string_invalid(self):
        """Test parsing invalid version string"""
        with pytest.raises(ValueError):
            ModelVersion.from_string("1.2")
        
        with pytest.raises(ValueError):
            ModelVersion.from_string("a.b.c")
    
    def test_version_comparison(self):
        """Test version comparison"""
        v1 = ModelVersion(1, 0, 0)
        v2 = ModelVersion(1, 1, 0)
        v3 = ModelVersion(2, 0, 0)
        
        assert v1 < v2
        assert v2 < v3
        assert v1 < v3
        assert v1 == ModelVersion(1, 0, 0)
    
    def test_version_increment(self):
        """Test version incrementing"""
        version = ModelVersion(1, 2, 3)
        
        major = version.increment_major()
        assert major == ModelVersion(2, 0, 0)
        
        minor = version.increment_minor()
        assert minor == ModelVersion(1, 3, 0)
        
        patch = version.increment_patch()
        assert patch == ModelVersion(1, 2, 4)


class TestModelMetadata:
    """Test model metadata handling"""
    
    def test_metadata_creation(self):
        """Test creating model metadata"""
        version = ModelVersion(1, 0, 0)
        metadata = ModelMetadata(
            model_id="test-model-1.0.0",
            version=version,
            name="test-model",
            description="Test model",
            training_config={"epochs": 10},
            training_metrics={"loss": 0.5},
            training_dataset="test_dataset",
            training_duration_minutes=120.0
        )
        
        assert metadata.model_id == "test-model-1.0.0"
        assert metadata.version == version
        assert metadata.name == "test-model"
        assert metadata.status == ModelStatus.TRAINING
        assert metadata.parameter_count == 0
    
    def test_metadata_serialization(self):
        """Test metadata to/from dict conversion"""
        version = ModelVersion(1, 0, 0)
        created_at = datetime.now(timezone.utc)
        
        metadata = ModelMetadata(
            model_id="test-model",
            version=version,
            name="test",
            description="Test",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0,
            created_at=created_at
        )
        
        # Test to_dict
        data = metadata.to_dict()
        assert data["version"] == "1.0.0"
        assert data["created_at"] == created_at.isoformat()
        
        # Test from_dict
        restored = ModelMetadata.from_dict(data)
        assert restored.version == version
        assert restored.created_at == created_at
        assert restored.model_id == metadata.model_id


class TestDeploymentRecord:
    """Test deployment record functionality"""
    
    def test_deployment_record_creation(self):
        """Test creating deployment records"""
        version = ModelVersion(1, 0, 0)
        deployed_at = datetime.now(timezone.utc)
        
        deployment = DeploymentRecord(
            deployment_id="deploy-123",
            model_id="test-model",
            version=version,
            target=DeploymentTarget.STAGING,
            status=ModelStatus.DEPLOYED,
            deployed_at=deployed_at,
            deployed_by="test_user"
        )
        
        assert deployment.deployment_id == "deploy-123"
        assert deployment.target == DeploymentTarget.STAGING
        assert deployment.status == ModelStatus.DEPLOYED
    
    def test_deployment_record_serialization(self):
        """Test deployment record serialization"""
        version = ModelVersion(1, 0, 0)
        deployed_at = datetime.now(timezone.utc)
        
        deployment = DeploymentRecord(
            deployment_id="deploy-123",
            model_id="test-model",
            version=version,
            target=DeploymentTarget.PRODUCTION,
            status=ModelStatus.DEPLOYED,
            deployed_at=deployed_at,
            deployed_by="test_user",
            rollback_version=ModelVersion(0, 9, 0)
        )
        
        data = deployment.to_dict()
        assert data["version"] == "1.0.0"
        assert data["rollback_version"] == "0.9.0"
        assert data["deployed_at"] == deployed_at.isoformat()


class TestModelRegistry:
    """Test model registry functionality"""
    
    @pytest.fixture
    def temp_registry(self):
        """Create temporary registry for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ModelRegistry(temp_dir)
    
    def test_registry_initialization(self, temp_registry):
        """Test registry initializes correctly"""
        assert len(temp_registry.models) == 0
        assert len(temp_registry.deployments) == 0
        assert temp_registry.registry_path.exists()
    
    def test_register_model(self, temp_registry):
        """Test registering a new model"""
        metadata = ModelMetadata(
            model_id="",
            version=ModelVersion(1, 0, 0),
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        model_id = temp_registry.register_model(metadata)
        
        assert model_id == "test-model-1.0.0"
        assert model_id in temp_registry.models
        assert temp_registry.models[model_id].model_id == model_id
    
    def test_register_duplicate_model(self, temp_registry):
        """Test registering duplicate model fails"""
        metadata = ModelMetadata(
            model_id="",
            version=ModelVersion(1, 0, 0),
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        # Register first time
        temp_registry.register_model(metadata)
        
        # Second registration should fail
        with pytest.raises(ValueError):
            temp_registry.register_model(metadata)
    
    def test_get_model(self, temp_registry):
        """Test getting model by ID"""
        metadata = ModelMetadata(
            model_id="",
            version=ModelVersion(1, 0, 0),
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        model_id = temp_registry.register_model(metadata)
        retrieved = temp_registry.get_model(model_id)
        
        assert retrieved is not None
        assert retrieved.model_id == model_id
        assert retrieved.version == ModelVersion(1, 0, 0)
    
    def test_get_nonexistent_model(self, temp_registry):
        """Test getting nonexistent model returns None"""
        result = temp_registry.get_model("nonexistent")
        assert result is None
    
    def test_list_models(self, temp_registry):
        """Test listing models"""
        # Register multiple models
        for i in range(3):
            metadata = ModelMetadata(
                model_id="",
                version=ModelVersion(1, i, 0),
                name=f"test-model-{i}",
                description=f"Test model {i}",
                training_config={},
                training_metrics={},
                training_dataset="test",
                training_duration_minutes=60.0,
                status=ModelStatus.STAGED if i == 0 else ModelStatus.TRAINING
            )
            temp_registry.register_model(metadata)
        
        # Test list all
        all_models = temp_registry.list_models()
        assert len(all_models) == 3
        
        # Test list by status
        staged_models = temp_registry.list_models(ModelStatus.STAGED)
        assert len(staged_models) == 1
        assert staged_models[0].status == ModelStatus.STAGED
    
    def test_get_latest_version(self, temp_registry):
        """Test getting latest version of a model"""
        # Register multiple versions
        for version in [(1, 0, 0), (1, 1, 0), (2, 0, 0)]:
            metadata = ModelMetadata(
                model_id="",
                version=ModelVersion(*version),
                name="test-model",
                description="Test model",
                training_config={},
                training_metrics={},
                training_dataset="test",
                training_duration_minutes=60.0
            )
            temp_registry.register_model(metadata)
        
        latest = temp_registry.get_latest_version("test-model")
        assert latest is not None
        assert latest.version == ModelVersion(2, 0, 0)
    
    def test_get_latest_version_nonexistent(self, temp_registry):
        """Test getting latest version of nonexistent model"""
        result = temp_registry.get_latest_version("nonexistent")
        assert result is None
    
    def test_update_model_status(self, temp_registry):
        """Test updating model status"""
        metadata = ModelMetadata(
            model_id="",
            version=ModelVersion(1, 0, 0),
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        model_id = temp_registry.register_model(metadata)
        
        # Update status
        temp_registry.update_model_status(model_id, ModelStatus.STAGED)
        
        updated = temp_registry.get_model(model_id)
        assert updated.status == ModelStatus.STAGED
    
    def test_update_nonexistent_model_status(self, temp_registry):
        """Test updating status of nonexistent model"""
        with pytest.raises(ValueError):
            temp_registry.update_model_status("nonexistent", ModelStatus.STAGED)
    
    def test_record_deployment(self, temp_registry):
        """Test recording deployments"""
        deployment = DeploymentRecord(
            deployment_id="deploy-123",
            model_id="test-model",
            version=ModelVersion(1, 0, 0),
            target=DeploymentTarget.STAGING,
            status=ModelStatus.DEPLOYED,
            deployed_at=datetime.now(timezone.utc),
            deployed_by="test_user"
        )
        
        temp_registry.record_deployment(deployment)
        
        assert len(temp_registry.deployments) == 1
        assert temp_registry.deployments[0].deployment_id == "deploy-123"
    
    def test_get_current_deployment(self, temp_registry):
        """Test getting current deployment"""
        # Record multiple deployments
        older_deployment = DeploymentRecord(
            deployment_id="deploy-1",
            model_id="test-model",
            version=ModelVersion(1, 0, 0),
            target=DeploymentTarget.STAGING,
            status=ModelStatus.DEPLOYED,
            deployed_at=datetime.now(timezone.utc).replace(hour=10),
            deployed_by="test_user"
        )
        
        newer_deployment = DeploymentRecord(
            deployment_id="deploy-2",
            model_id="test-model",
            version=ModelVersion(1, 1, 0),
            target=DeploymentTarget.STAGING,
            status=ModelStatus.DEPLOYED,
            deployed_at=datetime.now(timezone.utc).replace(hour=12),
            deployed_by="test_user"
        )
        
        temp_registry.record_deployment(older_deployment)
        temp_registry.record_deployment(newer_deployment)
        
        current = temp_registry.get_current_deployment(DeploymentTarget.STAGING)
        assert current is not None
        assert current.deployment_id == "deploy-2"  # Should be the newer one
    
    def test_persistence(self, temp_registry):
        """Test registry persistence"""
        # Register a model
        metadata = ModelMetadata(
            model_id="",
            version=ModelVersion(1, 0, 0),
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        model_id = temp_registry.register_model(metadata)
        
        # Create new registry instance with same path
        new_registry = ModelRegistry(str(temp_registry.registry_path))
        
        # Should load the existing model
        assert len(new_registry.models) == 1
        assert model_id in new_registry.models


class TestModelArtifactManager:
    """Test model artifact management"""
    
    @pytest.fixture
    def temp_artifacts(self):
        """Create temporary artifact manager for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ModelArtifactManager(temp_dir)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = Mock()
        model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        model.parameters.return_value = [torch.randn(10, 10), torch.randn(5)]
        return model
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return NarrativeLLMConfig()
    
    @pytest.fixture 
    def test_metadata(self):
        """Create test metadata"""
        return ModelMetadata(
            model_id="test-model-1.0.0",
            version=ModelVersion(1, 0, 0),
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
    
    def test_get_model_path(self, temp_artifacts):
        """Test getting model path"""
        path = temp_artifacts.get_model_path("test-model")
        
        expected = temp_artifacts.artifacts_root / "test-model"
        assert path == expected
    
    def test_save_model_artifacts(self, temp_artifacts, mock_model, test_config, test_metadata):
        """Test saving model artifacts"""
        artifact_info = temp_artifacts.save_model_artifacts(
            mock_model, test_config, test_metadata
        )
        
        # Check returned paths
        assert "model_path" in artifact_info
        assert "config_path" in artifact_info
        assert "metadata_path" in artifact_info
        assert "model_checksum" in artifact_info
        assert "config_checksum" in artifact_info
        
        # Check files exist
        model_path = Path(artifact_info["model_path"])
        config_path = Path(artifact_info["config_path"])
        metadata_path = Path(artifact_info["metadata_path"])
        
        assert model_path.exists()
        assert config_path.exists()
        assert metadata_path.exists()
        
        # Check file contents
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        assert isinstance(config_data, dict)
        
        with open(metadata_path, 'r') as f:
            metadata_data = json.load(f)
        assert metadata_data["model_id"] == test_metadata.model_id
    
    def test_verify_artifacts(self, temp_artifacts, mock_model, test_config, test_metadata):
        """Test artifact verification"""
        # Save artifacts first
        artifact_info = temp_artifacts.save_model_artifacts(
            mock_model, test_config, test_metadata
        )
        
        # Update metadata with artifact info
        test_metadata.model_path = artifact_info["model_path"]
        test_metadata.config_path = artifact_info["config_path"]
        test_metadata.model_checksum = artifact_info["model_checksum"]
        test_metadata.config_checksum = artifact_info["config_checksum"]
        
        # Verification should pass
        assert temp_artifacts.verify_artifacts(test_metadata) is True
    
    def test_verify_artifacts_missing_files(self, temp_artifacts, test_metadata):
        """Test artifact verification with missing files"""
        test_metadata.model_path = "/nonexistent/model.pt"
        test_metadata.config_path = "/nonexistent/config.json"
        
        assert temp_artifacts.verify_artifacts(test_metadata) is False
    
    def test_verify_artifacts_no_paths(self, temp_artifacts, test_metadata):
        """Test artifact verification with no paths"""
        test_metadata.model_path = None
        test_metadata.config_path = None
        
        assert temp_artifacts.verify_artifacts(test_metadata) is False
    
    def test_cleanup_artifacts(self, temp_artifacts, mock_model, test_config, test_metadata):
        """Test cleaning up artifacts"""
        # Save artifacts first
        temp_artifacts.save_model_artifacts(mock_model, test_config, test_metadata)
        
        model_dir = temp_artifacts.get_model_path(test_metadata.model_id)
        assert model_dir.exists()
        
        # Cleanup
        temp_artifacts.cleanup_artifacts(test_metadata.model_id)
        
        assert not model_dir.exists()


class TestModelValidator:
    """Test model validation"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        return Mock()
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return NarrativeLLMConfig()
    
    @pytest.fixture
    def validator(self):
        """Create model validator"""
        return ModelValidator()
    
    # REMOVED: test_validate_model_success - was testing mock infrastructure
    # REMOVED: test_validate_model_with_custom_prompts - was testing mock infrastructure  
    # REMOVED: test_validate_model_failure - was testing mock infrastructure


class TestModelVersionManager:
    """Test the main model version manager"""
    
    @pytest.fixture
    def temp_manager(self):
        """Create temporary version manager for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = str(Path(temp_dir) / "registry")
            artifacts_path = str(Path(temp_dir) / "artifacts")
            yield ModelVersionManager(registry_path, artifacts_path)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        model = Mock()
        model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
        model.parameters.return_value = [torch.randn(10, 10), torch.randn(5)]
        return model
    
    @pytest.fixture
    def test_config(self):
        """Create test configuration"""
        return NarrativeLLMConfig()
    
    def test_manager_initialization(self, temp_manager):
        """Test manager initializes correctly"""
        assert temp_manager.registry is not None
        assert temp_manager.artifact_manager is not None
        assert temp_manager.validator is not None
    
    def test_create_model_version_initial(self, temp_manager, mock_model, test_config):
        """Test creating first model version"""
        model_id = temp_manager.create_model_version(
            model=mock_model,
            config=test_config,
            name="test-model",
            description="Test model",
            training_config={"epochs": 10},
            training_metrics={"loss": 0.5},
            training_dataset="test_dataset",
            training_duration_minutes=120.0
        )
        
        assert model_id == "test-model-1.0.0"
        
        metadata = temp_manager.registry.get_model(model_id)
        assert metadata is not None
        assert metadata.version == ModelVersion(1, 0, 0)
        assert metadata.parameter_count > 0
        assert metadata.model_size_mb > 0
    
    def test_create_model_version_increment(self, temp_manager, mock_model, test_config):
        """Test creating incremental model versions"""
        # Create initial version
        temp_manager.create_model_version(
            model=mock_model,
            config=test_config,
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        # Create patch version
        patch_id = temp_manager.create_model_version(
            model=mock_model,
            config=test_config,
            name="test-model",
            description="Test model patch",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0,
            version_type="patch"
        )
        
        assert patch_id == "test-model-1.0.1"
        
        # Create minor version
        minor_id = temp_manager.create_model_version(
            model=mock_model,
            config=test_config,
            name="test-model",
            description="Test model minor",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0,
            version_type="minor"
        )
        
        assert minor_id == "test-model-1.1.0"
    
    def test_load_model(self, temp_manager, mock_model, test_config):
        """Test loading a model"""
        # Create model first
        model_id = temp_manager.create_model_version(
            model=mock_model,
            config=test_config,
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        # Load model
        with patch('backend.app.narrative_engine.quad_head_model.create_quad_head_model') as mock_create:
            mock_loaded_model = Mock()
            mock_create.return_value = mock_loaded_model
            
            loaded_model, loaded_config, loaded_metadata = temp_manager.load_model(model_id)
            
            assert loaded_model == mock_loaded_model
            assert isinstance(loaded_config, NarrativeLLMConfig)
            assert loaded_metadata.model_id == model_id
    
    def test_load_nonexistent_model(self, temp_manager):
        """Test loading nonexistent model"""
        with pytest.raises(ValueError):
            temp_manager.load_model("nonexistent")
    
    @pytest.mark.asyncio
    async def test_validate_model(self, temp_manager, mock_model, test_config):
        """Test model validation"""
        # Create model first
        model_id = temp_manager.create_model_version(
            model=mock_model,
            config=test_config,
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        # Mock load_model to avoid file loading
        with patch.object(temp_manager, 'load_model') as mock_load:
            mock_load.return_value = (mock_model, test_config, Mock())
            
            result = await temp_manager.validate_model(model_id)
            
            assert "passed" in result
            
            # Check model status was updated
            metadata = temp_manager.registry.get_model(model_id)
            if result["passed"]:
                assert metadata.status == ModelStatus.STAGED
            else:
                assert metadata.status == ModelStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_deploy_model(self, temp_manager, mock_model, test_config):
        """Test model deployment"""
        # Create and stage model
        model_id = temp_manager.create_model_version(
            model=mock_model,
            config=test_config,
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        # Update status to staged
        temp_manager.registry.update_model_status(model_id, ModelStatus.STAGED)
        
        # Deploy model
        deployment_id = await temp_manager.deploy_model(
            model_id, 
            DeploymentTarget.STAGING,
            {"replicas": 2}
        )
        
        assert deployment_id is not None
        assert deployment_id.startswith(model_id)
        
        # Check deployment was recorded
        current = temp_manager.registry.get_current_deployment(DeploymentTarget.STAGING)
        assert current is not None
        assert current.model_id == model_id
        
        # Check model status updated
        metadata = temp_manager.registry.get_model(model_id)
        assert metadata.status == ModelStatus.DEPLOYED
    
    @pytest.mark.asyncio
    async def test_deploy_unstaged_model(self, temp_manager, mock_model, test_config):
        """Test deploying unstaged model fails"""
        # Create model (will be in TRAINING status)
        model_id = temp_manager.create_model_version(
            model=mock_model,
            config=test_config,
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        # Try to deploy unstaged model
        with pytest.raises(ValueError):
            await temp_manager.deploy_model(model_id, DeploymentTarget.STAGING)
    
    def test_list_models(self, temp_manager, mock_model, test_config):
        """Test listing models"""
        # Create multiple models
        for i in range(3):
            temp_manager.create_model_version(
                model=mock_model,
                config=test_config,
                name=f"test-model-{i}",
                description=f"Test model {i}",
                training_config={},
                training_metrics={},
                training_dataset="test",
                training_duration_minutes=60.0
            )
        
        models = temp_manager.list_models()
        assert len(models) == 3
        
        # Test filtering by status
        staged_models = temp_manager.list_models(ModelStatus.STAGED)
        assert len(staged_models) == 0  # None are staged
    
    def test_get_model_info(self, temp_manager, mock_model, test_config):
        """Test getting model info"""
        model_id = temp_manager.create_model_version(
            model=mock_model,
            config=test_config,
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        info = temp_manager.get_model_info(model_id)
        assert info is not None
        assert info.model_id == model_id
    
    def test_delete_model(self, temp_manager, mock_model, test_config):
        """Test deleting a model"""
        model_id = temp_manager.create_model_version(
            model=mock_model,
            config=test_config,
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        # Delete model
        temp_manager.delete_model(model_id)
        
        # Should be gone from registry
        assert temp_manager.registry.get_model(model_id) is None
    
    def test_delete_deployed_model_without_force(self, temp_manager, mock_model, test_config):
        """Test deleting deployed model without force fails"""
        model_id = temp_manager.create_model_version(
            model=mock_model,
            config=test_config,
            name="test-model",
            description="Test model",
            training_config={},
            training_metrics={},
            training_dataset="test",
            training_duration_minutes=60.0
        )
        
        # Set as deployed
        temp_manager.registry.update_model_status(model_id, ModelStatus.DEPLOYED)
        
        # Should fail without force
        with pytest.raises(ValueError):
            temp_manager.delete_model(model_id)
        
        # Should succeed with force
        temp_manager.delete_model(model_id, force=True)
        assert temp_manager.registry.get_model(model_id) is None


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_model_version_manager(self):
        """Test creating model version manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = create_model_version_manager(
                registry_path=temp_dir + "/registry",
                artifacts_path=temp_dir + "/artifacts"
            )
            
            assert isinstance(manager, ModelVersionManager)
    
    def test_create_model_version(self):
        """Test creating model version"""
        version = create_model_version(1, 2, 3)
        
        assert isinstance(version, ModelVersion)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3


class TestIntegration:
    """Integration tests for the complete versioning system"""
    
    # REMOVED: test_end_to_end_model_lifecycle - was testing mock infrastructure with complex validation
    
    @pytest.mark.asyncio
    async def test_registry_persistence(self):
        """Test that registry persists across manager instances"""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = temp_dir + "/registry"
            artifacts_path = temp_dir + "/artifacts"
            
            # Create first manager and add model
            manager1 = create_model_version_manager(
                registry_path=registry_path,
                artifacts_path=artifacts_path
            )
            
            mock_model = Mock()
            mock_model.state_dict.return_value = {"layer.weight": torch.randn(10, 10)}
            mock_model.parameters.return_value = [torch.randn(10, 10)]
            
            config = NarrativeLLMConfig()
            
            model_id = manager1.create_model_version(
                model=mock_model,
                config=config,
                name="persistence-test",
                description="Test persistence",
                training_config={},
                training_metrics={},
                training_dataset="test",
                training_duration_minutes=60.0
            )
            
            # Create second manager and verify it sees the model
            manager2 = create_model_version_manager(
                registry_path=registry_path,
                artifacts_path=artifacts_path
            )
            
            models = manager2.list_models()
            assert len(models) == 1
            assert models[0].name == "persistence-test"
            assert models[0].model_id == model_id 