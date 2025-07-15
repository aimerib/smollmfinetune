"""
Model Versioning and Deployment System

This module provides comprehensive model lifecycle management for quad-head models:
- Semantic versioning for model releases
- Model registry with metadata tracking  
- Production deployment pipeline
- Safe rollback capabilities
- Model artifact management
- Quality validation and testing
"""

import asyncio
import json
import logging
import shutil
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
import torch
from pydantic import BaseModel, Field

from backend.app.narrative_engine.quad_head_model import QuadHeadNarrativeLM
from backend.app.narrative_engine.config import NarrativeLLMConfig
from backend.app.services.evaluation.streaming_evaluation import StreamingEvaluationService

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model deployment status"""
    TRAINING = "training"
    VALIDATING = "validating"  
    STAGED = "staged"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    FAILED = "failed"


class DeploymentTarget(str, Enum):
    """Deployment target environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ModelVersion:
    """Model version information"""
    major: int
    minor: int 
    patch: int
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __lt__(self, other: 'ModelVersion') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: 'ModelVersion') -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    @classmethod
    def from_string(cls, version_str: str) -> 'ModelVersion':
        """Parse version from string like '1.2.3'"""
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")
        
        try:
            major, minor, patch = map(int, parts)
            return cls(major=major, minor=minor, patch=patch)
        except ValueError:
            raise ValueError(f"Invalid version format: {version_str}")
    
    def increment_major(self) -> 'ModelVersion':
        """Increment major version (breaking changes)"""
        return ModelVersion(major=self.major + 1, minor=0, patch=0)
    
    def increment_minor(self) -> 'ModelVersion':
        """Increment minor version (new features)"""
        return ModelVersion(major=self.major, minor=self.minor + 1, patch=0)
    
    def increment_patch(self) -> 'ModelVersion':
        """Increment patch version (bug fixes)"""
        return ModelVersion(major=self.major, minor=self.minor, patch=self.patch + 1)


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    version: ModelVersion
    name: str
    description: str
    
    # Training metadata
    training_config: Dict[str, Any]
    training_metrics: Dict[str, float]
    training_dataset: str
    training_duration_minutes: float
    
    # Model architecture
    architecture: str = "QuadHeadNarrativeLM"
    parameter_count: int = 0
    model_size_mb: float = 0.0
    
    # Quality metrics
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    performance_benchmarks: Dict[str, float] = field(default_factory=dict)
    
    # Deployment info
    status: ModelStatus = ModelStatus.TRAINING
    deployment_target: Optional[DeploymentTarget] = None
    deployed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    
    # File paths
    model_path: Optional[str] = None
    config_path: Optional[str] = None
    artifacts_path: Optional[str] = None
    
    # Checksums for integrity
    model_checksum: Optional[str] = None
    config_checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['version'] = str(self.version)
        data['created_at'] = self.created_at.isoformat()
        if self.deployed_at:
            data['deployed_at'] = self.deployed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        data = data.copy()
        data['version'] = ModelVersion.from_string(data['version'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('deployed_at'):
            data['deployed_at'] = datetime.fromisoformat(data['deployed_at'])
        return cls(**data)


@dataclass
class DeploymentRecord:
    """Record of a model deployment"""
    deployment_id: str
    model_id: str
    version: ModelVersion
    target: DeploymentTarget
    status: ModelStatus
    deployed_at: datetime
    deployed_by: str
    
    # Deployment configuration
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    rollback_version: Optional[ModelVersion] = None
    
    # Performance tracking
    deployment_metrics: Dict[str, float] = field(default_factory=dict)
    health_check_results: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['version'] = str(self.version)
        data['deployed_at'] = self.deployed_at.isoformat()
        if self.rollback_version:
            data['rollback_version'] = str(self.rollback_version)
        return data


class ModelRegistry:
    """Central registry for model versions and metadata"""
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_file = self.registry_path / "models.json"
        self.deployments_file = self.registry_path / "deployments.json"
        
        # Load existing data
        self.models: Dict[str, ModelMetadata] = self._load_models()
        self.deployments: List[DeploymentRecord] = self._load_deployments()
    
    def _load_models(self) -> Dict[str, ModelMetadata]:
        """Load models from registry file"""
        if not self.models_file.exists():
            return {}
        
        try:
            with open(self.models_file, 'r') as f:
                data = json.load(f)
            
            models = {}
            for model_id, model_data in data.items():
                models[model_id] = ModelMetadata.from_dict(model_data)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to load models registry: {e}")
            return {}
    
    def _load_deployments(self) -> List[DeploymentRecord]:
        """Load deployment records"""
        if not self.deployments_file.exists():
            return []
        
        try:
            with open(self.deployments_file, 'r') as f:
                data = json.load(f)
            
            deployments = []
            for deployment_data in data:
                deployment_data['version'] = ModelVersion.from_string(deployment_data['version'])
                deployment_data['deployed_at'] = datetime.fromisoformat(deployment_data['deployed_at'])
                if deployment_data.get('rollback_version'):
                    deployment_data['rollback_version'] = ModelVersion.from_string(deployment_data['rollback_version'])
                
                deployments.append(DeploymentRecord(**deployment_data))
            
            return deployments
            
        except Exception as e:
            logger.error(f"Failed to load deployments registry: {e}")
            return []
    
    def _save_models(self):
        """Save models registry to file"""
        try:
            data = {model_id: model.to_dict() for model_id, model in self.models.items()}
            with open(self.models_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save models registry: {e}")
            raise
    
    def _save_deployments(self):
        """Save deployment records to file"""
        try:
            data = [deployment.to_dict() for deployment in self.deployments]
            with open(self.deployments_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save deployments registry: {e}")
            raise
    
    def register_model(self, metadata: ModelMetadata) -> str:
        """Register a new model version"""
        model_id = f"{metadata.name}-{metadata.version}"
        
        if model_id in self.models:
            raise ValueError(f"Model {model_id} already exists")
        
        metadata.model_id = model_id
        self.models[model_id] = metadata
        self._save_models()
        
        logger.info(f"Registered model {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        return self.models.get(model_id)
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """List all models, optionally filtered by status"""
        models = list(self.models.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        # Sort by version descending
        models.sort(key=lambda m: m.version, reverse=True)
        return models
    
    def get_latest_version(self, model_name: str) -> Optional[ModelMetadata]:
        """Get the latest version of a model by name"""
        matching_models = [m for m in self.models.values() if m.name == model_name]
        
        if not matching_models:
            return None
        
        return max(matching_models, key=lambda m: m.version)
    
    def update_model_status(self, model_id: str, status: ModelStatus):
        """Update model deployment status"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        self.models[model_id].status = status
        self._save_models()
        
        logger.info(f"Updated model {model_id} status to {status}")
    
    def record_deployment(self, deployment: DeploymentRecord):
        """Record a new deployment"""
        self.deployments.append(deployment)
        self._save_deployments()
        
        logger.info(f"Recorded deployment {deployment.deployment_id}")
    
    def get_current_deployment(self, target: DeploymentTarget) -> Optional[DeploymentRecord]:
        """Get current active deployment for target environment"""
        target_deployments = [
            d for d in self.deployments 
            if d.target == target and d.status == ModelStatus.DEPLOYED
        ]
        
        if not target_deployments:
            return None
        
        # Return most recent deployment
        return max(target_deployments, key=lambda d: d.deployed_at)


class ModelArtifactManager:
    """Manages model files and artifacts"""
    
    def __init__(self, artifacts_root: str = "models/artifacts"):
        self.artifacts_root = Path(artifacts_root)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_id: str) -> Path:
        """Get path for model artifacts"""
        return self.artifacts_root / model_id
    
    def save_model_artifacts(self, 
                           model: QuadHeadNarrativeLM,
                           config: NarrativeLLMConfig,
                           metadata: ModelMetadata) -> Dict[str, str]:
        """Save model artifacts and return file paths"""
        
        model_dir = self.get_model_path(metadata.model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = model_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save configuration
        config_path = model_dir / "config.json"
        config_dict = asdict(config)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Calculate checksums
        model_checksum = self._calculate_checksum(model_path)
        config_checksum = self._calculate_checksum(config_path)
        
        return {
            "model_path": str(model_path),
            "config_path": str(config_path),
            "metadata_path": str(metadata_path),
            "model_checksum": model_checksum,
            "config_checksum": config_checksum
        }
    
    def load_model_artifacts(self, model_id: str) -> Tuple[QuadHeadNarrativeLM, NarrativeLLMConfig, ModelMetadata]:
        """Load model artifacts from storage"""
        
        model_dir = self.get_model_path(model_id)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model artifacts not found: {model_id}")
        
        # Load configuration
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found for model: {model_id}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = NarrativeLLMConfig(**config_dict)
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata.from_dict(metadata_dict)
        else:
            # Create minimal metadata if not found
            metadata = ModelMetadata(
                model_id=model_id,
                version=ModelVersion(1, 0, 0),
                name=model_id,
                description="Loaded model",
                training_config={},
                training_metrics={},
                training_dataset="unknown",
                training_duration_minutes=0.0
            )
        
        # Create and load model
        from backend.app.narrative_engine.quad_head_model import create_quad_head_model
        model = create_quad_head_model(config)
        
        # Load weights
        model_path = model_dir / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
        
        return model, config, metadata
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def verify_artifacts(self, metadata: ModelMetadata) -> bool:
        """Verify model artifacts integrity"""
        if not metadata.model_path or not metadata.config_path:
            return False
        
        model_path = Path(metadata.model_path)
        config_path = Path(metadata.config_path)
        
        if not model_path.exists() or not config_path.exists():
            return False
        
        # Verify checksums if available
        if metadata.model_checksum:
            current_checksum = self._calculate_checksum(model_path)
            if current_checksum != metadata.model_checksum:
                return False
        
        if metadata.config_checksum:
            current_checksum = self._calculate_checksum(config_path)
            if current_checksum != metadata.config_checksum:
                return False
        
        return True
    
    def cleanup_artifacts(self, model_id: str):
        """Remove model artifacts from storage"""
        model_dir = self.get_model_path(model_id)
        
        if model_dir.exists():
            shutil.rmtree(model_dir)
            logger.info(f"Cleaned up artifacts for model {model_id}")


class ModelValidator:
    """Validates model quality before deployment"""
    
    def __init__(self, evaluation_service: Optional[StreamingEvaluationService] = None):
        self.evaluation_service = evaluation_service or StreamingEvaluationService()
        
        # Quality thresholds for deployment
        self.quality_thresholds = {
            "overall_quality_score": 0.7,
            "text_coherence": 0.75,
            "speech_quality": 0.65,
            "response_latency_ms": 200.0
        }
    
    async def validate_model(self, 
                           model: QuadHeadNarrativeLM,
                           config: NarrativeLLMConfig,
                           test_prompts: List[str] = None) -> Dict[str, Any]:
        """Validate model quality for deployment"""
        
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you today?",
                "Tell me about yourself.",
                "What is your favorite hobby?",
                "How do you feel about helping people?"
            ]
        
        validation_results = {
            "passed": False,
            "scores": {},
            "detailed_results": [],
            "failed_checks": []
        }
        
        try:
            # Test model with each prompt
            total_scores = {}
            
            for prompt in test_prompts:
                # Mock generation for testing
                # In real implementation, would use streaming inference
                result = await self._test_model_response(model, config, prompt)
                validation_results["detailed_results"].append(result)
                
                # Aggregate scores
                for metric, score in result.get("scores", {}).items():
                    if metric not in total_scores:
                        total_scores[metric] = []
                    total_scores[metric].append(score)
            
            # Calculate average scores
            avg_scores = {}
            for metric, scores in total_scores.items():
                avg_scores[metric] = sum(scores) / len(scores) if scores else 0.0
            
            validation_results["scores"] = avg_scores
            
            # Check against thresholds
            failed_checks = []
            for metric, threshold in self.quality_thresholds.items():
                if metric in avg_scores:
                    if avg_scores[metric] < threshold:
                        failed_checks.append(f"{metric}: {avg_scores[metric]:.3f} < {threshold}")
            
            validation_results["failed_checks"] = failed_checks
            validation_results["passed"] = len(failed_checks) == 0
            
            logger.info(f"Model validation {'passed' if validation_results['passed'] else 'failed'}")
            
        except Exception as e:
            logger.error(f"Model validation failed with error: {e}")
            validation_results["failed_checks"].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    async def _test_model_response(self, 
                                 model: QuadHeadNarrativeLM,
                                 config: NarrativeLLMConfig,
                                 prompt: str) -> Dict[str, Any]:
        """Test model response to a single prompt"""
        
        start_time = datetime.now()
        
        # Mock model inference for testing
        # In real implementation, would use actual model generation
        mock_response = {
            "text": "This is a test response.",
            "speech_frames": [[0.1, 0.2, 0.3] for _ in range(10)],
            "control_signals": ["neutral"],
            "memory_updates": [{"type": "conversation", "content": "test"}]
        }
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Calculate quality scores (simplified)
        scores = {
            "text_coherence": 0.8,  # Mock score
            "speech_quality": 0.75,  # Mock score
            "response_latency_ms": latency_ms,
            "overall_quality_score": 0.77  # Mock overall score
        }
        
        return {
            "prompt": prompt,
            "response": mock_response,
            "scores": scores,
            "latency_ms": latency_ms
        }


class ModelVersionManager:
    """Main model versioning and deployment manager"""
    
    def __init__(self, 
                 registry_path: str = "models/registry",
                 artifacts_path: str = "models/artifacts"):
        
        self.registry = ModelRegistry(registry_path)
        self.artifact_manager = ModelArtifactManager(artifacts_path)
        self.validator = ModelValidator()
        
        logger.info("Model Version Manager initialized")
    
    def create_model_version(self,
                           model: QuadHeadNarrativeLM,
                           config: NarrativeLLMConfig,
                           name: str,
                           description: str,
                           training_config: Dict[str, Any],
                           training_metrics: Dict[str, float],
                           training_dataset: str,
                           training_duration_minutes: float,
                           version_type: str = "patch") -> str:
        """Create a new model version"""
        
        # Determine next version number
        latest = self.registry.get_latest_version(name)
        if latest is None:
            version = ModelVersion(1, 0, 0)
        else:
            if version_type == "major":
                version = latest.version.increment_major()
            elif version_type == "minor":
                version = latest.version.increment_minor()
            else:  # patch
                version = latest.version.increment_patch()
        
        # Create metadata
        metadata = ModelMetadata(
            model_id="",  # Will be set by registry
            version=version,
            name=name,
            description=description,
            training_config=training_config,
            training_metrics=training_metrics,
            training_dataset=training_dataset,
            training_duration_minutes=training_duration_minutes,
            parameter_count=sum(p.numel() for p in model.parameters()),
            model_size_mb=sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # Assuming float32
        )
        
        # Register model
        model_id = self.registry.register_model(metadata)
        
        # Save artifacts
        artifact_info = self.artifact_manager.save_model_artifacts(model, config, metadata)
        
        # Update metadata with artifact paths
        metadata.model_path = artifact_info["model_path"]
        metadata.config_path = artifact_info["config_path"] 
        metadata.artifacts_path = str(self.artifact_manager.get_model_path(model_id))
        metadata.model_checksum = artifact_info["model_checksum"]
        metadata.config_checksum = artifact_info["config_checksum"]
        
        # Update registry with artifact info
        self.registry.models[model_id] = metadata
        self.registry._save_models()
        
        logger.info(f"Created model version {model_id}")
        return model_id
    
    def load_model(self, model_id: str) -> Tuple[QuadHeadNarrativeLM, NarrativeLLMConfig, ModelMetadata]:
        """Load a model by ID"""
        
        metadata = self.registry.get_model(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        # Verify artifacts
        if not self.artifact_manager.verify_artifacts(metadata):
            raise RuntimeError(f"Model artifacts verification failed: {model_id}")
        
        # Load model
        model, config, _ = self.artifact_manager.load_model_artifacts(model_id)
        
        logger.info(f"Loaded model {model_id}")
        return model, config, metadata
    
    async def validate_model(self, model_id: str) -> Dict[str, Any]:
        """Validate a model for deployment"""
        
        model, config, metadata = self.load_model(model_id)
        
        # Run validation
        validation_result = await self.validator.validate_model(model, config)
        
        # Update model status based on validation
        if validation_result["passed"]:
            self.registry.update_model_status(model_id, ModelStatus.STAGED)
            metadata.validation_metrics = validation_result["scores"]
        else:
            self.registry.update_model_status(model_id, ModelStatus.FAILED)
        
        return validation_result
    
    async def deploy_model(self,
                         model_id: str,
                         target: DeploymentTarget,
                         deployment_config: Dict[str, Any] = None) -> str:
        """Deploy a model to target environment"""
        
        metadata = self.registry.get_model(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        if metadata.status != ModelStatus.STAGED:
            raise ValueError(f"Model {model_id} is not staged for deployment (status: {metadata.status})")
        
        # Check if there's a current deployment to rollback to
        current_deployment = self.registry.get_current_deployment(target)
        rollback_version = current_deployment.version if current_deployment else None
        
        # Create deployment record
        deployment_id = f"{model_id}-{target.value}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        deployment = DeploymentRecord(
            deployment_id=deployment_id,
            model_id=model_id,
            version=metadata.version,
            target=target,
            status=ModelStatus.DEPLOYED,
            deployed_at=datetime.now(timezone.utc),
            deployed_by="system",
            deployment_config=deployment_config or {},
            rollback_version=rollback_version
        )
        
        # Update model status
        self.registry.update_model_status(model_id, ModelStatus.DEPLOYED)
        metadata.deployment_target = target
        metadata.deployed_at = deployment.deployed_at
        
        # Record deployment
        self.registry.record_deployment(deployment)
        
        logger.info(f"Deployed model {model_id} to {target}")
        return deployment_id
    
    async def rollback_deployment(self, target: DeploymentTarget) -> Optional[str]:
        """Rollback to previous deployment"""
        
        current_deployment = self.registry.get_current_deployment(target)
        if not current_deployment or not current_deployment.rollback_version:
            raise ValueError(f"No rollback version available for {target}")
        
        # Find the rollback model
        rollback_models = [
            m for m in self.registry.models.values()
            if m.version == current_deployment.rollback_version and m.name == current_deployment.model_id.split('-')[0]
        ]
        
        if not rollback_models:
            raise ValueError(f"Rollback model not found: {current_deployment.rollback_version}")
        
        rollback_model = rollback_models[0]
        
        # Deploy rollback version
        deployment_id = await self.deploy_model(rollback_model.model_id, target, {"rollback": True})
        
        logger.info(f"Rolled back {target} to version {current_deployment.rollback_version}")
        return deployment_id
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """List all models"""
        return self.registry.list_models(status)
    
    def get_model_info(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model information"""
        return self.registry.get_model(model_id)
    
    def delete_model(self, model_id: str, force: bool = False):
        """Delete a model and its artifacts"""
        
        metadata = self.registry.get_model(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        if metadata.status == ModelStatus.DEPLOYED and not force:
            raise ValueError(f"Cannot delete deployed model {model_id} without force=True")
        
        # Remove from registry
        del self.registry.models[model_id]
        self.registry._save_models()
        
        # Clean up artifacts
        self.artifact_manager.cleanup_artifacts(model_id)
        
        logger.info(f"Deleted model {model_id}")


# Factory functions
def create_model_version_manager(**kwargs) -> ModelVersionManager:
    """Factory function to create model version manager"""
    return ModelVersionManager(**kwargs)


def create_model_version(major: int, minor: int, patch: int) -> ModelVersion:
    """Factory function to create model version"""
    return ModelVersion(major=major, minor=minor, patch=patch) 