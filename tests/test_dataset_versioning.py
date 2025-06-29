"""
Unit tests for dataset versioning and lineage tracking.

Tests cover:
- Hash computation consistency
- Dataset lock file format and updates
- CI check logic for dataset changes
- Telemetry integration with dataset manifests
"""

import json
import hashlib
import tempfile
import os
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, mock_open

from app.utils.dataset_versioning import (
    DatasetManifest,
    DatasetVersioning,
    compute_dataset_hash,
    update_dataset_lock,
    check_dataset_consistency,
    get_dataset_manifest_for_run
)


class TestDatasetManifest:
    """Test DatasetManifest data class"""
    
    def test_manifest_creation_with_required_fields(self):
        """DatasetManifest should create successfully with all required fields"""
        manifest = DatasetManifest(
            name="test_dataset",
            hash_sha256="abc123def456",
            num_samples=1000,
            schema_version="1.0",
            source_url="datasets/test_dataset/",
            created_at="2025-01-27T10:00:00Z"
        )
        
        assert manifest.name == "test_dataset"
        assert manifest.hash_sha256 == "abc123def456"
        assert manifest.num_samples == 1000
        assert manifest.schema_version == "1.0"
        assert manifest.source_url == "datasets/test_dataset/"
        assert manifest.created_at == "2025-01-27T10:00:00Z"
    
    def test_manifest_to_dict(self):
        """DatasetManifest should serialize to dictionary correctly"""
        manifest = DatasetManifest(
            name="test_dataset",
            hash_sha256="abc123def456",
            num_samples=1000,
            schema_version="1.0",
            source_url="datasets/test_dataset/",
            created_at="2025-01-27T10:00:00Z"
        )
        
        manifest_dict = manifest.to_dict()
        
        assert manifest_dict["name"] == "test_dataset"
        assert manifest_dict["hash_sha256"] == "abc123def456"
        assert manifest_dict["num_samples"] == 1000
        assert manifest_dict["schema_version"] == "1.0"
        assert manifest_dict["source_url"] == "datasets/test_dataset/"
        assert manifest_dict["created_at"] == "2025-01-27T10:00:00Z"
    
    def test_manifest_from_dict(self):
        """DatasetManifest should deserialize from dictionary correctly"""
        manifest_dict = {
            "name": "test_dataset",
            "hash_sha256": "abc123def456",
            "num_samples": 1000,
            "schema_version": "1.0",
            "source_url": "datasets/test_dataset/",
            "created_at": "2025-01-27T10:00:00Z"
        }
        
        manifest = DatasetManifest.from_dict(manifest_dict)
        
        assert manifest.name == "test_dataset"
        assert manifest.hash_sha256 == "abc123def456"
        assert manifest.num_samples == 1000


class TestComputeDatasetHash:
    """Test dataset hash computation"""
    
    def test_compute_hash_single_file(self):
        """compute_dataset_hash should return consistent SHA256 for single file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('{"test": "data"}')
            f.flush()
            
            try:
                hash1 = compute_dataset_hash(f.name)
                hash2 = compute_dataset_hash(f.name)
                
                assert hash1 == hash2
                assert len(hash1) == 64  # SHA256 hex length
                assert isinstance(hash1, str)
            finally:
                os.unlink(f.name)
    
    def test_compute_hash_directory_deterministic(self):
        """compute_dataset_hash should return deterministic hash for directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            file1 = Path(temp_dir) / "file1.json"
            file2 = Path(temp_dir) / "file2.json"
            
            file1.write_text('{"sample": 1}')
            file2.write_text('{"sample": 2}')
            
            hash1 = compute_dataset_hash(temp_dir)
            hash2 = compute_dataset_hash(temp_dir)
            
            assert hash1 == hash2
            assert len(hash1) == 64
    
    def test_compute_hash_directory_order_independence(self):
        """Hash should be same regardless of file discovery order"""
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                # Create same files in different order
                files = [("a.json", '{"a": 1}'), ("b.json", '{"b": 2}'), ("c.json", '{"c": 3}')]
                
                # Directory 1: create in order
                for name, content in files:
                    (Path(temp_dir1) / name).write_text(content)
                
                # Directory 2: create in reverse order
                for name, content in reversed(files):
                    (Path(temp_dir2) / name).write_text(content)
                
                hash1 = compute_dataset_hash(temp_dir1)
                hash2 = compute_dataset_hash(temp_dir2)
                
                assert hash1 == hash2
    
    def test_compute_hash_different_content_different_hash(self):
        """Different file content should produce different hashes"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f1:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f2:
                f1.write('{"test": "data1"}')
                f2.write('{"test": "data2"}')
                f1.flush()
                f2.flush()
                
                try:
                    hash1 = compute_dataset_hash(f1.name)
                    hash2 = compute_dataset_hash(f2.name)
                    
                    assert hash1 != hash2
                finally:
                    os.unlink(f1.name)
                    os.unlink(f2.name)


class TestUpdateDatasetLock:
    """Test dataset lock file management"""
    
    def test_update_lock_creates_new_file(self):
        """update_dataset_lock should create new lock file when none exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset"
            dataset_path.mkdir()
            
            # Create sample dataset file
            (dataset_path / "data.json").write_text('{"sample": "data"}')
            
            manifest = update_dataset_lock(str(dataset_path), "test_dataset")
            
            lock_file = dataset_path / "dataset.lock"
            assert lock_file.exists()
            
            # Verify lock file content
            lock_data = json.loads(lock_file.read_text())
            assert lock_data["name"] == "test_dataset"
            assert "hash_sha256" in lock_data
            assert lock_data["num_samples"] >= 0
            assert lock_data["schema_version"] == "1.0"
            assert manifest.name == "test_dataset"
    
    def test_update_lock_updates_existing_file(self):
        """update_dataset_lock should update existing lock file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset"
            dataset_path.mkdir()
            lock_file = dataset_path / "dataset.lock"
            
            # Create initial lock file
            initial_manifest = {
                "name": "test_dataset",
                "hash_sha256": "old_hash",
                "num_samples": 100,
                "schema_version": "1.0",
                "source_url": str(dataset_path),
                "created_at": "2025-01-01T00:00:00Z"
            }
            lock_file.write_text(json.dumps(initial_manifest, indent=2))
            
            # Create new data
            (dataset_path / "new_data.json").write_text('{"sample": "new_data"}')
            
            manifest = update_dataset_lock(str(dataset_path), "test_dataset")
            
            # Verify update
            lock_data = json.loads(lock_file.read_text())
            assert lock_data["hash_sha256"] != "old_hash"
            assert lock_data["name"] == "test_dataset"
            assert manifest.hash_sha256 != "old_hash"
    
    def test_update_lock_preserves_name_and_metadata(self):
        """update_dataset_lock should preserve name and other metadata when updating"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset"
            dataset_path.mkdir()
            lock_file = dataset_path / "dataset.lock"
            
            # Create data file
            (dataset_path / "data.json").write_text('{"sample": "data"}')
            
            # Initial registration
            manifest1 = update_dataset_lock(str(dataset_path), "test_dataset")
            original_created_at = manifest1.created_at
            
            # Add more data and update
            (dataset_path / "more_data.json").write_text('{"sample": "more"}')
            manifest2 = update_dataset_lock(str(dataset_path), "test_dataset")
            
            assert manifest2.name == "test_dataset"
            assert manifest2.hash_sha256 != manifest1.hash_sha256
            # Should preserve original creation time
            lock_data = json.loads(lock_file.read_text())
            assert lock_data["created_at"] == original_created_at


class TestCheckDatasetConsistency:
    """Test CI check logic for dataset consistency"""
    
    def test_check_consistency_passes_when_consistent(self):
        """check_dataset_consistency should pass when dataset matches lock file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset"
            dataset_path.mkdir()
            
            # Create dataset
            (dataset_path / "data.json").write_text('{"sample": "data"}')
            
            # Create lock file
            manifest = update_dataset_lock(str(dataset_path), "test_dataset")
            
            # Check should pass
            result = check_dataset_consistency(str(dataset_path))
            assert result.is_consistent == True
            assert result.expected_hash == manifest.hash_sha256
            assert result.actual_hash == manifest.hash_sha256
    
    def test_check_consistency_fails_when_inconsistent(self):
        """check_dataset_consistency should fail when dataset doesn't match lock file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset"
            dataset_path.mkdir()
            lock_file = dataset_path / "dataset.lock"
            
            # Create initial dataset and lock
            (dataset_path / "data.json").write_text('{"sample": "data"}')
            manifest = update_dataset_lock(str(dataset_path), "test_dataset")
            
            # Modify dataset without updating lock
            (dataset_path / "new_data.json").write_text('{"sample": "new"}')
            
            # Check should fail
            result = check_dataset_consistency(str(dataset_path))
            assert result.is_consistent == False
            assert result.expected_hash == manifest.hash_sha256
            assert result.actual_hash != manifest.hash_sha256
    
    def test_check_consistency_fails_when_no_lock_file(self):
        """check_dataset_consistency should fail when no lock file exists"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset"
            dataset_path.mkdir()
            
            # Create dataset without lock file
            (dataset_path / "data.json").write_text('{"sample": "data"}')
            
            # Check should fail
            result = check_dataset_consistency(str(dataset_path))
            assert result.is_consistent == False
            assert result.expected_hash is None
            assert "No lock file found" in result.error_message


class TestTelemetryIntegration:
    """Test integration with telemetry SDK"""
    
    def test_get_dataset_manifest_for_run_single_dataset(self):
        """get_dataset_manifest_for_run should return manifest for datasets used in run"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset"
            dataset_path.mkdir()
            
            # Create dataset and lock
            (dataset_path / "data.json").write_text('{"sample": "data"}')
            manifest = update_dataset_lock(str(dataset_path), "test_dataset")
            
            # Get manifest for telemetry
            manifests = get_dataset_manifest_for_run([str(dataset_path)])
            
            assert len(manifests) == 1
            assert manifests[0]["name"] == "test_dataset"
            assert manifests[0]["hash_sha256"] == manifest.hash_sha256
    
    def test_get_dataset_manifest_for_run_multiple_datasets(self):
        """get_dataset_manifest_for_run should handle multiple datasets"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset1_path = Path(temp_dir) / "dataset1"
            dataset2_path = Path(temp_dir) / "dataset2"
            dataset1_path.mkdir()
            dataset2_path.mkdir()
            
            # Create datasets
            (dataset1_path / "data.json").write_text('{"sample": "data1"}')
            (dataset2_path / "data.json").write_text('{"sample": "data2"}')
            
            manifest1 = update_dataset_lock(str(dataset1_path), "dataset1")
            manifest2 = update_dataset_lock(str(dataset2_path), "dataset2")
            
            # Get manifests
            manifests = get_dataset_manifest_for_run([str(dataset1_path), str(dataset2_path)])
            
            assert len(manifests) == 2
            names = [m["name"] for m in manifests]
            assert "dataset1" in names
            assert "dataset2" in names
    
    @patch('app.utils.telemetry_sdk.log')
    def test_dataset_manifest_auto_attached_to_telemetry(self, mock_log):
        """Dataset manifests should be automatically attached to telemetry runs"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset"
            dataset_path.mkdir()
            
            # Create dataset
            (dataset_path / "data.json").write_text('{"sample": "data"}')
            manifest = update_dataset_lock(str(dataset_path), "test_dataset")
            
            # Simulate training run with dataset attachment
            from app.utils.dataset_versioning import attach_dataset_to_telemetry
            attach_dataset_to_telemetry([str(dataset_path)])
            
            # Verify telemetry was called with dataset info
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0][0]
            assert "datasets" in call_args
            assert len(call_args["datasets"]) == 1
            assert call_args["datasets"][0]["name"] == "test_dataset"


class TestDatasetVersioningClass:
    """Test DatasetVersioning utility class"""
    
    def test_register_dataset_creates_manifest(self):
        """DatasetVersioning.register() should create proper manifest"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset"
            dataset_path.mkdir()
            
            # Create dataset
            (dataset_path / "data.json").write_text('{"sample": "data"}')
            
            versioning = DatasetVersioning()
            manifest = versioning.register(str(dataset_path), "test_dataset")
            
            assert manifest.name == "test_dataset"
            assert len(manifest.hash_sha256) == 64
            assert manifest.num_samples >= 0
            assert manifest.schema_version == "1.0"
    
    def test_check_consistency_method(self):
        """DatasetVersioning.check_consistency() should validate dataset state"""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset"
            dataset_path.mkdir()
            
            # Create dataset and register
            (dataset_path / "data.json").write_text('{"sample": "data"}')
            
            versioning = DatasetVersioning()
            versioning.register(str(dataset_path), "test_dataset")
            
            # Check should pass
            result = versioning.check_consistency(str(dataset_path))
            assert result.is_consistent == True 