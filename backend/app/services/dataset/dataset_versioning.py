"""
Dataset Versioning & Lineage Tracking

Provides reproducible experiment tracking by versioning datasets with content hashes,
metadata manifests, and integration with the telemetry SDK.

Key Features:
- Deterministic SHA256 hashing of datasets (files or directories)
- dataset.lock manifest files with version metadata
- CI consistency checks for dataset changes
- Automatic telemetry integration for experiment tracking
"""

import json
import hashlib
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, NamedTuple

logger = logging.getLogger(__name__)


@dataclass
class DatasetManifest:
    """Dataset version manifest containing metadata and hash"""
    name: str
    hash_sha256: str
    num_samples: int
    schema_version: str
    source_url: str
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetManifest':
        """Create manifest from dictionary"""
        return cls(**data)


class ConsistencyResult(NamedTuple):
    """Result of dataset consistency check"""
    is_consistent: bool
    expected_hash: Optional[str]
    actual_hash: Optional[str]
    error_message: Optional[str] = None


def compute_dataset_hash(path: str) -> str:
    """
    Compute deterministic SHA256 hash for a dataset file or directory.
    
    For directories, files are sorted alphabetically and hashed in order
    to ensure deterministic results regardless of filesystem ordering.
    
    Args:
        path: Path to dataset file or directory
        
    Returns:
        Hexadecimal SHA256 hash string
    """
    path_obj = Path(path)
    hasher = hashlib.sha256()
    
    if path_obj.is_file():
        # Single file - hash contents directly
        with open(path_obj, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
    
    elif path_obj.is_dir():
        # Directory - hash all files in sorted order, excluding lock files
        files = []
        for file_path in path_obj.rglob('*'):
            if (file_path.is_file() and 
                not file_path.name.startswith('.') and 
                file_path.name != 'dataset.lock'):
                # Store relative path for consistent hashing
                rel_path = file_path.relative_to(path_obj)
                files.append((str(rel_path), file_path))
        
        # Sort by relative path for deterministic ordering
        files.sort(key=lambda x: x[0])
        
        for rel_path, file_path in files:
            # Hash the relative path first
            hasher.update(rel_path.encode('utf-8'))
            
            # Then hash the file contents
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
    
    else:
        raise ValueError(f"Path does not exist or is not a file/directory: {path}")
    
    return hasher.hexdigest()


def count_dataset_samples(path: str) -> int:
    """
    Count number of samples in a dataset.
    
    For JSON files, counts objects. For JSONL, counts lines.
    For directories, sums across all data files.
    
    Args:
        path: Path to dataset file or directory
        
    Returns:
        Number of samples found
    """
    path_obj = Path(path)
    total_samples = 0
    
    def count_file_samples(file_path: Path) -> int:
        """Count samples in a single file"""
        if not file_path.exists():
            return 0
            
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    elif isinstance(data, dict):
                        # Check for common dataset formats
                        if 'dataset' in data and isinstance(data['dataset'], list):
                            return len(data['dataset'])
                        elif 'samples' in data and isinstance(data['samples'], list):
                            return len(data['samples'])
                        else:
                            return 1  # Single object
                    else:
                        return 1
            
            elif file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return sum(1 for line in f if line.strip())
            
            else:
                # For other file types, assume 1 sample per file
                return 1
                
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not count samples in {file_path}: {e}")
            return 0
    
    if path_obj.is_file():
        total_samples = count_file_samples(path_obj)
    
    elif path_obj.is_dir():
        # Count samples across all data files
        for file_path in path_obj.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.json', '.jsonl']:
                total_samples += count_file_samples(file_path)
    
    return total_samples


def update_dataset_lock(dataset_path: str, name: str, source_url: Optional[str] = None) -> DatasetManifest:
    """
    Create or update dataset.lock file for a dataset.
    
    Args:
        dataset_path: Path to dataset directory or file
        name: Human-readable name for dataset
        source_url: Optional source URL (defaults to dataset_path)
        
    Returns:
        DatasetManifest object with updated information
    """
    path_obj = Path(dataset_path)
    
    if path_obj.is_file():
        lock_file = path_obj.parent / "dataset.lock"
        dataset_dir = path_obj.parent
    else:
        lock_file = path_obj / "dataset.lock"
        dataset_dir = path_obj
    
    # Check if lock file exists to preserve creation time
    existing_created_at = None
    if lock_file.exists():
        try:
            with open(lock_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_created_at = existing_data.get('created_at')
        except (json.JSONDecodeError, IOError):
            logger.warning(f"Could not read existing lock file: {lock_file}")
    
    # Compute current dataset hash and sample count
    dataset_hash = compute_dataset_hash(dataset_path)
    num_samples = count_dataset_samples(dataset_path)
    
    # Create manifest with current timestamp or preserve existing
    created_at = existing_created_at or datetime.now(timezone.utc).isoformat()
    
    manifest = DatasetManifest(
        name=name,
        hash_sha256=dataset_hash,
        num_samples=num_samples,
        schema_version="1.0",
        source_url=source_url or str(dataset_dir.resolve()),
        created_at=created_at
    )
    
    # Write lock file
    try:
        with open(lock_file, 'w', encoding='utf-8') as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Updated dataset lock: {lock_file} (hash: {dataset_hash[:12]}...)")
        
    except IOError as e:
        logger.error(f"Failed to write lock file: {e}")
        raise
    
    return manifest


def check_dataset_consistency(dataset_path: str) -> ConsistencyResult:
    """
    Check if dataset matches its lock file.
    
    Args:
        dataset_path: Path to dataset directory or file
        
    Returns:
        ConsistencyResult with validation information
    """
    path_obj = Path(dataset_path)
    
    if path_obj.is_file():
        lock_file = path_obj.parent / "dataset.lock"
    else:
        lock_file = path_obj / "dataset.lock"
    
    # Check if lock file exists
    if not lock_file.exists():
        return ConsistencyResult(
            is_consistent=False,
            expected_hash=None,
            actual_hash=None,
            error_message=f"No lock file found at {lock_file}"
        )
    
    try:
        # Read expected hash from lock file
        with open(lock_file, 'r', encoding='utf-8') as f:
            lock_data = json.load(f)
            expected_hash = lock_data.get('hash_sha256')
        
        if not expected_hash:
            return ConsistencyResult(
                is_consistent=False,
                expected_hash=None,
                actual_hash=None,
                error_message="Lock file missing hash_sha256 field"
            )
        
        # Compute actual hash
        actual_hash = compute_dataset_hash(dataset_path)
        
        is_consistent = (expected_hash == actual_hash)
        
        return ConsistencyResult(
            is_consistent=is_consistent,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
            error_message=None if is_consistent else "Dataset hash mismatch"
        )
        
    except (json.JSONDecodeError, IOError) as e:
        return ConsistencyResult(
            is_consistent=False,
            expected_hash=None,
            actual_hash=None,
            error_message=f"Error reading lock file: {e}"
        )


def get_dataset_manifest_for_run(dataset_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Get dataset manifests for telemetry tracking.
    
    Args:
        dataset_paths: List of paths to datasets used in a run
        
    Returns:
        List of manifest dictionaries for telemetry
    """
    manifests = []
    
    for dataset_path in dataset_paths:
        path_obj = Path(dataset_path)
        
        if path_obj.is_file():
            lock_file = path_obj.parent / "dataset.lock"
        else:
            lock_file = path_obj / "dataset.lock"
        
        if lock_file.exists():
            try:
                with open(lock_file, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                    manifests.append(manifest_data)
            
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read manifest for {dataset_path}: {e}")
                # Create minimal manifest
                manifests.append({
                    "name": Path(dataset_path).name,
                    "hash_sha256": "unknown",
                    "num_samples": 0,
                    "schema_version": "1.0",
                    "source_url": str(dataset_path),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(e)
                })
        else:
            logger.warning(f"No lock file found for dataset: {dataset_path}")
            # Create minimal manifest for unversioned dataset
            manifests.append({
                "name": Path(dataset_path).name,
                "hash_sha256": "unversioned",
                "num_samples": 0,
                "schema_version": "1.0",
                "source_url": str(dataset_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "error": "No lock file found"
            })
    
    return manifests


def attach_dataset_to_telemetry(dataset_paths: List[str]) -> None:
    """
    Attach dataset manifests to current telemetry run.
    
    Args:
        dataset_paths: List of dataset paths used in current run
    """
    try:
        from backend.app.core.telemetry_sdk import log
        
        manifests = get_dataset_manifest_for_run(dataset_paths)
        
        # Log dataset information to telemetry
        log({
            "datasets": manifests,
            "dataset_count": len(manifests),
            "dataset_attachment_timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        logger.info(f"Attached {len(manifests)} dataset manifests to telemetry")
        
    except ImportError:
        logger.warning("Telemetry SDK not available - dataset manifests not attached")
    except Exception as e:
        logger.error(f"Failed to attach datasets to telemetry: {e}")


class DatasetVersioning:
    """
    Main utility class for dataset versioning operations.
    
    Provides high-level interface for registering datasets,
    checking consistency, and managing version metadata.
    """
    
    def __init__(self, schema_version: str = "1.0"):
        self.schema_version = schema_version
    
    def register(self, dataset_path: str, name: str, source_url: Optional[str] = None) -> DatasetManifest:
        """
        Register a dataset by creating/updating its lock file.
        
        Args:
            dataset_path: Path to dataset file or directory
            name: Human-readable dataset name
            source_url: Optional source URL
            
        Returns:
            DatasetManifest with registration information
        """
        return update_dataset_lock(dataset_path, name, source_url)
    
    def check_consistency(self, dataset_path: str) -> ConsistencyResult:
        """
        Check if dataset is consistent with its lock file.
        
        Args:
            dataset_path: Path to dataset file or directory
            
        Returns:
            ConsistencyResult with validation details
        """
        return check_dataset_consistency(dataset_path)
    
    def get_manifests_for_telemetry(self, dataset_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Get dataset manifests for telemetry integration.
        
        Args:
            dataset_paths: List of dataset paths
            
        Returns:
            List of manifest dictionaries
        """
        return get_dataset_manifest_for_run(dataset_paths)
    
    def attach_to_telemetry(self, dataset_paths: List[str]) -> None:
        """
        Attach datasets to current telemetry run.
        
        Args:
            dataset_paths: List of dataset paths used in run
        """
        attach_dataset_to_telemetry(dataset_paths) 