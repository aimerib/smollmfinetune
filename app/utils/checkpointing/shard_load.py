"""
Checkpoint shard loader for reassembling sharded model states.

Handles loading multiple .safetensors shard files and combining them back
into a complete PyTorch state dictionary, with S3 download support.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from safetensors.torch import load_file
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class ShardLoader:
    """Loads sharded model checkpoints back into complete state dictionaries"""
    
    def __init__(self):
        """Initialize the shard loader"""
        logger.info("ShardLoader initialized")
    
    def _load_manifest(self, manifest_path: Path) -> Dict[str, Any]:
        """
        Load and validate checkpoint manifest.
        
        Args:
            manifest_path: Path to checkpoint.json manifest file
            
        Returns:
            Manifest dictionary
        """
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Validate manifest structure
        required_keys = ['shards', 'total_size', 'num_shards']
        for key in required_keys:
            if key not in manifest:
                raise ValueError(f"Invalid manifest: missing '{key}' field")
        
        if not isinstance(manifest['shards'], list):
            raise ValueError("Manifest 'shards' must be a list")
        
        if len(manifest['shards']) != manifest['num_shards']:
            raise ValueError("Manifest shard count mismatch")
        
        logger.info(f"Loaded manifest: {manifest['num_shards']} shards, {manifest['total_size'] / 1024 / 1024:.1f}MB total")
        return manifest
    
    def _download_manifest_from_s3(self, bucket: str, prefix: str, endpoint_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Download manifest file from S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix
            endpoint_url: Custom S3 endpoint
            
        Returns:
            Manifest dictionary
        """
        s3_config = {}
        if endpoint_url:
            s3_config['endpoint_url'] = endpoint_url
        
        s3_client = boto3.client('s3', **s3_config)
        
        # Download manifest
        manifest_key = f"{prefix}checkpoint.json" if prefix else "checkpoint.json"
        
        try:
            response = s3_client.get_object(Bucket=bucket, Key=manifest_key)
            manifest_data = response['Body'].read().decode('utf-8')
            manifest = json.loads(manifest_data)
            
            logger.info(f"Downloaded manifest from s3://{bucket}/{manifest_key}")
            return manifest
            
        except ClientError as e:
            logger.error(f"Failed to download manifest: {e}")
            raise
    
    def _download_file_from_s3(self, s3_client, bucket: str, s3_key: str, local_path: Path) -> bool:
        """
        Download a single file from S3 with retry logic.
        
        Args:
            s3_client: Boto3 S3 client
            bucket: S3 bucket name
            s3_key: S3 object key
            local_path: Local file path to save to
            
        Returns:
            True if successful, False otherwise
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                s3_client.download_file(bucket, s3_key, str(local_path))
                file_size = local_path.stat().st_size
                logger.info(f"Downloaded {local_path.name} ({file_size / 1024 / 1024:.1f}MB)")
                return True
                
            except ClientError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Download failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to download {s3_key} after {max_retries} attempts: {e}")
                    return False
        
        return False
    
    def _download_missing_shards(self, manifest: Dict[str, Any], local_dir: Path, 
                                bucket: str, prefix: str, endpoint_url: Optional[str] = None) -> bool:
        """
        Download only missing shard files from S3.
        
        Args:
            manifest: Checkpoint manifest
            local_dir: Local directory to download to
            bucket: S3 bucket name
            prefix: S3 key prefix
            endpoint_url: Custom S3 endpoint
            
        Returns:
            True if all missing shards downloaded successfully
        """
        s3_config = {}
        if endpoint_url:
            s3_config['endpoint_url'] = endpoint_url
        
        s3_client = boto3.client('s3', **s3_config)
        
        # Check which shards are missing
        missing_shards = []
        for shard_name in manifest['shards']:
            local_shard_path = local_dir / shard_name
            if not local_shard_path.exists():
                missing_shards.append(shard_name)
        
        if not missing_shards:
            logger.info("All shards already present locally")
            return True
        
        logger.info(f"Downloading {len(missing_shards)} missing shards...")
        
        # Download missing shards
        for shard_name in missing_shards:
            s3_key = f"{prefix}{shard_name}" if prefix else shard_name
            local_path = local_dir / shard_name
            
            if not self._download_file_from_s3(s3_client, bucket, s3_key, local_path):
                return False
        
        return True
    
    def _load_shards_into_state_dict(self, manifest: Dict[str, Any], shard_dir: Path) -> Dict[str, torch.Tensor]:
        """
        Load all shard files and combine into single state dictionary.
        
        Args:
            manifest: Checkpoint manifest
            shard_dir: Directory containing shard files
            
        Returns:
            Combined state dictionary
        """
        state_dict = {}
        
        for i, shard_filename in enumerate(manifest['shards']):
            shard_path = shard_dir / shard_filename
            
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            
            logger.info(f"Loading shard {i+1}/{len(manifest['shards'])}: {shard_filename}")
            
            # Load shard using safetensors
            shard_data = load_file(str(shard_path))
            
            # Check for key conflicts
            for key in shard_data:
                if key in state_dict:
                    raise ValueError(f"Duplicate key '{key}' found in multiple shards")
                state_dict[key] = shard_data[key]
        
        logger.info(f"Loaded {len(state_dict)} tensors from {len(manifest['shards'])} shards")
        return state_dict
    
    def load_from_directory(self, checkpoint_dir: Path) -> Dict[str, torch.Tensor]:
        """
        Load sharded checkpoint from local directory.
        
        Args:
            checkpoint_dir: Directory containing manifest and shard files
            
        Returns:
            Complete state dictionary
        """
        checkpoint_dir = Path(checkpoint_dir)
        manifest_path = checkpoint_dir / "checkpoint.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
        
        # Load manifest
        manifest = self._load_manifest(manifest_path)
        
        # Load and combine shards
        state_dict = self._load_shards_into_state_dict(manifest, checkpoint_dir)
        
        return state_dict
    
    def load_from_s3(self, bucket: str, prefix: str, local_cache_dir: Path, 
                     endpoint_url: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Load sharded checkpoint from S3-compatible storage.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix (folder path)
            local_cache_dir: Local directory to cache downloaded files
            endpoint_url: Custom S3 endpoint (for MinIO, Wasabi, etc.)
            
        Returns:
            Complete state dictionary
        """
        local_cache_dir = Path(local_cache_dir)
        local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download manifest first
        manifest = self._download_manifest_from_s3(bucket, prefix, endpoint_url)
        
        # Save manifest locally for future reference
        local_manifest_path = local_cache_dir / "checkpoint.json"
        with open(local_manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Download missing shards
        if not self._download_missing_shards(manifest, local_cache_dir, bucket, prefix, endpoint_url):
            raise RuntimeError("Failed to download required shard files")
        
        # Load and combine shards
        state_dict = self._load_shards_into_state_dict(manifest, local_cache_dir)
        
        logger.info(f"Successfully loaded checkpoint from s3://{bucket}/{prefix}")
        return state_dict
    
    def verify_checkpoint_integrity(self, checkpoint_dir: Path) -> bool:
        """
        Verify that all shard files are present and loadable.
        
        Args:
            checkpoint_dir: Directory containing checkpoint files
            
        Returns:
            True if checkpoint is complete and valid
        """
        try:
            manifest_path = checkpoint_dir / "checkpoint.json"
            if not manifest_path.exists():
                logger.error("Manifest file missing")
                return False
            
            manifest = self._load_manifest(manifest_path)
            
            # Check all shard files exist
            for shard_name in manifest['shards']:
                shard_path = checkpoint_dir / shard_name
                if not shard_path.exists():
                    logger.error(f"Missing shard file: {shard_name}")
                    return False
            
            # Try loading each shard
            for shard_name in manifest['shards']:
                shard_path = checkpoint_dir / shard_name
                try:
                    load_file(str(shard_path))
                except Exception as e:
                    logger.error(f"Failed to load shard {shard_name}: {e}")
                    return False
            
            logger.info("Checkpoint integrity verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint integrity check failed: {e}")
            return False 