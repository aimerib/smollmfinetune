"""
Checkpoint sharding writer for splitting large model states into manageable chunks.

Handles splitting PyTorch state dictionaries into multiple .safetensors files
under a specified size limit, with manifest-based management.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
from safetensors.torch import save_file
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class ShardWriter:
    """Writes model checkpoints as sharded .safetensors files"""
    
    def __init__(self, shard_size_gb: float = 4.0):
        """
        Initialize the shard writer.
        
        Args:
            shard_size_gb: Maximum size per shard in GB (default 4GB)
        """
        self.shard_size_gb = shard_size_gb
        self.shard_size_bytes = int(shard_size_gb * 1024 * 1024 * 1024)
        logger.info(f"ShardWriter initialized with max shard size: {shard_size_gb}GB")
    
    def _calculate_tensor_size(self, tensor: torch.Tensor) -> int:
        """Calculate tensor size in bytes"""
        return tensor.numel() * tensor.element_size()
    
    def _split_state_dict_into_shards(self, state_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Split state dictionary into multiple shards under size limit.
        
        Args:
            state_dict: PyTorch state dictionary
            
        Returns:
            List of shard dictionaries
        """
        shards = []
        current_shard = {}
        current_shard_size = 0
        
        # Sort keys by tensor size (largest first) for better packing
        sorted_items = sorted(
            state_dict.items(),
            key=lambda item: self._calculate_tensor_size(item[1]),
            reverse=True
        )
        
        for key, tensor in sorted_items:
            tensor_size = self._calculate_tensor_size(tensor)
            
            # If this tensor alone exceeds shard size, put it in its own shard
            if tensor_size > self.shard_size_bytes:
                if current_shard:
                    shards.append(current_shard)
                    current_shard = {}
                    current_shard_size = 0
                
                shards.append({key: tensor})
                logger.warning(f"Large tensor '{key}' ({tensor_size / 1024 / 1024:.1f}MB) exceeds shard size limit")
                continue
            
            # If adding this tensor would exceed limit, start new shard
            if current_shard_size + tensor_size > self.shard_size_bytes:
                if current_shard:
                    shards.append(current_shard)
                current_shard = {}
                current_shard_size = 0
            
            current_shard[key] = tensor
            current_shard_size += tensor_size
        
        # Add final shard if not empty
        if current_shard:
            shards.append(current_shard)
        
        logger.info(f"Split state dict into {len(shards)} shards")
        return shards
    
    def save_shards(self, state_dict: Dict[str, torch.Tensor], output_dir: Path) -> Path:
        """
        Save state dictionary as sharded .safetensors files.
        
        Args:
            state_dict: PyTorch state dictionary to save
            output_dir: Directory to save shards
            
        Returns:
            Path to manifest file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into shards
        shards = self._split_state_dict_into_shards(state_dict)
        
        # Save each shard
        shard_filenames = []
        total_size = 0
        
        for i, shard in enumerate(shards):
            # Generate shard filename with zero-padding
            shard_filename = f"pytorch_model-{i+1:05d}-of-{len(shards):05d}.safetensors"
            shard_path = output_dir / shard_filename
            
            # Save shard using safetensors
            save_file(shard, str(shard_path))
            
            # Calculate size
            shard_size = shard_path.stat().st_size
            total_size += shard_size
            
            shard_filenames.append(shard_filename)
            logger.info(f"Saved shard {i+1}/{len(shards)}: {shard_filename} ({shard_size / 1024 / 1024:.1f}MB)")
        
        # Create manifest
        manifest = {
            "shards": shard_filenames,
            "total_size": total_size,
            "num_shards": len(shards),
            "shard_size_gb": self.shard_size_gb,
            "created_at": time.time(),
            "tensor_count": len(state_dict)
        }
        
        manifest_path = output_dir / "checkpoint.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Checkpoint sharded: {len(shards)} files, {total_size / 1024 / 1024:.1f}MB total")
        logger.info(f"Manifest saved: {manifest_path}")
        
        return manifest_path
    
    def upload_to_s3(self, local_dir: Path, bucket: str, prefix: str = "", 
                     endpoint_url: Optional[str] = None, progress_callback: Optional[callable] = None) -> bool:
        """
        Upload sharded checkpoint to S3-compatible storage.
        
        Args:
            local_dir: Local directory containing shards and manifest
            bucket: S3 bucket name
            prefix: S3 key prefix (folder path)
            endpoint_url: Custom S3 endpoint (for MinIO, Wasabi, etc.)
            progress_callback: Optional progress callback function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize S3 client
            s3_config = {}
            if endpoint_url:
                s3_config['endpoint_url'] = endpoint_url
            
            s3_client = boto3.client('s3', **s3_config)
            
            # Find all files to upload
            files_to_upload = []
            for file_path in local_dir.glob("*"):
                if file_path.is_file():
                    s3_key = f"{prefix}{file_path.name}" if prefix else file_path.name
                    files_to_upload.append((file_path, s3_key))
            
            logger.info(f"Uploading {len(files_to_upload)} files to s3://{bucket}/{prefix}")
            
            # Upload each file with retry logic
            for i, (local_path, s3_key) in enumerate(files_to_upload):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        file_size = local_path.stat().st_size
                        logger.info(f"Uploading {local_path.name} ({file_size / 1024 / 1024:.1f}MB) [{i+1}/{len(files_to_upload)}]")
                        
                        # Use multipart upload for large files
                        if file_size > 100 * 1024 * 1024:  # 100MB threshold  
                            s3_client.upload_file(
                                str(local_path), 
                                bucket, 
                                s3_key,
                                Config=boto3.s3.transfer.TransferConfig(
                                    multipart_threshold=1024 * 25,  # 25MB
                                    max_concurrency=10,
                                    multipart_chunksize=1024 * 25,
                                    use_threads=True
                                )
                            )
                        else:
                            s3_client.upload_file(str(local_path), bucket, s3_key)
                        
                        if progress_callback:
                            progress_callback(i + 1, len(files_to_upload), local_path.name)
                        
                        break  # Success, exit retry loop
                        
                    except ClientError as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            logger.warning(f"Upload failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Failed to upload {local_path.name} after {max_retries} attempts: {e}")
                            return False
            
            logger.info(f"Successfully uploaded checkpoint to s3://{bucket}/{prefix}")
            return True
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False 