"""
Unit tests for checkpoint sharding functionality.

Tests the core functionality of splitting model and optimizer states into 
sharded .safetensors files with manifest-based loading.
"""
import pytest
import torch
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil

from app.utils.checkpointing.shard_save import ShardWriter
from app.utils.checkpointing.shard_load import ShardLoader


class TestShardWriter:
    """Tests for checkpoint shard writing functionality"""
    
    def test_init_shard_writer(self):
        """ShardWriter should initialize with shard size configuration"""
        writer = ShardWriter(shard_size_gb=2.0)
        assert writer.shard_size_gb == 2.0
        assert writer.shard_size_bytes == 2.0 * 1024 * 1024 * 1024
    
    def test_create_dummy_state_dict(self):
        """Should create dummy tensors for testing purposes"""
        writer = ShardWriter(shard_size_gb=0.001)  # 1MB for testing
        
        # Create dummy model state dict
        state_dict = {
            'layer1.weight': torch.randn(1000, 1000),  # ~4MB tensor
            'layer1.bias': torch.randn(1000),
            'layer2.weight': torch.randn(500, 500),    # ~1MB tensor
            'layer2.bias': torch.randn(500),
            'optimizer.state': torch.randn(2000, 2000) # ~16MB tensor
        }
        
        assert len(state_dict) == 5
        assert state_dict['layer1.weight'].shape == (1000, 1000)
    
    def test_calculate_tensor_size(self):
        """Should correctly calculate tensor size in bytes"""
        writer = ShardWriter(shard_size_gb=1.0)
        
        tensor = torch.randn(100, 100)  # 100*100*4 bytes for float32
        size = writer._calculate_tensor_size(tensor)
        assert size == 100 * 100 * 4  # 40,000 bytes
    
    def test_split_state_dict_into_shards(self):
        """Should split large state dict into multiple shards under size limit"""
        writer = ShardWriter(shard_size_gb=0.001)  # 1MB limit
        
        state_dict = {
            'small_tensor': torch.randn(100, 100),     # ~40KB
            'medium_tensor': torch.randn(500, 500),    # ~1MB
            'large_tensor': torch.randn(1000, 1000),   # ~4MB 
        }
        
        shards = writer._split_state_dict_into_shards(state_dict)
        
        # Should create multiple shards since large_tensor exceeds 1MB limit
        assert len(shards) >= 2
        assert all(isinstance(shard, dict) for shard in shards)
    
    def test_save_shards_to_directory(self):
        """Should save shards as .safetensors files with manifest"""
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ShardWriter(shard_size_gb=0.001)
            output_dir = Path(temp_dir)
            
            state_dict = {
                'tensor1': torch.randn(100, 100),
                'tensor2': torch.randn(200, 200),
            }
            
            manifest_path = writer.save_shards(state_dict, output_dir)
            
            # Should create manifest file
            assert manifest_path.exists()
            assert manifest_path.name == "checkpoint.json"
            
            # Should create shard files
            shard_files = list(output_dir.glob("*.safetensors"))
            assert len(shard_files) > 0
    
    def test_manifest_format(self):
        """Manifest should contain expected metadata structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ShardWriter(shard_size_gb=1.0)
            output_dir = Path(temp_dir)
            
            state_dict = {'test_tensor': torch.randn(10, 10)}
            manifest_path = writer.save_shards(state_dict, output_dir)
            
            with open(manifest_path) as f:
                manifest = json.load(f)
            
            # Verify manifest structure
            assert 'shards' in manifest
            assert 'total_size' in manifest
            assert 'num_shards' in manifest
            assert 'shard_size_gb' in manifest
            assert isinstance(manifest['shards'], list)
            assert all(shard.endswith('.safetensors') for shard in manifest['shards'])
    
    @patch('boto3.client')
    def test_upload_to_s3(self, mock_boto):
        """Should upload shards to S3-compatible storage"""
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3
        
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = ShardWriter(shard_size_gb=1.0)
            output_dir = Path(temp_dir)
            
            # Create dummy files
            (output_dir / "shard-00001.safetensors").write_text("dummy")
            (output_dir / "checkpoint.json").write_text('{"shards": ["shard-00001.safetensors"]}')
            
            writer.upload_to_s3(output_dir, "test-bucket", "test-prefix/")
            
            # Should have called upload_file for each file
            assert mock_s3.upload_file.call_count >= 2


class TestShardLoader:
    """Tests for checkpoint shard loading functionality"""
    
    def test_init_shard_loader(self):
        """ShardLoader should initialize correctly"""
        loader = ShardLoader()
        assert loader is not None
    
    def test_load_manifest(self):
        """Should load and validate manifest file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "checkpoint.json"
            manifest_data = {
                'shards': ['shard-00001.safetensors', 'shard-00002.safetensors'],
                'total_size': 1000000,
                'num_shards': 2,
                'shard_size_gb': 1.0
            }
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f)
            
            loader = ShardLoader()
            manifest = loader._load_manifest(manifest_path)
            
            assert manifest == manifest_data
            assert len(manifest['shards']) == 2
    
    def test_download_missing_shards_from_s3(self):
        """Should download only missing shard files from S3"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create one shard locally, leave one missing
            (output_dir / "shard-00001.safetensors").write_text("existing")
            
            manifest = {
                'shards': ['shard-00001.safetensors', 'shard-00002.safetensors']
            }
            
            loader = ShardLoader()
            with patch.object(loader, '_download_file_from_s3') as mock_download:
                loader._download_missing_shards(manifest, output_dir, "bucket", "prefix/")
                
                # Should only download the missing shard
                mock_download.assert_called_once()
                args = mock_download.call_args[0]
                assert 'shard-00002.safetensors' in args[2]  # filename
    
    def test_load_shards_into_state_dict(self):
        """Should combine multiple shards back into single state dict"""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ShardLoader()
            shard_dir = Path(temp_dir)
            
            # Create mock shard files (we'll mock the safetensors loading)
            (shard_dir / "shard-00001.safetensors").write_text("dummy")
            (shard_dir / "shard-00002.safetensors").write_text("dummy")
            
            manifest = {
                'shards': ['shard-00001.safetensors', 'shard-00002.safetensors']
            }
            
            # Mock safetensors.torch.load_file to return test tensors
            with patch('app.utils.checkpointing.shard_load.load_file') as mock_load:
                mock_load.side_effect = [
                    {'tensor1': torch.randn(10, 10)},
                    {'tensor2': torch.randn(20, 20)}
                ]
                
                state_dict = loader._load_shards_into_state_dict(manifest, shard_dir)
                
                assert 'tensor1' in state_dict
                assert 'tensor2' in state_dict
                assert state_dict['tensor1'].shape == (10, 10)
                assert state_dict['tensor2'].shape == (20, 20)
    
    def test_load_from_local_directory(self):
        """Should load complete state dict from local sharded checkpoint"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            
            # Create manifest
            manifest = {
                'shards': ['shard-00001.safetensors'],
                'total_size': 400,
                'num_shards': 1
            }
            manifest_path = checkpoint_dir / "checkpoint.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            
            # Create dummy shard file
            (checkpoint_dir / "shard-00001.safetensors").write_text("dummy")
            
            # Mock the shard loading
            with patch('app.utils.checkpointing.shard_load.load_file') as mock_load:
                test_tensor = torch.randn(10, 10)
                mock_load.return_value = {'test_tensor': test_tensor}
                
                loader = ShardLoader()
                state_dict = loader.load_from_directory(checkpoint_dir)
                
                assert 'test_tensor' in state_dict
                torch.testing.assert_close(state_dict['test_tensor'], test_tensor)
    
    @patch('boto3.client')
    def test_load_from_s3_with_resume(self, mock_boto):
        """Should load from S3 and resume from partial download"""
        mock_s3 = Mock()
        mock_boto.return_value = mock_s3
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = ShardLoader()
            
            # Mock S3 manifest download
            manifest_data = {
                'shards': ['shard-00001.safetensors'],
                'total_size': 400
            }
            
            with patch.object(loader, '_download_manifest_from_s3') as mock_download_manifest:
                mock_download_manifest.return_value = manifest_data
                
                # Mock the download to actually create the shard file
                def mock_download_file(s3_client, bucket, s3_key, local_path):
                    # Create the file to simulate successful download
                    local_path.write_text("dummy_content")
                    return True
                
                with patch.object(loader, '_download_file_from_s3', side_effect=mock_download_file):
                    with patch.object(loader, '_load_shards_into_state_dict') as mock_load_shards:
                        test_state = {'loaded_tensor': torch.randn(5, 5)}
                        mock_load_shards.return_value = test_state
                        
                        state_dict = loader.load_from_s3("bucket", "prefix/", Path(temp_dir))
                        
                        assert state_dict == test_state


class TestIntegration:
    """Integration tests for save/load round trip"""
    
    def test_save_and_load_round_trip(self):
        """Should save shards and reload identical state dict"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            
            # Create original state dict
            original_state = {
                'layer1.weight': torch.randn(100, 100),
                'layer1.bias': torch.randn(100),
                'layer2.weight': torch.randn(50, 50),
            }
            
            # Save as shards
            writer = ShardWriter(shard_size_gb=0.001)  # Small size to force sharding
            manifest_path = writer.save_shards(original_state, checkpoint_dir)
            
            # Load back
            loader = ShardLoader()
            loaded_state = loader.load_from_directory(checkpoint_dir)
            
            # Should be identical
            assert set(original_state.keys()) == set(loaded_state.keys())
            for key in original_state:
                torch.testing.assert_close(original_state[key], loaded_state[key])
    
    def test_large_model_sharding(self):
        """Should handle large models with multiple shards correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            
            # Create large state dict that will definitely be sharded
            original_state = {
                f'layer_{i}.weight': torch.randn(500, 500) 
                for i in range(10)  # 10 * 1MB tensors
            }
            
            writer = ShardWriter(shard_size_gb=0.002)  # 2MB per shard
            manifest_path = writer.save_shards(original_state, checkpoint_dir)
            
            # Should create multiple shards
            shard_files = list(checkpoint_dir.glob("*.safetensors"))
            assert len(shard_files) > 1
            
            # Load and verify
            loader = ShardLoader()
            loaded_state = loader.load_from_directory(checkpoint_dir)
            
            assert len(loaded_state) == len(original_state)
            for key in original_state:
                torch.testing.assert_close(original_state[key], loaded_state[key]) 