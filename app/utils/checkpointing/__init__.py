"""
Checkpoint sharding utilities for distributed training.

Enables saving and loading large model checkpoints as sharded .safetensors files
with S3-compatible storage support for fault-tolerant training on spot instances.
"""

from .shard_save import ShardWriter
from .shard_load import ShardLoader

__all__ = ['ShardWriter', 'ShardLoader'] 