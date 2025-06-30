#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Training Script

Example script demonstrating how to integrate the telemetry SDK
for experiment tracking and reproducibility.
"""

import argparse
import json
import torch
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

# Import telemetry SDK
from app.utils.telemetry_sdk import init, log, capture_cfg

# Import for S3 heartbeat support
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


@dataclass
class SFTConfig:
    """SFT training configuration"""
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_steps: int = 500
    warmup_steps: int = 50
    save_steps: int = 100
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    output_dir: str = "sft_output"


class HeartbeatWriter:
    """Writes training heartbeat to S3 for spot instance monitoring"""
    
    def __init__(self, s3_bucket: Optional[str] = None, run_id: Optional[str] = None,
                 interval_seconds: int = 300):
        self.s3_bucket = s3_bucket or os.getenv('S3_HEARTBEAT_BUCKET')
        self.run_id = run_id or os.getenv('RUN_ID', f'run_{int(time.time())}')
        self.instance_id = os.getenv('INSTANCE_ID', 'unknown')
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None
        
        if self.s3_bucket and HAS_BOTO3:
            self.s3_client = boto3.client('s3')
            self._start()
        else:
            if self.s3_bucket and not HAS_BOTO3:
                print("⚠️  Heartbeat monitoring disabled: boto3 not installed")
            self.s3_client = None
    
    def _write_heartbeat(self):
        """Write heartbeat to S3"""
        if not self.s3_client:
            return
            
        heartbeat_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'instance_id': self.instance_id,
            'run_id': self.run_id,
            'status': 'alive'
        }
        
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=f'heartbeats/{self.run_id}',
                Body=json.dumps(heartbeat_data),
                ContentType='application/json'
            )
            print(f"💓 Heartbeat written at {heartbeat_data['timestamp']}")
        except Exception as e:
            print(f"❌ Failed to write heartbeat: {e}")
    
    def _heartbeat_loop(self):
        """Background thread for writing heartbeats"""
        while not self._stop_event.is_set():
            self._write_heartbeat()
            self._stop_event.wait(self.interval_seconds)
    
    def _start(self):
        """Start heartbeat thread"""
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        print(f"💓 Heartbeat monitoring started (interval: {self.interval_seconds}s)")
    
    def stop(self):
        """Stop heartbeat thread"""
        if self._thread:
            self._stop_event.set()
            self._thread.join()
            print("💓 Heartbeat monitoring stopped")


def prepare_dummy_dataset():
    """Prepare a dummy dataset for demonstration"""
    dummy_data = [
        {"input": "What is machine learning?", "output": "Machine learning is a field of AI..."},
        {"input": "Explain neural networks", "output": "Neural networks are computational models..."},
        {"input": "What is fine-tuning?", "output": "Fine-tuning is the process of adapting..."},
    ]
    return dummy_data


@capture_cfg
def train_sft_model(config: SFTConfig, heartbeat_writer: Optional[HeartbeatWriter] = None) -> Dict[str, Any]:
    """
    Train SFT model with telemetry tracking.
    
    The @capture_cfg decorator automatically logs the configuration.
    """
    print(f"🚀 Starting SFT training with config: {config}")
    
    # Prepare dataset
    dataset = prepare_dummy_dataset()
    log({"dataset_size": len(dataset)})
    
    # Simulate training loop
    for step in range(config.max_steps):
        # Simulate training step
        simulated_loss = 3.0 * torch.exp(-torch.tensor(step * 0.01))
        simulated_lr = config.learning_rate * (1 - step / config.max_steps)
        
        # Log metrics every 10 steps
        if step % 10 == 0:
            log({
                "train_loss": simulated_loss.item(),
                "learning_rate": simulated_lr,
                "step": step,
                "gpu_memory_mb": 2048  # Simulated
            })
            
            print(f"Step {step}: loss={simulated_loss:.4f}, lr={simulated_lr:.2e}")
        
        # Save checkpoint
        if step % config.save_steps == 0 and step > 0:
            checkpoint_path = Path(config.output_dir) / f"checkpoint-{step}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Create sharded checkpoint if enabled
            if hasattr(config, 'shard_size_gb') and config.shard_size_gb > 0:
                print(f"🔄 Creating sharded checkpoint (shard size: {config.shard_size_gb}GB)...")
                # Note: This is a demo script, so we just simulate sharding
                shards_dir = checkpoint_path / "shards"
                shards_dir.mkdir(exist_ok=True)
                
                # Create demo manifest
                import json
                manifest = {
                    "shards": [f"shard-{i:05d}.safetensors" for i in range(1, 3)],
                    "total_size": 1000000,
                    "num_shards": 2,
                    "shard_size_gb": config.shard_size_gb
                }
                
                with open(shards_dir / "checkpoint.json", 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                print(f"💾 Demo sharded checkpoint created: {shards_dir}")
            
            log({
                "checkpoint_saved": str(checkpoint_path),
                "checkpoint_step": step,
                "sharded": hasattr(config, 'shard_size_gb') and config.shard_size_gb > 0
            })
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Final results
    final_results = {
        "final_loss": simulated_loss.item(),
        "total_steps": config.max_steps,
        "model_saved": True,
        "output_path": config.output_dir
    }
    
    log(final_results)
    return final_results


def main():
    """Main training script entry point"""
    parser = argparse.ArgumentParser(description="SFT Training with Telemetry SDK")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-135M-Instruct", 
                       help="Base model name")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--max-steps", type=int, default=500,
                       help="Maximum training steps")
    parser.add_argument("--output-dir", default="sft_output",
                       help="Output directory")
    parser.add_argument("--run-name", default="sft_experiment",
                       help="Experiment run name")
    
    # Checkpoint sharding arguments
    parser.add_argument("--shard-size-gb", type=float, default=0,
                       help="Shard size in GB (0 = disabled)")
    parser.add_argument("--s3-bucket", default=None,
                       help="S3 bucket for checkpoint storage")
    parser.add_argument("--s3-prefix", default=None,
                       help="S3 prefix/folder for checkpoints")
    parser.add_argument("--s3-endpoint-url", default=None,
                       help="Custom S3 endpoint URL (for MinIO, Wasabi, etc.)")
    parser.add_argument("--resume-from", default=None,
                       help="Resume from checkpoint (supports sharded checkpoints)")
    
    # Heartbeat monitoring arguments
    parser.add_argument("--heartbeat-bucket", default=None,
                       help="S3 bucket for heartbeat monitoring (defaults to --s3-bucket)")
    parser.add_argument("--heartbeat-interval", type=int, default=300,
                       help="Heartbeat interval in seconds (default: 300)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SFTConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        output_dir=args.output_dir
    )
    
    # Add checkpoint sharding configuration
    config.shard_size_gb = args.shard_size_gb
    config.s3_bucket = args.s3_bucket
    config.s3_prefix = args.s3_prefix or f"sft_checkpoints/{args.run_name}/"
    config.s3_endpoint_url = args.s3_endpoint_url
    config.resume_from_checkpoint = args.resume_from
    
    # Initialize telemetry SDK
    print("🔧 Initializing telemetry tracking...")
    run_id = init(
        run_name=args.run_name,
        cfg=config.__dict__,
        backends=['sqlite', 'csv'],  # Use SQLite + CSV fallback
        sqlite_path="experiments.db",
        csv_path="experiments.csv"
    )
    
    print(f"📊 Telemetry initialized: {run_id}")
    
    # Set up heartbeat monitoring for spot instances
    heartbeat_bucket = args.heartbeat_bucket or args.s3_bucket
    heartbeat_writer = None
    
    if heartbeat_bucket:
        heartbeat_writer = HeartbeatWriter(
            s3_bucket=heartbeat_bucket,
            run_id=run_id,
            interval_seconds=args.heartbeat_interval
        )
    
    try:
        # Run training
        results = train_sft_model(config, heartbeat_writer=heartbeat_writer)
        
        print(f"✅ Training completed successfully!")
        print(f"📈 Final loss: {results['final_loss']:.4f}")
        print(f"💾 Model saved to: {results['output_path']}")
        print(f"📊 View results: python -m app.utils.telemetry_sdk.cli {run_id}")
        
    except Exception as e:
        log({"error": str(e), "status": "failed"})
        print(f"❌ Training failed: {e}")
        raise
    finally:
        # Stop heartbeat monitoring
        if heartbeat_writer:
            heartbeat_writer.stop()


if __name__ == "__main__":
    main() 