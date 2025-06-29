#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Training Script

Example script demonstrating how to integrate the telemetry SDK
for experiment tracking and reproducibility.
"""

import argparse
import json
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

# Import telemetry SDK
from app.utils.telemetry_sdk import init, log, capture_cfg


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


def prepare_dummy_dataset():
    """Prepare a dummy dataset for demonstration"""
    dummy_data = [
        {"input": "What is machine learning?", "output": "Machine learning is a field of AI..."},
        {"input": "Explain neural networks", "output": "Neural networks are computational models..."},
        {"input": "What is fine-tuning?", "output": "Fine-tuning is the process of adapting..."},
    ]
    return dummy_data


@capture_cfg
def train_sft_model(config: SFTConfig) -> Dict[str, Any]:
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
            
            log({
                "checkpoint_saved": str(checkpoint_path),
                "checkpoint_step": step
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
    
    args = parser.parse_args()
    
    # Create configuration
    config = SFTConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        output_dir=args.output_dir
    )
    
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
    
    try:
        # Run training
        results = train_sft_model(config)
        
        print(f"✅ Training completed successfully!")
        print(f"📈 Final loss: {results['final_loss']:.4f}")
        print(f"💾 Model saved to: {results['output_path']}")
        print(f"📊 View results: python -m app.utils.telemetry_sdk.cli {run_id}")
        
    except Exception as e:
        log({"error": str(e), "status": "failed"})
        print(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main() 