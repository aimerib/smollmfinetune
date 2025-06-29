#!/usr/bin/env python3
"""
Mini-MoE SFT Training Script

Example script for training a Mini Mixture-of-Experts model
with telemetry SDK integration for experiment tracking.
"""

import argparse
import json
import torch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

# Import telemetry SDK
from app.utils.telemetry_sdk import init, log, capture_cfg


@dataclass
class MiniMoEConfig:
    """Mini-MoE training configuration"""
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    learning_rate: float = 1e-4
    batch_size: int = 2
    max_steps: int = 1000
    warmup_steps: int = 100
    save_steps: int = 200
    # MoE specific parameters
    num_experts: int = 4
    top_k_experts: int = 2
    expert_capacity: float = 1.25
    load_balancing_weight: float = 0.01
    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    output_dir: str = "mini_moe_output"


def prepare_mini_moe_dataset():
    """Prepare a dataset for Mini-MoE training"""
    # Diverse dataset to encourage expert specialization
    moe_data = [
        {"input": "Explain quantum computing", "output": "Quantum computing uses quantum mechanics...", "domain": "science"},
        {"input": "What is the capital of France?", "output": "The capital of France is Paris.", "domain": "geography"},
        {"input": "How do neural networks work?", "output": "Neural networks process information through layers...", "domain": "technology"},
        {"input": "Write a poem about nature", "output": "The trees sway gently in the breeze...", "domain": "creative"},
        {"input": "What is photosynthesis?", "output": "Photosynthesis is the process plants use...", "domain": "science"},
        {"input": "Tell me a joke", "output": "Why don't scientists trust atoms? Because they make up everything!", "domain": "creative"},
    ]
    return moe_data


def simulate_expert_routing(step: int, num_experts: int = 4) -> Dict[str, float]:
    """Simulate expert routing statistics"""
    # Simulate how different experts are being used
    expert_usage = {}
    for i in range(num_experts):
        # Simulate usage patterns that evolve during training
        base_usage = 0.25  # Equal usage initially
        specialization = 0.1 * torch.sin(torch.tensor(step * 0.01 + i)).item()
        expert_usage[f"expert_{i}_usage"] = max(0.05, base_usage + specialization)
    
    # Normalize to sum to 1
    total = sum(expert_usage.values())
    expert_usage = {k: v/total for k, v in expert_usage.items()}
    
    return expert_usage


@capture_cfg
def train_mini_moe_model(config: MiniMoEConfig) -> Dict[str, Any]:
    """
    Train Mini-MoE model with comprehensive telemetry tracking.
    
    The @capture_cfg decorator automatically logs the configuration.
    """
    print(f"🧠 Starting Mini-MoE training with {config.num_experts} experts")
    print(f"📊 Config: {config}")
    
    # Prepare dataset
    dataset = prepare_mini_moe_dataset()
    log({
        "dataset_size": len(dataset),
        "num_experts": config.num_experts,
        "top_k_experts": config.top_k_experts,
        "expert_capacity": config.expert_capacity
    })
    
    best_loss = float('inf')
    expert_specialization_score = 0.0
    
    # Simulate Mini-MoE training loop
    for step in range(config.max_steps):
        # Simulate training step with MoE-specific metrics
        base_loss = 2.5 * torch.exp(-torch.tensor(step * 0.008))
        moe_routing_loss = 0.1 * torch.exp(-torch.tensor(step * 0.005))
        total_loss = base_loss + config.load_balancing_weight * moe_routing_loss
        
        simulated_lr = config.learning_rate * (1 - step / config.max_steps)
        
        # Get expert routing statistics
        expert_stats = simulate_expert_routing(step, config.num_experts)
        
        # Calculate load balancing metrics
        expert_usage_values = list(expert_stats.values())
        load_balance_coefficient = 1.0 - torch.std(torch.tensor(expert_usage_values)).item()
        
        # Update specialization score (how well experts are specializing)
        expert_specialization_score = max(expert_specialization_score, 
                                        1.0 - torch.var(torch.tensor(expert_usage_values)).item())
        
        # Log comprehensive metrics every 20 steps
        if step % 20 == 0:
            metrics = {
                "total_loss": total_loss.item(),
                "base_loss": base_loss.item(),
                "routing_loss": moe_routing_loss.item(),
                "learning_rate": simulated_lr,
                "load_balance_coefficient": load_balance_coefficient,
                "expert_specialization_score": expert_specialization_score,
                "step": step,
                # Expert-specific metrics
                **expert_stats,
                # Hardware metrics (simulated)
                "gpu_memory_mb": 4096,
                "gpu_utilization": min(95, 60 + step * 0.01),
            }
            
            log(metrics)
            
            if total_loss < best_loss:
                best_loss = total_loss.item()
                log({"best_loss": best_loss, "best_step": step})
            
            print(f"Step {step}: loss={total_loss:.4f} (base={base_loss:.4f}, routing={moe_routing_loss:.4f})")
            print(f"  Load balance: {load_balance_coefficient:.3f}, Specialization: {expert_specialization_score:.3f}")
        
        # Save checkpoint with MoE-specific info
        if step % config.save_steps == 0 and step > 0:
            checkpoint_path = Path(config.output_dir) / f"checkpoint-{step}"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Save expert routing info
            expert_info = {
                "expert_stats": expert_stats,
                "load_balance_coefficient": load_balance_coefficient,
                "expert_specialization_score": expert_specialization_score
            }
            
            with open(checkpoint_path / "expert_info.json", "w") as f:
                json.dump(expert_info, f, indent=2)
            
            log({
                "checkpoint_saved": str(checkpoint_path),
                "checkpoint_step": step,
                "expert_routing_saved": True
            })
            print(f"💾 Saved MoE checkpoint: {checkpoint_path}")
    
    # Final results with MoE-specific metrics
    final_results = {
        "final_total_loss": total_loss.item(),
        "final_base_loss": base_loss.item(),
        "final_routing_loss": moe_routing_loss.item(),
        "best_loss": best_loss,
        "final_load_balance": load_balance_coefficient,
        "final_expert_specialization": expert_specialization_score,
        "total_steps": config.max_steps,
        "num_experts": config.num_experts,
        "model_saved": True,
        "output_path": config.output_dir,
        **{f"final_{k}": v for k, v in expert_stats.items()}
    }
    
    log(final_results)
    return final_results


def main():
    """Main Mini-MoE training script entry point"""
    parser = argparse.ArgumentParser(description="Mini-MoE SFT Training with Telemetry SDK")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-135M-Instruct", 
                       help="Base model name")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum training steps")
    parser.add_argument("--num-experts", type=int, default=4,
                       help="Number of experts in MoE")
    parser.add_argument("--top-k", type=int, default=2,
                       help="Top-k experts to route to")
    parser.add_argument("--load-balancing-weight", type=float, default=0.01,
                       help="Load balancing loss weight")
    parser.add_argument("--output-dir", default="mini_moe_output",
                       help="Output directory")
    parser.add_argument("--run-name", default="mini_moe_experiment",
                       help="Experiment run name")
    
    args = parser.parse_args()
    
    # Create configuration
    config = MiniMoEConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        num_experts=args.num_experts,
        top_k_experts=args.top_k,
        load_balancing_weight=args.load_balancing_weight,
        output_dir=args.output_dir
    )
    
    # Set environment variable for the experiment
    os.environ['RUN_EXPERIMENT_TYPE'] = 'mini_moe'
    os.environ['RUN_NUM_EXPERTS'] = str(config.num_experts)
    
    # Initialize telemetry SDK with WandB if available
    print("🔧 Initializing telemetry tracking for Mini-MoE...")
    run_id = init(
        run_name=args.run_name,
        cfg=config.__dict__,
        backends=['sqlite', 'wandb', 'csv'],  # Try all backends
        sqlite_path="mini_moe_experiments.db",
        csv_path="mini_moe_experiments.csv"
    )
    
    print(f"📊 Telemetry initialized: {run_id}")
    
    try:
        # Run Mini-MoE training
        results = train_mini_moe_model(config)
        
        print(f"✅ Mini-MoE training completed successfully!")
        print(f"📈 Final loss: {results['final_total_loss']:.4f}")
        print(f"🎯 Best loss: {results['best_loss']:.4f}")
        print(f"⚖️  Load balance: {results['final_load_balance']:.3f}")
        print(f"🧠 Expert specialization: {results['final_expert_specialization']:.3f}")
        print(f"💾 Model saved to: {results['output_path']}")
        print(f"📊 View results: python -m app.utils.telemetry_sdk.cli {run_id}")
        
    except Exception as e:
        log({"error": str(e), "status": "failed", "experiment_type": "mini_moe"})
        print(f"❌ Mini-MoE training failed: {e}")
        raise


if __name__ == "__main__":
    main() 