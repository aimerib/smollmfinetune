#!/usr/bin/env python3
"""
Triple-Head SFT Training Script

This script provides the interface for R4-6 Triple-Head SFT training.
It delegates to the main SFT implementation with appropriate defaults.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """Main entry point for triple-head SFT training"""
    
    print("🎭 Triple-Head SFT Training Pipeline")
    print("Delegating to main SFT implementation...")
    
    # Import the real SFT script
    try:
        from scripts.run_sft import main as sft_main, SFTConfig
        import json
        
        # Parse arguments specifically for triple-head training
        parser = argparse.ArgumentParser(description="Triple-Head SFT Training (R4-6)")
        parser.add_argument("--character-name", type=str, default="Clara", 
                          help="Character name for training")
        parser.add_argument("--dataset", type=str, 
                          help="Path to training dataset (will generate synthetic if not provided)")
        parser.add_argument("--output-dir", type=str, default="training_output/triple_head_sft", 
                          help="Output directory")
        parser.add_argument("--max-steps", type=int, default=1000, 
                          help="Maximum training steps")
        parser.add_argument("--batch-size", type=int, default=2, 
                          help="Training batch size")
        parser.add_argument("--enable-memory-training", action="store_true", default=True,
                          help="Enable memory head training (Method B)")
        parser.add_argument("--use-wandb", action="store_true", 
                          help="Use WandB for experiment tracking")
        parser.add_argument("--synthetic-data-size", type=int, default=500,
                          help="Size of synthetic dataset to generate")
        
        args = parser.parse_args()
        
        # Create configuration for triple-head training
        config = SFTConfig(
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
            output_dir=args.output_dir,
            character_name=args.character_name,
            dataset_path=args.dataset,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            enable_memory_training=args.enable_memory_training,
            use_wandb=args.use_wandb,
            synthetic_data_size=args.synthetic_data_size,
            
            # Triple-head specific settings
            text_weight=1.0,
            control_weight=1.0,
            memory_weight=1.0,
            memory_embedding_weight=0.7,
            memory_metadata_weight=0.3,
            
            # Training configuration optimized for triple-head
            learning_rate=3e-5,
            warmup_steps=100,
            logging_steps=10,
            eval_steps=100,
            save_steps=200,
            
            # WandB settings
            wandb_project="narrative-triple-head-sft",
            wandb_name=f"triple-head-{args.character_name}",
        )
        
        print(f"📁 Output directory: {config.output_dir}")
        print(f"🎭 Character: {config.character_name}")
        print(f"🧠 Memory training: {'enabled' if config.enable_memory_training else 'disabled'}")
        print(f"📊 Synthetic data size: {config.synthetic_data_size}")
        print(f"🚀 Max steps: {config.max_steps}")
        
        # Save config for reference
        os.makedirs(config.output_dir, exist_ok=True)
        config_path = Path(config.output_dir) / "triple_head_config.json"
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        print(f"💾 Configuration saved to: {config_path}")
        
        # Override sys.argv to pass config to the main SFT script
        sys.argv = [
            "run_sft.py",
            "--character-name", config.character_name,
            "--output-dir", config.output_dir,
            "--max-steps", str(config.max_steps),
            "--batch-size", str(config.batch_size),
            "--synthetic-data-size", str(config.synthetic_data_size),
        ]
        
        if config.dataset_path:
            sys.argv.extend(["--dataset", config.dataset_path])
        if config.use_wandb:
            sys.argv.append("--use-wandb")
        if not config.enable_memory_training:
            sys.argv.append("--disable-memory-training")
        
        # Run the main SFT implementation
        print("\n🔄 Starting SFT training...")
        sft_main()
        
    except ImportError as e:
        print(f"❌ Failed to import SFT implementation: {e}")
        print("Make sure the main SFT script is available at scripts/run_sft.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 