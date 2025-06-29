#!/usr/bin/env python3
"""
Lightweight SFT Training for CI

This script runs a minimal training job for continuous integration testing.
It trains a tiny model for ~50 steps on a 100-sample synthetic dataset to
catch regressions in training pipeline without requiring GPU resources.

Usage:
    python scripts/run_sft_ci.py --output-dir ci_output --max-steps 50
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Add the app directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

# Import telemetry SDK for experiment tracking
from utils.telemetry_sdk import init, log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CITrainingConfig:
    """Configuration for CI training run"""
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"  # Tiny model
    learning_rate: float = 2e-4
    batch_size: int = 1  # Small batch for CI
    max_steps: int = 50
    output_dir: str = "ci_training_output"
    quick_mode: bool = False
    dataset_size: int = 100  # Small synthetic dataset


def generate_toy_dataset(size: int = 100) -> List[Dict[str, Any]]:
    """
    Generate a small synthetic dataset for CI testing.
    
    Args:
        size: Number of samples to generate
        
    Returns:
        List of training samples in chat format
    """
    logger.info(f"Generating {size} toy training samples...")
    
    # Simple synthetic conversations for testing
    toy_samples = []
    
    for i in range(size):
        sample = {
            "messages": [
                {
                    "role": "user", 
                    "content": f"Hello! What's your favorite hobby? (Sample {i+1})"
                },
                {
                    "role": "assistant", 
                    "content": f"I enjoy reading and learning new things! There's always something fascinating to discover. (Response {i+1})"
                }
            ]
        }
        toy_samples.append(sample)
    
    logger.info(f"Generated {len(toy_samples)} toy samples")
    return toy_samples


def save_dataset_to_jsonl(samples: List[Dict[str, Any]], output_path: Path) -> None:
    """Save dataset samples to JSONL format"""
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    logger.info(f"Saved dataset to {output_path}")


def run_ci_training(config: CITrainingConfig) -> Dict[str, Any]:
    """
    Run lightweight training for CI testing.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary with training results
    """
    logger.info("Starting CI training...")
    
    try:
        # Import training dependencies
        import torch
        from transformers import (
            AutoTokenizer, 
            AutoModelForCausalLM,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling
        )
        from datasets import Dataset
        
        # Force CPU usage for CI
        device = "cpu"
        torch.cuda.is_available = lambda: False  # Force CPU mode
        
        logger.info(f"Using device: {device}")
        
        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate toy dataset
        toy_samples = generate_toy_dataset(config.dataset_size)
        
        # Save dataset for debugging
        dataset_path = output_dir / "toy_dataset.jsonl"
        save_dataset_to_jsonl(toy_samples, dataset_path)
        
        # Load tokenizer and model
        logger.info(f"Loading model: {config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,  # Use FP32 for CPU
            device_map=None  # Don't use device mapping on CPU
        )
        
        # Prepare dataset
        def tokenize_function(examples):
            # Convert chat format to text for language modeling
            texts = []
            
            # Handle batch of examples
            messages_batch = examples['messages']
            for messages in messages_batch:
                if isinstance(messages, list):
                    # Join messages into single text for causal LM
                    text_parts = []
                    for msg in messages:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        text_parts.append(f"{role}: {content}")
                    text = "\n".join(text_parts)
                else:
                    text = str(messages)
                texts.append(text)
            
            # Tokenize with proper settings for causal LM
            tokenized = tokenizer(
                texts,
                padding="max_length",  # Use max_length padding for consistent batches
                truncation=True,
                max_length=256,  # Short sequences for CI
                return_tensors=None  # Don't return tensors here, let the data collator handle it
            )
            
            # For causal LM, labels should be the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_list(toy_samples)
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=dataset.column_names  # Remove original columns to avoid conflicts
        )
        
        # Training arguments optimized for CI
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            max_steps=config.max_steps,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=config.learning_rate,
            weight_decay=0.01,
            logging_steps=max(1, config.max_steps // 5),  # Log a few times during training
            save_steps=config.max_steps,  # Save at end only
            save_total_limit=1,
            remove_unused_columns=True,  # Remove unused columns (this is default)
            dataloader_num_workers=0,  # No multiprocessing for CI
            dataloader_pin_memory=False,  # Disable pin memory for CPU
            fp16=False,  # No mixed precision on CPU
            report_to=[],  # Disable wandb/tensorboard for CI
            push_to_hub=False,  # Don't push to hub
            ddp_find_unused_parameters=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Run training
        logger.info("Starting training...")
        start_time = time.time()
        
        training_result = trainer.train()
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        # Save the model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed in {training_duration:.1f}s")
        
        # Log metrics to telemetry
        final_loss = training_result.training_loss
        log({
            "final_loss": final_loss,
            "training_duration": training_duration,
            "max_steps": config.max_steps,
            "dataset_size": config.dataset_size,
            "status": "completed"
        })
        
        return {
            "final_loss": final_loss,
            "training_duration": training_duration,
            "output_path": str(output_dir),
            "checkpoint_files": list(output_dir.glob("**/*.safetensors"))
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        log({"error": str(e), "status": "failed"})
        raise


def main():
    """Main entry point for CI training script"""
    parser = argparse.ArgumentParser(description="Lightweight SFT Training for CI")
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                       help="Base model name")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--max-steps", type=int, default=50,
                       help="Maximum training steps")
    parser.add_argument("--output-dir", default="ci_training_output",
                       help="Output directory")
    parser.add_argument("--dataset-size", type=int, default=100,
                       help="Size of synthetic dataset")
    parser.add_argument("--quick-mode", action="store_true",
                       help="Enable quick mode for testing")
    
    args = parser.parse_args()
    
    # Create configuration
    config = CITrainingConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        dataset_size=args.dataset_size,
        quick_mode=args.quick_mode
    )
    
    # Initialize telemetry tracking
    logger.info("🔧 Initializing telemetry tracking...")
    run_id = init(
        run_name="ci_sft_training",
        cfg=config.__dict__,
        backends=['csv'],  # Use CSV only for CI (no SQLite/WandB dependencies)
    )
    
    logger.info(f"📊 Telemetry initialized: {run_id}")
    
    try:
        # Run training
        results = run_ci_training(config)
        
        logger.info("✅ CI Training completed successfully!")
        logger.info(f"📈 Final loss: {results['final_loss']:.4f}")
        logger.info(f"⏱️  Duration: {results['training_duration']:.1f}s")
        logger.info(f"💾 Output: {results['output_path']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ CI Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 