#!/usr/bin/env python3
"""
DPO Training Script for Generation Head.

Implements Direct Preference Optimization for the generation head,
focusing on text quality, creativity, and factual accuracy.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

import torch
from transformers import AutoTokenizer, TrainingArguments
from datasets import Dataset as HFDataset
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.narrative_engine.model import NarrativeLLM
from backend.app.narrative_engine.config import NarrativeLLMConfig
from backend.app.narrative_engine.dpo_trainer import (
    GenerationDPOTrainer,
    TripleHeadDPOConfig,
    prepare_dpo_dataset,
    load_head_specific_preferences
)
from backend.app.services.training.rlhf_trainer import prepare_preference_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train generation head with DPO")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to base model or SFT checkpoint")
    parser.add_argument("--ref-model-path", type=str, default=None,
                       help="Path to reference model (defaults to model-path)")
    parser.add_argument("--output-dir", type=str, default="models/generation_dpo",
                       help="Output directory for trained model")
    
    # Data arguments
    parser.add_argument("--preference-dir", type=str, default="preference_data",
                       help="Directory containing preference files")
    parser.add_argument("--preference-file", type=str, default=None,
                       help="Specific preference file (overrides preference-dir)")
    
    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DPO beta (KL penalty)")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Number of warmup steps")
    
    # Logging arguments
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="generation-dpo",
                       help="WandB project name")
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"generation_dpo_{Path(args.model_path).stem}"
        )
    
    # Load preferences
    logger.info("Loading preference data...")
    if args.preference_file:
        # Use specific file
        preferences = prepare_preference_dataset(args.preference_file)
    else:
        # Load from directory
        preferences = load_head_specific_preferences(
            Path(args.preference_dir), 
            "generation"
        )
    
    if not preferences:
        logger.error("No preference data found!")
        return
    
    logger.info(f"Loaded {len(preferences)} preference pairs")
    
    # Initialize models
    logger.info(f"Loading model from {args.model_path}")
    config = NarrativeLLMConfig()
    model = NarrativeLLM(config)
    
    # Load checkpoint if available
    if Path(args.model_path).exists():
        # Load adapter or checkpoint
        checkpoint_path = Path(args.model_path)
        if (checkpoint_path / "adapter_config.json").exists():
            # PEFT adapter
            model.load_adapter(str(checkpoint_path), adapter_name="sft")
            model.set_adapter("sft")
        else:
            # Full model checkpoint
            logger.info("Loading full model checkpoint...")
            # Implementation depends on checkpoint format
    
    # Load reference model
    ref_model = None
    if args.ref_model_path:
        logger.info(f"Loading reference model from {args.ref_model_path}")
        ref_model = NarrativeLLM(config)
        # Load reference checkpoint
        if Path(args.ref_model_path).exists():
            # Similar loading logic
            pass
    
    # Create DPO config
    dpo_config = TripleHeadDPOConfig(
        head_type="generation",
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps
    )
    
    # Prepare dataset
    logger.info("Preparing DPO dataset...")
    tokenizer = model.tokenizer
    dpo_dataset = prepare_dpo_dataset(preferences, tokenizer, args.max_length)
    
    # Convert to paired format for DPO (chosen and rejected in same batch)
    paired_data = []
    for item in dpo_dataset:
        # DPO expects pairs in the batch
        paired_data.append({
            'input_ids': tokenizer(item['chosen_text'], 
                                 max_length=args.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors="pt")['input_ids'].squeeze(0),
            'attention_mask': tokenizer(item['chosen_text'],
                                      max_length=args.max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors="pt")['attention_mask'].squeeze(0),
            'labels': tokenizer(item['chosen_text'],
                              max_length=args.max_length,
                              padding='max_length',
                              truncation=True,
                              return_tensors="pt")['input_ids'].squeeze(0),
            'is_chosen': True
        })
        paired_data.append({
            'input_ids': tokenizer(item['rejected_text'],
                                 max_length=args.max_length,
                                 padding='max_length',
                                 truncation=True,
                                 return_tensors="pt")['input_ids'].squeeze(0),
            'attention_mask': tokenizer(item['rejected_text'],
                                      max_length=args.max_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors="pt")['attention_mask'].squeeze(0),
            'labels': tokenizer(item['rejected_text'],
                              max_length=args.max_length,
                              padding='max_length',
                              truncation=True,
                              return_tensors="pt")['input_ids'].squeeze(0),
            'is_chosen': False
        })
    
    # Create HuggingFace dataset
    hf_dataset = HFDataset.from_list(paired_data)
    
    # Split into train/eval
    split_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if args.use_wandb else "none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )
    
    # Create DPO trainer
    trainer = GenerationDPOTrainer(
        model=model,
        ref_model=ref_model,
        config=dpo_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting DPO training for generation head...")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save DPO config
    dpo_config_dict = {
        "training_type": "generation_dpo",
        "base_model": str(args.model_path),
        "ref_model": str(args.ref_model_path) if args.ref_model_path else "self",
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "training_samples": len(train_dataset),
    }
    
    with open(Path(args.output_dir) / "dpo_config.json", "w") as f:
        json.dump(dpo_config_dict, f, indent=2)
    
    logger.info("DPO training complete!")
    
    # Final evaluation
    if len(eval_dataset) > 0:
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        # Save evaluation results
        with open(Path(args.output_dir) / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({"final_eval": eval_results})
            wandb.finish()


if __name__ == "__main__":
    main() 