#!/usr/bin/env python3
"""
Coordinated DPO Training Script for Triple-Head Architecture.

Implements simultaneous Direct Preference Optimization across all three heads
with cross-head regularization and balanced optimization.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

import torch
from transformers import TrainingArguments
from datasets import Dataset as HFDataset
import wandb
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.narrative_engine.model import NarrativeLLM
from backend.app.narrative_engine.config import NarrativeLLMConfig
from backend.app.narrative_engine.dpo_trainer import (
    CoordinatedDPOTrainer,
    TripleHeadDPOConfig,
    load_head_specific_preferences
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_all_preferences(preference_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load preferences for all three heads"""
    preferences = {}
    
    for head_type in ["generation", "control", "memory"]:
        prefs = load_head_specific_preferences(preference_dir, head_type)
        if prefs:
            preferences[head_type] = prefs
            logger.info(f"Loaded {len(prefs)} {head_type} preferences")
        else:
            logger.warning(f"No preferences found for {head_type} head")
    
    return preferences


def align_preferences(preferences: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Align preferences across heads to create unified training samples.
    
    For conversations that have ratings across all heads, combine them.
    For missing ratings, use defaults or skip.
    """
    aligned = []
    
    # Create index by conversation_id
    conv_index = {}
    
    for head_type, head_prefs in preferences.items():
        for pref in head_prefs:
            conv_id = pref.get('conversation_id', f"{head_type}_{len(conv_index)}")
            if conv_id not in conv_index:
                conv_index[conv_id] = {}
            conv_index[conv_id][head_type] = pref
    
    # Combine preferences for each conversation
    for conv_id, heads_data in conv_index.items():
        # Skip if we don't have all three heads
        # (or we could use defaults for missing heads)
        if len(heads_data) < 2:  # At least 2 heads needed
            continue
        
        combined = {
            'conversation_id': conv_id,
            'prompt': heads_data.get('generation', {}).get('prompt', ''),
            'chosen': heads_data.get('generation', {}).get('chosen', ''),
            'rejected': heads_data.get('generation', {}).get('rejected', ''),
        }
        
        # Add head-specific ratings
        for head_type in ['generation', 'control', 'memory']:
            if head_type in heads_data:
                combined[f'{head_type}_data'] = heads_data[head_type]
            else:
                # Use default ratings if missing
                combined[f'{head_type}_data'] = {
                    'reward': 0.5,
                    'coordination': 5
                }
        
        aligned.append(combined)
    
    return aligned


def prepare_triple_head_dataset(aligned_preferences: List[Dict[str, Any]], 
                              tokenizer, max_length: int = 512) -> HFDataset:
    """Prepare dataset for coordinated DPO training"""
    processed_data = []
    
    for pref in aligned_preferences:
        # Basic text data
        prompt = pref['prompt']
        chosen = pref['chosen']
        rejected = pref.get('rejected', chosen)  # Use chosen as rejected if missing
        
        # Format for model
        chosen_text = f"{prompt}\n{chosen}"
        rejected_text = f"{prompt}\n{rejected}"
        
        # Tokenize
        chosen_encoding = tokenizer(
            chosen_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_encoding = tokenizer(
            rejected_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Create paired samples for DPO
        # Chosen sample
        processed_data.append({
            'input_ids': chosen_encoding['input_ids'].squeeze(0),
            'attention_mask': chosen_encoding['attention_mask'].squeeze(0),
            'labels': chosen_encoding['input_ids'].squeeze(0),
            'is_chosen': True,
            'generation_reward': pref['generation_data'].get('reward', 0.7),
            'control_reward': pref['control_data'].get('reward', 0.7),
            'memory_reward': pref['memory_data'].get('reward', 0.7),
        })
        
        # Rejected sample
        processed_data.append({
            'input_ids': rejected_encoding['input_ids'].squeeze(0),
            'attention_mask': rejected_encoding['attention_mask'].squeeze(0),
            'labels': rejected_encoding['input_ids'].squeeze(0),
            'is_chosen': False,
            'generation_reward': 0.3,  # Lower reward for rejected
            'control_reward': 0.3,
            'memory_reward': 0.3,
        })
    
    return HFDataset.from_list(processed_data)


def main():
    parser = argparse.ArgumentParser(description="Coordinated DPO training for all heads")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to base model or SFT checkpoint")
    parser.add_argument("--ref-model-path", type=str, default=None,
                       help="Path to reference model (defaults to model-path)")
    parser.add_argument("--output-dir", type=str, default="models/triple_head_dpo",
                       help="Output directory for trained model")
    
    # Data arguments
    parser.add_argument("--preference-dir", type=str, default="preference_data",
                       help="Directory containing preference files for all heads")
    
    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Training batch size (will be doubled for pairs)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                       help="Base learning rate")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DPO beta (KL penalty)")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Head-specific weights
    parser.add_argument("--generation-weight", type=float, default=1.0,
                       help="Weight for generation head loss")
    parser.add_argument("--control-weight", type=float, default=0.8,
                       help="Weight for control head loss")
    parser.add_argument("--memory-weight", type=float, default=0.6,
                       help="Weight for memory head loss")
    parser.add_argument("--coordination-weight", type=float, default=0.4,
                       help="Weight for cross-head coordination")
    
    # Advanced features
    parser.add_argument("--cross-head-regularization", type=float, default=0.01,
                       help="Cross-head regularization strength")
    parser.add_argument("--emotional-arc-weight", type=float, default=0.2,
                       help="Emotional arc preservation weight")
    
    # Logging arguments
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="triple-head-dpo",
                       help="WandB project name")
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"coordinated_dpo_{Path(args.model_path).stem}"
        )
    
    # Load preferences for all heads
    logger.info(f"Loading preferences from {args.preference_dir}")
    all_preferences = load_all_preferences(Path(args.preference_dir))
    
    if not all_preferences:
        logger.error("No preference data found for any head!")
        return
    
    # Align preferences across heads
    logger.info("Aligning preferences across heads...")
    aligned_preferences = align_preferences(all_preferences)
    logger.info(f"Created {len(aligned_preferences)} aligned preference samples")
    
    if not aligned_preferences:
        logger.error("No aligned preferences found!")
        return
    
    # Initialize models
    logger.info(f"Loading model from {args.model_path}")
    config = NarrativeLLMConfig()
    model = NarrativeLLM(config)
    
    # Load checkpoint if available
    if Path(args.model_path).exists():
        checkpoint_path = Path(args.model_path)
        if (checkpoint_path / "adapter_config.json").exists():
            model.load_adapter(str(checkpoint_path), adapter_name="sft")
            model.set_adapter("sft")
        else:
            logger.info("Loading full model checkpoint...")
    
    # Load reference model
    ref_model = None
    if args.ref_model_path:
        logger.info(f"Loading reference model from {args.ref_model_path}")
        ref_model = NarrativeLLM(config)
        # Load reference checkpoint
    
    # Create DPO config
    dpo_config = TripleHeadDPOConfig(
        head_type="all",
        learning_rate=args.learning_rate,
        beta=args.beta,
        head_specific_lr={
            "generation": args.learning_rate,
            "control": args.learning_rate * 0.5,  # Lower LR for control
            "memory": args.learning_rate * 0.2    # Even lower for memory
        },
        generation_weight=args.generation_weight,
        control_weight=args.control_weight,
        memory_weight=args.memory_weight,
        coordination_weight=args.coordination_weight,
        cross_head_regularization=args.cross_head_regularization,
        emotional_arc_weight=args.emotional_arc_weight,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
    )
    
    # Prepare dataset
    logger.info("Preparing coordinated DPO dataset...")
    tokenizer = model.tokenizer
    dataset = prepare_triple_head_dataset(aligned_preferences, tokenizer, args.max_length)
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size * 2,  # *2 for pairs
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=200,  # More warmup for complex training
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
        group_by_length=True,  # Group similar lengths for efficiency
    )
    
    # Create coordinated DPO trainer
    trainer = CoordinatedDPOTrainer(
        model=model,
        ref_model=ref_model,
        config=dpo_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting coordinated DPO training for all heads...")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    logger.info(f"Head weights: Gen={args.generation_weight}, Ctrl={args.control_weight}, Mem={args.memory_weight}")
    
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save comprehensive config
    config_dict = {
        "training_type": "coordinated_dpo",
        "base_model": str(args.model_path),
        "ref_model": str(args.ref_model_path) if args.ref_model_path else "self",
        "beta": args.beta,
        "learning_rates": dpo_config.head_specific_lr,
        "head_weights": {
            "generation": args.generation_weight,
            "control": args.control_weight,
            "memory": args.memory_weight,
            "coordination": args.coordination_weight
        },
        "cross_head_regularization": args.cross_head_regularization,
        "emotional_arc_weight": args.emotional_arc_weight,
        "num_epochs": args.num_epochs,
        "training_samples": len(train_dataset),
        "aligned_conversations": len(aligned_preferences),
    }
    
    with open(Path(args.output_dir) / "coordinated_dpo_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info("Coordinated DPO training complete!")
    
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
            
            # Log head-specific metrics if available
            if 'loss/generation' in eval_results:
                wandb.log({
                    "final_generation_loss": eval_results.get('loss/generation', 0),
                    "final_control_loss": eval_results.get('loss/control', 0),
                    "final_memory_loss": eval_results.get('loss/memory', 0),
                    "final_regularization_loss": eval_results.get('loss/regularization', 0),
                })
            
            wandb.finish()


if __name__ == "__main__":
    main() 