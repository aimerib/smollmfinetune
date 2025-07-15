#!/usr/bin/env python3
"""
Training script for Generation Head Reward Model.

Trains a reward model to evaluate text generation quality, creativity, and factual accuracy.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset as HFDataset
import wandb

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.narrative_engine.reward_models import (
    GenerationRewardModel, 
    load_head_preferences,
    RewardModelOutput
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GenerationPreferenceDataset(Dataset):
    """Dataset for generation head preference pairs"""
    
    def __init__(self, preferences: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.preferences = preferences
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        pref = self.preferences[idx]
        
        # Tokenize prompt + chosen response
        chosen_text = f"{pref['prompt']}\n{pref.get('chosen', '')}"
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize prompt + rejected response
        rejected_text = f"{pref['prompt']}\n{pref.get('rejected', '')}"
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Calculate reward from ratings
        reward = pref.get('reward', 0.5)
        
        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(0),
            'reward_diff': torch.tensor(reward - 0.5, dtype=torch.float)  # Center around 0
        }


class RewardModelTrainer(Trainer):
    """Custom trainer for reward model with preference pairs"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute ranking loss for preference pairs"""
        # Get rewards for chosen samples
        chosen_outputs = model(
            input_ids=inputs['chosen_input_ids'],
            attention_mask=inputs['chosen_attention_mask']
        )
        chosen_rewards = chosen_outputs.reward
        
        # Get rewards for rejected samples
        rejected_outputs = model(
            input_ids=inputs['rejected_input_ids'],
            attention_mask=inputs['rejected_attention_mask']
        )
        rejected_rewards = rejected_outputs.reward
        
        # Ranking loss - chosen should have higher reward
        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # Add margin loss for better separation
        margin = 0.5
        margin_loss = torch.nn.functional.relu(margin - (chosen_rewards - rejected_rewards)).mean()
        
        total_loss = loss + 0.1 * margin_loss
        
        if return_outputs:
            outputs = {
                'loss': total_loss,
                'chosen_rewards': chosen_rewards.mean().item(),
                'rejected_rewards': rejected_rewards.mean().item(),
                'reward_diff': (chosen_rewards - rejected_rewards).mean().item()
            }
            return total_loss, outputs
        
        return total_loss


def main():
    parser = argparse.ArgumentParser(description="Train generation head reward model")
    parser.add_argument("--preference-dir", type=str, default="preference_data",
                       help="Directory containing preference files")
    parser.add_argument("--output-dir", type=str, default="models/generation_rm",
                       help="Output directory for trained model")
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct",
                       help="Base model for reward model")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="generation-reward-model",
            config=vars(args)
        )
    
    # Load preferences
    logger.info(f"Loading generation preferences from {args.preference_dir}")
    preference_file = Path(args.preference_dir) / "generation_preferences.jsonl"
    
    if not preference_file.exists():
        logger.error(f"Preference file not found: {preference_file}")
        return
    
    preferences = load_head_preferences(preference_file, "generation")
    logger.info(f"Loaded {len(preferences)} preference pairs")
    
    # Initialize model and tokenizer
    logger.info(f"Initializing reward model with {args.base_model}")
    model = GenerationRewardModel(base_model_name=args.base_model)
    tokenizer = model.tokenizer
    
    # Create dataset
    dataset = GenerationPreferenceDataset(preferences, tokenizer, args.max_length)
    
    # Convert to HuggingFace dataset format
    hf_dataset = HFDataset.from_list([dataset[i] for i in range(len(dataset))])
    
    # Split into train/eval (90/10)
    split_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
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
    )
    
    # Create trainer
    trainer = RewardModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save model config
    config = {
        "model_type": "generation_reward",
        "base_model": args.base_model,
        "training_preferences": len(preferences),
        "max_length": args.max_length,
    }
    
    with open(Path(args.output_dir) / "reward_model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info("Training complete!")
    
    # Test the model
    if len(eval_dataset) > 0:
        logger.info("Running final evaluation...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({"final_eval": eval_results})
            wandb.finish()


if __name__ == "__main__":
    main() 