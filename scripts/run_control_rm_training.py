#!/usr/bin/env python3
"""
Training script for Control Head Reward Model.

Trains a reward model to evaluate emotional appropriateness and personality consistency.
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
from transformers import TrainingArguments, Trainer
from datasets import Dataset as HFDataset
import wandb
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.narrative_engine.reward_models import (
    ControlRewardModel, 
    load_head_preferences,
    prepare_control_preferences
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ControlPreferenceDataset(Dataset):
    """Dataset for control head preference pairs"""
    
    def __init__(self, preferences: List[Dict[str, Any]], num_control_tokens: int = 64):
        self.preferences = preferences
        self.num_control_tokens = num_control_tokens
        
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        pref = self.preferences[idx]
        
        # In practice, these would be actual control token outputs from the model
        # For now, using mock data based on ratings
        reward = pref.get('reward', 0.5)
        
        # Create mock control tokens based on emotional ratings
        emotional_appropriateness = pref.get('emotional_appropriateness', 5) / 10
        personality_consistency = pref.get('personality_consistency', 5) / 10
        mood_matching = pref.get('mood_matching', 5) / 10
        
        # Generate control token distribution
        control_tokens = torch.zeros(self.num_control_tokens)
        
        # Activate tokens based on ratings (mock behavior)
        if emotional_appropriateness > 0.7:
            control_tokens[0:10] = torch.rand(10) * emotional_appropriateness
        if personality_consistency > 0.7:
            control_tokens[10:20] = torch.rand(10) * personality_consistency
        if mood_matching > 0.7:
            control_tokens[20:30] = torch.rand(10) * mood_matching
            
        # Add some noise
        control_tokens += torch.rand(self.num_control_tokens) * 0.1
        
        # Normalize to probabilities
        control_tokens = torch.sigmoid(control_tokens)
        
        return {
            'control_tokens': control_tokens,
            'reward': torch.tensor(reward, dtype=torch.float),
            'emotional_appropriateness': torch.tensor(emotional_appropriateness),
            'personality_consistency': torch.tensor(personality_consistency),
            'mood_matching': torch.tensor(mood_matching),
        }


class ControlRewardTrainer(Trainer):
    """Custom trainer for control reward model"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute regression loss for reward prediction"""
        # Get predicted rewards
        outputs = model(control_tokens=inputs['control_tokens'])
        predicted_rewards = outputs.reward
        
        # Get true rewards
        true_rewards = inputs['reward']
        
        # MSE loss for reward prediction
        loss = torch.nn.functional.mse_loss(predicted_rewards, true_rewards)
        
        # Add consistency regularization
        # Penalize large differences in rewards for similar emotional scores
        if 'emotional_appropriateness' in inputs:
            emotion_diff = torch.abs(inputs['emotional_appropriateness'].unsqueeze(1) - 
                                   inputs['emotional_appropriateness'].unsqueeze(0))
            reward_diff = torch.abs(predicted_rewards.unsqueeze(1) - 
                                  predicted_rewards.unsqueeze(0))
            
            # Where emotions are similar, rewards should be similar
            consistency_loss = (reward_diff * torch.exp(-5 * emotion_diff)).mean()
            loss = loss + 0.1 * consistency_loss
        
        if return_outputs:
            return loss, {
                'loss': loss,
                'predicted_rewards': predicted_rewards.mean().item(),
                'true_rewards': true_rewards.mean().item(),
            }
        
        return loss


def main():
    parser = argparse.ArgumentParser(description="Train control head reward model")
    parser.add_argument("--preference-dir", type=str, default="preference_data",
                       help="Directory containing preference files")
    parser.add_argument("--output-dir", type=str, default="models/control_rm",
                       help="Output directory for trained model")
    parser.add_argument("--num-control-tokens", type=int, default=64,
                       help="Number of control tokens")
    parser.add_argument("--hidden-size", type=int, default=256,
                       help="Hidden size for reward model")
    parser.add_argument("--num-epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="control-reward-model",
            config=vars(args)
        )
    
    # Load preferences
    logger.info(f"Loading control preferences from {args.preference_dir}")
    preference_file = Path(args.preference_dir) / "control_preferences.jsonl"
    
    if not preference_file.exists():
        logger.error(f"Preference file not found: {preference_file}")
        return
    
    preferences = load_head_preferences(preference_file, "control")
    logger.info(f"Loaded {len(preferences)} preference pairs")
    
    # Process preferences for control head
    processed_preferences = prepare_control_preferences(preferences)
    
    # Initialize model
    logger.info(f"Initializing control reward model")
    model = ControlRewardModel(
        num_control_tokens=args.num_control_tokens,
        hidden_size=args.hidden_size
    )
    
    # Create dataset
    dataset = ControlPreferenceDataset(preferences, args.num_control_tokens)
    
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
        warmup_steps=50,
        logging_steps=10,
        eval_steps=25,
        save_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if args.use_wandb else "none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = ControlRewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    
    # Save model config
    config = {
        "model_type": "control_reward",
        "num_control_tokens": args.num_control_tokens,
        "hidden_size": args.hidden_size,
        "training_preferences": len(preferences),
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