#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) Pipeline for Triple-Head Narrative Engine

This script orchestrates the SFT phase, taking a prepared dataset and training 
the NarrativeLLM using the TripleHeadLoss function, while logging metrics to WandB.

Features:
- Triple-head model training (generation, control, memory)
- Synthetic data generation if no dataset provided
- Memory label generation for Method B training
- Comprehensive evaluation and monitoring
- WandB integration for experiment tracking
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dataclasses import dataclass
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import narrative engine components
from narrative_engine import (
    NarrativeLLM,
    NarrativeLLMConfig,
    create_narrative_model,
    TripleHeadLoss,
    DatasetProcessor,
    run_evaluation_suite
)
from narrative_engine.clara_trainer import CLARATrainer
from scripts.generate_synthetic_conversations import SyntheticDataGenerator
from app.utils.openai_client import get_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Configuration for SFT training"""
    
    # Model configuration
    base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    control_head_dim: int = 256
    memory_embedding_dim: int = 768
    memory_metadata_dim: int = 4
    
    # Training configuration
    output_dir: str = "sft_output"
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    max_steps: int = 1000
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    
    # Loss weights
    text_weight: float = 1.0
    control_weight: float = 1.0
    memory_weight: float = 1.0
    memory_embedding_weight: float = 0.7
    memory_metadata_weight: float = 0.3
    
    # Data configuration
    dataset_path: Optional[str] = None
    synthetic_data_size: int = 500
    max_seq_length: int = 512
    character_name: str = "Clara"
    
    # Evaluation configuration
    run_evaluation: bool = True
    evaluation_output: Optional[str] = None
    
    # WandB configuration
    use_wandb: bool = False
    wandb_project: str = "narrative-sft"
    wandb_name: Optional[str] = None
    
    # Memory training configuration
    enable_memory_training: bool = True
    memory_generation_model: str = "gpt-4o-mini"
    memory_diversity_threshold: float = 0.3


class TripleHeadDataset(Dataset):
    """Dataset for triple-head training with memory labels"""
    
    def __init__(self, 
                 samples: List[Dict[str, Any]], 
                 tokenizer,
                 max_length: int = 512,
                 enable_memory_training: bool = True):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_memory_training = enable_memory_training
        
        # Pad token setup
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Created TripleHeadDataset with {len(samples)} samples")
        logger.info(f"Memory training enabled: {enable_memory_training}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract conversation text
        messages = sample.get('messages', [])
        if not messages:
            raise ValueError(f"Sample {idx} has no messages")
        
        # Format conversation for training
        conversation_text = self._format_conversation(messages)
        
        # Tokenize
        encoding = self.tokenizer(
            conversation_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare base tensors
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        
        # Create loss mask (only train on assistant responses)
        loss_mask = self._create_loss_mask(conversation_text, input_ids)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'loss_mask': loss_mask,
        }
        
        # Add control labels if available
        if 'control_labels' in sample:
            control_labels = torch.tensor(sample['control_labels'], dtype=torch.float32)
            result['control_labels'] = control_labels
        
        # Add memory labels if available and enabled
        if self.enable_memory_training and 'memory_labels' in sample:
            memory_labels = torch.tensor(sample['memory_labels'], dtype=torch.float32)
            result['memory_labels'] = memory_labels
        
        return result
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation messages into training text"""
        formatted_parts = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                formatted_parts.append(f"<|system|>\n{content}<|end|>\n")
            elif role == 'user':
                formatted_parts.append(f"<|user|>\n{content}<|end|>\n")
            elif role == 'assistant':
                formatted_parts.append(f"<|assistant|>\n{content}<|end|>\n")
        
        return ''.join(formatted_parts)
    
    def _create_loss_mask(self, conversation_text: str, input_ids: torch.Tensor) -> torch.Tensor:
        """Create loss mask to only train on assistant responses"""
        # Simple approach: train on all tokens for now
        # In production, would parse the conversation format and mask appropriately
        return torch.ones_like(input_ids, dtype=torch.float32)


class TripleHeadTrainer(Trainer):
    """Custom trainer for triple-head architecture"""
    
    def __init__(self, 
                 loss_fn: TripleHeadLoss,
                 enable_memory_training: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        self.enable_memory_training = enable_memory_training
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss using TripleHeadLoss"""
        
        # Forward pass
        outputs = model(**inputs)
        
        # Extract outputs
        text_logits = outputs.get('text_logits', outputs.get('generation_logits'))
        control_logits = outputs.get('action_logits', outputs.get('control_logits'))
        memory_embedding = outputs.get('memory_embedding')
        memory_metadata = outputs.get('memory_metadata')
        
        # Extract labels
        labels = inputs.get('labels')
        control_labels = inputs.get('control_labels')
        memory_labels = inputs.get('memory_labels') if self.enable_memory_training else None
        loss_mask = inputs.get('loss_mask')
        
        # Compute loss using TripleHeadLoss
        loss_dict = self.loss_fn(
            text_logits=text_logits,
            control_logits=control_logits,
            memory_embedding=memory_embedding,
            memory_metadata=memory_metadata,
            labels=labels,
            control_labels=control_labels,
            memory_labels=memory_labels,
            loss_mask=loss_mask
        )
        
        total_loss = loss_dict['total_loss']
        
        # Log individual loss components
        if hasattr(self, 'state') and self.state.global_step % self.args.logging_steps == 0:
            for key, value in loss_dict.items():
                if key != 'total_loss' and torch.is_tensor(value):
                    self.log({f"train/{key}": value.item()})
        
        if return_outputs:
            return total_loss, outputs
        return total_loss


class SyntheticMemoryGenerator:
    """Generate synthetic memory labels for training"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.client = get_client()
        self.model_name = model_name
        
    def generate_memory_labels(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Generate memory labels for a conversation"""
        
        messages = conversation.get('messages', [])
        if not messages:
            return {}
        
        # Extract the last assistant message for memory generation
        assistant_msg = None
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                assistant_msg = msg.get('content', '')
                break
        
        if not assistant_msg:
            return {}
        
        # Generate memory vector and metadata
        memory_prompt = f"""
        Analyze this conversation turn and generate memory information:
        
        Assistant Response: "{assistant_msg}"
        
        Generate a memory profile with:
        1. A semantic embedding vector (768 dimensions, normalized)
        2. Memory metadata (0-1 scale):
           - importance: How important is this moment to remember?
           - surprise: How surprising or unexpected was this interaction?
           - valence: Emotional tone (-1 to 1, converted to 0-1)
           - persistence: How long should this memory last?
        
        Return as JSON with:
        {{
            "embedding": [768 random normalized values],
            "importance": 0.0-1.0,
            "surprise": 0.0-1.0, 
            "valence": 0.0-1.0,
            "persistence": 0.0-1.0
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a memory analysis expert. Generate memory profiles for conversations."},
                    {"role": "user", "content": memory_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse response
            content = response.choices[0].message.content
            memory_data = json.loads(content)
            
            # Validate and create memory labels
            embedding = memory_data.get('embedding', [])
            if len(embedding) != 768:
                # Generate random normalized embedding if not provided
                embedding = np.random.normal(0, 1, 768)
                embedding = embedding / np.linalg.norm(embedding)
                embedding = embedding.tolist()
            
            # Extract metadata
            importance = float(memory_data.get('importance', 0.5))
            surprise = float(memory_data.get('surprise', 0.3))
            valence = float(memory_data.get('valence', 0.5))
            persistence = float(memory_data.get('persistence', 0.5))
            
            # Combine embedding and metadata
            memory_labels = embedding + [importance, surprise, valence, persistence]
            
            return {
                'memory_labels': memory_labels,
                'memory_metadata': {
                    'importance': importance,
                    'surprise': surprise,
                    'valence': valence,
                    'persistence': persistence
                }
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate memory labels: {e}")
            # Return random memory labels as fallback
            embedding = np.random.normal(0, 1, 768)
            embedding = embedding / np.linalg.norm(embedding)
            memory_labels = embedding.tolist() + [0.5, 0.3, 0.5, 0.5]
            
            return {
                'memory_labels': memory_labels,
                'memory_metadata': {
                    'importance': 0.5,
                    'surprise': 0.3,
                    'valence': 0.5,
                    'persistence': 0.5
                }
            }


def load_or_generate_dataset(config: SFTConfig) -> List[Dict[str, Any]]:
    """Load existing dataset or generate synthetic data"""
    
    if config.dataset_path and Path(config.dataset_path).exists():
        logger.info(f"Loading dataset from {config.dataset_path}")
        
        with open(config.dataset_path, 'r') as f:
            if config.dataset_path.endswith('.json'):
                data = json.load(f)
            elif config.dataset_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                raise ValueError(f"Unsupported dataset format: {config.dataset_path}")
        
        logger.info(f"Loaded {len(data)} samples from dataset")
        return data
    
    else:
        logger.info(f"Generating {config.synthetic_data_size} synthetic samples")
        
        # Create synthetic data generator
        generator = SyntheticDataGenerator()
        
        # Generate character profile
        character = {
            'name': config.character_name,
            'personality': 'friendly, helpful, intelligent',
            'background': 'An AI assistant focused on helpful conversation',
            'speaking_style': 'Clear, engaging, and supportive'
        }
        
        # Generate synthetic conversations
        synthetic_data = []
        for i in range(config.synthetic_data_size):
            try:
                conversation = generator.generate_conversation(
                    character=character,
                    num_turns=3,
                    conversation_type='casual_chat'
                )
                synthetic_data.append(conversation)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Generated {i + 1}/{config.synthetic_data_size} synthetic samples")
                    
            except Exception as e:
                logger.warning(f"Failed to generate sample {i}: {e}")
                continue
        
        logger.info(f"Generated {len(synthetic_data)} synthetic samples")
        return synthetic_data


def enhance_dataset_with_labels(dataset: List[Dict[str, Any]], config: SFTConfig) -> List[Dict[str, Any]]:
    """Enhance dataset with control and memory labels"""
    
    enhanced_dataset = []
    memory_generator = SyntheticMemoryGenerator(config.memory_generation_model)
    
    logger.info("Enhancing dataset with control and memory labels...")
    
    for i, sample in enumerate(dataset):
        enhanced_sample = sample.copy()
        
        # Generate control labels (simplified for demo)
        # In production, this would use sophisticated analysis
        control_labels = [0.0] * 64  # Assuming 64 control tokens
        
        # Simple heuristic control token activation
        messages = sample.get('messages', [])
        if messages:
            assistant_msg = ""
            for msg in messages:
                if msg.get('role') == 'assistant':
                    assistant_msg = msg.get('content', '')
                    break
            
            # Activate some control tokens based on content
            if 'happy' in assistant_msg.lower():
                control_labels[0] = 1.0  # mood_happy
            if 'excited' in assistant_msg.lower():
                control_labels[1] = 1.0  # mood_excited
            if '!' in assistant_msg:
                control_labels[2] = 0.8  # enthusiasm
        
        enhanced_sample['control_labels'] = control_labels
        
        # Generate memory labels if enabled
        if config.enable_memory_training:
            memory_data = memory_generator.generate_memory_labels(sample)
            enhanced_sample.update(memory_data)
        
        enhanced_dataset.append(enhanced_sample)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Enhanced {i + 1}/{len(dataset)} samples")
    
    logger.info(f"Enhanced {len(enhanced_dataset)} samples with labels")
    return enhanced_dataset


def setup_training_environment(config: SFTConfig):
    """Setup training environment and WandB"""
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize WandB if enabled
    if config.use_wandb:
        import wandb
        
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name or f"sft-{config.character_name}",
            config=config.__dict__
        )
        logger.info("Initialized WandB tracking")
    
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Main SFT training function"""
    
    parser = argparse.ArgumentParser(description="Triple-Head SFT Training")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--dataset", type=str, help="Path to training dataset")
    parser.add_argument("--output-dir", type=str, default="sft_output", help="Output directory")
    parser.add_argument("--character-name", type=str, default="Clara", help="Character name")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--use-wandb", action="store_true", help="Use WandB for logging")
    parser.add_argument("--synthetic-data-size", type=int, default=500, help="Synthetic data size")
    parser.add_argument("--disable-memory-training", action="store_true", help="Disable memory head training")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = SFTConfig(**config_dict)
    else:
        config = SFTConfig()
    
    # Override config with command line arguments
    if args.dataset:
        config.dataset_path = args.dataset
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.character_name:
        config.character_name = args.character_name
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.use_wandb:
        config.use_wandb = True
    if args.synthetic_data_size:
        config.synthetic_data_size = args.synthetic_data_size
    if args.disable_memory_training:
        config.enable_memory_training = False
    
    logger.info("Starting Triple-Head SFT Training")
    logger.info(f"Configuration: {config}")
    
    # Setup environment
    setup_training_environment(config)
    
    try:
        # Load or generate dataset
        dataset = load_or_generate_dataset(config)
        
        if not dataset:
            raise ValueError("No dataset available for training")
        
        # Enhance dataset with labels
        enhanced_dataset = enhance_dataset_with_labels(dataset, config)
        
        # Create model
        logger.info("Creating triple-head model...")
        model_config = NarrativeLLMConfig(
            base_model_name=config.base_model,
            control_head_dim=config.control_head_dim
        )
        model = NarrativeLLM(model_config)
        tokenizer = model.tokenizer
        
        # Create loss function
        loss_fn = TripleHeadLoss(
            text_weight=config.text_weight,
            control_weight=config.control_weight,
            memory_weight=config.memory_weight,
            memory_embedding_weight=config.memory_embedding_weight,
            memory_metadata_weight=config.memory_metadata_weight
        )
    
        # Create dataset
        train_dataset = TripleHeadDataset(
            enhanced_dataset,
            tokenizer,
            max_length=config.max_seq_length,
            enable_memory_training=config.enable_memory_training
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            logging_steps=config.logging_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            report_to="wandb" if config.use_wandb else "none",
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Create trainer
        trainer = TripleHeadTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            loss_fn=loss_fn,
            enable_memory_training=config.enable_memory_training,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        logger.info("🚀 Starting SFT training...")
        trainer.train()
        
        # Save model
        logger.info("💾 Saving trained model...")
        trainer.save_model()
        
        # Run evaluation if enabled
        if config.run_evaluation:
            logger.info("🧪 Running evaluation...")
            evaluation_results = run_evaluation_suite(
                checkpoint_path=config.output_dir,
                model=model,
                tokenizer=tokenizer,
                training_history=trainer.state.log_history,
                output_json=config.evaluation_output,
                log_to_wandb=config.use_wandb
            )
            
            logger.info(f"Evaluation results: {evaluation_results}")
        
        # Save final config
        with open(Path(config.output_dir) / "sft_config.json", 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        
        logger.info("✅ SFT training completed successfully!")
        
        if config.use_wandb:
            import wandb
            wandb.finish()
        
    except Exception as e:
        logger.error(f"❌ SFT training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 