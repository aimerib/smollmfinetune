#!/usr/bin/env python3
"""
C.L.A.R.A. Loop Training Manager

Extends the existing TrainingManager to support dual-head training for
the Control Layer for Attentional Recirculation Augmentation architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
from transformers import (
    TrainingArguments, 
    DataCollatorForLanguageModeling,
    TrainerCallback,
    Trainer
)
import sys
sys.path.append('..')

# Import our existing infrastructure
from app.utils.training import TrainingManager, TrainingCallback
from narrative_engine.model import CLARALoopSmolLM, CLARALoopConfig, create_clara_loop_model

logger = logging.getLogger(__name__)


@dataclass 
class CLARADataCollator:
    """
    Data collator for C.L.A.R.A. Loop dual-head training.
    
    Handles both generation targets and control token labels.
    """
    
    tokenizer: Any
    mlm: bool = False
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch with both generation and control labels"""
        
        # Standard language modeling collation
        batch = self._collate_generation_targets(features)
        
        # Add control token labels
        control_labels = self._collate_control_labels(features)
        if control_labels is not None:
            batch["control_labels"] = control_labels
        
        return batch
    
    def _collate_generation_targets(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Handle standard language modeling collation"""
        # Use standard DataCollatorForLanguageModeling behavior
        standard_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=self.mlm,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors
        )
        
        # Extract features without control labels for standard collation
        gen_features = []
        for feature in features:
            gen_feature = {k: v for k, v in feature.items() if k != 'control_labels'}
            gen_features.append(gen_feature)
        
        return standard_collator(gen_features)
    
    def _collate_control_labels(self, features: List[Dict[str, Any]]) -> Optional[torch.Tensor]:
        """Extract and pad control token labels"""
        if not any('control_labels' in feature for feature in features):
            return None
        
        # Extract control labels (multi-hot encoded vectors)
        control_labels = []
        for feature in features:
            if 'control_labels' in feature:
                control_labels.append(feature['control_labels'])
            else:
                # Default to zeros if no control labels provided
                control_labels.append([0.0] * 31)  # Our 31 control tokens
        
        return torch.tensor(control_labels, dtype=torch.float32)


class CLARATrainer(Trainer):
    """
    Custom trainer for C.L.A.R.A. Loop dual-head architecture.
    
    Handles both generation loss and control token loss.
    """
    
    def __init__(self, control_loss_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.control_loss_weight = control_loss_weight
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute combined loss for dual heads"""
        
        # Forward pass through model
        outputs = model(**inputs)
        
        # Extract losses
        losses = outputs.get('losses', {})
        
        # Combine losses
        total_loss = 0
        loss_components = {}
        
        if 'generation_loss' in losses:
            gen_loss = losses['generation_loss']
            total_loss += gen_loss
            loss_components['generation_loss'] = gen_loss.item()
        
        if 'control_loss' in losses:
            control_loss = losses['control_loss']
            weighted_control_loss = control_loss * self.control_loss_weight
            total_loss += weighted_control_loss
            loss_components['control_loss'] = control_loss.item()
            loss_components['weighted_control_loss'] = weighted_control_loss.item()
        
        # Log loss components
        if hasattr(self, 'state') and self.state.global_step % self.args.logging_steps == 0:
            for key, value in loss_components.items():
                self.log({f"train/{key}": value})
        
        if return_outputs:
            return total_loss, outputs
        return total_loss


class CLARATrainingCallback(TrainingCallback):
    """
    Enhanced training callback for C.L.A.R.A. Loop that tracks
    both generation and control token metrics.
    """
    
    def __init__(self, control_tokens: List[Dict[str, Any]], **kwargs):
        super().__init__(**kwargs)
        self.control_tokens = control_tokens
        self.control_token_accuracies = {}
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Enhanced logging with control token metrics"""
        super().on_log(args, state, control, logs, **kwargs)
        
        if logs:
            # Track control-specific metrics
            control_metrics = {k: v for k, v in logs.items() if 'control' in k}
            if control_metrics:
                self.status_queue.put({
                    'type': 'control_metrics',
                    'step': state.global_step,
                    'metrics': control_metrics
                })


class CLARALoopTrainingManager(TrainingManager):
    """
    Training manager for C.L.A.R.A. Loop dual-head architecture.
    
    Extends the existing TrainingManager with support for:
    - Dual-head model training (generation + control)
    - Control token annotation and labeling
    - Emotional momentum validation
    - Surprise-weighted loss functions
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.control_loss_weight = 1.0
        
    def create_control_token_annotations(self, dataset: List[Dict[str, Any]], 
                                       character: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Annotate dataset with control token labels for dual-head training.
        
        For now, this creates synthetic control token labels based on:
        - Message sentiment analysis
        - Character personality traits
        - Conversation context
        """
        
        # Load control tokens
        tokens_path = Path("content/worlds/Default World/tokens.json")
        with open(tokens_path, 'r') as f:
            control_tokens = json.load(f)
        
        # Build token mappings
        token_to_id = {token['token']: i for i, token in enumerate(control_tokens)}
        
        annotated_dataset = []
        
        for sample in dataset:
            # Create annotated sample
            annotated_sample = sample.copy()
            
            # Analyze the assistant's response for control token annotation
            messages = sample.get('messages', [])
            if len(messages) >= 3:  # User, System, Assistant
                user_msg = messages[0]['content']
                assistant_msg = messages[2]['content']
                
                # Generate control token labels based on content analysis
                control_labels = self._generate_control_labels(
                    user_msg, assistant_msg, character, token_to_id
                )
                
                annotated_sample['control_labels'] = control_labels
            
            annotated_dataset.append(annotated_sample)
        
        logger.info(f"Annotated {len(dataset)} samples with control token labels")
        return annotated_dataset
    
    def _generate_control_labels(self, user_msg: str, assistant_msg: str, 
                               character: Dict[str, Any], token_to_id: Dict[str, int]) -> List[float]:
        """
        Generate control token labels for a conversation turn.
        
        This is a simplified heuristic-based approach for the spike.
        In production, this would be replaced with:
        - Manual annotation tools
        - More sophisticated NLP analysis
        - Active learning approaches
        """
        
        # Initialize all tokens as inactive
        labels = [0.0] * len(token_to_id)
        
        user_lower = user_msg.lower()
        assistant_lower = assistant_msg.lower()
        
        # Mood detection based on content
        if any(word in assistant_lower for word in ['happy', 'excited', 'great', 'wonderful', 'love']):
            if '<mood_happy_1>' in token_to_id:
                labels[token_to_id['<mood_happy_1>']] = 1.0
        elif any(word in assistant_lower for word in ['nervous', 'worried', 'anxious']):
            if '<mood_nervous_1>' in token_to_id:
                labels[token_to_id['<mood_nervous_1>']] = 1.0
        
        # Blush detection
        if any(word in user_lower for word in ['beautiful', 'pretty', 'gorgeous', 'cute', 'amazing']):
            if '<blush>' in token_to_id:
                labels[token_to_id['<blush>']] = 1.0
        
        # Touch detection
        if any(word in user_lower for word in ['touch', 'pat', 'hug', 'hold']):
            if '<touch_gentle>' in token_to_id:
                labels[token_to_id['<touch_gentle>']] = 1.0
        
        # Relationship progression (simple heuristic)
        if any(word in user_lower for word in ['friend', 'like you', 'enjoy talking']):
            if '<relationship_affinity_1>' in token_to_id:
                labels[token_to_id['<relationship_affinity_1>']] = 1.0
        
        # Curiosity detection
        if '?' in user_msg:
            if '<curiosity_piqued>' in token_to_id:
                labels[token_to_id['<curiosity_piqued>']] = 1.0
        
        return labels
    
    def _create_clara_model(self, config: Dict[str, Any]) -> CLARALoopSmolLM:
        """Create C.L.A.R.A. Loop model instead of standard model"""
        clara_config = CLARALoopConfig(
            base_model_name=self.base_model,
            control_head_dim=config.get('control_head_dim', 256),
            **config
        )
        
        model = CLARALoopSmolLM(clara_config)
        return model
    
    def _setup_clara_lora_model(self, model: CLARALoopSmolLM, config: Dict[str, Any], 
                              character: Dict[str, Any] = None, dataset_size: int = 0):
        """Setup LoRA for C.L.A.R.A. Loop model (only on base model, not control head)"""
        
        # Apply LoRA only to the base model, not the control head
        # The control head should train fully to learn control token mappings
        base_model = model.base_model
        
        # Use parent class method to setup LoRA on base model
        base_model_with_lora = super()._setup_lora_model(base_model, config, character, dataset_size)
        
        # Replace the base model in our C.L.A.R.A. model
        model.base_model = base_model_with_lora
        
        # Ensure control head parameters are trainable
        for param in model.control_head.parameters():
            param.requires_grad = True
        
        # Freeze recirculation layers for now (can be unfrozen later)
        for param in model.recirculation_embedding.parameters():
            param.requires_grad = False
        
        logger.info("Applied LoRA to base model, control head fully trainable")
        return model
    
    def start_clara_training(self, character: Dict[str, Any], dataset: List[Dict[str, Any]], 
                           config: Dict[str, Any]):
        """Start C.L.A.R.A. Loop training with dual-head support"""
        
        logger.info("🎭 Starting C.L.A.R.A. Loop dual-head training")
        
        # Add control token annotations to dataset
        annotated_dataset = self.create_control_token_annotations(dataset, character)
        
        # Update config for dual-head training
        enhanced_config = config.copy()
        enhanced_config.update({
            'clara_mode': True,
            'control_loss_weight': config.get('control_loss_weight', 1.0),
            'control_head_dim': config.get('control_head_dim', 256),
            'enable_control_validation': config.get('enable_control_validation', True),
            # Smaller batch size for dual-head stability
            'batch_size': config.get('batch_size', 1),
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 4),
            # Shorter training for spike validation
            'max_steps': config.get('max_steps', 200),
            'learning_rate': config.get('learning_rate', 5e-5)  # Lower LR for stability
        })
        
        # Start training with enhanced dataset and config
        return self.start_training(character, annotated_dataset, enhanced_config)
    
    def _training_worker(self, character: Dict[str, Any], dataset: List[Dict[str, Any]], 
                        config: Dict[str, Any]):
        """Enhanced training worker with C.L.A.R.A. Loop support"""
        
        if config.get('clara_mode', False):
            # Use C.L.A.R.A. Loop specific training
            return self._clara_training_worker(character, dataset, config)
        else:
            # Fall back to standard training
            return super()._training_worker(character, dataset, config)
    
    def _clara_training_worker(self, character: Dict[str, Any], dataset: List[Dict[str, Any]], 
                             config: Dict[str, Any]):
        """C.L.A.R.A. Loop specific training worker"""
        
        character_name = character.get('name', 'unknown')
        
        try:
            logger.info("🎭 Starting C.L.A.R.A. Loop training worker...")
            self.is_training = True
            self.should_stop = False
            
            # Load control tokens for callback
            tokens_path = Path("content/worlds/Default World/tokens.json")
            with open(tokens_path, 'r') as f:
                control_tokens = json.load(f)
            
            # Create C.L.A.R.A. Loop model
            logger.info("🧠 Creating C.L.A.R.A. Loop dual-head model...")
            model = self._create_clara_model(config)
            tokenizer = model.tokenizer
            
            # Setup LoRA on base model only
            model = self._setup_clara_lora_model(model, config, character, len(dataset))
            
            # Prepare dataset (same as parent, but with control labels)
            logger.info("📊 Preparing C.L.A.R.A. Loop dataset...")
            # ... (dataset preparation similar to parent)
            
            # Use custom data collator for dual-head training
            data_collator = CLARADataCollator(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # Setup training arguments (similar to parent)
            training_args = TrainingArguments(
                output_dir=str(self.project_dir / f"adapters/{character_name}_clara"),
                per_device_train_batch_size=config.get('batch_size', 1),  # Smaller batch for dual-head
                gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4),
                max_steps=config.get('max_steps', 500),  # Shorter for spike
                learning_rate=config.get('learning_rate', 1e-4),  # Lower LR for stability
                fp16=config.get('fp16', False) and self.device == "cuda",
                logging_steps=config.get('logging_steps', 10),
                save_steps=config.get('save_steps', 50),
                report_to="none",
                remove_unused_columns=False,  # Important for dual-head
                dataloader_pin_memory=(self.device == "cuda"),
                dataloader_num_workers=0,
            )
            
            # Create enhanced callback for C.L.A.R.A. Loop
            callback = CLARATrainingCallback(
                status_queue=self.status_queue,
                character=character,
                control_tokens=control_tokens,
                log_interval=config.get('logging_steps', 10)
            )
            
            # Create custom trainer
            trainer = CLARATrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,  # Will need proper dataset preparation
                data_collator=data_collator,
                callbacks=[callback],
                control_loss_weight=config.get('control_loss_weight', 1.0)
            )
            
            # Store trainer reference
            self.trainer = trainer
            
            # Start training
            logger.info("🚀 Starting C.L.A.R.A. Loop dual-head training...")
            trainer.train()
            
            # Save model
            trainer.save_model()
            logger.info("✅ C.L.A.R.A. Loop training complete!")
            
            self.status_queue.put({
                'type': 'clara_training_complete',
                'output_dir': training_args.output_dir,
                'message': 'C.L.A.R.A. Loop dual-head training completed successfully!'
            })
            
        except Exception as e:
            import traceback
            error_msg = f"C.L.A.R.A. Loop training failed: {str(e)}"
            traceback_str = traceback.format_exc()
            logger.error(f"❌ {error_msg}")
            logger.error(f"🔍 Full traceback:\n{traceback_str}")
            
            self.status_queue.put({
                'type': 'error',
                'message': error_msg,
                'traceback': traceback_str
            })
        finally:
            self.is_training = False
    
    def evaluate_clara_loop(self, model: CLARALoopSmolLM, test_conversations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate C.L.A.R.A. Loop emotional consistency and recirculation effectiveness.
        
        Returns metrics like:
        - Control token accuracy
        - Emotional momentum persistence
        - Surprise detection effectiveness
        """
        
        model.eval()
        metrics = {
            'control_token_accuracy': 0.0,
            'emotional_momentum_score': 0.0,
            'surprise_detection_accuracy': 0.0,
            'overall_clara_score': 0.0
        }
        
        # TODO: Implement evaluation logic
        # For spike, return placeholder metrics
        
        return metrics


# Factory function for easy integration
def create_clara_training_manager(**kwargs) -> CLARALoopTrainingManager:
    """Create a C.L.A.R.A. Loop training manager"""
    return CLARALoopTrainingManager(**kwargs)


if __name__ == "__main__":
    print("🎭 C.L.A.R.A. Loop Training Manager")
    print("Ready for dual-head emotional recirculation training!") 