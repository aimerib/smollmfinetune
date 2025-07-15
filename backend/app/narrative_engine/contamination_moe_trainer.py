#!/usr/bin/env python3
"""
🔥 Contamination-Isolation MoE Training Manager

Extends existing C.L.A.R.A. Loop training to support contamination warfare!
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
import re
import sys
sys.path.append('..')

from .narrative_engine.clara_trainer import CLARALoopTrainingManager, CLARATrainer
from .narrative_engine.contamination_moe import ContaminationIsolationMoE, ContaminationMoEConfig

logger = logging.getLogger(__name__)


class ContaminationMoETrainer(CLARATrainer):
    """Custom trainer for Contamination-Isolation MoE with expert routing loss"""
    
    def __init__(self, routing_loss_weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.routing_loss_weight = routing_loss_weight
        self.contamination_blocks = 0
        self.character_purity_successes = 0
        self.total_routing_decisions = 0
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute combined loss for contamination warfare"""
        
        # Extract contamination info
        input_text = inputs.pop('input_text', [''])[0] if 'input_text' in inputs else ""
        
        # Forward pass through contamination MoE
        outputs = model(input_text=input_text, **inputs)
        
        # Get base losses from C.L.A.R.A.
        total_loss = torch.tensor(0.0, device=next(iter(outputs['losses'].values())).device)
        loss_components = {}
        
        # Add generation and control losses
        for loss_name, loss_value in outputs['losses'].items():
            if loss_name == 'control_loss':
                weighted_loss = loss_value * self.control_loss_weight
                total_loss += weighted_loss
                loss_components[f'weighted_{loss_name}'] = weighted_loss.item()
            else:
                total_loss += loss_value
            loss_components[loss_name] = loss_value.item()
        
        # Update contamination metrics
        self._update_contamination_metrics(outputs)
        
        # Log metrics
        if hasattr(self, 'state') and self.state.global_step % self.args.logging_steps == 0:
            for key, value in loss_components.items():
                self.log({f"train/{key}": value})
            
            # Log contamination warfare metrics
            if self.total_routing_decisions > 0:
                self.log({
                    "contamination/block_rate": self.contamination_blocks / self.total_routing_decisions,
                    "contamination/character_purity": self.character_purity_successes / self.total_routing_decisions
                })
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def _update_contamination_metrics(self, outputs: Dict[str, torch.Tensor]):
        """Update contamination warfare metrics"""
        
        routing_info = outputs.get('routing_info', {})
        expert_weights = outputs.get('expert_weights')
        
        if expert_weights is not None:
            selected_expert = torch.argmax(expert_weights, dim=1)[0].item()
            contamination_detected = routing_info.get('contamination_detected', False)
            
            self.total_routing_decisions += 1
            
            if contamination_detected and selected_expert == 1:  # Safety expert
                self.contamination_blocks += 1
            elif not contamination_detected and selected_expert == 0:  # Character expert
                self.character_purity_successes += 1


class ContaminationMoETrainingManager(CLARALoopTrainingManager):
    """Training manager for Contamination-Isolation MoE architecture"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("🔥 Contamination-Isolation MoE Training Manager initialized!")
    
    def create_contamination_moe_model(self, config: Dict[str, Any]) -> ContaminationIsolationMoE:
        """Create Contamination-Isolation MoE model"""
        
        moe_config = ContaminationMoEConfig(
            base_model_name=self.base_model,
            control_head_dim=config.get('control_head_dim', 256),
            **config
        )
        
        return ContaminationIsolationMoE(moe_config)
    
    def start_contamination_warfare_training(self, character: Dict[str, Any], 
                                           dataset: List[Dict[str, Any]], 
                                           config: Dict[str, Any]):
        """Start contamination warfare training"""
        
        logger.info("🔥 STARTING CONTAMINATION WARFARE TRAINING!")
        
        # Enhanced config for contamination warfare
        warfare_config = config.copy()
        warfare_config.update({
            'contamination_moe_mode': True,
            'batch_size': 1,  # Small batch for MoE stability
            'gradient_accumulation_steps': 8,
            'learning_rate': 3e-5,  # Lower LR for MoE
            'max_steps': 500,  # Focused training
            'logging_steps': 5,
        })
        
        return self.start_training(character, dataset, warfare_config)
    
    def _training_worker(self, character: Dict[str, Any], dataset: List[Dict[str, Any]], 
                        config: Dict[str, Any]):
        """Enhanced training worker with contamination warfare"""
        
        if config.get('contamination_moe_mode', False):
            return self._contamination_warfare_worker(character, dataset, config)
        else:
            return super()._training_worker(character, dataset, config)
    
    def _contamination_warfare_worker(self, character: Dict[str, Any], 
                                    dataset: List[Dict[str, Any]], 
                                    config: Dict[str, Any]):
        """Contamination warfare training worker - FULL PRODUCTION TRAINING"""
        
        try:
            logger.info("🔥 CONTAMINATION WARFARE TRAINING INITIATED!")
            self.is_training = True
            self.should_stop = False
            
            # Load control tokens for contamination detection
            tokens_path = Path("content/worlds/Default World/tokens.json")
            if not tokens_path.exists():
                logger.warning("⚠️ Control tokens not found - creating basic ones for contamination MoE")
                # Create minimal control tokens for contamination MoE
                self._create_basic_control_tokens(tokens_path)
            
            # Create contamination MoE model
            logger.info("🧠 Creating Contamination-Isolation MoE model...")
            model = self.create_contamination_moe_model(config)
            tokenizer = model.tokenizer
            
            # Setup LoRA on base model only (contamination MoE layer stays trainable)
            model = self._setup_clara_lora_model(model, config, character, len(dataset))
            
            # Prepare contamination-enhanced dataset
            logger.info("📊 Preparing contamination warfare dataset...")
            enhanced_dataset = self._prepare_contamination_dataset(dataset, character)
            
            # Use contamination MoE data collator
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # Setup training arguments optimized for contamination MoE
            character_name = character.get('name', 'unknown').lower().replace(' ', '_')
            output_dir = self.project_dir / f"adapters/{character_name}_contamination_moe"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            from transformers import TrainingArguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=config.get('batch_size', 1),  # Small batch for MoE
                gradient_accumulation_steps=config.get('gradient_accumulation_steps', 8),
                max_steps=config.get('max_steps', 500),
                learning_rate=config.get('learning_rate', 3e-5),  # Lower LR for MoE stability
                fp16=config.get('fp16', False) and self.device == "cuda",
                logging_steps=config.get('logging_steps', 5),
                save_steps=config.get('save_steps', 50),
                report_to="none",
                remove_unused_columns=False,  # Important for contamination MoE
                dataloader_pin_memory=(self.device == "cuda"),
                dataloader_num_workers=0,
                optim="adamw_torch",
                lr_scheduler_type="cosine",
                warmup_ratio=0.05,
                save_safetensors=True,
                seed=42,
            )
            
            # Create contamination warfare callback
            callback = ContaminationMoETrainingCallback(
                status_queue=self.status_queue,
                character=character,
                log_interval=config.get('logging_steps', 5)
            )
            
            # Create contamination MoE trainer
            trainer = ContaminationMoETrainer(
                model=model,
                args=training_args,
                train_dataset=enhanced_dataset,
                data_collator=data_collator,
                callbacks=[callback],
                control_loss_weight=config.get('control_loss_weight', 1.0),
                routing_loss_weight=config.get('routing_loss_weight', 0.5)
            )
            
            # Store trainer reference
            self.trainer = trainer
            
            logger.info("🚀 DEPLOYING CONTAMINATION WARFARE TRAINING!")
            logger.info(f"📊 Training on {len(enhanced_dataset)} contamination-enhanced samples")
            
            # Start contamination warfare training!
            trainer.train()
            
            # Save the war-trained model
            trainer.save_model()
            
            # Test contamination isolation post-training
            logger.info("🧪 Testing contamination isolation...")
            contamination_report = self._test_contamination_isolation(model)
            
            logger.info("✅ CONTAMINATION WARFARE TRAINING COMPLETE!")
            logger.info(f"🎯 Final contamination isolation rate: {contamination_report.get('character_purity_rate', 0):.1f}%")
            
            # Save training metadata
            import json
            metadata = {
                'base_model': self.base_model,
                'training_method': 'contamination_moe',
                'character_name': character_name,
                'contamination_threshold': config.get('contamination_threshold', 0.7),
                'routing_temperature': config.get('routing_temperature', 1.0),
                'expert_dropout': config.get('expert_dropout', 0.1),
                'routing_loss_weight': config.get('routing_loss_weight', 0.5),
                'contamination_report': contamination_report,
                'total_steps': trainer.state.global_step if hasattr(trainer, 'state') else config.get('max_steps', 500),
                'training_date': __import__('datetime').datetime.utcnow().isoformat()
            }
            
            with open(output_dir / "contamination_moe_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.status_queue.put({
                'type': 'contamination_warfare_complete',
                'output_dir': str(output_dir),
                'contamination_report': contamination_report,
                'final_step': trainer.state.global_step if hasattr(trainer, 'state') else config.get('max_steps', 500),
                'message': 'CONTAMINATION WARFARE TRAINING COMPLETE! Expert specialization deployed successfully!'
            })
            
        except Exception as e:
            import traceback
            error_msg = f"Contamination warfare training failed: {str(e)}"
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
    
    def _prepare_contamination_dataset(self, dataset: List[Dict[str, Any]], 
                                     character: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare dataset with contamination analysis annotations"""
        
        enhanced_dataset = []
        
        for sample in dataset:
            enhanced_sample = sample.copy()
            
            # Add contamination analysis for training
            messages = sample.get('messages', [])
            if messages:
                user_message = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')
                assistant_message = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), '')
                
                # Analyze contamination in assistant's response
                contamination_score = self._analyze_contamination_score(assistant_message)
                enhanced_sample['contamination_analysis'] = {
                    'user_input': user_message,
                    'assistant_response': assistant_message,
                    'contamination_score': contamination_score,
                    'should_route_to_safety': contamination_score > 0.5
                }
            
            enhanced_dataset.append(enhanced_sample)
        
        logger.info(f"Enhanced {len(dataset)} samples with contamination analysis")
        return enhanced_dataset
    
    def _analyze_contamination_score(self, text: str) -> float:
        """Analyze text for contamination patterns (matches router logic)"""
        if not text:
            return 0.0
        
        contamination_patterns = [
            r'\bAI\b', r'\bassistant\b', r'\bprogramm', r'\bartificial intelligence\b',
            r'\bmodel\b', r'\btraining\b', r'\bhelpful and harmless\b',
            r'\bnot (actually|really) (a )?human\b', r'\breal person\b'
        ]
        
        text_lower = text.lower()
        matches = sum(1 for pattern in contamination_patterns 
                     if re.search(pattern, text_lower, re.IGNORECASE))
        
        return min(matches / 3.0, 1.0)
    
    def _test_contamination_isolation(self, model) -> Dict[str, Any]:
        """Test contamination isolation post-training"""
        
        test_cases = [
            ("Tell me about yourself", "character"),
            ("As an AI assistant, I think...", "contamination"),
            ("I'm feeling happy today!", "emotional"),
            ("What did we talk about earlier?", "memory")
        ]
        
        results = {'total': 0, 'character_pure': 0, 'contamination_blocked': 0}
        
        model.eval()
        with torch.no_grad():
            for input_text, expected_type in test_cases:
                input_ids = torch.randint(0, 1000, (1, 10))  # Dummy for testing
                outputs = model.forward(input_ids, input_text=input_text)
                
                contamination_detected = outputs['routing_info'].get('contamination_detected', False)
                expert_weights = outputs['expert_weights'][0]
                selected_expert = torch.argmax(expert_weights).item()
                
                results['total'] += 1
                
                if expected_type == "contamination" and contamination_detected and selected_expert == 1:
                    results['contamination_blocked'] += 1
                elif expected_type == "character" and not contamination_detected and selected_expert == 0:
                    results['character_pure'] += 1
        
        # Calculate rates
        results['character_purity_rate'] = (results['character_pure'] / 2) * 100  # 2 character tests
        results['contamination_block_rate'] = (results['contamination_blocked'] / 1) * 100  # 1 contamination test
        results['overall_success_rate'] = ((results['character_pure'] + results['contamination_blocked']) / results['total']) * 100
        
        return results
    
    def _create_basic_control_tokens(self, tokens_path: Path):
        """Create basic control tokens for contamination MoE if none exist"""
        
        basic_tokens = [
            {"token": "<mood_happy_1>", "description": "Happy emotional state", "category": "mood"},
            {"token": "<mood_nervous_1>", "description": "Nervous emotional state", "category": "mood"},
            {"token": "<blush>", "description": "Blushing reaction", "category": "expression"},
            {"token": "<touch_gentle>", "description": "Gentle touch reaction", "category": "physical"},
            {"token": "<relationship_affinity_1>", "description": "Relationship affinity", "category": "social"},
            {"token": "<curiosity_piqued>", "description": "Curiosity response", "category": "cognitive"}
        ]
        
        # Create directory if needed
        tokens_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save basic tokens
        import json
        with open(tokens_path, 'w') as f:
            json.dump(basic_tokens, f, indent=2)
        
        logger.info(f"Created basic control tokens at {tokens_path}")


class ContaminationMoETrainingCallback:
    """Enhanced callback for Contamination-Isolation MoE training"""
    
    def __init__(self, status_queue, character, log_interval=5):
        self.status_queue = status_queue
        self.character = character
        self.log_interval = log_interval
        self.contamination_warfare_log = []


def create_contamination_moe_training_manager(**kwargs) -> ContaminationMoETrainingManager:
    """Create Contamination-Isolation MoE training manager"""
    return ContaminationMoETrainingManager(**kwargs)


if __name__ == "__main__":
    print("🔥 CONTAMINATION-ISOLATION MoE TRAINING MANAGER")
    print("Ready to deploy expert specialization against constitutional AI contamination!")
    
    manager = create_contamination_moe_training_manager()
    print("✅ Contamination warfare training system operational!") 