import threading
import queue
import time
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    TrainerCallback, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset


class TrainingCallback(TrainerCallback):
    """Custom callback for real-time training updates"""
    
    def __init__(self, status_queue: queue.Queue):
        self.status_queue = status_queue
        self.start_time = time.time()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when training logs are available"""
        if logs and 'loss' in logs:
            elapsed_time = time.time() - self.start_time
            
            self.status_queue.put({
                'type': 'log',
                'step': state.global_step,
                'loss': logs['loss'],
                'learning_rate': logs.get('learning_rate', 0),
                'elapsed_time': elapsed_time,
                'epoch': state.epoch
            })
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the start of training"""
        self.status_queue.put({
            'type': 'train_begin',
            'total_steps': state.max_steps
        })
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        self.status_queue.put({
            'type': 'train_end',
            'final_step': state.global_step
        })
    
    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved"""
        self.status_queue.put({
            'type': 'checkpoint_saved',
            'step': state.global_step,
            'checkpoint_dir': args.output_dir
        })


class TrainingManager:
    """Manages model training with pause/resume functionality"""
    
    def __init__(self, base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
        self.base_model = base_model
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Training state
        self.is_training = False
        self.is_paused = False
        self.should_stop = False
        
        # Communication queues
        self.status_queue = queue.Queue()
        self.command_queue = queue.Queue()
        
        # Training objects
        self.trainer = None
        self.training_thread = None
        
        # Metrics storage
        self.loss_history = []
        self.current_metrics = {}
        
        # Paths
        self.project_dir = Path("training_output")
        self.project_dir.mkdir(exist_ok=True)
    
    def _load_base_model(self):
        """Load the base model and tokenizer"""
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def _setup_lora_model(self, model, config: Dict[str, Any]):
        """Setup LoRA configuration for the model"""
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=config.get('lora_r', 8),
            lora_alpha=config.get('lora_alpha', 32),
            target_modules=config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=config.get('lora_dropout', 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        return get_peft_model(model, lora_config)
    
    def _training_worker(self, character: Dict[str, Any], dataset: List[Dict[str, Any]], 
                        config: Dict[str, Any]):
        """Worker function that runs training in a separate thread"""
        try:
            self.is_training = True
            self.should_stop = False
            
            # Load model and tokenizer
            model, tokenizer = self._load_base_model()
            model = self._setup_lora_model(model, config)
            
            # Prepare dataset
            from .dataset import DatasetManager
            dataset_manager = DatasetManager()
            processed_dataset = dataset_manager.prepare_for_training(dataset, tokenizer)
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
            )
            
            # Calculate training steps
            batch_size = config.get('batch_size', 2)
            gradient_accumulation = config.get('gradient_accumulation_steps', 2)
            epochs = config.get('epochs', 3)
            total_steps = (len(processed_dataset) * epochs) // (batch_size * gradient_accumulation)
            
            # Setup output directory
            character_name = character.get('name', 'unknown').lower().replace(' ', '_')
            output_dir = self.project_dir / f"adapters/{character_name}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation,
                num_train_epochs=epochs,
                learning_rate=config.get('learning_rate', 2e-5),
                fp16=config.get('fp16', False),
                optim="adamw_torch",
                logging_steps=config.get('logging_steps', 10),
                save_steps=config.get('save_steps', 100),
                save_strategy="steps",
                evaluation_strategy="no",
                report_to="none",
                dataloader_pin_memory=False,
                dataloader_num_workers=0,
                max_grad_norm=config.get('max_grad_norm', 1.0),
                warmup_steps=config.get('warmup_steps', 10),
                remove_unused_columns=False,
            )
            
            # Create trainer
            callback = TrainingCallback(self.status_queue)
            
            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=processed_dataset,
                data_collator=data_collator,
                callbacks=[callback],
            )
            
            # Training loop with pause/resume support
            self.trainer.train()
            
            # Save final model
            self.trainer.save_model()
            self.status_queue.put({
                'type': 'training_complete',
                'output_dir': str(output_dir)
            })
            
        except Exception as e:
            self.status_queue.put({
                'type': 'error',
                'message': str(e)
            })
        finally:
            self.is_training = False
            self.is_paused = False
    
    def start_training(self, character: Dict[str, Any], dataset: List[Dict[str, Any]], 
                      config: Dict[str, Any]):
        """Start training in a background thread"""
        if self.is_training:
            raise RuntimeError("Training is already in progress")
        
        # Clear previous state
        self.loss_history.clear()
        self.current_metrics.clear()
        
        # Start training thread
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(character, dataset, config),
            daemon=True
        )
        self.training_thread.start()
    
    def pause_training(self):
        """Pause training and save checkpoint"""
        if not self.is_training or self.is_paused:
            return False
        
        self.is_paused = True
        if self.trainer:
            # Save current checkpoint
            self.trainer.save_model()
            self.status_queue.put({
                'type': 'training_paused',
                'checkpoint_saved': True
            })
        return True
    
    def resume_training(self):
        """Resume paused training"""
        if not self.is_paused:
            return False
        
        self.is_paused = False
        self.status_queue.put({
            'type': 'training_resumed'
        })
        return True
    
    def stop_training(self):
        """Stop training completely"""
        self.should_stop = True
        
        if self.trainer:
            self.trainer.save_model()
        
        self.is_training = False
        self.is_paused = False
        
        self.status_queue.put({
            'type': 'training_stopped'
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        # Process any new status updates
        while not self.status_queue.empty():
            try:
                status = self.status_queue.get_nowait()
                self._process_status_update(status)
            except queue.Empty:
                break
        
        return self.current_metrics.copy()
    
    def _process_status_update(self, status: Dict[str, Any]):
        """Process a status update from the training thread"""
        status_type = status.get('type')
        
        if status_type == 'log':
            # Update current metrics
            self.current_metrics.update({
                'current_step': status.get('step', 0),
                'current_loss': status.get('loss', 0),
                'learning_rate': status.get('learning_rate', 0),
                'elapsed_time': status.get('elapsed_time', 0),
                'epoch': status.get('epoch', 0)
            })
            
            # Add to loss history
            loss = status.get('loss', 0)
            self.loss_history.append(loss)
            
            # Calculate loss delta
            if len(self.loss_history) > 1:
                loss_delta = self.loss_history[-1] - self.loss_history[-2]
                self.current_metrics['loss_delta'] = loss_delta
            
            self.current_metrics['loss_history'] = self.loss_history.copy()
        
        elif status_type == 'train_begin':
            self.current_metrics['total_steps'] = status.get('total_steps', 0)
            self.current_metrics['training_started'] = True
        
        elif status_type in ['train_end', 'training_complete']:
            self.current_metrics['training_complete'] = True
            self.is_training = False
        
        elif status_type == 'error':
            self.current_metrics['error'] = status.get('message', 'Unknown error')
            self.is_training = False
    
    def get_available_checkpoints(self, character_name: str) -> List[str]:
        """Get list of available checkpoints for a character"""
        character_dir = self.project_dir / f"adapters/{character_name.lower().replace(' ', '_')}"
        
        if not character_dir.exists():
            return []
        
        checkpoints = []
        for item in character_dir.iterdir():
            if item.is_dir() and item.name.startswith('checkpoint-'):
                checkpoints.append(str(item))
        
        return sorted(checkpoints)
    
    def get_training_status(self) -> str:
        """Get current training status"""
        if self.is_training and self.is_paused:
            return 'paused'
        elif self.is_training:
            return 'training'
        elif self.current_metrics.get('training_complete'):
            return 'complete'
        elif self.current_metrics.get('error'):
            return 'error'
        else:
            return 'idle' 