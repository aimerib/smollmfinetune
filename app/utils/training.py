import threading
import queue
import time
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    TrainerCallback, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Configure logging
logger = logging.getLogger(__name__)


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
        # Conservative device selection - prefer CPU for stability during debugging
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            # MPS can be problematic with training, use with caution
            self.device = "mps"
            print("⚠️  Using MPS device - if training hangs, switch to CPU")
        else:
            self.device = "cpu"
        
        print(f"🔧 TrainingManager initialized with device: {self.device}")
        
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
        # Fix device mapping for different backends
        if self.device == "cuda":
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            # For MPS and CPU, don't use device_map, load normally and move to device
            device_map = None
            torch_dtype = torch.float32
        
        print(f"Loading model {self.base_model} on device {self.device}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        # For non-CUDA devices, manually move to device
        if self.device != "cuda":
            model = model.to(self.device)
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model loaded successfully on {self.device}")
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
            print("🚀 Starting training worker thread...")
            self.is_training = True
            self.should_stop = False
            
            # Check dependencies
            print("🔍 Checking dependencies...")
            try:
                import torch
                import transformers
                import peft
                import datasets
                print(f"✅ Dependencies OK - PyTorch: {torch.__version__}, Transformers: {transformers.__version__}")
            except ImportError as e:
                raise RuntimeError(f"Missing dependency: {e}")
            
            print("📦 Loading base model and tokenizer...")
            # Load model and tokenizer
            model, tokenizer = self._load_base_model()
            
            print("⚙️ Setting up LoRA configuration...")
            model = self._setup_lora_model(model, config)
            
            # Prepare dataset
            print("📊 Preparing dataset for training...")
            from .dataset import DatasetManager
            dataset_manager = DatasetManager()
            processed_dataset = dataset_manager.prepare_for_training(dataset, tokenizer)
            print(f"✅ Dataset prepared: {len(processed_dataset)} samples")
            
            # Data collator
            print("🔧 Setting up data collator...")
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8,
            )
            
            # Calculate training steps
            print("📈 Calculating training parameters...")
            batch_size = config.get('batch_size', 2)
            gradient_accumulation = config.get('gradient_accumulation_steps', 2)
            epochs = config.get('epochs', 3)
            dataset_size = len(processed_dataset)
            
            # Calculate total steps (can be overridden by UI)
            total_steps = (dataset_size * epochs) // (batch_size * gradient_accumulation)
            
            # Optional manual override coming from the UI – helpful when working
            # with very large datasets where a full epoch would be overkill.
            if config.get('max_steps_override') and config['max_steps_override'] > 0:
                print(f"🔧 Overriding total_steps: {total_steps} → {config['max_steps_override']}")
                total_steps = int(config['max_steps_override'])
            
            print(f"📊 Training calculation:")
            print(f"   Dataset size: {dataset_size} samples")
            print(f"   Epochs: {epochs}")
            print(f"   Batch size: {batch_size}")
            print(f"   Gradient accumulation: {gradient_accumulation}")
            print(f"   Formula: ({dataset_size} × {epochs}) ÷ ({batch_size} × {gradient_accumulation}) = {total_steps} steps")
            
            # Setup output directory
            character_name = character.get('name', 'unknown').lower().replace(' ', '_')
            output_dir = self.project_dir / f"adapters/{character_name}"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Output directory: {output_dir}")
            
            # Training arguments
            print("⚙️ Setting up training arguments...")
            
            # Adjust settings based on device
            use_fp16 = config.get('fp16', False) and self.device == "cuda"  # Only use FP16 on CUDA
            if self.device == "mps" and config.get('fp16', False):
                print("⚠️  FP16 disabled on MPS (Apple Silicon) for stability")
            
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation,
                max_steps=total_steps,
                learning_rate=config.get('learning_rate', 2e-5),
                fp16=use_fp16,
                optim="adamw_torch",
                logging_steps=config.get('logging_steps', 10),
                save_steps=config.get('save_steps', 100),
                save_strategy="steps",
                eval_strategy="no",  # Fixed: was 'evaluation_strategy'
                report_to="none",
                # MPS/CPU optimizations
                dataloader_pin_memory=(self.device == "cuda"),  # Only pin memory on CUDA
                dataloader_num_workers=0,  # Avoid multiprocessing issues
                max_grad_norm=config.get('max_grad_norm', 1.0),
                warmup_steps=config.get('warmup_steps', 10),
                remove_unused_columns=False,
                # Additional stability settings
                greater_is_better=False,
                load_best_model_at_end=False,
                seed=42,
            )
            
            print(f"✅ Using max_steps={total_steps} for precise control (instead of epochs)")
            
            print(f"⚙️ Training args configured for {self.device}")
            
            # Create trainer
            print("🏗️ Creating trainer instance...")
            callback = TrainingCallback(self.status_queue)
            
            print(f"🧠 Model parameters: {model.num_parameters():,}")
            print(f"📊 Training on {len(processed_dataset)} samples")
            
            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=processed_dataset,
                data_collator=data_collator,
                callbacks=[callback],
            )
            
            # Verify the trainer is using our calculated steps
            actual_max_steps = self.trainer.args.max_steps
            print(f"✅ Trainer created successfully!")
            print(f"🔍 Verification: Trainer will run for {actual_max_steps} steps (expected: {total_steps})")
            
            if actual_max_steps != total_steps:
                print(f"⚠️  WARNING: Step count mismatch! Expected {total_steps}, but trainer has {actual_max_steps}")
            
            print("🚀 Starting training loop...")
            
            # Add a small delay to ensure everything is set up properly
            import time
            time.sleep(1)
            
            # Manually trigger the start callback
            self.status_queue.put({
                'type': 'train_begin',
                'total_steps': total_steps,
                'message': f'Training started: {total_steps} steps over {epochs} epochs'
            })
            
            # Training loop with pause/resume support
            print("📊 About to call trainer.train()...")
            try:
                self.trainer.train()
                print("🎉 trainer.train() completed successfully!")
            except Exception as train_error:
                print(f"❌ Error during trainer.train(): {train_error}")
                raise
            
            print("🎉 Training completed successfully!")
            
            # Save final model
            self.trainer.save_model()
            self.status_queue.put({
                'type': 'training_complete',
                'output_dir': str(output_dir)
            })
            
        except Exception as e:
            import traceback
            error_msg = f"Training failed: {str(e)}"
            traceback_str = traceback.format_exc()
            print(f"❌ {error_msg}")
            print(f"🔍 Full traceback:\n{traceback_str}")
            
            self.status_queue.put({
                'type': 'error',
                'message': error_msg,
                'traceback': traceback_str
            })
        finally:
            print("🔄 Training worker cleanup...")
            self.is_training = False
            self.is_paused = False
    
    def start_training(self, character: Dict[str, Any], dataset: List[Dict[str, Any]], 
                      config: Dict[str, Any]):
        """Start training in a background thread"""
        if self.is_training:
            raise RuntimeError("Training is already in progress")
        
        print(f"🚀 Starting training for character: {character.get('name', 'Unknown')}")
        print(f"📊 Dataset size: {len(dataset)} samples")
        print(f"⚙️ Config: {config}")
        
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
        print("✅ Training thread started successfully")
    
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