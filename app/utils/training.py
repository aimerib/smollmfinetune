import threading
import queue
import time
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    TrainerCallback, DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import shutil
import zipfile
import datetime
import random
from .metrics import CharacterConsistencyMetrics, TrainingQualityTracker
from .monitoring import AdvancedMonitor

# Configure logging
logger = logging.getLogger(__name__)


class TrainingCallback(TrainerCallback):
    """Enhanced callback for real-time training updates with character consistency tracking"""
    
    def __init__(self, status_queue: queue.Queue, character: Dict[str, Any] = None, 
                 quality_tracker: TrainingQualityTracker = None,
                 monitor: AdvancedMonitor = None,
                 log_interval: int = 10):
        self.status_queue = status_queue
        self.start_time = time.time()
        self.character = character
        self.consistency_metrics = CharacterConsistencyMetrics()
        self.quality_tracker = quality_tracker
        self.monitor = monitor
        self.log_interval = log_interval
        self.eval_dataset_samples = []  # Cache for consistency evaluation
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when training logs are available"""
        if logs and 'loss' in logs and state.global_step % self.log_interval == 0:
            elapsed_time = time.time() - self.start_time
            
            log_data = {
                'type': 'log',
                'step': state.global_step,
                'loss': logs['loss'],
                'learning_rate': logs.get('learning_rate', 0),
                'elapsed_time': elapsed_time,
                'epoch': state.epoch
            }
            
            # Add evaluation metrics if available
            if 'eval_loss' in logs:
                log_data['eval_loss'] = logs['eval_loss']
            
            self.status_queue.put(log_data)
            
            # Track quality metrics
            if self.quality_tracker:
                self.quality_tracker.add_training_step(state.global_step, logs['loss'])
                
                # Get training health and log warnings
                health_status = self.quality_tracker.get_training_health()
                if health_status['warnings']:
                    log_data['training_warnings'] = health_status['warnings']
                    log_data['training_health'] = health_status['status']
                    self.status_queue.put({
                        'type': 'training_health',
                        'health_status': health_status,
                        'step': state.global_step
                    })
            
            # Log to advanced monitoring
            if self.monitor:
                metrics = {
                    'loss': logs['loss'],
                    'learning_rate': logs.get('learning_rate', 0),
                    'elapsed_time': elapsed_time,
                    'epoch': state.epoch
                }
                if 'eval_loss' in logs:
                    metrics['eval_loss'] = logs['eval_loss']
                
                self.monitor.log_metrics(state.global_step, metrics)
                
                if self.quality_tracker:
                    self.monitor.log_training_health(state.global_step, health_status)
    
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, eval_dataloader=None, **kwargs):
        """Called during evaluation - perform character consistency checks"""
        if (self.character and model and tokenizer and 
            hasattr(eval_dataloader, 'dataset') and 
            state.global_step % (self.log_interval * 5) == 0):  # Every 5 log intervals
            
            try:
                # Sample a few examples for consistency evaluation
                dataset = eval_dataloader.dataset
                sample_indices = random.sample(range(len(dataset)), min(5, len(dataset)))
                
                consistency_scores = []
                for idx in sample_indices:
                    try:
                        # Reconstruct the sample format for evaluation
                        input_ids = dataset[idx]['input_ids']
                        decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
                        
                        # Simple parsing to extract user/assistant parts
                        if 'User:' in decoded and 'Assistant:' in decoded:
                            parts = decoded.split('User:')[-1].split('Assistant:')
                            if len(parts) >= 2:
                                user_content = parts[0].strip()
                                assistant_content = parts[1].strip()
                                
                                # Create sample in expected format
                                sample = {
                                    'messages': [
                                        {'role': 'system', 'content': ''},
                                        {'role': 'user', 'content': user_content},
                                        {'role': 'assistant', 'content': assistant_content}
                                    ]
                                }
                                
                                # Evaluate consistency
                                scores = self.consistency_metrics.evaluate_character_consistency(sample, self.character)
                                consistency_scores.append(scores)
                    except Exception as e:
                        logger.debug(f"Error evaluating sample {idx}: {e}")
                        continue
                
                if consistency_scores:
                    # Log consistency metrics
                    if self.monitor:
                        self.monitor.log_character_consistency_metrics(state.global_step, consistency_scores)
                    
                    # Add to status queue
                    avg_consistency = sum(score['overall_consistency'] for score in consistency_scores) / len(consistency_scores)
                    self.status_queue.put({
                        'type': 'consistency_evaluation',
                        'step': state.global_step,
                        'avg_consistency': avg_consistency,
                        'consistency_scores': consistency_scores
                    })
                    
            except Exception as e:
                logger.debug(f"Error in consistency evaluation: {e}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the start of training"""
        self.status_queue.put({
            'type': 'train_begin',
            'total_steps': state.max_steps
        })
        
        if self.monitor:
            self.monitor.log_training_config(
                {
                    'max_steps': state.max_steps,
                    'learning_rate': args.learning_rate,
                    'batch_size': args.per_device_train_batch_size,
                    'fp16': args.fp16
                },
                self.character or {}
            )
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        self.status_queue.put({
            'type': 'train_end',
            'final_step': state.global_step
        })
        
        if self.monitor:
            final_metrics = {
                'final_step': state.global_step,
                'training_completed': True
            }
            self.monitor.log_final_results(final_metrics)
    
    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved"""
        self.status_queue.put({
            'type': 'checkpoint_saved',
            'step': state.global_step,
            'checkpoint_dir': args.output_dir
        })


class TrainingManager:
    """Enhanced training manager with character consistency metrics and advanced features"""
    
    def __init__(self, base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct", 
                 force_gpu: bool = False):
        self.default_base_model = base_model
        self.base_model = base_model
        
        # Enhanced device selection with better error handling
        self.device = self._select_device(force_gpu)
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
        
        # Enhanced metrics and monitoring
        self.loss_history = []
        self.current_metrics = {}
        self.quality_tracker = None
        self.monitor = None
        
        # Paths
        self.project_dir = Path("training_output")
        self.project_dir.mkdir(exist_ok=True)
        self.exports_dir = self.project_dir / "exports"
        self.exports_dir.mkdir(exist_ok=True)
        
        # Remember context for resuming
        self._last_character: Optional[Dict[str, Any]] = None
        self._last_dataset: Optional[List[Dict[str, Any]]] = None
        self._last_config: Optional[Dict[str, Any]] = None
        
        # Advanced features configuration
        self.advanced_config = {
            'enable_validation': True,
            'early_stopping_patience': 3,
            'adaptive_lora': False,
            'enhanced_quality_filtering': False,
            'enable_tensorboard': False,
            'enable_wandb': False,
            'configurable_logging_freq': 10
        }
    
    def _select_device(self, force_gpu: bool = False) -> str:
        """Enhanced device selection with better error handling"""
        try:
            if force_gpu and torch.cuda.is_available():
                # Test CUDA availability
                torch.cuda.empty_cache()
                return "cuda"
            elif torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                # Test MPS availability
                try:
                    test_tensor = torch.tensor([1.0]).to("mps")
                    return "mps"
                except Exception as e:
                    logger.warning(f"MPS device test failed: {e}")
                    return "cpu"
            else:
                return "cpu"
        except Exception as e:
            logger.warning(f"Device selection failed: {e}, falling back to CPU")
            return "cpu"
    
    def configure_advanced_features(self, config: Dict[str, Any]):
        """Configure advanced training features"""
        self.advanced_config.update(config)
        logger.info(f"Advanced features configured: {config}")
    
    def get_adaptive_lora_config(self, character: Dict[str, Any], dataset_size: int) -> Dict[str, str]:
        """Calculate adaptive LoRA parameters based on character complexity"""
        if not self.advanced_config.get('adaptive_lora', False):
            return {'lora_r': 16, 'lora_alpha': 16, 'lora_dropout': 0.1}
        
        # Analyze character complexity
        description_length = len(character.get('description', ''))
        personality_traits = len(character.get('personality', '').split(','))
        has_examples = bool(character.get('mes_example', ''))
        
        complexity_score = (
            (description_length / 1000) * 0.4 +
            (personality_traits / 10) * 0.3 +
            (1.0 if has_examples else 0.0) * 0.3
        )
        
        # Adjust parameters based on complexity and dataset size
        if complexity_score > 0.7 and dataset_size > 200:
            config = {'lora_r': 32, 'lora_alpha': 32, 'lora_dropout': 0.05}
            logger.info(f"High complexity character detected - using r=32")
        elif complexity_score < 0.3 or dataset_size < 100:
            config = {'lora_r': 8, 'lora_alpha': 8, 'lora_dropout': 0.15}
            logger.info(f"Simple character or small dataset - using r=8")
        else:
            config = {'lora_r': 16, 'lora_alpha': 16, 'lora_dropout': 0.1}
            logger.info(f"Standard character complexity - using r=16")
        
        logger.info(f"Character complexity score: {complexity_score:.2f}, dataset size: {dataset_size}")
        return config
    
    def _apply_quality_filter(self, dataset: List[Dict[str, Any]], character: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply enhanced quality filtering to the dataset"""
        from .metrics import CharacterConsistencyMetrics
        
        if not dataset:
            return dataset
        
        metrics = CharacterConsistencyMetrics()
        filtered_dataset = []
        char_name = character.get('name', '').lower()
        
        for sample in dataset:
            try:
                # Basic quality checks
                if len(sample.get('messages', [])) < 3:
                    continue
                    
                response = sample['messages'][2]['content']
                
                # Filter out responses that break character (third-person narration)
                if char_name and char_name in response.lower():
                    # Check if it's problematic third-person reference
                    lines = response.split('\n')
                    skip_sample = False
                    for line in lines:
                        if char_name in line.lower() and not line.strip().startswith('"'):
                            skip_sample = True
                            break
                    if skip_sample:
                        continue
                        
                # Filter out very short responses
                if len(response.split()) < 8:
                    continue
                    
                # Filter out meta-commentary
                if any(phrase in response.lower() for phrase in [
                    'as an ai', 'i cannot', 'i\'m not able', 'my training', 'as an assistant'
                ]):
                    continue
                
                # Evaluate character consistency
                scores = metrics.evaluate_character_consistency(sample, character)
                if scores['overall_consistency'] >= 0.3:  # Minimum threshold
                    filtered_dataset.append(sample)
                    
            except Exception as e:
                logger.debug(f"Error filtering sample: {e}")
                # Keep the sample if filtering fails
                filtered_dataset.append(sample)
        
        logger.info(f"Quality filtering: {len(dataset)} → {len(filtered_dataset)} samples")
        return filtered_dataset
    
    def set_base_model(self, model_name: str):
        """Update the base model for training"""
        self.base_model = model_name
        logger.info(f"Base model updated to: {model_name}")
    
    def _load_base_model(self):
        """Load the base model and tokenizer with enhanced error handling"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Configure device settings
                if self.device == "cuda":
                    device_map = "auto"
                    torch_dtype = torch.float16
                else:
                    device_map = None
                    torch_dtype = torch.float32
                
                print(f"Loading model {self.base_model} on device {self.device} (attempt {retry_count + 1})")
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
                
                # For non-CUDA devices, manually move to device
                if self.device != "cuda":
                    model = model.to(self.device)
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                print(f"✅ Model loaded successfully on {self.device}")
                return model, tokenizer
                
            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"GPU OOM on attempt {retry_count + 1}: {e}")
                if self.device == "cuda":
                    logger.warning("Falling back to CPU due to GPU OOM")
                    self.device = "cpu"
                    torch.cuda.empty_cache()
                else:
                    raise RuntimeError("Out of memory even on CPU") from e
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device == "cuda":
                    logger.warning(f"GPU memory error on attempt {retry_count + 1}: {e}")
                    self.device = "cpu"
                    torch.cuda.empty_cache()
                else:
                    logger.error(f"Runtime error loading model: {e}")
                    if retry_count == max_retries - 1:
                        raise
                        
            except Exception as e:
                logger.error(f"Unexpected error loading model on attempt {retry_count + 1}: {e}")
                if retry_count == max_retries - 1:
                    raise
                    
            retry_count += 1
            
        raise RuntimeError(f"Failed to load model after {max_retries} attempts")
    
    def _setup_lora_model(self, model, config: Dict[str, Any], character: Dict[str, Any] = None, dataset_size: int = 0):
        """Setup LoRA configuration for the model with adaptive parameters"""
        model = prepare_model_for_kbit_training(model)
        
        # Get adaptive LoRA configuration if enabled
        if character and dataset_size > 0:
            adaptive_config = self.get_adaptive_lora_config(character, dataset_size)
            # Override config values with adaptive ones if not explicitly set
            for key, value in adaptive_config.items():
                if key not in config or config[key] == 16:  # Default value
                    config[key] = value
        
        # Apply guideline-driven defaults for character LoRA
        r_val = config.get('lora_r', 16)
        alpha_val = config.get('lora_alpha', r_val)  # Use α = r for character training
        dropout_val = config.get('lora_dropout', 0.1)
        target_modules_val = config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
        
        logger.info(f"LoRA Configuration: r={r_val}, α={alpha_val}, dropout={dropout_val}")
        
        lora_config = LoraConfig(
            r=r_val,
            lora_alpha=alpha_val,
            target_modules=target_modules_val,
            lora_dropout=dropout_val,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        return get_peft_model(model, lora_config)
    
    def _training_worker(self, character: Dict[str, Any], dataset: List[Dict[str, Any]], 
                        config: Dict[str, Any]):
        """Enhanced worker function with validation, monitoring, and better error handling"""
        character_name = character.get('name', 'unknown')
        
        # Initialize monitoring and quality tracking
        self.quality_tracker = TrainingQualityTracker()
        
        # Initialize advanced monitoring if enabled
        if (self.advanced_config.get('enable_tensorboard') or 
            self.advanced_config.get('enable_wandb')):
            self.monitor = AdvancedMonitor(
                character_name=character_name,
                enable_tensorboard=self.advanced_config.get('enable_tensorboard', False),
                enable_wandb=self.advanced_config.get('enable_wandb', False),
                wandb_project="character-lora-training"
            )
        
        try:
            print("🚀 Starting enhanced training worker thread...")
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
            # Load model and tokenizer with enhanced error handling
            model, tokenizer = self._load_base_model()
            
            print("⚙️ Setting up LoRA configuration...")
            # Use adaptive LoRA parameters if enabled
            model = self._setup_lora_model(model, config, character, len(dataset))
            
            # Apply sample selection if specified
            max_samples = config.get('max_samples', len(dataset))
            if max_samples < len(dataset):
                print(f"🎲 Randomly selecting {max_samples} samples from {len(dataset)} total samples...")
                # Create a copy and shuffle to ensure randomization
                dataset_copy = dataset.copy()
                random.shuffle(dataset_copy)
                selected_dataset = dataset_copy[:max_samples]
                print(f"✅ Selected {len(selected_dataset)} samples for training")
            else:
                selected_dataset = dataset
                print(f"📊 Using all {len(dataset)} samples for training")
            
            # Enhanced quality filtering if enabled
            if self.advanced_config.get('enhanced_quality_filtering', False):
                print("🔍 Applying enhanced quality filtering...")
                selected_dataset = self._apply_quality_filter(selected_dataset, character)
                print(f"✅ Quality filtering complete: {len(selected_dataset)} samples remain")
            
            # Prepare dataset with validation split
            print("📊 Preparing dataset for training...")
            from .dataset import DatasetManager
            dataset_manager = DatasetManager()
            include_system_prompts = config.get('include_system_prompts', False)
            
            # Create validation split if enabled
            if self.advanced_config.get('enable_validation', True) and len(selected_dataset) > 20:
                val_split = 0.1  # 10% for validation
                val_size = max(2, int(len(selected_dataset) * val_split))
                train_data = selected_dataset[:-val_size]
                val_data = selected_dataset[-val_size:]
                
                processed_train_dataset = dataset_manager.prepare_for_training(
                    train_data, tokenizer, include_system_prompts=include_system_prompts
                )
                processed_val_dataset = dataset_manager.prepare_for_training(
                    val_data, tokenizer, include_system_prompts=include_system_prompts
                )
                
                print(f"✅ Dataset prepared with validation split:")
                print(f"   Training: {len(processed_train_dataset)} samples")
                print(f"   Validation: {len(processed_val_dataset)} samples")
                
            else:
                # No validation split
                processed_train_dataset = dataset_manager.prepare_for_training(
                    selected_dataset, tokenizer, include_system_prompts=include_system_prompts
                )
                processed_val_dataset = None
                print(f"✅ Dataset prepared: {len(processed_train_dataset)} samples (no validation)")
            
            if include_system_prompts:
                print("   Including system prompts in training data")
            else:
                print("   System prompts removed (LoRA will internalize character behavior)")
            
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
            epochs = config.get('epochs', 6)  # 5-10 epochs recommended for character LoRA
            dataset_size = len(selected_dataset)
            
            # Calculate total steps (can be overridden by UI)
            total_steps = (dataset_size * epochs) // (batch_size * gradient_accumulation)
            
            # Optional manual override coming from the UI – helpful when working
            # with very large datasets where a full epoch would be overkill.
            if config.get('max_steps_override') and config['max_steps_override'] > 0:
                print(f"🔧 Overriding total_steps: {total_steps} → {config['max_steps_override']}")
                total_steps = int(config['max_steps_override'])
            
            print(f"📊 Training calculation:")
            if max_samples < len(dataset):
                print(f"   Original dataset: {len(dataset)} samples")
                print(f"   Selected samples: {dataset_size} samples (randomly sampled)")
            else:
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
            
            # Enhanced training arguments
            print("⚙️ Setting up enhanced training arguments...")
            
            # Adjust settings based on device
            use_fp16 = config.get('fp16', False) and self.device == "cuda"  # Only use FP16 on CUDA
            if self.device == "mps" and config.get('fp16', False):
                print("⚠️  FP16 disabled on MPS (Apple Silicon) for stability")
            
            # Configure logging frequency
            log_freq = self.advanced_config.get('configurable_logging_freq', 10)
            
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation,
                max_steps=total_steps,
                learning_rate=config.get('learning_rate', 2e-4),
                fp16=use_fp16,
                optim="adamw_torch",
                
                # Enhanced logging and evaluation
                logging_steps=log_freq,
                save_steps=config.get('save_steps', 50),
                save_strategy="steps",
                
                # Validation and early stopping configuration
                eval_strategy="steps" if processed_val_dataset else "no",
                eval_steps=max(25, log_freq * 2) if processed_val_dataset else None,
                load_best_model_at_end=processed_val_dataset is not None,
                metric_for_best_model="eval_loss" if processed_val_dataset else None,
                greater_is_better=False,
                
                # Learning rate scheduling
                lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),
                warmup_ratio=config.get('warmup_ratio', 0.05),
                warmup_steps=config.get('warmup_steps', max(5, total_steps // 20)),
                
                # Performance optimizations
                dataloader_pin_memory=(self.device == "cuda"),
                dataloader_num_workers=0,  # Avoid multiprocessing issues
                max_grad_norm=config.get('max_grad_norm', 1.0),
                remove_unused_columns=False,
                
                # Additional stability and monitoring
                report_to="none",  # We handle logging manually
                save_total_limit=3,  # Keep only 3 checkpoints
                save_safetensors=True,
                seed=42,
                
                # Advanced settings
                prediction_loss_only=True,
                include_inputs_for_metrics=False,
            )
            
            print(f"✅ Using max_steps={total_steps} for precise control (instead of epochs)")
            
            print(f"⚙️ Training args configured for {self.device}")
            
            # Create enhanced trainer with callbacks
            print("🏗️ Creating enhanced trainer instance...")
            
            # Create enhanced callback
            callback = TrainingCallback(
                status_queue=self.status_queue,
                character=character,
                quality_tracker=self.quality_tracker,
                monitor=self.monitor,
                log_interval=log_freq
            )
            
            # Prepare callbacks list
            callbacks = [callback]
            
            # Add early stopping if validation is enabled
            if processed_val_dataset is not None:
                early_stopping = EarlyStoppingCallback(
                    early_stopping_patience=self.advanced_config.get('early_stopping_patience', 3),
                    early_stopping_threshold=0.001
                )
                callbacks.append(early_stopping)
                print(f"✅ Early stopping enabled with patience={self.advanced_config.get('early_stopping_patience', 3)}")
            
            print(f"🧠 Model parameters: {model.num_parameters():,}")
            print(f"📊 Training on {len(processed_train_dataset)} samples")
            if processed_val_dataset:
                print(f"📊 Validation on {len(processed_val_dataset)} samples")
            
            self.trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=processed_train_dataset,
                eval_dataset=processed_val_dataset,
                data_collator=data_collator,
                callbacks=callbacks,
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
            
            # Training loop with pause/resume support – allow resuming from checkpoint
            resume_cp = config.get('resume_from_checkpoint')
            if resume_cp:
                print(f"🔄 Resuming training from checkpoint: {resume_cp}")
            print("📊 About to call trainer.train()...")
            try:
                self.trainer.train(resume_from_checkpoint=resume_cp if resume_cp else None)
                print("🎉 trainer.train() completed successfully!")
            except Exception as train_error:
                print(f"❌ Error during trainer.train(): {train_error}")
                raise
            
            print("🎉 Training completed successfully!")
            
            # Save final model
            print("💾 Saving final model...")
            self.trainer.save_model()
            
            # Log final metrics
            if self.monitor:
                try:
                    self.monitor.create_loss_curve_plot()
                except Exception as e:
                    logger.warning(f"Failed to create final plots: {e}")
            
            self.status_queue.put({
                'type': 'training_complete',
                'output_dir': str(output_dir),
                'final_metrics': self.current_metrics.copy()
            })
            
            print("🎉 Training completed successfully!")
            
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
            
            # Cleanup monitoring resources
            if self.monitor:
                try:
                    self.monitor.finish()
                except Exception as e:
                    logger.warning(f"Error finishing monitoring: {e}")
    
    def start_training(self, character: Dict[str, Any], dataset: List[Dict[str, Any]], 
                      config: Dict[str, Any]):
        """Start training in a background thread"""
        if self.is_training:
            raise RuntimeError("Training is already in progress")
        
        print(f"🚀 Starting training for character: {character.get('name', 'Unknown')}")
        max_samples = config.get('max_samples', len(dataset))
        if max_samples < len(dataset):
            print(f"📊 Dataset: {max_samples} samples selected from {len(dataset)} total")
        else:
            print(f"📊 Dataset size: {len(dataset)} samples")
        print(f"⚙️ Config: {config}")
        
        # Clear previous state
        self.loss_history.clear()
        self.current_metrics.clear()
        
        # Cache context for potential resume later
        self._last_character = character
        self._last_dataset = dataset
        self._last_config = config.copy()
        
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
            # Request trainer to stop after current step
            try:
                if hasattr(self.trainer, 'control'):
                    self.trainer.control.should_training_stop = True
            except Exception as _e:
                pass
            self.status_queue.put({
                'type': 'training_paused',
                'checkpoint_saved': True
            })
        return True
    
    def resume_training(self):
        """Resume paused training"""
        if not self.is_paused or self.is_training:
            return False

        # Locate latest checkpoint for character
        if not self._last_character or not self._last_dataset or not self._last_config:
            logger.error("No previous training context cached – cannot resume.")
            return False

        character_name = self._last_character.get('name', 'unknown').lower().replace(' ', '_')
        adapter_dir = self.project_dir / f"adapters/{character_name}"
        latest_ckpt = self._latest_checkpoint_dir(adapter_dir)
        if not latest_ckpt:
            logger.error("No checkpoint found to resume from.")
            return False

        # Inject resume checkpoint path into config copy
        cfg = self._last_config.copy()
        cfg['resume_from_checkpoint'] = str(latest_ckpt)

        # Reset flags and start new training thread
        self.is_paused = False
        self.start_training(self._last_character, self._last_dataset, cfg)
        self.status_queue.put({'type': 'training_resumed'})
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
        """Process a status update from the training thread with enhanced metrics"""
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
            
            # Add validation loss if available
            if 'eval_loss' in status:
                self.current_metrics['eval_loss'] = status['eval_loss']
            
            # Add training warnings if present
            if 'training_warnings' in status:
                self.current_metrics['training_warnings'] = status['training_warnings']
            if 'training_health' in status:
                self.current_metrics['training_health'] = status['training_health']
            
            # Add to loss history
            loss = status.get('loss', 0)
            self.loss_history.append(loss)
            
            # Calculate loss delta
            if len(self.loss_history) > 1:
                loss_delta = self.loss_history[-1] - self.loss_history[-2]
                self.current_metrics['loss_delta'] = loss_delta
            
            self.current_metrics['loss_history'] = self.loss_history.copy()
        
        elif status_type == 'training_health':
            # Update training health status
            health_status = status.get('health_status', {})
            self.current_metrics.update({
                'training_health_status': health_status.get('status', 'unknown'),
                'health_warnings': health_status.get('warnings', []),
                'health_recommendations': health_status.get('recommendations', []),
                'avg_consistency': health_status.get('avg_consistency', 0)
            })
        
        elif status_type == 'consistency_evaluation':
            # Update character consistency metrics
            self.current_metrics.update({
                'character_consistency': status.get('avg_consistency', 0),
                'consistency_last_eval_step': status.get('step', 0)
            })
        
        elif status_type == 'train_begin':
            self.current_metrics['total_steps'] = status.get('total_steps', 0)
            self.current_metrics['training_started'] = True
        
        elif status_type in ['train_end', 'training_complete']:
            self.current_metrics['training_complete'] = True
            if 'final_metrics' in status:
                self.current_metrics.update(status['final_metrics'])
            self.is_training = False
        
        elif status_type == 'error':
            self.current_metrics['error'] = status.get('message', 'Unknown error')
            self.current_metrics['error_traceback'] = status.get('traceback', '')
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
    
    # ------------------------------------------------------------------
    # 🔧 Asset-management helpers
    # ------------------------------------------------------------------

    def _adapter_dir(self, character_name: str) -> Path:
        return self.project_dir / f"adapters/{character_name.lower().replace(' ', '_')}"

    def clear_training_assets(self, character_name: str) -> bool:
        """Delete the LoRA adapter folder (and all checkpoints)."""
        target_dir = self._adapter_dir(character_name)
        if target_dir.exists():
            shutil.rmtree(target_dir)
            logger.info(f"🗑️ Cleared training assets at {target_dir}")
            return True
        logger.warning(f"No training assets found for {character_name} at {target_dir}")
        return False

    def _zip_dir(self, source_dir: Path, zip_name: str) -> Path:
        """Utility to zip an entire directory tree and return resulting path."""
        zip_path = self.exports_dir / f"{zip_name}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_dir.rglob('*'):
                zf.write(file_path, file_path.relative_to(source_dir))
        logger.info(f"📦 Created export zip {zip_path} (source {source_dir})")
        return zip_path

    def export_lora(self, character_name: str) -> Path:
        """Zip the final adapter folder for sharing/deployment."""
        adapter_dir = self._adapter_dir(character_name)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter directory not found for {character_name}: {adapter_dir}")
        return self._zip_dir(adapter_dir, f"{character_name}_lora_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

    def _latest_checkpoint_dir(self, adapter_dir: Path) -> Optional[Path]:
        checkpoints = [p for p in adapter_dir.iterdir() if p.is_dir() and p.name.startswith('checkpoint-')]
        if not checkpoints:
            return None
        # Sort by integer after 'checkpoint-'
        checkpoints.sort(key=lambda p: int(p.name.split('-')[-1]))
        return checkpoints[-1]

    def export_latest_checkpoint(self, character_name: str) -> Optional[Path]:
        """Zip the most recent checkpoint directory (if any)."""
        adapter_dir = self._adapter_dir(character_name)
        checkpoint_dir = self._latest_checkpoint_dir(adapter_dir)
        if not checkpoint_dir:
            logger.warning(f"No checkpoints found for {character_name}")
            return None
        zip_name = f"{character_name}_{checkpoint_dir.name}_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        return self._zip_dir(checkpoint_dir, zip_name) 