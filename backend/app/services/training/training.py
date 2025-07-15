import threading
import queue
import torch
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    TrainerCallback, DataCollatorForLanguageModeling, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import shutil
import zipfile
import datetime
import time
import random
import json
from backend.app.core.metrics.metrics import CharacterConsistencyMetrics, TrainingQualityTracker
from backend.app.core.monitoring import AdvancedMonitor


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
        self.last_log_metrics: Dict[str, Any] = {}
        self.last_consistency_metrics: Dict[str, Any] = {}
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when training logs are available"""
        if logs and 'loss' in logs:
            self.last_log_metrics = logs.copy()

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
            if 'eval_loss' in logs and logs['eval_loss'] is not None:
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
                if 'eval_loss' in logs and logs['eval_loss'] is not None:
                    metrics['eval_loss'] = logs['eval_loss']
                
                # NEW: Add personality alignment metrics if available from quality tracker
                if self.quality_tracker:
                    wandb_metrics = self.quality_tracker.get_wandb_metrics()
                    if 'avg_personality_alignment' in wandb_metrics:
                        metrics['avg_personality_alignment'] = wandb_metrics['avg_personality_alignment']
                        metrics['recent_personality_alignment'] = wandb_metrics.get('recent_personality_alignment', wandb_metrics['avg_personality_alignment'])
                
                self.monitor.log_metrics(state.global_step, metrics)
                
                if self.quality_tracker:
                    self.monitor.log_training_health(state.global_step, health_status)
    
    def on_evaluate(self, args, state, control, model=None, tokenizer=None, eval_dataloader=None, **kwargs):
        """Called during evaluation - perform character consistency checks and personality alignment"""
        if (self.character and model and tokenizer and 
            hasattr(eval_dataloader, 'dataset') and 
            state.global_step % (self.log_interval * 5) == 0):  # Every 5 log intervals
            
            try:
                # Sample a few examples for consistency evaluation
                dataset = eval_dataloader.dataset
                sample_indices = random.sample(range(len(dataset)), min(5, len(dataset)))
                
                consistency_scores = []
                personality_alignment_scores = []
                evaluated_samples_for_ui = []
                
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
                                
                                # Evaluate character consistency
                                scores = self.consistency_metrics.evaluate_character_consistency(sample, self.character)
                                consistency_scores.append(scores)
                                
                                # NEW: Evaluate personality alignment if character has Big-Five scores
                                personality_score = None
                                big_five_scores = self.character.get('big_five_scores')
                                if big_five_scores and isinstance(big_five_scores, dict):
                                    try:
                                        from .evaluation.personality_metric import calculate_personality_alignment
                                        personality_score = calculate_personality_alignment(
                                            assistant_content, 
                                            big_five_scores
                                        )
                                        personality_alignment_scores.append(personality_score)
                                        logger.debug(f"Personality alignment for sample {idx}: {personality_score:.3f}")
                                    except Exception as e:
                                        logger.warning(f"Failed to calculate personality alignment for sample {idx}: {e}")
                                        personality_score = None
                                
                                evaluated_samples_for_ui.append({
                                    'user': user_content,
                                    'assistant': assistant_content,
                                    'scores': scores,
                                    'personality_alignment': personality_score
                                })
                    except Exception as e:
                        logger.debug(f"Error evaluating sample {idx}: {e}")
                        continue
                
                if consistency_scores:
                    # Log consistency metrics
                    if self.monitor:
                        self.monitor.log_character_consistency_metrics(state.global_step, consistency_scores)
                    
                    # Add consistency scores to quality tracker
                    avg_consistency = sum(score['overall_consistency'] for score in consistency_scores) / len(consistency_scores)
                    
                    # NEW: Add personality alignment scores to quality tracker
                    if personality_alignment_scores:
                        avg_personality_alignment = sum(personality_alignment_scores) / len(personality_alignment_scores)
                        if self.quality_tracker:
                            self.quality_tracker.add_personality_alignment_score(avg_personality_alignment)
                        logger.info(f"Step {state.global_step}: Avg personality alignment = {avg_personality_alignment:.3f}")
                    
                    # ✅ Store both for UI and checkpoint saving
                    self.last_consistency_metrics = {
                        'character_consistency': avg_consistency,  # Key that UI expects
                        'avg_consistency': avg_consistency,        # Backup key
                        'consistency_last_eval_step': state.global_step,
                        'evaluated_samples': evaluated_samples_for_ui,
                    }
                    
                    # Add personality alignment to stored metrics if available
                    if personality_alignment_scores:
                        self.last_consistency_metrics['avg_personality_alignment'] = avg_personality_alignment

                    self.status_queue.put({
                        'type': 'consistency_evaluation',
                        'step': state.global_step,
                        'character_consistency': avg_consistency,  # Key that UI expects
                        'avg_consistency': avg_consistency,         # Backup key
                        'consistency_scores': consistency_scores,
                        'personality_alignment_scores': personality_alignment_scores,
                        'avg_personality_alignment': sum(personality_alignment_scores) / len(personality_alignment_scores) if personality_alignment_scores else None,
                        'evaluated_samples': evaluated_samples_for_ui
                    })
                    
            except Exception as e:
                logger.debug(f"Error in consistency evaluation: {e}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the start of training"""
        self.status_queue.put({
            'type': 'train_begin',
            'total_steps': state.max_steps
        })
        
        # ✅ Store training start time for elapsed time calculations
        self._training_start_time = time.time()
        
        # ✅ Store all metadata attributes from args for checkpoint saving
        self._base_model_name = getattr(args, 'base_model_name', 'unknown')
        self._training_method = getattr(args, 'training_method', 'lora')
        self._use_dora = getattr(args, 'use_dora', False)
        self._use_rslora = getattr(args, 'use_rslora', False)
        self._lora_r = getattr(args, 'lora_r', 0)
        self._lora_alpha = getattr(args, 'lora_alpha', 0)
        self._lora_dropout = getattr(args, 'lora_dropout', 0.1)
        self._target_modules = getattr(args, 'target_modules', [])
        self._dataset_size = getattr(args, 'dataset_size', 0)
        
        logger.info(f"🏁 Training metadata stored: method={self._training_method}, r={self._lora_r}, base={self._base_model_name}")
        
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
        checkpoint_dir = kwargs.get('output_dir')
        if not checkpoint_dir:
            # Fallback for older transformers versions
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

        self.status_queue.put({
            'type': 'checkpoint_saved',
            'step': state.global_step,
            'checkpoint_dir': checkpoint_dir
        })

        # Save latest metrics to checkpoint directory
        metrics_to_save = {
            'current_step': state.global_step,
            'current_epoch': state.epoch,
            'elapsed_time': time.time() - getattr(self, '_training_start_time', time.time()),
            'timestamp': time.time()
        }
        metrics_to_save.update(self.last_log_metrics)
        metrics_to_save.update(self.last_consistency_metrics)

        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            
            # Save training summary (for metrics)
            summary_path = checkpoint_path / "training_summary.json"
            with summary_path.open('w') as f:
                json.dump(metrics_to_save, f, indent=4)
            
            # ✅ NEW: Save complete training metadata (for comparison charts)
            try:
                # Wait a moment to ensure checkpoint is fully written
                import time as time_module
                time_module.sleep(0.5)
                
                metadata_path = checkpoint_path / "training_metadata.json"
                
                # Create comprehensive metadata from stored attributes
                checkpoint_metadata = {
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "timestamp": time_module.time(),
                    "character_name": self.character.get('name', 'unknown') if self.character else 'unknown',
                    "base_model": getattr(self, '_base_model_name', 'unknown'),
                    "training_method": getattr(self, '_training_method', 'lora'),
                    "use_dora": getattr(self, '_use_dora', False),
                    "use_rslora": getattr(self, '_use_rslora', False),
                    "lora_r": getattr(self, '_lora_r', 0),
                    "lora_alpha": getattr(self, '_lora_alpha', 0),
                    "lora_dropout": getattr(self, '_lora_dropout', 0.1),
                    "target_modules": getattr(self, '_target_modules', []),
                    "dataset_size": getattr(self, '_dataset_size', 0),
                    "total_steps": state.max_steps if hasattr(state, 'max_steps') else 0
                }
                
                # Remove None values to keep JSON clean
                checkpoint_metadata = {k: v for k, v in checkpoint_metadata.items() if v is not None}
                
                with open(metadata_path, 'w') as f:
                    json.dump(checkpoint_metadata, f, indent=2)
                
                logger.info(f"✅ Checkpoint metadata saved to {metadata_path}")
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint metadata: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # ✅ NEW: Checkpoint sharding support
            shard_size_gb = getattr(args, 'shard_size_gb', None)
            if shard_size_gb and shard_size_gb > 0:
                try:
                    from ..checkpointing import ShardWriter
                    import torch
                    
                    logger.info(f"🔄 Creating sharded checkpoint (shard size: {shard_size_gb}GB)...")
                    
                    # Get model state dict (includes adapter weights)
                    model_state_dict = kwargs.get('model', args._current_model).state_dict()
                    
                    # Add optimizer state if available
                    optimizer_state = kwargs.get('optimizer')
                    if optimizer_state and hasattr(optimizer_state, 'state_dict'):
                        optimizer_dict = optimizer_state.state_dict()
                        # Prefix optimizer keys to avoid conflicts
                        for key, value in optimizer_dict.items():
                            model_state_dict[f"optimizer.{key}"] = value
                    
                    # Create shards
                    shard_writer = ShardWriter(shard_size_gb=shard_size_gb)
                    shards_dir = checkpoint_path / "shards"
                    manifest_path = shard_writer.save_shards(model_state_dict, shards_dir)
                    
                    logger.info(f"✅ Sharded checkpoint saved: {manifest_path}")
                    
                    # Upload to S3 if configured
                    s3_bucket = getattr(args, 's3_bucket', None)
                    if s3_bucket:
                        s3_prefix = getattr(args, 's3_prefix', f"checkpoints/{self.character.get('name', 'unknown')}/{state.global_step}/")
                        endpoint_url = getattr(args, 's3_endpoint_url', None)
                        
                        logger.info(f"📤 Uploading shards to S3: s3://{s3_bucket}/{s3_prefix}")
                        success = shard_writer.upload_to_s3(
                            shards_dir, 
                            s3_bucket, 
                            s3_prefix,
                            endpoint_url=endpoint_url
                        )
                        
                        if success:
                            logger.info("✅ Shards uploaded to S3 successfully")
                        else:
                            logger.warning("⚠️ Failed to upload shards to S3")
                    
                except Exception as e:
                    logger.error(f"Failed to create sharded checkpoint: {e}")
                    import traceback
                    logger.error(traceback.format_exc())


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
        self.eval_loss_history = []  # ✅ NEW: Track validation loss history
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
        from backend.app.core.metrics.metrics import CharacterConsistencyMetrics
        
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
    
    def has_preference_data(self, character_name: str, min_preferences: int = 100) -> bool:
        """
        Check if a character has sufficient preference data for RLHF.
        
        Args:
            character_name: Name of the character
            min_preferences: Minimum number of preference pairs required
            
        Returns:
            True if sufficient preferences exist
        """
        from .rlhf_trainer import has_sufficient_preferences
        return has_sufficient_preferences(character_name, min_preferences)
    
    def run_rlhf_training(self, character_name: str, sft_adapter_path: str, 
                         algorithm: str = "grpo", config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Run RLHF training on top of an SFT adapter.
        
        Args:
            character_name: Name of the character
            sft_adapter_path: Path to the SFT adapter
            algorithm: RLHF algorithm ("grpo" or "ppo")
            config: Optional RLHF configuration
            
        Returns:
            Path to RLHF adapter or None if failed
        """
        from .rlhf_trainer import run_rlhf, prepare_preference_dataset, RLHFConfig
        from pathlib import Path
        
        # Look for preference logs
        pref_path = Path(f"content/worlds/Default World/characters/{character_name}/preference_logs.ndjson")
        
        if not pref_path.exists():
            logger.error(f"No preference data found for {character_name}")
            return None
        
        try:
            # Prepare preference dataset
            pref_dataset = prepare_preference_dataset(pref_path)
            logger.info(f"Loaded {len(pref_dataset)} preference pairs for RLHF")
            
            # Create RLHF config
            if config is None:
                config = {}
            
            rlhf_config = RLHFConfig(
                algorithm=algorithm.lower(),
                output_dir=str(self._adapter_dir(character_name) / "rlhf_output"),
                **config
            )
            
            # Run RLHF training
            logger.info(f"Starting {algorithm.upper()} training for {character_name}")
            rlhf_adapter_path = run_rlhf(
                model_path=sft_adapter_path,
                pref_dataset=pref_dataset,
                config=rlhf_config
            )
            
            logger.info(f"RLHF training complete. Adapter saved to: {rlhf_adapter_path}")
            return rlhf_adapter_path
            
        except Exception as e:
            logger.error(f"RLHF training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
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
        """Setup LoRA or DoRA configuration for the model with adaptive parameters"""
        if self.device == "cuda":
            model = prepare_model_for_kbit_training(model)
        
        # Get adaptive LoRA configuration if enabled
        if character and dataset_size > 0:
            adaptive_config = self.get_adaptive_lora_config(character, dataset_size)
            # Override config values with adaptive ones if not explicitly set
            for key, value in adaptive_config.items():
                if key not in config or config[key] == 16:  # Default value
                    config[key] = value
        
        # Apply guideline-driven defaults for character PEFT
        r_val = config.get('lora_r', 16)
        alpha_val = config.get('lora_alpha', r_val * 2)
        dropout_val = config.get('lora_dropout', 0.0) # 0.0 is the recommended for DoRA
        target_modules_val = config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
        finetune_method = config.get('finetune_method', 'dora').lower()
        use_rslora = config.get('use_rslora', False)
        use_dora = config.get('use_dora', True)
        
        # Determine method based on config
        if finetune_method == 'dora' or use_dora:
            use_dora = True
            use_rslora = False  # DoRA and RSLoRA are mutually exclusive
            logger.info(f"DoRA Configuration: r={r_val}, α={alpha_val}, dropout={dropout_val}")
        elif use_rslora:
            use_dora = False
            logger.info(f"RSLoRA Configuration: r={r_val}, α={alpha_val}/√{r_val}, dropout={dropout_val}")
        else:
            logger.info(f"Standard LoRA Configuration: r={r_val}, α={alpha_val}, dropout={dropout_val}")

        common_peft_params = {
            'r': r_val,
            'lora_alpha': alpha_val,
            'target_modules': target_modules_val,
            'lora_dropout': dropout_val,
            'bias': "none",
            'task_type': "CAUSAL_LM",
        }
        
        # Add method-specific parameters
        if use_dora:
            common_peft_params['use_dora'] = True
            # Add DoRA-specific optimizations for better performance
            ephemeral_gpu_offload = config.get('ephemeral_gpu_offload', False)
            if ephemeral_gpu_offload and self.device == "cuda":
                from peft import LoraRuntimeConfig
                common_peft_params['runtime_config'] = LoraRuntimeConfig(ephemeral_gpu_offload=True)
                logger.info("DoRA ephemeral GPU offload enabled for performance")
        elif use_rslora:
            common_peft_params['use_rslora'] = True
        
        peft_config = LoraConfig(**common_peft_params)
        
        return get_peft_model(model, peft_config)
    
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
            
            # Configure logging frequency - check both config sources
            log_freq = config.get('logging_steps', self.advanced_config.get('logging_steps', 10))
            
            save_steps_val = config.get('save_steps', 50)

            # When load_best_model_at_end is True, save_steps must be a multiple of eval_steps.
            # We adjust eval_steps to be the closest divisor of save_steps to the original target.
            eval_steps_val = None
            if processed_val_dataset:
                original_eval_steps = max(25, log_freq * 2)
                # Check for compatibility.
                if save_steps_val > 0 and save_steps_val % original_eval_steps == 0:
                    eval_steps_val = original_eval_steps
                else:
                    # Find a compatible evaluation step count.
                    if save_steps_val > 0:
                        # Find all divisors of save_steps_val
                        divs = {i for i in range(1, int(save_steps_val**0.5) + 1) if save_steps_val % i == 0}
                        divs.update({save_steps_val // d for d in divs})
                        
                        if divs:
                            # Find the divisor closest to original_eval_steps, preferring the higher value on a tie.
                            eval_steps_val = min(divs, key=lambda d: (abs(d - original_eval_steps), -d))
                        else:
                            # Fallback, though this case is unlikely if save_steps_val > 0
                            eval_steps_val = save_steps_val
                    else:
                        # If save_steps is 0 or less, let transformers handle potential errors.
                        eval_steps_val = original_eval_steps

                    if eval_steps_val != original_eval_steps:
                        print(f"⚠️  Adjusted eval_steps from {original_eval_steps} to {eval_steps_val} to be compatible with save_steps={save_steps_val}")
            
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
                save_steps=save_steps_val,
                save_strategy="steps",
                
                # Validation and early stopping configuration
                eval_strategy="steps" if processed_val_dataset else "no",
                eval_steps=eval_steps_val,
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
            
            # ✅ Add metadata attributes to training_args for callback access
            finetune_method = config.get('finetune_method', 'lora').lower()
            use_rslora = config.get('use_rslora', False) or finetune_method == 'rslora'
            use_dora = config.get('use_dora', False) or finetune_method == 'dora'
            
            training_args.base_model_name = self.base_model
            training_args.training_method = finetune_method
            training_args.use_dora = use_dora
            training_args.use_rslora = use_rslora
            training_args.lora_r = config.get('lora_r', 16)
            training_args.lora_alpha = config.get('lora_alpha', config.get('lora_r', 16))
            training_args.lora_dropout = config.get('lora_dropout', 0.1)
            training_args.target_modules = config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
            training_args.dataset_size = len(selected_dataset)
            training_args.character_name = character_name
            
            # ✅ NEW: Add checkpoint sharding configuration
            training_args.shard_size_gb = config.get('shard_size_gb', 0)  # 0 = disabled
            training_args.s3_bucket = config.get('s3_bucket', None)
            training_args.s3_prefix = config.get('s3_prefix', None)
            training_args.s3_endpoint_url = config.get('s3_endpoint_url', None)
            
            logger.info(f"🏷️ Training args enhanced with metadata: {finetune_method}, r={training_args.lora_r}, base={self.base_model}")
            
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
            import time as time_module
            time_module.sleep(1)
            
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
            
            # Save base model and training metadata for inference compatibility
            # Extract values from config for metadata
            finetune_method = config.get('finetune_method', 'lora').lower()
            use_rslora = config.get('use_rslora', False)
            use_dora = config.get('use_dora', False)
            r_val = config.get('lora_r', 16)
            alpha_val = config.get('lora_alpha', r_val)
            dropout_val = config.get('lora_dropout', 0.1)
            target_modules_val = config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
            
            metadata = {
                'base_model': self.base_model,
                'training_method': finetune_method,
                'use_rslora': use_rslora,
                'use_dora': use_dora,
                'lora_r': r_val,
                'lora_alpha': alpha_val,
                'lora_dropout': dropout_val,
                'target_modules': target_modules_val,
                'character_name': character_name,
                'training_date': datetime.datetime.utcnow().isoformat(),
                'total_steps': total_steps,
                'dataset_size': len(selected_dataset)
            }
            
            metadata_path = output_dir / "training_metadata.json"
            with metadata_path.open('w') as f:
                json.dump(metadata, f, indent=4)
            print(f"✅ Training metadata saved to {metadata_path}")
            
            # Log final metrics
            if self.monitor:
                try:
                    self.monitor.create_loss_curve_plot()
                except Exception as e:
                    logger.warning(f"Failed to create final plots: {e}")
            
            # Also save final metrics to adapter directory
            final_metrics_to_save = self.current_metrics.copy()
            final_metrics_to_save.update(metadata)  # Include metadata in summary
            summary_path = output_dir / "training_summary.json"
            with summary_path.open('w') as f:
                json.dump(final_metrics_to_save, f, indent=4)
            
            # Check if RLHF should be run after SFT
            if config.get('enable_rlhf', False) and self.has_preference_data(character_name):
                print("🧠 Starting RLHF phase...")
                self.status_queue.put({
                    'type': 'rlhf_starting',
                    'message': 'Starting RLHF training phase'
                })
                
                # Get RLHF configuration
                rlhf_algorithm = config.get('rlhf_algorithm', 'grpo')
                rlhf_config = config.get('rlhf_config', {})
                
                # Run RLHF training
                rlhf_adapter_path = self.run_rlhf_training(
                    character_name=character_name,
                    sft_adapter_path=str(output_dir),
                    algorithm=rlhf_algorithm,
                    config=rlhf_config
                )
                
                if rlhf_adapter_path:
                    print(f"✅ RLHF training complete! Adapter saved to: {rlhf_adapter_path}")
                    self.status_queue.put({
                        'type': 'rlhf_complete',
                        'rlhf_adapter_path': rlhf_adapter_path
                    })
                    # Update final output directory to RLHF adapter
                    final_output_dir = rlhf_adapter_path
                else:
                    print("⚠️ RLHF training failed, using SFT adapter only")
                    final_output_dir = str(output_dir)
            else:
                final_output_dir = str(output_dir)
            
            self.status_queue.put({
                'type': 'training_complete',
                'output_dir': final_output_dir,
                'final_metrics': self.current_metrics.copy(),
                'sft_adapter_path': str(output_dir),
                'rlhf_adapter_path': rlhf_adapter_path if config.get('enable_rlhf', False) else None
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
        self.eval_loss_history.clear()  # ✅ NEW: Clear validation loss history
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
    
    def _load_sharded_checkpoint(self, manifest_path: Path, cache_dir: Path = None) -> Optional[str]:
        """
        Load a sharded checkpoint and return the path to the reassembled checkpoint.
        
        Args:
            manifest_path: Path to checkpoint.json manifest file
            cache_dir: Optional directory to cache downloaded files (for S3 checkpoints)
            
        Returns:
            Path to loaded checkpoint directory or None if failed
        """
        try:
            from .checkpointing import ShardLoader
            import tempfile
            
            logger.info(f"📥 Loading sharded checkpoint: {manifest_path}")
            
            # Handle S3 manifest URLs
            if str(manifest_path).startswith('s3://'):
                # Parse S3 URL: s3://bucket/prefix/checkpoint.json
                s3_url = str(manifest_path)
                parts = s3_url[5:].split('/', 1)  # Remove 's3://' and split
                bucket = parts[0]
                prefix = parts[1].rsplit('/', 1)[0] + '/' if len(parts) > 1 else ""
                
                if cache_dir is None:
                    cache_dir = Path(tempfile.mkdtemp(prefix="sharded_checkpoint_"))
                
                loader = ShardLoader()
                state_dict = loader.load_from_s3(bucket, prefix, cache_dir)
                
                # Create temporary checkpoint directory and save state dict
                checkpoint_dir = cache_dir / "assembled_checkpoint"
                checkpoint_dir.mkdir(exist_ok=True)
                
                import torch
                torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
                logger.info(f"✅ Sharded checkpoint assembled to: {checkpoint_dir}")
                
                return str(checkpoint_dir)
            
            else:
                # Local manifest file
                manifest_dir = manifest_path.parent
                loader = ShardLoader()
                state_dict = loader.load_from_directory(manifest_dir)
                
                # Save assembled checkpoint in same directory
                checkpoint_dir = manifest_dir.parent / "assembled_checkpoint"
                checkpoint_dir.mkdir(exist_ok=True)
                
                import torch
                torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
                logger.info(f"✅ Sharded checkpoint assembled to: {checkpoint_dir}")
                
                return str(checkpoint_dir)
        
        except Exception as e:
            logger.error(f"Failed to load sharded checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def resume_training(self):
        """Resume paused training with support for sharded checkpoints"""
        if not self.is_paused or self.is_training:
            return False

        # Locate latest checkpoint for character
        if not self._last_character or not self._last_dataset or not self._last_config:
            logger.error("No previous training context cached – cannot resume.")
            return False

        character_name = self._last_character.get('name', 'unknown').lower().replace(' ', '_')
        adapter_dir = self.project_dir / f"adapters/{character_name}"
        latest_ckpt = self._latest_checkpoint_dir(adapter_dir)
        
        # Check for sharded checkpoint if regular checkpoint not found
        if not latest_ckpt:
            # Look for sharded checkpoints
            for item in adapter_dir.glob("checkpoint-*/shards/checkpoint.json"):
                sharded_ckpt = self._load_sharded_checkpoint(item)
                if sharded_ckpt:
                    latest_ckpt = Path(sharded_ckpt)
                    break
        
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
        # Process any new status updates (critical for status changes)
        updates_processed = 0
        while not self.status_queue.empty() and updates_processed < 50:  # Prevent infinite loop
            try:
                status = self.status_queue.get_nowait()
                self._process_status_update(status)
                updates_processed += 1
            except queue.Empty:
                break
        
        # Debug log if many updates were processed
        if updates_processed > 10:
            logger.debug(f"Processed {updates_processed} status updates in get_metrics()")
        
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
            if 'eval_loss' in status and status['eval_loss'] is not None:
                self.current_metrics['eval_loss'] = status['eval_loss']
                # ✅ NEW: Track validation loss history for charting
                self.eval_loss_history.append(status['eval_loss'])
            
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
            
            # ✅ NEW: Include validation loss history in metrics for charting
            if hasattr(self, 'eval_loss_history') and self.eval_loss_history:
                self.current_metrics['eval_loss_history'] = self.eval_loss_history.copy()
        
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
                'consistency_last_eval_step': status.get('step', 0),
                'evaluated_samples': status.get('evaluated_samples', [])
            })
            
            # NEW: Update personality alignment metrics if available
            if status.get('avg_personality_alignment') is not None:
                self.current_metrics.update({
                    'avg_personality_alignment': status.get('avg_personality_alignment', 0),
                    'personality_alignment_last_eval_step': status.get('step', 0)
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
    
    def get_wandb_url(self) -> Optional[str]:
        """Get the Wandb URL if monitoring is active."""
        if self.monitor and self.monitor.enable_wandb:
            return self.monitor.wandb_url
        return None

    def get_tensorboard_logdir(self) -> Optional[str]:
        """Get the TensorBoard log directory if monitoring is active."""
        if self.monitor and self.monitor.enable_tensorboard:
            return self.monitor.tensorboard_logdir
        return None

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
    
    def merge_and_export_model(self, character_name: str, checkpoint_path: Optional[str] = None) -> Path:
        """Merge LoRA/DoRA weights into base model and export as a complete model"""
        from peft import PeftModel
        
        adapter_dir = self._adapter_dir(character_name)
        if checkpoint_path:
            merge_path = Path(checkpoint_path)
        else:
            merge_path = adapter_dir
        
        if not merge_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {merge_path}")
        
        # Load training metadata to get the correct base model
        metadata_path = merge_path / "training_metadata.json"
        if metadata_path.exists():
            with metadata_path.open('r') as f:
                metadata = json.load(f)
            base_model_name = metadata.get('base_model', self.base_model)
        else:
            base_model_name = self.base_model
            logger.warning(f"No metadata found, using default base model: {base_model_name}")
        
        print(f"🔄 Loading base model: {base_model_name}")
        
        # Load base model
        if self.device == "cuda":
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = None
            torch_dtype = torch.float32
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        if self.device != "cuda":
            base_model = base_model.to(self.device)
        
        # Load PEFT model
        print(f"🔄 Loading PEFT adapter from: {merge_path}")
        peft_model = PeftModel.from_pretrained(base_model, str(merge_path))
        
        # Merge weights
        print("🔄 Merging LoRA/DoRA weights...")
        merged_model = peft_model.merge_and_unload()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create export directory
        timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        export_dir = self.exports_dir / f"{character_name}_merged_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Save merged model and tokenizer
        print(f"💾 Saving merged model to: {export_dir}")
        merged_model.save_pretrained(export_dir, safe_serialization=True)
        tokenizer.save_pretrained(export_dir)
        
        # Save merge metadata
        merge_metadata = {
            'base_model': base_model_name,
            'adapter_path': str(merge_path),
            'character_name': character_name,
            'merge_date': datetime.datetime.utcnow().isoformat(),
            'merged_model_path': str(export_dir)
        }
        
        if metadata_path.exists():
            merge_metadata['training_metadata'] = metadata
        
        merge_metadata_path = export_dir / "merge_metadata.json"
        with merge_metadata_path.open('w') as f:
            json.dump(merge_metadata, f, indent=4)
        
        # Zip the merged model
        zip_path = self._zip_dir(export_dir, f"{character_name}_merged_{timestamp}")
        
        # Clean up the unzipped directory to save space
        shutil.rmtree(export_dir)
        
        print(f"✅ Model merged and exported to: {zip_path}")
        return zip_path

    def add_metadata_to_existing_model(self, character_name: str, base_model: str, 
                                      training_method: str = "dora", checkpoint_path: Optional[str] = None) -> bool:
        """Add metadata to an existing model that doesn't have training_metadata.json"""
        adapter_dir = self._adapter_dir(character_name)
        if checkpoint_path:
            target_path = Path(checkpoint_path)
        else:
            target_path = adapter_dir
        
        if not target_path.exists():
            logger.error(f"Adapter path not found: {target_path}")
            return False
        
        metadata_path = target_path / "training_metadata.json"
        if metadata_path.exists():
            logger.info(f"Metadata already exists at {metadata_path}")
            return True
        
        # Create metadata based on adapter_config.json if available
        adapter_config_path = target_path / "adapter_config.json"
        if adapter_config_path.exists():
            with adapter_config_path.open('r') as f:
                adapter_config = json.load(f)
            
            r_val = adapter_config.get('r', 16)
            alpha_val = adapter_config.get('lora_alpha', r_val)
            dropout_val = adapter_config.get('lora_dropout', 0.1)
            target_modules = adapter_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
            use_dora = adapter_config.get('use_dora', training_method.lower() == 'dora')
            use_rslora = adapter_config.get('use_rslora', training_method.lower() == 'rslora')
        else:
            # Default values
            r_val = 16
            alpha_val = 16
            dropout_val = 0.1
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            use_dora = training_method.lower() == 'dora'
            use_rslora = training_method.lower() == 'rslora'
        
        metadata = {
            'base_model': base_model,
            'training_method': training_method.lower(),
            'use_rslora': use_rslora,
            'use_dora': use_dora,
            'lora_r': r_val,
            'lora_alpha': alpha_val,
            'lora_dropout': dropout_val,
            'target_modules': target_modules,
            'character_name': character_name,
            'training_date': 'unknown',
            'total_steps': 'unknown',
            'dataset_size': 'unknown',
            'metadata_added_manually': True
        }
        
        with metadata_path.open('w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"✅ Metadata added to {metadata_path}")
        return True
    
    def add_metadata_to_checkpoint(self, character_name: str, checkpoint_name: str, 
                                 base_model: str, training_method: str = "dora") -> bool:
        """Add metadata to a specific checkpoint that doesn't have training_metadata.json"""
        adapter_dir = self._adapter_dir(character_name)
        checkpoint_path = adapter_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint path not found: {checkpoint_path}")
            return False
        
        metadata_path = checkpoint_path / "training_metadata.json"
        if metadata_path.exists():
            logger.info(f"Metadata already exists at {metadata_path}")
            return True
        
        # Read adapter config if available
        adapter_config_path = checkpoint_path / "adapter_config.json"
        if adapter_config_path.exists():
            with adapter_config_path.open('r') as f:
                adapter_config = json.load(f)
            
            r_val = adapter_config.get('r', 16)
            alpha_val = adapter_config.get('lora_alpha', r_val)
            dropout_val = adapter_config.get('lora_dropout', 0.1)
            target_modules = adapter_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
            use_dora = adapter_config.get('use_dora', training_method.lower() == 'dora')
            use_rslora = adapter_config.get('use_rslora', training_method.lower() == 'rslora')
        else:
            # Default values
            r_val = 16
            alpha_val = 16
            dropout_val = 0.1
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            use_dora = training_method.lower() == 'dora'
            use_rslora = training_method.lower() == 'rslora'
        
        # Extract step number from checkpoint name
        import re
        step_match = re.search(r'checkpoint-(\d+)', checkpoint_name)
        checkpoint_step = int(step_match.group(1)) if step_match else 0
        
        metadata = {
            'base_model': base_model,
            'training_method': training_method.lower(),
            'use_rslora': use_rslora,
            'use_dora': use_dora,
            'lora_r': r_val,
            'lora_alpha': alpha_val,
            'lora_dropout': dropout_val,
            'target_modules': target_modules,
            'character_name': character_name,
            'training_date': 'unknown',
            'total_steps': checkpoint_step,
            'dataset_size': 'unknown',
            'checkpoint_step': checkpoint_step,
            'metadata_added_manually': True
        }
        
        with metadata_path.open('w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"✅ Metadata added to checkpoint: {metadata_path}")
        return True
    
    def export_runtime_packet(self, character_name: str) -> str:
        """
        Export a character runtime packet containing all assets needed for deployment.
        
        Args:
            character_name: Name of the character to export
            
        Returns:
            str: Path to the created runtime packet directory
            
        Raises:
            FileNotFoundError: If character or adapter not found
            ValueError: If required files are missing
        """
        import shutil
        from pathlib import Path
        
        # Create runtime packets directory
        runtime_packets_dir = Path("runtime_packets")
        runtime_packets_dir.mkdir(exist_ok=True)
        
        # Create character-specific export directory
        packet_dir = runtime_packets_dir / character_name
        if packet_dir.exists():
            shutil.rmtree(packet_dir)
        packet_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎮 Creating runtime packet for '{character_name}' at {packet_dir}")
        
        # Step 1: Find character data across all worlds
        character_data = self._find_character_in_worlds(character_name)
        if not character_data:
            raise FileNotFoundError(f"Character '{character_name}' not found in any world")
        
        char_core_path, world_path = character_data
        
        # Step 2: Find the best available adapter (prefer RLHF over SFT)
        adapter_source = self._find_best_adapter(character_name)
        if not adapter_source:
            raise FileNotFoundError(f"No trained adapter found for character '{character_name}'")
        
        adapter_path, adapter_type = adapter_source
        logger.info(f"Using {adapter_type} adapter from: {adapter_path}")
        
        # Step 3: Copy adapter.safetensors
        source_adapter = adapter_path / "adapter.safetensors"
        if not source_adapter.exists():
            raise FileNotFoundError(f"adapter.safetensors not found at {source_adapter}")
        
        shutil.copy2(source_adapter, packet_dir / "adapter.safetensors")
        logger.info(f"✅ Copied adapter.safetensors ({adapter_type})")
        
        # Step 4: Copy character_core.json
        shutil.copy2(char_core_path, packet_dir / "character_core.json")
        logger.info(f"✅ Copied character_core.json")
        
        # Step 5: Copy world_lore.json
        world_lore_path = world_path / "world_lore.json"
        if world_lore_path.exists():
            shutil.copy2(world_lore_path, packet_dir / "world_lore.json")
            logger.info(f"✅ Copied world_lore.json")
        else:
            # Create minimal world lore if missing
            minimal_lore = {
                "meta": {"version": 1},
                "facts": {},
                "timeline": [],
                "factions": [],
                "places": []
            }
            with open(packet_dir / "world_lore.json", 'w') as f:
                json.dump(minimal_lore, f, indent=2)
            logger.warning(f"⚠️ Created minimal world_lore.json (original not found)")
        
        # Step 6: Copy tokens.json
        tokens_path = world_path / "tokens.json"
        if tokens_path.exists():
            shutil.copy2(tokens_path, packet_dir / "tokens.json")
            logger.info(f"✅ Copied tokens.json")
        else:
            # Create minimal tokens if missing
            minimal_tokens = []
            with open(packet_dir / "tokens.json", 'w') as f:
                json.dump(minimal_tokens, f, indent=2)
            logger.warning(f"⚠️ Created empty tokens.json (original not found)")
        
        # Step 7: Create runtime_config.json
        runtime_config = self._create_runtime_config(adapter_path, adapter_type)
        with open(packet_dir / "runtime_config.json", 'w') as f:
            json.dump(runtime_config, f, indent=2)
        logger.info(f"✅ Created runtime_config.json")
        
        # Step 8: Create export manifest
        manifest = {
            "character_name": character_name,
            "export_date": datetime.datetime.now(datetime.UTC).isoformat(),
            "adapter_type": adapter_type,
            "world_name": world_path.name,
            "files": [
                "adapter.safetensors",
                "character_core.json", 
                "world_lore.json",
                "tokens.json",
                "runtime_config.json"
            ],
            "format_version": "1.0"
        }
        
        with open(packet_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"✅ Created manifest.json")
        
        logger.info(f"🎉 Runtime packet created successfully at: {packet_dir}")
        return str(packet_dir)
    
    def _find_character_in_worlds(self, character_name: str) -> Optional[tuple[Path, Path]]:
        """
        Find character_core.json file across all worlds.
        
        Returns:
            tuple[Path, Path] | None: (character_core_path, world_path) if found, None otherwise
        """
        worlds_root = Path("content/worlds")
        if not worlds_root.exists():
            return None
        
        for world_dir in worlds_root.iterdir():
            if not world_dir.is_dir():
                continue
            
            char_core_path = world_dir / "characters" / character_name / "character_core.json"
            if char_core_path.exists():
                return char_core_path, world_dir
        
        return None
    
    def _find_best_adapter(self, character_name: str) -> Optional[tuple[Path, str]]:
        """
        Find the best available adapter for a character.
        Prefers RLHF adapter over SFT adapter.
        
        Returns:
            tuple[Path, str] | None: (adapter_path, adapter_type) if found, None otherwise
        """
        adapter_dir = self._adapter_dir(character_name)
        if not adapter_dir.exists():
            return None
        
        # Check for RLHF adapter first (preferred)
        rlhf_grpo_path = adapter_dir / "rlhf_output" / "adapter_grpo"
        rlhf_ppo_path = adapter_dir / "rlhf_output" / "adapter_ppo"
        
        for rlhf_path, rlhf_type in [(rlhf_grpo_path, "RLHF-GRPO"), (rlhf_ppo_path, "RLHF-PPO")]:
            if rlhf_path.exists() and (rlhf_path / "adapter.safetensors").exists():
                return rlhf_path, rlhf_type
        
        # Fall back to SFT adapter
        if (adapter_dir / "adapter.safetensors").exists():
            return adapter_dir, "SFT"
        
        return None
    
    def _create_runtime_config(self, adapter_path: Path, adapter_type: str) -> dict:
        """
        Create runtime configuration for the exported packet.
        
        Args:
            adapter_path: Path to the adapter directory
            adapter_type: Type of adapter (SFT, RLHF-GRPO, RLHF-PPO)
            
        Returns:
            dict: Runtime configuration
        """
        # Load training metadata to get base model
        metadata_path = adapter_path / "training_metadata.json"
        base_model = self.base_model  # Default fallback
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                base_model = metadata.get('base_model', self.base_model)
            except Exception as e:
                logger.warning(f"Failed to read metadata: {e}")
        
        # Create tokenizer path based on base model
        tokenizer_path = f".cache/tokenizers/{base_model.replace('/', '-')}-patched"
        
        config = {
            "base_model": base_model,
            "adapter_path": "adapter.safetensors",
            "tokenizer_path": tokenizer_path,
            "character_file": "character_core.json",
            "world_file": "world_lore.json",
            "tokens_file": "tokens.json",
            "adapter_type": adapter_type,
            "format_version": "1.0"
        }
        
        # Add adapter-specific configuration if metadata exists
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                config["training_metadata"] = {
                    "training_method": metadata.get('training_method', 'lora'),
                    "use_dora": metadata.get('use_dora', False),
                    "use_rslora": metadata.get('use_rslora', False),
                    "lora_r": metadata.get('lora_r', 16),
                    "lora_alpha": metadata.get('lora_alpha', 16),
                    "lora_dropout": metadata.get('lora_dropout', 0.1),
                    "target_modules": metadata.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
                }
            except Exception as e:
                logger.warning(f"Failed to include training metadata: {e}")
        
        return config 