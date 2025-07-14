"""
RLHF (Reinforcement Learning from Human Feedback) trainer implementation.

Supports both GRPO (Group-Relative Policy Optimization) and PPO (Proximal Policy Optimization)
for preference-based fine-tuning of language models.
"""
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel, get_peft_model, LoraConfig

# TRL imports with fallback
try:
    from trl import GRPOTrainer, PPOTrainer, GRPOConfig, PPOConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    logging.warning("TRL library not available. RLHF features will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """Configuration for RLHF training"""
    # Algorithm selection
    algorithm: str = "grpo"  # "grpo", "ppo", or "dpo"
    
    # Training hyperparameters (GRPO defaults from R1-11 spec)
    learning_rate: float = 5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 500
    
    # GRPO-specific parameters
    num_generations: int = 6  # For group-relative scoring
    max_prompt_length: int = 4000
    max_completion_length: int = 2500
    beta: float = 0.0  # KL penalty (0 = no reference model needed)
    
    # PPO-specific parameters
    kl_penalty: float = 0.1
    ppo_batch_size: int = 16
    
    # DPO-specific parameters
    dpo_beta: float = 0.1  # DPO KL penalty
    reference_free: bool = False  # Use reference-free DPO
    label_smoothing: float = 0.0  # Label smoothing for DPO
    
    # Output and logging
    output_dir: str = "rlhf_output"
    report_to: str = "wandb"
    logging_steps: int = 10
    save_steps: int = 100
    
    # Advanced options
    fp16: bool = True
    gradient_checkpointing: bool = True
    strict_kl: bool = False  # If True, use beta > 0 for GRPO
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for trainer"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def prepare_preference_dataset(preference_path: Union[str, Path]) -> Dataset:
    """
    Prepare a preference dataset from NDJSON logs.
    
    Args:
        preference_path: Path to preference_logs.ndjson file
        
    Returns:
        HuggingFace Dataset with columns: prompt, chosen, rejected
    """
    preference_path = Path(preference_path)
    
    if not preference_path.exists():
        raise FileNotFoundError(f"Preference file not found: {preference_path}")
    
    # Load preference data
    preferences = []
    with open(preference_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Handle multiple rejected options by selecting one
                    if isinstance(data.get('rejected'), list):
                        # For now, select the first rejected option
                        # In future, could use all rejected options
                        rejected = data['rejected'][0] if data['rejected'] else ""
                    else:
                        rejected = data.get('rejected', "")
                    
                    preferences.append({
                        'prompt': data['prompt'],
                        'chosen': data['chosen'],
                        'rejected': rejected
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid preference entry: {e}")
    
    if not preferences:
        raise ValueError("No valid preferences found in file")
    
    logger.info(f"Loaded {len(preferences)} preference pairs")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(preferences)
    
    return dataset


def load_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer for RLHF training.
    
    Args:
        model_path: Path to base model or SFT checkpoint
        device: Device to load model on
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading kwargs
    model_kwargs = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "device_map": device
    }
    
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    
    # Load model
    try:
        # Try loading as a PEFT model first
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    return model, tokenizer


def run_rlhf(
    model_path: str,
    pref_dataset: Dataset,
    config: Optional[RLHFConfig] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    **kwargs
) -> str:
    """
    Run RLHF training with specified algorithm.
    
    Args:
        model_path: Path to SFT model/adapter
        pref_dataset: Preference dataset
        config: RLHF configuration
        tokenizer: Optional tokenizer (will load if not provided)
        **kwargs: Additional trainer arguments
        
    Returns:
        Path to saved RLHF model/adapter
    """
    if not TRL_AVAILABLE:
        raise ImportError("TRL library is required for RLHF training. Install with: pip install trl")
    
    # Use default config if not provided
    if config is None:
        config = RLHFConfig()
    
    # Load model and tokenizer
    if tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
    
    # Update config with any additional kwargs
    config_dict = config.to_dict()
    config_dict.update(kwargs)
    
    # Select trainer based on algorithm
    if config.algorithm.lower() == "grpo":
        logger.info("Using GRPO (Group-Relative Policy Optimization) trainer")
        
        # Configure GRPO
        training_args = GRPOConfig(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_generations=config.num_generations,
            max_prompt_length=config.max_prompt_length,
            max_completion_length=config.max_completion_length,
            max_steps=config.max_steps,
            beta=config.beta if config.strict_kl else 0.0,
            report_to=config.report_to,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            fp16=config.fp16,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        
        # Create GRPO trainer
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=pref_dataset,
            tokenizer=tokenizer,
        )
        
    elif config.algorithm.lower() == "ppo":
        logger.info("Using PPO (Proximal Policy Optimization) trainer")
        
        # Configure PPO
        training_args = PPOConfig(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            batch_size=config.ppo_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            ppo_epochs=4,  # Standard PPO epochs
            max_steps=config.max_steps,
            kl_penalty=config.kl_penalty,
            report_to=config.report_to,
            log_with=config.report_to,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            fp16=config.fp16,
        )
        
        # Create PPO trainer
        trainer = PPOTrainer(
            model=model,
            config=training_args,
            dataset=pref_dataset,
            tokenizer=tokenizer,
        )
        
    elif config.algorithm.lower() == "dpo":
        logger.info("Using DPO (Direct Preference Optimization)")
        
        # Import DPO components
        try:
            from narrative_engine.dpo_trainer import GenerationDPOTrainer, TripleHeadDPOConfig
            from transformers import TrainingArguments
        except ImportError:
            logger.warning("DPO trainer not found. Falling back to TRL DPOTrainer if available.")
            try:
                from trl import DPOTrainer, DPOConfig
                use_trl_dpo = True
            except ImportError:
                raise ImportError("Neither narrative_engine DPO nor TRL DPO is available")
        else:
            use_trl_dpo = False
        
        if use_trl_dpo:
            # Use TRL's DPO implementation
            training_args = DPOConfig(
                output_dir=config.output_dir,
                learning_rate=config.learning_rate,
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                max_steps=config.max_steps,
                beta=config.dpo_beta,
                warmup_ratio=config.warmup_ratio,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                report_to=config.report_to,
                fp16=config.fp16,
                gradient_checkpointing=config.gradient_checkpointing,
            )
            
            trainer = DPOTrainer(
                model=model,
                args=training_args,
                train_dataset=pref_dataset,
                tokenizer=tokenizer,
            )
        else:
            # Use our custom triple-head DPO implementation
            dpo_config = TripleHeadDPOConfig(
                head_type="generation",  # Focus on generation head for standard DPO
                learning_rate=config.learning_rate,
                beta=config.dpo_beta,
                max_length=config.max_prompt_length,
                batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                warmup_steps=int(config.max_steps * config.warmup_ratio),
            )
            
            training_args = TrainingArguments(
                output_dir=config.output_dir,
                max_steps=config.max_steps,
                per_device_train_batch_size=config.per_device_train_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                warmup_ratio=config.warmup_ratio,
                logging_steps=config.logging_steps,
                save_steps=config.save_steps,
                report_to=config.report_to,
                fp16=config.fp16,
                gradient_checkpointing=config.gradient_checkpointing,
                remove_unused_columns=False,
            )
            
            trainer = GenerationDPOTrainer(
                model=model,
                config=dpo_config,
                args=training_args,
                train_dataset=pref_dataset,
                tokenizer=tokenizer,
            )
    
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}. Choose 'grpo', 'ppo', or 'dpo'")
    
    # Log training start
    logger.info(f"Starting {config.algorithm.upper()} training with {len(pref_dataset)} preference pairs")
    logger.info(f"Max steps: {config.max_steps}, Batch size: {config.per_device_train_batch_size}")
    
    # Train the model
    trainer.train()
    
    # Save the model
    output_path = Path(config.output_dir) / f"adapter_{config.algorithm.lower()}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(str(output_path))
    logger.info(f"Saved {config.algorithm.upper()} model to {output_path}")
    
    # Save training config for reference
    config_path = output_path / "rlhf_config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    return str(output_path)


def has_sufficient_preferences(character_name: str, min_preferences: int = 100) -> bool:
    """
    Check if a character has sufficient preference data for RLHF.
    
    Args:
        character_name: Name of the character
        min_preferences: Minimum number of preference pairs required
        
    Returns:
        True if sufficient preferences exist
    """
    # Look for preference logs in character data directory
    pref_path = Path(f"content/worlds/Default World/characters/{character_name}/preference_logs.ndjson")
    
    if not pref_path.exists():
        return False
    
    # Count preference entries
    count = 0
    try:
        with open(pref_path, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception as e:
        logger.warning(f"Error reading preference file: {e}")
        return False
    
    return count >= min_preferences


# Convenience function for UI integration
def get_rlhf_algorithms() -> List[str]:
    """Get list of available RLHF algorithms"""
    return ["GRPO", "PPO", "DPO"]


def get_default_rlhf_config(algorithm: str = "grpo") -> Dict[str, Any]:
    """Get default configuration for RLHF algorithm"""
    config = RLHFConfig(algorithm=algorithm.lower())
    return config.to_dict() 