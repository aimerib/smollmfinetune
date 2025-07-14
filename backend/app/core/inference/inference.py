import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
import json
import uuid

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Import observability components
from .observability import (
    ObservabilityLogger, 
    InferenceObservabilityData, 
    generate_request_id,
    extract_token_probabilities,
    get_observability_logger
)

# Import NarrativeLLM for advanced adapter management
try:
    from ..narrative_engine.model import NarrativeLLM, create_narrative_model
    from ..narrative_engine.config import NarrativeLLMConfig
    NARRATIVE_ENGINE_AVAILABLE = True
except ImportError:
    try:
        # Try absolute import for standalone scripts
        from .narrative_engine.model import NarrativeLLM, create_narrative_model
        from .narrative_engine.config import NarrativeLLMConfig
        NARRATIVE_ENGINE_AVAILABLE = True
    except ImportError:
        NARRATIVE_ENGINE_AVAILABLE = False


class InferenceManager:
    """Manages model inference and testing"""
    
    def __init__(self, base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
        self.default_base_model = base_model
        self.base_model = base_model
        # Better device handling
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        logger.info(f"InferenceManager initialized with device: {self.device}")
        
        self.loaded_models = {}  # Cache for loaded models
        self.max_cached_models = 2  # Limit cache size to prevent memory issues
        self.cache_access_count = {}  # Track model usage for LRU eviction
        self.project_dir = Path("training_output")
        
        # NEW: NarrativeLLM management for hot-swapping
        self.narrative_models = {}  # character_name -> NarrativeLLM instance
        self.character_adapters = {}  # character_name -> {adapter_type: adapter_path}
        self.active_adapters = {}  # character_name -> currently_active_adapter_name
        
        if NARRATIVE_ENGINE_AVAILABLE:
            logger.info("✨ NarrativeLLM hot-swapping capabilities enabled")
        else:
            logger.warning("⚠️ NarrativeLLM not available - enhanced adapter features disabled")
    
    def set_base_model(self, model_name: str):
        """Update the base model for inference"""
        self.base_model = model_name
        # Clear cache since base model changed
        self.clear_model_cache()
        logger.info(f"Base model updated to: {model_name}")
    
    def _load_base_model(self):
        """Load the base model and tokenizer"""
        logger.info(f"Loading base model: {self.base_model}")
        
        # Better device mapping and dtype handling
        if self.device == "cuda":
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            # For MPS and CPU, don't use device_map
            device_map = None
            torch_dtype = torch.float32
        
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
        
        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Model has {model.num_parameters():,} parameters")
        
        return model, tokenizer
    
    def get_available_models(self) -> List[str]:
        """Get list of available trained models"""
        if not self.project_dir.exists():
            return []
        
        models = []
        adapters_dir = self.project_dir / "adapters"
        
        if adapters_dir.exists():
            for character_dir in adapters_dir.iterdir():
                if character_dir.is_dir():
                    # Check if it contains adapter files
                    if (character_dir / "adapter_config.json").exists():
                        models.append(f"LoRA: {character_dir.name}")
                    
                    # Check for checkpoints
                    for checkpoint_dir in character_dir.iterdir():
                        if (checkpoint_dir.is_dir() and 
                            checkpoint_dir.name.startswith('checkpoint-') and
                            (checkpoint_dir / "adapter_config.json").exists()):
                            models.append(f"Checkpoint: {character_dir.name}/{checkpoint_dir.name}")
        
        # Add base model option
        models.insert(0, f"Base: {self.base_model}")
        
        return models
    
    def _get_model_path(self, model_identifier: str) -> Optional[Path]:
        """Resolve a model identifier to a filesystem path."""
        if model_identifier.startswith("Base:"):
            return None
            
        parts = model_identifier.split(": ", 1)[1].split("/")
        character_name = parts[0]
        
        if len(parts) > 1:  # Checkpoint
            checkpoint_name = parts[1]
            adapter_path = self.project_dir / "adapters" / character_name / checkpoint_name
        else:  # Final LoRA
            adapter_path = self.project_dir / "adapters" / character_name
        
        return adapter_path if adapter_path.exists() else None

    def get_model_metrics(self, model_identifier: str) -> Dict[str, Any]:
        """Load training metrics from the training_summary.json file."""
        # Handle raw HuggingFace model IDs (base models) - they don't have training metrics
        if not model_identifier.startswith(("Base:", "LoRA:", "Checkpoint:")) and "/" in model_identifier:
            logger.debug(f"Base model {model_identifier} has no training metrics")
            return {}
        
        model_path = self._get_model_path(model_identifier)
        if not model_path:
            return {}

        summary_path = model_path / "training_summary.json"
        if summary_path.exists():
            with summary_path.open('r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {summary_path}")
                    return {}
        return {}
    
    def get_model_metadata(self, model_identifier: str) -> Dict[str, Any]:
        """Load training metadata including base model information."""
        # Handle raw HuggingFace model IDs (base models) - they don't have training metadata
        if not model_identifier.startswith(("Base:", "LoRA:", "Checkpoint:")) and "/" in model_identifier:
            logger.debug(f"Base model {model_identifier} has no training metadata")
            return {}
        
        model_path = self._get_model_path(model_identifier)
        if not model_path:
            return {}

        metadata_path = model_path / "training_metadata.json"
        if metadata_path.exists():
            with metadata_path.open('r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {metadata_path}")
                    return {}
        return {}
    
    def get_required_base_model(self, model_identifier: str) -> Optional[str]:
        """Get the base model required for this adapter."""
        metadata = self.get_model_metadata(model_identifier)
        return metadata.get('base_model')
    
    def load_model(self, model_path: str) -> tuple:
        """Load a specific model (base, LoRA, or checkpoint)"""
        if model_path in self.loaded_models:
            logger.info(f"Using cached model: {model_path}")
            # Update access count for LRU
            self.cache_access_count[model_path] = self.cache_access_count.get(model_path, 0) + 1
            return self.loaded_models[model_path]
        
        # Check if cache is full and evict least recently used if needed
        if len(self.loaded_models) >= self.max_cached_models:
            # Find least recently used model (lowest access count)
            lru_model = min(self.cache_access_count.items(), key=lambda x: x[1])[0]
            logger.info(f"Cache full - evicting least used model: {lru_model}")
            
            # Remove from cache
            del self.loaded_models[lru_model]
            del self.cache_access_count[lru_model]
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Loading model: {model_path}")
        
        # For LoRA/DoRA models, check if we need a specific base model
        if model_path.startswith("LoRA:") or model_path.startswith("Checkpoint:"):
            required_base_model = self.get_required_base_model(model_path)
            if required_base_model and required_base_model != self.base_model:
                logger.info(f"Adapter requires base model: {required_base_model}, current: {self.base_model}")
                logger.info(f"Switching to required base model: {required_base_model}")
                
                # Temporarily switch to the required base model
                original_base_model = self.base_model
                self.base_model = required_base_model
                
                try:
                    base_model, tokenizer = self._load_base_model()
                except Exception as e:
                    # Restore original base model on failure
                    self.base_model = original_base_model
                    raise RuntimeError(f"Failed to load required base model {required_base_model}: {e}") from e
            elif not required_base_model:
                # No metadata found - this is an old model
                # Try to detect the correct base model based on adapter dimensions
                logger.warning(f"No metadata found for {model_path}. Attempting to detect correct base model.")
                
                # First try with current base model
                try:
                    base_model, tokenizer = self._load_base_model()
                    # Try loading the adapter to see if dimensions match
                    adapter_path = self._get_model_path(model_path)
                    if adapter_path:
                        config_path = adapter_path / "adapter_config.json"
                        if config_path.exists():
                            with open(config_path, 'r') as f:
                                adapter_config = json.load(f)
                            
                            # Check if this looks like a mismatch based on common patterns
                            if 'modules' in adapter_config:
                                # Look for dimension hints in the config
                                pass  # Current base model should work
                except Exception:
                    # If current base model fails, try 360M (common for DoRA models)
                    logger.info("Current base model failed, trying SmolLM2-360M for compatibility")
                    original_base_model = self.base_model
                    self.base_model = "HuggingFaceTB/SmolLM2-360M-Instruct"
                    
                    try:
                        base_model, tokenizer = self._load_base_model()
                    except Exception as e:
                        # Restore original and fail
                        self.base_model = original_base_model
                        raise RuntimeError(f"Failed to load adapter with both 135M and 360M base models. Please retrain the model or check compatibility: {e}") from e
            else:
                base_model, tokenizer = self._load_base_model()
        else:
            # Load base model and tokenizer
            base_model, tokenizer = self._load_base_model()
        
        if model_path.startswith("Base:"):
            # Use base model as-is
            model = base_model
            logger.info("Using base model without adapters")
        elif model_path.startswith("LoRA:") or model_path.startswith("Checkpoint:"):
            # Extract character name and optional checkpoint
            adapter_path = self._get_model_path(model_path)
            if not adapter_path:
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            # Load LoRA adapter with DoRA optimization if needed
            logger.info(f"Loading LoRA/DoRA adapter from: {adapter_path}")
            
            # Check if DoRA optimization should be enabled
            metadata = self.get_model_metadata(model_path)
            ephemeral_gpu_offload = (metadata.get('use_dora', False) and 
                                   self.device == "cuda")
            
            if ephemeral_gpu_offload:
                logger.info("Enabling DoRA ephemeral GPU offload for inference optimization")
                model = PeftModel.from_pretrained(base_model, str(adapter_path), 
                                                ephemeral_gpu_offload=True)
            else:
                model = PeftModel.from_pretrained(base_model, str(adapter_path))
                
            # Set model to eval mode for DoRA optimization
            if metadata.get('use_dora', False):
                model.eval()
                logger.info("DoRA model set to eval mode for optimal performance")
        else:
            raise ValueError(f"Invalid model path format: {model_path}")
        
        # Cache the loaded model
        self.loaded_models[model_path] = (model, tokenizer)
        self.cache_access_count[model_path] = 1  # Initialize access count
        logger.info(f"Model loaded and cached: {model_path} (cache size: {len(self.loaded_models)}/{self.max_cached_models})")
        
        return model, tokenizer
    
    def _format_chat_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """Format prompt using proper chat template for SmolLM2-135M-Instruct"""
        
        messages = []
        
        # Add system prompt if provided
        if system_prompt is not None:
            if system_prompt.strip():  # Only add if not empty
                messages.append({"role": "system", "content": system_prompt})
            # If system_prompt is empty string, we deliberately skip adding any system message
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _generate_with_model(self, model, tokenizer, prompt: str, max_tokens: int,
                           temperature: float, top_p: float, repetition_penalty: float,
                           do_sample: bool, system_prompt: Optional[str], seed: Optional[int] = None,
                           enable_observability: bool = False, request_id: Optional[str] = None,
                           model_path: Optional[str] = None) -> str:
        """Helper method to generate response with a loaded model and tokenizer"""
        import torch
        import random
        import numpy as np
        
        # Set seeds for reproducible generation if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # Ensure deterministic behavior
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            random.seed(seed)
            np.random.seed(seed)
            logger.debug(f"Set random seed to {seed} for reproducible generation")
        
        messages = self._format_chat_prompt(prompt, system_prompt)
        
        # Try to use the model's chat template
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            logger.debug("Using tokenizer's chat template")
            
            # For empty system prompt, we want to avoid any default system message
            if system_prompt == "":
                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
        else:
            # Fallback formatting for models without chat template
            logger.debug("Using fallback prompt formatting")
            if system_prompt and system_prompt.strip():
                formatted_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
            else:
                formatted_prompt = f"User: {prompt}\nAssistant:"
        
        logger.debug(f"Formatted prompt: {formatted_prompt[:200]}...")
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Ensure inputs are on the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        logger.debug(f"Input tokens: {inputs['input_ids'].shape}")
        logger.debug(f"Model device: {next(model.parameters()).device}")
        logger.debug(f"Input device: {inputs['input_ids'].device}")
        
        # Generate response
        with torch.no_grad():
            # Set seed again right before generation for extra determinism
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Configure generation parameters based on observability needs
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                # SmolLM2 specific optimizations
                "use_cache": True,
                # Prevent empty responses
                "min_new_tokens": 1,
                # Better stopping criteria
                "early_stopping": False,
            }
            
            # Add observability flags if enabled
            if enable_observability:
                generation_kwargs.update({
                    "output_attentions": True,
                    "output_hidden_states": True,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                })
                logger.info(f"Enabling observability for request {request_id}")
            else:
                generation_kwargs.update({
                    "output_attentions": False,
                    "output_hidden_states": False,
                })
            
            outputs = model.generate(**inputs, **generation_kwargs)
        
        # Handle different output formats based on observability settings
        if enable_observability:
            # outputs is a GenerateDecoderOnlyOutput object
            generated_sequences = outputs.sequences
            logger.debug(f"Generated sequences shape: {generated_sequences.shape}")
            
            # Decode response (only the new tokens)
            input_length = inputs['input_ids'].shape[-1]
            generated_tokens = generated_sequences[0][input_length:]
            
            logger.debug(f"New tokens generated: {len(generated_tokens)}")
            
            if len(generated_tokens) == 0:
                logger.warning("No new tokens generated!")
                return "No response generated. Try adjusting generation parameters."
            
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Log observability data if enabled
            if request_id and model_path:
                try:
                    # Extract observability data
                    attention_weights = outputs.attentions if hasattr(outputs, 'attentions') and outputs.attentions else []
                    hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') and outputs.hidden_states else []
                    token_probs = extract_token_probabilities(outputs, tokenizer, top_k=10)
                    
                    # Create observability data structure
                    obs_data = InferenceObservabilityData(
                        request_id=request_id,
                        prompt=prompt,
                        response=response,
                        model_path=model_path,
                        generation_config={
                            "temperature": temperature,
                            "top_p": top_p,
                            "max_tokens": max_tokens,
                            "repetition_penalty": repetition_penalty,
                            "do_sample": do_sample,
                            "seed": seed
                        },
                        attention_weights=attention_weights,
                        hidden_states=hidden_states,
                        token_probabilities=token_probs
                    )
                    
                    # Log the data
                    obs_logger = get_observability_logger()
                    obs_logger.log_inference_data(obs_data)
                    
                    logger.info(f"Logged observability data for request {request_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to log observability data: {e}")
        else:
            # outputs is just the token sequences
            logger.debug(f"Generated tokens: {outputs.shape}")
            
            # Decode response (only the new tokens)
            input_length = inputs['input_ids'].shape[-1]
            generated_tokens = outputs[0][input_length:]
            
            logger.debug(f"New tokens generated: {len(generated_tokens)}")
            
            if len(generated_tokens) == 0:
                logger.warning("No new tokens generated!")
                return "No response generated. Try adjusting generation parameters."
            
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        logger.debug(f"Raw decoded response: {response[:100]}...")
        
        # Clean up response
        response = response.strip()
        
        # Remove any remaining special tokens or artifacts
        if response.startswith(("User:", "Assistant:", "System:")):
            response = response.split(":", 1)[1].strip()
        
        logger.info(f"Final response length: {len(response)} characters")
        
        if not response:
            logger.warning("Empty response after processing!")
            return "Empty response generated. Check model and generation parameters."
        
        logger.info("Response generation completed successfully")
        return response
    
    def generate_response(self, model_path: str, prompt: str, max_tokens: int = 150,
                         temperature: float = 0.8, top_p: float = 0.9,
                         repetition_penalty: float = 1.1, do_sample: bool = True,
                         system_prompt: Optional[str] = None, seed: Optional[int] = None,
                         enable_observability: bool = False, request_id: Optional[str] = None) -> str:
        """Generate a response using the specified model"""
        try:
            logger.info(f"Generating response with model: {model_path}")
            logger.debug(f"Raw prompt: {prompt[:100]}...")
            logger.debug(f"Generation params: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
            if seed is not None:
                logger.info(f"Using seed {seed} for reproducible generation")
            
            # Handle raw HuggingFace model IDs (base models from comparison)
            if not model_path.startswith(("Base:", "LoRA:", "Checkpoint:")) and "/" in model_path:
                # This is a raw HuggingFace model ID, treat it as a base model
                logger.info(f"Detected raw HuggingFace model ID: {model_path}, treating as base model")
                original_base_model = self.base_model
                
                try:
                    # Temporarily switch to the requested base model
                    self.base_model = model_path
                    model, tokenizer = self._load_base_model()
                    
                    # Generate request ID if not provided and observability is enabled
                    if enable_observability and not request_id:
                        request_id = generate_request_id()
                        logger.info(f"Generated request ID for observability: {request_id}")
                    
                    # Generate the response with this model
                    response = self._generate_with_model(model, tokenizer, prompt, max_tokens, 
                                                       temperature, top_p, repetition_penalty, 
                                                       do_sample, system_prompt, seed,
                                                       enable_observability, request_id, model_path)
                    return response
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to load/generate with base model {model_path}: {e}") from e
                finally:
                    # Always restore the original base model after generation
                    self.base_model = original_base_model
            else:
                # Use the existing load_model logic for prefixed paths
                model, tokenizer = self.load_model(model_path)
            
            # Format prompt properly for chat models (no character context injection)
            logger.info(f"Formatting prompt for model: {model_path}")
            if system_prompt is not None:
                logger.info(f"Using system prompt: {system_prompt[:100]}..." if system_prompt else "Using empty system prompt")
            else:
                logger.info("Using default tokenizer system prompt")
            
            # Generate request ID if not provided and observability is enabled
            if enable_observability and not request_id:
                request_id = generate_request_id()
                logger.info(f"Generated request ID for observability: {request_id}")
            
            # Use the helper method for generation
            return self._generate_with_model(model, tokenizer, prompt, max_tokens, 
                                           temperature, top_p, repetition_penalty, 
                                           do_sample, system_prompt, seed, 
                                           enable_observability, request_id, model_path)
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
    
    def test_model_quality(self, model_path: str, test_prompts: List[str]) -> Dict[str, Any]:
        """Test model quality with a set of prompts"""
        results = {
            'model_path': model_path,
            'responses': [],
            'avg_length': 0,
            'response_diversity': 0,
        }
        
        responses = []
        for prompt in test_prompts:
            try:
                response = self.generate_response(model_path, prompt)
                responses.append({
                    'prompt': prompt,
                    'response': response,
                    'length': len(response.split())
                })
            except Exception as e:
                responses.append({
                    'prompt': prompt,
                    'response': f"Error: {str(e)}",
                    'length': 0
                })
        
        results['responses'] = responses
        
        # Calculate metrics
        if responses:
            lengths = [r['length'] for r in responses]
            results['avg_length'] = sum(lengths) / len(lengths) if lengths else 0
            
            # Response diversity (unique responses / total responses)
            unique_responses = len(set(r['response'] for r in responses if not r['response'].startswith('Error:')))
            results['response_diversity'] = unique_responses / len(responses) if responses else 0
        
        return results
    
    def compare_models(self, model_paths: List[str], test_prompts: List[str]) -> Dict[str, Any]:
        """Compare multiple models on the same set of prompts"""
        comparison = {
            'test_prompts': test_prompts,
            'models': {},
            'summary': {}
        }
        
        for model_path in model_paths:
            results = self.test_model_quality(model_path, test_prompts)
            comparison['models'][model_path] = results
        
        # Generate summary comparisons
        if comparison['models']:
            avg_lengths = {path: results['avg_length'] for path, results in comparison['models'].items()}
            diversities = {path: results['response_diversity'] for path, results in comparison['models'].items()}
            
            comparison['summary'] = {
                'best_avg_length': max(avg_lengths.items(), key=lambda x: x[1]) if avg_lengths else None,
                'best_diversity': max(diversities.items(), key=lambda x: x[1]) if diversities else None,
                'avg_length_range': (min(avg_lengths.values()), max(avg_lengths.values())) if avg_lengths else (0, 0),
                'diversity_range': (min(diversities.values()), max(diversities.values())) if diversities else (0, 0),
            }
        
        return comparison
    
    def create_chat_context(self, character: Dict[str, Any], conversation_history: List[Dict[str, str]] = None) -> str:
        """Create a chat context with character card and conversation history
        
        NOTE: This method is available for other use cases but is NOT used in model testing
        to ensure pure LoRA evaluation without character context injection.
        """
        from .character import CharacterManager
        
        char_manager = CharacterManager(world_manager=self.world_manager, client=self.client)
        card_block = char_manager.make_card_block(character)
        
        context_parts = [card_block]
        
        if conversation_history:
            context_parts.append("\n### Previous Conversation:")
            for turn in conversation_history:
                role = turn.get('role', 'user')
                content = turn.get('content', '')
                if role == 'user':
                    context_parts.append(f"User: {content}")
                else:
                    char_name = character.get('name', 'Assistant')
                    context_parts.append(f"{char_name}: {content}")
        
        return "\n".join(context_parts)
    
    def interactive_chat(self, model_path: str, character: Dict[str, Any]) -> None:
        """Start an interactive chat session (for CLI usage)"""
        print(f"Starting chat with {character.get('name', 'Character')} using {model_path}")
        print("Type 'quit' to exit, 'reset' to clear conversation history\n")
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'reset':
                    conversation_history.clear()
                    print("Conversation history cleared.\n")
                    continue
                elif not user_input:
                    continue
                
                # Create context with conversation history
                context = self.create_chat_context(character, conversation_history)
                full_prompt = f"{context}\nUser: {user_input}\n{character.get('name', 'Assistant')}:"
                
                # Generate response
                response = self.generate_response(model_path, full_prompt)
                
                print(f"{character.get('name', 'Assistant')}: {response}\n")
                
                # Update conversation history
                conversation_history.append({'role': 'user', 'content': user_input})
                conversation_history.append({'role': 'assistant', 'content': response})
                
                # Keep conversation history manageable
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-8:]
                
            except KeyboardInterrupt:
                print("\nChat interrupted.")
                break
            except Exception as e:
                print(f"Error: {str(e)}\n")
        
        logger.info("Chat session ended")

    def clear_model_cache(self):
        """Clear all cached models to free memory"""
        logger.info("Clearing model cache...")
        self.loaded_models.clear()
        self.cache_access_count.clear()
        
        # Also clear NarrativeLLM models
        self.narrative_models.clear()
        self.character_adapters.clear()
        self.active_adapters.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")

    # ============================================================================
    # NEW: NarrativeLLM Hot-Swapping Capabilities
    # ============================================================================
    
    def _find_all_character_adapters(self, character_name: str) -> Dict[str, Path]:
        """Find all available adapters for a character"""
        adapters = {}
        adapter_dir = self.project_dir / "adapters" / character_name
        
        if not adapter_dir.exists():
            return adapters
        
        # Check for SFT adapter (main directory)
        if (adapter_dir / "adapter.safetensors").exists():
            adapters["SFT"] = adapter_dir
        
        # Check for RLHF adapters
        rlhf_dir = adapter_dir / "rlhf_output"
        if rlhf_dir.exists():
            for rlhf_type in ["adapter_grpo", "adapter_ppo"]:
                rlhf_path = rlhf_dir / rlhf_type
                if rlhf_path.exists() and (rlhf_path / "adapter.safetensors").exists():
                    adapter_type = f"RLHF-{rlhf_type.split('_')[1].upper()}"
                    adapters[adapter_type] = rlhf_path
        
        # Check for checkpoints
        for checkpoint_dir in adapter_dir.iterdir():
            if (checkpoint_dir.is_dir() and 
                checkpoint_dir.name.startswith('checkpoint-') and
                (checkpoint_dir / "adapter.safetensors").exists()):
                checkpoint_name = checkpoint_dir.name
                adapters[f"Checkpoint-{checkpoint_name}"] = checkpoint_dir
        
        logger.info(f"Found {len(adapters)} adapters for {character_name}: {list(adapters.keys())}")
        return adapters
    
    def _get_preferred_adapter_type(self, adapters: Dict[str, Path]) -> str:
        """Get the preferred adapter type (RLHF > SFT > Checkpoint)"""
        # Preference order: RLHF-GRPO > RLHF-PPO > SFT > Checkpoints
        preference_order = ["RLHF-GRPO", "RLHF-PPO", "SFT"]
        
        for preferred in preference_order:
            if preferred in adapters:
                return preferred
        
        # Fall back to any checkpoint
        for adapter_type in adapters.keys():
            if adapter_type.startswith("Checkpoint-"):
                return adapter_type
        
        # Return first available if none match preferences
        return list(adapters.keys())[0] if adapters else None
    
    def load_character_with_hot_swap(self, character_name: str) -> str:
        """Load character into NarrativeLLM with hot-swapping capability"""
        if not NARRATIVE_ENGINE_AVAILABLE:
            return "❌ NarrativeLLM not available. Please install narrative engine components."
        
        try:
            # Find all available adapters for this character
            adapters = self._find_all_character_adapters(character_name)
            
            if not adapters:
                return f"❌ No adapters found for character '{character_name}'"
            
            # Store adapter info
            self.character_adapters[character_name] = adapters
            
            # Create NarrativeLLM instance
            config = NarrativeLLMConfig(base_model_name=self.base_model)
            narrative_model = NarrativeLLM(config)
            
            # Load all adapters into the model
            loaded_count = 0
            for adapter_type, adapter_path in adapters.items():
                try:
                    adapter_name = f"{character_name}_{adapter_type}"
                    narrative_model.load_adapter(str(adapter_path), adapter_name)
                    loaded_count += 1
                    logger.info(f"✅ Loaded {adapter_type} adapter for {character_name}")
                except Exception as e:
                    logger.error(f"❌ Failed to load {adapter_type} adapter: {e}")
            
            if loaded_count == 0:
                return f"❌ Failed to load any adapters for {character_name}"
            
            # Set default adapter (prefer RLHF > SFT)
            preferred_type = self._get_preferred_adapter_type(adapters)
            default_adapter = f"{character_name}_{preferred_type}"
            narrative_model.set_active_adapter(default_adapter)
            
            # Store the model and track active adapter
            self.narrative_models[character_name] = narrative_model
            self.active_adapters[character_name] = default_adapter
            
            logger.info(f"🎮 Character '{character_name}' loaded with {loaded_count} adapters, active: {preferred_type}")
            return f"✅ Loaded {character_name} with {loaded_count} adapters (active: {preferred_type})"
            
        except Exception as e:
            logger.error(f"Failed to load character {character_name}: {e}")
            return f"❌ Failed to load character: {str(e)}"
    
    def switch_character_adapter(self, character_name: str, adapter_type: str) -> str:
        """Hot-swap between character adapters"""
        if not NARRATIVE_ENGINE_AVAILABLE:
            return "❌ NarrativeLLM not available"
        
        if character_name not in self.narrative_models:
            return f"❌ Character '{character_name}' not loaded. Load it first with load_character_with_hot_swap()"
        
        if character_name not in self.character_adapters:
            return f"❌ No adapter info for character '{character_name}'"
        
        available_adapters = self.character_adapters[character_name]
        if adapter_type not in available_adapters:
            available_types = list(available_adapters.keys())
            return f"❌ Adapter type '{adapter_type}' not available. Available: {available_types}"
        
        try:
            model = self.narrative_models[character_name]
            adapter_name = f"{character_name}_{adapter_type}"
            
            # Switch to the requested adapter
            model.set_active_adapter(adapter_name)
            self.active_adapters[character_name] = adapter_name
            
            logger.info(f"🔄 Switched {character_name} to {adapter_type} adapter")
            return f"✅ Switched {character_name} to {adapter_type} mode"
            
        except Exception as e:
            logger.error(f"Failed to switch adapter for {character_name}: {e}")
            return f"❌ Failed to switch adapter: {str(e)}"
    
    def get_character_info(self, character_name: str) -> Dict[str, Any]:
        """Get information about a loaded character"""
        if character_name not in self.narrative_models:
            return {"loaded": False, "error": "Character not loaded"}
        
        available_adapters = self.character_adapters.get(character_name, {})
        active_adapter = self.active_adapters.get(character_name, "Unknown")
        
        # Extract just the adapter type from the full adapter name
        active_type = active_adapter.replace(f"{character_name}_", "") if active_adapter else "Unknown"
        
        # Safely get model device
        try:
            model_device = str(next(self.narrative_models[character_name].base_model.parameters()).device)
        except (StopIteration, AttributeError):
            model_device = "unknown"
        
        return {
            "loaded": True,
            "character_name": character_name,
            "available_adapters": list(available_adapters.keys()),
            "active_adapter": active_type,
            "total_adapters": len(available_adapters),
            "model_device": model_device
        }
    
    def generate_with_character(self, character_name: str, prompt: str, 
                              max_tokens: int = 150, temperature: float = 0.8,
                              top_p: float = 0.9, **kwargs) -> Dict[str, Any]:
        """Generate response using NarrativeLLM with control tokens"""
        if not NARRATIVE_ENGINE_AVAILABLE:
            return {"error": "NarrativeLLM not available"}
        
        if character_name not in self.narrative_models:
            return {"error": f"Character '{character_name}' not loaded"}
        
        try:
            model = self.narrative_models[character_name]
            tokenizer = model.tokenizer
            
            # Tokenize the input
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.base_model.device) for k, v in inputs.items()}
            
            # Generate with control tokens
            result = model.generate_with_control(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                user_input=prompt,
                previous_context="",  # Could be enhanced with conversation history
                **kwargs
            )
            
            # Get current adapter info
            active_adapter = self.active_adapters.get(character_name, "Unknown")
            active_type = active_adapter.replace(f"{character_name}_", "") if active_adapter else "Unknown"
            
            return {
                "response": result["generated_text"],
                "control_tokens": result.get("control_tokens", []),
                "emotional_state": result.get("emotional_state", {}),
                "active_adapter": active_type,
                "character_name": character_name,
                "surprise_score": result.get("surprise_score", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Generation failed for {character_name}: {e}")
            return {"error": f"Generation failed: {str(e)}"}
    
    def list_loaded_characters(self) -> List[Dict[str, Any]]:
        """List all loaded characters with their adapter information"""
        characters = []
        for character_name in self.narrative_models.keys():
            info = self.get_character_info(character_name)
            characters.append(info)
        return characters
    
    def unload_character(self, character_name: str) -> str:
        """Unload a character to free memory"""
        if character_name not in self.narrative_models:
            return f"❌ Character '{character_name}' not loaded"
        
        try:
            # Remove from all tracking dictionaries
            del self.narrative_models[character_name]
            if character_name in self.character_adapters:
                del self.character_adapters[character_name]
            if character_name in self.active_adapters:
                del self.active_adapters[character_name]
            
            # Clear GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"🗑️ Unloaded character '{character_name}'")
            return f"✅ Unloaded character '{character_name}'"
            
        except Exception as e:
            logger.error(f"Failed to unload character {character_name}: {e}")
            return f"❌ Failed to unload character: {str(e)}" 