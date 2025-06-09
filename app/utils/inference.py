import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class InferenceManager:
    """Manages model inference and testing"""
    
    def __init__(self, base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
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
        self.project_dir = Path("training_output")
    
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
    
    def load_model(self, model_path: str) -> tuple:
        """Load a specific model (base, LoRA, or checkpoint)"""
        if model_path in self.loaded_models:
            logger.info(f"Using cached model: {model_path}")
            return self.loaded_models[model_path]
        
        logger.info(f"Loading model: {model_path}")
        
        # Load base model and tokenizer
        base_model, tokenizer = self._load_base_model()
        
        if model_path.startswith("Base:"):
            # Use base model as-is
            model = base_model
            logger.info("Using base model without adapters")
        elif model_path.startswith("LoRA:") or model_path.startswith("Checkpoint:"):
            # Extract character name and optional checkpoint
            parts = model_path.split(": ", 1)[1].split("/")
            character_name = parts[0]
            
            if len(parts) > 1:  # Checkpoint
                checkpoint_name = parts[1]
                adapter_path = self.project_dir / "adapters" / character_name / checkpoint_name
            else:  # Final LoRA
                adapter_path = self.project_dir / "adapters" / character_name
            
            if not adapter_path.exists():
                raise FileNotFoundError(f"Model not found: {adapter_path}")
            
            # Load LoRA adapter
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            raise ValueError(f"Invalid model path format: {model_path}")
        
        # Cache the loaded model
        self.loaded_models[model_path] = (model, tokenizer)
        logger.info(f"Model loaded and cached: {model_path}")
        
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
    
    def generate_response(self, model_path: str, prompt: str, max_new_tokens: int = 150,
                         temperature: float = 0.8, top_p: float = 0.9,
                         repetition_penalty: float = 1.1, do_sample: bool = True,
                         system_prompt: Optional[str] = None) -> str:
        """Generate a response using the specified model"""
        try:
            logger.info(f"Generating response with model: {model_path}")
            logger.debug(f"Raw prompt: {prompt[:100]}...")
            logger.debug(f"Generation params: max_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}")
            
            model, tokenizer = self.load_model(model_path)
            
            # Format prompt properly for chat models (no character context injection)
            logger.info(f"Formatting prompt for model: {model_path}")
            if system_prompt is not None:
                logger.info(f"Using system prompt: {system_prompt[:100]}..." if system_prompt else "Using empty system prompt")
            else:
                logger.info("Using default tokenizer system prompt")
            
            messages = self._format_chat_prompt(prompt, system_prompt)
            
            # Try to use the model's chat template
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                logger.debug("Using tokenizer's chat template")
                
                # For empty system prompt, we want to avoid any default system message
                if system_prompt == "":
                    logger.debug("Attempting to bypass default system prompt")
                    # Some tokenizers might still inject default system prompt
                    # Try to apply template and check result
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    # Check if default system prompt was injected anyway
                    if "helpful AI assistant named SmolLM" in formatted_prompt:
                        logger.warning("Default system prompt detected in output, using fallback formatting")
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
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # Add some safety parameters
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            
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
        
        char_manager = CharacterManager()
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
    
    def clear_model_cache(self):
        """Clear the model cache to free up memory"""
        self.loaded_models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None 