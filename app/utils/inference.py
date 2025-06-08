import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class InferenceManager:
    """Manages model inference and testing"""
    
    def __init__(self, base_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
        self.base_model = base_model
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.loaded_models = {}  # Cache for loaded models
        self.project_dir = Path("training_output")
    
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
            return self.loaded_models[model_path]
        
        # Load base model and tokenizer
        base_model, tokenizer = self._load_base_model()
        
        if model_path.startswith("Base:"):
            # Use base model as-is
            model = base_model
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
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            raise ValueError(f"Invalid model path format: {model_path}")
        
        # Cache the loaded model
        self.loaded_models[model_path] = (model, tokenizer)
        
        return model, tokenizer
    
    def generate_response(self, model_path: str, prompt: str, max_new_tokens: int = 150,
                         temperature: float = 0.8, top_p: float = 0.9,
                         repetition_penalty: float = 1.1, do_sample: bool = True) -> str:
        """Generate a response using the specified model"""
        try:
            model, tokenizer = self.load_model(model_path)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
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
                )
            
            # Decode response (only the new tokens)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
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
        """Create a chat context with character card and conversation history"""
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