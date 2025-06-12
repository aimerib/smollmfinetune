"""
llama-cpp-python inference engine for GGUF models.
Handles GGUF model downloading, caching, and inference.
"""

import os
import asyncio
import threading
import secrets
import traceback
from typing import Optional, List, Tuple, Dict
import logging
from pathlib import Path
import hashlib
import requests
from tqdm import tqdm

from .inference_engines import InferenceEngine

logger = logging.getLogger(__name__)


class LlamaCppEngine(InferenceEngine):
    """llama-cpp-python engine for GGUF models with download/cache support"""

    # Class-level singleton to ensure model is loaded only once
    _instance = None
    _llm = None
    _model_loaded = False
    _initializing = False
    _generation_lock = None
    _generation_semaphore = None  # Prevent concurrent generation calls

    # GGUF cache directory
    if Path("/workspace").exists():
        _gguf_cache_dir = Path("/workspace") / ".cache" / "llamacpp_gguf"
    else:
        _gguf_cache_dir = Path.home() / ".cache" / "llamacpp_gguf"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, gguf_file: Optional[str] = None,
                 tokenizer_name: Optional[str] = None,
                 n_ctx: int = 4096,
                 n_threads: Optional[int] = None,
                 n_gpu_layers: int = -1):
        """
        Initialize LlamaCpp engine with GGUF model
        
        Args:
            gguf_file: GGUF model string in format 'repo_id/filename@tokenizer_repo'
            tokenizer_name: Optional tokenizer override
            n_ctx: Context window size
            n_threads: Number of threads (auto-detect if None)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all, 0 = CPU only)
        """
        # Initialize parent class first
        super().__init__()
        
        # Prevent re-initialization to avoid Streamlit infinite loops
        if hasattr(self, '_initialized'):
            current_gguf = getattr(self, 'gguf_file', None)
            if gguf_file is None or gguf_file == current_gguf:
                logger.debug(f"LlamaCppEngine already initialized with {current_gguf}, skipping")
                return

        # Configuration
        self.gguf_file = gguf_file
        self.tokenizer_name = tokenizer_name
        self.n_ctx = int(os.getenv('MAX_MODEL_LEN', str(n_ctx)))  # Use env var but fall back to parameter
        self.n_threads = n_threads or os.cpu_count()
        self.n_gpu_layers = n_gpu_layers
        
        # Force reload if model changed
        if hasattr(self, '_initialized') and gguf_file != getattr(self, 'gguf_file', None):
            logger.info(f"🔄 Model changed from {getattr(self, 'gguf_file', None)} to {gguf_file}, forcing reload")
            LlamaCppEngine._model_loaded = False
            LlamaCppEngine._llm = None
            LlamaCppEngine._initializing = False  # Reset initializing flag to allow re-initialization

        self._available = None
        
        # Mark as initialized
        self._initialized = True

        # Initialize the lock if not already done
        if LlamaCppEngine._generation_lock is None:
            LlamaCppEngine._generation_lock = threading.Lock()

        # Create GGUF cache directory
        LlamaCppEngine._gguf_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "Llama.cpp"

    def is_available(self) -> bool:
        """Check if llama-cpp-python is available"""
        if self._available is not None:
            return self._available

        try:
            from llama_cpp import Llama
            self._available = True
            logger.info("llama-cpp-python detected and available")
        except ImportError:
            self._available = False
            logger.debug("llama-cpp-python not installed")
        except Exception as e:
            self._available = False
            logger.debug(f"llama-cpp-python not available: {e}")

        return self._available

    def _download_gguf_file(self, repo_id: str, filename: str) -> Path:
        """Download GGUF file from HuggingFace if not cached"""
        # Create a unique cache key based on repo and filename
        cache_key = hashlib.md5(f"{repo_id}/{filename}".encode()).hexdigest()
        cached_path = self._gguf_cache_dir / f"{cache_key}_{filename}"
        
        # Check if already cached
        if cached_path.exists():
            logger.info(f"Using cached GGUF file: {cached_path}")
            return cached_path
        
        # Download from HuggingFace
        logger.info(f"Downloading GGUF file: {repo_id}/{filename}")
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size
            file_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(cached_path, 'wb') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Downloaded GGUF file to: {cached_path}")
            return cached_path
            
        except Exception as e:
            # Clean up partial download
            if cached_path.exists():
                cached_path.unlink()
            raise RuntimeError(f"Failed to download GGUF file: {e}")

    def _parse_gguf_model_string(self, model_string: str) -> Tuple[str, str, Optional[str]]:
        import json
        """Parse GGUF model string format: 'repo_id/filename@tokenizer_repo'
        
        Examples:
        - 'TheBloke/Llama-2-70B-GGUF/llama-2-70b.Q4_K_M.gguf@meta-llama/Llama-2-70b-hf'
        - 'TheBloke/Mistral-7B-GGUF/mistral-7b-instruct-v0.2.Q5_K_M.gguf'
        
        Returns: (repo_id, gguf_filename, tokenizer_repo)
        """
        # Split by @ to separate tokenizer if provided
        parts = model_string.split('@')
        gguf_part = parts[0]
        tokenizer_repo = parts[1] if len(parts) > 1 else None
        
        # Extract repo_id and filename
        if '/' not in gguf_part:
            raise ValueError(f"Invalid GGUF format: {model_string}. Expected 'repo_id/filename'")
        
        # Find the last / that separates repo from filename
        last_slash = gguf_part.rfind('/')
        repo_id = gguf_part[:last_slash]
        filename = gguf_part[last_slash + 1:]
        try:
            files_array = json.loads(f"{{\"filenames\": {filename}}}")
            filename = files_array["filenames"]
        except Exception as e:
            logger.error(f"Error parsing GGUF filename: {e}")
            traceback.print_exc()
            pass
        
        if not filename.endswith('.gguf') and not isinstance(filename, list):
            raise ValueError(f"Invalid GGUF filename: {filename}. Must end with .gguf or be a list of filenames.")
        
        return repo_id, filename, tokenizer_repo

    def _initialize_model(self):
        """Initialize llama-cpp model once (singleton pattern)"""
        with LlamaCppEngine._generation_lock:
            # Double-check pattern: check again inside the lock
            if LlamaCppEngine._model_loaded and LlamaCppEngine._llm is not None:
                return
            
            # Prevent concurrent initialization attempts
            if hasattr(LlamaCppEngine, '_initializing') and LlamaCppEngine._initializing:
                logger.info("Model initialization already in progress, waiting...")
                import time
                max_wait = 300  # 5 minutes max wait
                waited = 0
                while LlamaCppEngine._initializing and waited < max_wait:
                    time.sleep(1)
                    waited += 1
                
                if LlamaCppEngine._model_loaded and LlamaCppEngine._llm is not None:
                    return
                else:
                    raise RuntimeError("Model initialization by other thread failed")
            
            # Mark as initializing
            LlamaCppEngine._initializing = True

        try:
            from llama_cpp import Llama
            
            if not self.gguf_file:
                raise ValueError("GGUF file must be specified for LlamaCppEngine")

            # Parse GGUF specification
            repo_id, filename, tokenizer_repo = self._parse_gguf_model_string(self.gguf_file)
            
            # Download GGUF file if needed
            gguf_path = self._download_gguf_file(repo_id, filename)
            
            logger.info(f"🚀 Loading GGUF model from {gguf_path}")
            logger.info(f"🔧 Configuration: threads={self.n_threads}, gpu_layers={self.n_gpu_layers}, ctx={self.n_ctx}")
            
            # Set memory optimization environment variables
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Use first GPU
            os.environ.setdefault("LLAMA_CUBLAS", "1")  # Enable CUDA
            
            # Initialize llama-cpp-python with GPU optimization
            logger.info("Creating Llama instance...")
            LlamaCppEngine._llm = Llama(
                model_path=str(gguf_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=True,  # Keep verbose to debug GPU issues
                seed=-1,  # Allow random seeds per generation
                n_batch=512,  # Optimize batch size
                use_mlock=False,  # Disable mlock to avoid memory errors
                use_mmap=True,   # Use memory mapping
                main_gpu=0,      # Use first GPU
                offload_kqv=True,      # Offload KV cache to GPU
            )
            
            # Verify the model was created successfully
            if LlamaCppEngine._llm is None:
                raise RuntimeError("Failed to create Llama instance - returned None")
            
            # Test that the model is callable
            if not callable(LlamaCppEngine._llm):
                raise RuntimeError("Llama instance is not callable")
            
            # Store the model info for display
            self.model_display_name = f"{repo_id}/{filename}"
            
            LlamaCppEngine._model_loaded = True
            logger.info(f"Llama.cpp model loaded successfully! Instance type: {type(LlamaCppEngine._llm)}")

        except Exception as e:
            logger.error(f"Failed to load Llama.cpp model: {e}")
            LlamaCppEngine._model_loaded = False
            raise
        finally:
            # Always clear the initializing flag
            LlamaCppEngine._initializing = False

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template to messages using the jinja2 and token information embedded in the gguf file"""
        if not LlamaCppEngine._model_loaded or LlamaCppEngine._llm is None:
            self._initialize_model()
            
        try:
            # Extract chat template from the model metadata
            chat_template = None
            eos_token = None
            bos_token = None
            
            # Get model metadata
            model_metadata = LlamaCppEngine._llm.metadata
            
            # Extract template and tokens from metadata
            if 'tokenizer.chat_template' in model_metadata:
                chat_template = model_metadata['tokenizer.chat_template']
            
            # Extract EOS token with error handling
            if 'tokenizer.ggml.eos_token_id' in model_metadata:
                try:
                    eos_token_id = int(model_metadata['tokenizer.ggml.eos_token_id'])
                    # Get the actual token string
                    eos_token = LlamaCppEngine._llm.detokenize([eos_token_id]).decode('utf-8', errors='ignore')
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to extract EOS token: {e}")
                    eos_token = None
            
            # Extract BOS token with error handling
            if 'tokenizer.ggml.bos_token_id' in model_metadata:
                try:
                    bos_token_id = int(model_metadata['tokenizer.ggml.bos_token_id'])
                    # Get the actual token string  
                    bos_token = LlamaCppEngine._llm.detokenize([bos_token_id]).decode('utf-8', errors='ignore')
                except (ValueError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to extract BOS token: {e}")
                    bos_token = None
            
            # # Fallback tokens if not found
            # if not eos_token:
            #     eos_token = "</s>"
            # if not bos_token:
            #     bos_token = "<s>"
                
            # If we have a chat template, use Jinja2ChatFormatter
            if chat_template:
                from llama_cpp.llama_chat_format import Jinja2ChatFormatter
                
                formatter = Jinja2ChatFormatter(
                    template=chat_template,
                    eos_token=eos_token,
                    bos_token=bos_token,
                    stop_token_ids=[LlamaCppEngine._llm.token_eos()],
                )
                
                # Format the messages
                formatted = formatter(llama=LlamaCppEngine._llm, messages=messages)
                return formatted.prompt
            else:
                # Fallback to simple format if no template found
                logger.warning("No chat template found in model, using simple fallback format")
                formatted_parts = []
                for message in messages:
                    role = message.get('role', 'user')
                    content = message.get('content', '')
                    if role == 'system':
                        formatted_parts.append(f"System: {content}")
                    elif role == 'user':
                        formatted_parts.append(f"User: {content}")
                    elif role == 'assistant':
                        formatted_parts.append(f"Assistant: {content}")
                
                return "\n\n".join(formatted_parts) + "\n\nAssistant:"
                
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error applying chat template: {e}")
            # Ultimate fallback
            formatted_parts = []
            for message in messages:
                role = message.get('role', 'user')
                content = message.get('content', '')
                formatted_parts.append(f"{role.title()}: {content}")
            
            return "\n\n".join(formatted_parts) + "\n\nAssistant:"

    @classmethod
    def get_gguf_cache_info(cls) -> Dict[str, any]:
        """Get information about cached GGUF files"""
        cache_info = {
            'cache_dir': str(cls._gguf_cache_dir),
            'cached_files': [],
            'total_size_gb': 0
        }
        
        if cls._gguf_cache_dir.exists():
            total_size = 0
            for file in cls._gguf_cache_dir.glob('*.gguf'):
                size = file.stat().st_size
                total_size += size
                cache_info['cached_files'].append({
                    'filename': file.name,
                    'size_gb': size / (1024**3),
                    'path': str(file)
                })
            
            cache_info['total_size_gb'] = total_size / (1024**3)
        
        return cache_info

    @classmethod
    def clear_gguf_cache(cls) -> int:
        """Clear GGUF cache and return number of files deleted"""
        if not cls._gguf_cache_dir.exists():
            return 0
        
        count = 0
        for file in cls._gguf_cache_dir.glob('*.gguf'):
            file.unlink()
            count += 1
        
        logger.info(f"Cleared {count} GGUF files from cache")
        return count

    async def _generate_batch_raw(self, prompts: List[str], max_tokens: int = 160,
                                  temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                                  custom_stop_tokens: Optional[List[str]] = None, **sampling_kwargs) -> List[str]:
        """Generate multiple prompts sequentially with thread safety"""
        # Initialize semaphore if not already done (must be in async context)
        if LlamaCppEngine._generation_semaphore is None:
            LlamaCppEngine._generation_semaphore = asyncio.Semaphore(1)
            
        async with LlamaCppEngine._generation_semaphore:  # Ensure no concurrent generation
            try:
                logger.info(f"🚀 Llama.cpp generate_batch called with {len(prompts)} prompts")

                # Validate inputs
                if not prompts:
                    logger.warning("⚠️ Empty prompts list passed to generate_batch")
                    return []

                # Filter out empty prompts
                filtered_prompts = [p for p in prompts if p and isinstance(p, str)]
                if len(filtered_prompts) != len(prompts):
                    logger.warning(f"⚠️ Filtered out {len(prompts) - len(filtered_prompts)} invalid prompts")
                    prompts = filtered_prompts

                if not prompts:
                    logger.warning("⚠️ No valid prompts after filtering")
                    return []

                # Initialize model if needed
                self._initialize_model()

                if not LlamaCppEngine._model_loaded or LlamaCppEngine._llm is None:
                    raise RuntimeError("Llama.cpp model failed to load")

                # Better stop tokens - less aggressive to avoid empty responses
                stop_tokens = ["<|endoftext|>", "User:", "###", "<|user|>", "<|endofcard|>"]
                
                # Allow caller to override stop tokens
                if custom_stop_tokens is not None:
                    stop_tokens = list(custom_stop_tokens)

                # ✅ NEW: Extract enhanced sampling parameters from kwargs
                top_k = sampling_kwargs.get('top_k', 40)  # Default for llama.cpp
                min_p = sampling_kwargs.get('min_p', 0.05)  # Default min_p
                repetition_penalty = sampling_kwargs.get('repetition_penalty', 1.1)
                frequency_penalty = sampling_kwargs.get('frequency_penalty', 0.0)
                presence_penalty = sampling_kwargs.get('presence_penalty', 0.0)
                seed_override = sampling_kwargs.get('seed')
                
                # Build enhanced sampling parameters for llama.cpp
                enhanced_sampling = {
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k if top_k is not None else 40,
                    'min_p': min_p if min_p is not None else 0.05,
                    'repeat_penalty': repetition_penalty,
                    'frequency_penalty': frequency_penalty,
                    'presence_penalty': presence_penalty,
                    'stop': stop_tokens,
                    'seed': seed_override,
                }

                # Generate responses sequentially in a single thread
                results = []
                for i, prompt in enumerate(prompts):
                    try:
                        logger.debug(f"Generating prompt {i+1}/{len(prompts)}")
                        
                        # Generate with enhanced parameters
                        response = await asyncio.to_thread(
                            self._sync_generate_single_enhanced,
                            prompt,
                            enhanced_sampling
                        )
                        results.append(response)
                        
                    except Exception as e:
                        logger.error(f"Error generating response for prompt {i+1}: {e}")
                        results.append("Error: Generation failed")

                return results

            except Exception as e:
                logger.error("Llama.cpp generation failed: %s", e)
                raise

    async def generate_batch(self, prompts: List[str], max_tokens: int = 160,
                             temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                             custom_stop_tokens: Optional[List[str]] = None, **sampling_kwargs) -> List[str]:
        """Generate multiple prompts with thinking support"""
        from .inference_engines import apply_thinking_template, filter_thinking_tokens
        
        # Apply thinking templates to all prompts
        # modified_prompts = [apply_thinking_template(prompt, self.thinking_config) for prompt in prompts]
        
        # Generate responses
        responses = await self._generate_batch_raw(
            prompts=prompts,
            max_tokens=max_tokens,
            character_name=character_name,
            custom_stop_tokens=custom_stop_tokens,
            **sampling_kwargs  # ✅ Pass through sampling parameters
        )
        
        # Filter thinking tokens from all responses
        filtered_responses = [filter_thinking_tokens(response) for response in responses]
        
        return filtered_responses

    def _sync_generate_single_enhanced(self, prompt: str, sampling_params: dict) -> str:
        """Enhanced synchronous generation with full sampling parameter support"""
        try:
            # Ensure model is loaded and available
            if not LlamaCppEngine._model_loaded or LlamaCppEngine._llm is None:
                logger.error("Model not loaded or _llm is None, attempting to reinitialize")
                self._initialize_model()
                
            # Double-check after initialization
            if LlamaCppEngine._llm is None:
                raise RuntimeError("Model failed to initialize - _llm is None")
            
            # Generate with a random seed for each call if not specified
            if sampling_params.get('seed') is None:
                import random
                sampling_params['seed'] = random.randint(0, 2**31 - 1)
            
            # Use the generation lock to prevent concurrent access to _llm
            with LlamaCppEngine._generation_lock:
                # Final safety check
                if LlamaCppEngine._llm is None:
                    raise RuntimeError("_llm is None just before generation call")
                    
                logger.debug(f"Calling _llm generation with enhanced parameters")
                
                # Build parameters dict for llama-cpp-python
                generation_params = {
                    'max_tokens': sampling_params['max_tokens'],
                    'temperature': sampling_params['temperature'],
                    'top_p': sampling_params['top_p'],
                    'top_k': sampling_params['top_k'],
                    'min_p': sampling_params['min_p'],
                    'repeat_penalty': sampling_params['repeat_penalty'],
                    'frequency_penalty': sampling_params['frequency_penalty'],
                    'presence_penalty': sampling_params['presence_penalty'],
                    'stop': sampling_params['stop'],
                    'seed': sampling_params['seed'],
                    'echo': False,  # Don't echo the prompt
                }
                
                # Remove None values to avoid parameter errors
                generation_params = {k: v for k, v in generation_params.items() if v is not None}
                
                response = LlamaCppEngine._llm(prompt, **generation_params)
            
            # Extract text from response - handle both dict and string formats
            text = ""
            if isinstance(response, dict):
                if 'choices' in response and response['choices']:
                    text = response['choices'][0].get('text', '')
                elif 'content' in response:
                    text = response['content']
                else:
                    # Fallback: convert to string
                    text = str(response)
            else:
                text = str(response)
            
            # Clean up the response
            text = text.strip()
            
            # Check for empty or problematic responses
            if not text or text in ['', ' ', '\n', '\\n']:
                logger.warning(f"Empty or whitespace-only response generated for prompt: {prompt[:50]}...")
                text = "I need more context to respond appropriately."
            elif len(text) < 3:  # Very short responses are likely errors
                logger.warning(f"Very short response generated: '{text}'")
                text = "I apologize, but I need more information to provide a proper response."
            
            return text
            
        except Exception as e:
            logger.error(f"Enhanced single generation failed: {e}")
            return f"Error: Generation failed - {str(e)}"

    def _sync_generate_single(self, prompt: str, max_tokens: int, temperature: float, 
                             top_p: float, stop_tokens: List[str]) -> str:
        """Legacy synchronous generation for backward compatibility"""
        # Convert to enhanced format and delegate
        sampling_params = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': 40,  # Default
            'min_p': 0.05,  # Default
            'repeat_penalty': 1.1,  # Default
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'stop': stop_tokens,
            'seed': None,
        }
        return self._sync_generate_single_enhanced(prompt, sampling_params)

    async def _generate_raw(self, prompt: str, max_tokens: int = 160,
                           temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate a single prompt by delegating to _generate_batch_raw for consistency"""
        results = await self._generate_batch_raw(
            prompts=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return results[0] if results else "" 