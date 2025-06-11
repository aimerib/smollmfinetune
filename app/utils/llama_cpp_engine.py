"""
llama-cpp-python inference engine for GGUF models.
Handles GGUF model downloading, caching, and inference.
"""

import os
import asyncio
import threading
import secrets
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
        # Prevent re-initialization to avoid Streamlit infinite loops
        if hasattr(self, '_initialized'):
            current_gguf = getattr(self, 'gguf_file', None)
            if gguf_file is None or gguf_file == current_gguf:
                logger.debug(f"LlamaCppEngine already initialized with {current_gguf}, skipping")
                return

        # Configuration
        self.gguf_file = gguf_file
        self.tokenizer_name = tokenizer_name
        self.n_ctx = n_ctx
        self.n_threads = 8 # n_threads or os.cpu_count()
        self.n_gpu_layers = n_gpu_layers
        
        # Force reload if model changed
        if hasattr(self, '_initialized') and gguf_file != getattr(self, 'gguf_file', None):
            logger.info(f"🔄 Model changed from {getattr(self, 'gguf_file', None)} to {gguf_file}, forcing reload")
            LlamaCppEngine._model_loaded = False
            LlamaCppEngine._llm = None

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
        
        if not filename.endswith('.gguf'):
            raise ValueError(f"Invalid GGUF filename: {filename}. Must end with .gguf")
        
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
            
            # Store the model info for display
            self.model_display_name = f"{repo_id}/{filename}"
            
            LlamaCppEngine._model_loaded = True
            logger.info(f"Llama.cpp model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load Llama.cpp model: {e}")
            LlamaCppEngine._model_loaded = False
            raise
        finally:
            # Always clear the initializing flag
            LlamaCppEngine._initializing = False

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

    async def generate_batch(self, prompts: List[str], max_tokens: int = 160,
                             temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                             custom_stop_tokens: Optional[List[str]] = None) -> List[str]:
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
                stop_tokens = ["<|endoftext|>", "\n\nUser:", "###", "<|user|>", "<|endofcard|>"]
                
                # Allow caller to override stop tokens
                if custom_stop_tokens is not None:
                    stop_tokens = list(custom_stop_tokens)

                # Generate responses sequentially in a single thread
                results = []
                for i, prompt in enumerate(prompts):
                    try:
                        logger.debug(f"Generating prompt {i+1}/{len(prompts)}")
                        
                        # Generate with random seed for each prompt
                        response = await asyncio.to_thread(
                            self._sync_generate_single,
                            prompt,
                            max_tokens,
                            temperature,
                            top_p,
                            stop_tokens
                        )
                        results.append(response)
                        
                    except Exception as e:
                        logger.error(f"Error generating response for prompt {i+1}: {e}")
                        results.append("Error: Generation failed")

                return results

            except Exception as e:
                logger.error("Llama.cpp generation failed: %s", e)
                raise

    def _sync_generate_single(self, prompt: str, max_tokens: int, temperature: float, 
                             top_p: float, stop_tokens: List[str]) -> str:
        """Synchronous generation for a single prompt with improved error handling"""
        try:
            # Generate with a random seed for each call
            import random
            seed = random.randint(0, 2**31 - 1)
            
            response = LlamaCppEngine._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_tokens,
                echo=False,  # Don't echo the prompt
                seed=seed,   # Random seed for each generation
                repeat_penalty=1.1,  # Reduce repetition
            )
            
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
            logger.error(f"Single generation failed: {e}")
            return f"Error: Generation failed - {str(e)}"

    async def generate(self, prompt: str, max_tokens: int = 160,
                       temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                       custom_stop_tokens: Optional[List[str]] = None) -> str:
        """Generate a single prompt by delegating to generate_batch for consistency"""
        results = await self.generate_batch(
            prompts=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            character_name=character_name,
            custom_stop_tokens=custom_stop_tokens,
        )
        return results[0] if results else "" 