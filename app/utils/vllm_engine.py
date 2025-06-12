"""
vLLM inference engine for high-performance GPU deployment.
Handles regular HuggingFace models only (no GGUF support).
"""

import os
import asyncio
import threading
import secrets
import time
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, TypedDict, Literal, Union, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor
import logging
import torch
from pathlib import Path

from .inference_engines import InferenceEngine

logger = logging.getLogger(__name__)


# ===== Configuration Management =====

@dataclass
class VLLMConfig:
    """Configuration for vLLM engine"""
    model_name: str = "PocketDoc/Dans-PersonalityEngine-V1.3.0-24b"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.95
    max_model_len: int = 4096
    max_num_seqs: int = 100
    max_num_batched_tokens: int = 8192
    max_batch_size: int = 1000
    batch_timeout_ms: int = 50
    max_concurrent_requests: int = 1000
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
    
    # Cache settings
    cache_dir: Optional[str] = None
    force_offline: bool = False
    
    @classmethod
    def from_env(cls) -> 'VLLMConfig':
        """Load configuration from environment variables"""
        return cls(
            model_name=os.getenv('VLLM_MODEL', cls.model_name),
            tensor_parallel_size=int(os.getenv('VLLM_TENSOR_PARALLEL_SIZE', '1')),
            gpu_memory_utilization=float(os.getenv('VLLM_GPU_MEMORY_UTILIZATION', '0.95')),
            max_model_len=int(os.getenv('MAX_MODEL_LEN', '4096')),
            max_num_seqs=int(os.getenv('VLLM_MAX_NUM_SEQS', '100')),
            max_num_batched_tokens=int(os.getenv('VLLM_MAX_BATCHED_TOKENS', '8192')),
            max_batch_size=int(os.getenv('VLLM_MAX_BATCH_SIZE', '1000')),
            batch_timeout_ms=int(os.getenv('VLLM_BATCH_TIMEOUT_MS', '50')),
            max_concurrent_requests=int(os.getenv('MAX_CONCURRENT_REQUESTS', '1000')),
            cache_dir=os.getenv('HF_HOME'),
            force_offline=os.getenv('HF_HUB_OFFLINE', '0') == '1',
        )


class SamplingConfig(TypedDict, total=False):
    """Type hints for sampling parameters"""
    max_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int]
    min_p: Optional[float]
    repetition_penalty: float
    frequency_penalty: float
    presence_penalty: float
    seed: Optional[int]
    stop: Optional[List[str]]


# ===== Custom Exceptions =====

class VLLMInitializationError(Exception):
    """Raised when vLLM fails to initialize"""
    def __init__(self, original_error: Exception, config: VLLMConfig):
        self.original_error = original_error
        self.config = config
        super().__init__(self._create_message(original_error))
        
    def _create_message(self, error: Exception) -> str:
        error_str = str(error).lower()
        if "out of memory" in error_str or "cuda" in error_str:
            return (
                f"GPU out of memory. Try:\n"
                f"1. Reducing gpu_memory_utilization (current: {self.config.gpu_memory_utilization})\n"
                f"2. Using a smaller model\n"
                f"3. Reducing max_model_len (current: {self.config.max_model_len})\n"
                f"4. Clearing GPU memory with torch.cuda.empty_cache()\n"
                f"Original error: {error}"
            )
        elif "assertion" in error_str:
            return (
                f"vLLM assertion error - likely due to incompatible settings:\n"
                f"- Check tensor_parallel_size matches GPU count\n"
                f"- Verify model supports the configuration\n"
                f"Original error: {error}"
            )
        elif "fp8" in error_str:
            return (
                f"FP8 quantization error detected:\n"
                f"- Your GPU may not support FP8 operations\n"
                f"- Try disabling kv_cache_dtype='fp8_e5m2'\n"
                f"Original error: {error}"
            )
        else:
            return f"vLLM initialization failed: {error}"


# ===== Utility Classes =====

class BatchQueue:
    """Manages batching of requests for efficient processing"""
    def __init__(self, max_batch_size: int, timeout_ms: int):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
        
    async def add_request(self, request: Dict) -> asyncio.Future:
        """Add a request to the queue and return a future for the result"""
        future = asyncio.Future()
        await self.queue.put((request, future))
        return future
        
    async def process_batches(self, process_func: Callable):
        """Continuously process batches from the queue"""
        self._processing = True
        while self._processing:
            batch = []
            futures = []
            deadline = time.time() + (self.timeout_ms / 1000.0)
            
            # Collect items for batch
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    timeout = max(0, deadline - time.time())
                    request, future = await asyncio.wait_for(
                        self.queue.get(), timeout=timeout
                    )
                    batch.append(request)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break
                    
            if batch:
                try:
                    # Process the batch
                    results = await process_func(batch)
                    # Set results for all futures
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)
                except Exception as e:
                    # Set exception for all futures
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
                            
    def stop(self):
        """Stop processing batches"""
        self._processing = False


class RetryHandler:
    """Handles retry logic with exponential backoff"""
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        
    async def retry_with_backoff(self, func: Callable, *args, **kwargs):
        """Retry a function with exponential backoff and jitter"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == self.max_attempts - 1:
                    raise
                    
                # Exponential backoff with jitter
                delay = (2 ** attempt * self.base_delay) + random.uniform(0, 1)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
                
        raise last_exception


# ===== Model Manager =====

class VLLMModelManager:
    """Manages vLLM model loading and lifecycle"""
    def __init__(self):
        self._llm = None
        self._loaded = False
        self._loading_lock = threading.Lock()
        self._config: Optional[VLLMConfig] = None
        
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded and self._llm is not None
        
    def get_model(self):
        """Get the loaded model"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._llm
        
    def load_model(self, config: VLLMConfig) -> None:
        """Load the vLLM model with proper error handling"""
        with self._loading_lock:
            # Double-check pattern
            if self.is_loaded() and self._config == config:
                logger.info("Model already loaded with same configuration")
                return
                
            # Clear previous model if config changed
            if self._llm is not None and self._config != config:
                logger.info("Configuration changed, reloading model...")
                self.cleanup()
                
            self._config = config
            self._load_model_internal(config)
            
    def _load_model_internal(self, config: VLLMConfig) -> None:
        """Internal method to load the model"""
        try:
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            from vllm import LLM
            
            # Setup environment for HuggingFace
            env_backup = self._setup_hf_environment(config)
            
            try:
                # Validate model name for security
                self._validate_model_name(config.model_name)
                
                # Check if model is cached
                if not self._is_model_cached(config.model_name):
                    logger.info(f"Model not cached, downloading: {config.model_name}")
                    # Temporarily allow online access for download
                    os.environ.pop('HF_HUB_OFFLINE', None)
                    os.environ.pop('TRANSFORMERS_OFFLINE', None)
                
                logger.info(f"Loading vLLM model: {config.model_name}")
                logger.info(
                    f"Config: gpu_mem={config.gpu_memory_utilization}, "
                    f"max_seqs={config.max_num_seqs}, "
                    f"max_tokens={config.max_num_batched_tokens}"
                )
                
                # Determine KV cache dtype based on GPU capability
                kv_cache_dtype = self._get_optimal_kv_cache_dtype()
                
                self._llm = LLM(
                    model=config.model_name,
                    tensor_parallel_size=config.tensor_parallel_size,
                    gpu_memory_utilization=config.gpu_memory_utilization,
                    max_model_len=config.max_model_len,
                    max_num_seqs=config.max_num_seqs,
                    max_num_batched_tokens=config.max_num_batched_tokens,
                    enforce_eager=False,
                    trust_remote_code=True,
                    enable_prefix_caching=True,
                    disable_custom_all_reduce=True,
                    kv_cache_dtype=kv_cache_dtype,
                    calculate_kv_scales=(kv_cache_dtype == "fp8_e5m2"),
                )
                
                self._loaded = True
                logger.info("vLLM model loaded successfully!")
                
            finally:
                # Restore environment
                self._restore_environment(env_backup)
                
        except Exception as e:
            self._loaded = False
            self._llm = None
            raise VLLMInitializationError(e, config)
            
    def _setup_hf_environment(self, config: VLLMConfig) -> Dict[str, Optional[str]]:
        """Setup HuggingFace environment and return backup"""
        backup = {
            'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE'),
            'TRANSFORMERS_OFFLINE': os.environ.get('TRANSFORMERS_OFFLINE'),
            'HF_HOME': os.environ.get('HF_HOME'),
            'HF_HUB_CACHE': os.environ.get('HF_HUB_CACHE'),
        }
        
        if config.force_offline:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
        if config.cache_dir:
            os.environ['HF_HOME'] = config.cache_dir
            os.environ['HF_HUB_CACHE'] = config.cache_dir
        elif not os.environ.get('HF_HOME'):
            cache_dir = self._get_default_cache_dir()
            os.environ['HF_HOME'] = cache_dir
            os.environ['HF_HUB_CACHE'] = cache_dir
            logger.info(f"Using cache directory: {cache_dir}")
            
        return backup
        
    def _restore_environment(self, backup: Dict[str, Optional[str]]) -> None:
        """Restore environment variables from backup"""
        for key, value in backup.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)
                
    def _get_default_cache_dir(self) -> str:
        """Get default cache directory"""
        if os.path.exists("/workspace"):
            return "/workspace/.cache/vllm_hf"
        return os.path.expanduser("~/.cache/huggingface")
        
    def _validate_model_name(self, model_name: str) -> None:
        """Validate model name for security"""
        if ".." in model_name or model_name.startswith("/"):
            raise ValueError(f"Invalid model name: {model_name}")
            
    def _is_model_cached(self, model_name: str) -> bool:
        """Check if model is available in cache"""
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(model_name, local_files_only=True)
            logger.info(f"Found cached model: {model_name}")
            return True
        except Exception:
            return False
            
    def _get_optimal_kv_cache_dtype(self) -> str:
        """Determine optimal KV cache dtype based on GPU"""
        if not torch.cuda.is_available():
            return "auto"
            
        # Check GPU compute capability for FP8 support
        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        
        # FP8 requires compute capability 8.9+ (Ada Lovelace or newer)
        if capability[0] >= 8 and capability[1] >= 9:
            return "fp8_e5m2"
        else:
            return "auto"
            
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self._llm is not None:
            # vLLM doesn't have explicit cleanup, but clear references
            self._llm = None
            self._loaded = False
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Model manager cleaned up")


# ===== Main Engine Class =====

class VLLMEngine(InferenceEngine):
    """vLLM engine for high-performance cloud deployment with batching"""
    
    # Thread-safe singleton
    _instance = None
    _instance_lock = threading.Lock()
    
    # Shared resources
    _model_manager: Optional[VLLMModelManager] = None
    _batch_queue: Optional[BatchQueue] = None
    _request_semaphore: Optional[asyncio.Semaphore] = None
    _executor: Optional[ThreadPoolExecutor] = None
    _retry_handler: Optional[RetryHandler] = None
    
    def __new__(cls, *args, **kwargs):
        """Thread-safe singleton pattern"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
        
    def __init__(self, config: Optional[VLLMConfig] = None):
        """Initialize the engine with configuration"""
        # Prevent re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        super().__init__()
        
        # Load configuration
        self.config = config or VLLMConfig.from_env()
        
        # Initialize components
        self._initialize_components()
        
        # Mark as initialized
        self._initialized = True
        self._available = None
        
    def _initialize_components(self) -> None:
        """Initialize engine components"""
        if VLLMEngine._model_manager is None:
            VLLMEngine._model_manager = VLLMModelManager()
            
        if VLLMEngine._batch_queue is None:
            VLLMEngine._batch_queue = BatchQueue(
                max_batch_size=self.config.max_batch_size,
                timeout_ms=self.config.batch_timeout_ms
            )
            
        if VLLMEngine._request_semaphore is None:
            VLLMEngine._request_semaphore = asyncio.Semaphore(
                self.config.max_concurrent_requests
            )
            
        if VLLMEngine._executor is None:
            VLLMEngine._executor = ThreadPoolExecutor(max_workers=4)
            
        if VLLMEngine._retry_handler is None:
            VLLMEngine._retry_handler = RetryHandler(
                max_attempts=self.config.retry_max_attempts,
                base_delay=self.config.retry_base_delay
            )
            
    @property
    def name(self) -> str:
        return "vLLM"
        
    def is_available(self) -> bool:
        """Check if vLLM is available"""
        if self._available is not None:
            return self._available
            
        try:
            from vllm import LLM, SamplingParams
            self._available = torch.cuda.is_available()
            
            if self._available:
                logger.info("vLLM detected with GPU support")
            else:
                logger.info("vLLM available but no GPU detected")
                
        except ImportError:
            self._available = False
            logger.debug("vLLM not installed")
        except Exception as e:
            self._available = False
            logger.debug(f"vLLM not available: {e}")
            
        return self._available
        
    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded before use"""
        if not self._model_manager.is_loaded():
            self._model_manager.load_model(self.config)
            
    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template to messages"""
        self._ensure_model_loaded()
        
        try:
            # Setup offline mode for tokenizer operations
            env_backup = {'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE')}
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            try:
                tokenizer = self._model_manager.get_model().get_tokenizer()
                
                if hasattr(tokenizer, 'apply_chat_template'):
                    return tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # Fallback format
                    return self._simple_chat_format(messages)
                    
            finally:
                # Restore environment
                if env_backup['HF_HUB_OFFLINE'] is not None:
                    os.environ['HF_HUB_OFFLINE'] = env_backup['HF_HUB_OFFLINE']
                else:
                    os.environ.pop('HF_HUB_OFFLINE', None)
                    
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            return self._simple_chat_format(messages)
            
    def _simple_chat_format(self, messages: List[Dict[str, str]]) -> str:
        """Simple chat format fallback"""
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
        
    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 160,
        temperature: float = 0.8,
        top_p: float = 0.9,
        character_name: Optional[str] = None,
        custom_stop_tokens: Optional[List[str]] = None,
        **sampling_kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts"""
        # Use semaphore to limit concurrent requests
        async with self._request_semaphore:
            # Validate and filter prompts
            valid_prompts = self._validate_prompts(prompts)
            if not valid_prompts:
                return []
                
            # Prepare sampling configuration
            sampling_config = self._prepare_sampling_config(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                custom_stop_tokens=custom_stop_tokens,
                **sampling_kwargs
            )
            
            # Use retry handler for resilience
            try:
                results = await self._retry_handler.retry_with_backoff(
                    self._generate_batch_internal,
                    valid_prompts,
                    sampling_config
                )
                
                # Apply thinking token filtering if needed
                from .inference_engines import filter_thinking_tokens
                return [filter_thinking_tokens(r) for r in results]
                
            except Exception as e:
                logger.error(f"Batch generation failed after retries: {e}")
                return ["Error: Generation failed"] * len(prompts)
                
    async def _generate_batch_internal(
        self,
        prompts: List[str],
        sampling_config: SamplingConfig
    ) -> List[str]:
        """Internal batch generation with optimized processing"""
        self._ensure_model_loaded()
        
        from vllm import SamplingParams
        
        # Create sampling params
        sampling_params = SamplingParams(**sampling_config)
        
        # Execute generation
        request_outputs = await asyncio.to_thread(
            self._sync_generate_wrapper,
            prompts,
            sampling_params
        )
        
        # Process results
        results = []
        for output in request_outputs:
            if output.outputs:
                results.append(output.outputs[0].text.strip())
            else:
                results.append("Error: No output generated")
                
        return results
        
    def _sync_generate_wrapper(self, prompts: List[str], sampling_params):
        """Wrapper for synchronous vLLM generation"""
        return self._model_manager.get_model().generate(prompts, sampling_params)
        
    def _validate_prompts(self, prompts: List[str]) -> List[str]:
        """Validate and filter prompts"""
        valid_prompts = []
        
        for prompt in prompts:
            if prompt and isinstance(prompt, str) and prompt.strip():
                valid_prompts.append(prompt)
            else:
                logger.warning(f"Invalid prompt filtered: {prompt}")
                
        return valid_prompts
        
    def _prepare_sampling_config(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        custom_stop_tokens: Optional[List[str]] = None,
        **kwargs
    ) -> SamplingConfig:
        """Prepare sampling configuration"""
        # Default stop tokens
        stop_tokens = custom_stop_tokens or [
            "<|endoftext|>", "User:", "###", 
            "<|endofcard|>", "<|user|>"
        ]
        
        # Base configuration
        config: SamplingConfig = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop': stop_tokens,
            'seed': kwargs.get('seed', secrets.randbits(64)),
        }
        
        # Add optional parameters
        if 'top_k' in kwargs and kwargs['top_k'] is not None:
            config['top_k'] = int(kwargs['top_k'])
            
        if 'min_p' in kwargs and kwargs['min_p'] is not None:
            config['min_p'] = float(kwargs['min_p'])
            
        if 'repetition_penalty' in kwargs and kwargs['repetition_penalty'] != 1.0:
            config['repetition_penalty'] = float(kwargs['repetition_penalty'])
            
        if 'frequency_penalty' in kwargs and kwargs['frequency_penalty'] != 0.0:
            config['frequency_penalty'] = float(kwargs['frequency_penalty'])
            
        if 'presence_penalty' in kwargs and kwargs['presence_penalty'] != 0.0:
            config['presence_penalty'] = float(kwargs['presence_penalty'])
            
        return config
        
    async def _generate_raw(
        self,
        prompt: str,
        max_tokens: int = 160,
        temperature: float = 0.8,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate a single response"""
        results = await self.generate_batch(
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        return results[0] if results else ""
        
    @classmethod
    def cleanup(cls) -> None:
        """Cleanup all resources"""
        logger.info("Cleaning up vLLM engine...")
        
        # Stop batch queue
        if cls._batch_queue:
            cls._batch_queue.stop()
            
        # Cleanup model manager
        if cls._model_manager:
            cls._model_manager.cleanup()
            
        # Shutdown executor
        if cls._executor:
            cls._executor.shutdown(wait=True)
            
        # Clear references
        cls._batch_queue = None
        cls._model_manager = None
        cls._executor = None
        cls._request_semaphore = None
        cls._retry_handler = None
        
        logger.info("vLLM engine cleanup complete")
        
    def __del__(self):
        """Destructor to ensure cleanup"""
        # Only cleanup if this is the singleton instance being destroyed
        if self.__class__._instance is self:
            self.__class__.cleanup()
            self.__class__._instance = None