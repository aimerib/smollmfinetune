"""
vLLM inference engine for high-performance GPU deployment.
Handles regular HuggingFace models only (no GGUF support).
"""

import os
import asyncio
import threading
import secrets
from typing import Optional, List
import logging
import torch

from .inference_engines import InferenceEngine

logger = logging.getLogger(__name__)


def _sync_generate(prompts, sampling_params):
    """Synchronous wrapper for vLLM generate call"""
    # No lock needed here – vLLM's scheduler supports concurrent generate calls.
    return VLLMEngine._llm.generate(prompts, sampling_params)


class VLLMEngine(InferenceEngine):
    """vLLM engine for high-performance cloud deployment with batching"""

    # Class-level singleton to ensure model is loaded only once
    _instance = None
    _llm = None
    _model_loaded = False
    _initializing = False  # Add flag to prevent concurrent initialization
    _generation_lock = None  # Add class-level lock

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: Optional[str] = None, 
                 tensor_parallel_size: int = 1):
        # Initialize parent class first
        super().__init__()
        
        # ✅ FIX: Prevent re-initialization to avoid Streamlit infinite loops
        # Check if already initialized with same parameters
        if hasattr(self, '_initialized'):
            current_model = getattr(self, 'model_name', None)
            if model_name is None or model_name == current_model:
                return  # Already initialized with same or default model
        
        # Allow override via environment variable
        target_model = (
            model_name or
            os.getenv('VLLM_MODEL', 'PocketDoc/Dans-PersonalityEngine-V1.3.0-24b')
        )
        
        # Only proceed if model actually changed or first initialization
        if not hasattr(self, '_initialized') or target_model != getattr(self, 'model_name', None):
            self.model_name = target_model
            
            # Force reload if model changed
            if hasattr(self, '_initialized') and target_model != getattr(self, 'model_name', None):
                VLLMEngine._model_loaded = False
                VLLMEngine._llm = None

            self.tensor_parallel_size = 1

            self._sampling_params = None
            self._available = None
            self._batch_queue = []
            self._batch_size = 8  # Process in batches of 8
            self._max_batch_size = 1000  # vLLM can handle very large batches
            
            # Mark as initialized to prevent future re-initialization
            self._initialized = True

            # Initialize the lock if not already done
            if VLLMEngine._generation_lock is None:
                VLLMEngine._generation_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "vLLM"

    def is_available(self) -> bool:
        """Check if vLLM is available"""
        if self._available is not None:
            return self._available

        try:
            from vllm import LLM, SamplingParams
            import torch

            # Check if we have GPU (vLLM works best with GPU)
            gpu_available = torch.cuda.is_available()
            self._available = gpu_available  # Prefer GPU for vLLM

            if self._available:
                logger.info("vLLM detected with GPU support")
            else:
                logger.debug("vLLM available but no GPU detected")

        except ImportError:
            self._available = False
            logger.debug("vLLM not installed")
        except Exception as e:
            self._available = False
            logger.debug(f"vLLM not available: {e}")

        return self._available

    def _initialize_model(self):
        """Initialize vLLM model once (singleton pattern)"""
        # ✅ CRITICAL FIX: Add thread-safe initialization check
        with VLLMEngine._generation_lock:
            # Double-check pattern: check again inside the lock
            if VLLMEngine._model_loaded and VLLMEngine._llm is not None:
                return
            
            # Prevent concurrent initialization attempts
            if hasattr(VLLMEngine, '_initializing') and VLLMEngine._initializing:
                logger.info("Model initialization already in progress, waiting...")
                # Wait for other thread to finish initialization
                import time
                max_wait = 300  # 5 minutes max wait
                waited = 0
                while VLLMEngine._initializing and waited < max_wait:
                    time.sleep(1)
                    waited += 1
                
                if VLLMEngine._model_loaded and VLLMEngine._llm is not None:
                    return
                else:
                    raise RuntimeError("Model initialization by other thread failed")
            
            # Mark as initializing
            VLLMEngine._initializing = True

        try:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            from vllm import LLM, SamplingParams

            # Get configuration from environment or use defaults
            gpu_memory_util = float(
                os.getenv('VLLM_GPU_MEMORY_UTILIZATION', '0.90'))  # Increased
            # Reduced for memory
            max_model_len = int(os.getenv('VLLM_MAX_MODEL_LEN', '4096'))

            # Regular HuggingFace model loading only
            logger.info(f"Loading vLLM model {self.model_name} (this may take 1-2 minutes)...")
            
            VLLMEngine._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_util,
                max_model_len=max_model_len,
                enforce_eager=False,
                trust_remote_code=True,
            )
            
            self.model_display_name = self.model_name

            VLLMEngine._model_loaded = True
            logger.info(f"vLLM model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            VLLMEngine._model_loaded = False
            raise
        finally:
            # Always clear the initializing flag
            VLLMEngine._initializing = False

    async def _generate_batch_raw(self, prompts: List[str], max_tokens: int = 160,
                                  temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                                  custom_stop_tokens: Optional[List[str]] = None) -> List[str]:
        """Generate multiple prompts in a single batch (much more efficient)"""
        try:
            logger.info(
                f"🚀 vLLM generate_batch called with {len(prompts)} prompts")

            # Validate inputs to prevent tensor shape mismatches
            if not prompts:
                logger.warning(
                    "⚠️ Empty prompts list passed to generate_batch")
                return []

            # Filter out empty prompts that can cause tensor dimension errors
            filtered_prompts = [p for p in prompts if p and isinstance(p, str)]
            if len(filtered_prompts) != len(prompts):
                logger.warning(
                    f"⚠️ Filtered out {len(prompts) - len(filtered_prompts)} invalid prompts")
                prompts = filtered_prompts

            if not prompts:
                logger.warning("⚠️ No valid prompts after filtering")
                return []

            # Initialize model if needed (only happens once)
            self._initialize_model()

            if not VLLMEngine._model_loaded or VLLMEngine._llm is None:
                raise RuntimeError("vLLM model failed to load")

            from vllm import SamplingParams
            import asyncio
            import secrets

            # Base stop tokens list
            stop_tokens = ["\n\n", "<|endoftext|>",
                           "User:", "###", "<|endofcard|>", "<|user|>"]

            # Allow caller to override stop tokens for special generation modes
            if custom_stop_tokens is not None:
                stop_tokens = list(custom_stop_tokens)

            # Use a cryptographically strong random seed for each batch
            seed = secrets.randbits(64)

            # Sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_tokens,
                seed=seed,
            )

            # Run synchronous vLLM generate in a worker thread
            request_outputs = await asyncio.to_thread(
                _sync_generate, prompts, sampling_params
            )

            results: List[str] = []
            for request_output in request_outputs:
                if request_output.outputs:
                    generated_text = request_output.outputs[0].text
                    results.append(generated_text)
                else:
                    logger.warning("No outputs for prompt: %s", request_output.prompt)
                    results.append("Error: No output generated")

            # Strip whitespace / handle empties
            for idx, txt in enumerate(results):
                txt = txt.strip()
                if not txt:
                    logger.warning("Empty result for prompt '%s'", prompts[idx])
                    txt = "Error: Empty response"
                results[idx] = txt

            return results

        except Exception as e:
            logger.error("vLLM generation failed: %s", e)
            raise

    async def generate_batch(self, prompts: List[str], max_tokens: int = 160,
                             temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                             custom_stop_tokens: Optional[List[str]] = None) -> List[str]:
        """Generate multiple prompts in a single batch with thinking support"""
        from .inference_engines import apply_thinking_template, filter_thinking_tokens
        
        # Apply thinking templates to all prompts
        modified_prompts = [apply_thinking_template(prompt, self.thinking_config) for prompt in prompts]
        
        # Generate responses
        responses = await self._generate_batch_raw(
            prompts=modified_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            character_name=character_name,
            custom_stop_tokens=custom_stop_tokens
        )
        
        # Filter thinking tokens from all responses
        filtered_responses = [filter_thinking_tokens(response) for response in responses]
        
        return filtered_responses

    async def _generate_raw(self, prompt: str, max_tokens: int = 160,
                           temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate a single prompt by delegating to `_generate_batch_raw` for consistency."""
        results = await self._generate_batch_raw(
            prompts=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        # `_generate_batch_raw` returns one result per input prompt
        return results[0] if results else "" 