"""
vLLM inference engine for high-performance GPU deployment.
Handles regular HuggingFace models only (no GGUF support).
"""

import os
import asyncio
import threading
import secrets
from typing import Optional, List, Dict
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
    _generation_lock = None  # Add class-level async lock
    _batch_lock = None  # Separate lock for batch operations

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

            # Initialize the locks if not already done
            if VLLMEngine._generation_lock is None:
                VLLMEngine._generation_lock = threading.Lock()
            if VLLMEngine._batch_lock is None:
                VLLMEngine._batch_lock = asyncio.Lock()

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

            # Get configuration from environment or use optimized defaults for A100 80GB
            gpu_memory_util = float(
                os.getenv('VLLM_GPU_MEMORY_UTILIZATION', '0.95'))  # Reduced to leave more KV cache space
            max_model_len = int(os.getenv('MAX_MODEL_LEN', '4096'))
            
            # KV cache optimization for high-memory GPUs
            max_num_seqs = int(os.getenv('VLLM_MAX_NUM_SEQS', '100'))  # Increase concurrent sequences
            max_num_batched_tokens = int(os.getenv('VLLM_MAX_BATCHED_TOKENS', '8192'))  # Optimize batch token limit

            # Regular HuggingFace model loading only
            logger.info(f"Loading vLLM model {self.model_name} (this may take 1-2 minutes)...")
            logger.info(f"🔧 vLLM config: gpu_mem={gpu_memory_util}, max_seqs={max_num_seqs}, max_tokens={max_num_batched_tokens}")
            
            VLLMEngine._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_util,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                enforce_eager=False,
                trust_remote_code=True,
                # Enable block management optimizations for large VRAM
                enable_prefix_caching=True,
                disable_custom_all_reduce=True,
                kv_cache_dtype="fp8_e5m2",
                calculate_kv_scales=True,
            )
            
            self.model_display_name = self.model_name

            VLLMEngine._model_loaded = True
            logger.info(f"vLLM model loaded successfully!")

        except Exception as e:
            error_msg = f"Failed to load vLLM model: {e}"
            logger.error(error_msg)
            logger.exception("Full vLLM initialization traceback:")
            
            # ✅ FIX: Check for specific error types to provide better guidance
            error_str = str(e).lower()
            if "out of memory" in error_str or "cuda" in error_str:
                logger.error("💥 CUDA/Memory error detected - try reducing model size or clearing memory")
            elif "assertion" in error_str:
                logger.error("💥 vLLM assertion error - this may be due to incompatible settings")
            elif "fp8" in error_str:
                logger.error("💥 FP8 error detected - KV cache dtype incompatibility")
            
            VLLMEngine._model_loaded = False
            raise RuntimeError(f"vLLM initialization failed: {error_msg}")
        finally:
            # Always clear the initializing flag
            VLLMEngine._initializing = False

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template to messages using the transformers tokenizer"""
        if not VLLMEngine._model_loaded or VLLMEngine._llm is None:
            self._initialize_model()
            
        try:
            # Get the tokenizer from vLLM
            tokenizer = VLLMEngine._llm.get_tokenizer()
            
            # Use transformers apply_chat_template if available
            if hasattr(tokenizer, 'apply_chat_template'):
                formatted = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            else:
                # Fallback to simple format if apply_chat_template not available
                logger.warning("Tokenizer doesn't have apply_chat_template, using simple fallback format")
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
            logger.error(f"Error applying chat template in vLLM: {e}")
            # Ultimate fallback
            formatted_parts = []
            for message in messages:
                role = message.get('role', 'user')
                content = message.get('content', '')
                formatted_parts.append(f"{role.title()}: {content}")
            
            return "\n\n".join(formatted_parts) + "\n\nAssistant:"

    async def _generate_batch_raw(self, prompts: List[str], max_tokens: int = 160,
                                  temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                                  custom_stop_tokens: Optional[List[str]] = None) -> List[str]:
        """Generate multiple prompts in a single batch (much more efficient)"""
        # Ensure we have the batch lock to prevent concurrent batch issues
        if VLLMEngine._batch_lock is None:
            VLLMEngine._batch_lock = asyncio.Lock()
        
        async with VLLMEngine._batch_lock:
            try:
                logger.info(
                    f"🚀 vLLM generate_batch called with {len(prompts)} prompts")

                # Validate inputs to prevent tensor shape mismatches
                if not prompts:
                    logger.warning(
                        "⚠️ Empty prompts list passed to generate_batch")
                    return []

                # Filter out empty prompts that can cause tensor dimension errors
                filtered_prompts = [p for p in prompts if p and isinstance(p, str) and len(p.strip()) > 0]
                if len(filtered_prompts) != len(prompts):
                    logger.warning(
                        f"⚠️ Filtered out {len(prompts) - len(filtered_prompts)} invalid prompts")

                if not filtered_prompts:
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

                # Run synchronous vLLM generate with proper error handling
                try:
                    request_outputs = await asyncio.to_thread(
                        _sync_generate, filtered_prompts, sampling_params
                    )
                except Exception as vllm_error:
                    # If vLLM batch fails, try smaller batches or sequential processing
                    logger.warning(f"⚠️ vLLM batch generation failed: {vllm_error}")
                    
                    # Try splitting into smaller batches
                    if len(filtered_prompts) > 10:
                        logger.info("🔄 Attempting smaller batch sizes...")
                        batch_size = max(1, len(filtered_prompts) // 4)
                        all_results = []
                        
                        for i in range(0, len(filtered_prompts), batch_size):
                            small_batch = filtered_prompts[i:i + batch_size]
                            try:
                                small_outputs = await asyncio.to_thread(
                                    _sync_generate, small_batch, sampling_params
                                )
                                # Extract text from outputs
                                for output in small_outputs:
                                    if output.outputs:
                                        all_results.append(output.outputs[0].text)
                                    else:
                                        all_results.append("Error: No output generated")
                                        
                                # Small delay between batches to prevent overload
                                await asyncio.sleep(0.1)
                                
                            except Exception as small_batch_error:
                                logger.error(f"Small batch also failed: {small_batch_error}")
                                # Fill with error responses for this batch
                                all_results.extend(["Error: Generation failed"] * len(small_batch))
                        
                        return self._process_results(all_results, filtered_prompts)
                    else:
                        # For very small batches, fall back to sequential processing
                        logger.info("🔄 Falling back to sequential processing...")
                        return await self._sequential_fallback(filtered_prompts, sampling_params)

                # Process successful batch results
                results: List[str] = []
                for request_output in request_outputs:
                    if request_output.outputs:
                        generated_text = request_output.outputs[0].text
                        results.append(generated_text)
                    else:
                        logger.warning("No outputs for prompt: %s", request_output.prompt[:50])
                        results.append("Error: No output generated")

                return self._process_results(results, filtered_prompts)

            except Exception as e:
                logger.error("vLLM generation failed: %s", e)
                # Return error responses to maintain expected output length
                return ["Error: Generation failed"] * len(prompts)

    def _process_results(self, results: List[str], original_prompts: List[str]) -> List[str]:
        """Process and clean up generation results"""
        processed_results = []
        
        for idx, txt in enumerate(results):
            txt = txt.strip()
            if not txt:
                prompt_preview = original_prompts[idx][:50] if idx < len(original_prompts) else "unknown"
                logger.warning(f"Empty result for prompt '{prompt_preview}...'")
                txt = "Error: Empty response"
            processed_results.append(txt)

        return processed_results

    async def _sequential_fallback(self, prompts: List[str], sampling_params) -> List[str]:
        """Fallback to sequential generation when batch processing fails"""
        results = []
        
        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"Sequential generation {i+1}/{len(prompts)}")
                request_outputs = await asyncio.to_thread(
                    _sync_generate, [prompt], sampling_params
                )
                
                if request_outputs and request_outputs[0].outputs:
                    results.append(request_outputs[0].outputs[0].text)
                else:
                    results.append("Error: No output generated")
                    
                # Small delay to prevent overwhelming the engine
                if i < len(prompts) - 1:
                    await asyncio.sleep(0.05)
                    
            except Exception as seq_error:
                logger.error(f"Sequential generation failed for prompt {i+1}: {seq_error}")
                results.append("Error: Sequential generation failed")
        
        return results

    async def generate_batch(self, prompts: List[str], max_tokens: int = 160,
                             temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                             custom_stop_tokens: Optional[List[str]] = None) -> List[str]:
        """Generate multiple prompts in a single batch with thinking support"""
        from .inference_engines import apply_thinking_template, filter_thinking_tokens
        
        # Apply thinking templates to all prompts
        # modified_prompts = [apply_thinking_template(prompt, self.thinking_config) for prompt in prompts]
        
        # Generate responses
        responses = await self._generate_batch_raw(
            prompts=prompts,
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