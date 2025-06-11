"""
Inference engine adapters for different deployment environments.
Automatically selects the best engine based on available resources.
"""

import os
import asyncio
from abc import ABC, abstractmethod
import secrets
from typing import Optional, List, Tuple, Dict
import logging
import torch  # For memory management
from pathlib import Path
import hashlib
import requests
from tqdm import tqdm
import json


logger = logging.getLogger(__name__)


class InferenceEngine(ABC):
    """Abstract base class for inference engines"""

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 160,
                       temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine is available in current environment"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name for logging/display"""
        pass


class LMStudioEngine(InferenceEngine):
    """LM Studio engine for local Mac development"""

    def __init__(self):
        self._available = None

    @property
    def name(self) -> str:
        return "LM Studio"

    def is_available(self) -> bool:
        """Check if LM Studio is available"""
        if self._available is not None:
            return self._available

        try:
            import lmstudio as lms
            # Try to connect to LM Studio
            _ = lms.llm()
            self._available = True
            logger.info("LM Studio detected and available")
        except Exception as e:
            self._available = False
            logger.debug(f"LM Studio not available: {e}")

        return self._available

    async def generate(self, prompt: str, max_tokens: int = 160,
                       temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate using LM Studio"""
        try:
            import lmstudio as lms

            model = lms.llm()
            response = model.respond(prompt, config={
                "temperature": temperature,
                "topPSampling": top_p,
                "maxTokens": max_tokens,
            })

            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            raise RuntimeError(f"LM Studio generation failed: {str(e)}")


class VLLMEngine(InferenceEngine):
    """vLLM engine for high-performance cloud deployment with batching"""

    # Class-level singleton to ensure model is loaded only once
    _instance = None
    _llm = None
    _model_loaded = False
    _generation_lock = None  # Add class-level lock
    _gguf_cache_dir = Path.home() / ".cache" / "vllm_gguf"  # GGUF cache directory

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_name: Optional[str] = None, 
                 gguf_file: Optional[str] = None,
                 tokenizer_name: Optional[str] = None,
                 tensor_parallel_size: int = 1):
        # Only initialize once
        if hasattr(self, '_initialized') and model_name == self.model_name:
            return

        # Allow override via environment variable
        self.model_name = (
            model_name or
            os.getenv('VLLM_MODEL', 'PocketDoc/Dans-PersonalityEngine-V1.3.0-24b')
        )
        
        # GGUF configuration
        self.gguf_file = gguf_file
        self.tokenizer_name = tokenizer_name
        
        # Force reload if model changed
        if hasattr(self, '_initialized') and model_name != getattr(self, 'model_name', None):
            VLLMEngine._model_loaded = False
            VLLMEngine._llm = None

        # Auto-detect tensor parallel size based on GPU count if not specified
        if tensor_parallel_size == 1:
            try:
                import torch
                gpu_count = torch.cuda.device_count()
                # Use multiple GPUs for 24B model if available
                if gpu_count >= 2 and "24b" in self.model_name.lower():
                    self.tensor_parallel_size = min(gpu_count, 4)  # Max 4 GPUs
                else:
                    self.tensor_parallel_size = 1
            except:
                self.tensor_parallel_size = 1
        else:
            self.tensor_parallel_size = tensor_parallel_size

        self._sampling_params = None
        self._available = None
        self._batch_queue = []
        self._batch_size = 8  # Process in batches of 8
        self._max_batch_size = 1000  # vLLM can handle very large batches
        self._initialized = True

        # Initialize the lock if not already done
        if VLLMEngine._generation_lock is None:
            VLLMEngine._generation_lock = asyncio.Lock()

        # Create GGUF cache directory
        VLLMEngine._gguf_cache_dir.mkdir(parents=True, exist_ok=True)

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
        """Initialize vLLM model once (singleton pattern)"""
        if VLLMEngine._model_loaded:
            return

        try:
            from vllm import LLM, SamplingParams

            # Get configuration from environment or use defaults
            gpu_memory_util = float(
                os.getenv('VLLM_GPU_MEMORY_UTILIZATION', '0.90'))  # Increased
            # Reduced for memory
            max_model_len = int(os.getenv('VLLM_MAX_MODEL_LEN', '4096'))

            # Check if this is a GGUF model
            if self.gguf_file:
                # Parse GGUF specification
                repo_id, filename, tokenizer_repo = self._parse_gguf_model_string(self.gguf_file)
                
                # Download GGUF file if needed
                gguf_path = self._download_gguf_file(repo_id, filename)
                
                # Use specified tokenizer or default to base model
                if self.tokenizer_name:
                    tokenizer = self.tokenizer_name
                elif tokenizer_repo:
                    tokenizer = tokenizer_repo
                else:
                    # Try to infer base model from repo name
                    # e.g., TheBloke/Llama-2-70B-GGUF -> meta-llama/Llama-2-70b-hf
                    logger.warning(f"No tokenizer specified for GGUF, using repo ID: {repo_id}")
                    tokenizer = repo_id
                
                logger.info(f"Loading GGUF model from {gguf_path} with tokenizer {tokenizer}")
                
                VLLMEngine._llm = LLM(
                    model=str(gguf_path),
                    tokenizer=tokenizer,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_util,
                    max_model_len=max_model_len,
                    enforce_eager=False,
                    trust_remote_code=True,
                )
                
                # Store the model info for display
                self.model_display_name = f"{repo_id}/{filename}"
                
            else:
                # Regular HuggingFace model loading
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

    async def generate(self, prompt: str, max_tokens: int = 160,
                       temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                       custom_stop_tokens: Optional[List[str]] = None) -> str:
        """Generate a single prompt by delegating to `generate_batch` for consistency.

        Delegating avoids code duplication, keeps validation/stop-token logic in
        one place, and ensures any future fixes in `generate_batch` automatically
        apply here.
        """
        results = await self.generate_batch(
            prompts=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            character_name=character_name,
            custom_stop_tokens=custom_stop_tokens,
        )
        # `generate_batch` returns one result per input prompt
        return results[0] if results else ""

    async def generate_batch(self, prompts: List[str], max_tokens: int = 160,
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

            # Base stop tokens list
            stop_tokens = ["\n\n", "<|endoftext|>",
                           "User:", "###", "<|endofcard|>", "<|user|>"]

            # Allow caller to override stop tokens for special generation modes
            if custom_stop_tokens is not None:
                stop_tokens = list(custom_stop_tokens)

            # Add character name as stop token to prevent speaking for other characters
            if character_name:
                stop_tokens.extend(
                    [f"{character_name}:", f"\n{character_name}:"])

            # Use a really random seed every time using the gpu seed
            seed = secrets.randbits(64)

            # Create sampling parameters (optimized for character generation)
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop_tokens,
                # Character-optimized settings
                repetition_penalty=1.05,  # Slight penalty to reduce repetition
                frequency_penalty=0.1,    # Light penalty for frequent tokens
                presence_penalty=0.05,    # Encourage topic diversity
                top_k=-1,                # Disabled (use top_p)
                min_p=0.0,               # Disabled (use top_p)
                logprobs=None,           # Disabled for performance
                min_tokens=1,            # Ensure non-empty responses,
                seed=seed,
            )

            logger.info(
                f"🎯 Sending {len(prompts)} prompts to vLLM model (character: {character_name or 'none'})")

            try:
                # Generate all prompts in a single batch
                loop = asyncio.get_event_loop()
                outputs = await loop.run_in_executor(
                    None,
                    lambda: VLLMEngine._llm.generate(prompts, sampling_params)
                )
            except RuntimeError as e:
                # Handle tensor shape/index errors and other CUDA issues
                error_str = str(e)
                if any(err in error_str for err in ["index", "out of bounds", "size", "dimension", "shape"]):
                    logger.error(
                        f"⚠️ vLLM tensor shape/indexing error: {error_str}")
                    logger.info("🔄 Falling back to sequential generation")

                    # Fallback to sequential generation
                    outputs = []
                    for i, prompt in enumerate(prompts):
                        try:
                            # Process each prompt individually
                            single_output = await loop.run_in_executor(
                                None,
                                lambda: VLLMEngine._llm.generate(
                                    [prompt], sampling_params)
                            )
                            outputs.extend(single_output)
                            # Clear cache after each item
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception as inner_e:
                            logger.warning(
                                f"❌ Individual generation failed for prompt {i}: {inner_e}")
                            # Create an empty output placeholder to maintain alignment with input
                            from vllm.outputs import RequestOutput
                            outputs.append(RequestOutput(
                                request_id=f"error-{i}", prompt=prompt, prompt_token_ids=[], outputs=[]))
                else:
                    # Re-raise other errors
                    raise

            logger.info(f"📤 vLLM returned {len(outputs)} outputs")

            # Extract results
            results = []
            for i, output in enumerate(outputs):
                if output and hasattr(output, 'outputs') and output.outputs:
                    text = output.outputs[0].text.strip()
                    # logger.debug(f"vLLM output {i}: '{text[:100]}{'...' if len(text) > 100 else ''}' (length: {len(text)})")
                    results.append(text)
                else:
                    logger.warning(f"❌ vLLM output {i}: No outputs generated")
                    results.append("")

            # Ensure we return exactly the right number of results
            if len(results) < len(prompts):
                logger.warning(
                    f"⚠️ Results count mismatch: got {len(results)}, expected {len(prompts)}")
                # Pad with empty strings to match original prompt count
                results.extend([""] * (len(prompts) - len(results)))
            elif len(results) > len(prompts):
                logger.warning(
                    f"⚠️ Too many results: got {len(results)}, expected {len(prompts)}")
                # Truncate to match original prompt count
                results = results[:len(prompts)]

            # Force CUDA cache cleanup after batch generation to prevent memory fragmentation
            if torch.cuda.is_available():
                try:
                    # Clear cache
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.debug(f"Failed to clear CUDA cache: {e}")

            logger.info(
                f"✅ vLLM batch generation complete: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"💥 vLLM batch generation failed: {str(e)}")
            # Last resort fallback - return empty responses for all prompts
            logger.warning(
                "⚠️ Returning empty responses due to generation failure")
            return [""] * len(prompts)


class InferenceEngineFactory:
    """Factory to automatically select the best available inference engine"""

    @staticmethod
    def create_engine(preferred_engine: Optional[str] = None) -> InferenceEngine:
        """
        Create the best available inference engine.

        Args:
            preferred_engine: Force a specific engine ("lmstudio", "vllm")

        Returns:
            The best available inference engine
        """
        # Define engine priority (best to fallback)
        engines = [
            VLLMEngine(),       # Best for cloud GPU deployment
            LMStudioEngine(),   # Best for local Mac development
        ]

        # If user specified a preference, try that first
        if preferred_engine:
            logger.info(f"User requested preferred engine: {preferred_engine}")
            engine_map = {
                "lmstudio": LMStudioEngine(),
                "vllm": VLLMEngine(),
            }

            if preferred_engine.lower() in engine_map:
                preferred = engine_map[preferred_engine.lower()]
                logger.info(
                    f"Testing availability of preferred engine: {preferred.name}")
                if preferred.is_available():
                    logger.info(f"✅ Using preferred engine: {preferred.name}")
                    return preferred
                else:
                    logger.warning(
                        f"❌ Preferred engine {preferred.name} not available, falling back...")
            else:
                logger.warning(
                    f"❌ Unknown preferred engine: {preferred_engine}, falling back...")

        # Auto-select best available engine
        logger.info("🔍 Auto-detecting best available engine...")
        for engine in engines:
            logger.info(f"Testing engine: {engine.name}")
            if engine.is_available():
                logger.info(f"✅ Auto-selected inference engine: {engine.name}")
                return engine
            else:
                logger.info(f"❌ Engine {engine.name} not available")

        # This should never happen since TransformersEngine should always work
        raise RuntimeError("No inference engines available!")


# Convenience function for easy usage
def get_inference_engine(preferred: Optional[str] = None) -> InferenceEngine:
    """Get the best available inference engine"""
    return InferenceEngineFactory.create_engine(preferred)
