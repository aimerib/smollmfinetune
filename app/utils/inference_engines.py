"""
Inference engine adapters for different deployment environments.
Automatically selects the best engine based on available resources.
"""

import os
import asyncio
from abc import ABC, abstractmethod
import secrets
from typing import Optional, List
import logging


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
            model = lms.llm()
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
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: Optional[str] = None, tensor_parallel_size: int = 1):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
        
        # Allow override via environment variable
        self.model_name = (
            model_name or 
            os.getenv('VLLM_MODEL', 'PocketDoc/Dans-PersonalityEngine-V1.3.0-24b')
        )
        
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
        self._initialized = True
    
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
        if VLLMEngine._model_loaded:
            return
        
        try:
            from vllm import LLM, SamplingParams
            
            # Get configuration from environment or use defaults
            gpu_memory_util = float(os.getenv('VLLM_GPU_MEMORY_UTILIZATION', '0.90'))  # Increased
            max_model_len = int(os.getenv('VLLM_MAX_MODEL_LEN', '4096'))  # Reduced for memory
            
            logger.info(f"Loading vLLM model {self.model_name} (this may take 1-2 minutes)...")
            
            VLLMEngine._llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_util,
                max_model_len=max_model_len,
                enforce_eager=False,
                trust_remote_code=True,
            )
            
            VLLMEngine._model_loaded = True
            logger.info(f"vLLM model {self.model_name} loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            VLLMEngine._model_loaded = False
            raise
    
    async def generate(self, prompt: str, max_tokens: int = 160,
                      temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                      custom_stop_tokens: Optional[List[str]] = None) -> str:
        """Generate using vLLM with smart batching"""
        try:
            # Initialize model if needed (only happens once)
            self._initialize_model()
            
            if not VLLMEngine._model_loaded or VLLMEngine._llm is None:
                raise RuntimeError("vLLM model failed to load")
            
            from vllm import SamplingParams
            
            # Base stop tokens (robust defaults for normal dialogue)
            stop_tokens = ["\n\n", "<|endoftext|>", "User:", "###", "<|endofcard|>", "<|user|>"]
            
            # Allow caller to override stop tokens for special generation modes
            if custom_stop_tokens is not None:
                # Use a *copy* so that downstream modifications do not affect caller list
                stop_tokens = list(custom_stop_tokens)
            
            # Add character name as stop token to prevent speaking for other characters
            if character_name:
                stop_tokens.extend([f"{character_name}:", f"\n{character_name}:"])

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
                min_tokens=1,            # Ensure non-empty responses
                seed=seed,
            )
            
            # Generate using the singleton model instance
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None, 
                lambda: VLLMEngine._llm.generate([prompt], sampling_params)
            )
            
            if outputs and outputs[0].outputs:
                return outputs[0].outputs[0].text.strip()
            else:
                raise RuntimeError("vLLM returned empty output")
                
        except Exception as e:
            raise RuntimeError(f"vLLM generation failed: {str(e)}")
    
    async def generate_batch(self, prompts: List[str], max_tokens: int = 160,
                           temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                           custom_stop_tokens: Optional[List[str]] = None) -> List[str]:
        """Generate multiple prompts in a single batch (much more efficient)"""
        try:
            logger.info(f"🚀 vLLM generate_batch called with {len(prompts)} prompts")
            
            # Initialize model if needed (only happens once)
            self._initialize_model()
            
            if not VLLMEngine._model_loaded or VLLMEngine._llm is None:
                raise RuntimeError("vLLM model failed to load")
            
            from vllm import SamplingParams
            
            # Base stop tokens list
            stop_tokens = ["\n\n", "<|endoftext|>", "User:", "###", "<|endofcard|>", "<|user|>"]
            
            # Allow caller to override stop tokens for special generation modes
            if custom_stop_tokens is not None:
                stop_tokens = list(custom_stop_tokens)
            
            # Add character name as stop token to prevent speaking for other characters
            if character_name:
                stop_tokens.extend([f"{character_name}:", f"\n{character_name}:"])
            
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
                min_tokens=1,            # Ensure non-empty responses
            )
            
            logger.info(f"🎯 Sending {len(prompts)} prompts to vLLM model (character: {character_name or 'none'})")
            
            # Generate all prompts in a single batch
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None, 
                lambda: VLLMEngine._llm.generate(prompts, sampling_params)
            )
            
            logger.info(f"📤 vLLM returned {len(outputs)} outputs")
            
            # Extract results
            results = []
            for i, output in enumerate(outputs):
                if output.outputs:
                    text = output.outputs[0].text.strip()
                    # logger.debug(f"vLLM output {i}: '{text[:100]}{'...' if len(text) > 100 else ''}' (length: {len(text)})")
                    results.append(text)
                else:
                    logger.warning(f"❌ vLLM output {i}: No outputs generated")
                    results.append("")
            
            logger.info(f"✅ vLLM batch generation complete: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"💥 vLLM batch generation failed: {str(e)}")
            raise RuntimeError(f"vLLM batch generation failed: {str(e)}")



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
                logger.info(f"Testing availability of preferred engine: {preferred.name}")
                if preferred.is_available():
                    logger.info(f"✅ Using preferred engine: {preferred.name}")
                    return preferred
                else:
                    logger.warning(f"❌ Preferred engine {preferred.name} not available, falling back...")
            else:
                logger.warning(f"❌ Unknown preferred engine: {preferred_engine}, falling back...")
        
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