"""
Inference engine adapters for different deployment environments.
Automatically selects the best engine based on available resources.
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
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
                enforce_eager=True,
                trust_remote_code=True,
            )
            
            VLLMEngine._model_loaded = True
            logger.info(f"vLLM model {self.model_name} loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            VLLMEngine._model_loaded = False
            raise
    
    async def generate(self, prompt: str, max_tokens: int = 160,
                      temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate using vLLM with smart batching"""
        try:
            # Initialize model if needed (only happens once)
            self._initialize_model()
            
            if not VLLMEngine._model_loaded or VLLMEngine._llm is None:
                raise RuntimeError("vLLM model failed to load")
            
            from vllm import SamplingParams
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=["\n\n", "<|endoftext|>", "User:", "###", "<|endofcard|>"]  # Better stop tokens
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
                           temperature: float = 0.8, top_p: float = 0.9) -> List[str]:
        """Generate multiple prompts in a single batch (much more efficient)"""
        try:
            # Initialize model if needed (only happens once)
            self._initialize_model()
            
            if not VLLMEngine._model_loaded or VLLMEngine._llm is None:
                raise RuntimeError("vLLM model failed to load")
            
            from vllm import SamplingParams
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=["\n\n", "<|endoftext|>", "User:", "###", "<|endofcard|>"]
            )
            
            # Generate all prompts in a single batch
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None, 
                lambda: VLLMEngine._llm.generate(prompts, sampling_params)
            )
            
            # Extract results
            results = []
            for output in outputs:
                if output.outputs:
                    results.append(output.outputs[0].text.strip())
                else:
                    results.append("")
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"vLLM batch generation failed: {str(e)}")


class LlamaCppEngine(InferenceEngine):
    """llama-cpp-python engine as fallback option"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or self._find_model_path()
        self._llm = None
        self._available = None
    
    @property
    def name(self) -> str:
        return "llama-cpp-python"
    
    def _find_model_path(self) -> Optional[str]:
        """Try to find a suitable model file"""
        # Common model locations
        possible_paths = [
            "/models/model.gguf",
            "/app/models/model.gguf",
            "./models/model.gguf",
            # Add more paths as needed
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def is_available(self) -> bool:
        """Check if llama-cpp-python is available"""
        if self._available is not None:
            return self._available
        
        try:
            from llama_cpp import Llama
            
            # Check if we have a model file
            if self.model_path and os.path.exists(self.model_path):
                self._available = True
                logger.info(f"llama-cpp-python available with model: {self.model_path}")
            else:
                self._available = False
                logger.debug(f"llama-cpp-python model not found at: {self.model_path}")
                
        except ImportError:
            self._available = False
            logger.debug("llama-cpp-python not installed")
        except Exception as e:
            self._available = False
            logger.debug(f"llama-cpp-python not available: {e}")
        
        return self._available
    
    def _initialize_model(self):
        """Lazy initialization of llama-cpp model"""
        if self._llm is None:
            from llama_cpp import Llama
            
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Context length
                n_threads=None,  # Use all available threads
                verbose=False
            )
            logger.info(f"llama-cpp-python model loaded: {self.model_path}")
    
    async def generate(self, prompt: str, max_tokens: int = 160,
                      temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate using llama-cpp-python"""
        try:
            # Initialize model if needed
            self._initialize_model()
            
            # Generate in thread pool since llama-cpp is synchronous
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self._llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    echo=False,
                    stop=["\n\n", "<|endoftext|>"]
                )
            )
            
            if output and 'choices' in output and output['choices']:
                return output['choices'][0]['text'].strip()
            else:
                raise RuntimeError("llama-cpp-python returned empty output")
                
        except Exception as e:
            raise RuntimeError(f"llama-cpp-python generation failed: {str(e)}")


class TransformersEngine(InferenceEngine):
    """Transformers engine as universal fallback"""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._available = None
    
    @property
    def name(self) -> str:
        return "Transformers"
    
    def is_available(self) -> bool:
        """Transformers should always be available as it's a core dependency"""
        if self._available is not None:
            return self._available
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._available = True
            logger.info("Transformers engine available as fallback")
        except ImportError:
            self._available = False
            logger.error("Transformers not available - this should not happen!")
        
        return self._available
    
    def _initialize_model(self):
        """Lazy initialization of transformers model"""
        if self._model is None:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            logger.info(f"Transformers model loaded: {self.model_name}")
    
    async def generate(self, prompt: str, max_tokens: int = 160,
                      temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate using transformers"""
        try:
            import torch
            
            # Initialize model if needed
            self._initialize_model()
            
            # Tokenize
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            
            # Generate in thread pool
            loop = asyncio.get_event_loop()
            
            def _generate():
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self._tokenizer.eos_token_id,
                        eos_token_id=self._tokenizer.eos_token_id,
                    )
                
                # Decode only new tokens
                response = self._tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[-1]:], 
                    skip_special_tokens=True
                )
                return response.strip()
            
            result = await loop.run_in_executor(None, _generate)
            return result
            
        except Exception as e:
            raise RuntimeError(f"Transformers generation failed: {str(e)}")


class InferenceEngineFactory:
    """Factory to automatically select the best available inference engine"""
    
    @staticmethod
    def create_engine(preferred_engine: Optional[str] = None) -> InferenceEngine:
        """
        Create the best available inference engine.
        
        Args:
            preferred_engine: Force a specific engine ("lmstudio", "vllm", "llamacpp", "transformers")
        
        Returns:
            The best available inference engine
        """
        # Define engine priority (best to fallback)
        engines = [
            VLLMEngine(),       # Best for cloud GPU deployment
            LMStudioEngine(),   # Best for local Mac development
            LlamaCppEngine(),   # Good CPU fallback
            TransformersEngine() # Universal fallback
        ]
        
        # If user specified a preference, try that first
        if preferred_engine:
            engine_map = {
                "lmstudio": LMStudioEngine(),
                "vllm": VLLMEngine(), 
                "llamacpp": LlamaCppEngine(),
                "transformers": TransformersEngine()
            }
            
            if preferred_engine.lower() in engine_map:
                preferred = engine_map[preferred_engine.lower()]
                if preferred.is_available():
                    logger.info(f"Using preferred engine: {preferred.name}")
                    return preferred
                else:
                    logger.warning(f"Preferred engine {preferred.name} not available, falling back...")
        
        # Auto-select best available engine
        for engine in engines:
            if engine.is_available():
                logger.info(f"Auto-selected inference engine: {engine.name}")
                return engine
        
        # This should never happen since TransformersEngine should always work
        raise RuntimeError("No inference engines available!")


# Convenience function for easy usage
def get_inference_engine(preferred: Optional[str] = None) -> InferenceEngine:
    """Get the best available inference engine"""
    return InferenceEngineFactory.create_engine(preferred) 