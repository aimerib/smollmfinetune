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
import threading


logger = logging.getLogger(__name__)


class InferenceEngine(ABC):
    """Abstract base class for inference engines"""

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 160,
                       temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate text from prompt"""
        pass

    async def generate_batch(self, prompts: List[str], max_tokens: int = 160,
                            temperature: float = 0.8, top_p: float = 0.9, 
                            character_name: str = None,
                            custom_stop_tokens: Optional[List[str]] = None) -> List[str]:
        """Generate multiple prompts in batch (default implementation uses sequential generation)"""
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, max_tokens, temperature, top_p)
            results.append(result)
        return results

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


class InferenceEngineFactory:
    """Factory to automatically select the best available inference engine"""

    @staticmethod
    def create_engine(preferred_engine: Optional[str] = None) -> InferenceEngine:
        """
        Create the best available inference engine.

        Args:
            preferred_engine: Force a specific engine ("lmstudio", "vllm", "llamacpp")

        Returns:
            The best available inference engine
        """
        # Import engines here to avoid circular imports
        try:
            from .vllm_engine import VLLMEngine
        except ImportError:
            logger.warning("VLLMEngine not available")
            VLLMEngine = None
            
        try:
            from .llama_cpp_engine import LlamaCppEngine
        except ImportError:
            logger.warning("LlamaCppEngine not available")
            LlamaCppEngine = None

        # Define engine priority (best to fallback)
        engines = []
        if VLLMEngine:
            engines.append(VLLMEngine())
        if LlamaCppEngine:
            engines.append(LlamaCppEngine())
        engines.append(LMStudioEngine())

        # If user specified a preference, try that first
        if preferred_engine:
            logger.info(f"User requested preferred engine: {preferred_engine}")
            engine_map = {
                "lmstudio": LMStudioEngine(),
            }
            
            if VLLMEngine:
                engine_map["vllm"] = VLLMEngine()
            if LlamaCppEngine:
                engine_map["llamacpp"] = LlamaCppEngine()

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

        # This should never happen since LMStudioEngine should always work as fallback
        raise RuntimeError("No inference engines available!")


# Convenience function for easy usage
def get_inference_engine(preferred: Optional[str] = None) -> InferenceEngine:
    """Get the best available inference engine"""
    return InferenceEngineFactory.create_engine(preferred)
