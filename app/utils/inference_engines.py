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


def filter_thinking_tokens(response: str) -> str:
    """
    Filter out thinking tokens from model responses.
    
    Handles common thinking patterns:
    - <think>...</think>
    - <｜thinking｜>...</｜thinking｜>
    - Any content before </think> tag
    
    Args:
        response: Raw model response that may contain thinking tokens
        
    Returns:
        Filtered response with thinking content removed
    """
    if not response:
        return response
    
    # Find the last </think> tag (in case there are multiple thinking blocks)
    think_end_patterns = ['</think>', '</｜thinking｜>']
    
    for pattern in think_end_patterns:
        if pattern in response:
            # Split at the pattern and take everything after
            parts = response.split(pattern)
            if len(parts) > 1:
                # Take the last part (after the final thinking block)
                filtered_response = parts[-1].strip()
                
                # Remove any leading newlines or whitespace
                filtered_response = filtered_response.lstrip('\n').strip()
                
                if filtered_response:
                    logger.debug(f"Filtered thinking tokens. Original length: {len(response)}, Filtered length: {len(filtered_response)}")
                    return filtered_response
    
    # If no thinking tokens found, return original response
    return response


def apply_thinking_template(prompt: str, thinking_config: Dict) -> str:
    """
    Apply thinking template modifications to prompts.
    
    Args:
        prompt: The original prompt (should already be chat-templated)
        thinking_config: Configuration dict with 'enabled', 'template', 'qwen_thinking' keys
        
    Returns:
        Modified prompt with thinking template applied
    """
    if not thinking_config.get('enabled', False):
        return prompt
    
    template = thinking_config.get('template', '')
    
    if template == "Deepseek Template":
        # For Deepseek: Apply chat template first, then append prefill
        # The prompt should already be chat-templated at this point
        return prompt + "<think>\n\n"
    
    elif template == "Qwen3 Template":
        # For Qwen3: Append /think or /nothink to user message
        qwen_thinking = thinking_config.get('qwen_thinking', True)
        suffix = " /think" if qwen_thinking else " /nothink"
        
        # Try to parse and modify the user part of the chat template
        # Look for common patterns like <|user|>content<|endoftext|> or similar
        if '<|user|>' in prompt and '<|endoftext|>' in prompt:
            # DanChat format: find the user section and append suffix before <|endoftext|>
            parts = prompt.split('<|endoftext|>')
            for i, part in enumerate(parts):
                if '<|user|>' in part:
                    # Extract the user content and append suffix
                    user_content = part.split('<|user|>')[-1]
                    parts[i] = part.replace(user_content, user_content + suffix)
                    break
            return '<|endoftext|>'.join(parts)
        elif '<|im_start|>user' in prompt and '<|im_end|>' in prompt:
            # ChatML format: find user section and append suffix before <|im_end|>
            parts = prompt.split('<|im_end|>')
            for i, part in enumerate(parts):
                if '<|im_start|>user' in part:
                    parts[i] = part + suffix
                    break
            return '<|im_end|>'.join(parts)
        else:
            # Fallback: append to the end of the prompt
            return prompt + suffix
    
    return prompt


class InferenceEngine(ABC):
    """Abstract base class for inference engines"""

    def __init__(self):
        self.thinking_config = {'enabled': False}

    def set_thinking_config(self, thinking_config: Dict):
        """Set thinking model configuration"""
        self.thinking_config = thinking_config or {'enabled': False}

    @abstractmethod
    async def _generate_raw(self, prompt: str, max_tokens: int = 160,
                           temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate text from prompt (raw implementation without thinking filtering)"""
        pass

    async def generate(self, prompt: str, max_tokens: int = 160,
                       temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate text from prompt with thinking token support"""
        # Apply thinking template if enabled
        modified_prompt = apply_thinking_template(prompt, self.thinking_config)
        
        # Generate response
        response = await self._generate_raw(modified_prompt, max_tokens, temperature, top_p)
        
        # Filter thinking tokens from response (applied by default to all responses)
        filtered_response = filter_thinking_tokens(response)
        
        return filtered_response

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
        super().__init__()
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

    async def _generate_raw(self, prompt: str, max_tokens: int = 160,
                           temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate using LM Studio (raw implementation)"""
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
                "lmstudio": lambda: LMStudioEngine(),
            }
            
            if VLLMEngine:
                engine_map["vllm"] = lambda: VLLMEngine()
            if LlamaCppEngine:
                engine_map["llamacpp"] = lambda: LlamaCppEngine()

            if preferred_engine.lower() in engine_map:
                preferred = engine_map[preferred_engine.lower()]()
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
