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


def apply_thinking_template_to_messages(messages: List[Dict[str, str]], thinking_config: Dict) -> tuple[List[Dict[str, str]], str]:
    """
    Apply thinking template modifications to messages before chat templating.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        thinking_config: Configuration dict with 'enabled', 'template', 'qwen_thinking' keys
        
    Returns:
        Tuple of (modified_messages, prefill_text)
        - modified_messages: Messages with thinking modifications applied
        - prefill_text: Any text that should be appended after chat templating
    """
    if not thinking_config.get('enabled', False):
        return messages, ""
    
    template = thinking_config.get('template', '')
    modified_messages = messages.copy()
    prefill_text = ""
    
    if template == "Deepseek Template":
        # For Deepseek: Messages stay the same, but we add prefill after templating
        prefill_text = "<think>\n\n"
    
    elif template == "Qwen3 Template":
        # For Qwen3: Modify the last user message to include /think or /nothink
        qwen_thinking = thinking_config.get('qwen_thinking', True)
        suffix = " /think" if qwen_thinking else " /nothink"
        
        # Find the last user message and append the suffix
        for i in range(len(modified_messages) - 1, -1, -1):
            if modified_messages[i]['role'] == 'user':
                modified_messages[i] = {
                    'role': 'user',
                    'content': modified_messages[i]['content'] + suffix
                }
                break
    
    return modified_messages, prefill_text


def apply_thinking_template(prompt: str, thinking_config: Dict) -> str:
    """
    Legacy function for backward compatibility.
    This is a fallback for when we only have a pre-templated prompt.
    """
    if not thinking_config.get('enabled', False):
        return prompt
    
    template = thinking_config.get('template', '')
    
    if template == "Deepseek Template":
        return prompt + "<think>\n\n"
    elif template == "Qwen3 Template":
        qwen_thinking = thinking_config.get('qwen_thinking', True)
        suffix = " /think" if qwen_thinking else " /nothink"
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

    @abstractmethod
    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template to messages using engine-specific method"""
        pass

    async def generate_with_messages(self, messages: List[Dict[str, str]], max_tokens: int = 160,
                                    temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate text from messages with proper chat templating and thinking support"""
        # Apply thinking template modifications to messages
        modified_messages, prefill_text = apply_thinking_template_to_messages(messages, self.thinking_config)
        
        # Apply chat template
        templated_prompt = self.apply_chat_template(modified_messages)
        
        # Add prefill text if needed (for Deepseek)
        if prefill_text:
            templated_prompt += prefill_text
        
        # Generate response
        response = await self._generate_raw(templated_prompt, max_tokens, temperature, top_p)
        
        # Filter thinking tokens from response
        filtered_response = filter_thinking_tokens(response)
        
        return filtered_response

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

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply chat template to messages - LM Studio handles this automatically"""
        # LM Studio typically handles chat formatting automatically
        # For now, we'll use a simple fallback format
        try:
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
            logger.error(f"Error applying chat template in LM Studio: {e}")
            # Ultimate fallback
            return f"User: {messages[-1].get('content', '')}\n\nAssistant:"

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
