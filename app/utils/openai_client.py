"""
Simple OpenAI API client wrapper for text generation.
Supports both text completion and chat completion endpoints.
"""

import os
import asyncio
import logging
import traceback
from typing import List, Dict, Optional, Union, Any
import aiohttp
import json

logger = logging.getLogger(__name__)


class CompletionResponse:
    """Response object for completion requests"""
    def __init__(self, text: str, raw_response: Dict[str, Any]):
        self.text = text
        self.raw_response = raw_response
        self.usage = raw_response.get('usage', {})
        self.model = raw_response.get('model', '')
        self.finish_reason = None
        
        # Extract finish reason from response
        if 'choices' in raw_response and raw_response['choices']:
            choice = raw_response['choices'][0]
            self.finish_reason = choice.get('finish_reason')


class OpenAIClient:
    """Simple OpenAI API client for text generation"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 default_model: Optional[str] = None,
                 auto_detect_model: bool = True):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (defaults to OpenAI, but can be changed for compatible endpoints)
            default_model: Default model to use for requests
            auto_detect_model: If True, automatically detect the model from /v1/models endpoint on first use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', "dummy")
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Remove trailing slash from base_url
        self.base_url = self.base_url.rstrip('/')
        self.default_model = default_model
        self.auto_detect_model = auto_detect_model
        self._model_detected = False  # Track if we've already detected the model
        
        self._session = None
        self._session_loop = None  # Track which loop the session belongs to
    
    async def _ensure_model_detected(self):
        """
        Ensure model is detected if auto_detect_model is enabled
        This is called lazily on first API usage
        """
        if self.auto_detect_model and not self.default_model and not self._model_detected:
            try:
                detected_model = await self._detect_model()
                if detected_model:
                    self.default_model = detected_model
                    logger.info(f"Auto-detected model: {detected_model}")
                self._model_detected = True
            except Exception as e:
                logger.warning(f"Failed to auto-detect model: {e}")
                self._model_detected = True  # Don't keep trying on subsequent calls
    
    async def _detect_model(self) -> Optional[str]:
        """
        Detect the available model from the /v1/models endpoint
        
        Returns:
            Model ID string if found, None otherwise
        """
        try:
            # Use a fresh session for model detection to avoid any issues
            async with aiohttp.ClientSession(
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.get(f'{self.base_url}/models') as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and data['data'] and len(data['data']) > 0:
                            model_id = data['data'][0]['id']
                            return model_id
                    else:
                        logger.warning(f"Failed to fetch models: HTTP {response.status}")
        except Exception as e:
            logger.warning(f"Error detecting model: {e}")
        
        return None
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from the /v1/models endpoint
        
        Returns:
            List of model information dictionaries
        """
        try:
            # Use a fresh session for getting models to avoid any issues
            async with aiohttp.ClientSession(
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.get(f'{self.base_url}/models') as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('data', [])
                    else:
                        error_text = await response.text()
                        raise Exception(f"Models API error {response.status}: {error_text}")
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            raise
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper lifecycle management"""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
            
        session_invalid = False
        
        # Check various conditions that would make the session invalid
        if self._session is None:
            session_invalid = True
        elif self._session.closed:
            session_invalid = True
        elif self._session_loop != current_loop:
            # Session belongs to different loop, need to recreate
            session_invalid = True
        else:
            # Additional checks for session health
            try:
                if hasattr(self._session, '_loop') and self._session._loop.is_closed():
                    session_invalid = True
            except Exception:
                session_invalid = True
        
        if session_invalid:
            # Close existing session if it exists
            await self._cleanup_session()
            
            # Create new session
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
            self._session_loop = current_loop
            
        return self._session
    
    async def _cleanup_session(self):
        """Clean up the current session properly"""
        if self._session and not self._session.closed:
            try:
                await self._session.close()
                # Give a moment for cleanup
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.debug(f"Error closing session: {e}")
        
        self._session = None
        self._session_loop = None
    
    async def close(self):
        """Close the aiohttp session"""
        await self._cleanup_session()
    
    async def reset_session(self):
        """Reset the aiohttp session (useful when switching event loops)"""
        await self._cleanup_session()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def complete(self, 
                      prompt: str,
                      model: Optional[str] = None,
                      max_tokens: int = 150,
                      temperature: float = 0.8,
                      top_p: float = 0.9,
                      stop: Optional[List[str]] = None,
                      return_full_response: bool = False,
                      **kwargs) -> Union[str, CompletionResponse]:
        """
        Generate text completion from a prompt
        
        Args:
            prompt: Input prompt string
            model: Model to use (defaults to default_model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            stop: List of stop sequences
            return_full_response: If True, return CompletionResponse object, else just text
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text string or CompletionResponse object
        """
        # Ensure model is detected if needed
        await self._ensure_model_detected()
        
        payload = {
            'model': model or self.default_model,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            **kwargs
        }
        
        if stop:
            payload['stop'] = stop
        
        # Try with session, if it fails, try with fresh session
        for attempt in range(2):
            try:
                if attempt == 0:
                    # Try with managed session first
                    session = await self._get_session()
                    async with session.post(f'{self.base_url}/completions', json=payload) as response:
                        return await self._handle_response(response, return_full_response)
                else:
                    # If managed session fails, try with fresh session
                    await self._cleanup_session()
                    async with aiohttp.ClientSession(
                        headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as fresh_session:
                        async with fresh_session.post(f'{self.base_url}/completions', json=payload) as response:
                            return await self._handle_response(response, return_full_response)
                            
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Session request failed, retrying with fresh session: {e}")
                    continue
                else:
                    logger.error(f"Text completion failed: {e}")
                    traceback.print_exc()
                    raise
    
    async def chat_complete(self,
                           messages: List[Dict[str, str]],
                           model: Optional[str] = None,
                           max_tokens: int = 150,
                           temperature: float = 0.8,
                           top_p: float = 0.9,
                           stop: Optional[List[str]] = None,
                           return_full_response: bool = False,
                           **kwargs) -> Union[str, CompletionResponse]:
        """
        Generate chat completion from messages
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model to use (defaults to default_model)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter (0-1)
            stop: List of stop sequences
            return_full_response: If True, return CompletionResponse object, else just text
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text string or CompletionResponse object
        """
        # Ensure model is detected if needed
        await self._ensure_model_detected()
        
        payload = {
            'model': model or self.default_model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            **kwargs
        }
        
        if stop:
            payload['stop'] = stop
        
        # Try with session, if it fails, try with fresh session
        for attempt in range(2):
            try:
                if attempt == 0:
                    # Try with managed session first
                    session = await self._get_session()
                    async with session.post(f'{self.base_url}/chat/completions', json=payload) as response:
                        return await self._handle_response(response, return_full_response, is_chat=True)
                else:
                    # If managed session fails, try with fresh session
                    await self._cleanup_session()
                    async with aiohttp.ClientSession(
                        headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                        timeout=aiohttp.ClientTimeout(total=120)
                    ) as fresh_session:
                        async with fresh_session.post(f'{self.base_url}/chat/completions', json=payload) as response:
                            return await self._handle_response(response, return_full_response, is_chat=True)
                            
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Session request failed, retrying with fresh session: {e}")
                    continue
                else:
                    logger.error(f"Chat completion failed: {e}")
                    raise
    
    async def _handle_response(self, response, return_full_response: bool, is_chat: bool = False):
        """Handle API response and extract text"""
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"OpenAI API error {response.status}: {error_text}")
        
        data = await response.json()
        
        # Extract the generated text
        if 'choices' not in data or not data['choices']:
            raise Exception("No choices returned from API")
        
        if is_chat:
            message = data['choices'][0].get('message', {})
            text = message.get('content', '').strip()
        else:
            text = data['choices'][0].get('text', '').strip()
        
        if return_full_response:
            return CompletionResponse(text, data)
        else:
            return text
    
    async def generate(self,
                      prompt: Union[str, List[Dict[str, str]]],
                      model: Optional[str] = None,
                      max_tokens: int = 150,
                      temperature: float = 0.8,
                      top_p: float = 0.9,
                      stop: Optional[List[str]] = None,
                      return_full_response: bool = False,
                      **kwargs) -> Union[str, CompletionResponse]:
        """
        Universal generate method that automatically chooses completion or chat completion
        
        Args:
            prompt: String for text completion, or list of messages for chat completion
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: List of stop sequences
            return_full_response: If True, return CompletionResponse object, else just text
            **kwargs: Additional parameters
            
        Returns:
            Generated text string or CompletionResponse object
        """
        if isinstance(prompt, str):
            return await self.complete(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                return_full_response=return_full_response,
                **kwargs
            )
        elif isinstance(prompt, list):
            return await self.chat_complete(
                messages=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                return_full_response=return_full_response,
                **kwargs
            )
        else:
            raise ValueError("prompt must be a string or list of messages")
    
    async def generate_batch(self,
                            prompts: List[Union[str, List[Dict[str, str]]]],
                            model: Optional[str] = None,
                            max_tokens: int = 150,
                            temperature: float = 0.8,
                            top_p: float = 0.9,
                            stop: Optional[List[str]] = None,
                            return_full_response: bool = False,
                            max_concurrent: int = 5,
                            use_true_batching: bool = True,
                            **kwargs) -> List[Union[str, CompletionResponse]]:
        """
        Generate multiple completions with optimized batching
        
        Args:
            prompts: List of prompts (strings or message lists)
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: List of stop sequences
            return_full_response: If True, return CompletionResponse objects, else just text
            max_concurrent: Maximum concurrent requests (for non-batched mode)
            use_true_batching: If True, use OpenAI API 'n' parameter for identical prompts
            **kwargs: Additional parameters
            
        Returns:
            List of generated text strings or CompletionResponse objects
        """
        if not prompts:
            return []
        
        # Try to use true batching when possible
        if use_true_batching and self._can_use_true_batching(prompts):
            return await self._true_batch_generate(
                prompts, model, max_tokens, temperature, top_p, stop, return_full_response, **kwargs
            )
        else:
            # Fall back to concurrent individual requests
            return await self._concurrent_generate(
                prompts, model, max_tokens, temperature, top_p, stop, return_full_response, max_concurrent, **kwargs
            )
    
    def _can_use_true_batching(self, prompts: List[Union[str, List[Dict[str, str]]]]) -> bool:
        """Check if we can use true batching (all prompts are identical)"""
        if len(prompts) <= 1:
            return False
        
        # Check if all prompts are identical
        first_prompt = prompts[0]
        return all(prompt == first_prompt for prompt in prompts[1:])
    
    async def _true_batch_generate(self,
                                  prompts: List[Union[str, List[Dict[str, str]]]],
                                  model: Optional[str],
                                  max_tokens: int,
                                  temperature: float,
                                  top_p: float,
                                  stop: Optional[List[str]],
                                  return_full_response: bool,
                                  **kwargs) -> List[Union[str, CompletionResponse]]:
        """Generate using true batching with 'n' parameter"""
        prompt = prompts[0]  # All prompts are identical
        n = len(prompts)
        
        logger.info(f"🚀 Using TRUE BATCHING: {n} completions in single API call")
        
        if isinstance(prompt, str):
            # Text completion
            response = await self.complete(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                return_full_response=True,  # We need full response to extract multiple choices
                n=n,  # TRUE BATCHING PARAMETER
                **kwargs
            )
        else:
            # Chat completion
            response = await self.chat_complete(
                messages=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                return_full_response=True,  # We need full response to extract multiple choices
                n=n,  # TRUE BATCHING PARAMETER
                **kwargs
            )
        
        # Extract multiple results from the single batched response
        results = []
        raw_response = response.raw_response
        
        if 'choices' in raw_response:
            for choice in raw_response['choices']:
                if isinstance(prompt, str):
                    # Text completion
                    text = choice.get('text', '').strip()
                else:
                    # Chat completion
                    message = choice.get('message', {})
                    text = message.get('content', '').strip()
                
                if return_full_response:
                    # Create individual response objects
                    individual_response = {**raw_response, 'choices': [choice]}
                    results.append(CompletionResponse(text, individual_response))
                else:
                    results.append(text)
        
        # Pad with empty results if we didn't get enough choices
        while len(results) < n:
            if return_full_response:
                results.append(CompletionResponse("", {}))
            else:
                results.append("")
        
        return results[:n]  # Ensure we return exactly n results
    
    async def _concurrent_generate(self,
                                  prompts: List[Union[str, List[Dict[str, str]]]],
                                  model: Optional[str],
                                  max_tokens: int,
                                  temperature: float,
                                  top_p: float,
                                  stop: Optional[List[str]],
                                  return_full_response: bool,
                                  max_concurrent: int,
                                  **kwargs) -> List[Union[str, CompletionResponse]]:
        """Generate using concurrent individual requests (fallback method)"""
        logger.info(f"🔄 Using CONCURRENT mode: {len(prompts)} individual API calls (max {max_concurrent} concurrent)")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_single(prompt):
            async with semaphore:
                return await self.generate(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    return_full_response=return_full_response,
                    **kwargs
                )
        
        tasks = [generate_single(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch generation failed for prompt {i}: {result}")
                processed_results.append("Error: Generation failed" if not return_full_response else CompletionResponse("Error: Generation failed", {}))
            else:
                processed_results.append(result)
        
        return processed_results


# Global client instance
_client = None

def get_client() -> OpenAIClient:
    """Get or create global OpenAI client instance"""
    global _client
    if _client is None:
        _client = OpenAIClient()
    return _client

def set_client(client: OpenAIClient):
    """Set global OpenAI client instance"""
    global _client
    _client = client

async def cleanup_global_client():
    """Clean up the global client instance"""
    global _client
    if _client:
        try:
            await _client.close()
        except Exception as e:
            logger.debug(f"Error cleaning up global client: {e}")
        finally:
            _client = None

# Convenience functions for direct usage
async def generate_text(prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
    """Generate text using the global client"""
    client = get_client()
    return await client.generate(prompt, **kwargs)

async def generate_batch(prompts: List[Union[str, List[Dict[str, str]]]], **kwargs) -> List[str]:
    """Generate multiple texts using the global client"""
    client = get_client()
    return await client.generate_batch(prompts, **kwargs) 