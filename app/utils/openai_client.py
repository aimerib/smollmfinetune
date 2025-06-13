"""
Simple OpenAI API client wrapper for text generation.
Supports both text completion and chat completion endpoints.
"""

import os
import asyncio
import logging
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
                 default_model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI client
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (defaults to OpenAI, but can be changed for compatible endpoints)
            default_model: Default model to use for requests
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        self.default_model = default_model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Remove trailing slash from base_url
        self.base_url = self.base_url.rstrip('/')
        
        self._session = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            timeout = aiohttp.ClientTimeout(total=120)  # 2 minute timeout
            self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self._session
    
    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
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
        session = await self._get_session()
        
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
        
        try:
            async with session.post(f'{self.base_url}/completions', json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Extract the generated text
                if 'choices' not in data or not data['choices']:
                    raise Exception("No choices returned from API")
                
                text = data['choices'][0].get('text', '').strip()
                
                if return_full_response:
                    return CompletionResponse(text, data)
                else:
                    return text
                    
        except Exception as e:
            logger.error(f"Text completion failed: {e}")
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
        session = await self._get_session()
        
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
        
        try:
            async with session.post(f'{self.base_url}/chat/completions', json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error_text}")
                
                data = await response.json()
                
                # Extract the generated text
                if 'choices' not in data or not data['choices']:
                    raise Exception("No choices returned from API")
                
                message = data['choices'][0].get('message', {})
                text = message.get('content', '').strip()
                
                if return_full_response:
                    return CompletionResponse(text, data)
                else:
                    return text
                    
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
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
                            **kwargs) -> List[Union[str, CompletionResponse]]:
        """
        Generate multiple completions concurrently
        
        Args:
            prompts: List of prompts (strings or message lists)
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: List of stop sequences
            return_full_response: If True, return CompletionResponse objects, else just text
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters
            
        Returns:
            List of generated text strings or CompletionResponse objects
        """
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

# Convenience functions for direct usage
async def generate_text(prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
    """Generate text using the global client"""
    client = get_client()
    return await client.generate(prompt, **kwargs)

async def generate_batch(prompts: List[Union[str, List[Dict[str, str]]]], **kwargs) -> List[str]:
    """Generate multiple texts using the global client"""
    client = get_client()
    return await client.generate_batch(prompts, **kwargs) 