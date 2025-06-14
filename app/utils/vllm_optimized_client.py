"""
vLLM-optimized client for enhanced batching performance.
Extends the base OpenAI client with vLLM-specific optimizations.
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .openai_client import OpenAIClient, CompletionResponse

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Represents a batched request with metadata"""
    prompt: Union[str, List[Dict[str, str]]]
    request_id: str
    priority: int = 0
    max_tokens: int = 150
    temperature: float = 0.8
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class BatchConfig:
    """Configuration for vLLM batching optimization"""
    max_batch_size: int = 32  # Optimal for most vLLM setups
    max_wait_time: float = 0.1  # 100ms max wait for batching
    target_batch_size: int = 16  # Target size for efficiency
    priority_threshold: int = 5  # High priority requests bypass batching
    adaptive_batching: bool = True  # Adapt batch size based on performance
    token_budget_per_batch: int = 8192  # Max tokens per batch (input + output)


class VLLMOptimizedClient(OpenAIClient):
    """
    An enhanced client that optimizes batching for vLLM backends by:
    - Using continuous batching via a request queue
    - Implementing adaptive batch sizing based on performance
    - Grouping similar requests (e.g., same max_tokens) for better efficiency
    """
    
    def __init__(self, 
                 batch_config: Optional[BatchConfig] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_config = batch_config or BatchConfig()
        
        # Batching state - defer queue creation until needed
        self._request_queue = None
        self._batch_processor_task = None
        self._batch_results = {}
        self._batch_futures = {}
        self._request_counter = 0
        self._processor_started = False
        
        # Performance tracking
        self._batch_stats = {
            'total_requests': 0,
            'batched_requests': 0,
            'batch_efficiency': [],
            'avg_batch_size': 0,
            'throughput_history': []
        }
        
        # Don't start batch processor during init - defer until needed
    
    def _normalize_prompt(self, prompt: Any) -> Union[str, List[Dict[str, str]]]:
        """
        Normalizes a prompt to be either a string or a list of chat messages.
        Handles the custom {'prompt': str, ...} dict format from dataset generation.
        """
        if isinstance(prompt, dict) and 'prompt' in prompt and 'role' not in prompt:
            # This is our custom prompt dictionary, not a chat message
            return prompt['prompt']
        # It's either a string, a list of chat messages, or something else we pass through
        return prompt

    def _ensure_batch_processor(self):
        """
        Ensure batch processor is running.
        Handles cases where no event loop is available (e.g., during Streamlit init).
        """
        # Avoid starting if already running
        if self._processor_started and self._batch_processor_task and not self._batch_processor_task.done():
            return

        try:
            # Check for a running event loop
            loop = asyncio.get_running_loop()
            
            # Initialize queue if it doesn't exist
            if self._request_queue is None:
                self._request_queue = asyncio.Queue()

            # Start processor task if not running
            if self._batch_processor_task is None or self._batch_processor_task.done():
                self._batch_processor_task = loop.create_task(self._batch_processor())
                self._processor_started = True
                logger.info("✅ vLLM batch processor started successfully.")

        except RuntimeError:
            # No running event loop, log and continue without batching
            if not self._processor_started: # Log only once
                 logger.warning("⚠️ No running event loop. vLLM-optimized client will fall back to standard requests.")
            # Ensure processor is marked as not started so fallbacks are used.
            self._processor_started = False
    
    async def _batch_processor(self):
        """Main batch processing loop optimized for vLLM"""
        logger.info("🚀 Starting vLLM-optimized batch processor")
        
        while True:
            try:
                batch_requests = await self._collect_batch()
                if batch_requests:
                    await self._process_batch(batch_requests)
                else:
                    # Small delay when no requests
                    await asyncio.sleep(0.01)
                    
            except asyncio.CancelledError:
                logger.info("Batch processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # Brief pause on error
    
    async def _collect_batch(self) -> List[BatchRequest]:
        """Collect requests from the queue to form a dynamically sized batch."""
        batch = []
        start_time = time.time()
        total_tokens = 0
        
        # Wait for the first request with a timeout
        try:
            # Block until the first request arrives, but with a timeout
            request = await asyncio.wait_for(self._request_queue.get(), timeout=self.batch_config.max_wait_time)
            
            # Check token budget
            request_tokens = self._estimate_tokens(request)
            if request_tokens <= self.batch_config.token_budget_per_batch:
                batch.append(request)
                total_tokens += request_tokens
                self._request_queue.task_done()
            else:
                logger.warning(f"Request {request.request_id} is too large for batch budget, processing individually.")
                asyncio.create_task(self._process_single_request(request))

        except asyncio.TimeoutError:
            return [] # No requests came in time, return empty batch

        # Greedily collect more requests if they are available
        while (
            len(batch) < self.batch_config.max_batch_size and
            total_tokens < self.batch_config.token_budget_per_batch and
            not self._request_queue.empty()
        ):
            try:
                request = self._request_queue.get_nowait()
                
                # Check token budget
                request_tokens = self._estimate_tokens(request)
                if total_tokens + request_tokens <= self.batch_config.token_budget_per_batch:
                    batch.append(request)
                    total_tokens += request_tokens
                    self._request_queue.task_done()
                else:
                    # If the request doesn't fit, put it back and stop collecting
                    self._request_queue.put_nowait(request)
                    break
            except asyncio.QueueEmpty:
                break
        
        # Adaptive batching: adjust target size based on performance
        if self.batch_config.adaptive_batching:
            self._adapt_batch_size(len(batch))
        
        return batch
    
    def _estimate_tokens(self, request: BatchRequest) -> int:
        """Estimate token count for a request to manage batch budget."""
        # Simple estimation: average token length is ~1.33 chars
        # This is a rough heuristic and can be improved with a real tokenizer
        prompt_data = self._normalize_prompt(request.prompt)
        prompt_text = ""
        if isinstance(prompt_data, list): # Handle chat messages
            try:
                prompt_text = " ".join([msg.get('content', '') for msg in prompt_data if isinstance(msg, dict)])
            except TypeError:
                logger.warning(f"Could not join prompt messages, estimating from raw data: {prompt_data}")
                prompt_text = str(prompt_data)

        elif isinstance(prompt_data, str):
            prompt_text = prompt_data
        
        return len(prompt_text.split()) + request.max_tokens
    
    def _adapt_batch_size(self, current_batch_size: int):
        """Adapt batch size based on recent throughput."""
        if not self.batch_config.adaptive_batching:
            return
        if len(self._batch_stats['batch_efficiency']) > 10:
            recent_efficiency = np.mean(self._batch_stats['batch_efficiency'][-10:])
            
            if recent_efficiency > 0.8 and current_batch_size < self.batch_config.max_batch_size:
                # High efficiency, try larger batches
                self.batch_config.target_batch_size = min(
                    self.batch_config.target_batch_size + 2,
                    self.batch_config.max_batch_size
                )
            elif recent_efficiency < 0.6:
                # Low efficiency, try smaller batches
                self.batch_config.target_batch_size = max(
                    self.batch_config.target_batch_size - 2,
                    4
                )
    
    async def _process_batch(self, batch_requests: List[BatchRequest]):
        """Process a batch of requests with vLLM optimization"""
        if not batch_requests:
            return
        
        start_time = time.time()
        batch_size = len(batch_requests)
        
        logger.debug(f"Processing batch of {batch_size} requests")
        
        # Group requests by similarity for better vLLM efficiency
        grouped_batches = self._group_similar_requests(batch_requests)
        
        for group in grouped_batches:
            await self._process_request_group(group)
        
        # Update statistics
        processing_time = time.time() - start_time
        self._update_batch_stats(batch_size, processing_time)
    
    def _group_similar_requests(self, requests: List[BatchRequest]) -> List[List[BatchRequest]]:
        """Group requests by similar generation parameters to optimize vLLM processing."""
        groups = defaultdict(list)
        for req in requests:
            # Group by all generation parameters except the prompt itself
            group_key = (
                req.max_tokens,
                req.temperature,
                req.top_p,
                tuple(req.stop) if req.stop else None
            )
            groups[group_key].append(req)
        return list(groups.values())
    
    async def _process_request_group(self, group: List[BatchRequest]):
        """Process a group of requests with identical parameters."""
        if not group:
            return

        try:
            # Use true batching if supported and beneficial
            if len(group) > 1 and self._can_use_true_batching_group(group):
                await self._process_batch_group(group)
            else:
                # Process concurrently if true batching is not possible
                await asyncio.gather(*(self._process_single_request(req) for req in group))
        except Exception as e:
            logger.error(f"Error processing request group: {e}", exc_info=True)
            for req in group:
                self._batch_futures[req.request_id].set_exception(e)
    
    async def _process_single_request(self, request: BatchRequest):
        """Process a single request, for high priority or non-batchable requests."""
        try:
            normalized_prompt = self._normalize_prompt(request.prompt)
            response = await super().generate(
                prompt=normalized_prompt,
                model=self.default_model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
                **request.kwargs
            )
            self._batch_futures[request.request_id].set_result(response)
        except Exception as e:
            logger.error(f"Error processing single request {request.request_id}: {e}", exc_info=True)
            if request.request_id in self._batch_futures:
                self._batch_futures[request.request_id].set_exception(e)
    
    async def _process_batch_group(self, group: List[BatchRequest]):
        """Process a batch of requests using the underlying client's batch generation."""
        try:
            # Use true batching for efficiency
            responses = await self._true_batch_group(group)
            
            # Distribute responses back to their corresponding futures
            for request, response in zip(group, responses):
                if request.request_id in self._batch_futures:
                    self._batch_futures[request.request_id].set_result(response)
        except Exception as e:
            logger.error(f"Error processing batch group: {e}", exc_info=True)
            for request in group:
                if request.request_id in self._batch_futures:
                    self._batch_futures[request.request_id].set_exception(e)
    
    async def _true_batch_group(self, group: List[BatchRequest]) -> List[str]:
        """Performs a true batch generation for a group of requests."""
        first_req = group[0]
        prompts = [self._normalize_prompt(req.prompt) for req in group]
        
        responses = await super()._true_batch_generate(
            prompts=prompts,
            model=self.default_model,
            max_tokens=first_req.max_tokens,
            temperature=first_req.temperature,
            top_p=first_req.top_p,
            stop=first_req.stop,
            return_full_response=False,
            **first_req.kwargs
        )
        return responses
    
    def _update_batch_stats(self, batch_size: int, processing_time: float):
        """Update performance statistics after processing a batch."""
        if processing_time > 0:
            throughput = batch_size / processing_time
            self._batch_stats['throughput_history'].append(throughput)
            
            # Keep last 50 stats for moving average
            self._batch_stats['throughput_history'] = self._batch_stats['throughput_history'][-50:]

        self._batch_stats['total_requests'] += batch_size
        self._batch_stats['batched_requests'] += batch_size
        
        # Update average batch size
        total_batched = self._batch_stats['batched_requests']
        if total_batched > 0:
            current_avg = self._batch_stats['avg_batch_size']
            self._batch_stats['avg_batch_size'] = ((current_avg * (total_batched - batch_size)) + batch_size) / total_batched
    
    def _can_use_true_batching_group(self, group: List[BatchRequest]) -> bool:
        """Check if true batching can be used for a group."""
        prompts = [self._normalize_prompt(req.prompt) for req in group]
        return self._can_use_true_batching(prompts)
    
    async def generate_optimized(self,
                               prompt: Union[str, List[Dict[str, str]]],
                               priority: int = 0,
                               **kwargs) -> str:
        """Generate with vLLM-optimized batching"""
        self._ensure_batch_processor()
        
        if not self._processor_started:
            logger.warning("Batch processor not running, falling back to standard batch generation.")
            # The base class `generate_batch` expects strings or message lists
            # We need to extract the prompt string here for the fallback.
            string_prompts = [self._normalize_prompt(prompt)] if isinstance(prompt, str) else [self._normalize_prompt(p) for p in prompt]
            return await super().generate_batch(prompts=string_prompts, **kwargs)

        request_ids = []
        futures = []
        
        # Create request
        request_id = f"req_{self._request_counter}"
        self._request_counter += 1
        
        request = BatchRequest(
            prompt=prompt,
            request_id=request_id,
            priority=priority,
            max_tokens=kwargs.get('max_tokens', 150),
            temperature=kwargs.get('temperature', 0.8),
            top_p=kwargs.get('top_p', 0.9),
            stop=kwargs.get('stop'),
            kwargs={k: v for k, v in kwargs.items() 
                   if k not in ['max_tokens', 'temperature', 'top_p', 'stop']}
        )
        
        # Create future for result
        future = asyncio.Future()
        self._batch_futures[request_id] = future
        
        # High priority requests bypass batching
        if priority >= self.batch_config.priority_threshold:
            return await self.generate(prompt=prompt, **kwargs)
        
        # Add to batch queue
        await self._request_queue.put(request)
        
        try:
            # Wait for result
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            # Cleanup on timeout
            self._batch_futures.pop(request_id, None)
            raise
        finally:
            # Cleanup
            self._batch_futures.pop(request_id, None)
    
    async def generate_batch_optimized(self,
                                     prompts: List[Union[str, List[Dict[str, str]]]],
                                     priority: int = 0,
                                     **kwargs) -> List[str]:
        """Generate multiple completions with vLLM optimization"""
        if not prompts:
            return []
        
        # For large batches, use the optimized batching
        if len(prompts) > 10:
            tasks = [
                self.generate_optimized(prompt, priority=priority, **kwargs)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)
        else:
            # For small batches, use existing logic
            return await super().generate_batch(prompts, **kwargs)
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get current batching performance statistics"""
        total_reqs = self._batch_stats['total_requests']
        batched_reqs = self._batch_stats['batched_requests']
        
        return {
            'total_requests': total_reqs,
            'batched_requests': batched_reqs,
            'batch_utilization': batched_reqs / max(total_reqs, 1),
            'avg_batch_size': self._batch_stats['avg_batch_size'],
            'avg_efficiency': np.mean(self._batch_stats['batch_efficiency']) if self._batch_stats['batch_efficiency'] else 0,
            'current_queue_size': self._request_queue.qsize(),
            'active_futures': len(self._batch_futures)
        }
    
    async def close(self):
        """Close the client and cleanup resources"""
        # Cancel batch processor
        if self._batch_processor_task and not self._batch_processor_task.done():
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup futures
        for future in self._batch_futures.values():
            if not future.done():
                future.cancel()
        
        self._batch_futures.clear()
        
        # Close parent
        await super().close() 