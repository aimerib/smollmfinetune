"""
Adapter Management System for hot-swappable character models.

Handles loading, caching, versioning and benchmarking of character adapters
with support for triple-head models.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import shutil
import numpy as np

import torch
from safetensors.torch import load_file, save_file
from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)


@dataclass
class AdapterVersion:
    """Version information for an adapter"""
    version: str
    path: str
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdapterMetadata:
    """Metadata for character adapters"""
    character_id: str
    adapter_type: str = "standard"  # standard, triple_head, memory_only
    memory_head_trained: bool = False
    control_head_trained: bool = False
    memory_vector_dim: int = 768
    control_vocab_size: int = 1000
    training_steps: int = 0
    base_model: str = ""
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "character_id": self.character_id,
            "adapter_type": self.adapter_type,
            "memory_head_trained": self.memory_head_trained,
            "control_head_trained": self.control_head_trained,
            "memory_vector_dim": self.memory_vector_dim,
            "control_vocab_size": self.control_vocab_size,
            "training_steps": self.training_steps,
            "base_model": self.base_model,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdapterMetadata':
        """Create from dictionary"""
        return cls(**data)


class AdapterManager:
    """
    Manages hot-swappable adapters for character models.
    
    Features:
    - Zero-downtime adapter swapping
    - Version management and rollback
    - Performance benchmarking
    - Multi-adapter ensemble inference
    - Memory-efficient caching
    """
    
    def __init__(self, base_model_path: Optional[str] = None,
                 cache_dir: str = "adapter_cache",
                 max_loaded_adapters: int = 5):
        """
        Initialize adapter manager.
        
        Args:
            base_model_path: Path to base model
            cache_dir: Directory for adapter cache
            max_loaded_adapters: Maximum adapters to keep in memory
        """
        self.base_model_path = base_model_path or "HuggingFaceTB/SmolLM2-135M-Instruct"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Loaded adapters cache
        self.loaded_adapters: Dict[str, Any] = {}
        self.max_loaded_adapters = max_loaded_adapters
        self.adapter_access_times: Dict[str, float] = {}
        
        # Version tracking
        self.version_history: Dict[str, List[AdapterVersion]] = {}
        self.active_versions: Dict[str, str] = {}
        
        # Base model (lazy loaded)
        self._base_model = None
        self._tokenizer = None
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, float]]] = {}
        
        # Load lock for thread safety
        self._load_lock = asyncio.Lock()
        
        logger.info(f"AdapterManager initialized with cache dir: {self.cache_dir}")
    
    async def load_adapter(self, character_id: str, adapter_path: str,
                         version: Optional[str] = None,
                         metadata: Optional[AdapterMetadata] = None) -> float:
        """
        Load an adapter into memory.
        
        Args:
            character_id: Unique character identifier
            adapter_path: Path to adapter files
            version: Version identifier
            metadata: Adapter metadata
            
        Returns:
            Load time in seconds
        """
        start_time = time.time()
        
        async with self._load_lock:
            try:
                # Check cache capacity
                if len(self.loaded_adapters) >= self.max_loaded_adapters:
                    await self._evict_least_used_adapter()
                
                # Load adapter configuration
                adapter_config = PeftConfig.from_pretrained(adapter_path)
                
                # Get or create base model
                base_model = await self._get_base_model()
                
                # Load adapter weights
                logger.info(f"Loading adapter for {character_id} from {adapter_path}")
                adapter_model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    adapter_name=character_id
                )
                
                # Store in cache
                self.loaded_adapters[character_id] = {
                    "model": adapter_model,
                    "config": adapter_config,
                    "metadata": metadata or AdapterMetadata(character_id=character_id),
                    "path": adapter_path,
                    "loaded_at": time.time()
                }
                
                # Update access time
                self.adapter_access_times[character_id] = time.time()
                
                # Track version
                if version:
                    self._add_version_history(character_id, version, adapter_path, metadata)
                    self.active_versions[character_id] = version
                
                load_time = time.time() - start_time
                logger.info(f"Adapter {character_id} loaded in {load_time:.2f}s")
                
                return load_time
                
            except Exception as e:
                logger.error(f"Failed to load adapter {character_id}: {e}")
                raise
    
    async def hot_swap_adapter(self, character_id: str, new_adapter_path: str,
                             version: Optional[str] = None) -> float:
        """
        Hot-swap an adapter without downtime.
        
        Args:
            character_id: Character to update
            new_adapter_path: Path to new adapter
            version: New version identifier
            
        Returns:
            Swap time in seconds
        """
        start_time = time.time()
        
        try:
            # Pre-load new adapter to verify it works
            temp_id = f"{character_id}_swap_temp"
            await self.load_adapter(temp_id, new_adapter_path)
            
            async with self._load_lock:
                # Remove old adapter if exists
                if character_id in self.loaded_adapters:
                    old_adapter = self.loaded_adapters.pop(character_id)
                    # Clean up GPU memory
                    del old_adapter["model"]
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Move temp adapter to correct ID
                self.loaded_adapters[character_id] = self.loaded_adapters.pop(temp_id)
                self.loaded_adapters[character_id]["path"] = new_adapter_path
                
                # Update version
                if version:
                    self.active_versions[character_id] = version
                
                swap_time = time.time() - start_time
                logger.info(f"Hot-swapped adapter {character_id} in {swap_time:.2f}s")
                
                return swap_time
                
        except Exception as e:
            logger.error(f"Hot-swap failed for {character_id}: {e}")
            # Clean up temp adapter if exists
            if temp_id in self.loaded_adapters:
                del self.loaded_adapters[temp_id]
            raise
    
    async def rollback_adapter(self, character_id: str, target_version: str) -> bool:
        """
        Rollback to a previous adapter version.
        
        Args:
            character_id: Character to rollback
            target_version: Version to rollback to
            
        Returns:
            Success status
        """
        if character_id not in self.version_history:
            logger.error(f"No version history for {character_id}")
            return False
        
        # Find target version
        target_info = None
        for version_info in self.version_history[character_id]:
            if version_info.version == target_version:
                target_info = version_info
                break
        
        if not target_info:
            logger.error(f"Version {target_version} not found for {character_id}")
            return False
        
        # Perform hot-swap to target version
        try:
            await self.hot_swap_adapter(
                character_id,
                target_info.path,
                target_version
            )
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    async def ensemble_inference(self, prompt: str, character_ids: List[str],
                               aggregation: str = "weighted",
                               weights: Optional[List[float]] = None,
                               **kwargs) -> Dict[str, Any]:
        """
        Run inference with multiple character adapters.
        
        Args:
            prompt: Input prompt
            character_ids: List of characters to use
            aggregation: How to combine results (weighted, voting, best)
            weights: Weights for weighted aggregation
            **kwargs: Generation parameters
            
        Returns:
            Ensemble results with individual responses
        """
        results = {}
        
        # Generate with each character
        tasks = []
        for char_id in character_ids:
            if char_id in self.loaded_adapters:
                task = self._generate_with_adapter(char_id, prompt, **kwargs)
                tasks.append((char_id, task))
        
        # Wait for all results
        responses = await asyncio.gather(*[task for _, task in tasks])
        
        # Store individual results
        for (char_id, _), response in zip(tasks, responses):
            results[char_id] = {
                "response": response,
                "metadata": self.loaded_adapters[char_id]["metadata"].to_dict()
            }
        
        # Aggregate if requested
        if aggregation == "weighted" and len(results) > 1:
            # Implement weighted combination (simplified)
            if weights is None:
                weights = [1.0 / len(results)] * len(results)
            
            # For now, just return the weighted random choice
            # In production, would implement proper ensemble
            import random
            chosen = random.choices(
                list(results.keys()),
                weights=weights[:len(results)]
            )[0]
            results["ensemble"] = results[chosen]["response"]
        
        return results
    
    async def benchmark_adapter(self, character_id: str, 
                              num_samples: int = 10,
                              test_prompts: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Benchmark adapter performance.
        
        Args:
            character_id: Character to benchmark
            num_samples: Number of test runs
            test_prompts: Custom test prompts
            
        Returns:
            Performance metrics
        """
        if character_id not in self.loaded_adapters:
            raise ValueError(f"Adapter {character_id} not loaded")
        
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you today?",
                "Tell me about yourself.",
                "What do you think about the weather?",
                "What are your goals and dreams?",
                "Describe your perfect day."
            ]
        
        # Run benchmarks
        inference_times = []
        tokens_generated = []
        memory_usage_start = self._get_gpu_memory_usage()
        
        for i in range(num_samples):
            prompt = test_prompts[i % len(test_prompts)]
            
            start_time = time.time()
            response = await self._generate_with_adapter(
                character_id, prompt, max_tokens=100
            )
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            tokens_generated.append(len(response.split()))
        
        memory_usage_end = self._get_gpu_memory_usage()
        
        # Calculate metrics
        metrics = {
            "avg_inference_time_ms": np.mean(inference_times) * 1000,
            "p95_inference_time_ms": np.percentile(inference_times, 95) * 1000,
            "tokens_per_second": np.sum(tokens_generated) / np.sum(inference_times),
            "memory_usage_mb": memory_usage_end - memory_usage_start,
            "num_samples": num_samples
        }
        
        # Store in history
        if character_id not in self.performance_history:
            self.performance_history[character_id] = []
        self.performance_history[character_id].append({
            "timestamp": time.time(),
            **metrics
        })
        
        return metrics
    
    def get_active_adapter(self, character_id: str) -> Optional[str]:
        """Get path of currently active adapter"""
        if character_id in self.loaded_adapters:
            return self.loaded_adapters[character_id]["path"]
        return None
    
    def get_active_version(self, character_id: str) -> Optional[str]:
        """Get currently active version"""
        return self.active_versions.get(character_id)
    
    def get_adapter_info(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Get adapter information"""
        if character_id not in self.loaded_adapters:
            return None
        
        adapter = self.loaded_adapters[character_id]
        metadata = adapter["metadata"]
        
        return {
            "character_id": character_id,
            "adapter_type": metadata.adapter_type,
            "memory_head_trained": metadata.memory_head_trained,
            "control_head_trained": metadata.control_head_trained,
            "memory_vector_dim": metadata.memory_vector_dim,
            "path": adapter["path"],
            "loaded_at": adapter["loaded_at"],
            "version": self.active_versions.get(character_id),
            "memory_usage_mb": self._estimate_adapter_memory(adapter)
        }
    
    def get_adapter_memory_usage(self, character_id: str) -> float:
        """Get estimated memory usage in MB"""
        if character_id not in self.loaded_adapters:
            return 0.0
        
        return self._estimate_adapter_memory(self.loaded_adapters[character_id])
    
    def get_version_history(self, character_id: str) -> List[Dict[str, Any]]:
        """Get version history for character"""
        if character_id not in self.version_history:
            return []
        
        return [
            {
                "version": v.version,
                "path": v.path,
                "created_at": v.created_at,
                "metadata": v.metadata,
                "performance": v.performance_metrics
            }
            for v in self.version_history[character_id]
        ]
    
    async def _get_base_model(self):
        """Get or load base model"""
        if self._base_model is None:
            logger.info(f"Loading base model: {self.base_model_path}")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
            self._base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        
        return self._base_model
    
    async def _evict_least_used_adapter(self):
        """Evict least recently used adapter"""
        if not self.loaded_adapters:
            return
        
        # Find LRU adapter
        lru_id = min(
            self.adapter_access_times.keys(),
            key=lambda k: self.adapter_access_times.get(k, 0)
        )
        
        logger.info(f"Evicting adapter {lru_id} to free memory")
        
        # Remove from cache
        if lru_id in self.loaded_adapters:
            del self.loaded_adapters[lru_id]
            del self.adapter_access_times[lru_id]
            
            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def _generate_with_adapter(self, character_id: str, prompt: str,
                                   max_tokens: int = 150, **kwargs) -> str:
        """Generate text using specific adapter"""
        if character_id not in self.loaded_adapters:
            raise ValueError(f"Adapter {character_id} not loaded")
        
        # Update access time
        self.adapter_access_times[character_id] = time.time()
        
        # Get model and tokenizer
        adapter_model = self.loaded_adapters[character_id]["model"]
        tokenizer = self._tokenizer
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = adapter_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.9),
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
    
    def _add_version_history(self, character_id: str, version: str,
                           path: str, metadata: Optional[AdapterMetadata]):
        """Add version to history"""
        if character_id not in self.version_history:
            self.version_history[character_id] = []
        
        version_info = AdapterVersion(
            version=version,
            path=path,
            created_at=time.time(),
            metadata=metadata.to_dict() if metadata else {}
        )
        
        self.version_history[character_id].append(version_info)
    
    def _estimate_adapter_memory(self, adapter: Dict[str, Any]) -> float:
        """Estimate adapter memory usage in MB"""
        # Rough estimation based on adapter config
        config = adapter.get("config")
        if not config:
            return 100.0  # Default estimate
        
        # Calculate based on adapter dimensions
        # This is a simplified estimation
        vocab_size = getattr(config, "vocab_size", 50000)
        hidden_size = getattr(config, "hidden_size", 768)
        num_layers = getattr(config, "num_hidden_layers", 12)
        
        # Rough calculation: embeddings + layer weights
        param_count = vocab_size * hidden_size + num_layers * hidden_size * hidden_size * 4
        
        # Convert to MB (assuming float16)
        memory_mb = (param_count * 2) / (1024 * 1024)
        
        return memory_mb
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0 