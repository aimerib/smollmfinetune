"""
Data Pipeline for Narrative Engine

This module provides the DatasetProcessor class that handles tokenization,
validation, and preparation of DatasetSample objects for training the
dual-head Narrative-LLM.
"""

import torch
from typing import Dict, List, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import logging

from .data_schema import DatasetSample, Turn

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Processes DatasetSample objects for training.
    
    Handles:
    - Tokenization of conversation turns
    - Creation of loss masks for dual-head architecture  
    - Channel identification (text vs action)
    - Batch processing for DataLoader
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048):
        """
        Initialize the DatasetProcessor.
        
        Args:
            tokenizer: HuggingFace tokenizer for text processing
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Special tokens for channel identification
        self.text_channel_token = "<|text|>"
        self.action_channel_token = "<|action|>"
        
        # Add special tokens if they don't exist
        special_tokens = [self.text_channel_token, self.action_channel_token]
        new_tokens = [token for token in special_tokens if token not in self.tokenizer.get_vocab()]
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
    
    def process_sample(self, sample: DatasetSample) -> Dict[str, torch.Tensor]:
        """
        Process a single DatasetSample into tensors ready for training.
        
        Args:
            sample: DatasetSample object to process
            
        Returns:
            Dictionary containing:
            - input_ids: Token IDs for the conversation
            - attention_mask: Attention mask for padding
            - loss_mask: Mask indicating which tokens to include in loss (assistant turns only)
            - channel_mask: Mask indicating text (0) vs action (1) channels
        """
        # Build the conversation text with special tokens
        conversation_parts = []
        channel_markers = []  # Track which parts correspond to which channels
        
        # Add session context
        conversation_parts.append(f"Session: {sample.session_id}")
        channel_markers.append("context")
        
        # Add persona mix context
        persona_text = ", ".join([f"{name}({weight:.1f})" for name, weight in sample.persona_mix.items()])
        conversation_parts.append(f"Persona: {persona_text}")
        channel_markers.append("context")
        
        # Add memory slots
        if sample.memory_slots:
            memory_text = " | ".join(sample.memory_slots)
            conversation_parts.append(f"Memory: {memory_text}")
            channel_markers.append("context")
        
        # Process conversation turns
        for turn in sample.turns:
            if turn.sender == "user":
                conversation_parts.append(f"User: {turn.text}")
                channel_markers.append("user")
            else:  # assistant
                # Add channel marker before assistant response
                channel_token = self.action_channel_token if turn.channel == "action" else self.text_channel_token
                conversation_parts.append(f"{channel_token} Assistant: {turn.text}")
                channel_markers.append(turn.channel)
        
        # Join all parts
        full_text = "\n".join(conversation_parts)
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Create loss and channel masks
        loss_mask, channel_mask = self._create_masks(full_text, input_ids, sample.turns)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "channel_mask": channel_mask
        }
    
    def _create_masks(self, full_text: str, input_ids: torch.Tensor, turns: List[Turn]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create loss mask and channel mask for dual-head training.
        
        Args:
            full_text: The full conversation text
            input_ids: Tokenized input IDs
            turns: List of conversation turns
            
        Returns:
            Tuple of (loss_mask, channel_mask)
            - loss_mask: 1 for assistant tokens to include in loss, 0 otherwise
            - channel_mask: 0 for text channel, 1 for action channel
        """
        # Initialize masks
        seq_len = input_ids.shape[0]
        loss_mask = torch.zeros(seq_len, dtype=torch.long)
        channel_mask = torch.zeros(seq_len, dtype=torch.long)
        
        # Tokenize the full text to get token positions
        tokens = self.tokenizer.tokenize(full_text)
        
        # Find assistant turn positions
        assistant_positions = []
        text = full_text
        
        for turn in turns:
            if turn.sender == "assistant":
                # Find the position of this assistant turn
                channel_token = self.action_channel_token if turn.channel == "action" else self.text_channel_token
                search_pattern = f"{channel_token} Assistant: {turn.text}"
                
                start_pos = text.find(search_pattern)
                if start_pos != -1:
                    # Calculate token positions approximately
                    char_to_token_ratio = len(tokens) / len(full_text) if len(full_text) > 0 else 0
                    
                    # Find assistant response start (after "Assistant: ")
                    assistant_start = start_pos + len(f"{channel_token} Assistant: ")
                    assistant_end = assistant_start + len(turn.text)
                    
                    # Convert character positions to approximate token positions
                    token_start = int(assistant_start * char_to_token_ratio)
                    token_end = int(assistant_end * char_to_token_ratio)
                    
                    # Clamp to sequence length
                    token_start = max(0, min(token_start, seq_len - 1))
                    token_end = max(token_start + 1, min(token_end, seq_len))
                    
                    assistant_positions.append((token_start, token_end, turn.channel))
        
        # Apply masks
        for token_start, token_end, channel in assistant_positions:
            # Mark for loss calculation (assistant turns only)
            loss_mask[token_start:token_end] = 1
            
            # Mark channel type (0 for text, 1 for action)
            channel_value = 1 if channel == "action" else 0
            channel_mask[token_start:token_end] = channel_value
        
        return loss_mask, channel_mask
    
    def process_batch(self, samples: List[DatasetSample]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of DatasetSample objects.
        
        Args:
            samples: List of DatasetSample objects
            
        Returns:
            Dictionary with batched tensors
        """
        batch_results = [self.process_sample(sample) for sample in samples]
        
        # Stack tensors
        batched = {}
        for key in batch_results[0].keys():
            batched[key] = torch.stack([result[key] for result in batch_results])
        
        return batched
    
    def validate_batch(self, samples: List[DatasetSample]) -> tuple[bool, List[str]]:
        """
        Validate a batch of DatasetSample objects.
        
        Args:
            samples: List of DatasetSample objects to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for i, sample in enumerate(samples):
            try:
                # Validate using Pydantic
                DatasetSample(**sample.model_dump())
                
                # Additional validation
                if len(sample.turns) == 0:
                    errors.append(f"Sample {i}: No turns found")
                
                if not sample.session_id:
                    errors.append(f"Sample {i}: Empty session_id")
                
                if sum(sample.persona_mix.values()) <= 0:
                    errors.append(f"Sample {i}: Invalid persona_mix weights")
                
                # Check for assistant turns
                has_assistant = any(turn.sender == "assistant" for turn in sample.turns)
                if not has_assistant:
                    errors.append(f"Sample {i}: No assistant turns found")
                    
            except Exception as e:
                errors.append(f"Sample {i}: Validation error - {str(e)}")
        
        return len(errors) == 0, errors
    
    def create_dataloader(
        self, 
        samples: List[DatasetSample], 
        batch_size: int = 8, 
        shuffle: bool = True,
        **dataloader_kwargs
    ) -> DataLoader:
        """
        Create a PyTorch DataLoader for training.
        
        Args:
            samples: List of DatasetSample objects
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            **dataloader_kwargs: Additional arguments for DataLoader
            
        Returns:
            PyTorch DataLoader
        """
        dataset = NarrativeDataset(samples, self)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            **dataloader_kwargs
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of processed samples
            
        Returns:
            Batched tensors
        """
        # Stack tensors
        batched = {}
        for key in batch[0].keys():
            batched[key] = torch.stack([item[key] for item in batch])
        
        return batched


class NarrativeDataset(Dataset):
    """
    PyTorch Dataset wrapper for DatasetSample objects.
    """
    
    def __init__(self, samples: List[DatasetSample], processor: DatasetProcessor):
        """
        Initialize the dataset.
        
        Args:
            samples: List of DatasetSample objects
            processor: DatasetProcessor for tokenization
        """
        self.samples = samples
        self.processor = processor
        
        # Pre-validate samples
        is_valid, errors = processor.validate_batch(samples)
        if not is_valid:
            logger.warning(f"Dataset validation issues: {errors}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processor.process_sample(self.samples[idx]) 