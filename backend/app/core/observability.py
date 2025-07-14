"""
Inference Observability System (R3-4)

Provides logging and storage of model intermediate states during inference:
- Capture attention weights, hidden states, and token probabilities
- Store observability data with request IDs for later inspection
- JSON-based storage with tensor shape metadata
- Integration with existing error handling infrastructure
"""

import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceObservabilityData:
    """Data structure for storing inference observability information"""
    
    request_id: str
    prompt: str
    response: str
    model_path: str
    generation_config: Dict[str, Any]
    attention_weights: List[torch.Tensor]
    hidden_states: List[torch.Tensor]
    token_probabilities: Dict[str, float]
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Set timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, handling tensors appropriately"""
        data = {
            'request_id': self.request_id,
            'prompt': self.prompt,
            'response': self.response,
            'model_path': self.model_path,
            'generation_config': self.generation_config,
            'token_probabilities': self.token_probabilities,
            'timestamp': self.timestamp
        }
        
        # Store tensor shapes instead of actual tensors for JSON compatibility
        data['attention_weights_shape'] = [list(tensor.shape) for tensor in self.attention_weights]
        data['hidden_states_shape'] = [list(tensor.shape) for tensor in self.hidden_states]
        
        return data


class ObservabilityLogger:
    """Handles logging and retrieval of inference observability data"""
    
    def __init__(self, log_dir: str = "training_output/observability_logs"):
        """
        Initialize the observability logger
        
        Args:
            log_dir: Directory to store observability log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ObservabilityLogger initialized with log directory: {self.log_dir}")
    
    def log_inference_data(self, data: InferenceObservabilityData) -> bool:
        """
        Log inference observability data to a JSON file
        
        Args:
            data: InferenceObservabilityData instance to log
            
        Returns:
            bool: True if logging successful, False otherwise
        """
        try:
            log_file = self.log_dir / f"{data.request_id}.json"
            
            # Convert to dictionary for JSON serialization
            log_data = data.to_dict()
            
            # Write to file
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"Logged observability data for request {data.request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log observability data for request {data.request_id}: {e}")
            return False
    
    def get_log_data(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve logged observability data for a specific request ID
        
        Args:
            request_id: The request ID to retrieve data for
            
        Returns:
            Dict containing the logged data, or None if not found
        """
        try:
            log_file = self.log_dir / f"{request_id}.json"
            
            if not log_file.exists():
                logger.warning(f"No observability log found for request {request_id}")
                return None
            
            with open(log_file, 'r') as f:
                data = json.load(f)
            
            logger.debug(f"Retrieved observability data for request {request_id}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve observability data for request {request_id}: {e}")
            return None
    
    def list_available_logs(self) -> List[str]:
        """
        List all available request IDs that have observability logs
        
        Returns:
            List of request IDs
        """
        try:
            log_files = list(self.log_dir.glob("*.json"))
            request_ids = [f.stem for f in log_files]
            
            # Sort by modification time (newest first)
            request_ids.sort(key=lambda req_id: (self.log_dir / f"{req_id}.json").stat().st_mtime, reverse=True)
            
            logger.debug(f"Found {len(request_ids)} observability logs")
            return request_ids
            
        except Exception as e:
            logger.error(f"Failed to list available observability logs: {e}")
            return []
    
    def delete_log(self, request_id: str) -> bool:
        """
        Delete observability log for a specific request ID
        
        Args:
            request_id: The request ID to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            log_file = self.log_dir / f"{request_id}.json"
            
            if not log_file.exists():
                logger.warning(f"No observability log found for request {request_id}")
                return False
            
            log_file.unlink()
            logger.info(f"Deleted observability log for request {request_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete observability log for request {request_id}: {e}")
            return False
    
    def cleanup_old_logs(self, max_logs: int = 100) -> int:
        """
        Clean up old observability logs, keeping only the most recent ones
        
        Args:
            max_logs: Maximum number of logs to keep
            
        Returns:
            int: Number of logs deleted
        """
        try:
            available_logs = self.list_available_logs()
            
            if len(available_logs) <= max_logs:
                return 0
            
            # Delete oldest logs
            logs_to_delete = available_logs[max_logs:]
            deleted_count = 0
            
            for request_id in logs_to_delete:
                if self.delete_log(request_id):
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old observability logs")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old observability logs: {e}")
            return 0


def generate_request_id() -> str:
    """Generate a unique request ID for observability tracking"""
    return f"req_{uuid.uuid4().hex[:12]}"


def extract_token_probabilities(outputs, tokenizer, top_k: int = 10) -> Dict[str, float]:
    """
    Extract top-k token probabilities from model outputs
    
    Args:
        outputs: Model generation outputs
        tokenizer: Tokenizer for decoding tokens
        top_k: Number of top tokens to extract
        
    Returns:
        Dict mapping tokens to their probabilities
    """
    try:
        if not hasattr(outputs, 'scores') or not outputs.scores:
            return {}
        
        # Get the last score tensor (for the final generated token)
        last_scores = outputs.scores[-1][0]  # Shape: [vocab_size]
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(last_scores, dim=-1)
        
        # Get top-k tokens and their probabilities
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(probabilities)))
        
        # Convert to dictionary
        token_probs = {}
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            try:
                token = tokenizer.decode([idx], skip_special_tokens=True)
                if token:  # Only include non-empty tokens
                    token_probs[token] = prob
            except Exception:
                # Skip tokens that can't be decoded
                continue
        
        return token_probs
        
    except Exception as e:
        logger.error(f"Failed to extract token probabilities: {e}")
        return {}


# Global observability logger instance
_observability_logger = None

def get_observability_logger() -> ObservabilityLogger:
    """Get global observability logger instance"""
    global _observability_logger
    if _observability_logger is None:
        _observability_logger = ObservabilityLogger()
    return _observability_logger 