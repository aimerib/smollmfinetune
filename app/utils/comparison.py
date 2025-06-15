import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import datetime

from .inference import InferenceManager

logger = logging.getLogger(__name__)

class ComparisonManager:
    """Manages comparison between different models and checkpoints."""

    def __init__(self, inference_manager: InferenceManager):
        self.inference_manager = inference_manager
        self.project_dir = Path("training_output")
        self.promotions_file = self.project_dir / "promotions.json"

    def compare_models_side_by_side(
        self,
        model_identifiers: List[str],
        prompt: str,
        generation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate responses from multiple models for the same prompt."""
        responses = {}
        for model_id in model_identifiers:
            try:
                response = self.inference_manager.generate_response(
                    model_path=model_id,
                    prompt=prompt,
                    **generation_config
                )
                responses[model_id] = response
            except Exception as e:
                logger.error(f"Error generating response for {model_id}: {e}")
                responses[model_id] = f"Error: {e}"
        return responses

    def get_comparison_metrics(self, model_identifiers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch and structure metrics for comparison radar chart."""
        metrics_data = {}
        for model_id in model_identifiers:
            metrics = self.inference_manager.get_model_metrics(model_id)
            # Normalize metrics for radar chart if needed
            # For now, we'll take them as-is
            metrics_data[model_id] = {
                'eval_loss': metrics.get('eval_loss'),
                'avg_consistency': metrics.get('avg_consistency'),
                'learning_rate': metrics.get('learning_rate'),
                'loss': metrics.get('loss')
            }
        return metrics_data
        
    def promote_checkpoint(self, character_name: str, checkpoint_id: str, reason: str = "") -> bool:
        """Mark a checkpoint as 'best' for a character, with human validation."""
        promotions = self._load_promotions()
        
        if character_name not in promotions:
            promotions[character_name] = {}
            
        promotions[character_name] = {
            'promoted_checkpoint': checkpoint_id,
            'reason': reason,
            'promoted_at': datetime.datetime.utcnow().isoformat()
        }
        
        return self._save_promotions(promotions)

    def get_promoted_checkpoint(self, character_name: str) -> Optional[str]:
        """Get the promoted checkpoint for a character."""
        promotions = self._load_promotions()
        return promotions.get(character_name, {}).get('promoted_checkpoint')

    def _load_promotions(self) -> Dict[str, Any]:
        """Load the promotions data from file."""
        if not self.promotions_file.exists():
            return {}
        with self.promotions_file.open('r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def _save_promotions(self, promotions: Dict[str, Any]) -> bool:
        """Save the promotions data to file."""
        try:
            with self.promotions_file.open('w') as f:
                json.dump(promotions, f, indent=4)
            return True
        except IOError:
            return False 