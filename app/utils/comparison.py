import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import datetime
import asyncio

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
        max_tokens: int,
        generation_config: Dict[str, Any],
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate responses from multiple models for the same prompt with consistent seeding for fair comparison."""
        # Generate a consistent seed if none provided
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        logger.info(f"Using seed {seed} for fair comparison across {len(model_identifiers)} models")
        
        responses = {}
        for model_id in model_identifiers:
            try:
                response = self.inference_manager.generate_response(
                    model_path=model_id,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    system_prompt="",
                    seed=seed,  # Use the same seed for all models
                    **generation_config
                )
                responses[model_id] = response
                logger.debug(f"Generated response for {model_id} with seed {seed}")
            except Exception as e:
                logger.error(f"Error generating response for {model_id}: {e}")
                responses[model_id] = f"Error: {e}"
        return responses

    def get_comparison_metrics(self, model_identifiers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch and structure metrics for comparison radar chart."""
        metrics_data = {}
        for model_id in model_identifiers:
            metrics = self.inference_manager.get_model_metrics(model_id)
            
            # Add current_loss and character_consistency from the metrics
            processed_metrics = {
                'eval_loss': metrics.get('eval_loss'),
                'avg_consistency': metrics.get('avg_consistency'),
                'character_consistency': metrics.get('character_consistency'),
                'current_loss': metrics.get('current_loss'),
                'learning_rate': metrics.get('learning_rate'),
                'loss': metrics.get('loss'),
                'final_loss': metrics.get('current_loss') or metrics.get('loss'),  # Use current_loss as final_loss
                'elapsed_time': metrics.get('elapsed_time'),
                'step': metrics.get('step', 0),  # Actual steps completed for this checkpoint
                'epoch': metrics.get('epoch', 0)
            }
            metrics_data[model_id] = processed_metrics
        return metrics_data

    def get_training_run_summary(self, model_identifiers: List[str]) -> Dict[str, Any]:
        """
        Extract shared training configuration that's common across all checkpoints.
        This should be displayed in a summary, not cluttering the comparison charts.
        """
        # Find a trained model (not base model) to get shared config
        trained_models = [m for m in model_identifiers if not m.startswith(("Base:", "HuggingFaceTB/"))]
        
        if not trained_models:
            return {}
        
        # Get metadata from the first trained model
        first_model = trained_models[0]
        metadata = self.inference_manager.get_model_metadata(first_model)
        
        if not metadata:
            return {}
        
        # Extract shared configuration
        summary = {
            'base_model': metadata.get('base_model', 'Unknown'),
            'training_method': metadata.get('training_method', 'lora').upper(),
            'dataset_size': metadata.get('dataset_size', 0),
            'total_configured_steps': metadata.get('total_steps', 0),
            'lora_rank': metadata.get('lora_r', 0),
            'lora_alpha': metadata.get('lora_alpha', 0),
            'lora_dropout': metadata.get('lora_dropout', 0.1),
            'character_name': metadata.get('character_name', 'Unknown'),
            'use_dora': metadata.get('use_dora', False),
            'use_rslora': metadata.get('use_rslora', False),
            'target_modules': metadata.get('target_modules', [])
        }
        
        return summary

    def get_variable_metrics_for_charts(self, model_identifiers: List[str], enhanced_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Extract only metrics that actually vary between checkpoints for meaningful charts.
        """
        variable_metrics = {}
        
        for model_id in model_identifiers:
            metrics = enhanced_metrics.get(model_id, {})
            metadata = self.inference_manager.get_model_metadata(model_id)
            
            # Focus on metrics that actually change between checkpoints
            model_variable_metrics = {
                # Core performance metrics (vary by checkpoint)
                'training_loss': metrics.get('current_loss') or metrics.get('loss', 0) or 0,
                'validation_loss': metrics.get('eval_loss') or 0,  # ✅ FIX: Handle None values properly
                'character_consistency': metrics.get('character_consistency', 0) or 0,
                
                # Progress metrics (vary by checkpoint)
                'actual_steps_completed': metrics.get('step', 0) or 0,
                'epochs_completed': metrics.get('epoch', 0) or 0,
                'training_time_elapsed': (metrics.get('elapsed_time') or 0) / 60 if metrics.get('elapsed_time') else 0,  # Convert to minutes
                
                # Learning state (may vary with scheduling)
                'learning_rate_at_checkpoint': metrics.get('learning_rate', 0) or 0,
                
                # Enhanced character evaluation metrics
                'personality_consistency': metrics.get('personality_consistency', 0) or 0,
                'speech_style': metrics.get('speech_style', 0) or 0,
                'emotional_authenticity': metrics.get('emotional_authenticity', 0) or 0,
                'scenario_appropriateness': metrics.get('scenario_appropriateness', 0) or 0,
                
                # Model type for grouping
                'is_base_model': model_id.startswith(("Base:", "HuggingFaceTB/")),
                'is_checkpoint': "checkpoint-" in model_id.lower(),
                'is_final_model': not model_id.startswith(("Base:", "HuggingFaceTB/", "Checkpoint:"))
            }
            
            variable_metrics[model_id] = model_variable_metrics
        
        return variable_metrics

    async def evaluate_character_consistency_with_judge(
        self,
        character: Dict[str, Any],
        user_prompt: str,
        model_response: str,
        dataset_manager
    ) -> Dict[str, float]:
        """
        Use the personality engine model as a judge to evaluate character consistency.
        This provides much more sophisticated evaluation than rule-based metrics.
        """
        char_name = character.get('name', 'the character')
        
        # Create character card block for context
        from .character import CharacterManager
        char_manager = CharacterManager()
        character_definition = char_manager.make_card_block(character)
        
        # Define evaluation criteria with personality engine as judge
        evaluation_criteria = {
            'personality_consistency': {
                'description': 'personality traits and behavioral patterns',
                'weight': 0.25
            },
            'speech_style': {
                'description': 'speaking style, tone, and dialogue patterns',
                'weight': 0.20
            },
            'emotional_authenticity': {
                'description': 'emotional responses and reactions',
                'weight': 0.20
            },
            'character_voice': {
                'description': 'unique voice and mannerisms',
                'weight': 0.15
            },
            'scenario_appropriateness': {
                'description': 'appropriateness of response to the given scenario',
                'weight': 0.15
            },
            'immersion_quality': {
                'description': 'maintaining character immersion without breaking the fourth wall',
                'weight': 0.05
            }
        }
        
        scores = {}
        
        for criterion, details in evaluation_criteria.items():
            judge_prompt = f"""You are an expert character consistency evaluator and roleplay director. Your task is to objectively assess how well a character response matches their established definition.

CHARACTER DEFINITION:
{character_definition}

USER PROMPT: "{user_prompt}"

CHARACTER RESPONSE: "{model_response}"

EVALUATION TASK:
Rate how well this response demonstrates {details['description']} for {char_name}, based on their character definition above.

Consider:
- Does the response match the character's established personality traits?
- Is the speaking style consistent with their defined voice?
- Are the emotional reactions authentic to this character?
- Does it maintain character immersion?

Rate on a scale of 0.0 to 1.0 where:
- 0.0 = Completely inconsistent with character
- 0.3 = Poor consistency, major issues
- 0.5 = Moderate consistency, some issues
- 0.7 = Good consistency, minor issues  
- 0.9 = Excellent consistency
- 1.0 = Perfect character consistency

Respond with ONLY a decimal number between 0.0 and 1.0, nothing else."""

            try:
                # Use the dataset manager's client (personality engine) as judge
                judge_response = await dataset_manager.client.chat_complete(
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_tokens=10,
                    temperature=0.1,  # Low temperature for consistent evaluation
                    top_p=0.9
                )
                
                # Extract score from response
                score_text = judge_response.strip()
                try:
                    score = float(score_text)
                    score = max(0.0, min(1.0, score))  # Clamp to valid range
                except ValueError:
                    logger.warning(f"Invalid score format from judge: {score_text}")
                    score = 0.5  # Default moderate score
                    
                scores[criterion] = score
                
            except Exception as e:
                logger.error(f"Error evaluating {criterion}: {e}")
                scores[criterion] = 0.5  # Default moderate score
        
        # Calculate weighted overall score
        overall_score = sum(scores[criterion] * details['weight'] 
                          for criterion, details in evaluation_criteria.items())
        scores['overall_consistency'] = overall_score
        
        return scores

    async def get_enhanced_comparison_metrics(
        self,
        model_identifiers: List[str],
        character: Dict[str, Any],
        test_prompt: str,
        responses: Dict[str, str],
        dataset_manager
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Get enhanced metrics with proper organization:
        - Base metrics (from files)
        - Training run summary (shared config)
        - Variable metrics (for charts)
        """
        # Get base metrics
        base_metrics = self.get_comparison_metrics(model_identifiers)
        
        # Add real-time character consistency evaluation for each model
        for model_id in model_identifiers:
            if model_id in responses and not responses[model_id].startswith("Error:"):
                try:
                    # Use personality engine as judge for character consistency
                    consistency_scores = await self.evaluate_character_consistency_with_judge(
                        character, test_prompt, responses[model_id], dataset_manager
                    )
                    
                    # Add the enhanced character consistency metrics
                    base_metrics[model_id]['character_consistency'] = consistency_scores['overall_consistency']
                    base_metrics[model_id]['personality_consistency'] = consistency_scores.get('personality_consistency', 0.5)
                    base_metrics[model_id]['speech_style'] = consistency_scores.get('speech_style', 0.5)
                    base_metrics[model_id]['emotional_authenticity'] = consistency_scores.get('emotional_authenticity', 0.5)
                    base_metrics[model_id]['character_voice'] = consistency_scores.get('character_voice', 0.5)
                    base_metrics[model_id]['scenario_appropriateness'] = consistency_scores.get('scenario_appropriateness', 0.5)
                    base_metrics[model_id]['immersion_quality'] = consistency_scores.get('immersion_quality', 0.5)
                    
                    logger.info(f"Enhanced character consistency for {model_id}: {consistency_scores['overall_consistency']:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating character consistency for {model_id}: {e}")
                    # Keep original metrics without enhancement
        
        # Get training run summary (shared config)
        training_summary = self.get_training_run_summary(model_identifiers)
        
        # Get variable metrics for charts
        variable_metrics = self.get_variable_metrics_for_charts(model_identifiers, base_metrics)
        
        return base_metrics, training_summary, variable_metrics
        
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