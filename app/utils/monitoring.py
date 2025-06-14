import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

# Optional imports for monitoring tools
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add pandas import for timestamp handling
try:
    import pandas as pd
except ImportError:
    # Fallback for timestamp handling
    import datetime
    class pd:
        class Timestamp:
            @staticmethod
            def now():
                return datetime.datetime.now()

logger = logging.getLogger(__name__)

class AdvancedMonitor:
    """Advanced monitoring integration for character LoRA training"""
    
    def __init__(self, 
                 character_name: str,
                 log_dir: str = "training_output/logs",
                 enable_tensorboard: bool = True,
                 enable_wandb: bool = False,
                 wandb_project: str = "character-lora-training",
                 wandb_entity: Optional[str] = None):
        
        self.character_name = character_name
        self.log_dir = Path(log_dir)
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        
        # Create log directory
        self.character_log_dir = self.log_dir / f"{character_name.replace(' ', '_')}"
        self.character_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard
        self.tb_writer = None
        if self.enable_tensorboard:
            try:
                self.tb_writer = SummaryWriter(log_dir=str(self.character_log_dir / "tensorboard"))
                logger.info(f"📊 TensorBoard logging enabled: {self.character_log_dir}/tensorboard")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.enable_tensorboard = False
        
        # Initialize Wandb
        self.wandb_run = None
        if self.enable_wandb:
            try:
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=f"character-{character_name}",
                    tags=["character-lora", "training"]
                )
                logger.info(f"🌐 Wandb logging enabled: {wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to initialize Wandb: {e}")
                self.enable_wandb = False
        
        # Track metrics history for local analysis
        self.metrics_history = []
    
    def log_training_config(self, config: Dict[str, Any], character: Dict[str, Any]):
        """Log training configuration and character info"""
        
        # Prepare metadata
        metadata = {
            'character_name': character.get('name', 'Unknown'),
            'character_description_length': len(character.get('description', '')),
            'has_personality': bool(character.get('personality', '')),
            'has_examples': bool(character.get('mes_example', '')),
            'training_config': config
        }
        
        # Save to local JSON
        with open(self.character_log_dir / "training_config.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Log to monitoring tools
        if self.enable_wandb and self.wandb_run:
            self.wandb_run.config.update(metadata)
        
        logger.info(f"📝 Training configuration logged for {self.character_name}")
    
    def log_metrics(self, step: int, metrics: Dict[str, Any]):
        """Log training metrics to all enabled monitoring tools"""
        
        # Add timestamp and step
        log_entry = {
            'step': step,
            'timestamp': pd.Timestamp.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(log_entry)
        
        # TensorBoard logging
        if self.enable_tensorboard and self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
        
        # Wandb logging
        if self.enable_wandb and self.wandb_run:
            wandb_metrics = {'step': step}
            for key, value in metrics.items():
                if isinstance(value, (int, float, str)):
                    wandb_metrics[key] = value
            self.wandb_run.log(wandb_metrics)
    
    def log_character_consistency_metrics(self, step: int, consistency_scores: List[Dict[str, float]]):
        """Log character consistency metrics with detailed breakdown"""
        
        if not consistency_scores:
            return
        
        # Aggregate scores
        aggregated = {}
        for key in consistency_scores[0].keys():
            values = [score[key] for score in consistency_scores]
            aggregated[f'consistency/{key}_mean'] = sum(values) / len(values)
            aggregated[f'consistency/{key}_min'] = min(values)
            aggregated[f'consistency/{key}_max'] = max(values)
            aggregated[f'consistency/{key}_std'] = (sum((x - aggregated[f'consistency/{key}_mean'])**2 for x in values) / len(values))**0.5
        
        self.log_metrics(step, aggregated)
        
        # Create detailed consistency report
        if step % 100 == 0:  # Every 100 steps
            self._save_consistency_report(step, consistency_scores)
    
    def log_training_health(self, step: int, health_status: Dict[str, Any]):
        """Log training health indicators"""
        
        health_metrics = {
            'health/status_numeric': {'healthy': 1.0, 'warning': 0.5, 'critical': 0.0}.get(health_status['status'], 0.0),
            'health/warning_count': len(health_status.get('warnings', [])),
            'health/avg_consistency': health_status.get('avg_consistency', 0),
            'health/steps_trained': health_status.get('steps_trained', 0)
        }
        
        self.log_metrics(step, health_metrics)
        
        # Log warnings as text
        if self.enable_wandb and self.wandb_run and health_status.get('warnings'):
            self.wandb_run.log({
                'health/warnings': '\n'.join(health_status['warnings']),
                'step': step
            })
    
    def log_dataset_stats(self, dataset_stats: Dict[str, Any]):
        """Log dataset statistics"""
        
        stats_to_log = {}
        for key, value in dataset_stats.items():
            if isinstance(value, (int, float)):
                stats_to_log[f'dataset/{key}'] = value
        
        # Log at step 0 (before training)
        self.log_metrics(0, stats_to_log)
        
        # Save detailed dataset analysis
        with open(self.character_log_dir / "dataset_analysis.json", 'w') as f:
            json.dump(dataset_stats, f, indent=2)
    
    def log_sample_evaluation(self, step: int, sample_idx: int, evaluation: Dict[str, Any]):
        """Log individual sample evaluations"""
        
        # Only log a subset to avoid overwhelming logs
        if sample_idx % 10 == 0:  # Every 10th sample
            sample_metrics = {f'sample_eval/{key}': value for key, value in evaluation.items() if isinstance(value, (int, float))}
            self.log_metrics(step, sample_metrics)
    
    def create_loss_curve_plot(self):
        """Create and save loss curve visualization"""
        if not self.metrics_history:
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # Extract loss data
            steps = [entry['step'] for entry in self.metrics_history if 'loss' in entry]
            losses = [entry['loss'] for entry in self.metrics_history if 'loss' in entry]
            
            if not steps:
                return
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
            
            # Add consistency scores if available
            consistency_steps = [entry['step'] for entry in self.metrics_history if 'consistency/overall_consistency_mean' in entry]
            consistency_scores = [entry['consistency/overall_consistency_mean'] for entry in self.metrics_history if 'consistency/overall_consistency_mean' in entry]
            
            if consistency_steps:
                plt.twinx()
                plt.plot(consistency_steps, consistency_scores, 'r-', linewidth=2, alpha=0.7, label='Character Consistency')
                plt.ylabel('Character Consistency Score', color='red')
            
            plt.xlabel('Training Steps')
            plt.ylabel('Loss', color='blue')
            plt.title(f'Training Progress - {self.character_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = self.character_log_dir / "training_progress.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Log to wandb if enabled
            if self.enable_wandb and self.wandb_run:
                self.wandb_run.log({"training_progress": wandb.Image(str(plot_path))})
            
            logger.info(f"📈 Training progress plot saved: {plot_path}")
            
        except ImportError:
            logger.debug("Matplotlib not available for plotting")
        except Exception as e:
            logger.warning(f"Failed to create loss curve plot: {e}")
    
    def _save_consistency_report(self, step: int, consistency_scores: List[Dict[str, float]]):
        """Save detailed consistency analysis report"""
        
        report = {
            'step': step,
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_samples': len(consistency_scores),
            'detailed_scores': consistency_scores,
            'summary': {}
        }
        
        # Calculate summary statistics
        for key in consistency_scores[0].keys():
            values = [score[key] for score in consistency_scores]
            report['summary'][key] = {
                'mean': sum(values) / len(values),
                'median': sorted(values)[len(values)//2],
                'min': min(values),
                'max': max(values),
                'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
            }
        
        # Save report
        report_path = self.character_log_dir / f"consistency_report_step_{step}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def log_final_results(self, final_metrics: Dict[str, Any]):
        """Log final training results and create summary"""
        
        # Log final metrics
        self.log_metrics(final_metrics.get('final_step', 0), final_metrics)
        
        # Create training summary
        summary = {
            'character_name': self.character_name,
            'final_metrics': final_metrics,
            'total_steps': len(self.metrics_history),
            'training_completed': True
        }
        
        # Save summary
        with open(self.character_log_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create final plots
        self.create_loss_curve_plot()
        
        logger.info(f"✅ Final training results logged for {self.character_name}")
    
    def finish(self):
        """Clean up monitoring resources"""
        
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.enable_wandb and self.wandb_run:
            self.wandb_run.finish()
        
        logger.info(f"🏁 Monitoring finished for {self.character_name}") 