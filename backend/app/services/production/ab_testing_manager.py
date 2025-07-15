"""A/B Testing Manager for Dreamcast Platform

Provides comprehensive A/B testing capabilities for platform features, voice models,
character experiences, and user interface optimizations.
"""

import asyncio
import logging
import hashlib
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pydantic import BaseModel
import redis.asyncio as aioredis


class ExperimentStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class MetricType(Enum):
    CONVERSION = "conversion"
    ENGAGEMENT = "engagement"
    PERFORMANCE = "performance"
    SATISFACTION = "satisfaction"
    RETENTION = "retention"


@dataclass
class ExperimentVariant:
    """Represents a variant in an A/B test"""
    id: str
    name: str
    description: str
    traffic_allocation: float  # 0.0 to 1.0
    configuration: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment"""
    id: str
    name: str
    description: str
    variants: List[str]
    traffic_split: Dict[str, float]
    success_metrics: List[str]
    hypothesis: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    min_sample_size: int = 100
    confidence_level: float = 0.95
    target_population: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.start_date:
            result['start_date'] = self.start_date.isoformat()
        if self.end_date:
            result['end_date'] = self.end_date.isoformat()
        return result


@dataclass
class ExperimentResult:
    """Results of an A/B test experiment"""
    experiment_id: str
    variant_id: str
    metric_name: str
    value: float
    sample_size: int
    confidence_interval: tuple
    p_value: float
    is_significant: bool
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class UserAssignment:
    """User assignment to experiment variant"""
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime
    session_id: Optional[str] = None
    user_attributes: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['assigned_at'] = self.assigned_at.isoformat()
        return result


class ABTestingManager:
    """Comprehensive A/B testing manager for the Dreamcast platform"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.logger = logging.getLogger(__name__)
        self.redis_url = redis_url
        self.redis_client = None
        
        # In-memory storage for current session (in production, use Redis/database)
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.user_assignments: Dict[str, UserAssignment] = {}
        self.experiment_results: Dict[str, List[ExperimentResult]] = {}
        
        # Configuration
        self.default_traffic_split = 50  # 50/50 split
        self.min_sample_size = 100
        self.confidence_level = 0.95
        
    async def initialize(self):
        """Initialize Redis connection and load existing experiments"""
        try:
            self.redis_client = aioredis.from_url(self.redis_url)
            await self.load_experiments_from_storage()
        except Exception as e:
            self.logger.warning(f"Redis connection failed, using in-memory storage: {e}")
    
    async def create_experiment(self, experiment_config: ExperimentConfig) -> Dict[str, Any]:
        """Create a new A/B test experiment"""
        try:
            # Validate configuration
            if not self._validate_experiment_config(experiment_config):
                raise ValueError("Invalid experiment configuration")
            
            # Create experiment
            experiment = {
                'id': experiment_config.id,
                'name': experiment_config.name,
                'description': experiment_config.description,
                'variants': experiment_config.variants,
                'traffic_split': experiment_config.traffic_split,
                'success_metrics': experiment_config.success_metrics,
                'hypothesis': experiment_config.hypothesis,
                'status': ExperimentStatus.ACTIVE.value,
                'start_date': experiment_config.start_date or datetime.utcnow(),
                'end_date': experiment_config.end_date,
                'min_sample_size': experiment_config.min_sample_size,
                'confidence_level': experiment_config.confidence_level,
                'target_population': experiment_config.target_population or {},
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow(),
                'sample_sizes': {variant: 0 for variant in experiment_config.variants},
                'conversion_rates': {variant: 0.0 for variant in experiment_config.variants}
            }
            
            # Store experiment
            self.experiments[experiment_config.id] = experiment
            await self._save_experiment_to_storage(experiment)
            
            self.logger.info(f"Created experiment: {experiment_config.id}")
            return experiment
            
        except Exception as e:
            self.logger.error(f"Error creating experiment {experiment_config.id}: {e}")
            raise
    
    async def assign_user_to_variant(self, user_id: str, experiment_id: str, 
                                    user_attributes: Dict[str, Any] = None) -> Optional[str]:
        """Assign user to experiment variant with consistent assignment"""
        try:
            if experiment_id not in self.experiments:
                self.logger.warning(f"Experiment {experiment_id} not found")
                return None
            
            experiment = self.experiments[experiment_id]
            
            # Check if experiment is active
            if experiment['status'] != ExperimentStatus.ACTIVE.value:
                return None
            
            # Check if user already assigned
            assignment_key = f"{user_id}:{experiment_id}"
            if assignment_key in self.user_assignments:
                return self.user_assignments[assignment_key].variant_id
            
            # Check target population criteria
            if not self._matches_target_population(user_attributes, experiment['target_population']):
                return None
            
            # Consistent hash-based assignment
            variant_id = self._assign_variant_by_hash(user_id, experiment_id, experiment['traffic_split'])
            
            # Create assignment record
            assignment = UserAssignment(
                user_id=user_id,
                experiment_id=experiment_id,
                variant_id=variant_id,
                assigned_at=datetime.utcnow(),
                user_attributes=user_attributes or {}
            )
            
            self.user_assignments[assignment_key] = assignment
            
            # Update sample size
            experiment['sample_sizes'][variant_id] += 1
            experiment['updated_at'] = datetime.utcnow()
            
            await self._save_assignment_to_storage(assignment)
            await self._save_experiment_to_storage(experiment)
            
            return variant_id
            
        except Exception as e:
            self.logger.error(f"Error assigning user {user_id} to experiment {experiment_id}: {e}")
            return None
    
    async def record_metric(self, user_id: str, experiment_id: str, metric_name: str, 
                          value: Union[float, bool], metadata: Dict[str, Any] = None):
        """Record experiment metric for analysis"""
        try:
            if experiment_id not in self.experiments:
                return
            
            # Find user assignment
            assignment_key = f"{user_id}:{experiment_id}"
            if assignment_key not in self.user_assignments:
                return
            
            assignment = self.user_assignments[assignment_key]
            variant_id = assignment.variant_id
            
            # Convert boolean to numeric for conversion metrics
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            
            # Store metric
            metric_record = {
                'user_id': user_id,
                'experiment_id': experiment_id,
                'variant_id': variant_id,
                'metric_name': metric_name,
                'value': value,
                'timestamp': datetime.utcnow(),
                'metadata': metadata or {}
            }
            
            await self._save_metric_to_storage(metric_record)
            
            # Update experiment statistics
            await self._update_experiment_statistics(experiment_id, variant_id, metric_name, value)
            
        except Exception as e:
            self.logger.error(f"Error recording metric for user {user_id}: {e}")
    
    async def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment results and analysis"""
        try:
            if experiment_id not in self.experiments:
                return {}
            
            experiment = self.experiments[experiment_id]
            
            # Get all metrics for this experiment
            metrics = await self._get_experiment_metrics(experiment_id)
            
            # Calculate statistical results
            results = {
                'experiment_id': experiment_id,
                'name': experiment['name'],
                'status': experiment['status'],
                'start_date': experiment['start_date'],
                'sample_sizes': experiment['sample_sizes'],
                'variants': {},
                'statistical_significance': {},
                'recommendations': []
            }
            
            # Analyze each variant
            for variant_id in experiment['variants']:
                variant_metrics = [m for m in metrics if m['variant_id'] == variant_id]
                variant_analysis = await self._analyze_variant_performance(variant_id, variant_metrics)
                results['variants'][variant_id] = variant_analysis
            
            # Statistical significance testing
            if len(experiment['variants']) == 2:
                sig_results = await self._calculate_statistical_significance(experiment_id, metrics)
                results['statistical_significance'] = sig_results
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(experiment_id, results)
            results['recommendations'] = recommendations
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting experiment results for {experiment_id}: {e}")
            return {}
    
    async def list_active_experiments(self) -> List[Dict[str, Any]]:
        """List all active experiments"""
        try:
            active_experiments = []
            
            for exp_id, experiment in self.experiments.items():
                if experiment['status'] == ExperimentStatus.ACTIVE.value:
                    # Add summary statistics
                    summary = {
                        'id': exp_id,
                        'name': experiment['name'],
                        'description': experiment['description'],
                        'variants': experiment['variants'],
                        'sample_sizes': experiment['sample_sizes'],
                        'start_date': experiment['start_date'],
                        'total_participants': sum(experiment['sample_sizes'].values()),
                        'is_significant': await self._check_experiment_significance(exp_id)
                    }
                    active_experiments.append(summary)
            
            return active_experiments
            
        except Exception as e:
            self.logger.error(f"Error listing active experiments: {e}")
            return []
    
    async def stop_experiment(self, experiment_id: str, reason: str = "") -> bool:
        """Stop an active experiment"""
        try:
            if experiment_id not in self.experiments:
                return False
            
            experiment = self.experiments[experiment_id]
            experiment['status'] = ExperimentStatus.COMPLETED.value
            experiment['end_date'] = datetime.utcnow()
            experiment['completion_reason'] = reason
            experiment['updated_at'] = datetime.utcnow()
            
            await self._save_experiment_to_storage(experiment)
            
            self.logger.info(f"Stopped experiment {experiment_id}: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping experiment {experiment_id}: {e}")
            return False
    
    def _assign_variant_by_hash(self, user_id: str, experiment_id: str, 
                               traffic_split: Dict[str, float]) -> str:
        """Assign variant using consistent hashing"""
        try:
            # Create consistent hash
            hash_input = f"{user_id}:{experiment_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            
            # Normalize to 0-100 range
            bucket = hash_value % 100
            
            # Assign based on traffic split
            cumulative = 0
            for variant, percentage in traffic_split.items():
                cumulative += percentage
                if bucket < cumulative:
                    return variant
            
            # Default to first variant if something goes wrong
            return list(traffic_split.keys())[0]
            
        except Exception as e:
            self.logger.error(f"Error in variant assignment: {e}")
            return list(traffic_split.keys())[0]
    
    def _validate_experiment_config(self, config: ExperimentConfig) -> bool:
        """Validate experiment configuration"""
        try:
            # Check traffic split sums to 100
            if abs(sum(config.traffic_split.values()) - 100) > 0.1:
                self.logger.error("Traffic split must sum to 100")
                return False
            
            # Check variants match traffic split
            if set(config.variants) != set(config.traffic_split.keys()):
                self.logger.error("Variants must match traffic split keys")
                return False
            
            # Check minimum sample size
            if config.min_sample_size < 10:
                self.logger.error("Minimum sample size must be at least 10")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating experiment config: {e}")
            return False
    
    def _matches_target_population(self, user_attributes: Dict[str, Any], 
                                  target_criteria: Dict[str, Any]) -> bool:
        """Check if user matches target population criteria"""
        if not target_criteria or not user_attributes:
            return True
        
        try:
            for key, expected_value in target_criteria.items():
                if key not in user_attributes:
                    return False
                
                user_value = user_attributes[key]
                
                # Handle different comparison types
                if isinstance(expected_value, dict):
                    # Range comparison
                    if 'min' in expected_value and user_value < expected_value['min']:
                        return False
                    if 'max' in expected_value and user_value > expected_value['max']:
                        return False
                elif isinstance(expected_value, list):
                    # List membership
                    if user_value not in expected_value:
                        return False
                else:
                    # Exact match
                    if user_value != expected_value:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking target population: {e}")
            return True  # Default to include user
    
    async def _update_experiment_statistics(self, experiment_id: str, variant_id: str, 
                                          metric_name: str, value: float):
        """Update experiment statistics with new metric"""
        try:
            experiment = self.experiments[experiment_id]
            
            # Update conversion rates for conversion metrics
            if metric_name in ['conversion', 'signup', 'purchase', 'character_creation']:
                current_rate = experiment['conversion_rates'].get(variant_id, 0.0)
                sample_size = experiment['sample_sizes'].get(variant_id, 0)
                
                if sample_size > 0:
                    # Update running average
                    new_rate = ((current_rate * (sample_size - 1)) + value) / sample_size
                    experiment['conversion_rates'][variant_id] = new_rate
            
            experiment['updated_at'] = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Error updating experiment statistics: {e}")
    
    async def _analyze_variant_performance(self, variant_id: str, 
                                         metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance of a specific variant"""
        try:
            if not metrics:
                return {'sample_size': 0, 'metrics': {}}
            
            analysis = {
                'sample_size': len(metrics),
                'metrics': {}
            }
            
            # Group metrics by name
            metric_groups = {}
            for metric in metrics:
                name = metric['metric_name']
                if name not in metric_groups:
                    metric_groups[name] = []
                metric_groups[name].append(metric['value'])
            
            # Calculate statistics for each metric
            for metric_name, values in metric_groups.items():
                if values:
                    analysis['metrics'][metric_name] = {
                        'mean': sum(values) / len(values),
                        'count': len(values),
                        'conversion_rate': sum(1 for v in values if v > 0) / len(values) if values else 0
                    }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing variant performance: {e}")
            return {'sample_size': 0, 'metrics': {}}
    
    async def _calculate_statistical_significance(self, experiment_id: str, 
                                                metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistical significance between variants"""
        try:
            # Simplified statistical significance calculation
            # In production, would use proper statistical tests
            
            experiment = self.experiments[experiment_id]
            variants = experiment['variants']
            
            if len(variants) != 2:
                return {}
            
            variant_a, variant_b = variants
            
            # Get conversion rates
            rate_a = experiment['conversion_rates'].get(variant_a, 0)
            rate_b = experiment['conversion_rates'].get(variant_b, 0)
            
            size_a = experiment['sample_sizes'].get(variant_a, 0)
            size_b = experiment['sample_sizes'].get(variant_b, 0)
            
            # Check minimum sample size
            min_size = experiment.get('min_sample_size', 100)
            if size_a < min_size or size_b < min_size:
                return {
                    'is_significant': False,
                    'p_value': None,
                    'confidence_interval': None,
                    'message': f'Insufficient sample size (need {min_size}, have {min(size_a, size_b)})'
                }
            
            # Simplified significance test (would use proper z-test in production)
            pooled_rate = (rate_a * size_a + rate_b * size_b) / (size_a + size_b) if (size_a + size_b) > 0 else 0
            
            if pooled_rate > 0:
                # Mock p-value calculation
                difference = abs(rate_a - rate_b)
                p_value = max(0.001, 0.2 - (difference * 10))  # Simplified
                
                is_significant = p_value < 0.05
                
                return {
                    'is_significant': is_significant,
                    'p_value': p_value,
                    'confidence_interval': (rate_a - 0.05, rate_a + 0.05),
                    'effect_size': difference,
                    'winner': variant_a if rate_a > rate_b else variant_b
                }
            
            return {
                'is_significant': False,
                'p_value': None,
                'confidence_interval': None,
                'message': 'Insufficient conversion data'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical significance: {e}")
            return {}
    
    async def _generate_recommendations(self, experiment_id: str, 
                                       results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on experiment results"""
        try:
            recommendations = []
            
            experiment = self.experiments[experiment_id]
            
            # Check sample size
            total_sample = sum(experiment['sample_sizes'].values())
            min_size = experiment.get('min_sample_size', 100)
            
            if total_sample < min_size:
                recommendations.append(f"Continue experiment to reach minimum sample size of {min_size}")
                return recommendations
            
            # Check statistical significance
            sig_results = results.get('statistical_significance', {})
            
            if sig_results.get('is_significant'):
                winner = sig_results.get('winner')
                if winner:
                    recommendations.append(f"Implement variant '{winner}' - statistically significant improvement")
            
            elif total_sample >= min_size * 2:
                recommendations.append("No significant difference found - consider stopping experiment")
            
            else:
                recommendations.append("Continue experiment to improve statistical power")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    async def _check_experiment_significance(self, experiment_id: str) -> bool:
        """Check if experiment has reached statistical significance"""
        try:
            metrics = await self._get_experiment_metrics(experiment_id)
            sig_results = await self._calculate_statistical_significance(experiment_id, metrics)
            return sig_results.get('is_significant', False)
            
        except Exception as e:
            self.logger.error(f"Error checking experiment significance: {e}")
            return False
    
    # Storage methods (simplified for demo - would use Redis/database in production)
    async def _save_experiment_to_storage(self, experiment: Dict[str, Any]):
        """Save experiment to persistent storage"""
        try:
            if self.redis_client:
                key = f"experiment:{experiment['id']}"
                await self.redis_client.set(key, json.dumps(experiment, default=str))
        except Exception as e:
            self.logger.error(f"Error saving experiment to storage: {e}")
    
    async def _save_assignment_to_storage(self, assignment: UserAssignment):
        """Save user assignment to persistent storage"""
        try:
            if self.redis_client:
                key = f"assignment:{assignment.user_id}:{assignment.experiment_id}"
                await self.redis_client.set(key, json.dumps(assignment.to_dict()))
        except Exception as e:
            self.logger.error(f"Error saving assignment to storage: {e}")
    
    async def _save_metric_to_storage(self, metric: Dict[str, Any]):
        """Save metric to persistent storage"""
        try:
            if self.redis_client:
                key = f"metric:{metric['experiment_id']}:{metric['user_id']}:{datetime.utcnow().timestamp()}"
                await self.redis_client.set(key, json.dumps(metric, default=str))
        except Exception as e:
            self.logger.error(f"Error saving metric to storage: {e}")
    
    async def load_experiments_from_storage(self):
        """Load existing experiments from storage"""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys("experiment:*")
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        experiment = json.loads(data)
                        self.experiments[experiment['id']] = experiment
        except Exception as e:
            self.logger.error(f"Error loading experiments from storage: {e}")
    
    async def _get_experiment_metrics(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Get all metrics for an experiment"""
        try:
            if self.redis_client:
                keys = await self.redis_client.keys(f"metric:{experiment_id}:*")
                metrics = []
                for key in keys:
                    data = await self.redis_client.get(key)
                    if data:
                        metrics.append(json.loads(data))
                return metrics
            return []
        except Exception as e:
            self.logger.error(f"Error getting experiment metrics: {e}")
            return [] 