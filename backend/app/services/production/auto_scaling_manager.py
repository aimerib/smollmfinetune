"""Auto-Scaling Manager for Dreamcast Platform

Provides intelligent auto-scaling capabilities based on CPU, memory, WebSocket connections,
and custom platform metrics. Includes predictive scaling and console-quality performance
optimization.
"""

import asyncio
import logging
import docker
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib


class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass
class ScalingDecision:
    """Represents a scaling decision"""
    service: str
    direction: ScalingDirection
    current_replicas: int
    target_replicas: int
    reason: str
    confidence: float
    scale_up: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        self.scale_up = self.direction == ScalingDirection.UP


@dataclass
class LoadPrediction:
    """Load prediction for a service"""
    service: str
    predicted_load: float
    confidence: float
    time_horizon: int  # minutes
    recommendation: ScalingDirection


class AutoScalingManager:
    """Intelligent auto-scaling manager for the Dreamcast platform"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.docker_client = docker.from_env()
        
        # Scaling thresholds
        self.metrics_threshold = {
            'cpu_high': 80,
            'cpu_low': 30,
            'memory_high': 85,
            'memory_low': 40,
            'response_time_high': 2000,  # ms
            'response_time_low': 500,    # ms
            'websocket_connections_high': 1000,
            'websocket_connections_low': 200,
            'error_rate_high': 0.05,    # 5%
            'queue_size_high': 50,
            'disk_usage_high': 85
        }
        
        # Scaling configuration
        self.min_replicas = {
            'platform-api': 2,
            'platform-client': 2,
            'inference-engine': 1,
            'celery-worker': 1
        }
        
        self.max_replicas = {
            'platform-api': 10,
            'platform-client': 6,
            'inference-engine': 4,
            'celery-worker': 8
        }
        
        # Cooldown periods (seconds)
        self.scale_up_cooldown = 300   # 5 minutes
        self.scale_down_cooldown = 600 # 10 minutes
        
        # Historical data for predictive scaling
        self.load_history: Dict[str, List[Dict]] = {
            'platform-api': [],
            'platform-client': [],
            'inference-engine': [],
            'celery-worker': []
        }
        
        # Last scaling actions
        self.last_scaling: Dict[str, datetime] = {}
        
        # Predictive models
        self.prediction_models: Dict[str, Any] = {}
        
    async def monitor_and_scale(self):
        """Main monitoring and scaling loop"""
        try:
            current_metrics = await self.get_current_metrics()
            
            # Check each service for scaling needs
            services = ['platform-api', 'platform-client', 'inference-engine', 'celery-worker']
            
            for service in services:
                # Check if scaling is needed
                should_scale = await self.evaluate_scaling_need(service, current_metrics)
                
                if should_scale:
                    scaling_decision = await self.make_scaling_decision(service, current_metrics)
                    
                    if scaling_decision.direction != ScalingDirection.NONE:
                        await self.execute_scaling(scaling_decision)
            
            # Update load history for predictive scaling
            await self.update_load_history(current_metrics)
            
        except Exception as e:
            self.logger.error(f"Error in auto-scaling monitor: {e}")
    
    async def evaluate_scaling_need(self, service: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate if a service needs scaling"""
        try:
            # Check cooldown period
            if not self._is_scaling_allowed(service):
                return False
            
            # Get service-specific metrics
            service_metrics = self._get_service_metrics(service, metrics)
            
            # Check various scaling triggers
            scale_triggers = [
                await self.should_scale_api(service_metrics) if service == 'platform-api' else False,
                await self.should_scale_client(service_metrics) if service == 'platform-client' else False,
                await self.should_scale_inference(service_metrics) if service == 'inference-engine' else False,
                await self.should_scale_workers(service_metrics) if service == 'celery-worker' else False
            ]
            
            return any(scale_triggers)
            
        except Exception as e:
            self.logger.error(f"Error evaluating scaling need for {service}: {e}")
            return False
    
    async def should_scale_api(self, metrics: Dict[str, Any]) -> bool:
        """Check if API service should scale"""
        try:
            cpu_usage = metrics.get('cpu_usage', 0)
            memory_usage = metrics.get('memory_usage', 0)
            response_time = metrics.get('response_time', 0)
            error_rate = metrics.get('error_rate', 0)
            active_requests = metrics.get('active_requests', 0)
            
            # Scale up conditions
            scale_up = (
                cpu_usage > self.metrics_threshold['cpu_high'] or
                memory_usage > self.metrics_threshold['memory_high'] or
                response_time > self.metrics_threshold['response_time_high'] or
                error_rate > self.metrics_threshold['error_rate_high'] or
                active_requests > 100
            )
            
            # Scale down conditions (all must be true)
            scale_down = (
                cpu_usage < self.metrics_threshold['cpu_low'] and
                memory_usage < self.metrics_threshold['memory_low'] and
                response_time < self.metrics_threshold['response_time_low'] and
                error_rate < 0.01 and  # Less than 1%
                active_requests < 20
            )
            
            return scale_up or scale_down
            
        except Exception as e:
            self.logger.error(f"Error checking API scaling: {e}")
            return False
    
    async def should_scale_client(self, metrics: Dict[str, Any]) -> bool:
        """Check if React client service should scale"""
        try:
            cpu_usage = metrics.get('cpu_usage', 0)
            memory_usage = metrics.get('memory_usage', 0)
            active_users = metrics.get('active_users', 0)
            page_load_time = metrics.get('page_load_time', 0)
            
            # Scale up conditions
            scale_up = (
                cpu_usage > self.metrics_threshold['cpu_high'] or
                memory_usage > self.metrics_threshold['memory_high'] or
                active_users > 500 or
                page_load_time > 3000  # 3 seconds
            )
            
            # Scale down conditions
            scale_down = (
                cpu_usage < self.metrics_threshold['cpu_low'] and
                memory_usage < self.metrics_threshold['memory_low'] and
                active_users < 50 and
                page_load_time < 1000  # 1 second
            )
            
            return scale_up or scale_down
            
        except Exception as e:
            self.logger.error(f"Error checking client scaling: {e}")
            return False
    
    async def should_scale_inference(self, metrics: Dict[str, Any]) -> bool:
        """Check if inference engine should scale"""
        try:
            cpu_usage = metrics.get('cpu_usage', 0)
            memory_usage = metrics.get('memory_usage', 0)
            gpu_usage = metrics.get('gpu_usage', 0)
            queue_size = metrics.get('queue_size', 0)
            inference_latency = metrics.get('inference_latency', 0)
            
            # Scale up conditions (more conservative for expensive inference)
            scale_up = (
                (cpu_usage > 90 or gpu_usage > 90) or
                memory_usage > 90 or
                queue_size > self.metrics_threshold['queue_size_high'] or
                inference_latency > 5000  # 5 seconds
            )
            
            # Scale down conditions (very conservative)
            scale_down = (
                cpu_usage < 20 and
                memory_usage < 30 and
                gpu_usage < 20 and
                queue_size < 5 and
                inference_latency < 1000  # 1 second
            )
            
            return scale_up or scale_down
            
        except Exception as e:
            self.logger.error(f"Error checking inference scaling: {e}")
            return False
    
    async def should_scale_workers(self, metrics: Dict[str, Any]) -> bool:
        """Check if Celery workers should scale"""
        try:
            cpu_usage = metrics.get('cpu_usage', 0)
            memory_usage = metrics.get('memory_usage', 0)
            queue_size = metrics.get('queue_size', 0)
            active_tasks = metrics.get('active_tasks', 0)
            failed_tasks = metrics.get('failed_tasks', 0)
            
            # Scale up conditions
            scale_up = (
                cpu_usage > self.metrics_threshold['cpu_high'] or
                memory_usage > self.metrics_threshold['memory_high'] or
                queue_size > 20 or
                active_tasks > 10 or
                failed_tasks > 5
            )
            
            # Scale down conditions
            scale_down = (
                cpu_usage < self.metrics_threshold['cpu_low'] and
                memory_usage < self.metrics_threshold['memory_low'] and
                queue_size < 2 and
                active_tasks < 2 and
                failed_tasks < 1
            )
            
            return scale_up or scale_down
            
        except Exception as e:
            self.logger.error(f"Error checking worker scaling: {e}")
            return False
    
    def calculate_websocket_scaling(self, websocket_metrics: Dict[str, Any]) -> ScalingDecision:
        """Calculate scaling decision for WebSocket connections"""
        try:
            active_connections = websocket_metrics.get('active_connections', 0)
            connection_rate = websocket_metrics.get('connection_rate', 0)
            message_throughput = websocket_metrics.get('message_throughput', 0)
            
            current_replicas = 3  # Default current replicas
            
            # Determine scaling direction
            if (active_connections > self.metrics_threshold['websocket_connections_high'] or
                connection_rate > 10 or message_throughput > 1000):
                
                target_replicas = min(current_replicas + 2, self.max_replicas.get('platform-api', 10))
                direction = ScalingDirection.UP
                reason = f"High WebSocket load: {active_connections} connections, {connection_rate} conn/s"
                
            elif (active_connections < self.metrics_threshold['websocket_connections_low'] and
                  connection_rate < 2 and message_throughput < 200):
                
                target_replicas = max(current_replicas - 1, self.min_replicas.get('platform-api', 2))
                direction = ScalingDirection.DOWN
                reason = f"Low WebSocket load: {active_connections} connections"
                
            else:
                target_replicas = current_replicas
                direction = ScalingDirection.NONE
                reason = "WebSocket load within normal range"
            
            confidence = 0.8  # WebSocket scaling confidence
            
            return ScalingDecision(
                service='platform-api',
                direction=direction,
                current_replicas=current_replicas,
                target_replicas=target_replicas,
                reason=reason,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating WebSocket scaling: {e}")
            return ScalingDecision(
                service='platform-api',
                direction=ScalingDirection.NONE,
                current_replicas=3,
                target_replicas=3,
                reason="Error in calculation",
                confidence=0.0
            )
    
    async def make_scaling_decision(self, service: str, metrics: Dict[str, Any]) -> ScalingDecision:
        """Make intelligent scaling decision for a service"""
        try:
            current_replicas = await self.get_service_replicas(service)
            service_metrics = self._get_service_metrics(service, metrics)
            
            # Get load prediction
            prediction = await self.predict_load(service, service_metrics)
            
            # Determine scaling direction and target
            direction, target_replicas, reason, confidence = await self._calculate_scaling_target(
                service, current_replicas, service_metrics, prediction
            )
            
            return ScalingDecision(
                service=service,
                direction=direction,
                current_replicas=current_replicas,
                target_replicas=target_replicas,
                reason=reason,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error making scaling decision for {service}: {e}")
            return ScalingDecision(
                service=service,
                direction=ScalingDirection.NONE,
                current_replicas=await self.get_service_replicas(service),
                target_replicas=await self.get_service_replicas(service),
                reason="Error in decision making",
                confidence=0.0
            )
    
    async def _calculate_scaling_target(self, service: str, current_replicas: int, 
                                       metrics: Dict[str, Any], prediction: LoadPrediction) -> tuple:
        """Calculate scaling target based on metrics and prediction"""
        try:
            cpu_usage = metrics.get('cpu_usage', 0)
            memory_usage = metrics.get('memory_usage', 0)
            
            # Base scaling calculation on resource usage
            if cpu_usage > self.metrics_threshold['cpu_high'] or memory_usage > self.metrics_threshold['memory_high']:
                # Scale up
                scale_factor = max(cpu_usage, memory_usage) / 70  # Target 70% usage
                target_replicas = min(
                    int(current_replicas * scale_factor),
                    self.max_replicas.get(service, 10)
                )
                direction = ScalingDirection.UP
                reason = f"High resource usage: CPU {cpu_usage}%, Memory {memory_usage}%"
                confidence = 0.9
                
            elif (cpu_usage < self.metrics_threshold['cpu_low'] and 
                  memory_usage < self.metrics_threshold['memory_low']):
                # Scale down
                target_replicas = max(
                    current_replicas - 1,
                    self.min_replicas.get(service, 1)
                )
                direction = ScalingDirection.DOWN
                reason = f"Low resource usage: CPU {cpu_usage}%, Memory {memory_usage}%"
                confidence = 0.7
                
            else:
                # No scaling needed based on current metrics
                target_replicas = current_replicas
                direction = ScalingDirection.NONE
                reason = "Resource usage within normal range"
                confidence = 0.8
            
            # Consider prediction if available
            if prediction and prediction.confidence > 0.7:
                if prediction.recommendation == ScalingDirection.UP and direction != ScalingDirection.DOWN:
                    direction = ScalingDirection.UP
                    target_replicas = min(target_replicas + 1, self.max_replicas.get(service, 10))
                    reason += f" + Predicted load increase: {prediction.predicted_load:.1f}"
                    confidence = min(confidence + 0.1, 1.0)
            
            return direction, target_replicas, reason, confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating scaling target: {e}")
            return ScalingDirection.NONE, current_replicas, "Calculation error", 0.0
    
    async def execute_scaling(self, decision: ScalingDecision):
        """Execute scaling decision"""
        try:
            if decision.direction == ScalingDirection.NONE:
                return
            
            self.logger.info(f"Executing scaling: {decision.service} from {decision.current_replicas} "
                           f"to {decision.target_replicas} replicas. Reason: {decision.reason}")
            
            # Update service replicas (this would interact with Docker Swarm or Kubernetes)
            await self.update_service_replicas(decision.service, decision.target_replicas)
            
            # Record scaling action
            self.last_scaling[decision.service] = decision.timestamp
            
            # Notify monitoring system
            await self.notify_scaling_event(decision)
            
        except Exception as e:
            self.logger.error(f"Error executing scaling for {decision.service}: {e}")
    
    async def update_service_replicas(self, service: str, target_replicas: int):
        """Update service replica count"""
        try:
            # This would interact with Docker Swarm or Kubernetes API
            # For now, simulate the action
            self.logger.info(f"Updating {service} to {target_replicas} replicas")
            
            # In real implementation:
            # service = self.docker_client.services.get(service)
            # service.update(mode=ServiceMode('replicated', replicas=target_replicas))
            
        except Exception as e:
            self.logger.error(f"Error updating service replicas: {e}")
            raise
    
    async def get_service_replicas(self, service: str) -> int:
        """Get current replica count for a service"""
        try:
            # This would query Docker Swarm or Kubernetes
            # For now, return default values
            defaults = {
                'platform-api': 3,
                'platform-client': 2,
                'inference-engine': 1,
                'celery-worker': 2
            }
            return defaults.get(service, 1)
            
        except Exception as e:
            self.logger.error(f"Error getting service replicas: {e}")
            return 1
    
    async def predict_load(self, service: str, current_metrics: Dict[str, Any]) -> Optional[LoadPrediction]:
        """Predict future load for a service"""
        try:
            if service not in self.load_history:
                return None
            
            history = self.load_history[service]
            
            if len(history) < 10:  # Need minimum data for prediction
                return None
            
            # Extract features and target
            features = []
            targets = []
            
            for i, record in enumerate(history[-30:]):  # Last 30 data points
                # Features: hour of day, cpu, memory, request rate
                hour = record['timestamp'].hour
                cpu = record.get('cpu_usage', 0)
                memory = record.get('memory_usage', 0)
                requests = record.get('requests_per_second', 0)
                
                features.append([hour, cpu, memory, requests])
                targets.append(cpu)  # Predict CPU usage as load indicator
            
            if len(features) < 5:
                return None
            
            # Train simple linear regression model
            model = LinearRegression()
            X = np.array(features)
            y = np.array(targets)
            model.fit(X, y)
            
            # Predict next data point
            current_hour = datetime.utcnow().hour
            current_cpu = current_metrics.get('cpu_usage', 0)
            current_memory = current_metrics.get('memory_usage', 0)
            current_requests = current_metrics.get('requests_per_second', 0)
            
            next_features = np.array([[current_hour + 1, current_cpu, current_memory, current_requests]])
            predicted_load = model.predict(next_features)[0]
            
            # Calculate confidence (simplified)
            confidence = max(0.5, min(0.9, 1.0 - abs(predicted_load - current_cpu) / 100))
            
            # Determine recommendation
            if predicted_load > 80:
                recommendation = ScalingDirection.UP
            elif predicted_load < 30:
                recommendation = ScalingDirection.DOWN
            else:
                recommendation = ScalingDirection.NONE
            
            return LoadPrediction(
                service=service,
                predicted_load=predicted_load,
                confidence=confidence,
                time_horizon=60,  # 1 hour
                recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting load for {service}: {e}")
            return None
    
    def predict_load(self, historical_data: List[Dict], current_hour: int) -> float:
        """Predict load based on historical patterns"""
        try:
            if len(historical_data) < 3:
                return 75.0  # Default prediction
            
            # Find similar hours in history
            similar_hours = [record for record in historical_data 
                           if abs(record['hour'] - current_hour) <= 1]
            
            if similar_hours:
                avg_load = sum(record['avg_load'] for record in similar_hours) / len(similar_hours)
                return avg_load
            
            # Fall back to overall average
            return sum(record['avg_load'] for record in historical_data) / len(historical_data)
            
        except Exception as e:
            self.logger.error(f"Error in load prediction: {e}")
            return 75.0
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system and service metrics"""
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Mock service metrics (in production, would collect from actual services)
            metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'response_time': 150.0,
                'error_rate': 0.02,
                'active_requests': 45,
                'requests_per_second': 23.4,
                'active_users': 156,
                'page_load_time': 800.0,
                'queue_size': 8,
                'active_tasks': 4,
                'failed_tasks': 0,
                'inference_latency': 300.0,
                'gpu_usage': 65.0,
                'active_connections': 250,
                'connection_rate': 3,
                'message_throughput': 45
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting current metrics: {e}")
            return {}
    
    async def update_load_history(self, metrics: Dict[str, Any]):
        """Update load history for predictive scaling"""
        try:
            timestamp = datetime.utcnow()
            
            for service in self.load_history:
                service_metrics = self._get_service_metrics(service, metrics)
                
                record = {
                    'timestamp': timestamp,
                    'cpu_usage': service_metrics.get('cpu_usage', 0),
                    'memory_usage': service_metrics.get('memory_usage', 0),
                    'requests_per_second': service_metrics.get('requests_per_second', 0),
                    'response_time': service_metrics.get('response_time', 0)
                }
                
                self.load_history[service].append(record)
                
                # Keep only last 100 records per service
                if len(self.load_history[service]) > 100:
                    self.load_history[service].pop(0)
                    
        except Exception as e:
            self.logger.error(f"Error updating load history: {e}")
    
    async def notify_scaling_event(self, decision: ScalingDecision):
        """Notify monitoring system of scaling event"""
        try:
            notification = {
                'type': 'scaling_event',
                'service': decision.service,
                'direction': decision.direction.value,
                'from_replicas': decision.current_replicas,
                'to_replicas': decision.target_replicas,
                'reason': decision.reason,
                'confidence': decision.confidence,
                'timestamp': decision.timestamp.isoformat()
            }
            
            # In production, would send to monitoring service
            self.logger.info(f"Scaling notification: {notification}")
            
        except Exception as e:
            self.logger.error(f"Error sending scaling notification: {e}")
    
    def _is_scaling_allowed(self, service: str) -> bool:
        """Check if scaling is allowed based on cooldown periods"""
        if service not in self.last_scaling:
            return True
        
        time_since_last = (datetime.utcnow() - self.last_scaling[service]).total_seconds()
        
        # Use shorter cooldown for scale up (more urgent)
        min_cooldown = min(self.scale_up_cooldown, self.scale_down_cooldown)
        
        return time_since_last >= min_cooldown
    
    def _get_service_metrics(self, service: str, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract service-specific metrics from all metrics"""
        # In a real implementation, this would extract service-specific metrics
        # For now, return the same metrics for all services
        return all_metrics.copy()
    
    async def start_auto_scaling(self, interval: int = 60):
        """Start the auto-scaling monitoring loop"""
        self.logger.info(f"Starting auto-scaling service (interval: {interval}s)")
        
        while True:
            try:
                await self.monitor_and_scale()
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(10)  # Short retry delay 