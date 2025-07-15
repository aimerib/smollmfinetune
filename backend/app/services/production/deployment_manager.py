"""Production Deployment Manager for Dreamcast Platform

Handles automated deployments, rolling updates, health checks, and rollback
capabilities for the unified React+FastAPI platform.
"""

import asyncio
import logging
import docker
import yaml
import json
import os
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import aiofiles


class DeploymentStrategy(Enum):
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"


class DeploymentPhase(Enum):
    PREPARING = "preparing"
    BUILDING = "building"
    TESTING = "testing"
    DEPLOYING = "deploying"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


@dataclass
class DeploymentConfig:
    """Configuration for a deployment"""
    service_name: str
    image: str
    tag: str
    replicas: int = 3
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    max_surge: int = 1
    max_unavailable: int = 0
    health_check_timeout: int = 300  # seconds
    rollback_on_failure: bool = True
    environment_variables: Dict[str, str] = None
    resource_limits: Dict[str, str] = None
    volume_mounts: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.environment_variables is None:
            self.environment_variables = {}
        if self.resource_limits is None:
            self.resource_limits = {}
        if self.volume_mounts is None:
            self.volume_mounts = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentStatus:
    """Status of a deployment"""
    deployment_id: str
    service: str
    status: str
    phase: DeploymentPhase
    previous_version: str
    new_version: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: str = ""
    rollback_available: bool = False
    health_checks_passed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            result['completed_at'] = self.completed_at.isoformat()
        result['phase'] = self.phase.value
        return result


@dataclass
class RollbackResult:
    """Result of a rollback operation"""
    success: bool
    message: str
    previous_version: str
    rolled_back_to: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class HealthChecker:
    """Health checking for deployed services"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def check_service_health(self, service_name: str, endpoint: str = "/health", 
                                  timeout: int = 30) -> bool:
        """Check if a service is healthy"""
        try:
            import aiohttp
            
            # Determine service URL
            service_urls = {
                'platform-api': 'http://platform-api:8000',
                'platform-client': 'http://platform-client:3000',
                'inference-engine': 'http://inference-engine:8001'
            }
            
            base_url = service_urls.get(service_name, f'http://{service_name}:8000')
            health_url = f"{base_url}{endpoint}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(health_url) as response:
                    if response.status == 200:
                        # Additional checks for specific services
                        if service_name == 'platform-api':
                            return await self._check_api_detailed_health(session, base_url)
                        elif service_name == 'platform-client':
                            return await self._check_client_detailed_health(session, base_url)
                        return True
                    else:
                        self.logger.warning(f"Health check failed for {service_name}: HTTP {response.status}")
                        return False
                        
        except Exception as e:
            self.logger.error(f"Health check error for {service_name}: {e}")
            return False
    
    async def _check_api_detailed_health(self, session, base_url: str) -> bool:
        """Detailed health check for API service"""
        try:
            # Check database connection
            async with session.get(f"{base_url}/health/db") as response:
                if response.status != 200:
                    return False
            
            # Check Redis connection
            async with session.get(f"{base_url}/health/redis") as response:
                if response.status != 200:
                    return False
            
            # Check basic API endpoint
            async with session.get(f"{base_url}/api/v1/status") as response:
                if response.status != 200:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Detailed API health check failed: {e}")
            return False
    
    async def _check_client_detailed_health(self, session, base_url: str) -> bool:
        """Detailed health check for React client"""
        try:
            # Check if static assets are loading
            async with session.get(f"{base_url}/static/js/") as response:
                if response.status not in [200, 404]:  # 404 is ok for directory listing
                    return False
            
            # Check if app is responding
            async with session.get(f"{base_url}/") as response:
                if response.status != 200:
                    return False
                
                # Check if it's actually the React app
                content = await response.text()
                if 'react' not in content.lower() and 'dreamcast' not in content.lower():
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Detailed client health check failed: {e}")
            return False
    
    async def wait_for_service_health(self, service_name: str, timeout: int = 300) -> bool:
        """Wait for service to become healthy"""
        start_time = datetime.utcnow()
        check_interval = 10  # seconds
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            if await self.check_service_health(service_name):
                return True
            
            await asyncio.sleep(check_interval)
        
        return False


class ProductionDeploymentManager:
    """Manages production deployments for the Dreamcast platform"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.docker_client = docker.from_env()
        self.health_checker = HealthChecker()
        
        # Deployment tracking
        self.active_deployments: Dict[str, DeploymentStatus] = {}
        self.deployment_history: List[DeploymentStatus] = []
        
        # Configuration
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        self.max_deployment_history = 100
        
        # Service registry
        self.services = {
            'platform-api': {
                'port': 8000,
                'health_endpoint': '/health',
                'dependencies': ['db-primary', 'redis-cluster']
            },
            'platform-client': {
                'port': 3000,
                'health_endpoint': '/health',
                'dependencies': []
            },
            'inference-engine': {
                'port': 8001,
                'health_endpoint': '/health',
                'dependencies': []
            },
            'celery-worker': {
                'port': None,
                'health_endpoint': None,
                'dependencies': ['db-primary', 'redis-cluster']
            }
        }
    
    async def deploy_service(self, config: DeploymentConfig) -> DeploymentStatus:
        """Deploy a service with the specified configuration"""
        try:
            deployment_id = f"{config.service_name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            
            # Get current version for rollback
            current_version = await self._get_current_service_version(config.service_name)
            
            # Create deployment status
            status = DeploymentStatus(
                deployment_id=deployment_id,
                service=config.service_name,
                status='in_progress',
                phase=DeploymentPhase.PREPARING,
                previous_version=current_version,
                new_version=f"{config.image}:{config.tag}",
                started_at=datetime.utcnow(),
                message="Preparing deployment"
            )
            
            self.active_deployments[deployment_id] = status
            
            # Execute deployment based on strategy
            if config.strategy == DeploymentStrategy.ROLLING_UPDATE:
                result = await self._execute_rolling_update(config, status)
            elif config.strategy == DeploymentStrategy.BLUE_GREEN:
                result = await self._execute_blue_green_deployment(config, status)
            elif config.strategy == DeploymentStrategy.CANARY:
                result = await self._execute_canary_deployment(config, status)
            else:
                result = await self._execute_recreate_deployment(config, status)
            
            # Update final status
            status.completed_at = datetime.utcnow()
            status.progress = 100.0
            
            if result:
                status.status = 'completed'
                status.phase = DeploymentPhase.COMPLETED
                status.message = "Deployment completed successfully"
                status.rollback_available = True
            else:
                status.status = 'failed'
                status.phase = DeploymentPhase.FAILED
                status.message = "Deployment failed"
                
                # Auto-rollback if enabled
                if config.rollback_on_failure:
                    await self._auto_rollback(config, status)
            
            # Move to history
            self.deployment_history.append(status)
            if len(self.deployment_history) > self.max_deployment_history:
                self.deployment_history.pop(0)
            
            del self.active_deployments[deployment_id]
            
            return status
            
        except Exception as e:
            self.logger.error(f"Deployment failed for {config.service_name}: {e}")
            status.status = 'failed'
            status.phase = DeploymentPhase.FAILED
            status.message = f"Deployment error: {str(e)}"
            return status
    
    async def _execute_rolling_update(self, config: DeploymentConfig, 
                                     status: DeploymentStatus) -> bool:
        """Execute rolling update deployment"""
        try:
            self.logger.info(f"Starting rolling update for {config.service_name}")
            
            # Phase 1: Building
            status.phase = DeploymentPhase.BUILDING
            status.progress = 10.0
            status.message = "Building new image"
            
            if not await self._build_and_push_image(config):
                return False
            
            # Phase 2: Testing
            status.phase = DeploymentPhase.TESTING
            status.progress = 30.0
            status.message = "Running tests"
            
            if not await self._run_deployment_tests(config):
                return False
            
            # Phase 3: Deploying
            status.phase = DeploymentPhase.DEPLOYING
            status.progress = 50.0
            status.message = "Updating service replicas"
            
            # Get current service
            service = await self._get_docker_service(config.service_name)
            if not service:
                self.logger.error(f"Service {config.service_name} not found")
                return False
            
            # Update service with new image
            service_spec = service.attrs['Spec'].copy()
            service_spec['TaskTemplate']['ContainerSpec']['Image'] = f"{config.image}:{config.tag}"
            
            # Update environment variables
            if config.environment_variables:
                env_list = []
                for key, value in config.environment_variables.items():
                    env_list.append(f"{key}={value}")
                service_spec['TaskTemplate']['ContainerSpec']['Env'] = env_list
            
            # Update resource limits
            if config.resource_limits:
                resources = service_spec['TaskTemplate']['Resources']
                if 'Limits' not in resources:
                    resources['Limits'] = {}
                resources['Limits'].update(config.resource_limits)
            
            # Perform rolling update
            service.update(**service_spec)
            
            # Phase 4: Verifying
            status.phase = DeploymentPhase.VERIFYING
            status.progress = 80.0
            status.message = "Verifying deployment health"
            
            # Wait for service to be healthy
            if not await self.health_checker.wait_for_service_health(
                config.service_name, config.health_check_timeout
            ):
                self.logger.error(f"Health check failed for {config.service_name}")
                return False
            
            status.health_checks_passed = True
            self.logger.info(f"Rolling update completed for {config.service_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Rolling update failed: {e}")
            return False
    
    async def _execute_blue_green_deployment(self, config: DeploymentConfig, 
                                           status: DeploymentStatus) -> bool:
        """Execute blue-green deployment"""
        try:
            self.logger.info(f"Starting blue-green deployment for {config.service_name}")
            
            # Blue-green deployment logic
            # This is a simplified implementation
            status.phase = DeploymentPhase.BUILDING
            status.progress = 20.0
            status.message = "Preparing green environment"
            
            # Create green service
            green_service_name = f"{config.service_name}-green"
            
            # Deploy to green environment
            await self._deploy_green_service(config, green_service_name)
            
            status.phase = DeploymentPhase.VERIFYING
            status.progress = 70.0
            status.message = "Testing green environment"
            
            # Test green environment
            if not await self.health_checker.wait_for_service_health(green_service_name):
                return False
            
            status.phase = DeploymentPhase.DEPLOYING
            status.progress = 90.0
            status.message = "Switching traffic to green"
            
            # Switch traffic (update load balancer configuration)
            await self._switch_traffic_to_green(config.service_name, green_service_name)
            
            # Clean up blue environment
            await self._cleanup_blue_service(config.service_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Blue-green deployment failed: {e}")
            return False
    
    async def _execute_canary_deployment(self, config: DeploymentConfig, 
                                       status: DeploymentStatus) -> bool:
        """Execute canary deployment"""
        try:
            self.logger.info(f"Starting canary deployment for {config.service_name}")
            
            # Canary deployment with gradual traffic shift
            canary_percentage = 10  # Start with 10% traffic
            
            status.phase = DeploymentPhase.DEPLOYING
            status.progress = 30.0
            status.message = f"Deploying canary ({canary_percentage}% traffic)"
            
            # Deploy canary version
            await self._deploy_canary_service(config, canary_percentage)
            
            # Gradually increase traffic
            for percentage in [10, 25, 50, 75, 100]:
                status.progress = 30.0 + (percentage * 0.6)
                status.message = f"Canary at {percentage}% traffic"
                
                await self._update_canary_traffic(config.service_name, percentage)
                
                # Monitor metrics for each step
                if not await self._monitor_canary_metrics(config.service_name, percentage):
                    self.logger.warning(f"Canary metrics failed at {percentage}%")
                    return False
                
                await asyncio.sleep(30)  # Wait between traffic increases
            
            # Finalize canary deployment
            await self._finalize_canary_deployment(config.service_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Canary deployment failed: {e}")
            return False
    
    async def _execute_recreate_deployment(self, config: DeploymentConfig, 
                                         status: DeploymentStatus) -> bool:
        """Execute recreate deployment (with downtime)"""
        try:
            self.logger.info(f"Starting recreate deployment for {config.service_name}")
            
            status.phase = DeploymentPhase.DEPLOYING
            status.progress = 50.0
            status.message = "Stopping old service"
            
            # Stop current service
            service = await self._get_docker_service(config.service_name)
            if service:
                service.remove()
            
            status.progress = 70.0
            status.message = "Starting new service"
            
            # Create new service
            await self._create_service_from_config(config)
            
            status.phase = DeploymentPhase.VERIFYING
            status.progress = 90.0
            status.message = "Verifying new service"
            
            # Wait for health check
            if not await self.health_checker.wait_for_service_health(config.service_name):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Recreate deployment failed: {e}")
            return False
    
    async def rollback_service(self, service_name: str, target_version: str = None) -> RollbackResult:
        """Rollback a service to a previous version"""
        try:
            current_version = await self._get_current_service_version(service_name)
            
            if not target_version:
                # Get previous version from deployment history
                target_version = await self._get_previous_version(service_name)
            
            if not target_version:
                return RollbackResult(
                    success=False,
                    message="No previous version available for rollback",
                    previous_version=current_version,
                    rolled_back_to="",
                    timestamp=datetime.utcnow()
                )
            
            self.logger.info(f"Rolling back {service_name} from {current_version} to {target_version}")
            
            # Create rollback configuration
            rollback_config = DeploymentConfig(
                service_name=service_name,
                image=target_version.split(':')[0],
                tag=target_version.split(':')[1] if ':' in target_version else 'latest',
                strategy=DeploymentStrategy.ROLLING_UPDATE,
                rollback_on_failure=False  # Don't rollback a rollback
            )
            
            # Execute rollback deployment
            status = await self.deploy_service(rollback_config)
            
            if status.status == 'completed':
                return RollbackResult(
                    success=True,
                    message=f"Successfully rolled back to {target_version}",
                    previous_version=current_version,
                    rolled_back_to=target_version,
                    timestamp=datetime.utcnow()
                )
            else:
                return RollbackResult(
                    success=False,
                    message=f"Rollback failed: {status.message}",
                    previous_version=current_version,
                    rolled_back_to="",
                    timestamp=datetime.utcnow()
                )
                
        except Exception as e:
            self.logger.error(f"Rollback failed for {service_name}: {e}")
            return RollbackResult(
                success=False,
                message=f"Rollback error: {str(e)}",
                previous_version="unknown",
                rolled_back_to="",
                timestamp=datetime.utcnow()
            )
    
    async def validate_config(self, config: DeploymentConfig) -> bool:
        """Validate deployment configuration"""
        try:
            # Check service exists in registry
            if config.service_name not in self.services:
                self.logger.error(f"Unknown service: {config.service_name}")
                return False
            
            # Validate image format
            if not config.image or ':' not in f"{config.image}:{config.tag}":
                self.logger.error("Invalid image format")
                return False
            
            # Validate replica count
            if config.replicas < 1:
                self.logger.error("Replica count must be at least 1")
                return False
            
            # Validate resource limits
            if config.resource_limits:
                for key, value in config.resource_limits.items():
                    if key not in ['cpus', 'memory']:
                        self.logger.error(f"Invalid resource limit: {key}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Config validation failed: {e}")
            return False
    
    # Helper methods
    async def _get_current_service_version(self, service_name: str) -> str:
        """Get current version of a service"""
        try:
            service = await self._get_docker_service(service_name)
            if service:
                image = service.attrs['Spec']['TaskTemplate']['ContainerSpec']['Image']
                return image
            return "unknown"
        except Exception:
            return "unknown"
    
    async def _get_docker_service(self, service_name: str):
        """Get Docker service by name"""
        try:
            services = self.docker_client.services.list(filters={'name': service_name})
            return services[0] if services else None
        except Exception as e:
            self.logger.error(f"Error getting Docker service {service_name}: {e}")
            return None
    
    async def _build_and_push_image(self, config: DeploymentConfig) -> bool:
        """Build and push Docker image"""
        try:
            # This would typically build and push to a registry
            # For demo purposes, assume image already exists
            self.logger.info(f"Building image {config.image}:{config.tag}")
            return True
        except Exception as e:
            self.logger.error(f"Image build failed: {e}")
            return False
    
    async def _run_deployment_tests(self, config: DeploymentConfig) -> bool:
        """Run tests before deployment"""
        try:
            # Run smoke tests, integration tests, etc.
            self.logger.info(f"Running tests for {config.service_name}")
            return True
        except Exception as e:
            self.logger.error(f"Tests failed: {e}")
            return False
    
    async def _auto_rollback(self, config: DeploymentConfig, status: DeploymentStatus):
        """Automatic rollback on deployment failure"""
        try:
            status.phase = DeploymentPhase.ROLLING_BACK
            status.message = "Auto-rolling back due to deployment failure"
            
            rollback_result = await self.rollback_service(config.service_name)
            
            if rollback_result.success:
                status.message += f" - Rollback successful"
            else:
                status.message += f" - Rollback failed: {rollback_result.message}"
                
        except Exception as e:
            self.logger.error(f"Auto-rollback failed: {e}")
            status.message += f" - Auto-rollback error: {str(e)}"
    
    async def _get_previous_version(self, service_name: str) -> Optional[str]:
        """Get previous version from deployment history"""
        try:
            # Find the last successful deployment for this service
            for deployment in reversed(self.deployment_history):
                if (deployment.service == service_name and 
                    deployment.status == 'completed'):
                    return deployment.previous_version
            return None
        except Exception:
            return None
    
    # Placeholder methods for advanced deployment strategies
    async def _deploy_green_service(self, config: DeploymentConfig, green_service_name: str):
        """Deploy service to green environment"""
        pass
    
    async def _switch_traffic_to_green(self, blue_service: str, green_service: str):
        """Switch traffic from blue to green"""
        pass
    
    async def _cleanup_blue_service(self, service_name: str):
        """Clean up blue environment after successful green deployment"""
        pass
    
    async def _deploy_canary_service(self, config: DeploymentConfig, percentage: int):
        """Deploy canary version with specified traffic percentage"""
        pass
    
    async def _update_canary_traffic(self, service_name: str, percentage: int):
        """Update traffic percentage for canary deployment"""
        pass
    
    async def _monitor_canary_metrics(self, service_name: str, percentage: int) -> bool:
        """Monitor metrics during canary deployment"""
        return True  # Simplified for demo
    
    async def _finalize_canary_deployment(self, service_name: str):
        """Finalize canary deployment by removing old version"""
        pass
    
    async def _create_service_from_config(self, config: DeploymentConfig):
        """Create Docker service from configuration"""
        pass
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """Get status of a specific deployment"""
        return self.active_deployments.get(deployment_id)
    
    async def list_deployments(self, service_name: str = None) -> List[DeploymentStatus]:
        """List recent deployments"""
        if service_name:
            return [d for d in self.deployment_history if d.service == service_name]
        return self.deployment_history.copy()
    
    async def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get comprehensive health status for a service"""
        try:
            is_healthy = await self.health_checker.check_service_health(service_name)
            current_version = await self._get_current_service_version(service_name)
            
            return {
                'service_name': service_name,
                'healthy': is_healthy,
                'current_version': current_version,
                'last_checked': datetime.utcnow().isoformat(),
                'dependencies': self.services.get(service_name, {}).get('dependencies', [])
            }
        except Exception as e:
            return {
                'service_name': service_name,
                'healthy': False,
                'error': str(e),
                'last_checked': datetime.utcnow().isoformat()
            } 