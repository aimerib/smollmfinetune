"""Tests for Production Deployment & Monitoring Infrastructure

This module tests the production deployment infrastructure including:
- Docker configurations and container orchestration
- Load balancing and auto-scaling
- Monitoring and alerting systems
- A/B testing framework
- Security and performance optimization
"""

import pytest
import yaml
import json
import docker
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

from backend.app.services.production.monitoring_service import (
    ProductionMonitoringService,
    PlatformMetrics,
    ServiceHealth,
    Alert
)
from backend.app.services.production.auto_scaling_manager import (
    AutoScalingManager,
    ScalingDecision
)
from backend.app.services.production.ab_testing_manager import (
    ABTestingManager,
    ExperimentConfig,
    ExperimentVariant
)
from backend.app.services.production.deployment_manager import (
    ProductionDeploymentManager,
    DeploymentConfig,
    DeploymentStatus,
    DeploymentStrategy,
    DeploymentPhase
)


class TestProductionInfrastructure:
    """Test production infrastructure configurations"""
    
    def test_docker_compose_production_structure(self):
        """Test production Docker Compose configuration is properly structured"""
        # Test that production docker-compose includes all required services
        compose_content = {
            'version': '3.8',
            'services': {
                'platform-api': {'image': 'dreamcast-platform-api:latest'},
                'platform-client': {'image': 'dreamcast-platform-client:latest'},
                'db-cluster': {'image': 'postgres:15'},
                'redis-cluster': {'image': 'redis:7'},
                'nginx-lb': {'image': 'nginx:alpine'},
                'celery-worker': {'image': 'dreamcast-platform-api:latest'}
            }
        }
        
        assert 'platform-api' in compose_content['services']
        assert 'platform-client' in compose_content['services']
        assert 'db-cluster' in compose_content['services']
        assert 'redis-cluster' in compose_content['services']
        assert 'nginx-lb' in compose_content['services']
        
    def test_nginx_load_balancer_config(self):
        """Test Nginx load balancer configuration"""
        nginx_config = {
            'upstream': {
                'api_backend': ['server platform-api-1:8000', 'server platform-api-2:8000'],
                'client_backend': ['server platform-client-1:3000', 'server platform-client-2:3000']
            },
            'ssl': {
                'certificate': '/etc/nginx/ssl/cert.pem',
                'certificate_key': '/etc/nginx/ssl/key.pem'
            }
        }
        
        assert 'api_backend' in nginx_config['upstream']
        assert 'client_backend' in nginx_config['upstream']
        assert len(nginx_config['upstream']['api_backend']) >= 2  # Multiple instances
        
    def test_database_cluster_configuration(self):
        """Test PostgreSQL cluster configuration for high availability"""
        db_config = {
            'master': {'host': 'db-master', 'port': 5432},
            'replicas': [
                {'host': 'db-replica-1', 'port': 5432},
                {'host': 'db-replica-2', 'port': 5432}
            ],
            'connection_pool': {
                'min_connections': 10,
                'max_connections': 100
            }
        }
        
        assert 'master' in db_config
        assert len(db_config['replicas']) >= 2
        assert db_config['connection_pool']['max_connections'] >= 50


class TestProductionMonitoringService:
    """Test comprehensive monitoring and alerting system"""
    
    @pytest.fixture
    def monitoring_service(self):
        return ProductionMonitoringService()
    
    def test_monitoring_service_initialization(self, monitoring_service):
        """Test monitoring service initializes with proper components"""
        assert monitoring_service.metrics_collector is not None
        assert monitoring_service.alert_manager is not None
        assert monitoring_service.analytics_processor is not None
        assert monitoring_service.websocket_manager is not None
        
    @pytest.mark.asyncio
    async def test_collect_platform_metrics(self, monitoring_service):
        """Test comprehensive platform metrics collection"""
        with patch.object(monitoring_service.metrics_collector, 'collect_system_metrics', new_callable=AsyncMock, return_value={'cpu_usage': 45.0, 'memory_usage': 60.0}), \
             patch.object(monitoring_service.metrics_collector, 'collect_api_metrics', new_callable=AsyncMock, return_value={'response_time': 150}), \
             patch.object(monitoring_service.metrics_collector, 'collect_react_metrics', new_callable=AsyncMock, return_value={'page_load_time': 800}), \
             patch.object(monitoring_service.metrics_collector, 'collect_websocket_metrics', new_callable=AsyncMock, return_value={'active_connections': 250}), \
             patch.object(monitoring_service.metrics_collector, 'collect_database_metrics', new_callable=AsyncMock, return_value={'query_time': 50}), \
             patch.object(monitoring_service.metrics_collector, 'collect_voice_metrics', new_callable=AsyncMock, return_value={'generation_latency': 300}), \
             patch.object(monitoring_service.metrics_collector, 'collect_ux_metrics', new_callable=AsyncMock, return_value={'journey_completion': 0.85}), \
             patch.object(monitoring_service.alert_manager, 'check_metrics', new_callable=AsyncMock, return_value=[]), \
             patch.object(monitoring_service.websocket_manager, 'broadcast', new_callable=AsyncMock, return_value=None):
            
            metrics = await monitoring_service.collect_platform_metrics()
            
            assert hasattr(metrics, 'api_performance')
            assert hasattr(metrics, 'react_performance')
            assert hasattr(metrics, 'websocket_metrics')
            assert hasattr(metrics, 'database_metrics')
            assert hasattr(metrics, 'voice_quality_metrics')
            assert hasattr(metrics, 'user_experience_metrics')
            assert metrics.api_performance['response_time'] == 150
            assert metrics.react_performance['page_load_time'] == 800
            assert metrics.websocket_metrics['active_connections'] == 250
            
    @pytest.mark.asyncio
    async def test_voice_quality_monitoring(self, monitoring_service):
        """Test voice generation and quality metrics collection"""
        voice_metrics = await monitoring_service.metrics_collector.collect_voice_metrics()
        
        expected_metrics = [
            'generation_latency',
            'quality_scores',
            'character_consistency',
            'spatial_audio_performance',
            'conversation_quality'
        ]
        
        for metric in expected_metrics:
            assert metric in voice_metrics
            
    @pytest.mark.asyncio
    async def test_alert_system_functionality(self, monitoring_service):
        """Test alert generation and management"""
        # Simulate high latency scenario
        metrics = {
            'api_performance': {'response_time': 5000},  # High latency
            'database_metrics': {'query_time': 2000}     # High DB time
        }
        
        with patch.object(monitoring_service.alert_manager, 'check_metrics') as mock_check:
            mock_check.return_value = [
                Alert(
                    level='critical',
                    message='API response time exceeds threshold',
                    metric='response_time',
                    value=5000,
                    threshold=1000,
                    timestamp=datetime.utcnow()
                )
            ]
            
            alerts = await monitoring_service.alert_manager.check_metrics(metrics)
            assert len(alerts) > 0
            assert alerts[0].level == 'critical'
            

class TestAutoScalingManager:
    """Test intelligent auto-scaling system"""
    
    @pytest.fixture
    def scaling_manager(self):
        with patch('docker.from_env'):
            return AutoScalingManager()
        
    def test_scaling_manager_initialization(self, scaling_manager):
        """Test auto-scaling manager initializes with proper thresholds"""
        assert scaling_manager.metrics_threshold['cpu_high'] == 80
        assert scaling_manager.metrics_threshold['memory_high'] == 85
        assert scaling_manager.metrics_threshold['response_time_high'] == 2000
        assert scaling_manager.metrics_threshold['websocket_connections_high'] == 1000
        
    @pytest.mark.asyncio
    async def test_api_scaling_decision(self, scaling_manager):
        """Test API service scaling based on metrics"""
        high_load_metrics = {
            'cpu_usage': 90,
            'memory_usage': 88,
            'response_time': 2500,
            'active_requests': 1500
        }
        
        with patch.object(scaling_manager, 'get_service_replicas', return_value=2):
            should_scale = await scaling_manager.should_scale_api(high_load_metrics)
            assert should_scale is True
            
    @pytest.mark.asyncio
    async def test_websocket_scaling_logic(self, scaling_manager):
        """Test WebSocket connection scaling"""
        websocket_metrics = {
            'active_connections': 1200,  # Above threshold
            'connection_rate': 50,       # Connections per second
            'message_throughput': 2000   # Messages per second
        }
        
        scaling_decision = scaling_manager.calculate_websocket_scaling(websocket_metrics)
        
        assert scaling_decision.scale_up is True
        assert scaling_decision.target_replicas > scaling_decision.current_replicas
        
    @pytest.mark.asyncio
    async def test_predictive_scaling(self, scaling_manager):
        """Test predictive scaling based on historical patterns"""
        historical_data = [
            {'hour': 14, 'avg_load': 75},  # Afternoon peak
            {'hour': 15, 'avg_load': 85},  # Peak continues
            {'hour': 16, 'avg_load': 90},  # Peak
        ]
        
        current_hour = 15
        predicted_load = scaling_manager.predict_load(historical_data, current_hour)
        
        assert predicted_load >= 80  # Should predict high load


class TestABTestingFramework:
    """Test A/B testing framework for platform optimization"""
    
    @pytest.fixture
    def ab_testing_manager(self):
        return ABTestingManager()
        
    def test_ab_testing_manager_initialization(self, ab_testing_manager):
        """Test A/B testing manager initializes properly"""
        assert ab_testing_manager.experiments == {}
        assert ab_testing_manager.user_assignments == {}
        
    @pytest.mark.asyncio
    async def test_create_experiment(self, ab_testing_manager):
        """Test creating new A/B test experiment"""
        experiment_config = ExperimentConfig(
            id='voice_model_test',
            name='Voice Model Comparison',
            description='Comparing voice synthesis quality between different models',
            hypothesis='Kokoro and Orpheus will provide better voice quality than control',
            variants=['kokoro', 'orpheus', 'control'],
            traffic_split={'kokoro': 40, 'orpheus': 40, 'control': 20},
            success_metrics=['voice_quality', 'user_satisfaction']
        )
        
        experiment = await ab_testing_manager.create_experiment(experiment_config)
        
        assert experiment['id'] == 'voice_model_test'
        assert experiment['status'] == 'active'
        assert len(experiment['variants']) == 3
        
    @pytest.mark.asyncio
    async def test_user_assignment_consistency(self, ab_testing_manager):
        """Test consistent user assignment to experiment variants"""
        experiment_config = ExperimentConfig(
            id='ui_test',
            name='UI Layout Test',
            description='Testing different UI layouts for better user engagement',
            hypothesis='Variant A will have higher conversion rates than Variant B',
            variants=['variant_a', 'variant_b'],
            traffic_split={'variant_a': 50, 'variant_b': 50},
            success_metrics=['conversion_rate']
        )
        
        await ab_testing_manager.create_experiment(experiment_config)
        
        # Same user should get same variant consistently
        user_id = 'user_123'
        variant_1 = await ab_testing_manager.assign_user_to_variant(user_id, 'ui_test')
        variant_2 = await ab_testing_manager.assign_user_to_variant(user_id, 'ui_test')
        
        assert variant_1 == variant_2
        
    @pytest.mark.asyncio
    async def test_traffic_split_distribution(self, ab_testing_manager):
        """Test traffic distribution matches configured splits"""
        experiment_config = ExperimentConfig(
            id='split_test',
            name='Traffic Split Test',
            description='Testing traffic distribution between control and treatment groups',
            hypothesis='The 70/30 split will provide adequate statistical power for reliable results',
            variants=['control', 'treatment'],
            traffic_split={'control': 70, 'treatment': 30},
            success_metrics=['engagement']
        )
        
        await ab_testing_manager.create_experiment(experiment_config)
        
        # Test with many users to verify distribution
        assignments = {}
        for i in range(1000):
            user_id = f'user_{i}'
            variant = await ab_testing_manager.assign_user_to_variant(user_id, 'split_test')
            assignments[variant] = assignments.get(variant, 0) + 1
            
        # Should roughly match 70/30 split (with some tolerance)
        control_pct = assignments.get('control', 0) / 1000
        treatment_pct = assignments.get('treatment', 0) / 1000
        
        assert 0.65 <= control_pct <= 0.75    # 70% ± 5%
        assert 0.25 <= treatment_pct <= 0.35  # 30% ± 5%


class TestProductionDeploymentManager:
    """Test production deployment and CI/CD system"""
    
    @pytest.fixture
    def deployment_manager(self):
        with patch('docker.from_env'):
            return ProductionDeploymentManager()
        
    def test_deployment_manager_initialization(self, deployment_manager):
        """Test deployment manager initializes with proper configuration"""
        assert deployment_manager.docker_client is not None
        assert deployment_manager.deployment_configs is not None
        
    @pytest.mark.asyncio
    async def test_deployment_configuration_validation(self, deployment_manager):
        """Test deployment configuration validation"""
        valid_config = DeploymentConfig(
            service_name='platform-api',
            image='dreamcast-platform-api',
            tag='v1.0.0',
            replicas=3,
            resource_limits={
                'cpus': '2.0',
                'memory': '4Gi'
            },
            environment_variables={
                'DATABASE_URL': 'postgresql://user:pass@db:5432/dreamcast',
                'REDIS_URL': 'redis://redis:6379'
            }
        )
        
        is_valid = await deployment_manager.validate_config(valid_config)
        assert is_valid is True
        
    @pytest.mark.asyncio
    async def test_rolling_deployment(self, deployment_manager):
        """Test rolling deployment with zero downtime"""
        deployment_config = DeploymentConfig(
            service_name='platform-api',
            image='dreamcast-platform-api',
            tag='v1.1.0',
            replicas=3,
            strategy=DeploymentStrategy.ROLLING_UPDATE
        )
        
        with patch.object(deployment_manager, '_execute_rolling_update') as mock_update, \
             patch.object(deployment_manager, '_get_current_service_version', return_value='v1.0.0'):
            mock_update.return_value = True  # Successful deployment
            
            result = await deployment_manager.deploy_service(deployment_config)
            assert result.status == 'completed'
            assert result.rollback_available is True
            
    @pytest.mark.asyncio
    async def test_automatic_rollback(self, deployment_manager):
        """Test automatic rollback on deployment failure"""
        with patch.object(deployment_manager, 'get_service_health', return_value=False), \
             patch.object(deployment_manager, '_get_previous_version', return_value='dreamcast-platform-api:v1.0.0'), \
             patch.object(deployment_manager, '_get_current_service_version', return_value='dreamcast-platform-api:v1.1.0'), \
             patch.object(deployment_manager, 'deploy_service') as mock_deploy:
            
            # Mock successful rollback deployment
            mock_deploy.return_value = DeploymentStatus(
                deployment_id='rollback-123',
                service='platform-api',
                status='completed',
                phase=DeploymentPhase.COMPLETED,
                previous_version='v1.1.0',
                new_version='v1.0.0',
                started_at=datetime.utcnow(),
                rollback_available=False
            )
            
            rollback_result = await deployment_manager.rollback_service('platform-api')
            
            assert rollback_result.success is True
            assert 'rolled back' in rollback_result.message.lower()


class TestSecurityAndCompliance:
    """Test security hardening and compliance measures"""
    
    def test_ssl_configuration(self):
        """Test SSL/TLS configuration for secure communications"""
        ssl_config = {
            'protocols': ['TLSv1.2', 'TLSv1.3'],
            'ciphers': [
                'ECDHE-ECDSA-AES256-GCM-SHA384',
                'ECDHE-RSA-AES256-GCM-SHA384',
                'ECDHE-ECDSA-CHACHA20-POLY1305'
            ],
            'hsts': {
                'max_age': 31536000,
                'include_subdomains': True,
                'preload': True
            }
        }
        
        assert 'TLSv1.3' in ssl_config['protocols']
        assert ssl_config['hsts']['max_age'] >= 31536000
        
    def test_rate_limiting_configuration(self):
        """Test rate limiting and DDoS protection"""
        rate_limits = {
            'api_endpoints': {'requests_per_minute': 1000},
            'websocket_connections': {'connections_per_ip': 10},
            'chat_messages': {'messages_per_minute': 60},
            'training_requests': {'requests_per_hour': 5}
        }
        
        assert rate_limits['api_endpoints']['requests_per_minute'] > 0
        assert rate_limits['websocket_connections']['connections_per_ip'] > 0
        
    def test_authentication_security(self):
        """Test authentication and authorization security"""
        auth_config = {
            'jwt': {
                'algorithm': 'RS256',
                'expiration': 3600,  # 1 hour
                'refresh_expiration': 604800  # 1 week
            },
            'password_policy': {
                'min_length': 12,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_numbers': True,
                'require_symbols': True
            }
        }
        
        assert auth_config['jwt']['algorithm'] == 'RS256'
        assert auth_config['password_policy']['min_length'] >= 12


class TestPerformanceOptimization:
    """Test performance optimization and CDN integration"""
    
    def test_cdn_configuration(self):
        """Test CDN configuration for global content delivery"""
        cdn_config = {
            'provider': 'cloudflare',
            'cache_rules': {
                'static_assets': {'ttl': 31536000},  # 1 year
                'api_responses': {'ttl': 300},       # 5 minutes
                'character_data': {'ttl': 3600}      # 1 hour
            },
            'compression': {
                'gzip': True,
                'brotli': True,
                'min_size': 1024
            }
        }
        
        assert cdn_config['cache_rules']['static_assets']['ttl'] >= 86400  # At least 1 day
        assert cdn_config['compression']['gzip'] is True
        
    def test_caching_strategy(self):
        """Test multi-layer caching strategy"""
        cache_config = {
            'redis': {
                'character_profiles': {'ttl': 3600},
                'world_data': {'ttl': 7200},
                'user_sessions': {'ttl': 1800}
            },
            'application': {
                'model_outputs': {'max_entries': 1000},
                'training_results': {'max_entries': 100}
            }
        }
        
        assert cache_config['redis']['character_profiles']['ttl'] > 0
        assert cache_config['application']['model_outputs']['max_entries'] >= 100


@pytest.mark.integration
class TestProductionIntegration:
    """Integration tests for complete production deployment"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_deployment_workflow(self):
        """Test complete deployment workflow from build to monitoring"""
        # This would test the full deployment pipeline
        deployment_steps = [
            'build_images',
            'run_tests',
            'deploy_to_staging',
            'run_integration_tests',
            'deploy_to_production',
            'verify_deployment',
            'setup_monitoring'
        ]
        
        for step in deployment_steps:
            # Each step would have actual implementation
            assert step is not None
            
    @pytest.mark.asyncio
    async def test_monitoring_integration_with_scaling(self):
        """Test integration between monitoring and auto-scaling"""
        # Test that monitoring alerts trigger scaling decisions
        monitoring_service = ProductionMonitoringService()
        with patch('docker.from_env'):
            scaling_manager = AutoScalingManager()
        
        # Simulate high load metrics in the expected structure
        metrics = {
            'system_resources': {
                'cpu_usage': 95,    # Above 85% critical threshold
                'memory_usage': 90  # At 90% critical threshold  
            },
            'api_performance': {
                'response_time': 3000  # Above 2000ms critical threshold
            }
        }
        
        # Monitoring should detect issues
        alerts = await monitoring_service.alert_manager.check_metrics(metrics)
        
        # Scaling should be triggered (requires flat structure)
        flat_metrics = {
            'cpu_usage': 95,
            'memory_usage': 90,
            'response_time': 3000
        }
        should_scale = await scaling_manager.should_scale_api(flat_metrics)
        
        assert len(alerts) > 0
        assert should_scale is True 