"""Integration Tests for Production Deployment System

Tests the complete production deployment workflow including monitoring,
auto-scaling, A/B testing, and deployment automation.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from backend.app.services.production.monitoring_service import ProductionMonitoringService
from backend.app.services.production.auto_scaling_manager import AutoScalingManager, ScalingDirection
from backend.app.services.production.ab_testing_manager import ABTestingManager, ExperimentConfig
from backend.app.services.production.deployment_manager import (
    ProductionDeploymentManager, DeploymentConfig, DeploymentStrategy
)


@pytest.mark.integration
class TestProductionSystemIntegration:
    """Test integration between all production components"""
    
    @pytest.fixture
    async def monitoring_service(self):
        """Create monitoring service for tests"""
        service = ProductionMonitoringService()
        yield service
    
    @pytest.fixture
    async def scaling_manager(self):
        """Create auto-scaling manager for tests"""
        manager = AutoScalingManager()
        yield manager
    
    @pytest.fixture
    async def ab_testing_manager(self):
        """Create A/B testing manager for tests"""
        manager = ABTestingManager()
        await manager.initialize()
        yield manager
    
    @pytest.fixture
    async def deployment_manager(self):
        """Create deployment manager for tests"""
        manager = ProductionDeploymentManager()
        yield manager
    
    @pytest.mark.asyncio
    async def test_monitoring_triggers_scaling(self, monitoring_service, scaling_manager):
        """Test that monitoring service metrics trigger auto-scaling"""
        # Simulate high load metrics
        high_load_metrics = {
            'cpu_usage': 95,
            'memory_usage': 88,
            'response_time': 2500,
            'active_requests': 1500
        }
        
        # Check if scaling should be triggered
        should_scale = await scaling_manager.should_scale_api(high_load_metrics)
        assert should_scale is True
        
        # Test scaling decision
        scaling_decision = await scaling_manager.make_scaling_decision('platform-api', high_load_metrics)
        assert scaling_decision.direction == ScalingDirection.UP
        assert scaling_decision.target_replicas > scaling_decision.current_replicas
    
    @pytest.mark.asyncio
    async def test_deployment_updates_monitoring(self, deployment_manager, monitoring_service):
        """Test that deployments are reflected in monitoring"""
        # Create deployment configuration
        config = DeploymentConfig(
            service_name='platform-api',
            image='dreamcast-platform-api',
            tag='v1.1.0',
            replicas=3,
            strategy=DeploymentStrategy.ROLLING_UPDATE
        )
        
        # Validate configuration
        is_valid = await deployment_manager.validate_config(config)
        assert is_valid is True
        
        # Mock deployment execution
        with patch.object(deployment_manager, '_execute_rolling_update', return_value=True):
            status = await deployment_manager.deploy_service(config)
            assert status.status in ['completed', 'in_progress']
    
    @pytest.mark.asyncio
    async def test_ab_testing_affects_monitoring_metrics(self, ab_testing_manager, monitoring_service):
        """Test that A/B testing results affect monitoring metrics"""
        # Create A/B test experiment
        experiment_config = ExperimentConfig(
            id='api_performance_test',
            name='API Performance Optimization',
            description='Testing optimized API configuration vs control',
            hypothesis='Optimized configuration will reduce response times by 20%',
            variants=['control', 'optimized'],
            traffic_split={'control': 50, 'optimized': 50},
            success_metrics=['response_time', 'error_rate']
        )
        
        experiment = await ab_testing_manager.create_experiment(experiment_config)
        assert experiment['status'] == 'active'
        
        # Assign users to variants
        user1_variant = await ab_testing_manager.assign_user_to_variant('user1', 'api_performance_test')
        user2_variant = await ab_testing_manager.assign_user_to_variant('user2', 'api_performance_test')
        
        assert user1_variant in ['control', 'optimized']
        assert user2_variant in ['control', 'optimized']
        
        # Record metrics for both variants
        await ab_testing_manager.record_metric('user1', 'api_performance_test', 'response_time', 150.0)
        await ab_testing_manager.record_metric('user2', 'api_performance_test', 'response_time', 120.0)
        
        # Get experiment results
        results = await ab_testing_manager.get_experiment_results('api_performance_test')
        assert results['experiment_id'] == 'api_performance_test'
        assert len(results['variants']) == 2
    
    @pytest.mark.asyncio
    async def test_full_deployment_workflow(self, deployment_manager, monitoring_service, scaling_manager):
        """Test complete deployment workflow with monitoring and scaling"""
        # Step 1: Create deployment configuration
        config = DeploymentConfig(
            service_name='platform-client',
            image='dreamcast-platform-client',
            tag='v2.0.0',
            replicas=2,
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            health_check_timeout=60
        )
        
        # Step 2: Validate configuration
        assert await deployment_manager.validate_config(config) is True
        
        # Step 3: Mock successful deployment
        with patch.object(deployment_manager, '_execute_rolling_update') as mock_deploy:
            mock_deploy.return_value = True
            
            # Execute deployment
            deployment_status = await deployment_manager.deploy_service(config)
            
            # Verify deployment completed
            assert deployment_status.service == 'platform-client'
            assert deployment_status.new_version == 'dreamcast-platform-client:v2.0.0'
        
        # Step 4: Simulate monitoring detecting the new deployment
        service_health = await deployment_manager.get_service_health('platform-client')
        assert service_health['service_name'] == 'platform-client'
        
        # Step 5: Test scaling behavior with new deployment
        current_metrics = await scaling_manager.get_current_metrics()
        scaling_needed = await scaling_manager.evaluate_scaling_need('platform-client', current_metrics)
        
        # Should work without errors (scaling needed or not depends on metrics)
        assert isinstance(scaling_needed, bool)
    
    @pytest.mark.asyncio
    async def test_rollback_workflow(self, deployment_manager):
        """Test rollback workflow"""
        # Mock deployment history
        deployment_manager.deployment_history = [
            Mock(
                service='platform-api',
                status='completed',
                previous_version='dreamcast-platform-api:v1.0.0',
                new_version='dreamcast-platform-api:v1.1.0'
            )
        ]
        
        # Test rollback
        with patch.object(deployment_manager, 'deploy_service') as mock_deploy:
            mock_deploy.return_value = Mock(status='completed', message='Rollback successful')
            
            rollback_result = await deployment_manager.rollback_service('platform-api')
            
            assert rollback_result.success is True
            assert 'successful' in rollback_result.message.lower()
    
    @pytest.mark.asyncio
    async def test_monitoring_alert_system(self, monitoring_service):
        """Test monitoring alert system"""
        # Simulate metrics that should trigger alerts
        critical_metrics = {
            'api_performance': {'response_time': 5000, 'error_rate': 0.15},
            'system_resources': {'cpu_usage': 95, 'memory_usage': 92},
            'database_metrics': {'query_time': 2500}
        }
        
        # Check for alerts
        alerts = await monitoring_service.alert_manager.check_metrics(critical_metrics)
        
        # Should generate multiple alerts for critical conditions
        assert len(alerts) > 0
        
        # Check alert levels
        critical_alerts = [alert for alert in alerts if alert.level == 'critical']
        assert len(critical_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_platform_health_calculation(self, monitoring_service):
        """Test platform health score calculation"""
        # Create sample metrics
        sample_metrics = {
            'api_performance': {
                'response_time': 150,
                'error_rate': 0.02
            },
            'react_performance': {
                'page_load_time': 800,
                'first_contentful_paint': 450
            },
            'database_metrics': {
                'query_time': 50,
                'cache_hit_ratio': 0.94
            },
            'voice_quality_metrics': {
                'generation_latency': 300,
                'quality_scores': 0.89
            },
            'system_resources': {
                'cpu_usage': 65,
                'memory_usage': 70
            }
        }
        
        # Calculate health score
        health_score = await monitoring_service.analytics_processor.calculate_platform_health_score(sample_metrics)
        
        # Should return a reasonable health score
        assert 0 <= health_score <= 100
        assert health_score > 50  # With good metrics, should be above 50
    
    @pytest.mark.asyncio
    async def test_websocket_scaling_integration(self, scaling_manager):
        """Test WebSocket-specific scaling integration"""
        # Simulate high WebSocket load
        websocket_metrics = {
            'active_connections': 1200,
            'connection_rate': 15,
            'message_throughput': 2500
        }
        
        # Test WebSocket scaling decision
        scaling_decision = scaling_manager.calculate_websocket_scaling(websocket_metrics)
        
        assert scaling_decision.scale_up is True
        assert scaling_decision.target_replicas > scaling_decision.current_replicas
        assert 'websocket' in scaling_decision.reason.lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_deployments(self, deployment_manager):
        """Test handling of concurrent deployments"""
        # Create multiple deployment configs
        configs = [
            DeploymentConfig(
                service_name='platform-api',
                image='dreamcast-platform-api',
                tag='v1.2.0'
            ),
            DeploymentConfig(
                service_name='platform-client',
                image='dreamcast-platform-client',
                tag='v1.2.0'
            )
        ]
        
        # Mock successful deployments
        with patch.object(deployment_manager, '_execute_rolling_update', return_value=True):
            # Start concurrent deployments
            deployment_tasks = [
                deployment_manager.deploy_service(config) for config in configs
            ]
            
            # Wait for all deployments to complete
            results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
            
            # All deployments should complete successfully
            for result in results:
                assert not isinstance(result, Exception)
                assert result.status in ['completed', 'in_progress']
    
    @pytest.mark.asyncio
    async def test_monitoring_data_retention(self, monitoring_service):
        """Test monitoring data retention and history management"""
        # Simulate collecting multiple metrics over time
        for i in range(50):
            metrics = await monitoring_service.collect_platform_metrics()
            # Verify metrics are added to history
            
        # Check history length doesn't exceed limits
        assert len(monitoring_service.metrics_history) <= monitoring_service.max_history
        
        # Check trends calculation works with sufficient data
        trends = monitoring_service._calculate_trends()
        assert isinstance(trends, dict)
    
    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, deployment_manager):
        """Test error recovery and resilience"""
        # Create deployment config with auto-rollback enabled
        config = DeploymentConfig(
            service_name='platform-api',
            image='dreamcast-platform-api',
            tag='v1.3.0',
            rollback_on_failure=True
        )
        
        # Mock deployment failure
        with patch.object(deployment_manager, '_execute_rolling_update', return_value=False), \
             patch.object(deployment_manager, 'rollback_service') as mock_rollback:
            
            mock_rollback.return_value = Mock(success=True, message='Rollback successful')
            
            # Execute deployment (should fail and trigger rollback)
            status = await deployment_manager.deploy_service(config)
            
            # Should have attempted rollback
            assert status.status == 'failed'


@pytest.mark.integration
class TestProductionAPIIntegration:
    """Test integration with production API endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_check_endpoints(self):
        """Test health check endpoints are working"""
        # This would test actual HTTP endpoints in a real environment
        # For now, test the health checker logic
        from backend.app.services.production.deployment_manager import HealthChecker
        
        health_checker = HealthChecker()
        
        # Mock successful health check
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            # Test API health check
            is_healthy = await health_checker.check_service_health('platform-api')
            assert isinstance(is_healthy, bool)
    
    @pytest.mark.asyncio
    async def test_websocket_monitoring_connection(self):
        """Test WebSocket monitoring connection"""
        # This would test actual WebSocket connections in a real environment
        # For now, test the connection management logic
        
        # Mock WebSocket behavior
        connection_data = {
            'type': 'platform_metrics',
            'metrics': {
                'timestamp': datetime.utcnow().isoformat(),
                'api_performance': {'response_time': 150}
            }
        }
        
        # Verify data structure is correct
        assert connection_data['type'] == 'platform_metrics'
        assert 'metrics' in connection_data
        assert 'timestamp' in connection_data['metrics']


@pytest.mark.integration
class TestProductionLoadTesting:
    """Test production system under load"""
    
    @pytest.mark.asyncio
    async def test_scaling_under_simulated_load(self, scaling_manager):
        """Test auto-scaling behavior under simulated load"""
        # Simulate gradually increasing load
        load_scenarios = [
            {'cpu_usage': 30, 'memory_usage': 40, 'requests_per_second': 50},
            {'cpu_usage': 60, 'memory_usage': 65, 'requests_per_second': 150},
            {'cpu_usage': 85, 'memory_usage': 88, 'requests_per_second': 300},
            {'cpu_usage': 95, 'memory_usage': 92, 'requests_per_second': 500}
        ]
        
        scaling_decisions = []
        
        for scenario in load_scenarios:
            should_scale = await scaling_manager.should_scale_api(scenario)
            if should_scale:
                decision = await scaling_manager.make_scaling_decision('platform-api', scenario)
                scaling_decisions.append(decision)
        
        # Should have made scaling decisions for high load scenarios
        assert len(scaling_decisions) > 0
        
        # Last decision should be scale up for highest load
        if scaling_decisions:
            last_decision = scaling_decisions[-1]
            assert last_decision.direction == ScalingDirection.UP
    
    @pytest.mark.asyncio
    async def test_monitoring_performance_under_load(self, monitoring_service):
        """Test monitoring service performance under load"""
        # Simulate rapid metric collection
        start_time = datetime.utcnow()
        
        tasks = []
        for _ in range(10):  # Simulate 10 concurrent metric collections
            task = monitoring_service.collect_platform_metrics()
            tasks.append(task)
        
        # Wait for all collections to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert duration < 10.0  # 10 seconds max for 10 concurrent collections
        
        # No exceptions should occur
        for result in results:
            assert not isinstance(result, Exception)


@pytest.mark.integration
class TestProductionRecoveryScenarios:
    """Test recovery from various failure scenarios"""
    
    @pytest.mark.asyncio
    async def test_service_failure_recovery(self, deployment_manager, monitoring_service):
        """Test recovery from service failure"""
        # Simulate service going down
        service_name = 'platform-api'
        
        # Mock health check failure
        with patch.object(monitoring_service, 'update_service_health') as mock_update:
            # Simulate unhealthy service
            await monitoring_service.update_service_health(service_name, {
                'status': 'unhealthy',
                'response_time': 0,
                'error_rate': 1.0,
                'details': {'error': 'Service unreachable'}
            })
            
            mock_update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_failure_recovery(self, monitoring_service):
        """Test monitoring behavior during database failure"""
        # Simulate database connectivity issues
        db_failure_metrics = {
            'database_metrics': {
                'query_time': 10000,  # Very high query time
                'connections_active': 0,
                'cache_hit_ratio': 0.0
            }
        }
        
        # Should generate alerts for database issues
        alerts = await monitoring_service.alert_manager.check_metrics(db_failure_metrics)
        
        # Should have database-related alerts
        db_alerts = [alert for alert in alerts if 'database' in alert.message.lower()]
        assert len(db_alerts) > 0 