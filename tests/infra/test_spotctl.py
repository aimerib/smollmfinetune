"""
Unit tests for spot instance orchestrator CLI.

Tests the spotctl command-line tool that manages spot GPU instances for
cost-effective weekend/overnight training runs with automatic resume.
"""
import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta, timezone
import boto3
from botocore.exceptions import ClientError
import click
from click.testing import CliRunner

# Import the CLI we'll build
from scripts.spotctl import (
    cli,
    SpotOrchestrator,
    RunPodProvider,
    AWSProvider,
    HeartbeatMonitor,
    CostTracker
)


class TestSpotOrchestrator:
    """Test the core SpotOrchestrator class"""
    
    @patch.dict(os.environ, {'AWS_REGION': 'us-east-1'})
    @patch('scripts.spotctl.boto3.client')
    def test_init_with_aws_provider(self, mock_boto):
        """Should initialize with AWS provider by default"""
        mock_boto.return_value = Mock()
        orchestrator = SpotOrchestrator(provider='aws')
        assert isinstance(orchestrator.provider, AWSProvider)
        assert orchestrator.provider_name == 'aws'
    
    @patch.dict(os.environ, {'RUNPOD_API_KEY': 'test_key'})
    def test_init_with_runpod_provider(self):
        """Should initialize with RunPod provider when specified"""
        orchestrator = SpotOrchestrator(provider='runpod')
        assert isinstance(orchestrator.provider, RunPodProvider)
        assert orchestrator.provider_name == 'runpod'
    
    def test_invalid_provider_raises_error(self):
        """Should raise error for invalid provider"""
        with pytest.raises(ValueError, match="Unknown provider"):
            SpotOrchestrator(provider='invalid')
    
    @patch.dict(os.environ, {'AWS_REGION': 'us-east-1'})
    @patch('scripts.spotctl.subprocess.check_output')
    @patch('scripts.spotctl.boto3.client')
    def test_deploy_creates_spot_instance(self, mock_boto, mock_subprocess):
        """Should deploy a new spot instance with correct configuration"""
        mock_ec2 = Mock()
        mock_boto.return_value = mock_ec2
        mock_subprocess.return_value = b'abc123\n'
        
        # Mock security group response
        mock_ec2.describe_security_groups.return_value = {
            'SecurityGroups': [{
                'GroupId': 'sg-12345'
            }]
        }
        
        # Mock the spot instance creation response
        mock_ec2.request_spot_instances.return_value = {
            'SpotInstanceRequests': [{
                'SpotInstanceRequestId': 'sir-12345678',
                'State': 'open'
            }]
        }
        
        orchestrator = SpotOrchestrator(provider='aws')
        result = orchestrator.deploy(
            instance_type='p4d.24xlarge',
            max_price=10.0,
            git_sha='abc123',
            resume_from=None
        )
        
        assert result['instance_id'] == 'sir-12345678'
        assert result['status'] == 'launching'
        
        # Verify correct spot request parameters
        mock_ec2.request_spot_instances.assert_called_once()
        call_args = mock_ec2.request_spot_instances.call_args[1]
        # Check LaunchSpecification contains correct instance type
        assert call_args['LaunchSpecification']['InstanceType'] == 'p4d.24xlarge'
        assert call_args['SpotPrice'] == '10.0'
    
    @patch.dict(os.environ, {'AWS_REGION': 'us-east-1'})
    @patch('scripts.spotctl.boto3.client')
    def test_deploy_with_resume_includes_checkpoint(self, mock_boto):
        """Should include checkpoint restoration in bootstrap when resuming"""
        mock_boto.return_value = Mock()
        orchestrator = SpotOrchestrator(provider='aws')
        
        with patch.object(orchestrator.provider, 'launch_instance') as mock_launch:
            mock_launch.return_value = {'instance_id': 'i-12345', 'status': 'running'}
            
            orchestrator.deploy(
                instance_type='p4d.24xlarge',
                max_price=10.0,
                git_sha='abc123',
                resume_from='s3://bucket/checkpoint-500/shards/checkpoint.json'
            )
            
            # Check that bootstrap script includes resume command
            call_args = mock_launch.call_args[1]
            bootstrap_script = call_args['user_data']
            assert '--resume-from s3://bucket/checkpoint-500/shards/checkpoint.json' in bootstrap_script
    
    @patch.dict(os.environ, {'AWS_REGION': 'us-east-1'})
    @patch('scripts.spotctl.boto3.client')
    def test_get_status_returns_instance_info(self, mock_boto):
        """Should return current instance status and metadata"""
        mock_boto.return_value = Mock()
        orchestrator = SpotOrchestrator(provider='aws')
        
        with patch.object(orchestrator.provider, 'get_instance_status') as mock_status:
            mock_status.return_value = {
                'instance_id': 'i-12345',
                'state': 'running',
                'public_ip': '54.123.45.67',
                'launch_time': datetime.now().isoformat(),
                'spot_price': 2.5
            }
            
            status = orchestrator.get_status()
            
            assert status['instance_id'] == 'i-12345'
            assert status['state'] == 'running'
            assert 'uptime_hours' in status
            assert 'estimated_cost' in status
    
    @patch.dict(os.environ, {'AWS_REGION': 'us-east-1'})
    @patch('scripts.spotctl.boto3.client')
    def test_terminate_stops_instance(self, mock_boto):
        """Should terminate the running instance"""
        # Mock EC2 client with proper return structure for get_status() call
        mock_ec2 = Mock()
        mock_ec2.describe_spot_instance_requests.return_value = {
            'SpotInstanceRequests': [{
                'InstanceId': 'i-12345',
                'State': 'active'
            }]
        }
        mock_ec2.describe_instances.return_value = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': 'i-12345',
                    'State': {'Name': 'running'},
                    'PublicIpAddress': '54.123.45.67',
                    'LaunchTime': datetime.now(),
                    'InstanceType': 'p4d.24xlarge'
                }]
            }]
        }
        mock_boto.return_value = mock_ec2
        orchestrator = SpotOrchestrator(provider='aws')
        
        with patch.object(orchestrator.provider, 'terminate_instance') as mock_terminate:
            mock_terminate.return_value = True
            
            result = orchestrator.terminate()
            
            assert result is True
            mock_terminate.assert_called_once()
    
    @patch.dict(os.environ, {'AWS_REGION': 'us-east-1'})
    @patch('scripts.spotctl.boto3.client')
    @patch('scripts.spotctl.subprocess.run')
    def test_tail_logs_streams_from_instance(self, mock_subprocess, mock_boto):
        """Should stream logs from running instance"""
        mock_boto.return_value = Mock()
        orchestrator = SpotOrchestrator(provider='aws')
        
        # Mock the metadata loading to return instance info
        with patch.object(orchestrator, '_load_instance_metadata') as mock_metadata:
            mock_metadata.return_value = {'instance_id': 'i-12345'}
            
            with patch.object(orchestrator.provider, 'get_instance_status') as mock_status:
                mock_status.return_value = {
                    'instance_id': 'i-12345',
                    'public_ip': '54.123.45.67',
                    'state': 'running'
                }
                
                orchestrator.tail_logs(lines=50)
                
                # Should SSH into instance and tail logs
                mock_subprocess.assert_called_once()
                ssh_command = mock_subprocess.call_args[0][0]
                assert 'ssh' in ssh_command
                assert '54.123.45.67' in ' '.join(ssh_command)
                assert 'tail -f' in ' '.join(ssh_command)


class TestHeartbeatMonitor:
    """Test the heartbeat monitoring system"""
    
    def test_write_heartbeat_to_s3(self):
        """Should write heartbeat timestamp to S3"""
        mock_s3 = Mock()
        
        monitor = HeartbeatMonitor(
            s3_bucket='training-heartbeats',
            run_id='run_123',
            s3_client=mock_s3
        )
        
        monitor.write_heartbeat()
        
        # Should write current timestamp to S3
        mock_s3.put_object.assert_called_once()
        call_args = mock_s3.put_object.call_args[1]
        assert call_args['Bucket'] == 'training-heartbeats'
        assert call_args['Key'] == 'heartbeats/run_123'
        
        # Verify timestamp format
        body = json.loads(call_args['Body'])
        assert 'timestamp' in body
        assert 'instance_id' in body
    
    @patch.dict(os.environ, {'AWS_REGION': 'us-east-1'})
    @patch('scripts.spotctl.boto3.client')
    def test_check_heartbeat_age(self, mock_boto):
        """Should calculate heartbeat age correctly"""
        # Mock S3 response with 10-minute old heartbeat
        old_timestamp = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        mock_response = {
            'Body': Mock(read=lambda: json.dumps({
                'timestamp': old_timestamp,
                'instance_id': 'i-12345'
            }).encode())
        }
        
        mock_s3 = Mock()
        mock_s3.get_object.return_value = mock_response
        mock_boto.return_value = mock_s3
        
        monitor = HeartbeatMonitor(
            s3_bucket='training-heartbeats',
            run_id='run_123',
            s3_client=mock_s3
        )
        
        age_minutes = monitor.get_heartbeat_age()
        
        assert 9 <= age_minutes <= 11  # Allow for small timing differences
    
    def test_is_stale_detects_old_heartbeat(self):
        """Should detect when heartbeat is too old"""
        monitor = HeartbeatMonitor(
            s3_bucket='training-heartbeats',
            run_id='run_123'
        )
        
        with patch.object(monitor, 'get_heartbeat_age') as mock_age:
            # Fresh heartbeat
            mock_age.return_value = 3
            assert monitor.is_stale(threshold_minutes=5) is False
            
            # Stale heartbeat
            mock_age.return_value = 10
            assert monitor.is_stale(threshold_minutes=5) is True


class TestCostTracker:
    """Test the cost tracking and budget enforcement"""
    
    def test_calculate_spot_cost(self):
        """Should calculate cumulative spot instance costs"""
        tracker = CostTracker(daily_budget=100.0)
        
        # Test cost calculation
        cost = tracker.calculate_cost(
            hours_running=2.5,
            hourly_rate=3.0
        )
        
        assert cost == 7.5
    
    def test_track_daily_spend(self):
        """Should track spending and enforce daily budget"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = CostTracker(
                daily_budget=100.0,
                cost_log_path=Path(tmpdir) / 'costs.json'
            )
            
            # Record some spending
            tracker.record_spend('i-12345', 25.0)
            tracker.record_spend('i-67890', 30.0)
            
            assert tracker.get_daily_spend() == 55.0
            assert tracker.is_under_budget() is True
            
            # Exceed budget
            tracker.record_spend('i-11111', 50.0)
            assert tracker.get_daily_spend() == 105.0
            assert tracker.is_under_budget() is False
    
    def test_reset_daily_counter(self):
        """Should reset daily spending at midnight"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = CostTracker(
                daily_budget=100.0,
                cost_log_path=Path(tmpdir) / 'costs.json'
            )
            
            # Record spending
            tracker.record_spend('i-12345', 50.0)
            
            # Simulate next day
            with patch('scripts.spotctl.datetime') as mock_datetime:
                mock_datetime.now.return_value = datetime.now() + timedelta(days=1)
                mock_datetime.utcnow = mock_datetime.now
                
                # Should reset to 0
                assert tracker.get_daily_spend() == 0.0


class TestCLI:
    """Test the spotctl CLI commands"""
    
    def test_deploy_command(self):
        """Test spotctl deploy command"""
        runner = CliRunner()
        
        with patch('scripts.spotctl.SpotOrchestrator') as mock_orchestrator:
            mock_instance = Mock()
            mock_instance.deploy.return_value = {
                'instance_id': 'i-12345',
                'status': 'launching'
            }
            mock_orchestrator.return_value = mock_instance
            
            result = runner.invoke(cli, [
                'deploy',
                '--instance-type', 'p4d.24xlarge',
                '--max-price', '10.0',
                '--provider', 'aws'
            ])
            
            assert result.exit_code == 0
            assert 'Deploying spot instance' in result.output
            assert 'Instance ID: i-12345' in result.output
    
    def test_deploy_with_resume(self):
        """Test deploy command with checkpoint resume"""
        runner = CliRunner()
        
        with patch('scripts.spotctl.SpotOrchestrator') as mock_orchestrator:
            mock_instance = Mock()
            mock_instance.deploy.return_value = {
                'instance_id': 'i-12345',
                'status': 'launching'
            }
            mock_orchestrator.return_value = mock_instance
            
            result = runner.invoke(cli, [
                'deploy',
                '--resume',
                '--checkpoint', 's3://bucket/checkpoint.json'
            ])
            
            assert result.exit_code == 0
            # Verify resume parameters were passed
            deploy_call = mock_instance.deploy.call_args[1]
            assert deploy_call['resume_from'] == 's3://bucket/checkpoint.json'
    
    def test_status_command(self):
        """Test spotctl status command"""
        runner = CliRunner()
        
        with patch('scripts.spotctl.SpotOrchestrator') as mock_orchestrator:
            mock_instance = Mock()
            mock_instance.get_status.return_value = {
                'instance_id': 'i-12345',
                'state': 'running',
                'uptime_hours': 2.5,
                'estimated_cost': 7.5,
                'public_ip': '54.123.45.67'
            }
            mock_orchestrator.return_value = mock_instance
            
            result = runner.invoke(cli, ['status'])
            
            assert result.exit_code == 0
            assert 'i-12345' in result.output
            assert 'running' in result.output
            assert '54.123.45.67' in result.output
            assert '2.5 hours' in result.output
            assert '$7.50' in result.output
    
    def test_terminate_command(self):
        """Test spotctl terminate command"""
        runner = CliRunner()
        
        with patch('scripts.spotctl.SpotOrchestrator') as mock_orchestrator:
            mock_instance = Mock()
            mock_instance.terminate.return_value = True
            mock_orchestrator.return_value = mock_instance
            
            # Test with confirmation
            result = runner.invoke(cli, ['terminate'], input='y\n')
            
            assert result.exit_code == 0
            assert 'Instance terminated' in result.output
    
    def test_tail_logs_command(self):
        """Test spotctl tail-logs command"""
        runner = CliRunner()
        
        with patch('scripts.spotctl.SpotOrchestrator') as mock_orchestrator:
            mock_instance = Mock()
            mock_orchestrator.return_value = mock_instance
            
            result = runner.invoke(cli, ['tail-logs', '--lines', '100'])
            
            assert result.exit_code == 0
            mock_instance.tail_logs.assert_called_once_with(lines=100)


class TestRunPodProvider:
    """Test RunPod-specific provider implementation"""
    
    @patch('httpx.post')
    def test_launch_runpod_instance(self, mock_post):
        """Should launch RunPod instance via API"""
        provider = RunPodProvider(api_key='test_key')
        
        # Mock RunPod API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'id': 'pod_12345',
            'status': 'starting',
            'machine': {
                'gpu_type': 'RTX_A5000',
                'gpu_count': 1
            }
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = provider.launch_instance(
            instance_type='RTX_A5000',
            spot_price=1.5,
            user_data='#!/bin/bash\necho "Hello"'
        )
        
        assert result['instance_id'] == 'pod_12345'
        assert result['status'] == 'starting'
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert 'runpod.io' in call_args[0][0]
        assert call_args[1]['headers']['Authorization'] == 'Bearer test_key'
    
    @patch('httpx.get')
    def test_get_runpod_status(self, mock_get):
        """Should get RunPod instance status"""
        provider = RunPodProvider(api_key='test_key')
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'id': 'pod_12345',
            'status': 'running',
            'runtime': {
                'uptimeSeconds': 3600,
                'pods': [{
                    'ip': '192.168.1.100'
                }]
            },
            'costPerHour': 1.5
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        status = provider.get_instance_status('pod_12345')
        
        assert status['state'] == 'running'
        assert status['public_ip'] == '192.168.1.100'
        assert status['spot_price'] == 1.5


class TestGitHubActionIntegration:
    """Test GitHub Action auto-resume integration"""
    
    @patch.dict(os.environ, {'AWS_REGION': 'us-east-1'})
    @patch('scripts.spotctl.boto3.client')
    def test_check_and_resume_workflow(self, mock_boto):
        """Should detect stale heartbeat and trigger resume"""
        # This would be tested in the actual GitHub Action
        # Here we test the logic that the action would call
        
        mock_boto.return_value = Mock()
        
        monitor = HeartbeatMonitor(
            s3_bucket='training-heartbeats',
            run_id='run_123'
        )
        
        orchestrator = SpotOrchestrator(provider='aws')
        
        with patch.object(monitor, 'is_stale') as mock_stale:
            with patch.object(orchestrator, 'deploy') as mock_deploy:
                # Simulate stale heartbeat
                mock_stale.return_value = True
                
                # Function that GitHub Action would call
                from scripts.spotctl import check_and_resume
                
                result = check_and_resume(
                    monitor=monitor,
                    orchestrator=orchestrator,
                    checkpoint='s3://bucket/latest/checkpoint.json'
                )
                
                assert result['action'] == 'resumed'
                mock_deploy.assert_called_once_with(
                    resume_from='s3://bucket/latest/checkpoint.json'
                ) 