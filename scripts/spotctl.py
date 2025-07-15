#!/usr/bin/env python3
"""
Spot Instance Training Orchestrator CLI

Manages pre-emptible GPU instances for cost-effective model training with
automatic checkpoint resume after interruptions.
"""
import os
import sys
import json
import time
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import logging

import click
import boto3
from botocore.exceptions import ClientError
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rich console for pretty output
console = Console()


class ProviderInterface(ABC):
    """Abstract interface for cloud providers"""
    
    @abstractmethod
    def launch_instance(self, instance_type: str, spot_price: float, 
                       user_data: str, **kwargs) -> Dict[str, Any]:
        """Launch a spot instance"""
        pass
    
    @abstractmethod
    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get instance status and metadata"""
        pass
    
    @abstractmethod
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate an instance"""
        pass


class AWSProvider(ProviderInterface):
    """AWS EC2 spot instance provider"""
    
    def __init__(self):
        self.ec2 = boto3.client('ec2')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
    
    def launch_instance(self, instance_type: str, spot_price: float, 
                       user_data: str, **kwargs) -> Dict[str, Any]:
        """Launch an AWS spot instance"""
        try:
            # Create spot instance request
            response = self.ec2.request_spot_instances(
                InstanceCount=1,
                Type='one-time',
                InstanceInterruptionBehavior='terminate',
                LaunchSpecification={
                    'ImageId': self._get_ami_id(),
                    'InstanceType': instance_type,
                    'KeyName': os.getenv('AWS_KEY_NAME', 'training-key'),
                    'SecurityGroupIds': [self._get_or_create_security_group()],
                    'UserData': user_data,
                    'BlockDeviceMappings': [{
                        'DeviceName': '/dev/sda1',
                        'Ebs': {
                            'VolumeSize': 500,  # 500GB for models/checkpoints
                            'VolumeType': 'gp3',
                            'DeleteOnTermination': True
                        }
                    }],
                    'IamInstanceProfile': {
                        'Name': os.getenv('AWS_INSTANCE_PROFILE', 'training-instance-profile')
                    }
                },
                SpotPrice=str(spot_price),
                ValidUntil=datetime.now(timezone.utc) + timedelta(days=7)
            )
            
            request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
            
            return {
                'instance_id': request_id,
                'status': 'launching',
                'provider': 'aws',
                'region': self.region
            }
            
        except ClientError as e:
            logger.error(f"Failed to launch AWS instance: {e}")
            raise
    
    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get AWS instance status"""
        try:
            # Get spot request info
            spot_response = self.ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[instance_id]
            )
            
            if not spot_response['SpotInstanceRequests']:
                return {'state': 'not_found'}
            
            spot_request = spot_response['SpotInstanceRequests'][0]
            
            result = {
                'instance_id': instance_id,
                'state': spot_request['State'],
                'spot_price': float(spot_request.get('SpotPrice', 0)),
                'launch_time': spot_request.get('CreateTime', datetime.now(timezone.utc)).isoformat()
            }
            
            # If instance is fulfilled, get instance details
            if 'InstanceId' in spot_request:
                ec2_instance_id = spot_request['InstanceId']
                instance_response = self.ec2.describe_instances(
                    InstanceIds=[ec2_instance_id]
                )
                
                if instance_response['Reservations']:
                    instance = instance_response['Reservations'][0]['Instances'][0]
                    result.update({
                        'ec2_instance_id': ec2_instance_id,
                        'state': instance['State']['Name'],
                        'public_ip': instance.get('PublicIpAddress', 'N/A'),
                        'instance_type': instance['InstanceType']
                    })
            
            return result
            
        except ClientError as e:
            logger.error(f"Failed to get instance status: {e}")
            return {'state': 'error', 'error': str(e)}
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate AWS spot instance"""
        try:
            # Cancel spot request
            self.ec2.cancel_spot_instance_requests(
                SpotInstanceRequestIds=[instance_id]
            )
            
            # Get actual instance ID and terminate
            status = self.get_instance_status(instance_id)
            if 'ec2_instance_id' in status:
                self.ec2.terminate_instances(
                    InstanceIds=[status['ec2_instance_id']]
                )
            
            return True
            
        except ClientError as e:
            logger.error(f"Failed to terminate instance: {e}")
            return False
    
    def _get_ami_id(self) -> str:
        """Get appropriate Deep Learning AMI for the region"""
        # Default to Ubuntu Deep Learning AMI
        # In production, this would query for the latest AMI
        ami_map = {
            'us-east-1': 'ami-0c55b159cbfafe1f0',  # Example AMI ID
            'us-west-2': 'ami-0c55b159cbfafe1f0'
        }
        return os.getenv('AWS_AMI_ID', ami_map.get(self.region, 'ami-0c55b159cbfafe1f0'))
    
    def _get_or_create_security_group(self) -> str:
        """Get or create security group for training instances"""
        sg_name = 'training-spot-instances'
        
        try:
            # Check if security group exists
            response = self.ec2.describe_security_groups(
                GroupNames=[sg_name]
            )
            return response['SecurityGroups'][0]['GroupId']
            
        except ClientError:
            # Create security group
            response = self.ec2.create_security_group(
                GroupName=sg_name,
                Description='Security group for training spot instances'
            )
            
            sg_id = response['GroupId']
            
            # Add SSH access
            self.ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    'IpProtocol': 'tcp',
                    'FromPort': 22,
                    'ToPort': 22,
                    'IpRanges': [{'CidrIp': '0.0.0.0/0'}]
                }]
            )
            
            return sg_id


class RunPodProvider(ProviderInterface):
    """RunPod GPU instance provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('RUNPOD_API_KEY')
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY not provided")
        
        self.base_url = "https://api.runpod.io/v2"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def launch_instance(self, instance_type: str, spot_price: float, 
                       user_data: str, **kwargs) -> Dict[str, Any]:
        """Launch a RunPod instance"""
        try:
            # Map instance types to RunPod GPU types
            gpu_map = {
                'RTX_A5000': 'NVIDIA RTX A5000',
                'RTX_4090': 'NVIDIA GeForce RTX 4090',
                'A100': 'NVIDIA A100 80GB PCIe'
            }
            
            payload = {
                'cloudType': 'SECURE',  # Use spot/preemptible
                'gpuType': gpu_map.get(instance_type, instance_type),
                'gpuCount': 1,
                'containerDiskInGb': 100,
                'volumeInGb': 500,
                'minDownload': 100,  # Mbps
                'minUpload': 100,
                'bidPerGpu': spot_price,
                'dockerArgs': user_data,  # RunPod uses docker args instead
                'env': {
                    'AUTO_RESUME': 'true',
                    'REPO_URL': os.getenv('REPO_URL', 'https://github.com/aimerib/smollmfinetune.git')
                }
            }
            
            response = httpx.post(
                f"{self.base_url}/pods",
                headers=self.headers,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'instance_id': data['id'],
                'status': data.get('status', 'starting'),
                'provider': 'runpod',
                'gpu_type': instance_type
            }
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to launch RunPod instance: {e}")
            raise
    
    def get_instance_status(self, instance_id: str) -> Dict[str, Any]:
        """Get RunPod instance status"""
        try:
            response = httpx.get(
                f"{self.base_url}/pods/{instance_id}",
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant info
            runtime = data.get('runtime', {})
            pod_info = runtime.get('pods', [{}])[0] if runtime.get('pods') else {}
            
            return {
                'instance_id': instance_id,
                'state': data.get('status', 'unknown'),
                'public_ip': pod_info.get('ip', 'N/A'),
                'spot_price': data.get('costPerHour', 0),
                'launch_time': data.get('createdAt', datetime.now(timezone.utc).isoformat()),
                'gpu_type': data.get('machine', {}).get('gpu_type', 'unknown')
            }
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to get RunPod status: {e}")
            return {'state': 'error', 'error': str(e)}
    
    def terminate_instance(self, instance_id: str) -> bool:
        """Terminate RunPod instance"""
        try:
            response = httpx.delete(
                f"{self.base_url}/pods/{instance_id}",
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return True
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to terminate RunPod instance: {e}")
            return False


class HeartbeatMonitor:
    """Monitors training heartbeat for failure detection"""
    
    def __init__(self, s3_bucket: str, run_id: str, s3_client=None):
        self.s3_bucket = s3_bucket
        self.run_id = run_id
        self.s3_client = s3_client or boto3.client('s3')
        self.instance_id = os.getenv('INSTANCE_ID', 'unknown')
    
    def write_heartbeat(self) -> None:
        """Write heartbeat timestamp to S3"""
        heartbeat_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'instance_id': self.instance_id,
            'run_id': self.run_id,
            'status': 'alive'
        }
        
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=f'heartbeats/{self.run_id}',
                Body=json.dumps(heartbeat_data),
                ContentType='application/json'
            )
            logger.debug(f"Heartbeat written for run {self.run_id}")
        except ClientError as e:
            logger.error(f"Failed to write heartbeat: {e}")
    
    def get_heartbeat_age(self) -> float:
        """Get age of last heartbeat in minutes"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=f'heartbeats/{self.run_id}'
            )
            
            data = json.loads(response['Body'].read())
            last_heartbeat = datetime.fromisoformat(data['timestamp'])
            
            # Handle timezone-aware datetime
            if last_heartbeat.tzinfo is None:
                last_heartbeat = last_heartbeat.replace(tzinfo=timezone.utc)
            
            current_time = datetime.now(timezone.utc)
            age = (current_time - last_heartbeat).total_seconds() / 60
            
            return age
            
        except ClientError as e:
            logger.error(f"Failed to read heartbeat: {e}")
            return float('inf')  # Assume very old if can't read
    
    def is_stale(self, threshold_minutes: int = 10) -> bool:
        """Check if heartbeat is stale"""
        age = self.get_heartbeat_age()
        return age > threshold_minutes


class CostTracker:
    """Tracks spending and enforces budget limits"""
    
    def __init__(self, daily_budget: float, cost_log_path: Optional[Path] = None):
        self.daily_budget = daily_budget
        self.cost_log_path = cost_log_path or Path('spot_costs.json')
        self._ensure_log_file()
    
    def _ensure_log_file(self) -> None:
        """Ensure cost log file exists"""
        if not self.cost_log_path.exists():
            self.cost_log_path.write_text('{}')
    
    def _load_costs(self) -> Dict[str, Any]:
        """Load cost data from file"""
        try:
            return json.loads(self.cost_log_path.read_text())
        except json.JSONDecodeError:
            return {}
    
    def _save_costs(self, costs: Dict[str, Any]) -> None:
        """Save cost data to file"""
        self.cost_log_path.write_text(json.dumps(costs, indent=2))
    
    def calculate_cost(self, hours_running: float, hourly_rate: float) -> float:
        """Calculate cost for running time"""
        return hours_running * hourly_rate
    
    def record_spend(self, instance_id: str, amount: float) -> None:
        """Record spending for an instance"""
        costs = self._load_costs()
        
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in costs:
            costs[today] = {}
        
        if instance_id not in costs[today]:
            costs[today][instance_id] = 0
        
        costs[today][instance_id] += amount
        self._save_costs(costs)
        
        logger.info(f"Recorded ${amount:.2f} spend for {instance_id}")
    
    def get_daily_spend(self) -> float:
        """Get total spending for today"""
        costs = self._load_costs()
        today = datetime.now().strftime('%Y-%m-%d')
        
        if today not in costs:
            return 0.0
        
        return sum(costs[today].values())
    
    def is_under_budget(self) -> bool:
        """Check if we're under daily budget"""
        return self.get_daily_spend() < self.daily_budget


class AlertManager:
    """Manages notifications and alerts for training events"""
    
    def __init__(self):
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'email_user': os.getenv('EMAIL_USER'),
            'email_password': os.getenv('EMAIL_PASSWORD'),
            'email_to': os.getenv('EMAIL_TO')
        }
        
    def send_alert(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Send alert to all configured channels"""
        alert_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'message': message,
            'details': details or {}
        }
        
        if self.slack_webhook:
            self._send_slack_alert(alert_data)
        
        if self.discord_webhook:
            self._send_discord_alert(alert_data)
            
        if all(self.email_config.values()):
            self._send_email_alert(alert_data)
    
    def _send_slack_alert(self, alert_data: Dict[str, Any]):
        """Send alert to Slack"""
        try:
            color = {
                'success': '#36a64f',
                'warning': '#ff9500', 
                'error': '#ff0000',
                'info': '#36a64f'
            }.get(alert_data['event_type'], '#cccccc')
            
            payload = {
                'attachments': [{
                    'color': color,
                    'title': f"🤖 Spot Training: {alert_data['event_type'].title()}",
                    'text': alert_data['message'],
                    'fields': [
                        {'title': k, 'value': str(v), 'short': True}
                        for k, v in alert_data['details'].items()
                    ],
                    'footer': 'Spot Orchestrator',
                    'ts': int(datetime.fromisoformat(alert_data['timestamp']).timestamp())
                }]
            }
            
            response = httpx.post(self.slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_discord_alert(self, alert_data: Dict[str, Any]):
        """Send alert to Discord"""
        try:
            color = {
                'success': 0x36a64f,
                'warning': 0xff9500,
                'error': 0xff0000,
                'info': 0x36a64f
            }.get(alert_data['event_type'], 0xcccccc)
            
            embed = {
                'title': f"🤖 Spot Training: {alert_data['event_type'].title()}",
                'description': alert_data['message'],
                'color': color,
                'timestamp': alert_data['timestamp'],
                'fields': [
                    {'name': k, 'value': str(v), 'inline': True}
                    for k, v in alert_data['details'].items()
                ],
                'footer': {'text': 'Spot Orchestrator'}
            }
            
            payload = {'embeds': [embed]}
            
            response = httpx.post(self.discord_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
    
    def _send_email_alert(self, alert_data: Dict[str, Any]):
        """Send alert via email"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config['email_user']
            msg['To'] = self.email_config['email_to']
            msg['Subject'] = f"Spot Training Alert: {alert_data['event_type'].title()}"
            
            body = f"""
            Alert: {alert_data['message']}
            Time: {alert_data['timestamp']}
            
            Details:
            {chr(10).join(f"• {k}: {v}" for k, v in alert_data['details'].items())}
            
            --
            Spot Orchestrator Automated Alert
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['email_user'], self.email_config['email_password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


class MetricsCollector:
    """Collects and reports system metrics"""
    
    def __init__(self, cloudwatch_enabled: bool = True):
        self.cloudwatch_enabled = cloudwatch_enabled
        self.cloudwatch = boto3.client('cloudwatch') if cloudwatch_enabled else None
        self.metrics_buffer = []
        
    def record_metric(self, name: str, value: float, unit: str = 'Count', 
                     dimensions: Dict[str, str] = None):
        """Record a custom metric"""
        metric = {
            'MetricName': name,
            'Value': value,
            'Unit': unit,
            'Timestamp': datetime.now(timezone.utc),
            'Dimensions': [
                {'Name': k, 'Value': v} 
                for k, v in (dimensions or {}).items()
            ]
        }
        
        self.metrics_buffer.append(metric)
        
        # Flush buffer if it gets too large
        if len(self.metrics_buffer) >= 20:
            self.flush_metrics()
    
    def flush_metrics(self):
        """Send buffered metrics to CloudWatch"""
        if not self.cloudwatch_enabled or not self.metrics_buffer:
            return
            
        try:
            self.cloudwatch.put_metric_data(
                Namespace='SpotOrchestrator',
                MetricData=self.metrics_buffer
            )
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to send metrics to CloudWatch: {e}")
    
    def record_instance_metrics(self, instance_info: Dict[str, Any]):
        """Record instance-specific metrics"""
        dimensions = {
            'InstanceId': instance_info.get('instance_id', 'unknown'),
            'Provider': instance_info.get('provider', 'unknown')
        }
        
        if 'uptime_hours' in instance_info:
            self.record_metric('InstanceUptime', instance_info['uptime_hours'], 
                             'Count', dimensions)
        
        if 'estimated_cost' in instance_info:
            self.record_metric('EstimatedCost', instance_info['estimated_cost'], 
                             'Count', dimensions)
        
        if 'spot_price' in instance_info:
            self.record_metric('SpotPrice', instance_info['spot_price'], 
                             'Count', dimensions)


class CostOptimizer:
    """Advanced cost optimization strategies"""
    
    def __init__(self):
        self.instance_preferences = [
            # Ordered by cost-effectiveness
            {'type': 'p3.2xlarge', 'max_price': 3.0, 'provider': 'aws'},
            {'type': 'p3.8xlarge', 'max_price': 12.0, 'provider': 'aws'},
            {'type': 'RTX_A4000', 'max_price': 0.4, 'provider': 'runpod'},
            {'type': 'RTX_A5000', 'max_price': 0.6, 'provider': 'runpod'},
            {'type': 'p4d.24xlarge', 'max_price': 20.0, 'provider': 'aws'},
        ]
        
        self.off_peak_hours = range(22, 6)  # 10 PM to 6 AM UTC
        
    def get_optimal_instance(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Get the most cost-effective instance for requirements"""
        min_gpu_memory = requirements.get('min_gpu_memory_gb', 16)
        max_budget = requirements.get('max_hourly_cost', float('inf'))
        
        # Filter by requirements
        suitable_instances = []
        for instance in self.instance_preferences:
            gpu_memory = self._get_gpu_memory(instance['type'])
            if gpu_memory >= min_gpu_memory and instance['max_price'] <= max_budget:
                suitable_instances.append(instance)
        
        if not suitable_instances:
            raise ValueError("No suitable instances found for requirements")
        
        # Apply time-based pricing preferences
        current_hour = datetime.now(timezone.utc).hour
        if current_hour in self.off_peak_hours:
            # Prefer more powerful instances during off-peak
            return suitable_instances[-1]
        else:
            # Prefer cheaper instances during peak
            return suitable_instances[0]
    
    def _get_gpu_memory(self, instance_type: str) -> int:
        """Get GPU memory for instance type"""
        memory_map = {
            'p3.2xlarge': 16,
            'p3.8xlarge': 64,
            'p4d.24xlarge': 320,
            'RTX_A4000': 16,
            'RTX_A5000': 24,
            'RTX_4090': 24,
            'A100_80GB': 80
        }
        return memory_map.get(instance_type, 16)
    
    def suggest_schedule(self, training_hours: int) -> Dict[str, Any]:
        """Suggest optimal training schedule"""
        current_time = datetime.now(timezone.utc)
        
        # Find next off-peak window
        next_off_peak = current_time.replace(hour=22, minute=0, second=0, microsecond=0)
        if current_time.hour >= 22:
            next_off_peak += timedelta(days=1)
        
        off_peak_duration = 8  # 10 PM to 6 AM
        
        if training_hours <= off_peak_duration:
            return {
                'strategy': 'single_session',
                'start_time': next_off_peak,
                'estimated_savings': '30-50%'
            }
        else:
            sessions_needed = (training_hours + off_peak_duration - 1) // off_peak_duration
            return {
                'strategy': 'split_sessions',
                'sessions': sessions_needed,
                'start_time': next_off_peak,
                'estimated_savings': '20-40%'
            }


class InstanceChainer:
    """Implements instance chaining for cost optimization"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.chain_config_path = Path('.instance_chain.json')
        
    def deploy_with_chaining(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy with instance chaining strategy"""
        optimizer = CostOptimizer()
        
        # Try instances in order of preference
        for attempt, instance_config in enumerate(optimizer.instance_preferences):
            try:
                console.print(f"[yellow]Attempt {attempt + 1}: Trying {instance_config['type']} on {instance_config['provider']}[/yellow]")
                
                # Switch provider if needed
                if self.orchestrator.provider_name != instance_config['provider']:
                    self.orchestrator = SpotOrchestrator(provider=instance_config['provider'])
                
                result = self.orchestrator.deploy(
                    instance_type=instance_config['type'],
                    max_price=instance_config['max_price'],
                    **requirements
                )
                
                # Save successful configuration
                self._save_chain_config(instance_config, attempt)
                
                console.print(f"[green]✓ Successfully deployed {instance_config['type']}[/green]")
                return result
                
            except Exception as e:
                console.print(f"[red]✗ Failed: {e}[/red]")
                if attempt == len(optimizer.instance_preferences) - 1:
                    raise RuntimeError("All instance types failed to deploy")
                continue
    
    def _save_chain_config(self, config: Dict[str, Any], attempt: int):
        """Save chain configuration for analytics"""
        chain_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'successful_config': config,
            'attempts': attempt + 1
        }
        self.chain_config_path.write_text(json.dumps(chain_data, indent=2))


class SpotOrchestrator:
    """Main orchestrator for spot instance management"""
    
    def __init__(self, provider: str = 'aws'):
        self.provider_name = provider
        
        if provider == 'aws':
            self.provider = AWSProvider()
        elif provider == 'runpod':
            self.provider = RunPodProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Initialize all components
        daily_budget = float(os.getenv('DAILY_BUDGET', '100.0'))
        self.cost_tracker = CostTracker(daily_budget)
        self.alert_manager = AlertManager()
        # Only enable CloudWatch for AWS provider
        self.metrics_collector = MetricsCollector(cloudwatch_enabled=(provider == 'aws'))
        self.cost_optimizer = CostOptimizer()
        self.instance_chainer = InstanceChainer(self)
    
    def deploy(self, instance_type: str = 'p4d.24xlarge', 
               max_price: float = 10.0,
               git_sha: Optional[str] = None,
               resume_from: Optional[str] = None,
               use_chaining: bool = False,
               use_optimizer: bool = False) -> Dict[str, Any]:
        """Deploy a new spot instance with advanced features"""
        
        # Check budget before deploying
        if not self.cost_tracker.is_under_budget():
            daily_spend = self.cost_tracker.get_daily_spend()
            self.alert_manager.send_alert(
                'error', 
                f'Daily budget exceeded! Current spend: ${daily_spend:.2f}',
                {'daily_budget': self.cost_tracker.daily_budget, 'current_spend': daily_spend}
            )
            raise RuntimeError("Daily budget exceeded!")
        
        try:
            # Use instance chaining if requested
            if use_chaining:
                return self.instance_chainer.deploy_with_chaining({
                    'git_sha': git_sha,
                    'resume_from': resume_from
                })
            
            # Use cost optimizer if requested
            if use_optimizer:
                optimal_config = self.cost_optimizer.get_optimal_instance({
                    'min_gpu_memory_gb': 16,
                    'max_hourly_cost': max_price
                })
                instance_type = optimal_config['type']
                max_price = optimal_config['max_price']
                
                console.print(f"[blue]💡 Optimizer selected: {instance_type} @ ${max_price}/hr[/blue]")
            
            # Get current git SHA if not provided
            if not git_sha:
                git_sha = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD']
                ).decode().strip()
            
            # Create bootstrap script
            user_data = self._create_bootstrap_script(git_sha, resume_from)
            
            # Launch instance
            console.print(f"[yellow]🚀 Launching {instance_type} on {self.provider_name}...[/yellow]")
            result = self.provider.launch_instance(
                instance_type=instance_type,
                spot_price=max_price,
                user_data=user_data
            )
            
            # Save instance metadata
            self._save_instance_metadata(result, git_sha, resume_from)
            
            # Send success alert
            self.alert_manager.send_alert(
                'success',
                f'Successfully deployed {instance_type} instance',
                {
                    'instance_id': result['instance_id'],
                    'instance_type': instance_type,
                    'max_price': max_price,
                    'provider': self.provider_name,
                    'resume_from': resume_from or 'new_training'
                }
            )
            
            # Record deployment metrics
            self.metrics_collector.record_metric('InstanceDeployment', 1, 'Count', {
                'Provider': self.provider_name,
                'InstanceType': instance_type
            })
            
            return result
            
        except Exception as e:
            # Send failure alert
            self.alert_manager.send_alert(
                'error',
                f'Failed to deploy instance: {str(e)}',
                {
                    'instance_type': instance_type,
                    'provider': self.provider_name,
                    'error': str(e)
                }
            )
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current instance status with enhanced monitoring"""
        # Load saved instance metadata
        metadata = self._load_instance_metadata()
        if not metadata:
            return {'state': 'no_instance'}
        
        try:
            # Get live status from provider
            status = self.provider.get_instance_status(metadata['instance_id'])
            
            # Calculate uptime and cost
            if 'launch_time' in status and status['state'] == 'running':
                launch_time = datetime.fromisoformat(status['launch_time'])
                if launch_time.tzinfo is None:
                    launch_time = launch_time.replace(tzinfo=timezone.utc)
                
                uptime = datetime.now(timezone.utc) - launch_time
                uptime_hours = uptime.total_seconds() / 3600
                
                status['uptime_hours'] = round(uptime_hours, 2)
                status['estimated_cost'] = round(
                    uptime_hours * status.get('spot_price', 0), 2
                )
                
                # Check for long-running instances (potential cost alert)
                if uptime_hours > 12:
                    self.alert_manager.send_alert(
                        'warning',
                        f'Instance running for {uptime_hours:.1f} hours - consider checking progress',
                        {
                            'instance_id': metadata['instance_id'],
                            'uptime_hours': uptime_hours,
                            'estimated_cost': status['estimated_cost']
                        }
                    )
            
            # Record metrics
            self.metrics_collector.record_instance_metrics(status)
            self.metrics_collector.flush_metrics()
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get instance status: {e}")
            return {'state': 'error', 'error': str(e)}
    
    def terminate(self) -> bool:
        """Terminate current instance with cost tracking and alerts"""
        metadata = self._load_instance_metadata()
        if not metadata:
            logger.warning("No instance to terminate")
            return False
        
        try:
            # Record final costs
            status = self.get_status()
            final_cost = 0
            if 'estimated_cost' in status:
                final_cost = status['estimated_cost']
                self.cost_tracker.record_spend(
                    metadata['instance_id'],
                    final_cost
                )
            
            # Terminate instance
            success = self.provider.terminate_instance(metadata['instance_id'])
            
            if success:
                # Send termination alert
                self.alert_manager.send_alert(
                    'info',
                    f'Successfully terminated instance {metadata["instance_id"]}',
                    {
                        'instance_id': metadata['instance_id'],
                        'final_cost': final_cost,
                        'uptime_hours': status.get('uptime_hours', 0),
                        'provider': self.provider_name
                    }
                )
                
                # Record termination metrics
                self.metrics_collector.record_metric('InstanceTermination', 1, 'Count', {
                    'Provider': self.provider_name,
                    'Reason': 'manual'
                })
                self.metrics_collector.flush_metrics()
                
                # Clear metadata
                metadata_path = Path('.spot_instance.json')
                if metadata_path.exists():
                    metadata_path.unlink()
            else:
                self.alert_manager.send_alert(
                    'error',
                    f'Failed to terminate instance {metadata["instance_id"]}',
                    {'instance_id': metadata['instance_id']}
                )
            
            return success
            
        except Exception as e:
            self.alert_manager.send_alert(
                'error',
                f'Error during instance termination: {str(e)}',
                {'instance_id': metadata.get('instance_id', 'unknown'), 'error': str(e)}
            )
            return False
    
    def tail_logs(self, lines: int = 50) -> None:
        """Tail logs from running instance"""
        status = self.get_status()
        
        if status.get('state') != 'running':
            console.print("[red]No running instance found[/red]")
            return
        
        public_ip = status.get('public_ip')
        if not public_ip or public_ip == 'N/A':
            console.print("[red]Instance has no public IP[/red]")
            return
        
        # SSH into instance and tail logs
        ssh_key = os.getenv('SSH_KEY_PATH', '~/.ssh/id_rsa')
        ssh_command = [
            'ssh',
            '-o', 'StrictHostKeyChecking=no',
            '-i', ssh_key,
            f'ubuntu@{public_ip}',
            f'tail -f -n {lines} /var/log/training.log'
        ]
        
        console.print(f"[green]Connecting to {public_ip}...[/green]")
        subprocess.run(ssh_command)
    
    def _create_bootstrap_script(self, git_sha: str, resume_from: Optional[str]) -> str:
        """Create instance bootstrap script"""
        resume_cmd = f"--resume-from {resume_from}" if resume_from else ""
        
        script = f"""#!/bin/bash
set -e

# Log all output
exec &> >(tee -a /var/log/training.log)

echo "Starting bootstrap at $(date)"

# Update system
apt-get update
apt-get install -y git python3-pip python3-venv

# Clone repository at specific commit
cd /home/ubuntu
git clone {os.getenv('REPO_URL', 'https://github.com/aimerib/smollmfinetune.git')} repo
cd repo
git checkout {git_sha}

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r app/requirements.txt

# Setup AWS credentials from instance profile
aws configure set region {os.getenv('AWS_REGION', 'us-east-1')}

# Export instance ID for heartbeat
export INSTANCE_ID=$(ec2-metadata --instance-id | cut -d " " -f 2)

# Start training with resume
cd app
python scripts/run_sft.py \\
    --model-name "HuggingFaceTB/SmolLM2-360M-Instruct" \\
    --max-steps 10000 \\
    --save-steps 500 \\
    --shard-size-gb 2.0 \\
    --s3-bucket {os.getenv('S3_CHECKPOINT_BUCKET', 'training-checkpoints')} \\
    --s3-prefix "runs/{git_sha}/" \\
    {resume_cmd}

echo "Training completed at $(date)"
"""
        return script
    
    def _save_instance_metadata(self, instance_info: Dict[str, Any], 
                               git_sha: str, resume_from: Optional[str]) -> None:
        """Save instance metadata locally"""
        metadata = {
            **instance_info,
            'git_sha': git_sha,
            'resume_from': resume_from,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        Path('.spot_instance.json').write_text(json.dumps(metadata, indent=2))
    
    def _load_instance_metadata(self) -> Optional[Dict[str, Any]]:
        """Load saved instance metadata"""
        metadata_path = Path('.spot_instance.json')
        if not metadata_path.exists():
            return None
        
        try:
            return json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            return None


def check_and_resume(monitor: HeartbeatMonitor, orchestrator: SpotOrchestrator,
                    checkpoint: str) -> Dict[str, str]:
    """Check heartbeat and resume if needed (called by GitHub Action)"""
    if monitor.is_stale(threshold_minutes=10):
        logger.info("Heartbeat is stale, resuming training...")
        orchestrator.deploy(resume_from=checkpoint)
        return {'action': 'resumed'}
    else:
        return {'action': 'healthy'}


# CLI Commands
@click.group()
def cli():
    """Spot Instance Training Orchestrator"""
    pass


@cli.command()
@click.option('--instance-type', default='p4d.24xlarge', 
              help='Instance type (e.g., p4d.24xlarge, RTX_A5000)')
@click.option('--max-price', default=10.0, type=float,
              help='Maximum spot price per hour')
@click.option('--provider', default='aws', type=click.Choice(['aws', 'runpod']),
              help='Cloud provider')
@click.option('--resume', is_flag=True, help='Resume from checkpoint')
@click.option('--checkpoint', help='Checkpoint path to resume from')
@click.option('--use-chaining', is_flag=True, help='Try multiple instance types for best price')
@click.option('--use-optimizer', is_flag=True, help='Use cost optimizer to select instance')
def deploy(instance_type: str, max_price: float, provider: str, 
           resume: bool, checkpoint: Optional[str], use_chaining: bool, use_optimizer: bool):
    """Deploy a new spot instance for training with advanced features"""
    console.print("[bold blue]🚀 Deploying spot instance...[/bold blue]")
    
    orchestrator = SpotOrchestrator(provider=provider)
    
    try:
        result = orchestrator.deploy(
            instance_type=instance_type,
            max_price=max_price,
            resume_from=checkpoint if resume else None,
            use_chaining=use_chaining,
            use_optimizer=use_optimizer
        )
        
        console.print(f"[green]✅ Instance deployed successfully![/green]")
        console.print(f"Instance ID: {result['instance_id']}")
        console.print(f"Status: {result['status']}")
        console.print(f"Provider: {result.get('provider', 'unknown')}")
        
        if use_optimizer or use_chaining:
            console.print(f"[blue]💡 Cost optimization was used[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Deployment failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--provider', default='aws', type=click.Choice(['aws', 'runpod']),
              help='Cloud provider')
def status(provider: str):
    """Check status of current spot instance"""
    orchestrator = SpotOrchestrator(provider=provider)
    status_info = orchestrator.get_status()
    
    if status_info.get('state') == 'no_instance':
        console.print("[yellow]No active instance found[/yellow]")
        return
    
    # Create status table
    table = Table(title="Instance Status")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Instance", status_info.get('instance_id', 'N/A'))
    table.add_row("State", status_info.get('state', 'unknown'))
    
    if status_info.get('state') == 'running':
        table.add_row("Public IP", status_info.get('public_ip', 'N/A'))
        table.add_row("Uptime", f"{status_info.get('uptime_hours', 0)} hours")
        table.add_row("Cost", f"${status_info.get('estimated_cost', 0):.2f}")
    
    console.print(table)
    
    # Show daily spend
    cost_tracker = CostTracker(float(os.getenv('DAILY_BUDGET', '100.0')))
    daily_spend = cost_tracker.get_daily_spend()
    budget_status = "✓ Under budget" if cost_tracker.is_under_budget() else "✗ Over budget!"
    
    console.print(f"\nDaily spend: ${daily_spend:.2f} [{budget_status}]")


@cli.command()
@click.option('--provider', default='aws', type=click.Choice(['aws', 'runpod']),
              help='Cloud provider')
@click.confirmation_option(prompt='Are you sure you want to terminate the instance?')
def terminate(provider: str):
    """Terminate the current spot instance"""
    console.print("[yellow]Terminating instance...[/yellow]")
    
    orchestrator = SpotOrchestrator(provider=provider)
    success = orchestrator.terminate()
    
    if success:
        console.print("[green]✓ Instance terminated successfully[/green]")
    else:
        console.print("[red]✗ Failed to terminate instance[/red]")
        sys.exit(1)


@cli.command('tail-logs')
@click.option('--lines', default=50, type=int, help='Number of lines to tail')
@click.option('--provider', default='aws', type=click.Choice(['aws', 'runpod']),
              help='Cloud provider')
def tail_logs(lines: int, provider: str):
    """Tail training logs from running instance"""
    orchestrator = SpotOrchestrator(provider=provider)
    orchestrator.tail_logs(lines=lines)


@cli.command()
@click.option('--training-hours', default=8, type=int, help='Expected training duration in hours')
@click.option('--min-gpu-memory', default=16, type=int, help='Minimum GPU memory in GB')
@click.option('--max-budget', default=50.0, type=float, help='Maximum hourly budget')
def optimize(training_hours: int, min_gpu_memory: int, max_budget: float):
    """Get cost optimization suggestions"""
    console.print("[bold blue]🎯 Analyzing cost optimization opportunities...[/bold blue]")
    
    optimizer = CostOptimizer()
    
    # Get optimal instance
    try:
        optimal_instance = optimizer.get_optimal_instance({
            'min_gpu_memory_gb': min_gpu_memory,
            'max_hourly_cost': max_budget
        })
        
        console.print(f"[green]💡 Recommended Instance: {optimal_instance['type']}[/green]")
        console.print(f"Provider: {optimal_instance['provider']}")
        console.print(f"Max Price: ${optimal_instance['max_price']:.2f}/hour")
        
        # Get scheduling suggestions
        schedule = optimizer.suggest_schedule(training_hours)
        console.print(f"\n[blue]📅 Scheduling Recommendation:[/blue]")
        console.print(f"Strategy: {schedule['strategy']}")
        console.print(f"Estimated Savings: {schedule['estimated_savings']}")
        
        if schedule['strategy'] == 'single_session':
            console.print(f"Next optimal start: {schedule['start_time'].strftime('%Y-%m-%d %H:%M UTC')}")
        else:
            console.print(f"Sessions needed: {schedule['sessions']}")
            console.print(f"First session: {schedule['start_time'].strftime('%Y-%m-%d %H:%M UTC')}")
            
    except ValueError as e:
        console.print(f"[red]❌ {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--days', default=7, type=int, help='Number of days to show metrics for')
def metrics(days: int):
    """View collected metrics and analytics"""
    console.print("[bold blue]📊 Metrics Dashboard[/bold blue]")
    
    # Load cost data
    cost_tracker = CostTracker(float(os.getenv('DAILY_BUDGET', '100.0')))
    daily_spend = cost_tracker.get_daily_spend()
    
    # Create metrics table
    table = Table(title="Cost Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Today's Spend", f"${daily_spend:.2f}")
    table.add_row("Daily Budget", f"${cost_tracker.daily_budget:.2f}")
    table.add_row("Budget Remaining", f"${cost_tracker.daily_budget - daily_spend:.2f}")
    table.add_row("Budget Usage", f"{(daily_spend/cost_tracker.daily_budget)*100:.1f}%")
    
    console.print(table)
    
    # Show instance chain analytics if available
    chain_path = Path('.instance_chain.json')
    if chain_path.exists():
        try:
            chain_data = json.loads(chain_path.read_text())
            console.print(f"\n[blue]🔗 Last Deployment Analysis:[/blue]")
            console.print(f"Successful after {chain_data['attempts']} attempts")
            console.print(f"Used: {chain_data['successful_config']['type']} on {chain_data['successful_config']['provider']}")
        except json.JSONDecodeError:
            pass


@cli.command()
@click.option('--type', 'alert_type', default='info', type=click.Choice(['info', 'warning', 'error', 'success']),
              help='Type of test alert')
@click.option('--message', default='Test alert from spotctl', help='Test message')
def test_alerts(alert_type: str, message: str):
    """Test the alert system"""
    console.print("[bold blue]🔔 Testing alert system...[/bold blue]")
    
    alert_manager = AlertManager()
    
    # Check which channels are configured
    channels = []
    if alert_manager.slack_webhook:
        channels.append("Slack")
    if alert_manager.discord_webhook:
        channels.append("Discord")
    if all(alert_manager.email_config.values()):
        channels.append("Email")
    
    if not channels:
        console.print("[yellow]⚠️  No alert channels configured. Set SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL, or email env vars.[/yellow]")
        return
    
    console.print(f"Configured channels: {', '.join(channels)}")
    
    # Send test alert
    alert_manager.send_alert(
        alert_type,
        message,
        {
            'test_mode': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'channels': channels
        }
    )
    
    console.print(f"[green]✅ Test {alert_type} alert sent![/green]")


@cli.command()
@click.option('--days', default=30, type=int, help='Number of days to analyze')
def analytics(days: int):
    """View detailed cost analytics and usage patterns"""
    console.print("[bold blue]📈 Cost Analytics Dashboard[/bold blue]")
    
    # This would typically read from a more sophisticated cost tracking system
    # For now, we'll show basic information
    cost_tracker = CostTracker(float(os.getenv('DAILY_BUDGET', '100.0')))
    
    console.print(f"\n[blue]📊 {days}-Day Analysis[/blue]")
    console.print("Feature coming soon: Historical cost analysis, usage patterns, and savings recommendations")
    
    # Show current day summary
    daily_spend = cost_tracker.get_daily_spend()
    console.print(f"\n[green]Today's Summary:[/green]")
    console.print(f"• Spend: ${daily_spend:.2f}")
    console.print(f"• Budget: ${cost_tracker.daily_budget:.2f}")
    console.print(f"• Efficiency: {(1 - daily_spend/cost_tracker.daily_budget)*100:.1f}% budget remaining")
    
    # Show optimization suggestions
    if daily_spend > cost_tracker.daily_budget * 0.8:
        console.print("\n[yellow]💡 Optimization Suggestions:[/yellow]")
        console.print("• Consider using --use-optimizer flag for future deployments")
        console.print("• Try scheduling training during off-peak hours (10 PM - 6 AM UTC)")
        console.print("• Use --use-chaining to try cheaper instances first")


if __name__ == '__main__':
    cli() 