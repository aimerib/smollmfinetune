"""
Studio Performance Monitor for Multimodal Studio

Provides comprehensive performance monitoring including:
- Real-time system metrics collection
- Performance bottleneck detection
- Optimization recommendations
- Historical performance tracking
- Resource usage monitoring
"""

import asyncio
import json
import psutil
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    SYSTEM = "system"
    JOB_PERFORMANCE = "job_performance"
    STUDIO_USAGE = "studio_usage"
    RESOURCE_UTILIZATION = "resource_utilization"

class BottleneckSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    disk_io_read: float
    disk_io_write: float
    network_io_sent: float
    network_io_recv: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    load_average: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class JobMetrics:
    """Job performance metrics"""
    job_id: str
    processing_time: float
    memory_peak: float
    cpu_peak: float
    throughput: float  # samples per second
    error_rate: float
    queue_wait_time: float
    status: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class PerformanceBottleneck:
    """Detected performance bottleneck"""
    id: str
    type: str
    severity: BottleneckSeverity
    title: str
    description: str
    affected_components: List[str]
    metrics: Dict[str, Any]
    suggestions: List[str]
    detected_at: datetime = None
    
    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['detected_at'] = self.detected_at.isoformat()
        data['severity'] = self.severity.value
        return data

@dataclass
class OptimizationSuggestion:
    """AI-powered optimization suggestion"""
    id: str
    title: str
    description: str
    impact: str
    implementation_effort: str
    priority: int  # 1-10, 1 being highest
    category: str  # e.g., "memory", "cpu", "io", "workflow"
    expected_improvement: str
    metadata: Dict[str, Any]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

class SystemMetricsCollector:
    """Collects system-level performance metrics"""
    
    def __init__(self):
        self._last_disk_io = None
        self._last_network_io = None
        self._last_time = None
        
    async def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        
        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent
        
        # Disk I/O metrics
        disk_io = psutil.disk_io_counters()
        current_time = time.time()
        
        if self._last_disk_io and self._last_time:
            time_delta = current_time - self._last_time
            disk_io_read = (disk_io.read_bytes - self._last_disk_io.read_bytes) / time_delta / 1024 / 1024  # MB/s
            disk_io_write = (disk_io.write_bytes - self._last_disk_io.write_bytes) / time_delta / 1024 / 1024  # MB/s
        else:
            disk_io_read = 0
            disk_io_write = 0
        
        self._last_disk_io = disk_io
        
        # Network I/O metrics
        network_io = psutil.net_io_counters()
        
        if self._last_network_io and self._last_time:
            time_delta = current_time - self._last_time
            network_io_sent = (network_io.bytes_sent - self._last_network_io.bytes_sent) / time_delta / 1024 / 1024  # MB/s
            network_io_recv = (network_io.bytes_recv - self._last_network_io.bytes_recv) / time_delta / 1024 / 1024  # MB/s
        else:
            network_io_sent = 0
            network_io_recv = 0
        
        self._last_network_io = network_io
        self._last_time = current_time
        
        # Load average (Unix systems only)
        load_average = None
        try:
            load_average = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else None
        except:
            pass
        
        # GPU metrics (if available)
        gpu_usage = None
        gpu_memory = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_usage = gpu.load * 100
                gpu_memory = gpu.memoryUtil * 100
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to get GPU metrics: {e}")
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            disk_io_read=disk_io_read,
            disk_io_write=disk_io_write,
            network_io_sent=network_io_sent,
            network_io_recv=network_io_recv,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            load_average=load_average
        )

class BottleneckDetector:
    """Detects performance bottlenecks from metrics"""
    
    def __init__(self):
        self.thresholds = {
            'cpu_high': 80.0,
            'cpu_critical': 95.0,
            'memory_high': 85.0,
            'memory_critical': 95.0,
            'disk_high': 90.0,
            'disk_critical': 98.0,
            'disk_io_high': 100.0,  # MB/s
            'queue_time_high': 300.0,  # seconds
            'error_rate_high': 0.05,  # 5%
        }
    
    async def detect_bottlenecks(
        self,
        system_metrics: SystemMetrics,
        job_metrics: List[JobMetrics],
        historical_data: Optional[List[SystemMetrics]] = None
    ) -> List[PerformanceBottleneck]:
        """Detect performance bottlenecks from current metrics"""
        
        bottlenecks = []
        
        # System-level bottlenecks
        bottlenecks.extend(await self._detect_system_bottlenecks(system_metrics, historical_data))
        
        # Job-level bottlenecks
        bottlenecks.extend(await self._detect_job_bottlenecks(job_metrics))
        
        return bottlenecks
    
    async def _detect_system_bottlenecks(
        self,
        metrics: SystemMetrics,
        historical_data: Optional[List[SystemMetrics]] = None
    ) -> List[PerformanceBottleneck]:
        """Detect system-level bottlenecks"""
        
        bottlenecks = []
        
        # CPU bottleneck
        if metrics.cpu_usage >= self.thresholds['cpu_critical']:
            bottlenecks.append(PerformanceBottleneck(
                id=str(uuid.uuid4()),
                type="cpu",
                severity=BottleneckSeverity.CRITICAL,
                title="Critical CPU Usage",
                description=f"CPU usage is at {metrics.cpu_usage:.1f}%, indicating severe performance degradation",
                affected_components=["job_processing", "system_responsiveness"],
                metrics={"cpu_usage": metrics.cpu_usage},
                suggestions=[
                    "Reduce concurrent job processing",
                    "Optimize computationally intensive operations",
                    "Consider upgrading CPU or adding more cores",
                    "Implement CPU usage throttling"
                ]
            ))
        elif metrics.cpu_usage >= self.thresholds['cpu_high']:
            bottlenecks.append(PerformanceBottleneck(
                id=str(uuid.uuid4()),
                type="cpu",
                severity=BottleneckSeverity.HIGH,
                title="High CPU Usage",
                description=f"CPU usage is at {metrics.cpu_usage:.1f}%, which may cause performance issues",
                affected_components=["job_processing"],
                metrics={"cpu_usage": metrics.cpu_usage},
                suggestions=[
                    "Monitor CPU usage trends",
                    "Consider reducing batch sizes",
                    "Optimize algorithms for better CPU efficiency"
                ]
            ))
        
        # Memory bottleneck
        if metrics.memory_usage >= self.thresholds['memory_critical']:
            bottlenecks.append(PerformanceBottleneck(
                id=str(uuid.uuid4()),
                type="memory",
                severity=BottleneckSeverity.CRITICAL,
                title="Critical Memory Usage",
                description=f"Memory usage is at {metrics.memory_usage:.1f}%, risking system instability",
                affected_components=["job_processing", "system_stability"],
                metrics={"memory_usage": metrics.memory_usage},
                suggestions=[
                    "Immediately reduce memory-intensive operations",
                    "Clear unnecessary data from memory",
                    "Restart memory-heavy processes",
                    "Consider adding more RAM"
                ]
            ))
        elif metrics.memory_usage >= self.thresholds['memory_high']:
            bottlenecks.append(PerformanceBottleneck(
                id=str(uuid.uuid4()),
                type="memory",
                severity=BottleneckSeverity.HIGH,
                title="High Memory Usage",
                description=f"Memory usage is at {metrics.memory_usage:.1f}%, approaching concerning levels",
                affected_components=["job_processing"],
                metrics={"memory_usage": metrics.memory_usage},
                suggestions=[
                    "Monitor memory usage patterns",
                    "Implement memory cleanup routines",
                    "Optimize data structures for memory efficiency"
                ]
            ))
        
        # Disk space bottleneck
        if metrics.disk_usage >= self.thresholds['disk_critical']:
            bottlenecks.append(PerformanceBottleneck(
                id=str(uuid.uuid4()),
                type="disk_space",
                severity=BottleneckSeverity.CRITICAL,
                title="Critical Disk Space",
                description=f"Disk usage is at {metrics.disk_usage:.1f}%, system may become unstable",
                affected_components=["data_storage", "job_output", "system_logs"],
                metrics={"disk_usage": metrics.disk_usage},
                suggestions=[
                    "Immediately free up disk space",
                    "Delete old export files",
                    "Clean up temporary files",
                    "Archive old datasets"
                ]
            ))
        
        # Disk I/O bottleneck
        if metrics.disk_io_read + metrics.disk_io_write >= self.thresholds['disk_io_high']:
            bottlenecks.append(PerformanceBottleneck(
                id=str(uuid.uuid4()),
                type="disk_io",
                severity=BottleneckSeverity.MEDIUM,
                title="High Disk I/O",
                description=f"Disk I/O is at {metrics.disk_io_read + metrics.disk_io_write:.1f} MB/s, may cause slowdowns",
                affected_components=["data_processing", "export_operations"],
                metrics={
                    "disk_io_read": metrics.disk_io_read,
                    "disk_io_write": metrics.disk_io_write
                },
                suggestions=[
                    "Optimize file access patterns",
                    "Use SSD storage for better I/O performance",
                    "Implement data caching strategies"
                ]
            ))
        
        return bottlenecks
    
    async def _detect_job_bottlenecks(self, job_metrics: List[JobMetrics]) -> List[PerformanceBottleneck]:
        """Detect job-level bottlenecks"""
        
        bottlenecks = []
        
        if not job_metrics:
            return bottlenecks
        
        # Calculate average metrics
        avg_queue_time = sum(m.queue_wait_time for m in job_metrics) / len(job_metrics)
        avg_error_rate = sum(m.error_rate for m in job_metrics) / len(job_metrics)
        
        # Queue time bottleneck
        if avg_queue_time >= self.thresholds['queue_time_high']:
            bottlenecks.append(PerformanceBottleneck(
                id=str(uuid.uuid4()),
                type="queue_time",
                severity=BottleneckSeverity.HIGH,
                title="High Job Queue Times",
                description=f"Average queue time is {avg_queue_time:.1f} seconds, indicating processing bottleneck",
                affected_components=["job_queue", "user_experience"],
                metrics={"avg_queue_time": avg_queue_time},
                suggestions=[
                    "Increase processing workers",
                    "Optimize job processing algorithms",
                    "Implement job prioritization",
                    "Consider horizontal scaling"
                ]
            ))
        
        # Error rate bottleneck
        if avg_error_rate >= self.thresholds['error_rate_high']:
            bottlenecks.append(PerformanceBottleneck(
                id=str(uuid.uuid4()),
                type="error_rate",
                severity=BottleneckSeverity.HIGH,
                title="High Job Error Rate",
                description=f"Job error rate is {avg_error_rate:.1%}, indicating system issues",
                affected_components=["job_processing", "data_quality"],
                metrics={"avg_error_rate": avg_error_rate},
                suggestions=[
                    "Investigate job failure causes",
                    "Improve error handling and recovery",
                    "Validate input data quality",
                    "Check system resource availability"
                ]
            ))
        
        return bottlenecks

class OptimizationEngine:
    """Generates AI-powered optimization suggestions"""
    
    async def generate_suggestions(
        self,
        system_metrics: SystemMetrics,
        job_metrics: List[JobMetrics],
        bottlenecks: List[PerformanceBottleneck],
        user_id: str
    ) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on current state"""
        
        suggestions = []
        
        # System optimization suggestions
        suggestions.extend(await self._generate_system_suggestions(system_metrics, bottlenecks))
        
        # Job optimization suggestions
        suggestions.extend(await self._generate_job_suggestions(job_metrics, bottlenecks))
        
        # Workflow optimization suggestions
        suggestions.extend(await self._generate_workflow_suggestions(system_metrics, job_metrics))
        
        return suggestions
    
    async def _generate_system_suggestions(
        self,
        metrics: SystemMetrics,
        bottlenecks: List[PerformanceBottleneck]
    ) -> List[OptimizationSuggestion]:
        """Generate system-level optimization suggestions"""
        
        suggestions = []
        
        # Memory optimization
        if metrics.memory_usage > 70:
            suggestions.append(OptimizationSuggestion(
                id=str(uuid.uuid4()),
                title="Optimize Memory Usage",
                description="Implement memory cleanup and optimization strategies",
                impact="Reduce memory usage by 15-30%",
                implementation_effort="Medium",
                priority=3,
                category="memory",
                expected_improvement="Faster processing, reduced memory pressure",
                metadata={
                    "current_usage": metrics.memory_usage,
                    "target_usage": max(50, metrics.memory_usage - 20)
                }
            ))
        
        # CPU optimization
        if metrics.cpu_usage > 60:
            suggestions.append(OptimizationSuggestion(
                id=str(uuid.uuid4()),
                title="Implement CPU Load Balancing",
                description="Distribute CPU-intensive tasks more evenly",
                impact="Improve system responsiveness by 20-40%",
                implementation_effort="Medium",
                priority=2,
                category="cpu",
                expected_improvement="Smoother user experience, better multitasking",
                metadata={
                    "current_usage": metrics.cpu_usage,
                    "recommended_max": 80
                }
            ))
        
        # Storage optimization
        if metrics.disk_usage > 80:
            suggestions.append(OptimizationSuggestion(
                id=str(uuid.uuid4()),
                title="Implement Storage Cleanup",
                description="Set up automated cleanup of old files and temporary data",
                impact="Free up 20-50% disk space",
                implementation_effort="Low",
                priority=1,
                category="storage",
                expected_improvement="Prevent storage issues, improve system stability",
                metadata={
                    "current_usage": metrics.disk_usage,
                    "cleanup_targets": ["old_exports", "temp_files", "logs"]
                }
            ))
        
        return suggestions
    
    async def _generate_job_suggestions(
        self,
        job_metrics: List[JobMetrics],
        bottlenecks: List[PerformanceBottleneck]
    ) -> List[OptimizationSuggestion]:
        """Generate job processing optimization suggestions"""
        
        suggestions = []
        
        if not job_metrics:
            return suggestions
        
        # Analyze job performance patterns
        avg_processing_time = sum(m.processing_time for m in job_metrics) / len(job_metrics)
        avg_throughput = sum(m.throughput for m in job_metrics) / len(job_metrics)
        
        # Processing time optimization
        if avg_processing_time > 300:  # 5 minutes
            suggestions.append(OptimizationSuggestion(
                id=str(uuid.uuid4()),
                title="Optimize Job Processing Time",
                description="Implement parallel processing and algorithm optimization",
                impact="Reduce processing time by 30-50%",
                implementation_effort="High",
                priority=2,
                category="workflow",
                expected_improvement="Faster job completion, better user experience",
                metadata={
                    "current_avg_time": avg_processing_time,
                    "target_time": avg_processing_time * 0.6
                }
            ))
        
        # Throughput optimization
        if avg_throughput < 1.0:  # Less than 1 sample per second
            suggestions.append(OptimizationSuggestion(
                id=str(uuid.uuid4()),
                title="Improve Processing Throughput",
                description="Optimize data pipelines and batch processing",
                impact="Increase throughput by 2-5x",
                implementation_effort="Medium",
                priority=3,
                category="workflow",
                expected_improvement="Process more data in less time",
                metadata={
                    "current_throughput": avg_throughput,
                    "target_throughput": avg_throughput * 3
                }
            ))
        
        return suggestions
    
    async def _generate_workflow_suggestions(
        self,
        system_metrics: SystemMetrics,
        job_metrics: List[JobMetrics]
    ) -> List[OptimizationSuggestion]:
        """Generate workflow optimization suggestions"""
        
        suggestions = []
        
        # Caching suggestions
        suggestions.append(OptimizationSuggestion(
            id=str(uuid.uuid4()),
            title="Implement Result Caching",
            description="Cache frequently used datasets and processing results",
            impact="Reduce processing time by 40-60% for repeated operations",
            implementation_effort="Medium",
            priority=4,
            category="workflow",
            expected_improvement="Faster response times, reduced resource usage",
            metadata={
                "cache_types": ["dataset_cache", "validation_cache", "export_cache"]
            }
        ))
        
        # Batch processing suggestions
        suggestions.append(OptimizationSuggestion(
            id=str(uuid.uuid4()),
            title="Optimize Batch Processing",
            description="Implement smart batching based on system resources",
            impact="Improve resource utilization by 25-40%",
            implementation_effort="Medium",
            priority=5,
            category="workflow",
            expected_improvement="Better resource utilization, more consistent performance",
            metadata={
                "current_cpu": system_metrics.cpu_usage,
                "current_memory": system_metrics.memory_usage
            }
        ))
        
        return suggestions

class StudioPerformanceMonitor:
    """Main performance monitoring service"""
    
    def __init__(self):
        self.metrics_collector = SystemMetricsCollector()
        self.bottleneck_detector = BottleneckDetector()
        self.optimization_engine = OptimizationEngine()
        
        self.storage_path = Path("data/performance_metrics")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[SystemMetrics] = []
        self.job_metrics_history: Dict[str, List[JobMetrics]] = {}
        self.max_history_size = 1000
        
    async def collect_studio_metrics(self, user_id: str) -> Dict[str, Any]:
        """Collect comprehensive studio metrics for user"""
        
        # Collect system metrics
        system_metrics = await self.metrics_collector.collect_metrics()
        
        # Get job metrics for user
        job_metrics = self.job_metrics_history.get(user_id, [])
        recent_job_metrics = job_metrics[-50:] if job_metrics else []  # Last 50 jobs
        
        # Store in history
        self.metrics_history.append(system_metrics)
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        
        return {
            'system_metrics': system_metrics.to_dict(),
            'job_metrics': [m.to_dict() for m in recent_job_metrics],
            'summary': {
                'total_jobs': len(job_metrics),
                'recent_jobs': len(recent_job_metrics),
                'system_health': await self._calculate_system_health(system_metrics),
                'performance_trend': await self._calculate_performance_trend()
            }
        }
    
    async def detect_bottlenecks(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks from metrics"""
        
        system_metrics = SystemMetrics(**metrics['system_metrics'])
        job_metrics = [JobMetrics(**m) for m in metrics.get('job_metrics', [])]
        
        bottlenecks = await self.bottleneck_detector.detect_bottlenecks(
            system_metrics=system_metrics,
            job_metrics=job_metrics,
            historical_data=self.metrics_history[-10:]  # Last 10 data points
        )
        
        return [b.to_dict() for b in bottlenecks]
    
    async def generate_optimization_suggestions(self, user_id: str) -> List[Dict[str, Any]]:
        """Generate optimization suggestions for user"""
        
        # Get current metrics
        metrics = await self.collect_studio_metrics(user_id)
        
        # Detect bottlenecks
        bottlenecks_data = await self.detect_bottlenecks(metrics)
        bottlenecks = [PerformanceBottleneck(**b) for b in bottlenecks_data]
        
        # Generate suggestions
        system_metrics = SystemMetrics(**metrics['system_metrics'])
        job_metrics = [JobMetrics(**m) for m in metrics.get('job_metrics', [])]
        
        suggestions = await self.optimization_engine.generate_suggestions(
            system_metrics=system_metrics,
            job_metrics=job_metrics,
            bottlenecks=bottlenecks,
            user_id=user_id
        )
        
        return [s.to_dict() for s in suggestions]
    
    async def get_metrics_history(
        self,
        user_id: str,
        since: datetime,
        metric_type: Optional[MetricType] = None
    ) -> Dict[str, Any]:
        """Get historical metrics data"""
        
        # Filter system metrics by time
        filtered_system_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= since
        ]
        
        # Filter job metrics by time and user
        user_job_metrics = self.job_metrics_history.get(user_id, [])
        filtered_job_metrics = [
            m for m in user_job_metrics
            if m.timestamp >= since
        ]
        
        return {
            'system_metrics': [m.to_dict() for m in filtered_system_metrics],
            'job_metrics': [m.to_dict() for m in filtered_job_metrics],
            'time_range': {
                'start': since.isoformat(),
                'end': datetime.utcnow().isoformat()
            },
            'summary': {
                'system_data_points': len(filtered_system_metrics),
                'job_data_points': len(filtered_job_metrics)
            }
        }
    
    async def record_job_metrics(self, user_id: str, job_metrics: JobMetrics) -> None:
        """Record job performance metrics"""
        
        if user_id not in self.job_metrics_history:
            self.job_metrics_history[user_id] = []
        
        self.job_metrics_history[user_id].append(job_metrics)
        
        # Limit history size per user
        if len(self.job_metrics_history[user_id]) > self.max_history_size:
            self.job_metrics_history[user_id] = self.job_metrics_history[user_id][-self.max_history_size:]
    
    async def _calculate_system_health(self, metrics: SystemMetrics) -> str:
        """Calculate overall system health score"""
        
        # Simple health calculation based on resource usage
        health_score = 100.0
        
        # CPU impact
        if metrics.cpu_usage > 90:
            health_score -= 30
        elif metrics.cpu_usage > 70:
            health_score -= 15
        
        # Memory impact
        if metrics.memory_usage > 90:
            health_score -= 30
        elif metrics.memory_usage > 70:
            health_score -= 15
        
        # Disk impact
        if metrics.disk_usage > 95:
            health_score -= 25
        elif metrics.disk_usage > 85:
            health_score -= 10
        
        health_score = max(0, health_score)
        
        if health_score >= 80:
            return "excellent"
        elif health_score >= 60:
            return "good"
        elif health_score >= 40:
            return "fair"
        else:
            return "poor"
    
    async def _calculate_performance_trend(self) -> str:
        """Calculate performance trend from recent history"""
        
        if len(self.metrics_history) < 5:
            return "insufficient_data"
        
        # Compare recent metrics to older ones
        recent_metrics = self.metrics_history[-5:]
        older_metrics = self.metrics_history[-10:-5] if len(self.metrics_history) >= 10 else []
        
        if not older_metrics:
            return "stable"
        
        # Calculate average resource usage
        recent_avg = sum(m.cpu_usage + m.memory_usage for m in recent_metrics) / len(recent_metrics)
        older_avg = sum(m.cpu_usage + m.memory_usage for m in older_metrics) / len(older_metrics)
        
        change_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
        
        if change_percent > 10:
            return "degrading"
        elif change_percent < -10:
            return "improving"
        else:
            return "stable"
    
    async def cleanup_old_metrics(self, days: int = 30) -> None:
        """Cleanup old metrics data"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Clean system metrics
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_date
        ]
        
        # Clean job metrics
        for user_id in list(self.job_metrics_history.keys()):
            self.job_metrics_history[user_id] = [
                m for m in self.job_metrics_history[user_id]
                if m.timestamp >= cutoff_date
            ]
            
            # Remove empty user entries
            if not self.job_metrics_history[user_id]:
                del self.job_metrics_history[user_id]
        
        logger.info(f"Cleaned up metrics older than {days} days") 