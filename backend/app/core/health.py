"""
Health Check System for Production Deployment

Provides comprehensive health monitoring for the application including
database connectivity, Redis availability, and system resources.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class HealthChecker:
    """Comprehensive health check system for production deployment"""
    
    def __init__(self, db_path: str = "data/platform.db"):
        self.db_path = db_path
        self.start_time = time.time()
    
    def check_database(self) -> Dict[str, Any]:
        """Check database connectivity and basic operations"""
        try:
            db_file = Path(self.db_path)
            if not db_file.exists():
                return {
                    "status": "unhealthy",
                    "error": "Database file does not exist",
                    "path": str(db_file.absolute())
                }
            
            # Test database connection and basic query
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                
                if result != (1,):
                    return {
                        "status": "unhealthy",
                        "error": "Database query returned unexpected result"
                    }
                
                # Check if tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                return {
                    "status": "healthy",
                    "tables_count": len(tables),
                    "tables": tables[:5] if len(tables) <= 5 else tables[:5] + ["..."],
                    "file_size_mb": round(db_file.stat().st_size / (1024 * 1024), 2)
                }
                
        except sqlite3.OperationalError as e:
            return {
                "status": "unhealthy",
                "error": f"Database connection failed: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Database health check failed: {str(e)}"
            }
    
    def check_redis(self, redis_url: Optional[str] = None) -> Dict[str, Any]:
        """Check Redis connectivity if available"""
        if not redis_url:
            return {
                "status": "not_configured",
                "message": "Redis not configured"
            }
        
        try:
            import redis
            client = redis.from_url(redis_url, socket_connect_timeout=5)
            
            # Test basic operations
            test_key = f"healthcheck:{int(time.time())}"
            client.set(test_key, "test", ex=10)  # Expire in 10 seconds
            value = client.get(test_key)
            client.delete(test_key)
            
            if value != b"test":
                return {
                    "status": "unhealthy",
                    "error": "Redis read/write test failed"
                }
            
            info = client.info()
            return {
                "status": "healthy",
                "version": info.get("redis_version"),
                "memory_used_mb": round(info.get("used_memory", 0) / (1024 * 1024), 2),
                "connected_clients": info.get("connected_clients", 0)
            }
            
        except ImportError:
            return {
                "status": "not_available",
                "message": "Redis client not installed"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Redis connection failed: {str(e)}"
            }
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = round(memory.used / (1024 ** 3), 2)
            memory_total_gb = round(memory.total / (1024 ** 3), 2)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_used_gb = round(disk.used / (1024 ** 3), 2)
            disk_total_gb = round(disk.total / (1024 ** 3), 2)
            
            # GPU usage if available
            gpu_info = None
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_info = {
                        "name": gpu.name,
                        "memory_used_mb": round(gpu.memoryUsed, 2),
                        "memory_total_mb": round(gpu.memoryTotal, 2),
                        "memory_percent": round((gpu.memoryUsed / gpu.memoryTotal) * 100, 1),
                        "temperature": gpu.temperature,
                        "load_percent": round(gpu.load * 100, 1)
                    }
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"GPU monitoring failed: {e}")
            
            # Determine overall health
            status = "healthy"
            warnings = []
            
            if cpu_percent > 90:
                warnings.append("High CPU usage")
                status = "warning"
            if memory_percent > 90:
                warnings.append("High memory usage")
                status = "warning"
            if disk_percent > 90:
                warnings.append("High disk usage")
                status = "warning"
            
            if cpu_percent > 95 or memory_percent > 95 or disk_percent > 95:
                status = "critical"
            
            return {
                "status": status,
                "warnings": warnings,
                "cpu_percent": cpu_percent,
                "memory": {
                    "percent": memory_percent,
                    "used_gb": memory_used_gb,
                    "total_gb": memory_total_gb
                },
                "disk": {
                    "percent": round(disk_percent, 1),
                    "used_gb": disk_used_gb,
                    "total_gb": disk_total_gb
                },
                "gpu": gpu_info
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"System resource check failed: {str(e)}"
            }
    
    def check_training_output_directory(self) -> Dict[str, Any]:
        """Check training output directory accessibility"""
        try:
            training_dir = Path("training_output")
            
            if not training_dir.exists():
                training_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write access
            test_file = training_dir / f"healthcheck_{int(time.time())}.tmp"
            test_file.write_text("health check test")
            
            if not test_file.exists():
                return {
                    "status": "unhealthy",
                    "error": "Cannot write to training output directory"
                }
            
            test_file.unlink()  # Clean up
            
            # Count existing files
            adapter_count = len(list((training_dir / "adapters").glob("*"))) if (training_dir / "adapters").exists() else 0
            prompt_count = len(list((training_dir / "prompts").glob("*"))) if (training_dir / "prompts").exists() else 0
            
            return {
                "status": "healthy",
                "path": str(training_dir.absolute()),
                "adapters_count": adapter_count,
                "prompts_count": prompt_count
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": f"Training directory check failed: {str(e)}"
            }
    
    def get_comprehensive_health(self, redis_url: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive health status for all systems"""
        health_checks = {
            "database": self.check_database(),
            "redis": self.check_redis(redis_url),
            "system": self.check_system_resources(),
            "storage": self.check_training_output_directory()
        }
        
        # Determine overall status
        overall_status = "healthy"
        critical_issues = []
        warnings = []
        
        for service, check in health_checks.items():
            if check["status"] == "unhealthy":
                overall_status = "unhealthy"
                critical_issues.append(f"{service}: {check.get('error', 'Unknown error')}")
            elif check["status"] == "critical":
                overall_status = "critical" if overall_status != "unhealthy" else overall_status
                critical_issues.append(f"{service}: Critical resource usage")
            elif check["status"] == "warning":
                if overall_status == "healthy":
                    overall_status = "warning"
                warnings.extend(check.get("warnings", []))
        
        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": round(time.time() - self.start_time),
            "version": "1.0.0",  # Could be loaded from package info
            "checks": health_checks,
            "critical_issues": critical_issues,
            "warnings": warnings
        }


# Global health checker instance
_health_checker = None

def get_health_checker() -> HealthChecker:
    """Get global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker 