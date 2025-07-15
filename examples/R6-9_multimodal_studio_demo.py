#!/usr/bin/env python3
"""
R6-9 Multimodal Studio Production Features Demo

This script demonstrates all the production-ready features of the Multimodal Studio:
- Advanced Job Management with batch operations
- Dataset Quality Validation with comprehensive reporting
- Multi-format Export Management with progress tracking
- Real-time Performance Monitoring with bottleneck detection
- User Preferences and Workspace Management

Prerequisites:
- Running Character Creation Devkit backend (./start-devkit.sh)
- Valid user authentication
- Sample dataset for testing

Usage:
    python examples/R6-9_multimodal_studio_demo.py
"""

import asyncio
import json
import requests
import websockets
import time
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultimodalStudioDemo:
    """Demo client for Multimodal Studio Production Features"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", ws_base_url: str = "ws://localhost:8000"):
        self.api_base_url = api_base_url
        self.ws_base_url = ws_base_url
        self.session = requests.Session()
        self.auth_token = None
        
    def authenticate(self, username: str = "demo_user", password: str = "demo_password") -> bool:
        """Authenticate with the backend (simplified for demo)"""
        try:
            # In a real scenario, you'd use proper authentication
            # For demo purposes, we'll use a mock token
            self.auth_token = "demo_token_12345"
            self.session.headers.update({'Authorization': f'Bearer {self.auth_token}'})
            logger.info("✅ Authentication successful")
            return True
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            return False
    
    def demo_job_management(self):
        """Demonstrate advanced job management features"""
        logger.info("\n🎛️ === ADVANCED JOB MANAGEMENT DEMO ===")
        
        try:
            # Get current jobs
            response = self.session.get(f"{self.api_base_url}/api/multimodal-studio/jobs")
            if response.status_code == 200:
                jobs = response.json()
                logger.info(f"📋 Found {len(jobs)} existing jobs")
                
                if jobs:
                    # Demonstrate batch operations
                    job_ids = [job['id'] for job in jobs[:3]]  # Take first 3 jobs
                    
                    # Batch pause operation
                    batch_request = {
                        "action": "pause",
                        "job_ids": job_ids,
                        "options": {}
                    }
                    
                    response = self.session.post(
                        f"{self.api_base_url}/api/multimodal-studio/jobs/batch-action",
                        json=batch_request
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        logger.info(f"⏸️ Batch pause operation: {len(results['results'])} jobs affected")
                        for result in results['results']:
                            status = "✅" if result['success'] else "❌"
                            logger.info(f"  {status} Job {result['job_id'][:8]}...")
                    
                    # Update job priority
                    if job_ids:
                        response = self.session.post(
                            f"{self.api_base_url}/api/multimodal-studio/jobs/{job_ids[0]}/priority",
                            json={"priority": 2}  # High priority
                        )
                        
                        if response.status_code == 200:
                            logger.info(f"🔼 Updated job priority to HIGH")
                
                # Get queue status
                response = self.session.get(f"{self.api_base_url}/api/multimodal-studio/jobs/queue")
                if response.status_code == 200:
                    queue_info = response.json()
                    logger.info(f"📊 Queue Status:")
                    logger.info(f"   Queued: {queue_info.get('queued_count', 0)}")
                    logger.info(f"   Active: {queue_info.get('active_count', 0)}")
                    logger.info(f"   Estimated wait: {queue_info.get('estimated_wait_time', 0)} minutes")
            
        except Exception as e:
            logger.error(f"❌ Job management demo failed: {e}")
    
    def demo_quality_validation(self):
        """Demonstrate dataset quality validation features"""
        logger.info("\n🔍 === DATASET QUALITY VALIDATION DEMO ===")
        
        try:
            # Mock dataset ID for demo
            dataset_id = "demo_dataset_001"
            
            # Start quality validation
            response = self.session.post(
                f"{self.api_base_url}/api/multimodal-studio/datasets/{dataset_id}/validate"
            )
            
            if response.status_code == 200:
                validation_result = response.json()
                report_id = validation_result.get('report_id')
                logger.info(f"🚀 Quality validation started (Report ID: {report_id})")
                
                # Simulate waiting for validation to complete
                logger.info("⏳ Waiting for validation to complete...")
                time.sleep(3)  # In reality, this would be WebSocket updates
                
                # Get validation reports
                response = self.session.get(
                    f"{self.api_base_url}/api/multimodal-studio/datasets/{dataset_id}/quality-reports"
                )
                
                if response.status_code == 200:
                    reports = response.json()
                    logger.info(f"📊 Retrieved {len(reports)} quality reports")
                    
                    if reports:
                        latest_report = reports[0]
                        metrics = latest_report.get('metrics', {})
                        
                        logger.info("📈 Quality Metrics:")
                        logger.info(f"   Overall Score: {metrics.get('overall_score', 0):.1f}/100")
                        logger.info(f"   Consistency: {metrics.get('consistency_score', 0):.1f}/100")
                        logger.info(f"   Diversity: {metrics.get('diversity_score', 0):.1f}/100")
                        logger.info(f"   Quality: {metrics.get('quality_score', 0):.1f}/100")
                        logger.info(f"   Character Adherence: {metrics.get('character_adherence', 0):.1f}/100")
                        
                        issues = latest_report.get('issues', [])
                        suggestions = latest_report.get('suggestions', [])
                        
                        logger.info(f"⚠️ Found {len(issues)} issues and {len(suggestions)} suggestions")
                        
                        # Apply improvement suggestions
                        if suggestions:
                            suggestion_ids = [s['id'] for s in suggestions[:2]]  # Apply first 2
                            
                            response = self.session.post(
                                f"{self.api_base_url}/api/multimodal-studio/datasets/{dataset_id}/quality-suggestions/apply",
                                json={"suggestion_ids": suggestion_ids}
                            )
                            
                            if response.status_code == 200:
                                results = response.json()
                                applied_count = len([r for r in results.get('applied', []) if r.get('success')])
                                logger.info(f"✨ Applied {applied_count} improvement suggestions")
            
        except Exception as e:
            logger.error(f"❌ Quality validation demo failed: {e}")
    
    def demo_export_management(self):
        """Demonstrate multi-format export management"""
        logger.info("\n📦 === MULTI-FORMAT EXPORT MANAGEMENT DEMO ===")
        
        try:
            # Create export configurations for different formats
            export_configs = [
                {
                    "name": "HuggingFace Research Export",
                    "format": "huggingface",
                    "options": {
                        "description": "Character dialogue dataset for research",
                        "license": "MIT",
                        "citation": "Demo Research Paper"
                    }
                },
                {
                    "name": "JSONL Simple Export",
                    "format": "jsonl",
                    "options": {
                        "include_fields": ["text", "character", "emotion"]
                    }
                },
                {
                    "name": "PyTorch Training Export",
                    "format": "pytorch",
                    "options": {}
                }
            ]
            
            created_configs = []
            
            for config in export_configs:
                response = self.session.post(
                    f"{self.api_base_url}/api/multimodal-studio/export-configs",
                    json=config
                )
                
                if response.status_code == 200:
                    created_config = response.json()
                    created_configs.append(created_config)
                    logger.info(f"📝 Created export config: {config['name']} ({config['format']})")
            
            # Get all export configurations
            response = self.session.get(f"{self.api_base_url}/api/multimodal-studio/export-configs")
            if response.status_code == 200:
                all_configs = response.json()
                logger.info(f"📚 Total export configurations: {len(all_configs)}")
            
            # Start exports using the created configurations
            dataset_id = "demo_dataset_001"
            export_jobs = []
            
            for config in created_configs[:2]:  # Export with first 2 configs
                response = self.session.post(
                    f"{self.api_base_url}/api/multimodal-studio/datasets/{dataset_id}/export",
                    json={"config_id": config['id']}
                )
                
                if response.status_code == 200:
                    export_job = response.json()
                    export_jobs.append(export_job)
                    logger.info(f"🚀 Started export: {config['name']} (Job ID: {export_job['id'][:8]}...)")
            
            # Demonstrate batch export
            if len(created_configs) >= 2:
                response = self.session.post(
                    f"{self.api_base_url}/api/multimodal-studio/exports/batch",
                    json={
                        "dataset_ids": [dataset_id, "demo_dataset_002"],
                        "config_id": created_configs[0]['id']
                    }
                )
                
                if response.status_code == 200:
                    batch_jobs = response.json()
                    logger.info(f"📦 Started batch export: {len(batch_jobs.get('export_jobs', []))} jobs")
            
            # Get export history
            response = self.session.get(f"{self.api_base_url}/api/multimodal-studio/exports")
            if response.status_code == 200:
                exports = response.json()
                logger.info(f"📋 Export history: {len(exports)} total exports")
                
                for export in exports[:3]:  # Show first 3
                    status_icon = {"completed": "✅", "processing": "⏳", "failed": "❌"}.get(export['status'], "❓")
                    logger.info(f"   {status_icon} {export['id'][:8]}... - {export['status']} ({export['progress']:.1f}%)")
            
        except Exception as e:
            logger.error(f"❌ Export management demo failed: {e}")
    
    def demo_performance_monitoring(self):
        """Demonstrate real-time performance monitoring"""
        logger.info("\n⚡ === REAL-TIME PERFORMANCE MONITORING DEMO ===")
        
        try:
            # Get current performance metrics
            response = self.session.get(f"{self.api_base_url}/api/multimodal-studio/performance/metrics")
            if response.status_code == 200:
                metrics = response.json()
                system_metrics = metrics.get('system_metrics', {})
                
                logger.info("📊 Current System Metrics:")
                logger.info(f"   CPU Usage: {system_metrics.get('cpu_usage', 0):.1f}%")
                logger.info(f"   Memory Usage: {system_metrics.get('memory_usage', 0):.1f}%")
                logger.info(f"   Disk Usage: {system_metrics.get('disk_usage', 0):.1f}%")
                logger.info(f"   Network I/O: {system_metrics.get('network_io_sent', 0):.2f} MB/s")
                
                summary = metrics.get('summary', {})
                logger.info(f"   System Health: {summary.get('system_health', 'unknown').upper()}")
                logger.info(f"   Performance Trend: {summary.get('performance_trend', 'unknown').upper()}")
            
            # Detect performance bottlenecks
            response = self.session.get(f"{self.api_base_url}/api/multimodal-studio/performance/bottlenecks")
            if response.status_code == 200:
                bottlenecks = response.json()
                
                if bottlenecks.get('bottlenecks'):
                    logger.info(f"⚠️ Detected {len(bottlenecks['bottlenecks'])} performance bottlenecks:")
                    
                    for bottleneck in bottlenecks['bottlenecks']:
                        severity_icon = {
                            "critical": "🔴",
                            "high": "🟠", 
                            "medium": "🟡",
                            "low": "🟢"
                        }.get(bottleneck['severity'], "❓")
                        
                        logger.info(f"   {severity_icon} {bottleneck['title']}")
                        logger.info(f"      {bottleneck['description']}")
                        logger.info(f"      Affects: {', '.join(bottleneck['affected_components'])}")
                else:
                    logger.info("✅ No performance bottlenecks detected")
            
            # Get optimization suggestions
            response = self.session.get(f"{self.api_base_url}/api/multimodal-studio/performance/optimization-suggestions")
            if response.status_code == 200:
                suggestions_data = response.json()
                suggestions = suggestions_data.get('suggestions', [])
                
                logger.info(f"💡 Optimization Suggestions ({len(suggestions)} available):")
                
                for suggestion in suggestions[:3]:  # Show top 3
                    priority_icon = "🔥" if suggestion['priority'] <= 2 else "⭐" if suggestion['priority'] <= 5 else "💡"
                    logger.info(f"   {priority_icon} {suggestion['title']}")
                    logger.info(f"      Impact: {suggestion['impact']}")
                    logger.info(f"      Effort: {suggestion['implementation_effort']}")
                    logger.info(f"      Category: {suggestion['category']}")
            
            # Get performance history
            response = self.session.get(f"{self.api_base_url}/api/multimodal-studio/performance/history?hours=1")
            if response.status_code == 200:
                history = response.json()
                system_data_points = len(history.get('system_metrics', []))
                job_data_points = len(history.get('job_metrics', []))
                
                logger.info(f"📈 Performance History (last 1 hour):")
                logger.info(f"   System data points: {system_data_points}")
                logger.info(f"   Job data points: {job_data_points}")
            
        except Exception as e:
            logger.error(f"❌ Performance monitoring demo failed: {e}")
    
    def demo_user_preferences(self):
        """Demonstrate user preferences and workspace management"""
        logger.info("\n🎨 === USER PREFERENCES & WORKSPACE MANAGEMENT DEMO ===")
        
        try:
            # Get current user preferences
            response = self.session.get(f"{self.api_base_url}/api/multimodal-studio/preferences")
            if response.status_code == 200:
                preferences = response.json()
                logger.info("⚙️ Current User Preferences:")
                logger.info(f"   Theme: {preferences.get('theme', 'light')}")
                logger.info(f"   Auto-save interval: {preferences.get('auto_save_interval', 30)}s")
                logger.info(f"   Notifications: {'✅' if preferences.get('notifications_enabled') else '❌'}")
                logger.info(f"   Keyboard shortcuts: {'✅' if preferences.get('keyboard_shortcuts_enabled') else '❌'}")
                logger.info(f"   Default export format: {preferences.get('default_export_format', 'jsonl')}")
            
            # Update preferences
            updated_preferences = {
                "theme": "dark",
                "auto_save_interval": 15,
                "notifications_enabled": True,
                "keyboard_shortcuts_enabled": True,
                "default_export_format": "huggingface"
            }
            
            response = self.session.put(
                f"{self.api_base_url}/api/multimodal-studio/preferences",
                json=updated_preferences
            )
            
            if response.status_code == 200:
                logger.info("✅ Updated user preferences successfully")
            
            # Create workspace configurations
            workspaces = [
                {
                    "name": "Research Workspace",
                    "description": "Configuration for academic research projects",
                    "configuration": {
                        "quality_thresholds": {"minimum_score": 85},
                        "export_preferences": {"default_format": "huggingface"},
                        "job_settings": {"default_priority": 2}
                    },
                    "tags": ["research", "academic"]
                },
                {
                    "name": "Production Workspace", 
                    "description": "High-performance configuration for production datasets",
                    "configuration": {
                        "quality_thresholds": {"minimum_score": 95},
                        "export_preferences": {"default_format": "pytorch"},
                        "job_settings": {"default_priority": 1}
                    },
                    "tags": ["production", "high-quality"]
                }
            ]
            
            created_workspaces = []
            
            for workspace in workspaces:
                response = self.session.post(
                    f"{self.api_base_url}/api/multimodal-studio/workspaces",
                    json=workspace
                )
                
                if response.status_code == 200:
                    created_workspace = response.json()
                    created_workspaces.append(created_workspace)
                    logger.info(f"🏠 Created workspace: {workspace['name']}")
            
            # Get all workspaces
            response = self.session.get(f"{self.api_base_url}/api/multimodal-studio/workspaces")
            if response.status_code == 200:
                workspaces = response.json()
                logger.info(f"📚 Total workspaces: {len(workspaces)}")
                
                for workspace in workspaces:
                    default_icon = "⭐" if workspace.get('is_default') else "📁"
                    logger.info(f"   {default_icon} {workspace['name']} ({len(workspace.get('tags', []))} tags)")
            
            # Set default workspace
            if created_workspaces:
                response = self.session.post(
                    f"{self.api_base_url}/api/multimodal-studio/workspaces/{created_workspaces[0]['id']}/set-default"
                )
                
                if response.status_code == 200:
                    logger.info(f"⭐ Set '{created_workspaces[0]['name']}' as default workspace")
            
        except Exception as e:
            logger.error(f"❌ User preferences demo failed: {e}")
    
    async def demo_websocket_features(self):
        """Demonstrate real-time WebSocket features"""
        logger.info("\n🔄 === REAL-TIME WEBSOCKET FEATURES DEMO ===")
        
        try:
            # Mock user ID for demo
            user_id = "demo_user_123"
            
            # Connect to job monitoring WebSocket
            jobs_ws_url = f"{self.ws_base_url}/api/multimodal-studio/ws/jobs?user_id={user_id}"
            
            logger.info("🔌 Connecting to job monitoring WebSocket...")
            
            async with websockets.connect(jobs_ws_url) as websocket:
                logger.info("✅ Connected to job monitoring WebSocket")
                
                # Listen for a few messages
                for i in range(3):
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(message)
                        
                        message_type = data.get('type', 'unknown')
                        logger.info(f"📨 Received WebSocket message: {message_type}")
                        
                        if message_type == 'job_progress':
                            job_id = data.get('job_id', 'unknown')[:8]
                            progress = data.get('progress', 0)
                            logger.info(f"   Job {job_id}... progress: {progress:.1f}%")
                        
                        elif message_type == 'queue_update':
                            queue_data = data.get('data', {})
                            queued = queue_data.get('queued_count', 0)
                            active = queue_data.get('active_count', 0)
                            logger.info(f"   Queue status: {queued} queued, {active} active")
                        
                    except asyncio.TimeoutError:
                        logger.info("⏰ No WebSocket message received (timeout)")
                        break
            
            # Connect to performance monitoring WebSocket
            perf_ws_url = f"{self.ws_base_url}/api/multimodal-studio/ws/performance?user_id={user_id}"
            
            logger.info("🔌 Connecting to performance monitoring WebSocket...")
            
            async with websockets.connect(perf_ws_url) as websocket:
                logger.info("✅ Connected to performance monitoring WebSocket")
                
                # Listen for performance updates
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    if data.get('type') == 'performance_metrics':
                        metrics = data.get('data', {})
                        logger.info("📊 Real-time performance update received")
                        logger.info(f"   CPU: {metrics.get('cpu_usage', 0):.1f}%")
                        logger.info(f"   Memory: {metrics.get('memory_usage', 0):.1f}%")
                
                except asyncio.TimeoutError:
                    logger.info("⏰ No performance WebSocket message received (timeout)")
            
        except Exception as e:
            logger.error(f"❌ WebSocket demo failed: {e}")
    
    def demo_keyboard_shortcuts(self):
        """Demonstrate keyboard shortcuts and UX features"""
        logger.info("\n⌨️ === KEYBOARD SHORTCUTS & UX FEATURES DEMO ===")
        
        shortcuts = [
            ("?", "Show help dialog", "General"),
            ("Ctrl+N", "New job", "Jobs"),
            ("Ctrl+S", "Save", "General"),
            ("Ctrl+,", "Preferences", "General"),
            ("F11", "Toggle fullscreen", "View"),
            ("Ctrl+Tab", "Next tab", "Navigation"),
            ("Ctrl+Shift+Tab", "Previous tab", "Navigation"),
            ("Space", "Pause/Resume", "Jobs"),
            ("Delete", "Cancel job", "Jobs"),
            ("Ctrl+F", "Search", "General")
        ]
        
        logger.info("⌨️ Available Keyboard Shortcuts:")
        
        categories = {}
        for shortcut, action, category in shortcuts:
            if category not in categories:
                categories[category] = []
            categories[category].append((shortcut, action))
        
        for category, items in categories.items():
            logger.info(f"\n   📂 {category}:")
            for shortcut, action in items:
                logger.info(f"      {shortcut:15} - {action}")
        
        logger.info("\n🎨 UX Features:")
        logger.info("   📱 Responsive Design: Mobile, tablet, and desktop optimized")
        logger.info("   🖱️ Drag & Drop: Intuitive file uploads and job reordering")
        logger.info("   🌙 Theme Support: Light, dark, and auto themes")
        logger.info("   ♿ Accessibility: Screen reader support, high contrast mode")
        logger.info("   💾 Auto-save: Configurable auto-save intervals")
        logger.info("   🏠 Workspaces: Named workspace presets and configurations")
        logger.info("   ⚡ Performance: Virtual lists, debounced updates, optimizations")
    
    async def run_full_demo(self):
        """Run the complete Multimodal Studio production features demo"""
        logger.info("🚀 === MULTIMODAL STUDIO PRODUCTION FEATURES DEMO ===")
        logger.info("This demo showcases all R6-9 production-ready features")
        logger.info("=" * 60)
        
        # Authenticate
        if not self.authenticate():
            logger.error("❌ Authentication failed. Please check your credentials.")
            return
        
        # Run all demos
        self.demo_job_management()
        self.demo_quality_validation()
        self.demo_export_management()
        self.demo_performance_monitoring()
        self.demo_user_preferences()
        
        # WebSocket demos (async)
        await self.demo_websocket_features()
        
        # UX features demo
        self.demo_keyboard_shortcuts()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 === DEMO COMPLETED SUCCESSFULLY ===")
        logger.info("All R6-9 production features demonstrated!")
        logger.info("\nNext steps:")
        logger.info("1. 🌐 Open the React client at http://localhost:3000")
        logger.info("2. 🎛️ Navigate to the Multimodal Studio")
        logger.info("3. 🎮 Try the features interactively!")
        logger.info("4. 📚 Read the documentation for detailed usage")
        logger.info("\nThe future of character dataset creation is production-ready! 🚀")

async def main():
    """Main demo function"""
    print("🎬 Starting R6-9 Multimodal Studio Production Features Demo...")
    print("Make sure the Character Creation Devkit backend is running!")
    print("Use: ./start-devkit.sh")
    print()
    
    # Create demo instance
    demo = MultimodalStudioDemo()
    
    # Run the complete demo
    await demo.run_full_demo()

if __name__ == "__main__":
    asyncio.run(main()) 