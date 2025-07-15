#!/usr/bin/env python3
"""
Streamlit Web Dashboard for Spot Instance Orchestrator

A beautiful, interactive web interface for managing spot GPU instances,
monitoring costs, viewing metrics, and controlling training runs.

Usage:
    streamlit run scripts/spot_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import subprocess
import time
from typing import Dict, Any, Optional, List

# Import our orchestrator
from spotctl import (
    SpotOrchestrator, CostTracker, AlertManager, 
    MetricsCollector, CostOptimizer, InstanceChainer
)

st.set_page_config(
    page_title="Spot Training Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
    .cost-alert {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_instance_status() -> Dict[str, Any]:
    """Load current instance status"""
    try:
        orchestrator = SpotOrchestrator()
        return orchestrator.get_status()
    except Exception as e:
        return {'state': 'error', 'error': str(e)}

def load_cost_data() -> Dict[str, Any]:
    """Load cost tracking data"""
    cost_tracker = CostTracker(float(os.getenv('DAILY_BUDGET', '100.0')))
    return {
        'daily_spend': cost_tracker.get_daily_spend(),
        'daily_budget': cost_tracker.daily_budget,
        'under_budget': cost_tracker.is_under_budget()
    }

def load_optimization_data() -> Dict[str, Any]:
    """Load cost optimization suggestions"""
    optimizer = CostOptimizer()
    try:
        optimal_instance = optimizer.get_optimal_instance({
            'min_gpu_memory_gb': 16,
            'max_hourly_cost': 50.0
        })
        schedule = optimizer.suggest_schedule(8)  # 8 hours default
        return {
            'optimal_instance': optimal_instance,
            'schedule': schedule
        }
    except Exception as e:
        return {'error': str(e)}

def create_cost_chart(cost_data: Dict[str, Any]) -> go.Figure:
    """Create cost visualization chart"""
    fig = go.Figure()
    
    # Budget vs Spend
    fig.add_trace(go.Bar(
        name='Daily Budget',
        x=['Budget'],
        y=[cost_data['daily_budget']],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Current Spend',
        x=['Spend'],
        y=[cost_data['daily_spend']],
        marker_color='red' if cost_data['daily_spend'] > cost_data['daily_budget'] else 'green'
    ))
    
    fig.update_layout(
        title="Daily Budget vs Spend",
        xaxis_title="",
        yaxis_title="Cost ($)",
        barmode='group'
    )
    
    return fig

def create_instance_timeline() -> go.Figure:
    """Create instance uptime timeline"""
    # This would be enhanced with real data
    fig = go.Figure()
    
    # Mock timeline data
    now = datetime.now()
    timeline_data = [
        {'start': now - timedelta(hours=3), 'end': now, 'instance': 'p4d.24xlarge', 'cost': 45.0},
        {'start': now - timedelta(hours=8), 'end': now - timedelta(hours=3), 'instance': 'p3.2xlarge', 'cost': 15.0}
    ]
    
    for i, data in enumerate(timeline_data):
        fig.add_trace(go.Scatter(
            x=[data['start'], data['end']],
            y=[i, i],
            mode='lines+markers',
            name=f"{data['instance']} (${data['cost']})",
            line=dict(width=10),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Instance Timeline",
        xaxis_title="Time",
        yaxis_title="Instance",
        height=300
    )
    
    return fig

def main():
    """Main dashboard interface"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Spot Training Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Provider selection
        provider = st.selectbox(
            "Cloud Provider",
            ["aws", "runpod"],
            help="Select your preferred cloud provider"
        )
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
        
        # Settings
        st.header("⚙️ Settings")
        daily_budget = st.number_input(
            "Daily Budget ($)",
            min_value=10.0,
            max_value=1000.0,
            value=float(os.getenv('DAILY_BUDGET', '100.0')),
            step=10.0
        )
        
        # Alert configuration
        st.header("🔔 Alerts")
        slack_webhook = st.text_input(
            "Slack Webhook URL",
            value=os.getenv('SLACK_WEBHOOK_URL', ''),
            type="password",
            help="Webhook URL for Slack notifications"
        )
        
        discord_webhook = st.text_input(
            "Discord Webhook URL", 
            value=os.getenv('DISCORD_WEBHOOK_URL', ''),
            type="password",
            help="Webhook URL for Discord notifications"
        )
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "💰 Cost Analytics", "🎯 Optimization", 
        "🚀 Deploy", "📋 Logs"
    ])
    
    with tab1:
        st.header("Instance Overview")
        
        # Load current status
        status = load_instance_status()
        
        # Status cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if status.get('state') == 'running':
                st.success("🟢 Instance Running")
                st.metric("Instance ID", status.get('instance_id', 'N/A'))
            elif status.get('state') == 'no_instance':
                st.info("⚪ No Active Instance")
            else:
                st.error("🔴 Instance Error")
        
        with col2:
            if status.get('state') == 'running':
                st.metric(
                    "Uptime", 
                    f"{status.get('uptime_hours', 0):.1f} hours"
                )
                
        with col3:
            if status.get('state') == 'running':
                st.metric(
                    "Estimated Cost",
                    f"${status.get('estimated_cost', 0):.2f}"
                )
        
        with col4:
            if status.get('state') == 'running':
                st.metric(
                    "Spot Price",
                    f"${status.get('spot_price', 0):.2f}/hr"
                )
        
        # Instance details
        if status.get('state') == 'running':
            st.subheader("Instance Details")
            
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.info(f"**Provider:** {provider}")
                st.info(f"**Public IP:** {status.get('public_ip', 'N/A')}")
                
            with details_col2:
                st.info(f"**Instance Type:** {status.get('instance_type', 'N/A')}")
                st.info(f"**State:** {status.get('state', 'unknown')}")
        
        # Instance timeline
        st.subheader("Instance Timeline")
        timeline_fig = create_instance_timeline()
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    with tab2:
        st.header("Cost Analytics")
        
        # Load cost data
        cost_data = load_cost_data()
        
        # Cost overview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Daily Spend",
                f"${cost_data['daily_spend']:.2f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Daily Budget", 
                f"${cost_data['daily_budget']:.2f}"
            )
        
        with col3:
            remaining = cost_data['daily_budget'] - cost_data['daily_spend']
            st.metric(
                "Budget Remaining",
                f"${remaining:.2f}",
                delta=f"{(remaining/cost_data['daily_budget']*100):.1f}% remaining"
            )
        
        # Budget alert
        if not cost_data['under_budget']:
            st.error("⚠️ Daily budget exceeded! Consider terminating instances or increasing budget.")
        elif cost_data['daily_spend'] > cost_data['daily_budget'] * 0.8:
            st.warning("⚠️ Approaching daily budget limit (80% used)")
        
        # Cost visualization
        cost_fig = create_cost_chart(cost_data)
        st.plotly_chart(cost_fig, use_container_width=True)
        
        # Cost breakdown table
        st.subheader("Cost Breakdown")
        # This would be populated with real data
        cost_breakdown = pd.DataFrame({
            'Instance': ['p4d.24xlarge', 'p3.2xlarge'],
            'Runtime (hrs)': [2.5, 1.8],
            'Cost per Hour': [18.0, 3.0],
            'Total Cost': [45.0, 5.4],
            'Provider': ['AWS', 'AWS']
        })
        st.dataframe(cost_breakdown, use_container_width=True)
    
    with tab3:
        st.header("Cost Optimization")
        
        # Load optimization data
        opt_data = load_optimization_data()
        
        if 'error' not in opt_data:
            # Optimal instance recommendation
            st.subheader("💡 Recommended Instance")
            
            opt_col1, opt_col2, opt_col3 = st.columns(3)
            
            with opt_col1:
                st.info(f"**Instance Type:** {opt_data['optimal_instance']['type']}")
                
            with opt_col2:
                st.info(f"**Provider:** {opt_data['optimal_instance']['provider']}")
                
            with opt_col3:
                st.info(f"**Max Price:** ${opt_data['optimal_instance']['max_price']:.2f}/hr")
            
            # Scheduling recommendations
            st.subheader("📅 Scheduling Optimization")
            schedule = opt_data['schedule']
            
            st.success(f"**Strategy:** {schedule['strategy']}")
            st.success(f"**Estimated Savings:** {schedule['estimated_savings']}")
            
            if schedule['strategy'] == 'single_session':
                st.info(f"**Next Optimal Start:** {schedule['start_time'].strftime('%Y-%m-%d %H:%M UTC')}")
            else:
                st.info(f"**Sessions Needed:** {schedule['sessions']}")
                st.info(f"**First Session:** {schedule['start_time'].strftime('%Y-%m-%d %H:%M UTC')}")
            
            # Optimization tips
            st.subheader("💡 Optimization Tips")
            st.markdown("""
            - **Off-peak hours** (10 PM - 6 AM UTC) offer 30-50% savings
            - **Instance chaining** tries cheaper options first
            - **Spot interruption** is handled automatically with checkpoints
            - **Budget alerts** prevent overspending
            """)
        else:
            st.error(f"Optimization error: {opt_data['error']}")
    
    with tab4:
        st.header("Deploy New Instance")
        
        with st.form("deploy_form"):
            st.subheader("Deployment Configuration")
            
            deploy_col1, deploy_col2 = st.columns(2)
            
            with deploy_col1:
                instance_type = st.selectbox(
                    "Instance Type",
                    ["p4d.24xlarge", "p3.8xlarge", "p3.2xlarge", "RTX_A5000", "RTX_A4000"],
                    help="Select GPU instance type"
                )
                
                max_price = st.number_input(
                    "Max Price ($/hour)",
                    min_value=0.1,
                    max_value=50.0,
                    value=10.0,
                    step=0.5
                )
                
            with deploy_col2:
                use_optimizer = st.checkbox(
                    "Use Cost Optimizer",
                    help="Automatically select optimal instance"
                )
                
                use_chaining = st.checkbox(
                    "Use Instance Chaining", 
                    help="Try multiple instance types for best price"
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                resume_training = st.checkbox("Resume from Checkpoint")
                checkpoint_path = st.text_input(
                    "Checkpoint Path (S3)",
                    placeholder="s3://bucket/checkpoint-500/shards/checkpoint.json",
                    disabled=not resume_training
                )
            
            # Deploy button
            submitted = st.form_submit_button("🚀 Deploy Instance", type="primary")
            
            if submitted:
                with st.spinner("Deploying instance..."):
                    try:
                        orchestrator = SpotOrchestrator(provider=provider)
                        result = orchestrator.deploy(
                            instance_type=instance_type,
                            max_price=max_price,
                            resume_from=checkpoint_path if resume_training else None,
                            use_optimizer=use_optimizer,
                            use_chaining=use_chaining
                        )
                        
                        st.success(f"✅ Instance deployed successfully!")
                        st.json(result)
                        
                        # Auto-refresh in 5 seconds
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Deployment failed: {str(e)}")
        
        # Quick actions
        st.subheader("Quick Actions")
        
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("📊 Refresh Status", type="secondary"):
                st.rerun()
        
        with action_col2:
            if st.button("🛑 Terminate Instance", type="secondary"):
                if status.get('state') == 'running':
                    with st.spinner("Terminating instance..."):
                        orchestrator = SpotOrchestrator(provider=provider)
                        success = orchestrator.terminate()
                        if success:
                            st.success("✅ Instance terminated successfully!")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("❌ Failed to terminate instance")
                else:
                    st.info("No running instance to terminate")
        
        with action_col3:
            if st.button("🔔 Test Alerts", type="secondary"):
                alert_manager = AlertManager()
                alert_manager.send_alert(
                    'info',
                    'Test alert from Spot Dashboard',
                    {'source': 'dashboard', 'timestamp': datetime.now(timezone.utc).isoformat()}
                )
                st.success("✅ Test alert sent!")
    
    with tab5:
        st.header("Training Logs")
        
        if status.get('state') == 'running':
            st.subheader("📋 Live Logs")
            
            # Log streaming would be implemented here
            st.info("🔄 Live log streaming coming soon...")
            
            # For now, show basic log information
            st.code("""
[2024-01-15 10:30:15] Training started...
[2024-01-15 10:30:16] Loading model: SmolLM2-360M-Instruct
[2024-01-15 10:30:17] Dataset loaded: 1000 samples
[2024-01-15 10:30:18] Training step 1/10000
[2024-01-15 10:30:19] Loss: 2.45
[2024-01-15 10:30:20] Training step 2/10000
...
            """, language="text")
            
            # Log controls
            log_col1, log_col2 = st.columns(2)
            
            with log_col1:
                if st.button("📥 Download Logs"):
                    st.info("Log download functionality coming soon...")
            
            with log_col2:
                lines = st.select_slider(
                    "Lines to show",
                    options=[50, 100, 200, 500, 1000],
                    value=100
                )
        else:
            st.info("No running instance to show logs for.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🚀 **Spot Training Dashboard** - Built with ❤️ for cost-effective AI training"
    )

if __name__ == "__main__":
    main() 