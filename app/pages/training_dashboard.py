"""
📊 Training Dashboard Page

Real-time training dashboard with advanced monitoring features including:
- Training controls (pause, resume, stop)
- Real-time metrics and health monitoring  
- Interactive loss curves and progress charts
- Character consistency tracking
- TensorBoard and Wandb integration
- Training health warnings and recommendations

Extracted from main app.py for better maintainability.
"""

import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
from utils.async_training import async_training_service

def render_consistency_deep_dive(metrics: Dict[str, Any]):
    """Renders the expandable deep-dive for character consistency."""
    
    consistency = metrics.get('character_consistency', 0)
    
    with st.expander(f"Character Consistency: {consistency:.2f}", expanded=False):
        st.markdown("##### 🔬 Consistency Score Deep Dive")
        st.markdown(
            """
            This shows a detailed breakdown of the consistency score based on a random sample of validation data. 
            It helps diagnose *why* the score is what it is.
            """
        )

        # ✨ NEW: Personality Drift Analysis (R1-9)
        if metrics.get('personality_drift_available', False) or 'personality_drift_result' in metrics:
            st.markdown("---")
            st.markdown("#### 🎭 Personality Drift Analysis")
            
            if 'personality_drift_result' in metrics:
                drift_result = metrics['personality_drift_result']
                
                # Import here to avoid circular imports
                try:
                    from ..components.personality_drift_analyzer import render_personality_drift_chart
                    
                    # Create beautiful radar chart
                    fig = render_personality_drift_chart(drift_result, "Personality Consistency Check")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show drift insights
                    drift_col1, drift_col2, drift_col3 = st.columns(3)
                    
                    with drift_col1:
                        drift_magnitude = drift_result.drift_magnitude
                        drift_color = "#10b981" if drift_magnitude < 0.1 else "#f59e0b" if drift_magnitude < 0.3 else "#ef4444"
                        st.markdown(f"""
                            <div style="text-align: center; padding: 0.8rem; background: rgba(255,255,255,0.05); border-radius: 6px; border-left: 3px solid {drift_color};">
                                <h5 style="margin: 0; color: {drift_color};">Drift Magnitude</h5>
                                <h3 style="margin: 0.3rem 0; color: {drift_color};">{drift_magnitude:.3f}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with drift_col2:
                        confidence = drift_result.confidence_score
                        conf_color = "#10b981" if confidence > 0.8 else "#f59e0b" if confidence > 0.6 else "#ef4444"
                        st.markdown(f"""
                            <div style="text-align: center; padding: 0.8rem; background: rgba(255,255,255,0.05); border-radius: 6px; border-left: 3px solid {conf_color};">
                                <h5 style="margin: 0; color: {conf_color};">Confidence</h5>
                                <h3 style="margin: 0.3rem 0; color: {conf_color};">{confidence:.0%}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with drift_col3:
                        samples = drift_result.samples_analyzed
                        sample_color = "#10b981" if samples >= 40 else "#f59e0b" if samples >= 20 else "#ef4444"
                        st.markdown(f"""
                            <div style="text-align: center; padding: 0.8rem; background: rgba(255,255,255,0.05); border-radius: 6px; border-left: 3px solid {sample_color};">
                                <h5 style="margin: 0; color: {sample_color};">Samples</h5>
                                <h3 style="margin: 0.3rem 0; color: {sample_color};">{samples}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Quick interpretation
                    if drift_magnitude < 0.1:
                        st.success("🎉 **Excellent personality consistency!** The model is staying true to the authored personality.")
                    elif drift_magnitude < 0.2:
                        st.info("✅ **Good personality consistency.** Minor variations are normal and expected.")
                    else:
                        st.warning("⚠️ **Personality drift detected.** Consider reviewing training data for personality consistency.")
                        
                except ImportError as e:
                    st.error(f"Could not load personality drift analyzer: {e}")
            else:
                st.info("💡 **Personality drift analysis will be available once training progresses.**")

        evaluated_samples = metrics.get('evaluated_samples', [])
        if not evaluated_samples:
            st.info("No detailed evaluation samples available for this step yet.")
            return

        # Helper to display score with color/icon
        def display_score(name, value, explanation):
            # For meta commentary, lower is better (it's a penalty score).
            if name == "Meta Commentary":
                icon = "✅" if value < 0.1 else "❌"
            else:
                icon = "✅" if value >= 0.7 else "⚠️" if value >= 0.4 else "❌"
            
            # Format value to 2 decimal places
            score_str = f"{value:.2f}"
            
            st.markdown(f"{icon} **{name}:** `{score_str}`", help=explanation)

        for i, sample_eval in enumerate(evaluated_samples):
            st.markdown(f"---")
            st.markdown(f"**Sample Evaluation {i+1}** (from validation set)")
            
            # Use columns for a cleaner layout
            col_convo, col_scores = st.columns([2, 1])

            with col_convo:
                st.markdown(f"**👤 User:**")
                st.info(sample_eval['user'])
                st.markdown(f"**🎭 Assistant's Response:**")
                st.info(sample_eval['assistant'])
            
            with col_scores:
                scores = sample_eval.get('scores', {})
                if scores:
                    st.markdown("**Score Breakdown:**")
                    
                    display_score("Overall", scores.get('overall_consistency', 0), "The weighted average of all consistency metrics.")
                    display_score("Name Consistency", scores.get('name_consistency', 0), "Checks for incorrect third-person self-references (e.g., 'CharacterName did...'). Should be 1.0.")
                    display_score("Personality", scores.get('personality_alignment', 0), "Aligns response with character's defined personality traits.")
                    display_score("Speech Pattern", scores.get('speech_pattern', 0), "Compares speech patterns (e.g., use of '...') to character examples.")
                    display_score("Response Quality", scores.get('response_quality', 0), "Evaluates response length and relevance to the user's prompt.")
                    display_score("Voice", scores.get('voice_consistency', 0), "Checks for general tone, use of actions (*...*), and emotional expression.")
                    display_score("Meta Commentary", scores.get('meta_commentary', 0), "PENALTY for breaking character by mentioning being an AI. Lower is better (0.0 is best).")
                else:
                    st.warning("No score breakdown available for this sample.")

def render_healthy_run_example():
    """Renders a simple text example of a healthy training run loss curve."""
    st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(30, 41, 59, 0.5); border-radius: 12px; margin: 1rem 0;">
            <h4 style="color: #f1f5f9; margin-bottom: 1rem;">💡 Healthy Loss Curve Example</h4>
        </div>
    """, unsafe_allow_html=True)
    st.code("""
Loss
│
│ \\
│  \\
│   \\___
│       \\____
└─────────────────► Steps
A healthy run shows loss decreasing and stabilizing.
    """)

def page_training_dashboard():
    """Async training dashboard with real-time monitoring"""
    st.markdown('<h2 class="gradient-text">📊 Training Dashboard</h2>', unsafe_allow_html=True)
    
    # Get current training run ID from session state
    current_training_run_id = st.session_state.get('current_training_run_id')
    user_id = st.session_state.get('current_user_id', 1)
    
    # Show recent training runs
    with st.expander("📋 Recent Training Runs", expanded=bool(not current_training_run_id)):
        recent_runs = async_training_service.get_user_training_runs(user_id, limit=5)
        
        if recent_runs:
            for run in recent_runs:
                with st.container():
                    col_info, col_status, col_action = st.columns([3, 1, 1])
                    
                    with col_info:
                        st.write(f"**{run['character_name']}** ({run['training_method'].upper()})")
                        st.caption(f"Created: {run['created_at'][:19].replace('T', ' ')}")
                    
                    with col_status:
                        status = run['status']
                        status_colors = {
                            'queued': '🟡',
                            'processing': '🔵', 
                            'completed': '🟢',
                            'failed': '🔴',
                            'cancelled': '⚫'
                        }
                        st.write(f"{status_colors.get(status, '⚪')} {status.title()}")
                    
                    with col_action:
                        if st.button("📊 Monitor", key=f"monitor_{run['training_run_id']}"):
                            st.session_state.current_training_run_id = run['training_run_id']
                            st.rerun()
                    
                    st.divider()
        else:
            st.info("No training runs found. Start a training job first!")
    
    # If no active training run, prompt user to start one
    if not current_training_run_id:
        st.info("ℹ️ Select a training run above to monitor, or configure and start a new training job.")
        return
    
    # Get current training status
    training_status = async_training_service.get_training_status(current_training_run_id)
    
    if training_status['status'] == 'not_found':
        st.error("❌ Training run not found. Please select a different run.")
        if st.button("🔄 Refresh Training Runs"):
            st.session_state.current_training_run_id = None
            st.rerun()
        return
    
    # Display current training run info
    st.markdown(f"### 🎯 Monitoring Training Run #{current_training_run_id}")
    
    # Check if this is contamination MoE training
    config = training_status.get('config', {})
    is_contamination_moe = config.get('use_contamination_moe', False)
    
    if is_contamination_moe:
        st.success("🔥 **CONTAMINATION WARFARE TRAINING ACTIVE!**")
        st.info("Expert specialization eliminating constitutional AI contamination")
    
    # Training run details
    col_char, col_model, col_status = st.columns(3)
    
    with col_char:
        st.metric("Character", training_status['character_name'])
    
    with col_model:
        model_display = training_status['base_model'].split('/')[-1]
        if is_contamination_moe:
            model_display += " (Contamination MoE)"
        st.metric("Base Model", model_display)
    
    with col_status:
        status = training_status['status']
        status_colors = {
            'queued': ('🟡', 'warning'),
            'processing': ('🔵', 'info'), 
            'completed': ('🟢', 'success'),
            'failed': ('🔴', 'error'),
            'cancelled': ('⚫', 'info')
        }
        icon, color = status_colors.get(status, ('⚪', 'info'))
        st.metric("Status", f"{icon} {status.title()}")
    
    # Training controls (limited for async)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🛑 Cancel Training", 
                    disabled=status not in ['queued', 'processing'],
                    use_container_width=True):
            if async_training_service.cancel_training(current_training_run_id, user_id):
                st.success("✅ Training cancelled successfully")
                st.rerun()
            else:
                st.error("❌ Failed to cancel training")
    
    with col3:
        if st.button("🧪 Test Model", 
                    disabled=status not in ['completed'],
                    use_container_width=True):
            st.info("🚧 Model testing coming soon!")
    
    with col4:
        if st.button("📂 Open Results", 
                    disabled=not training_status.get('sft_adapter_path'),
                    use_container_width=True):
            if training_status.get('sft_adapter_path'):
                st.success(f"📁 SFT Adapter: {training_status['sft_adapter_path']}")
                if training_status.get('rlhf_adapter_path'):
                    st.success(f"🧠 RLHF Adapter: {training_status['rlhf_adapter_path']}")
    
    # Monitoring Dashboards Section
    advanced_config = st.session_state.get('advanced_training_config', {})
    wandb_enabled = advanced_config.get('enable_wandb', False)
    tensorboard_enabled = advanced_config.get('enable_tensorboard', False)

    if st.session_state.training_status != 'idle' and (wandb_enabled or tensorboard_enabled):
        with st.expander("📊 Monitoring Dashboards", expanded=True):
            mon_col1, mon_col2 = st.columns(2)

            with mon_col1:
                if wandb_enabled:
                    st.markdown("##### 🌐 Weights & Biases")
                    wandb_url = st.session_state.training_manager.get_wandb_url()
                    if wandb_url:
                        st.markdown(f'**[Open Wandb Run Page ↗]({wandb_url})**')
                        st.session_state.wandb_url_displayed = True
                    elif st.session_state.get('wandb_url_displayed'):
                         st.markdown('**[Wandb Run Page ↗](about:blank)** (Link was previously active)')
                    else:
                        st.info("Wandb URL will appear here once the run starts.")

            with mon_col2:
                if tensorboard_enabled:
                    st.markdown("##### 📈 TensorBoard")
                    if st.button("Launch TensorBoard", key="launch_tb"):
                        st.session_state.launch_tensorboard_request = True
                    
                    if st.session_state.get("tensorboard_launched"):
                        st.markdown("**[Open TensorBoard Dashboard ↗](http://localhost:6006)**")
                        st.caption("TensorBoard is running in the background.")

    # Show active training configuration
    with st.expander("🔍 **Active Training Configuration**", expanded=False):
        active_config = st.session_state.get('active_training_config', {})
        if active_config:
            config_display_col1, config_display_col2 = st.columns(2)
            
            with config_display_col1:
                st.write(f"**Log Every N Steps:** `{active_config.get('logging_steps', 10)}`")
                st.write(f"**Learning Rate:** `{active_config.get('learning_rate', 2e-4)}`")
                st.write(f"**Method:** `{active_config.get('finetune_method', 'lora').upper()}`")
                st.write(f"**LoRA Rank:** `{active_config.get('lora_r', 16)}`")
                
            with config_display_col2:
                st.write(f"**Batch Size:** `{active_config.get('batch_size', 2)}`")
                st.write(f"**Save Steps:** `{active_config.get('save_steps', 50)}`")
                st.write(f"**Max Samples:** `{active_config.get('max_samples', 'All')}`")
                st.write(f"**FP16:** `{active_config.get('fp16', False)}`")
        else:
            st.info("No active training configuration found.")

    # Get training metrics from database
    training_metrics = training_status.get('metrics', {})
    
    # Auto-refresh for active training
    if status in ['queued', 'processing']:
        st.markdown("🔄 **Auto-refreshing every 10 seconds...**")
        time.sleep(10)
        st.rerun()
    
    # Enhanced real-time metrics
    metrics_placeholder = st.empty()
    health_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Display training information based on status
    if status == 'queued':
        st.info("⏳ Training job is queued and waiting for a worker...")
        st.markdown(f"**Dataset Size:** {training_status.get('dataset_size', 'Unknown')} samples")
        st.markdown(f"**Training Method:** {training_status['training_method'].upper()}")
        return
    elif status == 'failed':
        st.error("❌ Training failed!")
        if 'error' in training_metrics:
            st.error(f"**Error:** {training_metrics['error']}")
        return
    elif status == 'cancelled':
        st.warning("⚫ Training was cancelled")
        return
    
    # Show training metrics for active/completed training
    if training_metrics:
        # Display training health alerts
        with health_placeholder.container():
            health_status = training_metrics.get('training_health_status', 'unknown')
            health_warnings = training_metrics.get('health_warnings', [])
            
            if health_status == 'critical':
                st.error("🚨 **Critical Training Issues Detected:**")
                for warning in health_warnings:
                    st.error(f"• {warning}")
            elif health_status == 'warning':
                st.warning("⚠️ **Training Warnings:**")
                for warning in health_warnings:
                    st.warning(f"• {warning}")
            elif health_status == 'healthy':
                st.success("✅ Training is healthy")
        
        with metrics_placeholder.container():
            # Primary metrics row
            if is_contamination_moe:
                col1, col2, col3, col4, col5 = st.columns(5)
            else:
                col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_loss = training_metrics.get('current_loss', 0)
                loss_delta = training_metrics.get('loss_delta', 0)
                delta_color = "normal" if abs(loss_delta) < 0.01 else ("inverse" if loss_delta < 0 else "off")
                
                st.metric(
                    "Training Loss",
                    f"{current_loss:.4f}",
                    delta=f"{loss_delta:.4f}" if loss_delta != 0 else None,
                    delta_color=delta_color
                )
            
            with col2:
                current_step = training_metrics.get('current_step', training_status.get('total_steps', 0))
                total_steps = training_status.get('total_steps', 1)
                progress_pct = (current_step/total_steps)*100 if total_steps > 0 else 0
                
                st.metric(
                    "Progress",
                    f"{current_step}/{total_steps}",
                    delta=f"{progress_pct:.1f}%"
                )
            
            with col3:
                lr = training_metrics.get('learning_rate', 0)
                st.metric(
                    "Learning Rate",
                    f"{lr:.2e}" if lr > 0 else "N/A"
                )
            
            with col4:
                # Calculate elapsed time from timestamps
                if training_status.get('started_at') and status == 'processing':
                    from datetime import datetime
                    import dateutil.parser
                    started = dateutil.parser.parse(training_status['started_at'])
                    now = datetime.now(started.tzinfo)
                    elapsed = int((now - started).total_seconds())
                elif training_status.get('completed_at') and training_status.get('started_at'):
                    started = dateutil.parser.parse(training_status['started_at'])
                    completed = dateutil.parser.parse(training_status['completed_at'])
                    elapsed = int((completed - started).total_seconds())
                else:
                    elapsed = int(training_metrics.get('elapsed_time', 0))
                
                st.metric(
                    "Elapsed Time",
                    f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"
                )
            
            # Contamination warfare specific metrics
            if is_contamination_moe:
                with col5:
                    contamination_blocks = training_metrics.get('contamination_blocks_total', 0)
                    purity_rate = training_metrics.get('character_purity_rate', 0)
                    st.metric(
                        "🛡️ Contamination Blocked",
                        f"{contamination_blocks}",
                        delta=f"{purity_rate:.1f}% purity" if purity_rate > 0 else None,
                        help="Total contamination instances isolated by expert routing"
                    )
            
            # Secondary metrics row (if validation is enabled)
            if 'eval_loss' in training_metrics or 'character_consistency' in training_metrics or 'avg_personality_alignment' in training_metrics or 'avg_lore_adherence' in training_metrics:
                st.markdown("---")
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    if 'eval_loss' in training_metrics:
                        eval_loss = training_metrics['eval_loss']
                        st.metric(
                            "Validation Loss",
                            f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else str(eval_loss),
                            help="Loss on validation set"
                        )
                
                with col6:
                    if 'character_consistency' in training_metrics:
                        # Use the new deep-dive renderer
                        render_consistency_deep_dive(training_metrics)
                    else:
                        st.metric(
                            "Character Consistency",
                            "N/A",
                            help="Calculated during evaluation."
                        )
                
                with col7:
                    if 'avg_consistency' in training_metrics:
                        avg_consistency = training_metrics['avg_consistency']
                        st.metric(
                            "Avg Consistency",
                            f"{avg_consistency:.2f}" if isinstance(avg_consistency, (int, float)) else str(avg_consistency),
                            help="Overall character consistency score"
                        )
                
                with col8:
                    training_health = training_metrics.get('training_health_status', 'unknown').title()
                    health_color = {"Healthy": "normal", "Warning": "inverse", "Critical": "off"}.get(training_health, "normal")
                    st.metric(
                        "Training Health",
                        training_health,
                        delta_color=health_color
                    )
            
            # 🔥 CONTAMINATION WARFARE METRICS (R3-5)
            if is_contamination_moe and training_metrics.get('contamination_report'):
                st.markdown("---")
                st.markdown("### 🔥 Contamination Warfare Results")
                
                contamination_report = training_metrics['contamination_report']
                
                warfare_col1, warfare_col2, warfare_col3, warfare_col4 = st.columns(4)
                
                with warfare_col1:
                    purity_rate = contamination_report.get('character_purity_rate', 0)
                    st.metric(
                        "🎭 Character Purity",
                        f"{purity_rate:.1f}%",
                        help="Percentage of character responses free from contamination"
                    )
                
                with warfare_col2:
                    block_rate = contamination_report.get('contamination_block_rate', 0)
                    st.metric(
                        "🛡️ Contamination Blocked",
                        f"{block_rate:.1f}%",
                        help="Percentage of contaminated inputs successfully isolated"
                    )
                
                with warfare_col3:
                    success_rate = contamination_report.get('overall_success_rate', 0)
                    st.metric(
                        "⚔️ Warfare Success",
                        f"{success_rate:.1f}%",
                        help="Overall contamination warfare effectiveness"
                    )
                
                with warfare_col4:
                    expert_accuracy = contamination_report.get('expert_routing_accuracy', 0)
                    st.metric(
                        "🎯 Expert Routing",
                        f"{expert_accuracy:.1f}%",
                        help="Accuracy of expert specialization routing"
                    )
                
                # Contamination warfare status
                if purity_rate >= 95:
                    st.success("🏆 **CONTAMINATION WARFARE VICTORY!** Character expert successfully isolated from constitutional AI contamination!")
                elif purity_rate >= 80:
                    st.info("🔥 **STRONG CONTAMINATION DEFENSE!** Expert specialization working effectively!")
                else:
                    st.warning("⚠️ **CONTAMINATION RESISTANCE NEEDS WORK** Consider adjusting routing threshold or training longer")
            
            # ✨ NEW: Advanced Metrics Row (R1-9) 
            if 'avg_personality_alignment' in training_metrics or 'avg_lore_adherence' in training_metrics:
                st.markdown("---")
                st.markdown("### 🎭 Advanced Character Metrics")
                
                adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
                
                with adv_col1:
                    if 'avg_personality_alignment' in training_metrics:
                        personality_score = training_metrics['avg_personality_alignment']
                        personality_delta = training_metrics.get('personality_alignment_delta', 0)
                        delta_color = "normal" if abs(personality_delta) < 0.05 else ("normal" if personality_delta > 0 else "inverse")
                        
                        st.metric(
                            "🧠 Personality Alignment",
                            f"{personality_score:.3f}" if isinstance(personality_score, (int, float)) else str(personality_score),
                            delta=f"{personality_delta:+.3f}" if personality_delta != 0 else None,
                            delta_color=delta_color,
                            help="How well responses match the authored personality profile (Big Five traits)"
                        )
                
                with adv_col2:
                    if 'avg_lore_adherence' in training_metrics:
                        lore_score = training_metrics['avg_lore_adherence']
                        lore_delta = training_metrics.get('lore_adherence_delta', 0)
                        delta_color = "normal" if abs(lore_delta) < 0.05 else ("normal" if lore_delta > 0 else "inverse")
                        
                        st.metric(
                            "📜 Lore Adherence",
                            f"{lore_score:.3f}" if isinstance(lore_score, (int, float)) else str(lore_score),
                            delta=f"{lore_delta:+.3f}" if lore_delta != 0 else None,
                            delta_color=delta_color,
                            help="How well responses respect and incorporate world lore facts"
                        )
                
                with adv_col3:
                    # Combined quality score
                    if 'avg_personality_alignment' in training_metrics and 'avg_lore_adherence' in training_metrics:
                        combined_score = (training_metrics['avg_personality_alignment'] + training_metrics['avg_lore_adherence']) / 2
                        st.metric(
                            "🎯 Overall Quality",
                            f"{combined_score:.3f}",
                            help="Combined personality alignment and lore adherence score"
                        )
                    elif 'character_consistency' in training_metrics:
                        consistency = training_metrics['character_consistency']
                        st.metric(
                            "🎭 Character Quality",
                            f"{consistency:.3f}" if isinstance(consistency, (int, float)) else str(consistency),
                            help="Overall character performance metric"
                        )
                
                with adv_col4:
                    # Performance indicator
                    if 'avg_personality_alignment' in training_metrics:
                        personality_score = training_metrics['avg_personality_alignment']
                        if personality_score > 0.8:
                            performance = "Excellent"
                            performance_color = "#10b981"
                        elif personality_score > 0.6:
                            performance = "Good"
                            performance_color = "#f59e0b"
                        else:
                            performance = "Needs Work"
                            performance_color = "#ef4444"
                        
                        st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 4px solid {performance_color};">
                                <h4 style="margin: 0; color: {performance_color};">Performance</h4>
                                <h3 style="margin: 0.5rem 0; color: {performance_color};">{performance}</h3>
                                <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1;">Character Training</p>
                            </div>
                        """, unsafe_allow_html=True)
        
        # Enhanced loss curve with multiple metrics
        if 'loss_history' in training_metrics and training_metrics['loss_history']:
            with chart_placeholder.container():
                # Use columns to place an info icon next to the title
                col_title, col_info = st.columns([0.95, 0.05])
                with col_title:
                    st.markdown("### Training Progress")
                with col_info:
                    with st.popover("ℹ️", help="Explain this chart"):
                        st.markdown("""
                        **What am I looking at?**
                        This chart shows how well the model is learning over time.

                        - **🔵 Training Loss (Blue Line):** This shows the error on the data the model is currently training on. It should always go down.
                        - **🟠 Validation Loss (Orange Line):** This shows the error on a separate set of data the model hasn't seen. It's a key indicator of how well the model will perform on new, unseen conversations.

                        **What's a good sign? ✅**
                        Both lines go down and then flatten out. This means the model is learning and generalizing well.

                        **What's a bad sign? 🚨**
                        The blue line keeps going down, but the orange line starts to go **up**. This is called **overfitting**. The model has memorized the training data instead of learning the character's personality. 
                        
                        **If you see overfitting, it's a good time to stop training.**
                        """)
                
                # Create enhanced visualization
                steps = list(range(len(training_metrics['loss_history'])))
                
                # Build chart data
                chart_data = pd.DataFrame({
                    'Step': steps,
                    'Training Loss': training_metrics['loss_history']
                })
                
                # Create figure with secondary y-axis for character consistency
                fig = px.line(
                    chart_data, x='Step', y='Training Loss',
                    title="Training Progress Over Time",
                    template="plotly_dark"
                )
                
                # Add validation loss if available
                if 'eval_loss' in training_metrics:
                    # Try to get eval history from metrics first, then from training manager
                    eval_history = training_metrics.get('eval_loss_history', [])
                    if not eval_history and hasattr(st.session_state.training_manager, 'eval_loss_history'):
                        eval_history = getattr(st.session_state.training_manager, 'eval_loss_history', [])
                    if eval_history and len(eval_history) > 0:
                        # ✅ IMPROVED: Better step alignment for validation loss
                        # Validation happens less frequently, so we need to space out the points
                        active_config = st.session_state.get('active_training_config', {})
                        eval_freq = active_config.get('eval_steps', 25)
                        eval_steps = [i * eval_freq for i in range(len(eval_history))]
                        
                        fig.add_scatter(
                            x=eval_steps, y=eval_history,
                            mode='lines+markers',
                            name='Validation Loss',
                            line=dict(color='orange', width=3),
                            marker=dict(size=8, color='orange')
                        )
                
                # Style the chart
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    xaxis_title="Training Steps",
                    yaxis_title="Loss",
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                # Add training health indicators
                if 'health_warnings' in training_metrics and training_metrics['health_warnings']:
                    warning_step = training_metrics.get('current_step', len(steps))
                    fig.add_vline(
                        x=warning_step,
                        line_dash="dot",
                        line_color="red",
                        annotation_text="Warning",
                        annotation_position="top right"
                    )
                
                chart_col, example_col = st.columns([3, 1])
                with chart_col:
                    st.plotly_chart(fig, use_container_width=True)
                with example_col:
                    render_healthy_run_example()
                
                # Character consistency chart (if available)
                if 'character_consistency' in training_metrics:
                    st.markdown("### Character Consistency")
                    
                    # Create a simple consistency indicator
                    consistency_score = training_metrics['character_consistency']
                    
                    col_chart1, col_chart2 = st.columns([1, 2])
                    
                    with col_chart1:
                        # Gauge-style visualization
                        consistency_score = float(consistency_score) if consistency_score is not None else 0.0
                        gauge_color = "#10b981" if consistency_score > 0.7 else "#f59e0b" if consistency_score > 0.4 else "#ef4444"
                        st.metric(
                            "Current Consistency Score",
                            f"{consistency_score:.2f}",
                            help="1.0 = Perfect character consistency, 0.0 = Poor consistency"
                        )
                        
                        # Progress bar using Streamlit's native progress bar
                        st.progress(consistency_score, text=f"Consistency: {consistency_score:.2f}")
                    
                    with col_chart2:
                        # Show consistency evaluation details if available
                        last_eval_step = training_metrics.get('consistency_last_eval_step', 0)
                        if last_eval_step > 0:
                            st.info(f"Last consistency evaluation at step {last_eval_step}")
                        
                        # Recommendations based on consistency score
                        if consistency_score < 0.3:
                            st.warning("💡 **Low consistency detected:** Consider reviewing dataset quality or adjusting training parameters")
                        elif consistency_score > 0.8:
                            st.success("🎉 **Excellent consistency:** Character is learning well!")
                        else:
                            st.info("📈 **Moderate consistency:** Training is progressing normally")
    
    # Auto-refresh for real-time updates and status change detection
    if st.session_state.training_status in ['training', 'dataset_generation']:
        time.sleep(2)
        st.rerun()
    elif not status_changed and st.session_state.training_status in ['training', 'paused']:
        # Extra safety check - ensure we catch status changes even if metrics processing is delayed
        time.sleep(1)
        st.rerun()

# Run the page function if this file is executed directly (for testing)
if __name__ == "__main__":
    page_training_dashboard() 