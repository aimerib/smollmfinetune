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
    """Real-time training dashboard"""
    st.markdown('<h2 class="gradient-text">📊 Training Dashboard</h2>', unsafe_allow_html=True)
    
    if st.session_state.training_status == 'idle':
        st.info("ℹ️ No training in progress. Configure and start training first.")
        return
    
    # Training controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("⏸️ Pause Training", disabled=st.session_state.training_status != 'training'):
            st.session_state.training_manager.pause_training()
            st.session_state.training_status = 'paused'
            st.rerun()
    
    with col2:
        if st.button("▶️ Resume Training", disabled=st.session_state.training_status != 'paused'):
            st.session_state.training_manager.resume_training()
            st.session_state.training_status = 'training'
            st.rerun()
    
    with col3:
        if st.button("🧪 Test Current Model", disabled=st.session_state.training_status == 'idle'):
            # Implement quick testing
            st.info("Testing current checkpoint...")
    
    with col4:
        if st.button("🛑 Stop Training", disabled=st.session_state.training_status not in ['training', 'paused']):
            st.session_state.training_manager.stop_training()
            st.session_state.training_status = 'complete'
            st.rerun()
    
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

    # Enhanced real-time metrics
    metrics_placeholder = st.empty()
    health_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Always get metrics first (this processes status queue)
    metrics = st.session_state.training_manager.get_metrics()
    
    # Then check for status changes (critical for completion detection)
    current_status = st.session_state.training_manager.get_training_status()
    status_changed = current_status != st.session_state.training_status
    if status_changed:
        st.session_state.training_status = current_status
        
        # Force immediate refresh when status changes (especially for completion)
        if current_status in ['complete', 'error']:
            st.success(f"🎉 Training {current_status}!") if current_status == 'complete' else st.error(f"❌ Training {current_status}")
            time.sleep(1)  # Brief pause to show the message
            st.rerun()
    
    if metrics:
        # Display training health alerts
        with health_placeholder.container():
            health_status = metrics.get('training_health_status', 'unknown')
            health_warnings = metrics.get('health_warnings', [])
            
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
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_loss = metrics.get('current_loss', 0)
                loss_delta = metrics.get('loss_delta', 0)
                delta_color = "normal" if abs(loss_delta) < 0.01 else ("inverse" if loss_delta < 0 else "off")
                
                st.metric(
                    "Training Loss",
                    f"{current_loss:.4f}",
                    delta=f"{loss_delta:.4f}",
                    delta_color=delta_color
                )
            
            with col2:
                current_step = metrics.get('current_step', 0)
                total_steps = metrics.get('total_steps', 1)
                progress_pct = (current_step/total_steps)*100 if total_steps > 0 else 0
                
                st.metric(
                    "Progress",
                    f"{current_step}/{total_steps}",
                    delta=f"{progress_pct:.1f}%"
                )
            
            with col3:
                lr = metrics.get('learning_rate', 0)
                st.metric(
                    "Learning Rate",
                    f"{lr:.2e}" if lr > 0 else "N/A"
                )
            
            with col4:
                elapsed = int(metrics.get('elapsed_time', 0))
                st.metric(
                    "Elapsed Time",
                    f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"
                )
            
            # Secondary metrics row (if validation is enabled)
            if 'eval_loss' in metrics or 'character_consistency' in metrics:
                st.markdown("---")
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    if 'eval_loss' in metrics:
                        eval_loss = metrics['eval_loss']
                        st.metric(
                            "Validation Loss",
                            f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else str(eval_loss),
                            help="Loss on validation set"
                        )
                
                with col6:
                    if 'character_consistency' in metrics:
                        # Use the new deep-dive renderer
                        render_consistency_deep_dive(metrics)
                    else:
                        st.metric(
                            "Character Consistency",
                            "N/A",
                            help="Calculated during evaluation."
                        )
                
                with col7:
                    if 'avg_consistency' in metrics:
                        avg_consistency = metrics['avg_consistency']
                        st.metric(
                            "Avg Consistency",
                            f"{avg_consistency:.2f}" if isinstance(avg_consistency, (int, float)) else str(avg_consistency),
                            help="Overall character consistency score"
                        )
                
                with col8:
                    training_health = metrics.get('training_health_status', 'unknown').title()
                    health_color = {"Healthy": "normal", "Warning": "inverse", "Critical": "off"}.get(training_health, "normal")
                    st.metric(
                        "Training Health",
                        training_health,
                        delta_color=health_color
                    )
        
        # Enhanced loss curve with multiple metrics
        if 'loss_history' in metrics and metrics['loss_history']:
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
                steps = list(range(len(metrics['loss_history'])))
                
                # Build chart data
                chart_data = pd.DataFrame({
                    'Step': steps,
                    'Training Loss': metrics['loss_history']
                })
                
                # Create figure with secondary y-axis for character consistency
                fig = px.line(
                    chart_data, x='Step', y='Training Loss',
                    title="Training Progress Over Time",
                    template="plotly_dark"
                )
                
                # Add validation loss if available
                if 'eval_loss' in metrics:
                    # Try to get eval history from metrics first, then from training manager
                    eval_history = metrics.get('eval_loss_history', [])
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
                if 'health_warnings' in metrics and metrics['health_warnings']:
                    warning_step = metrics.get('current_step', len(steps))
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
                if 'character_consistency' in metrics:
                    st.markdown("### Character Consistency")
                    
                    # Create a simple consistency indicator
                    consistency_score = metrics['character_consistency']
                    
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
                        last_eval_step = metrics.get('consistency_last_eval_step', 0)
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