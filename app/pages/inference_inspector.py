"""
🔍 Inference Inspector Page (R3-4)

This page allows developers to inspect model behavior during inference:
1. Browse available observability logs by request ID
2. View detailed request information and generation parameters
3. Visualize attention weights with interactive heatmaps
4. Analyze token probability distributions
5. Debug model decision-making processes

Enables "popping the hood" on language model inference for debugging and analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

from utils.observability import get_observability_logger
from utils.error_handling import streamlit_error_boundary


@streamlit_error_boundary
def page_inference_inspector():
    """Main inference inspector page"""
    st.markdown('<h2 class="gradient-text">🔍 Inference Inspector</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Debug Model Behavior**: Inspect intermediate states during model inference to understand 
    why a model makes specific decisions. View attention patterns, token probabilities, and generation metadata.
    """)
    
    # Initialize observability logger
    if 'observability_logger' not in st.session_state:
        st.session_state.observability_logger = get_observability_logger()
    
    obs_logger = st.session_state.observability_logger
    
    # Get available logs
    available_logs = obs_logger.list_available_logs()
    
    if not available_logs:
        st.info("ℹ️ **No observability logs available**. Generate responses with observability enabled to start collecting data.")
        
        with st.expander("🔧 How to Enable Observability"):
            st.markdown("""
            **For Developers**: Observability logging must be enabled during model inference.
            
            ```python
            # Enable observability in inference calls
            response = inference_manager.generate_response(
                model_path="your-model",
                prompt="your prompt",
                enable_observability=True,  # This captures intermediate states
                request_id="optional-custom-id"
            )
            ```
            
            **Performance Note**: Observability adds computational overhead by capturing attention weights 
            and hidden states, so it should only be enabled for debugging purposes.
            """)
        return
    
    st.success(f"📊 Found **{len(available_logs)}** observability logs")
    
    # Request ID selection interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search/filter functionality
        search_filter = st.text_input(
            "🔍 Filter request IDs",
            placeholder="Type to filter logs...",
            help="Filter logs by request ID or part of ID"
        )
        
        # Apply filter
        filtered_logs = available_logs
        if search_filter:
            filtered_logs = [log_id for log_id in available_logs if search_filter.lower() in log_id.lower()]
        
        if not filtered_logs:
            st.warning(f"No logs match filter: '{search_filter}'")
            return
        
        # Request ID selection
        selected_request_id = st.selectbox(
            "Select Request ID to Inspect",
            options=filtered_logs,
            help="Choose a request ID to view detailed observability data"
        )
    
    with col2:
        st.markdown("### Quick Actions")
        
        if st.button("🔄 Refresh Logs"):
            st.rerun()
        
        if st.button("🧹 Cleanup Old Logs"):
            deleted_count = obs_logger.cleanup_old_logs(max_logs=50)
            if deleted_count > 0:
                st.success(f"Deleted {deleted_count} old logs")
                st.rerun()
            else:
                st.info("No old logs to cleanup")
    
    if not selected_request_id:
        return
    
    # Load selected log data
    log_data = obs_logger.get_log_data(selected_request_id)
    
    if not log_data:
        st.error(f"❌ Could not load data for request ID: {selected_request_id}")
        return
    
    # Display request overview
    st.markdown("---")
    st.markdown(f"### 📋 Request Overview: `{selected_request_id}`")
    
    # Metadata in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**🤖 Model:** `{log_data.get('model_path', 'Unknown')}`")
        st.markdown(f"**⏰ Timestamp:** {log_data.get('timestamp', 'Unknown')}")
        
    with col2:
        gen_config = log_data.get('generation_config', {})
        st.markdown(f"**🌡️ Temperature:** {gen_config.get('temperature', 'N/A')}")
        st.markdown(f"**🎯 Top-p:** {gen_config.get('top_p', 'N/A')}")
        
    with col3:
        st.markdown(f"**📏 Max Tokens:** {gen_config.get('max_tokens', 'N/A')}")
        st.markdown(f"**🎲 Seed:** {gen_config.get('seed', 'None')}")
    
    # Prompt and Response
    st.markdown("### 💬 Conversation")
    
    with st.expander("📝 Prompt", expanded=True):
        st.code(log_data.get('prompt', ''), language='text')
    
    with st.expander("🤖 Response", expanded=True):
        st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #6366f1;">
            <p style="margin: 0; color: #f8fafc;">{log_data.get('response', '')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization tabs
    st.markdown("---")
    st.markdown("### 📊 Model Behavior Analysis")
    
    tab1, tab2, tab3 = st.tabs(["🧠 Attention Patterns", "📈 Token Probabilities", "🔧 Technical Details"])
    
    with tab1:
        render_attention_visualization(log_data)
    
    with tab2:
        render_token_probability_visualization(log_data)
    
    with tab3:
        render_technical_details(log_data)


def render_attention_visualization(log_data: Dict[str, Any]):
    """Render attention weight visualizations"""
    st.markdown("#### 🧠 Attention Weight Analysis")
    
    attention_shapes = log_data.get('attention_weights_shape', [])
    
    if not attention_shapes:
        st.info("ℹ️ No attention weights captured for this request.")
        st.markdown("""
        **Note**: Attention weights are only available when the model supports attention output 
        and observability was properly enabled during generation.
        """)
        return
    
    st.success(f"📊 Captured attention weights from **{len(attention_shapes)}** layers")
    
    # Display attention shapes info
    with st.expander("📏 Attention Weight Shapes", expanded=False):
        for i, shape in enumerate(attention_shapes):
            st.code(f"Layer {i}: {shape}", language='text')
            
        st.markdown("""
        **Shape format**: `[batch_size, num_heads, sequence_length, sequence_length]`
        - **Batch size**: Number of sequences processed together
        - **Num heads**: Number of attention heads in the layer
        - **Sequence length**: Length of input + generated tokens
        """)
    
    # Mock attention visualization (since we only stored shapes, not actual data)
    st.markdown("#### 🎯 Attention Heatmap (Sample)")
    st.info("💡 **Implementation Note**: This shows a sample attention pattern. In production, this would display actual attention weights from the stored tensor data.")
    
    # Create sample attention heatmap
    import numpy as np
    
    if attention_shapes:
        # Use the first layer's dimensions for the sample
        shape = attention_shapes[0]
        if len(shape) >= 3:
            seq_len = min(shape[-1], 20)  # Limit display size
            
            # Create sample attention data
            attention_data = np.random.rand(seq_len, seq_len)
            
            # Generate token labels
            tokens = [f"tok_{i}" for i in range(seq_len)]
            
            fig = go.Figure(data=go.Heatmap(
                z=attention_data,
                x=tokens,
                y=tokens,
                colorscale='Viridis',
                showscale=True,
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Attention Weights Heatmap (Sample Data)",
                xaxis_title="Key Tokens",
                yaxis_title="Query Tokens",
                width=600,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **How to read this heatmap**:
            - **Darker colors** = Higher attention weights
            - **X-axis (Key)**: Tokens being attended TO
            - **Y-axis (Query)**: Tokens attending FROM
            - **Diagonal patterns**: Self-attention
            - **Off-diagonal patterns**: Cross-token attention
            """)


def render_token_probability_visualization(log_data: Dict[str, Any]):
    """Render token probability distributions"""
    st.markdown("#### 📈 Token Probability Analysis")
    
    token_probs = log_data.get('token_probabilities', {})
    
    if not token_probs:
        st.info("ℹ️ No token probabilities captured for this request.")
        st.markdown("""
        **Note**: Token probabilities require `output_scores=True` during generation 
        and are extracted from the final generation step.
        """)
        return
    
    st.success(f"📊 Captured probabilities for **{len(token_probs)}** top tokens")
    
    # Create probability bar chart
    if token_probs:
        tokens = list(token_probs.keys())
        probabilities = list(token_probs.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=tokens,
                y=probabilities,
                name='Token Probability',
                marker_color='rgba(99, 102, 241, 0.8)',
                text=[f'{p:.3f}' for p in probabilities],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Top Token Probabilities",
            xaxis_title="Tokens",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Probability table
        with st.expander("📊 Detailed Probability Table"):
            prob_df = pd.DataFrame({
                'Token': tokens,
                'Probability': probabilities,
                'Percentage': [f'{p*100:.2f}%' for p in probabilities]
            })
            st.dataframe(prob_df, use_container_width=True)
        
        # Analysis insights
        st.markdown("#### 💡 Analysis Insights")
        
        max_prob = max(probabilities)
        max_token = tokens[probabilities.index(max_prob)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🎯 Highest Probability Token", f'"{max_token}"', f"{max_prob:.3f}")
            
        with col2:
            entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities)
            st.metric("🌀 Decision Entropy", f"{entropy:.2f}", help="Higher = more uncertain")
        
        if max_prob > 0.8:
            st.success("✅ **High Confidence**: The model was very confident in its token choice.")
        elif max_prob > 0.5:
            st.info("⚖️ **Medium Confidence**: The model had moderate confidence.")
        else:
            st.warning("⚠️ **Low Confidence**: The model was uncertain between multiple tokens.")


def render_technical_details(log_data: Dict[str, Any]):
    """Render technical details and metadata"""
    st.markdown("#### 🔧 Technical Implementation Details")
    
    # Generation configuration
    st.markdown("##### ⚙️ Generation Configuration")
    gen_config = log_data.get('generation_config', {})
    
    if gen_config:
        config_df = pd.DataFrame([
            {'Parameter': k, 'Value': v} for k, v in gen_config.items()
        ])
        st.dataframe(config_df, use_container_width=True)
    else:
        st.info("No generation configuration available.")
    
    # Model architecture info
    st.markdown("##### 🏗️ Model Architecture Information")
    
    attention_shapes = log_data.get('attention_weights_shape', [])
    hidden_shapes = log_data.get('hidden_states_shape', [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Attention Layers**")
        if attention_shapes:
            st.code(f"Layers: {len(attention_shapes)}\nShape format: {attention_shapes[0] if attention_shapes else 'N/A'}")
        else:
            st.code("No attention data captured")
    
    with col2:
        st.markdown("**Hidden States**")
        if hidden_shapes:
            st.code(f"Layers: {len(hidden_shapes)}\nShape format: {hidden_shapes[0] if hidden_shapes else 'N/A'}")
        else:
            st.code("No hidden state data captured")
    
    # Raw data export
    st.markdown("##### 📁 Raw Data Export")
    
    if st.button("📥 Download Raw JSON"):
        import json
        json_str = json.dumps(log_data, indent=2)
        st.download_button(
            label="💾 Download Observability Data",
            data=json_str,
            file_name=f"observability_{log_data.get('request_id', 'unknown')}.json",
            mime="application/json"
        )
    
    # Performance metrics
    st.markdown("##### ⚡ Performance Impact")
    
    st.info("""
    **Observability Performance Notes**:
    - Attention capture adds ~20-30% inference overhead
    - Hidden states capture adds ~10-15% memory usage  
    - Token probability extraction is minimal impact
    - Use only for debugging, not production inference
    """)


# Run the page function if this file is executed directly (for testing)
if __name__ == "__main__":
    page_inference_inspector() 