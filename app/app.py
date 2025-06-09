import os
# Disable Streamlit's autoreload file-watcher early to avoid PyTorch inspection
# errors. Must be set before importing Streamlit so that Streamlit reads the
# configuration on startup.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pandas as pd
import json
import time
import torch
# Prevent Streamlit from trying to treat ``torch.classes`` like a normal
# Python package – this triggers a RuntimeError during Streamlit's module
# scan on some versions.  Overriding ``__path__`` with an empty list neuters
# the problematic attribute and keeps PyTorch fully functional.
torch.classes.__path__ = []

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.character import CharacterManager
from utils.dataset import DatasetManager
from utils.training import TrainingManager
from utils.inference import InferenceManager

# Page config
st.set_page_config(
    page_title="🎭 Character AI Training Studio",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main theme variables */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
    }
    
    /* Override Streamlit's default styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom card styling */
    .custom-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--accent-color);
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-training { background-color: var(--warning-color); }
    .status-complete { background-color: var(--success-color); }
    .status-error { background-color: var(--error-color); }
    .status-idle { background-color: var(--text-secondary); }
    
    /* Animated gradient text */
    .gradient-text {
        background: linear-gradient(45deg, #6366f1, #8b5cf6, #06b6d4);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient 3s ease infinite;
        font-weight: 700;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
        border: none;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed var(--primary-color);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        background: rgba(99, 102, 241, 0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    
    /* Custom animations */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .pulse { animation: pulse 2s infinite; }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 8px;
        border: none;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'character_manager' not in st.session_state:
        st.session_state.character_manager = CharacterManager()
    if 'dataset_manager' not in st.session_state:
        # Auto-detect inference engine, but allow override
        preferred_engine = os.getenv('INFERENCE_ENGINE', None)
        st.session_state.dataset_manager = DatasetManager(preferred_engine=preferred_engine)
    if 'training_manager' not in st.session_state:
        st.session_state.training_manager = TrainingManager()
    if 'inference_manager' not in st.session_state:
        st.session_state.inference_manager = InferenceManager()
    if 'current_character' not in st.session_state:
        st.session_state.current_character = None
    if 'training_status' not in st.session_state:
        st.session_state.training_status = 'idle'
    if 'dataset_preview' not in st.session_state:
        st.session_state.dataset_preview = None
    if 'selected_engine' not in st.session_state:
        # Display current engine in sidebar
        st.session_state.selected_engine = st.session_state.dataset_manager.inference_engine.name

def render_header():
    """Render the beautiful header"""
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="gradient-text" style="font-size: 3rem; margin-bottom: 0.5rem;">
                🎭 Character AI Training Studio
            </h1>
            <p style="font-size: 1.2rem; color: #cbd5e1; margin-bottom: 2rem;">
                Transform character cards into intelligent AI companions
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: #f8fafc;">🚀 AI Studio</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=["📁 Character Upload", "🔍 Dataset Preview", "⚙️ Training Config", "📊 Training Dashboard", "🧪 Model Testing"],
            icons=["upload", "search", "gear", "graph-up", "flask"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#06b6d4", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "padding": "0.5rem 1rem",
                    "--hover-color": "rgba(99, 102, 241, 0.1)",
                },
                "nav-link-selected": {"background-color": "rgba(99, 102, 241, 0.2)"},
            }
        )
        
        # ------------------------------------------------------------------
        # Inference-engine selector
        # ------------------------------------------------------------------

        from app.utils.inference_engines import InferenceEngineFactory

        # Build a map of available engines (name ➜ internal key)
        _engine_key_map = {
            "vLLM": "vllm",
            "LM Studio": "lmstudio",
            "llama-cpp": "llamacpp",
            "Transformers": "transformers",
        }

        available_engine_names = []
        for friendly_name, key in _engine_key_map.items():
            try:
                if InferenceEngineFactory.create_engine(key).is_available():
                    available_engine_names.append(friendly_name)
            except Exception:
                # If creation fails we treat the engine as unavailable
                continue

        if not available_engine_names:
            available_engine_names = ["Transformers"]

        selected_engine_friendly = st.selectbox(
            "🛠️ Inference Engine",
            available_engine_names,
            index=available_engine_names.index(st.session_state.selected_engine)
            if st.session_state.selected_engine in available_engine_names else 0,
            help="Select which backend to use for text generation"
        )

        # If the user picked a different engine we recreate the DatasetManager
        if selected_engine_friendly != st.session_state.selected_engine:
            internal_key = _engine_key_map[selected_engine_friendly]
            st.session_state.selected_engine = selected_engine_friendly
            st.session_state.dataset_manager = DatasetManager(preferred_engine=internal_key)
            st.experimental_rerun()
        
        # Status indicator
        status_colors = {
            'idle': '#94a3b8',
            'training': '#f59e0b',
            'complete': '#10b981',
            'error': '#ef4444'
        }
        
        status_text = {
            'idle': 'Ready',
            'training': 'Training in Progress',
            'complete': 'Training Complete',
            'error': 'Error Occurred'
        }
        
        st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; align-items: center;">
                    <div class="status-indicator" style="background-color: {status_colors[st.session_state.training_status]};"></div>
                    <span style="font-weight: 500;">Status: {status_text[st.session_state.training_status]}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Current inference engine
        engine_model = "PersonalityEngine-24B" if st.session_state.selected_engine == "vLLM" else "Auto-detected"
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 0.5rem 0; color: #f8fafc;">Data Generation</h4>
                <p style="margin: 0; color: #06b6d4;"><strong>{st.session_state.selected_engine}</strong></p>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem; color: #94a3b8;">
                    Model: {engine_model}
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Character info
        if st.session_state.current_character:
            char = st.session_state.current_character
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0 0 0.5rem 0; color: #f8fafc;">Current Character</h4>
                    <p style="margin: 0; color: #cbd5e1;"><strong>{char.get('name', 'Unknown')}</strong></p>
                    <p style="margin: 0.25rem 0 0 0; font-size: 0.9rem; color: #94a3b8;">
                        {char.get('description', 'No description')[:100]}...
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    return selected

def page_character_upload():
    """Character upload and card management page"""
    st.markdown('<h2 class="gradient-text">📁 Character Card Upload</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: white; margin-top: 0;">Upload Your Character Card</h3>
                <p style="color: rgba(255,255,255,0.8);">
                    Upload a SillyTavern-compatible JSON character card to begin training your AI character.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a character card file",
            type=['json'],
            help="Upload a .json character card file"
        )
        
        if uploaded_file is not None:
            try:
                character_data = json.load(uploaded_file)
                st.session_state.current_character = character_data
                st.success("✅ Character card loaded successfully!")
                
                # Display character preview
                st.markdown("### Character Preview")
                
                preview_col1, preview_col2 = st.columns(2)
                
                with preview_col1:
                    st.markdown(f"**Name:** {character_data.get('name', 'Unknown')}")
                    st.markdown(f"**Example:** {character_data.get('mes_example', 'No example available')[:300]}{'...' if len(character_data.get('mes_example', 'No example available')) > 300 else ''}")
                    st.markdown(f"**Scenario:** {character_data.get('scenario', 'No scenario available')[:300]}{'...' if len(character_data.get('scenario', 'No scenario available')) > 300 else ''}")

                with preview_col2:
                    description = character_data.get('description', 'No description available')
                    st.markdown(f"**Description:** {description[:200]}{'...' if len(description) > 200 else ''}")
                
                # Personality preview
                if 'personality' in character_data:
                    st.markdown("**Personality:**")
                    personality = character_data['personality']
                    st.markdown(f"{personality[:300]}{'...' if len(personality) > 300 else ''}")
                
            except Exception as e:
                st.error(f"❌ Error loading character card: {str(e)}")
    
    with col2:
        st.markdown("""
            <div style="background: rgba(99, 102, 241, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
                <h4 style="color: #6366f1; margin-top: 0;">💡 Tips</h4>
                <ul style="color: #cbd5e1; font-size: 0.9rem;">
                    <li>Ensure your JSON file follows SillyTavern format</li>
                    <li>Rich character descriptions lead to better training results</li>
                    <li>Include personality traits and example dialogue</li>
                    <li>Keep descriptions concise but detailed</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

def page_dataset_preview():
    """Dataset generation and preview page"""
    st.markdown('<h2 class="gradient-text">🔍 Dataset Preview & Generation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Dataset Generation Settings")
        
        # Show current inference engine with model info
        if st.session_state.selected_engine == "vLLM":
            engine_info = f"🔧 Using **{st.session_state.selected_engine}** with **PersonalityEngine-24B** for high-quality character responses"
        else:
            engine_info = f"🔧 Using **{st.session_state.selected_engine}** for text generation"
        
        st.info(engine_info)
        
        with st.form("dataset_generation"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                num_samples = st.slider("Number of samples", 50, 1000, 200, step=50)
                temperature = st.slider("Temperature", 0.5, 1.2, 0.8, step=0.1)
            
            with col_b:
                max_tokens = st.slider("Max tokens per sample", 100, 800, 300, step=50)
                top_p = st.slider("Top-p", 0.7, 1.0, 0.9, step=0.05)
            
            generate_button = st.form_submit_button("🚀 Generate Dataset", use_container_width=True)
        
        if generate_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Generate dataset (run async function in sync context)
                with st.spinner("Generating synthetic dataset..."):
                    import asyncio
                    
                    # Create progress callback
                    def update_progress(p):
                        progress_bar.progress(p)
                        status_text.text(f"Generated {int(p * num_samples)} / {num_samples} samples")
                    
                    # Run the async generation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        dataset = loop.run_until_complete(
                            st.session_state.dataset_manager.generate_dataset(
                                st.session_state.current_character,
                                num_samples=num_samples,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                progress_callback=update_progress
                            )
                        )
                    finally:
                        loop.close()
                
                st.session_state.dataset_preview = dataset
                progress_bar.progress(1.0)
                status_text.text("Dataset generation complete!")
                st.success(f"✅ Generated {len(dataset)} samples successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating dataset: {str(e)}")
    
    with col2:
        if st.session_state.dataset_preview:
            st.markdown("### Dataset Statistics")
            dataset = st.session_state.dataset_preview
            
            # Stats
            avg_length = sum(len(sample['messages'][2]['content'].split()) for sample in dataset) / len(dataset)
            unique_responses = len(set(sample['messages'][2]['content'] for sample in dataset))
            
            st.metric("Total Samples", len(dataset))
            st.metric("Avg Response Length", f"{avg_length:.1f} words")
            st.metric("Unique Responses", f"{unique_responses}/{len(dataset)}")
            
            # Quality score
            quality_score = min(100, (unique_responses / len(dataset)) * 100)
            st.metric("Quality Score", f"{quality_score:.1f}%")
    
    # Dataset preview
    if st.session_state.dataset_preview:
        st.markdown("### Sample Preview")
        
        # Sample selector
        sample_idx = st.selectbox("Select sample to preview", range(min(20, len(st.session_state.dataset_preview))))
        
        if sample_idx is not None:
            sample = st.session_state.dataset_preview[sample_idx]
            
            # Display conversation
            for message in sample['messages']:
                role = message['role']
                content = message['content']
                
                if role == 'system':
                    st.markdown(f"**🔧 System:** {content[:200]}...")
                elif role == 'user':
                    st.markdown(f"**👤 User:** {content}")
                else:
                    st.markdown(f"**🎭 {st.session_state.current_character.get('name', 'Assistant')}:** {content}")
                
                st.markdown("---")

def page_training_config():
    """Training configuration page"""
    st.markdown('<h2 class="gradient-text">⚙️ Training Configuration</h2>', unsafe_allow_html=True)
    
    if not st.session_state.dataset_preview:
        st.warning("⚠️ Please generate a dataset first.")
        return
    
    dataset_size = len(st.session_state.dataset_preview)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Hyperparameter Configuration")
        
        with st.form("training_config"):
            # Basic settings
            st.markdown("#### Basic Settings")
            col_a, col_b = st.columns(2)
            
            with col_a:
                epochs = st.slider("Epochs", 1, 10, min(3, max(1, 800 // dataset_size)))
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4],
                    value=2e-5,
                    format_func=lambda x: f"{x:.0e}"
                )
                batch_size = st.selectbox("Batch Size", [1, 2, 4, 8], index=1)
            
            with col_b:
                gradient_accumulation = st.selectbox("Gradient Accumulation Steps", [1, 2, 4, 8], index=1)
                warmup_steps = st.slider("Warmup Steps", 0, 100, 10)
                max_grad_norm = st.slider("Max Gradient Norm", 0.5, 2.0, 1.0, step=0.1)
            
            # LoRA settings
            st.markdown("#### LoRA Configuration")
            col_c, col_d = st.columns(2)
            
            with col_c:
                lora_r = st.slider("LoRA Rank (r)", 4, 64, 8, step=4)
                lora_alpha = st.slider("LoRA Alpha", 8, 128, 32, step=8)
            
            with col_d:
                lora_dropout = st.slider("LoRA Dropout", 0.0, 0.2, 0.05, step=0.01)
                target_modules = st.multiselect(
                    "Target Modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out"],
                    default=["q_proj", "k_proj", "v_proj", "o_proj"]
                )
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                fp16 = st.checkbox("Enable FP16", value=False)
                save_steps = st.slider("Save Every N Steps", 50, 500, 100, step=50)
                logging_steps = st.slider("Log Every N Steps", 5, 50, 10, step=5)
                eval_steps = st.slider("Evaluation Steps", 50, 200, 100, step=25)
            
            start_training = st.form_submit_button("🚀 Start Training", use_container_width=True)
    
    with col2:
        st.markdown("### Training Recommendations")
        
        # Calculate training recommendations
        total_steps = (dataset_size * epochs) // (batch_size * gradient_accumulation)
        overfitting_risk = "High" if total_steps > 500 else "Medium" if total_steps > 200 else "Low"
        
        # Display recommendations
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0;">📊 Training Analysis</h4>
                <p><strong>Total Steps:</strong> {total_steps}</p>
                <p><strong>Overfitting Risk:</strong> <span style="color: {'#ef4444' if overfitting_risk == 'High' else '#f59e0b' if overfitting_risk == 'Medium' else '#10b981'}">{overfitting_risk}</span></p>
                <p><strong>Est. Time:</strong> {total_steps * 2 // 60} minutes</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Warnings
        if overfitting_risk == "High":
            st.warning("⚠️ High overfitting risk! Consider reducing epochs or increasing dataset size.")
        
        if learning_rate > 5e-5:
            st.warning("⚠️ High learning rate may cause instability.")
        
        # Tips
        st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
                <h4 style="color: #10b981; margin-top: 0;">💡 Tips</h4>
                <ul style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0;">
                    <li>Start with lower learning rates for stability</li>
                    <li>Monitor loss curves for overfitting</li>
                    <li>Use gradient accumulation for larger effective batch sizes</li>
                    <li>Higher LoRA rank = more parameters but slower training</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    if start_training:
        # Store training config and start training
        config = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation,
            'warmup_steps': warmup_steps,
            'max_grad_norm': max_grad_norm,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'target_modules': target_modules,
            'fp16': fp16,
            'save_steps': save_steps,
            'logging_steps': logging_steps,
            'eval_steps': eval_steps
        }
        
        try:
            st.session_state.training_config = config
            
            # Start training
            st.session_state.training_manager.start_training(
                st.session_state.current_character,
                st.session_state.dataset_preview,
                config
            )
            
            # Update status
            st.session_state.training_status = 'training'
            
            st.success("🚀 Training started! Switch to the Training Dashboard to monitor progress.")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to start training: {str(e)}")
            st.session_state.training_status = 'error'

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
    
    # Real-time metrics
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Update training status from manager
    current_status = st.session_state.training_manager.get_training_status()
    if current_status != st.session_state.training_status:
        st.session_state.training_status = current_status
    
    # Get training metrics
    metrics = st.session_state.training_manager.get_metrics()
    
    if metrics:
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Loss",
                    f"{metrics.get('current_loss', 0):.4f}",
                    delta=f"{metrics.get('loss_delta', 0):.4f}"
                )
            
            with col2:
                current_step = metrics.get('current_step', 0)
                total_steps = metrics.get('total_steps', 1)
                st.metric(
                    "Progress",
                    f"{current_step}/{total_steps}",
                    delta=f"{(current_step/total_steps)*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    "Learning Rate",
                    f"{metrics.get('learning_rate', 0):.2e}"
                )
            
            with col4:
                elapsed = metrics.get('elapsed_time', 0)
                st.metric(
                    "Elapsed Time",
                    f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}"
                )
        
        # Loss curve
        if 'loss_history' in metrics and metrics['loss_history']:
            with chart_placeholder.container():
                st.markdown("### Loss Curve")
                
                loss_df = pd.DataFrame({
                    'Step': range(len(metrics['loss_history'])),
                    'Loss': metrics['loss_history']
                })
                
                fig = px.line(
                    loss_df, x='Step', y='Loss',
                    title="Training Loss Over Time",
                    template="plotly_dark"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh for real-time updates
    if st.session_state.training_status == 'training':
        time.sleep(2)
        st.rerun()

def page_model_testing():
    """Model testing and inference page"""
    st.markdown('<h2 class="gradient-text">🧪 Model Testing</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Test Your Trained Model")
        
        # Model selection
        available_models = st.session_state.inference_manager.get_available_models()
        
        if not available_models:
            st.info("ℹ️ No trained models available. Complete training first.")
            return
        
        selected_model = st.selectbox("Select Model", available_models)
        
        # Test prompt
        test_prompt = st.text_area(
            "Enter your test prompt:",
            placeholder="Tell me about yourself...",
            height=100
        )
        
        # Generation settings
        with st.expander("⚙️ Generation Settings"):
            col_a, col_b = st.columns(2)
            with col_a:
                max_new_tokens = st.slider("Max New Tokens", 50, 500, 150)
                temperature = st.slider("Temperature", 0.1, 1.5, 0.8)
            with col_b:
                top_p = st.slider("Top-p", 0.1, 1.0, 0.9)
                repetition_penalty = st.slider("Repetition Penalty", 1.0, 1.5, 1.1)
        
        if st.button("🚀 Generate Response", use_container_width=True):
            if test_prompt.strip():
                with st.spinner("Generating response..."):
                    response = st.session_state.inference_manager.generate_response(
                        selected_model,
                        test_prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty
                    )
                
                st.markdown("### Response")
                st.markdown(f"""
                    <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #6366f1;">
                        <p style="margin: 0; color: #f8fafc;">{response}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please enter a test prompt.")
    
    with col2:
        st.markdown("### Quick Tests")
        
        quick_tests = [
            "What drives you in life?",
            "Describe your greatest fear.",
            "Tell me about your homeland.",
            "What's your biggest regret?",
            "How do you handle failure?"
        ]
        
        for i, prompt in enumerate(quick_tests):
            if st.button(f"🎯 {prompt}", key=f"quick_test_{i}"):
                # Auto-fill the test prompt
                st.session_state.quick_test_prompt = prompt
        
        # Model comparison
        st.markdown("### Model Comparison")
        
        if len(available_models) > 1:
            st.info("Compare responses from different model checkpoints to evaluate training progress.")
            
            if st.button("📊 Compare Models"):
                st.info("Model comparison feature coming soon!")
        else:
            st.info("Train multiple checkpoints to enable model comparison.")

def main():
    """Main app function"""
    init_session_state()
    render_header()
    
    # Sidebar navigation
    selected_page = render_sidebar()
    
    # Page routing
    if selected_page == "📁 Character Upload":
        page_character_upload()
    elif selected_page == "🔍 Dataset Preview":
        page_dataset_preview()
    elif selected_page == "⚙️ Training Config":
        page_training_config()
    elif selected_page == "📊 Training Dashboard":
        page_training_dashboard()
    elif selected_page == "🧪 Model Testing":
        page_model_testing()
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: #64748b; border-top: 1px solid rgba(100, 116, 139, 0.2); margin-top: 3rem;">
            <p>🎭 Character AI Training Studio • Built with ❤️ and Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 