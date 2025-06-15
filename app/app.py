import asyncio
from asyncio.log import logger
import os

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import time
import torch

torch.classes.__path__ = []

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.character import CharacterManager
from utils.dataset import DatasetManager, QualityLevel, GenerationConfig
from utils.training import TrainingManager
from utils.inference import InferenceManager
from utils.comparison import ComparisonManager

# Configure logging for debugging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log')  # File output
    ]
)

# ✅ FIX: Global singleton to prevent DatasetManager re-initialization
_GLOBAL_DATASET_MANAGER = None

def get_or_create_dataset_manager(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Get or create a singleton DatasetManager instance"""
    global _GLOBAL_DATASET_MANAGER
    
    if _GLOBAL_DATASET_MANAGER is None:
        logger.info(f"🔧 Creating singleton DatasetManager with model: {os.getenv('MODEL_NAME')}")
        _GLOBAL_DATASET_MANAGER = DatasetManager(
            api_key=api_key,
            base_url=base_url,
        )
        logger.info("✅ Singleton DatasetManager created successfully")
    else:
        logger.info("♻️ Reusing existing singleton DatasetManager")
    
    return _GLOBAL_DATASET_MANAGER

# Initialize session state
def init_session_state():
    if 'character_manager' not in st.session_state:
        st.session_state.character_manager = CharacterManager()
    
    # ✅ FIX: Use singleton DatasetManager 
    if 'dataset_manager' not in st.session_state:
        # Get OpenAI configuration from environment
        api_key = os.getenv('OPENAI_API_KEY', "empty")
        base_url = os.getenv('OPENAI_BASE_URL')
        
        try:
            # Use singleton function to prevent re-initialization
            st.session_state.dataset_manager = get_or_create_dataset_manager(
                api_key=api_key,
                base_url=base_url,
            )
            logger.info("✅ DatasetManager successfully set in session state")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DatasetManager: {e}")
            # Set a placeholder to prevent infinite retries
            st.session_state.dataset_manager = None
            st.error(f"Failed to initialize DatasetManager: {e}")
    
    if 'training_manager' not in st.session_state:
        st.session_state.training_manager = TrainingManager()
    if 'inference_manager' not in st.session_state:
        st.session_state.inference_manager = InferenceManager()
    if 'comparison_manager' not in st.session_state:
        st.session_state.comparison_manager = ComparisonManager(st.session_state.inference_manager)
    if 'current_character' not in st.session_state:
        st.session_state.current_character = None
    if 'training_status' not in st.session_state:
        st.session_state.training_status = 'idle'
    if 'dataset_preview' not in st.session_state:
        st.session_state.dataset_preview = None
    if 'dataset_metadata' not in st.session_state:
        st.session_state.dataset_metadata = {}
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = None

def render_healthy_run_example():
    """Renders an SVG example of a healthy training run loss curve."""
    svg_html = """
    <svg width="250" height="150" viewBox="0 0 300 150" xmlns="http://www.w3.org/2000/svg">
        <style>
            .bg { fill: rgba(30, 41, 59, 0.5); }
            .axis-line { stroke: #475569; stroke-width: 1.5; }
            .axis-text { fill: #94a3b8; font-family: 'Inter', sans-serif; font-size: 12px; }
            .loss-curve { stroke: #10b981; stroke-width: 2.5; fill: none; stroke-linecap: round; stroke-linejoin: round; }
            .grid-line { stroke: #334155; stroke-width: 1; stroke-dasharray: 2,3; }
            .title-text { fill: #f1f5f9; font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 500; text-anchor: middle; }
        </style>
        <rect width="300" height="150" class="bg" rx="12"/>
        
        <!-- Title -->
        <text x="150" y="22" class="title-text">💡 Example: Healthy Loss Curve</text>

        <!-- Axes -->
        <line x1="40" y1="35" x2="40" y2="125" class="axis-line"/> <!-- Y-axis -->
        <text x="35" y="40" class="axis-text" text-anchor="end">High</text>
        <text x="35" y="125" class="axis-text" text-anchor="end">Low</text>
        <text x="15" y="85" class="axis-text" transform="rotate(-90 15,85)">Loss</text>

        <line x1="40" y1="125" x2="280" y2="125" class="axis-line"/> <!-- X-axis -->
        <text x="160" y="142" class="axis-text" text-anchor="middle">Training Steps</text>
        
        <!-- Grid Lines -->
        <line x1="40" y1="35" x2="280" y2="35" class="grid-line" />
        <line x1="40" y1="80" x2="280" y2="80" class="grid-line" />

        <!-- Healthy Loss Curve Path -->
        <path d="M 45,50 C 90,55 120,100 270,115" class="loss-curve" />
    </svg>
    """
    st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
            <p style="text-align: center; font-size: 0.9rem; color: #cbd5e1; margin-bottom: 0.5rem;">
                A healthy run shows loss decreasing and stabilizing.
            </p>
            {svg_html}
        </div>
    """, unsafe_allow_html=True)


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

    /* Custom styles for Dataset Explorer */
    .data-cell {
        height: 120px;
        overflow-y: auto;
        padding: 0.5rem;
        border-radius: 6px;
        background-color: rgba(255, 255, 255, 0.03);
        font-size: 0.9em;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .row-divider {
        margin: 0.25rem 0;
        border-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Render the beautiful header
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

# Render the sidebar navigation
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
            options=["📁 Character Upload", "🔍 Dataset Preview", "📚 Dataset Explorer", "⚙️ Training Config", "📊 Training Dashboard", "🧪 Model Testing", "⚔️ Model Comparison"],
            icons=["upload", "search", "table", "gear", "graph-up", "flask", "shuffle"],
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
        
        # Training status
        status_colors = {
            'idle': '#94a3b8',
            'training': '#f59e0b',
            'paused': '#f97316',
            'complete': '#10b981',
            'error': '#ef4444'
        }
        
        status_text = {
            'idle': 'Ready',
            'training': 'Training in Progress',
            'dataset_generation': 'Dataset Generation in Progress',
            'paused': 'Paused',
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
        
        # Character and dataset info
        if st.session_state.current_character:
            char_name = st.session_state.current_character.get("name", "Unknown")
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0 0 0.5rem 0;">🎭 Current Character</h4>
                    <p style="margin: 0; font-weight: 500;">{char_name}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Dataset info
            dataset_info = st.session_state.dataset_manager.get_dataset_info(st.session_state.current_character)
            if dataset_info['exists']:
                st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="margin: 0 0 0.5rem 0;">📊 Dataset</h4>
                        <p style="margin: 0;">Samples: {dataset_info['sample_count']}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Quick actions for current character
        if st.session_state.current_character:
            with st.expander("🗃️ Character Assets", expanded=False):
                char_name = st.session_state.current_character.get("name", "unknown")

                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    if st.button("🗑️ Clear Training", key="clear_training_btn", help="Remove training artifacts"):
                        if st.session_state.training_manager.clear_training_assets(char_name):
                            st.success("Training assets cleared!")
                        else:
                            st.info("No training assets to remove.")

                with col_b:
                    if st.button("⬇️ Export LoRA", key="export_lora_btn", help="Export trained LoRA model"):
                        try:
                            zip_path = st.session_state.training_manager.export_lora(char_name)
                            st.success(f"LoRA exported to {zip_path}")
                        except Exception as e:
                            st.error(str(e))

                with col_c:
                    if st.button("⬇️ Export Checkpoint", key="export_ckpt_btn", help="Export latest checkpoint"):
                        zip_path = st.session_state.training_manager.export_latest_checkpoint(char_name)
                        if zip_path:
                            st.success(f"Checkpoint exported to {zip_path}")
                        else:
                            st.info("No checkpoints found to export.")
    
    return selected

# Character upload and card management page
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

        if st.session_state.current_character:
            st.markdown("### Character Preview")
            st.markdown(f"**Name:** {st.session_state.current_character.get('name', 'Unknown')}")
            st.markdown(f"**Example:** {st.session_state.current_character.get('mes_example', 'No example available')[:300]}{'...' if len(st.session_state.current_character.get('mes_example', 'No example available')) > 300 else ''}")
            st.markdown(f"**Scenario:** {st.session_state.current_character.get('scenario', 'No scenario available')[:300]}{'...' if len(st.session_state.current_character.get('scenario', 'No scenario available')) > 300 else ''}")
            st.markdown(f"**Description:** {st.session_state.current_character.get('description', 'No description available')[:200]}{'...' if len(st.session_state.current_character.get('description', 'No description available')) > 200 else ''}")
            st.markdown(f"**Personality:** {st.session_state.current_character.get('personality', 'No personality available')[:300]}{'...' if len(st.session_state.current_character.get('personality', 'No personality available')) > 300 else ''}")
        
        if uploaded_file is not None:
            try:
                character_data = json.load(uploaded_file)
                st.session_state.current_character = character_data
                
                # Auto-load existing dataset if available
                dataset_with_metadata = st.session_state.dataset_manager.load_dataset_with_metadata(character_data)
                if dataset_with_metadata:
                    existing_dataset, metadata = dataset_with_metadata
                    st.session_state.dataset_preview = existing_dataset
                    st.session_state.dataset_metadata = metadata
                    st.success(f"✅ Character card loaded with existing dataset ({len(existing_dataset)} samples)!")
                else:
                    st.session_state.dataset_preview = None
                    st.session_state.dataset_metadata = {}
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

# Dataset generation and preview page
def page_dataset_preview():
    """Dataset generation and preview page"""
    st.markdown('<h2 class="gradient-text">🔍 Dataset Preview & Generation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return
    
    # Check for existing dataset
    dataset_info = st.session_state.dataset_manager.get_dataset_info(st.session_state.current_character)
    
    # Enhanced generation mode tabs
    generation_tab, standard_tab, enhanced_tab, premium_tab, quality_tab = st.tabs([
        "📊 Overview", 
        "📝 Basic Generation",
        "⭐ Enhanced Generation", 
        "🌟 Premium Generation",
        "🎯 Factual Q&A Generation"
    ])
    
    with generation_tab:
        # Show existing dataset info if available
        if dataset_info['exists']:
            st.markdown("### 📂 Existing Dataset Found")
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Existing Samples", dataset_info['sample_count'])
            with col_info2:
                if st.button("📂 Load Existing", use_container_width=True):
                    # ✅ FIX: Check if generation is in progress
                    if st.session_state.get('_generating_dataset', False):
                        st.warning("⚠️ Dataset generation in progress. Please wait.")
                    else:
                        dataset_with_metadata = st.session_state.dataset_manager.load_dataset_with_metadata(st.session_state.current_character)
                    if dataset_with_metadata:
                        existing_dataset, metadata = dataset_with_metadata
                        st.session_state.dataset_preview = existing_dataset
                        st.session_state.dataset_metadata = metadata
                        st.success(f"✅ Loaded {len(existing_dataset)} existing samples!")
                            # ✅ FIX: Use st.experimental_rerun instead of st.rerun for better stability
                        st.rerun()
            with col_info3:
                if st.button("🗑️ Reset Dataset", use_container_width=True):
                    # ✅ FIX: Check if generation is in progress
                    if st.session_state.get('_generating_dataset', False):
                        st.warning("⚠️ Dataset generation in progress. Please wait.")
                    else:
                        if st.session_state.dataset_manager.delete_dataset(st.session_state.current_character):
                            st.session_state.dataset_preview = None
                            st.session_state.dataset_metadata = {}
                            st.success("✅ Dataset reset! Generate a new one below.")
                            # ✅ FIX: Use controlled rerun
                            st.rerun()
            
            # Show system prompt configuration
            system_config = dataset_info.get('system_prompt_config', {})
            if system_config:
                if system_config.get('type') == 'none':
                    st.info("💡 Dataset has no system prompts (removed after generation)")
                elif system_config.get('type') == 'custom':
                    prompt_text = system_config.get('prompt', '')
                    st.info(f"💡 Dataset uses custom system prompt: \"{prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}\"")
                elif system_config.get('type') == 'temporal':
                    st.info("💡 Dataset uses temporal context system prompts (varying per sample)")
                    
            st.markdown("---")
        
        # Dataset statistics
        if st.session_state.dataset_preview:
            st.markdown("### Dataset Statistics")
            dataset = st.session_state.dataset_preview
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Stats
            avg_length = sum(len(sample['messages'][2]['content'].split()) for sample in dataset) / len(dataset)
            unique_responses = len(set(sample['messages'][2]['content'] for sample in dataset))
            
            with col1:
                st.metric("Total Samples", len(dataset))
            with col2:
                st.metric("Avg Response Length", f"{avg_length:.1f} words")
            with col3:
                st.metric("Unique Responses", f"{unique_responses}/{len(dataset)}")
            with col4:
                # Quality score
                quality_score = min(100, (unique_responses / len(dataset)) * 100)
                st.metric("Quality Score", f"{quality_score:.1f}%")
        
        # Info about generation modes
        st.markdown("""
        ### 🎯 Generation Methods
        
        **Standard Generation**: Fast, direct generation with basic quality filtering.
        - Good for: Quick datasets, testing, small characters
        - Speed: ~1-2 samples per second
        
        **Quality-First Generation**: Generate many samples, then use AI to select the best.
        - Good for: Production models, complex characters, best quality
        - Speed: Slower but much higher quality
        """)
        
        # ----------------------------
        # ✨ Augment Baseline Questions
        # ----------------------------
        st.markdown("### ✨ Augment Baseline Questions")
        
        # Split the UI into tabs for better organization and to avoid nested expanders
        qa_tabs = st.tabs(["⚙️ Setup & Generation", "🔍 Review Generated Questions"])
        
        with qa_tabs[0]:
            # ✅ NEW: Sampling Configuration for Question Generation
            st.markdown("#### 🎛️ Question Generation Settings")
            
            # Import and use sampling configuration
            from utils.sampling_config import render_sampling_config_ui, SamplingConfig, get_model_preset
            # Create default config for question generation (different from generation - better for questions)
            qa_default_config = SamplingConfig(
                temperature=0.9,  # Slightly higher for more diverse questions
                repetition_penalty=1.05,
                max_tokens=100  # Questions are shorter
            )
            
            # ✅ FIX: Pass use_expander=False to avoid nested expander error
            qa_sampling_config = render_sampling_config_ui(
                current_config=qa_default_config,
                key_prefix="qa_gen",
                use_expander=False  # Important: Avoid nested expanders
            )
            
            # ✅ NEW: Generation Controls
            st.markdown("#### 📝 Question Controls")
            
            num_q = st.number_input(
                "Number of questions to generate",
                min_value=1,
                max_value=100,
                value=10,
                step=1,
                key="num_q_generate"
            )

            if st.button("🔮 Generate Questions", key="generate_questions_btn", use_container_width=True):
                with st.spinner("Generating questions..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        qs = loop.run_until_complete(
                            st.session_state.dataset_manager.suggest_user_questions(
                                st.session_state.current_character,
                                num_questions=int(num_q),
                                existing_dataset=st.session_state.dataset_preview,
                                **qa_sampling_config.to_dict()  # Pass all sampling parameters
                            )
                        )
                        st.session_state.generated_questions = qs
                    finally:
                        loop.close()
                st.success(f"Generated {len(st.session_state.generated_questions)} questions!")

        # Show generated questions in the second tab
        with qa_tabs[1]:
            # Show generated questions and allow user to add to baseline
            gen_data = st.session_state.get('generated_questions')
            if gen_data:
                st.markdown("#### Review Generated Questions")
                selections = []
                for idx, item in enumerate(gen_data):
                    # Layout: checkbox | question | context toggle
                    cols = st.columns([0.08, 0.72, 0.2])
                    with cols[0]:
                        include = st.checkbox(f"q{idx+1}", value=True, key=f"include_q_{idx}", label_visibility="hidden")
                        if include:
                            selections.append(item['question'])
                    with cols[1]:
                        st.markdown(f"**{idx+1}. {item['question']}**")
                    with cols[2]:
                        toggle_key = f"show_ctx_{idx}"
                        if st.button("Context ↕", key=f"btn_{toggle_key}"):
                            st.session_state[toggle_key] = not st.session_state.get(toggle_key, False)
                    # Display context when toggled
                    if st.session_state.get(toggle_key, False) and item['context']:
                        st.markdown("**Context used:**")
                        for j, ctx in enumerate(item['context']):
                            st.markdown(f"*Q{j+1}:* {ctx['user']}")
                            st.markdown(f"*A{j+1}:* {ctx['assistant']}")
                        st.markdown("---")

                if selections and st.button("➕ Add Selected Questions", key="add_selected_qs_btn", use_container_width=True):
                    st.session_state.dataset_manager.default_user_prompts.extend(selections)
                    st.success(f"Added {len(selections)} questions to baseline list.")
            else:
                st.info("No questions generated yet. Go to the Setup & Generation tab to create questions.")

        # 📦 Dataset Import / Export
        st.markdown("### 📦 Import / Export Dataset")
        with st.expander("Manage dataset files (download or import)"):
            # Export (download button)
            if dataset_info['exists']:
                raw_json = st.session_state.dataset_manager.export_dataset(st.session_state.current_character)
                if raw_json is not None:
                    st.download_button(
                        label="⬇️ Download Dataset JSON",
                        data=raw_json,
                        file_name="character_dataset.json",
                        mime="application/json",
                        use_container_width=True
                    )
            else:
                st.info("No dataset available for export yet.")

            st.markdown("---")

            # Import
            import_file = st.file_uploader(
                "Upload a dataset JSON to import",
                type=["json"],
                key="import_dataset_uploader"
            )

            merge_mode = st.radio(
                "Merge strategy",
                ("replace", "append"),
                horizontal=True,
                help="Replace will overwrite any existing dataset, append will add new unique samples"
            )

            if import_file is not None:
                if st.button("📥 Import Dataset", key="import_dataset_btn", use_container_width=True):
                    success = st.session_state.dataset_manager.import_dataset_from_bytes(
                        st.session_state.current_character,
                        import_file.getvalue(),
                        merge_mode=merge_mode
                    )
                    if success:
                        st.success("Dataset imported successfully!")
                        # Reload into preview
                        st.session_state.dataset_preview = st.session_state.dataset_manager.load_dataset(
                            st.session_state.current_character
                        )
                        # ✅ FIX: Use controlled rerun
                        st.rerun()
                    else:
                        st.error("Failed to import dataset. Check file format.")
    
    with standard_tab:
        st.markdown("### 📝 Basic Dataset Generation")
        
        st.info("""
        🎯 **Basic Mode**:
        - Standard generation with minimal quality filtering
        - Fast generation for testing and prototyping
        - Good for: Quick experiments, proof of concepts
        """)
        
        # Basic generation settings
        st.markdown("#### 🎛️ Basic Generation Settings")
        
        # Simple configuration for basic mode
        with st.form("basic_generation"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                basic_num_samples = st.slider(
                    "Number of samples",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="Number of samples to generate for basic testing"
                )
                
                basic_temperature = st.slider(
                    "Temperature",
                    min_value=0.1,
                    max_value=1.5,
                    value=0.8,
                    step=0.1,
                    help="Controls randomness in generation"
                )
            
            with col_b:
                basic_max_tokens = st.slider(
                    "Max tokens per response",
                    min_value=50,
                    max_value=500,
                    value=200,
                    step=50,
                    help="Maximum length of each response"
                )
                
                basic_use_custom_system = st.checkbox(
                    "Apply custom system prompt", 
                    value=False, 
                    help="Replace temporal prompts with a custom prompt after generation"
                )
            
            if basic_use_custom_system:
                basic_system_prompt = st.text_area(
                    "System Prompt for Training",
                    placeholder="You are a helpful assistant...\n\nLeave empty for no system prompt.",
                    height=100,
                    help="Custom system prompt to use for training",
                    key="basic_system_prompt"
                )
            else:
                basic_system_prompt = None
                st.info("Dataset will keep temporal context system prompts")
            
            # Generate button
            basic_generate_button = st.form_submit_button(
                "🚀 Generate Basic Dataset", 
                use_container_width=True,
                type="primary"
            )
        
        if basic_generate_button:
            # ✅ FIX: Set generation state to prevent UI interference
            st.session_state._generating_dataset = True
        
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Run basic generation using the same pattern as enhanced mode
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    dataset = loop.run_until_complete(
                        st.session_state.dataset_manager.generate_dataset(
                            st.session_state.current_character,
                            num_samples=basic_num_samples,
                            max_tokens=basic_max_tokens,
                            temperature=basic_temperature,
                            top_p=0.9,
                            progress_callback=lambda p: progress_bar.progress(p),
                            append_to_existing=True,
                            custom_system_prompt=basic_system_prompt if basic_use_custom_system else None,
                            extra_quality=False,  # No extra quality for basic mode
                            quality_level=QualityLevel.BASIC  # Use basic quality level
                        )
                    )
                finally:
                    loop.close()
                
                st.session_state.dataset_preview = dataset
                # Update metadata if we generated with custom system prompt
                if basic_use_custom_system:
                    st.session_state.dataset_metadata = {
                        'generation_method': 'basic',
                        'system_prompt_config': {
                            'type': 'custom',
                            'prompt': basic_system_prompt
                        }
                    }
                else:
                    st.session_state.dataset_metadata = {
                        'generation_method': 'basic',
                        'system_prompt_config': {
                            'type': 'temporal',
                            'prompt': None
                        }
                    }
                progress_bar.progress(1.0)
                status_text.text("Basic dataset generation complete!")
                st.success(f"✅ Generated {len(dataset)} samples successfully!")
            
                # ✅ FIX: Clear generation state before rerun to prevent loops
                st.session_state._generating_dataset = False
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating basic dataset: {str(e)}")
                # ✅ FIX: Always clear generation state on error
                st.session_state._generating_dataset = False
    
    with enhanced_tab:
        st.markdown("### ⭐ Enhanced Dataset Generation")
        
        st.info("""
        🎯 **Enhanced Mode**:
        - vLLM-optimized batching for better performance
        - Real-time quality filtering during generation
        - Character-specific quality criteria
        - Good for: Production datasets, balanced quality/speed
        """)
        
        # Replace the basic parameter sliders with sampling configuration
        st.markdown("#### 🎛️ Generation Settings")
        
        # Import and use sampling configuration
        from utils.sampling_config import render_sampling_config_ui, SamplingConfig, get_model_preset

        # Create default config
        default_config = SamplingConfig(
            temperature=0.8,
            max_tokens=1000,
            repetition_penalty=1.1,
        )
        
        # Render sampling configuration UI
        sampling_config = render_sampling_config_ui(
            current_config=default_config,
            key_prefix="standard_gen"
        )
        
        # System prompt configuration
        st.markdown("#### 📝 System Prompt Configuration for Training")
        
        with st.expander("ℹ️ How System Prompts Work", expanded=False):
            st.markdown("""
            **During Generation**: The dataset uses diverse temporal prompts (past/present/future relationships) to generate varied, contextual responses.
            
            **For Training**: You can optionally replace all these temporal prompts with a single custom prompt. This gives you:
            - Temporal diversity during generation
            - Consistent system prompt during training
            - Perfect for scenario-specific or multi-character setups
            """)
        
        use_custom_system = st.checkbox("Apply custom system prompt to dataset", value=False, help="Replace temporal prompts with a custom prompt after generation", key="standard_use_custom_system")
        
        if use_custom_system:
            system_prompt = st.text_area(
                "System Prompt for Training",
                placeholder="You are a helpful assistant...\n\nLeave empty for no system prompt.",
                height=100,
                help="After generation with temporal prompts, this will replace all system prompts in the dataset for consistent training.",
                key="standard_system_prompt"
            )
        else:
            system_prompt = None
            st.info("Dataset will keep temporal context system prompts (varies per sample)")
        
        # Generation form
        with st.form("dataset_generation"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Allow generating much larger synthetic datasets (up to 20k samples)
                num_samples = st.slider(
                    "Total samples target",
                    min_value=20,
                    max_value=2000,
                    value=80,
                    step=20,
                    help="Desired total size of the synthetic dataset. Research shows 20-100 samples is optimal for character LoRAs, with 200-500 for more complex characters. Larger datasets risk overfitting."
                )
                
                # Extra quality checkbox
                extra_quality = st.checkbox(
                    "🌟 EXTRA QUALITY", 
                    value=False, 
                    help="Paraphrase all questions before generation for cleaner, more varied prompts. Takes longer but significantly improves dataset quality.",
                    key="standard_extra_quality"
                )
            
            with col_b:
                # Show configuration summary
                st.markdown("**Configuration Summary:**")
                st.write(f"• Temperature: {sampling_config.temperature}")
                st.write(f"• Max Tokens: {sampling_config.max_tokens}")
                st.write(f"• System Prompt: {'Custom' if use_custom_system else 'Temporal'}")
            if extra_quality:
                    st.write("• Extra Quality: ✅ Enabled")
            
            # Submit button
            generate_button = st.form_submit_button(
                "🚀 Generate Dataset", 
                use_container_width=True,
                type="primary"
            )
        
        if generate_button:
            # ✅ FIX: Set generation state to prevent UI interference
            st.session_state._generating_dataset = True
        
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress callback (called by DatasetManager)
            def update_progress(p: float):
                """Update status text for current chunk progress."""
                status_text.text(
                    f"Generating samples... {p*100:.1f}%"
                )
            
            try:
                # Run generation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                # ✅ FIX: Don't pass duplicate parameters - use sampling_config.to_dict() only
                    dataset = loop.run_until_complete(
                        st.session_state.dataset_manager.generate_dataset(
                            st.session_state.current_character,
                            num_samples=num_samples,
                            progress_callback=lambda p: progress_bar.progress(p),
                            append_to_existing=True,
                            custom_system_prompt=system_prompt if use_custom_system else None,
                        extra_quality=extra_quality,
                        **sampling_config.to_dict()  # Pass all sampling parameters
                        )
                    )
                finally:
                    loop.close()
                
                st.session_state.dataset_preview = dataset
                # Update metadata if we generated with custom system prompt
                if use_custom_system:
                    st.session_state.dataset_metadata = {
                        'system_prompt_config': {
                            'type': 'custom',
                            'prompt': system_prompt
                        }
                    }
                else:
                    st.session_state.dataset_metadata = {
                        'system_prompt_config': {
                            'type': 'temporal',
                            'prompt': None
                        }
                    }
                progress_bar.progress(1.0)
                status_text.text("Dataset generation complete!")
                st.success(f"✅ Generated {len(dataset)} samples successfully!")
            
                # ✅ FIX: Clear generation state before rerun to prevent loops
                st.session_state._generating_dataset = False
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating dataset: {str(e)}")
                # ✅ FIX: Always clear generation state on error
                st.session_state._generating_dataset = False
    
    with premium_tab:
        st.markdown("### 🌟 Premium Dataset Generation")
        
        st.info("""
        🎯 **Premium Mode**:
        - Multi-phase generation with progressive refinement
        - Specialized character consistency judging
        - Automatic quality improvement through iterative refinement
        - Highest quality output with intelligent sample curation
        - Good for: Production characters, best possible quality
        """)
        
        # Premium generation settings
        with st.form("premium_generation"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                premium_samples = st.slider(
                    "Target high-quality samples",
                    min_value=20,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Premium mode generates 3x this amount and curates the best samples"
                )
                
                enable_refinement = st.checkbox(
                    "🔄 Progressive Refinement", 
                    value=True,
                    help="Automatically improve lower-quality samples through iterative refinement"
                )
                
                quality_threshold = st.slider(
                    "Quality Threshold (0-1)",
                    min_value=0.7,
                    max_value=0.95,
                    value=0.8,
                    step=0.05,
                    help="Minimum quality score for sample acceptance"
                )
            
            with col_b:
                premium_extra_quality = st.checkbox(
                    "🌟 MAXIMUM QUALITY", 
                    value=True, 
                    help="Enable all quality enhancement features for absolute best results"
                )
                
                refinement_iterations = st.slider(
                    "Max Refinement Iterations",
                    min_value=1,
                    max_value=5,
                    value=2,
                    help="Maximum attempts to improve each sample"
                )
                
                # vLLM optimization settings
                vllm_optimization = st.checkbox(
                    "🚀 vLLM Optimization",
                    value=True,
                    help="Use advanced batching for better performance"
                )
            
            # Premium system prompt configuration
            st.markdown("#### 📝 System Prompt Configuration")
            use_custom_system_premium = st.checkbox("Apply custom system prompt to dataset", value=False, key="premium_custom_system")
            
            if use_custom_system_premium:
                system_prompt_premium = st.text_area(
                    "System Prompt for Training",
                    placeholder="You are a helpful assistant...\n\nLeave empty for no system prompt.",
                    height=100,
                    key="premium_system_prompt"
                )
            else:
                system_prompt_premium = None
                st.info("Dataset will use temporal context system prompts")
            
            # Estimated processing time
            estimated_total_samples = premium_samples * 3
            estimated_time = estimated_total_samples / 8 / 60  # Rough estimate
            
            st.info(f"""
            📊 **Premium Generation Plan**:
            - Generate: ~{estimated_total_samples} diverse samples
            - Curate: {premium_samples} highest quality samples
            - Estimated time: ~{estimated_time:.1f} minutes
            - Quality threshold: {quality_threshold:.0%}
            """)
            
            # Submit button
            premium_generate_button = st.form_submit_button(
                "🌟 Generate Premium Dataset", 
                use_container_width=True,
                type="primary"
            )
        
        if premium_generate_button:
            # ✅ FIX: Set generation state to prevent UI interference
            st.session_state._generating_dataset = True
        
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Enhanced progress callback for premium mode
            def update_premium_progress(p: float):
                phase = "Processing..." if p < 0.6 else "Judging..." if p < 0.8 else "Refining..."
                status_text.text(f"Premium generation - {phase} {p*100:.1f}%")
                
            try:
                # Run premium generation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:                    
                    # Create enhanced generation config
                    generation_config = GenerationConfig(
                        use_vllm_optimization=vllm_optimization,
                        quality_threshold=quality_threshold,
                        enable_progressive_refinement=enable_refinement,
                        max_refinement_iterations=refinement_iterations,
                        judge_batch_size=20,
                        diversity_weight=0.3,
                        enable_real_time_filtering=True
                    )
                    
                    # Update dataset manager with new config
                    st.session_state.dataset_manager.generation_config = generation_config
                    
                    dataset = loop.run_until_complete(
                        st.session_state.dataset_manager.generate_dataset(
                            st.session_state.current_character,
                            num_samples=premium_samples,
                            progress_callback=lambda p: progress_bar.progress(p),
                            append_to_existing=True,
                            custom_system_prompt=system_prompt_premium if use_custom_system_premium else None,
                            extra_quality=premium_extra_quality,
                            quality_level=QualityLevel.PREMIUM,  # Use premium quality
                            temperature=0.8,
                            top_p=0.9,
                            max_tokens=400
                        )
                    )
                finally:
                    loop.close()
                
                st.session_state.dataset_preview = dataset
                # Update metadata
                if use_custom_system_premium:
                    st.session_state.dataset_metadata = {
                        'generation_method': 'premium',
                        'quality_threshold': quality_threshold,
                        'system_prompt_config': {
                            'type': 'custom',
                            'prompt': system_prompt_premium
                        }
                    }
                else:
                    st.session_state.dataset_metadata = {
                        'generation_method': 'premium',
                        'quality_threshold': quality_threshold,
                        'system_prompt_config': {
                            'type': 'temporal',
                            'prompt': None
                        }
                    }
                    
                progress_bar.progress(1.0)
                status_text.text("Premium dataset generation complete!")
                st.success(f"🌟 Generated {len(dataset)} premium quality samples!")
                
                # Show generation statistics if available
                if hasattr(st.session_state.dataset_manager, 'generation_stats'):
                    stats = st.session_state.dataset_manager.generation_stats
                    st.info(f"""
                    📊 **Premium Generation Statistics**:
                    - Average quality score: {stats.get('avg_quality_score', 0):.2f}
                    - Samples refined: {stats.get('refined_samples', 0)}
                    - Filter efficiency: {(1 - stats.get('filtered_out', 0) / max(stats.get('total_generated', 1), 1)) * 100:.1f}%
                    """)
            
                # ✅ FIX: Clear generation state before rerun to prevent loops
                st.session_state._generating_dataset = False
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating premium dataset: {str(e)}")
                logger.exception("Premium generation error:")
                # ✅ FIX: Always clear generation state on error
                st.session_state._generating_dataset = False
    
    with quality_tab:
        st.markdown("### 🎯 Factual Q&A Dataset Generation")
        
        st.info("""
        **How it works:** This method creates a high-quality, targeted dataset to teach a model the core facts about your character.
        1.  **Fact Extraction:** An LLM analyzes your character card to extract a simple list of key facts.
        2.  **Q&A Variation:** For each fact, an LLM generates multiple, varied question-and-answer pairs.
        3.  **Reinforcement:** This process reinforces the most important information, making it easier for the LoRA to learn.
        
        This approach is ideal for creating small, potent datasets that prevent model hallucination and ensure character consistency.
        """)

        with st.form("factual_qa_generation"):
            st.markdown("#### ⚙️ Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                num_facts = st.slider(
                    "Number of Core Facts to Extract",
                    min_value=5,
                    max_value=50,
                    value=15,
                    step=1,
                    help="The number of key facts to distill from the character card. This determines the breadth of the dataset."
                )
            with col2:
                variations_per_fact = st.slider(
                    "Q&A Variations per Fact",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                    help="How many different question-answer pairs to create for each fact. More variations reinforce learning."
                )

            total_samples = num_facts * variations_per_fact
            st.success(f"**Estimated dataset size:** ~{total_samples} samples.")

            start_button = st.form_submit_button("🎯 Generate Factual Dataset", use_container_width=True, type="primary")

        if start_button:
            st.session_state._generating_dataset = True
            
            stage_text = st.empty()
            progress_bar = st.progress(0)
            
            def update_progress(p):
                progress_bar.progress(p)

            def update_stage(stage_info):
                stage_text.info(f"**Status:** {stage_info['message']}")

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                dataset = loop.run_until_complete(
                    st.session_state.dataset_manager.generate_factual_qa_dataset(
                        character=st.session_state.current_character,
                        num_facts_to_use=num_facts,
                        variations_per_fact=variations_per_fact,
                        progress_callback=update_progress,
                        stage_callback=update_stage
                    )
                )
                
                st.session_state.dataset_preview = dataset
                st.session_state.dataset_metadata = {
                    'generation_method': 'factual_qa',
                    'system_prompt_config': {'type': 'none'} # Factual QA datasets don't use system prompts
                }

                progress_bar.progress(1.0)
                stage_text.success("✅ Factual Q&A dataset generated successfully!")
                st.success(f"Generated {len(dataset)} high-quality factual samples.")

                st.session_state._generating_dataset = False
                st.rerun()

            except Exception as e:
                st.error(f"❌ Error during factual Q&A generation: {str(e)}")
                logger.error(f"Factual Q&A generation error: {e}", exc_info=True)
                st.session_state._generating_dataset = False
            finally:
                loop.close()

    # Dataset preview (shown in all tabs)
    if st.session_state.dataset_preview:
        st.markdown("---")
        st.markdown("### 📋 Sample Preview")
        
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

# Training configuration page
def page_training_config():
    """Enhanced training configuration page with advanced features"""
    st.markdown('<h2 class="gradient-text">⚙️ Training Configuration</h2>', unsafe_allow_html=True)
    
    # Check if a profile needs to be applied
    if 'profile_to_apply' in st.session_state and st.session_state.profile_to_apply:
        profile = st.session_state.profile_to_apply
        
        # Store profile values in a dedicated session state key
        st.session_state.training_form_defaults = {
            **profile.get("advanced_training_config", {}),
            **profile.get("hyperparameters", {}),
            "base_model": profile.get("base_model")
        }
        
        # Clear the flag
        del st.session_state['profile_to_apply']
        st.toast("✅ Profile applied to configuration below!", icon="✨")

    # Get defaults from session state, or set to empty dict if not present
    defaults = st.session_state.get('training_form_defaults', {})
    if 'training_form_defaults' in st.session_state:
        del st.session_state['training_form_defaults'] # One-time use

    # If training is already running or paused, encourage user to switch to Dashboard
    if st.session_state.get('training_status') in ['training', 'paused']:
        st.info("🚧 Training is in progress. Please use the Training Dashboard to monitor or control the run.")
        return
    
    if not st.session_state.dataset_preview:
        st.warning("⚠️ Please generate a dataset first.")
        return
    
    dataset_size = len(st.session_state.dataset_preview)
    
    # Advanced Features Configuration
    with st.expander("🚀 Advanced Training Features", expanded=False):
        st.markdown("### Core Improvements")
        
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            enable_validation = st.checkbox(
                "Enable Validation Split & Early Stopping",
                value=defaults.get("enable_validation", True),
                help="Split dataset for validation and enable early stopping to prevent overfitting"
            )
            
            adaptive_lora = st.checkbox(
                "Adaptive LoRA Parameters",
                value=defaults.get("adaptive_lora", False),
                help="Automatically adjust LoRA rank and alpha based on character complexity"
            )
            
            enhanced_filtering = st.checkbox(
                "Enhanced Quality Filtering",
                value=defaults.get("enhanced_quality_filtering", False),
                help="Apply character-specific quality filters to training data"
            )
        
        with col_adv2:
            enable_tensorboard = st.checkbox(
                "Enable TensorBoard Monitoring",
                value=defaults.get("enable_tensorboard", False),
                help="Enable detailed TensorBoard logging for training analysis"
            )
            
            enable_wandb = st.checkbox(
                "Weights & Biases Integration",
                value=defaults.get("enable_wandb", False),
                help="Log training to Wandb for advanced experiment tracking"
            )
        
        st.markdown("### Logging Configuration")
        logging_freq = st.slider(
            "Logging Frequency (steps)",
            min_value=1,
            max_value=50,
            value=defaults.get("configurable_logging_freq", 10),
            help="How often to log training metrics"
        )
        
        early_stopping_patience = st.slider(
            "Early Stopping Patience",
            min_value=1,
            max_value=10,
            value=defaults.get("early_stopping_patience", 3),
            help="Number of evaluation steps without improvement before stopping"
        ) if enable_validation else 3
        
        # Force GPU option
        force_gpu = st.checkbox(
            "Force GPU Usage",
            value=defaults.get("force_gpu", False),
            help="Override device selection to use GPU (if available)"
        )
        
        # Store advanced config in session state
        st.session_state.advanced_training_config = {
            'enable_validation': enable_validation,
            'early_stopping_patience': early_stopping_patience,
            'adaptive_lora': adaptive_lora,
            'enhanced_quality_filtering': enhanced_filtering,
            'enable_tensorboard': enable_tensorboard,
            'enable_wandb': enable_wandb,
            'configurable_logging_freq': logging_freq,
            'force_gpu': force_gpu
        }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Training Configuration")
        
        # Base model selection
        with st.expander("🤖 Base Model Selection", expanded=True):
            st.markdown("**Select the base model for LoRA training**")
            
            # Popular small models for LoRA
            base_model_options = [
                "HuggingFaceTB/SmolLM2-135M-Instruct",  # Default
                "HuggingFaceTB/SmolLM2-360M-Instruct",
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "microsoft/Phi-3.5-mini-instruct",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "Custom (enter HF ID below)"
            ]
            
            # Get the index of the default base model
            try:
                default_model_index = base_model_options.index(defaults.get("base_model"))
            except ValueError:
                # If the model from profile is not in the standard list, select "Custom"
                default_model_index = base_model_options.index("Custom (enter HF ID below)")

            base_model_choice = st.selectbox(
                "Select base model",
                base_model_options,
                index=default_model_index,
                help="Smaller models train faster and work better for character LoRAs"
            )
            
            if base_model_choice == "Custom (enter HF ID below)":
                custom_base_model = st.text_input(
                    "HuggingFace Model ID",
                    value=defaults.get("base_model", ""), # Pre-fill custom model
                    placeholder="e.g., teknium/OpenHermes-2.5-Mistral-7B",
                    help="Enter any HuggingFace model ID compatible with PEFT/LoRA"
                )
                selected_base_model = custom_base_model if custom_base_model else base_model_options[0]
            else:
                selected_base_model = base_model_choice
            
            # Update the training manager's base model
            if st.session_state.training_manager.base_model != selected_base_model:
                st.session_state.training_manager.set_base_model(selected_base_model)
                st.session_state.inference_manager.set_base_model(selected_base_model)
                
            # Model size info
            model_size_info = {
                "HuggingFaceTB/SmolLM2-135M-Instruct": "135M params - Very fast, good for testing",
                "HuggingFaceTB/SmolLM2-360M-Instruct": "360M params - Fast, better quality",
                "HuggingFaceTB/SmolLM2-1.7B-Instruct": "1.7B params - Balanced speed/quality",
                "mistralai/Mistral-7B-Instruct-v0.2": "7B params - High quality, slower",
                "meta-llama/Llama-3.2-1B-Instruct": "1B params - Good balance",
                "meta-llama/Llama-3.2-3B-Instruct": "3B params - Better quality",
                "microsoft/Phi-3.5-mini-instruct": "3.8B params - Efficient & capable",
                "Qwen/Qwen2.5-0.5B-Instruct": "0.5B params - Very fast",
                "Qwen/Qwen2.5-1.5B-Instruct": "1.5B params - Good balance",
                "Qwen/Qwen2.5-3B-Instruct": "3B params - Better quality"
            }
            
            if selected_base_model in model_size_info:
                st.info(f"ℹ️ {model_size_info[selected_base_model]}")
            
            st.success(f"✅ Base model: {selected_base_model}")
        
        st.markdown("### Hyperparameter Configuration")
        
        with st.form("training_config"):
            # Basic settings
            st.markdown("#### Basic Settings")
            
            # Dynamically pick a sensible default epoch count based on dataset size
            if dataset_size >= 200:
                optimal_epochs = 3  # Reduced for larger datasets
            else:
                optimal_epochs = min(6, max(3, 10000 // dataset_size))  # 3-6 epochs for smaller datasets
            
            # Epochs with explanation
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                st.markdown("**Epochs**")
            with c2:
                with st.popover("ℹ️", help="Explain Epochs"):
                    st.markdown("""
                    An **epoch** is one full pass through the entire training dataset.
                    
                    - **Too few epochs:** The model might not learn enough about the character (underfitting).
                    - **Too many epochs:** The model might memorize the training data and lose its ability to be creative (overfitting).
                    
                    **Recommendation:** 5-6 epochs for small datasets (<100 samples), and 3-4 for larger ones is a good starting point.
                    """)
            epochs = st.slider("Epochs", 1, 1000, defaults.get("epochs", optimal_epochs), label_visibility="collapsed", help="How many times the model sees the entire dataset.")

            # Learning Rate with explanation
            lr_options = [1e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1e-4, 2e-4, 3e-4, 5e-4]
            default_lr = defaults.get("learning_rate", 2e-4)
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                st.markdown("**Learning Rate**")
            with c2:
                with st.popover("ℹ️", help="Explain Learning Rate"):
                    st.markdown("""
                    The **Learning Rate** controls how much the model's parameters are adjusted during each training step.
                    
                    - **Too high:** The model might learn too fast and become unstable, with loss jumping around wildly.
                    - **Too low:** Training will be very slow, and the model might get stuck.
                    
                    **Recommendation:** `2e-4` is a safe and effective starting point for most characters.
                    """)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=lr_options,
                value=default_lr,
                format_func=lambda x: f"{x:.0e}",
                help="5e-5 to 5e-4 recommended for character LoRA training",
                label_visibility="collapsed"
            )

            # Batch Size with explanation
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                st.markdown("**Batch Size**")
            with c2:
                with st.popover("ℹ️", help="Explain Batch Size"):
                    st.markdown("""
                    The **Batch Size** is the number of training samples processed before the model's internal parameters are updated.
                    - It's limited by your GPU memory (VRAM).
                    - A larger batch size can lead to more stable training, but uses more memory.
                    - If you run out of memory, lower this value. You can compensate for a small batch size by increasing **Gradient Accumulation Steps**.
                    
                    **Recommendation:** Start with 2 or 4 and adjust based on your hardware.
                    """)
            batch_size = st.selectbox("Batch Size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], 
                                      index=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].index(defaults.get("batch_size", 2)), 
                                      label_visibility="collapsed")
            
            gradient_accumulation = st.selectbox("Gradient Accumulation Steps", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], 
                                                 index=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].index(defaults.get("gradient_accumulation_steps", 2)))
            warmup_steps = st.slider("Warmup Steps", 0, 100, defaults.get("warmup_steps", 10), help="10-20 steps usually sufficient")
            max_grad_norm = st.slider("Max Gradient Norm", 0.5, 8.0, defaults.get("max_grad_norm", 1.0), step=0.1, help="1.0 is standard")
            
            # Sample selection for dataset
            st.markdown("**Dataset Sampling**")
            if dataset_size > 0:
                use_all_samples = st.checkbox("Use All Samples", value=not ("max_samples" in defaults and defaults["max_samples"] != dataset_size), help="Use the entire dataset for training")
                if not use_all_samples:
                    max_samples = st.slider(
                        "Number of Samples", 
                        min_value=1, 
                        max_value=dataset_size, 
                        value=defaults.get("max_samples", min(dataset_size, 100)),
                        help=f"Select subset from {dataset_size} total samples (randomized selection)"
                    )
                else:
                    max_samples = dataset_size
            else:
                max_samples = dataset_size
            
            # LoRA settings optimized for character training
            st.markdown("#### PEFT Configuration (Character-Optimized)")
            
            finetune_method = st.radio(
                "Fine-tuning Method",
                ("LoRA", "DoRA"),
                horizontal=True,
                index=["lora", "dora"].index(defaults.get("finetune_method", "lora")),
                help="Choose between LoRA and DoRA for fine-tuning. DoRA can offer more precise training."
            ).lower()
            
            default_r = defaults.get("lora_r", 16)  # Optimal for character LoRA per research
            # LoRA Rank with explanation
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                st.markdown("**Rank (r)**")
            with c2:
                with st.popover("ℹ️", help="Explain Rank (r)"):
                    st.markdown("""
                    The **Rank (r)** determines the number of trainable parameters in the PEFT adapter. It controls the 'capacity' of the adapter.
                    - **Higher Rank:** More parameters, allowing the model to learn more complex details. This also increases training time and VRAM usage.
                    - **Lower Rank:** Fewer parameters, faster training, less VRAM.
                    
                    **Recommendation:** `8` or `16` is highly effective for most characters. Use `32` for very complex characters with large datasets.
                    """)
            lora_r = st.slider("Rank (r)", 4, 256, default_r, step=4,
                               label_visibility="collapsed",
                               help="8-16 optimal for character LoRAs. Higher rank = more capacity but slower.")

            # LoRA Alpha with explanation
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                st.markdown("**Alpha**")
            with c2:
                with st.popover("ℹ️", help="Explain Alpha"):
                    st.markdown("""
                    **Alpha** is a scaling factor for the PEFT adjustments. Think of it as controlling the 'intensity' of the training.
                    - By setting **Alpha equal to Rank (α = r)**, you are using a standard configuration that works very well for character training. This helps balance the learning process.
                    - Deviating from this (e.g., alpha = 2 * rank) is an advanced technique and not typically recommended for characters.
                    
                    **Recommendation:** Keep this value the same as your Rank.
                    """)
            lora_alpha = st.slider("Alpha", 8, 1024, defaults.get("lora_alpha", default_r), step=8,
                                   label_visibility="collapsed",
                                   help="Set equal to rank (α = r) for character training")
            
            lora_dropout = st.slider("Dropout", 0.0, 0.2, defaults.get("lora_dropout", 0.1), step=0.01, 
                                     help="0.05-0.1 for regularization")
            target_modules = st.multiselect(
                "Target Modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                default=defaults.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),  # Focus on attention layers
                help="Attention layers (q,k,v,o) are most important for character behavior"
            )
            
            # --------------------------------------------------------------
            # Resume-from-checkpoint selection
            # --------------------------------------------------------------
            available_ckpts = st.session_state.training_manager.get_available_checkpoints(
                st.session_state.current_character.get("name", "unknown")
            )

            if available_ckpts:
                resume_ckpt_option = st.selectbox(
                    "Resume from checkpoint (optional)",
                    ["None"] + available_ckpts
                )
                resume_ckpt = None if resume_ckpt_option == "None" else resume_ckpt_option
            else:
                resume_ckpt = None
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                # Check if dataset has system prompts
                dataset_has_system = False
                if st.session_state.dataset_preview and len(st.session_state.dataset_preview) > 0:
                    first_sample = st.session_state.dataset_preview[0]
                    if 'messages' in first_sample and len(first_sample['messages']) > 0:
                        dataset_has_system = first_sample['messages'][0].get('role') == 'system'
                
                include_system_prompts = st.checkbox(
                    "Include System Prompts in Training",
                    value=defaults.get("include_system_prompts", False),
                    help="If checked, system prompts will be included in training data. Usually better to let LoRA internalize character behavior without system prompts.",
                    disabled=not dataset_has_system
                )
                
                if dataset_has_system and not include_system_prompts:
                    st.info("💡 System prompts will be removed during training (recommended for character LoRAs)")
                elif dataset_has_system and include_system_prompts:
                    st.warning("⚠️ Including system prompts in training - make sure this is intentional")
                elif not dataset_has_system:
                    st.info("ℹ️ Dataset has no system prompts")
                
                fp16 = st.checkbox("Enable FP16", value=defaults.get("fp16", True), help="Enables mixed precision training for better performance")
                save_steps = st.slider("Save Every N Steps", 1, 200, defaults.get("save_steps", 50), step=1)
                logging_steps = st.slider("Log Every N Steps", 1, 100, defaults.get("logging_steps", 5), step=1)
                eval_steps = st.slider("Evaluation Steps", 1, 100, defaults.get("eval_steps", 10), step=1)
                max_steps_override = st.number_input(
                    "Override Total Training Steps (0 = auto)",
                    min_value=0,
                    max_value=50000,
                    value=defaults.get("max_steps_override", 0),
                    step=100,
                    help="Manually set the total number of optimisation steps if you need finer control. Leave at 0 to use the computed value."
                )
            
            start_training = st.form_submit_button("🚀 Start Training", use_container_width=True)
    
    with col2:
        st.markdown("### Training Recommendations")
        
        # Profile I/O: Create a dedicated expander for this
        # with st.expander("💾 Save, Load, and Apply Training Profiles", expanded=True):
        profile_cols = st.columns(2)
        with profile_cols[0]:
            if st.button("💾 Save Current Profile", use_container_width=True, help="Save the current settings as a profile"):
                try:
                    # Use a more robust way to get the character name
                    char_name = st.session_state.current_character.get("name", "untitled")
                    
                    config_to_save = {
                        "character_name": char_name,
                        "base_model": selected_base_model,
                        "dataset_file": st.session_state.dataset_metadata.get('path'),
                        "advanced_training_config": st.session_state.get('advanced_training_config', {}),
                        "hyperparameters": {
                            'epochs': epochs,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'gradient_accumulation_steps': gradient_accumulation,
                            'warmup_steps': warmup_steps,
                            'max_grad_norm': max_grad_norm,
                            'max_samples': max_samples,
                            'finetune_method': finetune_method,
                            'lora_r': lora_r,
                            'lora_alpha': lora_alpha,
                            'lora_dropout': lora_dropout,
                            'target_modules': target_modules,
                            'include_system_prompts': include_system_prompts,
                            'fp16': fp16,
                            'save_steps': save_steps,
                            'logging_steps': logging_steps,
                            'eval_steps': eval_steps,
                            'max_steps_override': int(max_steps_override) if max_steps_override else 0,
                        }
                    }
                    
                    # Create profiles directory if it doesn't exist
                    profiles_dir = Path("profiles")
                    profiles_dir.mkdir(exist_ok=True)
                    
                    # Save the profile
                    save_path = profiles_dir / f"{char_name}_profile.json"
                    with open(save_path, 'w') as f:
                        json.dump(config_to_save, f, indent=4)
                    
                    st.toast(f"✅ Profile saved: {save_path.name}", icon="💾")
                    
                except Exception as e:
                    st.error(f"❌ Error saving profile: {e}")

        with profile_cols[1]:
            uploaded_profile = st.file_uploader(
                "Load Profile", 
                type=['json'], 
                label_visibility="collapsed",
                help="Upload a saved training profile"
            )

        if uploaded_profile:
            try:
                profile_data = json.load(uploaded_profile)
                
                # Store loaded profile in session state to be applied
                st.session_state.loaded_profile = profile_data
                
                st.success(f"📂 Profile '{uploaded_profile.name}' loaded!")
                st.info("Click 'Apply Profile' to update the configuration below.")

            except Exception as e:
                st.error(f"❌ Error loading profile: {e}")
        
        # Button to apply the loaded profile
        if 'loaded_profile' in st.session_state and st.session_state.loaded_profile:
            if st.button("✨ Apply Profile", use_container_width=True):
                # In a real app, you would now update all the widgets.
                # Streamlit makes this tricky without re-running the script.
                # The "correct" way is to store defaults in session state
                # and use them to set widget values.
                # We will add this logic in the next step.
                st.session_state.profile_to_apply = st.session_state.loaded_profile
                del st.session_state['loaded_profile'] # Clear after flagging
                st.rerun() # Rerun to apply the settings

        # Calculate training recommendations based on selected samples
        effective_dataset_size = max_samples if 'max_samples' in locals() else dataset_size
        total_steps = (effective_dataset_size * epochs) // (batch_size * gradient_accumulation)
        effective_batch_size = batch_size * gradient_accumulation
        
        # More nuanced overfitting risk calculation
        if effective_dataset_size < 50:
            if total_steps > 300:
                overfitting_risk = "Very High"
            elif total_steps > 200:
                overfitting_risk = "High"
            else:
                overfitting_risk = "Medium"
        elif effective_dataset_size < 100:
            if total_steps > 500:
                overfitting_risk = "High"
            elif total_steps > 300:
                overfitting_risk = "Medium"
            else:
                overfitting_risk = "Low"
        else:  # effective_dataset_size >= 100
            if total_steps > 1000:
                overfitting_risk = "Medium"
            else:
                overfitting_risk = "Low"
        
        # Display recommendations
        # Show both total and selected dataset info
        dataset_info = f"{effective_dataset_size} samples"
        if effective_dataset_size != dataset_size:
            dataset_info += f" (from {dataset_size} total)"
        
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0;">📊 Training Analysis</h4>
                <p><strong>Training Dataset:</strong> {dataset_info}</p>
                <p><strong>Total Steps:</strong> {total_steps}</p>
                <p><strong>Effective Batch Size:</strong> {effective_batch_size}</p>
                <p><strong>Overfitting Risk:</strong> <span style="color: {'#dc2626' if overfitting_risk == 'Very High' else '#ef4444' if overfitting_risk == 'High' else '#f59e0b' if overfitting_risk == 'Medium' else '#10b981'}">{overfitting_risk}</span></p>
                <p><strong>Est. Time:</strong> ~{max(1, total_steps * 2 // 60)} minutes</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Warnings based on configuration
        if overfitting_risk in ["High", "Very High"]:
            st.warning(f"⚠️ {overfitting_risk} overfitting risk! Consider: reducing epochs to {max(1, epochs-2)}, increasing dataset size, or lowering learning rate.")
        
        if effective_dataset_size > 300:
            st.info("💡 Large dataset detected. Consider using rank 32 for more model capacity.")
        
        if learning_rate >= 5e-4:
            st.warning("⚠️ High learning rate may cause training instability. Consider 2e-4 for safer training.")
            
        if epochs > 8:
            st.warning("⚠️ High epoch count increases overfitting risk. 5-6 epochs is usually sufficient.")
        
        # Tips
        st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
                <h4 style="color: #10b981; margin-top: 0;">💡 Character LoRA Best Practices</h4>
                <ul style="color: #cbd5e1; font-size: 0.9rem;">
                    <li><strong>Dataset:</strong> 50-100 samples optimal, 200-300 max</li>
                    <li><strong>Learning Rate:</strong> Start with 2e-4, use 1e-4 if unstable</li>
                    <li><strong>LoRA Rank:</strong> 8-16 for most characters, 32 for complex ones</li>
                    <li><strong>LoRA Alpha:</strong> Set equal to rank (α = r)</li>
                    <li><strong>Epochs:</strong> 5-6 for small datasets, 3-4 for larger ones</li>
                    <li><strong>Monitor:</strong> Stop if loss plateaus or increases</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    if start_training:
        # Enhanced training config with advanced features
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
            'eval_steps': eval_steps,
            'max_steps_override': int(max_steps_override) if max_steps_override else 0,
            'resume_from_checkpoint': resume_ckpt,
            'include_system_prompts': include_system_prompts,
            'max_samples': max_samples,
            'finetune_method': finetune_method,
            # Enhanced scheduling options
            'lr_scheduler_type': 'cosine',
            'warmup_ratio': 0.05
        }
        
        try:
            # Configure advanced features
            advanced_config = st.session_state.get('advanced_training_config', {})
            st.session_state.training_manager.configure_advanced_features(advanced_config)
            
            # Initialize with force_gpu if specified
            if advanced_config.get('force_gpu', False):
                from utils.training import TrainingManager
                st.session_state.training_manager = TrainingManager(
                    base_model=st.session_state.training_manager.base_model,
                    force_gpu=True
                )
                st.session_state.training_manager.configure_advanced_features(advanced_config)
            
            # Use different key to avoid widget conflict
            st.session_state.active_training_config = config
            
            # Show configuration summary
            st.success("✅ Training configuration complete!")
            
            # Display what features are enabled
            enabled_features = []
            if advanced_config.get('enable_validation', True):
                enabled_features.append("🔍 Validation & Early Stopping")
            if advanced_config.get('adaptive_lora', False):
                enabled_features.append("🎯 Adaptive LoRA Parameters")
            if advanced_config.get('enhanced_quality_filtering', False):
                enabled_features.append("✨ Enhanced Quality Filtering")
            if advanced_config.get('enable_tensorboard', False):
                enabled_features.append("📊 TensorBoard Monitoring")
            if advanced_config.get('enable_wandb', False):
                enabled_features.append("🌐 Wandb Integration")
            
            if enabled_features:
                st.info("🚀 **Enhanced Features Active:**\n\n" + "\n".join([f"• {feature}" for feature in enabled_features]))
            
            # Start enhanced training
            st.session_state.training_manager.start_training(
                st.session_state.current_character,
                st.session_state.dataset_preview,
                config
            )
            
            # Update status
            st.session_state.training_status = 'training'
            
            st.success("🚀 Enhanced training started! Switch to the Training Dashboard to monitor progress.")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to start training: {str(e)}")
            st.session_state.training_status = 'error'
            import traceback
            st.error(f"Debug info: {traceback.format_exc()}")

# Real-time training dashboard
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

    # Enhanced real-time metrics
    metrics_placeholder = st.empty()
    health_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Update training status from manager
    current_status = st.session_state.training_manager.get_training_status()
    if current_status != st.session_state.training_status:
        st.session_state.training_status = current_status
    
    # Get enhanced training metrics
    metrics = st.session_state.training_manager.get_metrics()
    
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
                if 'eval_loss' in metrics and hasattr(st.session_state.training_manager, 'eval_loss_history'):
                    eval_history = getattr(st.session_state.training_manager, 'eval_loss_history', [])
                    if eval_history:
                        eval_steps = list(range(0, len(eval_history) * (len(steps) // len(eval_history)), len(steps) // len(eval_history)))[:len(eval_history)]
                        fig.add_scatter(
                            x=eval_steps, y=eval_history,
                            mode='lines+markers',
                            name='Validation Loss',
                            line=dict(color='orange', dash='dash')
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
                        gauge_color = "green" if consistency_score > 0.7 else "orange" if consistency_score > 0.4 else "red"
                        st.metric(
                            "Current Consistency Score",
                            f"{consistency_score:.2f}",
                            help="1.0 = Perfect character consistency, 0.0 = Poor consistency"
                        )
                        
                        # Progress bar visualization
                        progress_bar_html = f"""
                        <div style="background-color: #f0f0f0; border-radius: 10px; padding: 3px;">
                            <div style="background-color: {gauge_color}; width: {consistency_score*100:.0f}%; 
                                        height: 20px; border-radius: 7px; text-align: center; color: white; 
                                        font-weight: bold; line-height: 20px;">
                                {consistency_score:.2f}
                            </div>
                        </div>
                        """
                        st.markdown(progress_bar_html, unsafe_allow_html=True)
                    
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
    
    # Auto-refresh for real-time updates
    if st.session_state.training_status == 'training':
        time.sleep(2)
        st.rerun()

# Model testing and inference page
def page_model_testing():
    """Model testing and inference page"""
    st.markdown('<h2 class="gradient-text">🧪 Model Testing</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Test Your Trained Model")
        
        st.info("🧪 **Pure LoRA Testing**: No character context is injected. Testing how well the LoRA learned character behavior during training.")
        
        
        # Model selection
        available_models = st.session_state.inference_manager.get_available_models()
        
        if not available_models:
            st.info("ℹ️ No trained models available. Complete training first.")
            return
        
        selected_model = st.selectbox("Select Trained Model", available_models, key="test_model_select")
        
        # Check if we have dataset metadata with system prompt info
        dataset_metadata = st.session_state.get('dataset_metadata', {})
        system_prompt_config = dataset_metadata.get('system_prompt_config', {})
        
        # Set default option based on dataset
        if system_prompt_config.get('type') == 'custom':
            default_option = "Dataset System Prompt"
            options = ["Dataset System Prompt", "Default (Tokenizer's built-in)", "Empty (No system prompt)", "Roleplay Director", "Custom"]
        elif system_prompt_config.get('type') == 'none':
            default_option = "Empty (No system prompt)"
            options = ["Empty (No system prompt)", "Default (Tokenizer's built-in)", "Roleplay Director", "Custom"]
        else:
            default_option = "Default (Tokenizer's built-in)"
            options = ["Default (Tokenizer's built-in)", "Empty (No system prompt)", "Roleplay Director", "Custom"]

        system_prompt_option = st.radio(
            "Choose system prompt strategy:",
            options,
            help="Test how the LoRA responds to different system prompts"
        )
        
        # Show debug info about what's being tested
        with st.expander("🔧 Test Configuration"):
            st.write(f"**Selected Model:** {selected_model}")
            st.write(f"**System Prompt Strategy:** {system_prompt_option}")
            st.write(f"**Character Context Injection:** No (Pure LoRA Test)")
            if st.session_state.current_character and not selected_model.startswith("Base:"):
                char = st.session_state.current_character
                st.write(f"**Testing Character:** {char.get('name', 'Unknown')}")
                st.info("💡 Testing how well the LoRA learned the character behavior")
            elif selected_model.startswith("Base:"):
                st.write("**Mode:** Base model testing")
            else:
                st.warning("⚠️ No character uploaded for LoRA comparison")
        
        # System prompt selection
        system_prompt = None
        if system_prompt_option == "Dataset System Prompt":
            system_prompt = system_prompt_config.get('prompt', '')
            if system_prompt:
                st.info(f"📊 Using system prompt from dataset generation")
            else:
                st.info(f"📊 Dataset was generated with no system prompt")
        elif system_prompt_option == "Empty (No system prompt)":
            system_prompt = ""
            st.info("🧪 Testing pure LoRA behavior without any system guidance")
        elif system_prompt_option == "Roleplay Director":
            system_prompt = "You are a scene director playing the role of a character in a never ending chat"
            st.info("🎭 Testing LoRA with roleplay-oriented system prompt")
        elif system_prompt_option == "Custom":
            system_prompt = st.text_area(
                "Enter custom system prompt:",
                placeholder="You are...",
                height=80
            )
            st.info("✏️ Testing LoRA with your custom system prompt")
        else:
            st.info("🤖 Using tokenizer's default system prompt (SmolLM assistant)")
        
        if system_prompt is not None and system_prompt_option not in ["Default (Tokenizer's built-in)", "Custom"]:
            st.code(f"System: {system_prompt if system_prompt else '[No system prompt]'}", language="text")
        elif system_prompt_option == "Custom" and system_prompt:
            st.code(f"System: {system_prompt}", language="text")

        # Test prompt
        test_prompt = st.text_area(
            "Enter your test prompt:",
            value=st.session_state.get('quick_test_prompt', ''),
            placeholder="Tell me about yourself...",
            height=100,
            key="main_test_prompt"
        )
        
        # Generation settings
        with st.expander("⚙️ Generation Settings"):
            # Import and use sampling configuration
            from utils.sampling_config import render_sampling_config_ui, SamplingConfig, get_model_preset
            
            # Try to get current model name for testing
            test_model = selected_model
            if test_model.startswith("Base:"):
                # For base models, try to get from inference manager
                test_model = getattr(st.session_state.inference_manager, 'base_model', None)
            
            # Check if we have a model-specific preset
            model_preset = get_model_preset(test_model) if test_model else None
            if model_preset:
                st.info(f"🎯 **{model_preset['name']}** preset available for this model")
            
            # Create testing-specific default config
            default_test_config = SamplingConfig(
                temperature=0.8,
                top_p=0.9,
                max_tokens=150,
                repetition_penalty=1.1,
            )
            
            # Render compact sampling configuration
            test_sampling_config = render_sampling_config_ui(
                current_config=default_test_config,
                model_name=test_model,
                key_prefix="model_test"
            )
        
        if st.button("🚀 Generate Response", use_container_width=True):
            if test_prompt.strip():
                with st.spinner("Generating response..."):
                    try:
                        # Build messages for OpenAI API
                        messages = []
                        if system_prompt_option != "Default (Tokenizer's built-in)" and system_prompt is not None:
                            if system_prompt:  # Only add if not empty
                                messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": test_prompt})
                        
                        # Use OpenAI API for testing
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            response = loop.run_until_complete(
                                st.session_state.dataset_manager.client.chat_complete(
                                    messages=messages,
                                    max_tokens=test_sampling_config.max_tokens,
                                    temperature=test_sampling_config.temperature,
                                    top_p=test_sampling_config.top_p
                                )
                            )
                        finally:
                            loop.close()
                    except Exception as e:
                        st.error(f"❌ Generation failed: {e}")
                        response = f"Error: {e}"
                
                st.markdown("### Response")
                st.markdown(f"""
                    <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #6366f1;">
                        <p style="margin: 0; color: #f8fafc;">{response}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show recent logs for debugging
                with st.expander("🔍 Debug Logs (Last 20 lines)"):
                    try:
                        with open('app.log', 'r') as f:
                            lines = f.readlines()
                            recent_logs = ''.join(lines[-20:])
                            st.code(recent_logs, language='text')
                    except FileNotFoundError:
                        st.info("No log file found yet.")
            else:
                st.warning("⚠️ Please enter a test prompt.")
    
    with col2:
        st.markdown("### System Prompt Guide")
        
        st.markdown("""
        **🧪 Empty System Prompt**
        - Tests pure LoRA learned behavior
        - No guidance from system prompt
        - Best for seeing raw character adaptation
        
        **🎭 Roleplay Director**
        - Encourages character roleplay
        - Tests how LoRA responds to roleplay cues
        - Good for interactive character testing
        
        **✏️ Custom System Prompt**
        - Test specific scenarios
        - Control system behavior precisely
        - Useful for targeted evaluation
        
        **🤖 Default (SmolLM)**
        - Uses built-in assistant prompt
        - May conflict with character training
        - Good for comparison baseline
        """)
        
        st.markdown("### Quick Tests")
        
        quick_tests = [
            "Who are you?",
            "What drives you in life?",
            "Describe your greatest fear.", 
            "Tell me about your past.",
            "What's your personality like?",
            "How do you speak to others?"
        ]
        
        for i, prompt in enumerate(quick_tests):
            if st.button(f"🎯 {prompt}", key=f"quick_test_{i}"):
                # Auto-fill the test prompt
                st.session_state.quick_test_prompt = prompt
                st.rerun()
        
        # Model comparison
        st.markdown("### Model Comparison")
        
        if len(available_models) > 1:
            st.info("Go to the '⚔️ Model Comparison' page from the sidebar to compare models side-by-side.")
        else:
            st.info("Train multiple checkpoints to enable model comparison.")

# Dataset explorer page
def page_dataset_explorer_v2():
    """
    An advanced UI for dataset exploration, curation, and management.
    Allows for bucketing, bulk actions, editing, and manual additions.
    """
    st.markdown('<h2 class="gradient-text">📚 Dataset Explorer & Curator</h2>', unsafe_allow_html=True)

    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return

    char_name = st.session_state.current_character.get("name", "unknown_char")
    bucket_key = f"dataset_buckets_{char_name}"
    selection_key = f"selection_{char_name}"

    # Initialize bucket and selection state for the current character
    if bucket_key not in st.session_state:
        main_dataset = st.session_state.get('dataset_preview', [])
        st.session_state[bucket_key] = {
            "main": main_dataset,
            "quarantined": [],
        }
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {}

    buckets = st.session_state[bucket_key]
    selection_state = st.session_state[selection_key]

    # --- TOP-LEVEL OVERVIEW & ACTIONS ---
    st.markdown("### 🗄️ Dataset Overview")
    total_samples = sum(len(v) for v in buckets.values())
    
    overview_cols = st.columns(4)
    overview_cols[0].metric("Total Active Samples", sum(len(v) for k, v in buckets.items() if k != 'quarantined'))
    overview_cols[1].metric("Quarantined Samples", len(buckets.get('quarantined', [])))
    overview_cols[2].metric("Total Buckets", len(buckets))

    with overview_cols[3]:
        # Consolidate all non-quarantined data for export
        active_data = [sample for b_name, b_list in buckets.items() if b_name != 'quarantined' for sample in b_list]
        if active_data:
            json_data = json.dumps(active_data, indent=2)
            st.download_button(
                label="⬇️ Export Active Dataset",
                data=json_data,
                file_name=f"{char_name}_active_dataset.json",
                mime="application/json",
                use_container_width=True
            )

    st.markdown("---")

    # --- DIALOG FOR EDITING A SAMPLE ---
    if st.session_state.get('item_to_edit'):
        item_info = st.session_state.item_to_edit
        sample_to_edit = buckets[item_info['bucket']][item_info['index']]

        @st.dialog("✍️ Edit Sample")
        def edit_dialog():
            st.markdown("### Edit Conversation Turn")
            user_content = st.text_area("User Prompt", sample_to_edit['messages'][1]['content'], height=150)
            asst_content = st.text_area("Assistant Response", sample_to_edit['messages'][2]['content'], height=200)

            if st.button("💾 Save Changes", use_container_width=True):
                buckets[item_info['bucket']][item_info['index']]['messages'][1]['content'] = user_content
                buckets[item_info['bucket']][item_info['index']]['messages'][2]['content'] = asst_content
                st.session_state.item_to_edit = None
                st.rerun()

        edit_dialog()

    # --- MAIN CURATION UI ---
    col1, col2 = st.columns([1, 2.5])

    with col1:
        st.markdown("### 🗃️ Buckets")
        bucket_names = list(buckets.keys())
        if 'selected_bucket' not in st.session_state or st.session_state.selected_bucket not in bucket_names:
            st.session_state.selected_bucket = bucket_names[0]

        def format_bucket_name(b_name):
            return f"{b_name.replace('_', ' ').title()} ({len(buckets.get(b_name, []))})"

        selected_bucket = st.radio(
            "Select a bucket:",
            bucket_names,
            format_func=format_bucket_name,
            key='selected_bucket'
        )
        
        with st.expander("Manage Buckets"):
            new_bucket_name = st.text_input("New bucket name", key="new_bucket_name").strip().lower().replace(" ", "_")
            if st.button("➕ Create Bucket", use_container_width=True):
                if new_bucket_name and new_bucket_name not in buckets:
                    buckets[new_bucket_name] = []
                    st.rerun()
                else:
                    st.warning("Invalid or duplicate bucket name.")
            
            # Allow deleting custom buckets
            custom_buckets = [b for b in bucket_names if b not in ['main', 'quarantined']]
            if custom_buckets:
                bucket_to_delete = st.selectbox("Delete bucket", custom_buckets)
                if st.button("🗑️ Delete Bucket", use_container_width=True):
                    # Move items to main before deleting
                    items_to_move = buckets.pop(bucket_to_delete, [])
                    buckets['main'].extend(items_to_move)
                    st.session_state.selected_bucket = 'main'
                    st.success(f"Deleted bucket '{bucket_to_delete}' and moved its contents to 'main'.")
                    st.rerun()

    with col2:
        st.markdown(f"### ✏️ Contents of `{selected_bucket}`")
        bucket_data = buckets[selected_bucket]
        
        # --- BULK ACTIONS ---
        selected_indices = [int(k.split('_')[1]) for k, v in selection_state.items() if v and k.startswith(selected_bucket)]
        if selected_indices:
            st.markdown("#### **Bulk Actions** for selected items")
            bulk_cols = st.columns(2)
            
            with bulk_cols[0]:
                other_buckets = [b for b in bucket_names if b != selected_bucket]
                if other_buckets:
                    target_bucket = st.selectbox("Move to bucket", other_buckets, key="bulk_move_target")
                    if st.button("➡️ Move Selected", use_container_width=True):
                        items_to_move = [bucket_data[i] for i in sorted(selected_indices, reverse=True)]
                        buckets[target_bucket].extend(items_to_move)
                        for i in sorted(selected_indices, reverse=True):
                            del bucket_data[i]
                        # Clear selection state
                        st.session_state[selection_key] = {}
                        st.success(f"Moved {len(items_to_move)} items to `{target_bucket}`.")
                        st.rerun()

            with bulk_cols[1]:
                if st.button("🗑️ Delete Selected Permanently", use_container_width=True, type="primary"):
                    for i in sorted(selected_indices, reverse=True):
                        del bucket_data[i]
                    st.session_state[selection_key] = {}
                    st.success(f"Permanently deleted {len(selected_indices)} items.")
                    st.rerun()
            st.markdown("---")

        # --- DATA TABLE ---
        if not bucket_data:
            st.info("This bucket is empty. You can add a sample below.")
        else:
            # Pagination
            page_size = st.select_slider("Items per page", [10, 25, 50, 100], value=25)
            page_count = (len(bucket_data) + page_size - 1) // page_size
            page_num = 1
            if page_count > 1:
                page_num = st.number_input("Page", 1, page_count, 1)
            
            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, len(bucket_data))
            page_data = bucket_data[start_idx:end_idx]

            # Header
            header_cols = st.columns([0.08, 0.38, 0.38, 0.08, 0.08])
            with header_cols[0]:
                select_all = st.checkbox("All", key=f"select_all_{selected_bucket}_{page_num}", label_visibility="hidden")
            header_cols[1].markdown("**User Prompt**")
            header_cols[2].markdown("**Assistant Response**")
            header_cols[3].markdown("**Actions**")
            header_cols[4].markdown("**Actions**")

            # Handle Select All
            if select_all:
                for i in range(start_idx, end_idx):
                    selection_state[f"{selected_bucket}_{i}"] = True
            
            # Display items
            for i, sample in enumerate(page_data):
                global_idx = start_idx + i
                item_key = f"{selected_bucket}_{global_idx}"
                
                row_cols = st.columns([0.08, 0.38, 0.38, 0.08, 0.08])
                with row_cols[0]:
                    is_selected = st.checkbox(" ", key=f"sel_{item_key}", value=selection_state.get(item_key, False), label_visibility="hidden")
                    selection_state[item_key] = is_selected

                with row_cols[1]:
                    st.markdown(f"<div class='data-cell'>{sample['messages'][1]['content']}</div>", unsafe_allow_html=True)
                with row_cols[2]:
                    st.markdown(f"<div class='data-cell'>{sample['messages'][2]['content']}</div>", unsafe_allow_html=True)
                
                with row_cols[3]:
                    if st.button("✏️", key=f"edit_{item_key}", help="Edit sample"):
                        st.session_state.item_to_edit = {"bucket": selected_bucket, "index": global_idx}
                        st.rerun()
                with row_cols[4]:
                    if st.button("🗑️", key=f"del_{item_key}", help="Delete sample"):
                        del bucket_data[global_idx]
                        st.rerun()
                st.markdown('<hr class="row-divider">', unsafe_allow_html=True)
        
        # --- ADD NEW SAMPLE ---
        with st.expander("✍️ Add a new sample to this bucket"):
            with st.form(key="new_sample_form", clear_on_submit=True):
                user_prompt = st.text_area("User Prompt", height=100)
                assistant_response = st.text_area("Assistant Response", height=150)
                
                if st.form_submit_button("Add Sample", use_container_width=True):
                    if user_prompt and assistant_response:
                        system_prompt = st.session_state.dataset_metadata.get('system_prompt_config', {}).get('prompt', '')
                        new_sample = {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": assistant_response}
                            ]
                        }
                        buckets[selected_bucket].append(new_sample)
                        st.success("Sample added!")
                        st.rerun()
                    else:
                        st.warning("Both fields are required.")
    
    # --- PERSIST CHANGES ---
    # Update the main dataset preview for other parts of the app
    # The "active" dataset for training is everything NOT in 'quarantined'.
    st.session_state.dataset_preview = [
        sample for bucket_name, bucket_list in buckets.items() 
        if bucket_name != 'quarantined' 
        for sample in bucket_list
    ]

def page_model_comparison():
    """Page for comparing different models side-by-side."""
    st.markdown('<h2 class="gradient-text">⚔️ Model Comparison Dashboard</h2>', unsafe_allow_html=True)

    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return

    char_name = st.session_state.current_character.get("name", "Unknown")
    st.markdown(f"### Comparing models for: **{char_name}**")

    # Get available models
    available_models = st.session_state.inference_manager.get_available_models()
    
    # Filter for models related to the current character, plus the base model
    character_models = [m for m in available_models if char_name.lower().replace(' ', '_') in m.lower() or "Base:" in m]
    
    if len(character_models) < 2:
        st.info("ℹ️ You need at least two trained models/checkpoints for this character to compare them. The base model is always available for comparison.")
        # Also add base if not present, in case no models are trained yet
        if not any("Base:" in m for m in character_models):
             character_models.append(f"Base: {st.session_state.inference_manager.base_model}")
        if len(character_models) < 2:
             return
    
    selected_models = st.multiselect(
        "Select models to compare (2 or more)",
        options=character_models,
        default=character_models[:2] if len(character_models) >= 2 else character_models,
        help="Choose checkpoints or final LoRA models to test side-by-side."
    )

    if len(selected_models) < 2:
        st.warning("⚠️ Please select at least two models to compare.")
        return

    # Test prompt
    prompt = st.text_area(
        "Enter a test prompt",
        "Who are you and what are your core beliefs?",
        height=100,
        key="comparison_prompt"
    )

    # Generation settings
    with st.expander("⚙️ Generation Settings"):
        from utils.sampling_config import render_sampling_config_ui, SamplingConfig
        default_test_config = SamplingConfig(
            temperature=0.7,
            top_p=0.9,
            max_tokens=200,
            repetition_penalty=1.1,
        )
        test_sampling_config = render_sampling_config_ui(
            current_config=default_test_config,
            key_prefix="model_comparison_gen"
        )
    
    if st.button("🚀 Compare Responses", use_container_width=True, type="primary"):
        if not prompt.strip():
            st.error("❌ Please enter a prompt.")
            return

        # Store results in session state to persist them
        with st.spinner("Generating responses and fetching metrics..."):
            sp_config = test_sampling_config.to_dict()
            if 'min_tokens' in sp_config:
                sp_config.pop('min_tokens')
            if 'max_tokens' in sp_config:
                sp_config.pop('max_tokens')
            comparison_results = st.session_state.comparison_manager.compare_models_side_by_side(
                model_identifiers=selected_models,
                prompt=prompt,
                generation_config=sp_config
            )
            metrics_data = st.session_state.comparison_manager.get_comparison_metrics(selected_models)
            
            st.session_state.comparison_data = {
                "responses": comparison_results,
                "metrics": metrics_data
            }
        st.success("✅ Comparison complete!")

    # Display comparison if data exists
    if 'comparison_data' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Comparison Results")

        responses = st.session_state.comparison_data['responses']
        metrics = st.session_state.comparison_data['metrics']

        # Side-by-side responses
        st.markdown("#### Side-by-Side Responses")
        cols = st.columns(len(selected_models))
        for i, model_id in enumerate(selected_models):
            with cols[i]:
                st.markdown(f"##### {model_id}")
                st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 8px; height: 300px; overflow-y: auto;">
                        {responses.get(model_id, "N/A")}
                    </div>
                """, unsafe_allow_html=True)
                
                # Promotion button
                if "Base:" not in model_id:
                    if st.button(f"🏆 Promote {model_id.split('/')[-1]}", key=f"promote_{i}", use_container_width=True):
                        checkpoint_id = model_id.split(': ')[1]
                        success = st.session_state.comparison_manager.promote_checkpoint(
                            character_name=char_name,
                            checkpoint_id=checkpoint_id,
                            reason=f"Promoted after comparing with prompt: '{prompt[:50]}...'"
                        )
                        if success:
                            st.success(f"✅ Promoted {checkpoint_id} as the best version!")
                        else:
                            st.error("❌ Failed to promote checkpoint.")
        
        # Display promoted checkpoint info
        promoted_checkpoint = st.session_state.comparison_manager.get_promoted_checkpoint(char_name)
        if promoted_checkpoint:
            st.success(f"🏆 **Promoted Model:** `{promoted_checkpoint}` is currently selected as the best version for this character.")


        # Radar Chart for Metrics
        st.markdown("#### Quantitative Metrics Comparison")
        
        # Check if we have any metrics data to plot
        if any(metrics.values()):
            fig = go.Figure()

            # Define metrics to plot and their properties
            metric_info = {
                'eval_loss': {'name': 'Eval Loss (1/x)', 'invert': True},
                'avg_consistency': {'name': 'Avg Consistency', 'invert': False},
                'loss': {'name': 'Train Loss (1/x)', 'invert': True},
            }
            metric_labels = list(metric_info.keys())
            
            # Find max value for normalization after inversion
            max_inverted_loss = 1
            all_values = []
            for model_id in selected_models:
                model_metrics = metrics.get(model_id, {})
                for label in metric_labels:
                    val = model_metrics.get(label)
                    if val is not None and metric_info[label]['invert'] and val > 0:
                        all_values.append(1 / val)
                    elif val is not None and not metric_info[label]['invert']:
                        all_values.append(val)
            
            max_radial_value = max(all_values) if all_values else 1

            for model_id in selected_models:
                model_metrics = metrics.get(model_id, {})
                values = []
                for label in metric_labels:
                    val = model_metrics.get(label)
                    if val is None:
                        values.append(0)
                    elif metric_info[label]['invert']:
                        values.append(1 / val if val > 0 else 0)
                    else:
                        values.append(val)
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[info['name'] for info in metric_info.values()],
                    fill='toself',
                    name=model_id
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max_radial_value * 1.1] # Add padding
                    )),
                showlegend=True,
                template="plotly_dark",
                title="Model Metrics Radar Chart (Higher is Better)"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No quantitative metrics found to generate a radar chart. Make sure `training_summary.json` exists for the selected models.")

# Main app function
def main():
    """Main app function"""
    init_session_state()

    # Handle TensorBoard launch request
    if st.session_state.get("launch_tensorboard_request"):
        logdir = st.session_state.training_manager.get_tensorboard_logdir()
        if logdir:
            # This is where we would use the run_terminal_cmd tool in a real scenario
            # For this example, we'll simulate the launch and set the state.
            st.session_state.tensorboard_launched = True
            st.info(f"TensorBoard is launching in the background with log directory: {logdir}")
        else:
            st.warning("TensorBoard log directory not yet available. Please wait a moment for training to start.")
        # Reset the request flag
        st.session_state.launch_tensorboard_request = False


    render_header()
    
    # Sidebar navigation
    selected_page = render_sidebar()
    
    # Page routing
    if selected_page == "📁 Character Upload":
        page_character_upload()
    elif selected_page == "🔍 Dataset Preview":
        page_dataset_preview()
    elif selected_page == "📚 Dataset Explorer":
        page_dataset_explorer_v2()
    elif selected_page == "⚙️ Training Config":
        page_training_config()
    elif selected_page == "📊 Training Dashboard":
        page_training_dashboard()
    elif selected_page == "🧪 Model Testing":
        page_model_testing()
    elif selected_page == "⚔️ Model Comparison":
        page_model_comparison()
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: #64748b; border-top: 1px solid rgba(100, 116, 139, 0.2); margin-top: 3rem;">
            <p>🎭 Character AI Training Studio • Built with ❤️ and Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 