import asyncio
from asyncio.log import logger
import os
# Disable Streamlit's autoreload file-watcher early to avoid PyTorch inspection
# errors. Must be set before importing Streamlit so that Streamlit reads the
# configuration on startup.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/.cache/vllm_hf")

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

# Set specific loggers to DEBUG for more detail
logging.getLogger('utils.inference').setLevel(logging.DEBUG)

# ✅ FIX: Global singleton to prevent DatasetManager re-initialization
_GLOBAL_DATASET_MANAGER = None

def get_or_create_dataset_manager(preferred_engine: Optional[str] = None, generation_model: Optional[str] = None):
    """Get or create a singleton DatasetManager instance to prevent re-initialization"""
    global _GLOBAL_DATASET_MANAGER
    
    if _GLOBAL_DATASET_MANAGER is None:
        logger.info(f"🔧 Creating singleton DatasetManager with engine: {preferred_engine}, model: {generation_model}")
        _GLOBAL_DATASET_MANAGER = DatasetManager(
            preferred_engine=preferred_engine,
            generation_model=generation_model
        )
        logger.info("✅ Singleton DatasetManager created successfully")
    else:
        logger.info("♻️ Reusing existing singleton DatasetManager")
    
    return _GLOBAL_DATASET_MANAGER

# Initialize session state
def init_session_state():
    if 'character_manager' not in st.session_state:
        st.session_state.character_manager = CharacterManager()
    
    # ✅ FIX: Use singleton DatasetManager to prevent re-initialization and infinite loops
    if 'dataset_manager' not in st.session_state:
        # Auto-detect inference engine, but allow override
        preferred_engine = os.getenv('INFERENCE_ENGINE', None)
        generation_model = os.getenv('GENERATION_MODEL', None)
        
        try:
            # Use singleton function to prevent re-initialization
            st.session_state.dataset_manager = get_or_create_dataset_manager(
                preferred_engine=preferred_engine,
                generation_model=generation_model
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
    if 'current_character' not in st.session_state:
        st.session_state.current_character = None
    if 'training_status' not in st.session_state:
        st.session_state.training_status = 'idle'
    if 'dataset_preview' not in st.session_state:
        st.session_state.dataset_preview = None
    if 'dataset_metadata' not in st.session_state:
        st.session_state.dataset_metadata = {}
    if 'selected_engine' not in st.session_state:
        # Only set if dataset_manager exists and is properly initialized
        if hasattr(st.session_state, 'dataset_manager') and st.session_state.dataset_manager:
            st.session_state.selected_engine = st.session_state.dataset_manager.inference_engine.name
        else:
            st.session_state.selected_engine = "Unknown"
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = None
    if 'generation_model' not in st.session_state:
        st.session_state.generation_model = None

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
            options=["📁 Character Upload", "🔍 Dataset Preview", "📚 Dataset Explorer", "⚙️ Training Config", "📊 Training Dashboard", "🧪 Model Testing"],
            icons=["upload", "search", "table", "gear", "graph-up", "flask"],
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
        
        # Engine info (read-only)
        current_engine = st.session_state.dataset_manager.inference_engine
        engine_name = getattr(current_engine, 'name', 'Unknown')
        
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 0.5rem 0;">🛠️ Default Engine</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #94a3b8;">{engine_name}</p>
                <p style="margin: 0; font-size: 0.8rem; color: #64748b;">Configure per-task in each section</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Global settings
        with st.expander("⚙️ Global Settings", expanded=False):
            # Global thinking model toggle
            st.markdown("**🧠 Thinking Models**")
            thinking_enabled = st.checkbox(
                "Enable thinking model support",
                value=st.session_state.get('global_thinking_enabled', False),
                help="Enable support for models that use <think></think> reasoning (Deepseek R1, QwQ, etc.)",
                key="global_thinking_toggle"
            )
            
            if thinking_enabled:
                os.environ['REASONING_PARSER'] = "deepseek_r1"
                thinking_template = st.selectbox(
                    "Default thinking template",
                    ["Deepseek Template", "Qwen3 Template"],
                    help="Default template for thinking models",
                    key="global_thinking_template"
                )
                
                # Store global thinking config - don't set the variable controlled by a widget
                st.session_state.global_thinking_enabled = True
                # REMOVED: st.session_state.global_thinking_template = thinking_template
                
                # Apply to current engine if it supports it
                thinking_config = {
                    'enabled': True,
                        'template': thinking_template,
                    'qwen_thinking': True
                }
                
                if hasattr(current_engine, 'set_thinking_config'):
                    current_engine.set_thinking_config(thinking_config)
                    
                
                    st.info(f"🧠 Using {thinking_template} globally")
                else:
                    st.session_state.global_thinking_enabled = False
            
            # Cache management
            st.markdown("**💾 Cache Management**")
            if st.button("🧹 Clear All Caches", help="Clear model and GGUF caches"):
                try:
                    # Clear GGUF cache
                    from utils.llama_cpp_engine import LlamaCppEngine
                    gguf_count = LlamaCppEngine.clear_gguf_cache()
                    
                    # Clear CUDA cache
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    st.success(f"✅ Cleared {gguf_count} GGUF files and CUDA cache")
                except Exception as e:
                    st.error(f"❌ Cache clear error: {e}")
        
        # Help section
        with st.expander("ℹ️ Quick Help", expanded=False):
            st.markdown("""
            **🎯 Per-Task Configuration**
            - **Dataset Generation**: Choose models & sampling in the generation tab
            - **Model Testing**: Switch engines & models in the testing section
            - **Training**: Configure base models in training config
            
            **💡 Tips**
            - Different models work best for different tasks
            - Use RpR model for creative generation
            - Use smaller models for testing
            - Each section remembers its settings
            """)
    
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
    
    # Generation mode tabs
    generation_tab, standard_tab, quality_tab = st.tabs([
        "📊 Overview", 
        "🚀 Standard Generation", 
        "⭐ Quality-First Generation"
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
            # ✅ NEW: Model Selection for Question Generation
            st.markdown("#### 🤖 Model Selection")
            col_qa_model1, col_qa_model2 = st.columns([2, 1])
            
            with col_qa_model1:
                # Available models for question generation
                if st.session_state.selected_engine == "vLLM":
                    qa_available_models = [
                        "Current Engine's Model",
                        "PocketDoc/Dans-PersonalityEngine-V1.3.0-24b",
                        "ArliAI/QwQ-32B-ArliAI-RpR-v4", 
                        "meta-llama/Llama-3.1-70B-Instruct",
                        "meta-llama/Llama-3.2-3B-Instruct",
                        "mistralai/Mistral-7B-Instruct-v0.2",
                        "Qwen/Qwen2.5-7B-Instruct",
                        "microsoft/Phi-3.5-mini-instruct",
                        "Custom (enter HF ID below)"
                    ]
                else:
                    qa_available_models = [
                        "Current Engine's Model",
                        "Switch to vLLM for model selection"
                    ]
                
                selected_qa_model = st.selectbox(
                    "Question Generation Model",
                    qa_available_models,
                    help="Choose the model for generating questions",
                    key="question_gen_model"
                )
                
                # Handle custom model input
                if selected_qa_model == "Custom (enter HF ID below)":
                    custom_qa_model = st.text_input(
                        "HuggingFace Model ID",
                        placeholder="e.g., teknium/OpenHermes-2.5-Mistral-7B",
                        help="Enter any compatible HuggingFace model ID",
                        key="custom_qa_model"
                    )
                    if custom_qa_model:
                        final_qa_model = custom_qa_model
                    else:
                        final_qa_model = None  # Use current
                elif selected_qa_model == "Current Engine's Model":
                    final_qa_model = None  # Use current
                elif selected_qa_model == "Switch to vLLM for model selection":
                    st.info("💡 Switch to vLLM engine in the sidebar for model selection")
                    final_qa_model = None  # Use current
                else:
                    final_qa_model = selected_qa_model
            
            with col_qa_model2:
                if final_qa_model and final_qa_model != getattr(st.session_state.dataset_manager.inference_engine, 'model_name', None):
                    if st.button("🔄 Load QA Model", help="Load the selected model for question generation", key="load_qa_model_btn"):
                        with st.spinner("Loading model..."):
                            try:
                                # Store current model info to restore later if needed
                                st.session_state.qa_temp_engine = st.session_state.dataset_manager
                                
                                # Create new dataset manager with selected model
                                st.session_state.dataset_manager = get_or_create_dataset_manager(
                                    preferred_engine=st.session_state.selected_engine,
                                    generation_model=final_qa_model
                                )
                                st.success(f"✅ Loaded {final_qa_model.split('/')[-1]}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to load model: {e}")
                
                # Show current model
                current_qa_model = getattr(st.session_state.dataset_manager.inference_engine, 'model_name', 'Unknown')
                st.info(f"**Current**: {current_qa_model.split('/')[-1] if '/' in current_qa_model else current_qa_model}")
            
            # ✅ NEW: Sampling Configuration for Question Generation
            st.markdown("#### 🎛️ Question Generation Settings")
            
            # Import and use sampling configuration
            from utils.sampling_config import render_sampling_config_ui, SamplingConfig, get_model_preset
            
            # Get current model name for preset detection
            current_qa_model = getattr(st.session_state.dataset_manager.inference_engine, 'model_name', None)
            
            # Check if we have a model-specific preset
            qa_model_preset = get_model_preset(current_qa_model) if current_qa_model else None
            if qa_model_preset:
                st.info(f"🎯 **{qa_model_preset['name']}** preset available for this model")
            
            # Create default config for question generation (different from generation - better for questions)
            qa_default_config = SamplingConfig(
                temperature=0.9,  # Slightly higher for more diverse questions
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.05,
                max_tokens=100  # Questions are shorter
            )
            
            # ✅ FIX: Pass use_expander=False to avoid nested expander error
            qa_sampling_config = render_sampling_config_ui(
                current_config=qa_default_config,
                model_name=current_qa_model,
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
                        
                        # ✅ NEW: Restore original engine if we temporarily switched
                        if st.session_state.get('qa_temp_engine') and final_qa_model is not None:
                            st.session_state.dataset_manager = st.session_state.qa_temp_engine
                            st.session_state.pop('qa_temp_engine', None)
                            st.success("✅ Restored original generation engine")
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

            # Show generated questions and allow user to add to baseline
            gen_data = st.session_state.get('generated_questions')
            if gen_data:
                st.markdown("#### Review Generated Questions")
                selections = []
                for idx, item in enumerate(gen_data):
                    # Layout: checkbox | question | context toggle
                    cols = st.columns([0.08, 0.72, 0.2])
                    with cols[0]:
                        include = st.checkbox(f"q{idx+1}", value=True, label_visibility="hidden")
                        if include:
                            selections.append(item['question'])
                    with cols[1]:
                        st.markdown(f"**{idx+1}. {item['question']}**")
                    with cols[2]:
                        toggle_key = f"show_ctx_{idx}"
                        if st.button("Context ↕"):
                            st.session_state[toggle_key] = not st.session_state.get(toggle_key, False)
                    # Display context when toggled
                    if st.session_state.get(toggle_key, False) and item['context']:
                        st.markdown("**Context used:**")
                        for j, ctx in enumerate(item['context']):
                            st.markdown(f"*Q{j+1}:* {ctx['user']}")
                            st.markdown(f"*A{j+1}:* {ctx['assistant']}")
                        st.markdown("---")

                if selections and st.button("➕ Add Selected Questions", use_container_width=True):
                    st.session_state.dataset_manager.default_user_prompts.extend(selections)
                    st.success(f"Added {len(selections)} questions to baseline list.")

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
        st.markdown("### 🚀 Standard Dataset Generation")
        
        # ✅ NEW: Model Selection for Dataset Generation
        st.markdown("#### 🤖 Model Selection")
        col_model1, col_model2 = st.columns([2, 1])
        
        with col_model1:
            # Available models for dataset generation
            if st.session_state.selected_engine == "vLLM":
                available_models = [
                    "PocketDoc/Dans-PersonalityEngine-V1.3.0-24b",  # Default
                    "ArliAI/QwQ-32B-ArliAI-RpR-v4", 
                    "meta-llama/Llama-3.1-70B-Instruct",
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "microsoft/Phi-3.5-mini-instruct",
                    "Custom (enter HF ID below)"
                ]
            else:
                available_models = [
                    "Current Llama.cpp Model",
                    "Switch to vLLM for model selection"
                ]
            
            selected_gen_model = st.selectbox(
                "Generation Model",
                available_models,
                help="Choose the model for dataset generation",
                key="dataset_gen_model"
            )
            
            # Handle custom model input
            if selected_gen_model == "Custom (enter HF ID below)":
                custom_gen_model = st.text_input(
                    "HuggingFace Model ID",
                    placeholder="e.g., teknium/OpenHermes-2.5-Mistral-7B",
                    help="Enter any compatible HuggingFace model ID",
                    key="custom_gen_model"
                )
                if custom_gen_model:
                    final_gen_model = custom_gen_model
                else:
                    final_gen_model = available_models[0]  # Default
            elif selected_gen_model == "Switch to vLLM for model selection":
                st.info("💡 Switch to vLLM engine in the sidebar for model selection")
                final_gen_model = None
            else:
                final_gen_model = selected_gen_model if selected_gen_model != "Current Llama.cpp Model" else None
        
        with col_model2:
            # Get current model name safely
            current_engine_model_name = getattr(st.session_state.dataset_manager.inference_engine, 'model_name', None)
            
            if final_gen_model and current_engine_model_name and final_gen_model != current_engine_model_name:
                if st.button("🔄 Load Model", help="Load the selected model for generation"):
                    with st.spinner("Loading model..."):
                        try:
                            # Create new dataset manager with selected model
                            st.session_state.dataset_manager = get_or_create_dataset_manager(
                                preferred_engine=st.session_state.selected_engine,
                                generation_model=final_gen_model
                            )
                            st.success(f"✅ Loaded {final_gen_model.split('/')[-1]}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to load model: {e}")
            
            # Show current model
            current_gen_model = getattr(st.session_state.dataset_manager.inference_engine, 'model_name', 'Unknown')
            st.info(f"**Current**: {current_gen_model.split('/')[-1] if '/' in current_gen_model else current_gen_model}")
        
        # Replace the basic parameter sliders with sampling configuration
        st.markdown("#### 🎛️ Generation Settings")
        
        # Import and use sampling configuration
        from utils.sampling_config import render_sampling_config_ui, SamplingConfig, get_model_preset
        
        # Get current model name for preset detection
        current_model = getattr(st.session_state.dataset_manager.inference_engine, 'model_name', None)
        
        # Check if we have a model-specific preset
        model_preset = get_model_preset(current_model) if current_model else None
        if model_preset:
            st.info(f"🎯 **{model_preset['name']}** preset available for this model")
        
        # Create default config
        default_config = SamplingConfig(
            temperature=0.8,
            top_p=0.9,
            max_tokens=400,
            repetition_penalty=1.1,
        )
        
        # Render sampling configuration UI
        sampling_config = render_sampling_config_ui(
            current_config=default_config,
            model_name=current_model,
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
                st.write(f"• Model: {current_model.split('/')[-1] if current_model and '/' in current_model else 'Unknown'}")
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
    
    with quality_tab:
        st.markdown("### ⭐ Quality-First Dataset Generation")
        
        st.info("""
        🎯 **How it works:**
        1. Generate a large number of diverse samples (e.g., 10,000)
        2. Use AI to evaluate each sample for quality and character consistency
        3. Curate the best samples while maintaining diversity
        
        This produces much higher quality datasets but takes more time.
        """)
        
        # Show current model info
        if st.session_state.selected_engine == "vLLM":
            if hasattr(st.session_state.dataset_manager.inference_engine, 'gguf_file') and st.session_state.dataset_manager.inference_engine.gguf_file:
                # GGUF model
                display_name = getattr(st.session_state.dataset_manager.inference_engine, 'model_display_name', 'GGUF Model')
                st.info(f"🔧 Using GGUF model **{display_name}** for generation")
            else:
                # Regular model
                model_name = getattr(st.session_state.dataset_manager.inference_engine, 'model_name', 'Unknown')
                model_display = model_name.split('/')[-1] if '/' in model_name else model_name
                st.info(f"🔧 Using **{model_display}** for generation")
        else:
            st.info(f"🔧 Using **{st.session_state.selected_engine}** for generation")
        
        # Quality generation settings
        col1, col2 = st.columns(2)
        
        with col1:
            raw_samples = st.number_input(
                "Samples to Generate",
                min_value=1000,
                max_value=50000,
                value=10000,
                step=1000,
                help="More samples = better final quality but longer generation time"
            )
            
            final_size = st.number_input(
                "Final Dataset Size",
                min_value=50,
                max_value=1000,
                value=200,
                help="Number of high-quality samples to keep"
            )
            
            quality_threshold = st.slider(
                "Quality Threshold (0-10)",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.5,
                help="Minimum quality score to keep a sample"
            )
        
        with col2:
            diversity_weight = st.slider(
                "Diversity Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Balance between pure quality (0) and diversity (1)"
            )
            
            judgment_batch_size = st.selectbox(
                "Judgment Batch Size",
                [10, 25, 50, 100],
                index=2,
                help="Larger batches are more efficient with vLLM"
            )
            
            # Extra quality checkbox for quality generation
            extra_quality_advanced = st.checkbox(
                "🌟 EXTRA QUALITY", 
                value=False, 
                help="Paraphrase all questions before generation for cleaner, more varied prompts. Takes longer but significantly improves dataset quality.",
                key="quality_extra_quality"
            )
        
        # ✅ NEW: Sampling Configuration for Quality Generation
        st.markdown("#### 🎛️ Generation Settings")
        
        # Import and use sampling configuration
        from utils.sampling_config import render_sampling_config_ui, SamplingConfig, get_model_preset
        
        # Get current model name for preset detection
        current_model = None
        if hasattr(st.session_state.dataset_manager.inference_engine, 'model_name'):
            current_model = st.session_state.dataset_manager.inference_engine.model_name
        elif hasattr(st.session_state.dataset_manager.inference_engine, 'gguf_file'):
            current_model = st.session_state.dataset_manager.inference_engine.gguf_file
        
        # Check if we have a model-specific preset
        model_preset = get_model_preset(current_model) if current_model else None
        if model_preset:
            st.info(f"🎯 **{model_preset['name']}** preset available for your model")
        
        # Create default config for quality generation
        default_quality_config = SamplingConfig(
            temperature=0.9,
            top_p=0.95,
            max_tokens=400,
            repetition_penalty=1.05,
        )
        
        # Render sampling configuration UI
        quality_sampling_config = render_sampling_config_ui(
            current_config=default_quality_config,
            model_name=current_model,
            key_prefix="quality_gen"
            )
        
        # System prompt configuration for quality generation
        st.markdown("### System Prompt Configuration for Training")
        use_custom_system_quality = st.checkbox("Apply custom system prompt to dataset", value=False, help="Replace temporal prompts with a custom prompt after generation", key="quality_custom_system")
        
        if use_custom_system_quality:
            system_prompt_quality = st.text_area(
                "System Prompt for Training",
                placeholder="You are a helpful assistant...\n\nLeave empty for no system prompt.",
                height=100,
                help="After generation with temporal prompts, this will replace all system prompts in the dataset for consistent training.",
                key="quality_system_prompt"
            )
        else:
            system_prompt_quality = None
            st.info("Dataset will keep temporal context system prompts (varies per sample)")
        
        # Judge model selection
        with st.expander("🧑‍⚖️ Judge Model Configuration", expanded=False):
            use_custom_judge = st.checkbox("Use custom judge model", value=False)
            
            if use_custom_judge:
                judge_model = st.text_input(
                    "HuggingFace Model ID",
                    placeholder="e.g., meta-llama/Llama-3.1-70B-Instruct",
                    help="vLLM will automatically download and use this model as judge"
                )
            else:
                judge_model = None
                st.info("Using the same model for generation and judging (PersonalityEngine-24B)")
        
        # Estimated time
        samples_per_sec = 2 if st.session_state.selected_engine == "vLLM" else 0.5
        judge_per_sec = 5 if st.session_state.selected_engine == "vLLM" else 1
        est_gen_time = raw_samples / samples_per_sec / 60
        est_judge_time = raw_samples / judge_per_sec / 60
        est_total_time = est_gen_time + est_judge_time
        
        # Adjust estimated time for extra quality
        extra_quality_time = 0
        if 'extra_quality_advanced' in locals() and extra_quality_advanced:
            # Estimate paraphrasing time: ~0.5 seconds per prompt, assuming ~100-200 prompts
            estimated_prompts = min(raw_samples // 50, 200)  # Rough estimate
            extra_quality_time = estimated_prompts * 0.5 / 60  # Convert to minutes
            
        total_with_extra = est_total_time + extra_quality_time
        
        time_info = f"""
        📊 **Estimated Processing Time**: ~{total_with_extra:.1f} minutes
        - Generation: ~{est_gen_time:.1f} minutes
        - Evaluation: ~{est_judge_time:.1f} minutes"""
        
        if extra_quality_time > 0:
            time_info += f"\n        - Extra Quality (Paraphrasing): ~{extra_quality_time:.1f} minutes"
            
        st.info(time_info)
        
        # Show extra quality warning if enabled
        if extra_quality_advanced:
            st.warning("🌟 EXTRA QUALITY enabled - Generation will take longer but produce cleaner, more varied prompts")
        
        # Quality generation button
        if st.button("⭐ Start Quality-First Generation", use_container_width=True, type="primary"):
            # ✅ FIX: Set generation state to prevent UI interference
            st.session_state._generating_dataset = True
            
            # Create placeholders for progress tracking
            stage_text = st.empty()
            progress_bar = st.progress(0)
            stats_placeholder = st.empty()
            
            # Callbacks
            def update_progress(current_progress):
                progress_bar.progress(current_progress)
            
            def update_stage(stage_info):
                stage_text.info(f"**Stage**: {stage_info['message']}")
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run quality-first generation
                dataset = loop.run_until_complete(
                    st.session_state.dataset_manager.generate_with_quality_curation(
                        character=st.session_state.current_character,
                        raw_samples_target=raw_samples,
                        final_dataset_size=final_size,
                        quality_threshold=quality_threshold,
                        diversity_weight=diversity_weight,
                        judgment_batch_size=judgment_batch_size,
                        judge_model=judge_model,
                        progress_callback=update_progress,
                        stage_callback=update_stage,
                        custom_system_prompt=system_prompt_quality if use_custom_system_quality else None,
                        extra_quality=extra_quality_advanced,
                        **quality_sampling_config.to_dict()  # Pass all sampling parameters
                    )
                )
                
                st.session_state.dataset_preview = dataset
                # Update metadata if we generated with custom system prompt
                if use_custom_system_quality:
                    st.session_state.dataset_metadata = {
                        'system_prompt_config': {
                            'type': 'custom',
                            'prompt': system_prompt_quality
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
                stage_text.success("✅ Quality-first generation complete!")
                
                # Show final stats
                st.success(f"""
                🎉 **Generation Complete!**
                - Generated: {raw_samples} raw samples
                - Curated: {len(dataset)} high-quality samples
                - Acceptance rate: {(len(dataset)/raw_samples)*100:.1f}%
                """)
                
                # ✅ FIX: Clear generation state before rerun
                st.session_state._generating_dataset = False
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error in quality-first generation: {str(e)}")
                logger.error(f"Quality generation error: {e}", exc_info=True)
                # ✅ FIX: Always clear generation state on error
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

# Training configuration page
def page_training_config():
    """Training configuration page"""
    st.markdown('<h2 class="gradient-text">⚙️ Training Configuration</h2>', unsafe_allow_html=True)
    
    # If training is already running or paused, encourage user to switch to Dashboard
    if st.session_state.get('training_status') in ['training', 'paused']:
        st.info("🚧 Training is in progress. Please use the Training Dashboard to monitor or control the run.")
        return
    
    if not st.session_state.dataset_preview:
        st.warning("⚠️ Please generate a dataset first.")
        return
    
    dataset_size = len(st.session_state.dataset_preview)
    
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
            
            base_model_choice = st.selectbox(
                "Select base model",
                base_model_options,
                help="Smaller models train faster and work better for character LoRAs"
            )
            
            if base_model_choice == "Custom (enter HF ID below)":
                custom_base_model = st.text_input(
                    "HuggingFace Model ID",
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
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Dynamically pick a sensible default epoch count based on dataset size
                if dataset_size >= 200:
                    optimal_epochs = 3  # Reduced for larger datasets
                else:
                    optimal_epochs = min(6, max(3, 600 // dataset_size))  # 3-6 epochs for smaller datasets
                epochs = st.slider("Epochs", 1, 10, optimal_epochs, help="5-10 epochs recommended by research for character LoRA")
                lr_options = [1e-4, 2e-4, 3e-4, 5e-4]
                default_lr = 2e-4  # More conservative default
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=lr_options,
                    value=default_lr,
                    format_func=lambda x: f"{x:.0e}",
                    help="1e-4 to 5e-4 recommended for character LoRA training"
                )
                batch_size = st.selectbox("Batch Size", [1, 2, 4, 8], index=1)
            
            with col_b:
                gradient_accumulation = st.selectbox("Gradient Accumulation Steps", [1, 2, 4, 8], index=1)  # Default to 2
                warmup_steps = st.slider("Warmup Steps", 0, 100, 10, help="10-20 steps usually sufficient")
                max_grad_norm = st.slider("Max Gradient Norm", 0.5, 2.0, 1.0, step=0.1, help="1.0 is standard")
                
                # Sample selection for dataset
                st.markdown("**Dataset Sampling**")
                if dataset_size > 0:
                    use_all_samples = st.checkbox("Use All Samples", value=True, help="Use the entire dataset for training")
                    if not use_all_samples:
                        max_samples = st.slider(
                            "Number of Samples", 
                            min_value=1, 
                            max_value=dataset_size, 
                            value=min(dataset_size, 100),
                            help=f"Select subset from {dataset_size} total samples (randomized selection)"
                        )
                    else:
                        max_samples = dataset_size
                else:
                    max_samples = dataset_size
            
            # LoRA settings optimized for character training
            st.markdown("#### LoRA Configuration (Character-Optimized)")
            col_c, col_d = st.columns(2)
            
            with col_c:
                default_r = 16  # Optimal for character LoRA per research
                lora_r = st.slider("LoRA Rank (r)", 4, 64, default_r, step=4, 
                                   help="8-16 optimal for character LoRAs. Higher rank = more capacity but slower.")
                # Alpha = rank for character training (not 2x)
                lora_alpha = st.slider("LoRA Alpha", 8, 64, default_r, step=8, 
                                       help="Set equal to rank (α = r) for character training")
            
            with col_d:
                lora_dropout = st.slider("LoRA Dropout", 0.0, 0.2, 0.1, step=0.01, 
                                         help="0.05-0.1 for regularization")
                target_modules = st.multiselect(
                    "Target Modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    default=["q_proj", "k_proj", "v_proj", "o_proj"],  # Focus on attention layers
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
                    value=False,
                    help="If checked, system prompts will be included in training data. Usually better to let LoRA internalize character behavior without system prompts.",
                    disabled=not dataset_has_system
                )
                
                if dataset_has_system and not include_system_prompts:
                    st.info("💡 System prompts will be removed during training (recommended for character LoRAs)")
                elif dataset_has_system and include_system_prompts:
                    st.warning("⚠️ Including system prompts in training - make sure this is intentional")
                elif not dataset_has_system:
                    st.info("ℹ️ Dataset has no system prompts")
                
                fp16 = st.checkbox("Enable FP16", value=True, help="Enables mixed precision training for better performance")
                save_steps = st.slider("Save Every N Steps", 25, 200, 50, step=25)
                logging_steps = st.slider("Log Every N Steps", 1, 20, 5, step=1)
                eval_steps = st.slider("Evaluation Steps", 25, 100, 50, step=25)
                max_steps_override = st.number_input(
                    "Override Total Training Steps (0 = auto)",
                    min_value=0,
                    max_value=50000,
                    value=0,
                    step=100,
                    help="Manually set the total number of optimisation steps if you need finer control. Leave at 0 to use the computed value."
                )
            
            start_training = st.form_submit_button("🚀 Start Training", use_container_width=True)
    
    with col2:
        st.markdown("### Training Recommendations")
        
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
            'eval_steps': eval_steps,
            'max_steps_override': int(max_steps_override) if max_steps_override else 0,
            'resume_from_checkpoint': resume_ckpt,
            'include_system_prompts': include_system_prompts,
            'max_samples': max_samples
        }
        
        try:
            # Use different key to avoid widget conflict
            st.session_state.active_training_config = config
            
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
                elapsed = int(metrics.get('elapsed_time', 0))  # Convert to int for formatting
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
        
        # ✅ NEW: Model Selection for Testing
        st.markdown("#### 🤖 Model & Engine Selection")
        
        col_engine, col_model = st.columns([1, 2])
        
        with col_engine:
            # Engine selection for testing
            test_engines = ["Current", "vLLM", "Llama.cpp"]
            selected_test_engine = st.selectbox(
                "Test Engine", 
                test_engines,
                help="Choose inference engine for testing",
                key="test_engine_select"
            )
        
        with col_model:
            if selected_test_engine == "vLLM":
                # vLLM model options
                vllm_test_models = [
                    "PocketDoc/Dans-PersonalityEngine-V1.3.0-24b",
                    "ArliAI/QwQ-32B-ArliAI-RpR-v4", 
                    "meta-llama/Llama-3.1-70B-Instruct",
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "Custom (enter HF ID below)"
                ]
                test_model_choice = st.selectbox(
                    "vLLM Model",
                    vllm_test_models,
                    help="Choose vLLM model for testing",
                    key="vllm_test_model"
                )
                
                if test_model_choice == "Custom (enter HF ID below)":
                    custom_test_model = st.text_input(
                        "HuggingFace Model ID",
                        placeholder="e.g., teknium/OpenHermes-2.5-Mistral-7B",
                        key="custom_test_model"
                    )
                    final_test_model = custom_test_model if custom_test_model else vllm_test_models[0]
                else:
                    final_test_model = test_model_choice
            
            elif selected_test_engine == "Llama.cpp":
                # Llama.cpp GGUF model input
                gguf_model = st.text_input(
                    "GGUF Model",
                    placeholder="TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf",
                    help="Format: repo_id/filename.gguf",
                    key="gguf_test_model"
                )
                final_test_model = gguf_model if gguf_model else None
            
            else:  # Current
                final_test_model = None
                st.info("Using current inference engine configuration")
        
        # Load test model button
        if final_test_model and selected_test_engine != "Current":
            if st.button("🔄 Load Test Model", help="Load selected model for testing"):
                with st.spinner(f"Loading {selected_test_engine} model..."):
                    try:
                        if selected_test_engine == "vLLM":
                            from utils.vllm_engine import VLLMEngine, VLLMConfig
                            st.session_state.test_engine = VLLMEngine(config=VLLMConfig(model_name=final_test_model))
                        elif selected_test_engine == "Llama.cpp":
                            from utils.llama_cpp_engine import LlamaCppEngine
                            st.session_state.test_engine = LlamaCppEngine(gguf_file=final_test_model)
                        
                        st.success(f"✅ Loaded {final_test_model.split('/')[-1]} with {selected_test_engine}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to load test model: {e}")
        
        # Show current test configuration
        current_test_engine = getattr(st.session_state, 'test_engine', st.session_state.dataset_manager.inference_engine)
        current_test_model_name = getattr(current_test_engine, 'model_name', 'Unknown')
        if hasattr(current_test_engine, 'gguf_file') and current_test_engine.gguf_file:
            current_test_model_name = current_test_engine.gguf_file
        
        st.info(f"**Current Test Setup**: {current_test_engine.name} with {current_test_model_name.split('/')[-1] if '/' in current_test_model_name else current_test_model_name}")
        
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
                    # Use test engine if available, otherwise fall back to inference manager
                    if hasattr(st.session_state, 'test_engine'):
                        # Direct engine generation
                        try:
                            # Build messages for chat template
                            messages = []
                            if system_prompt_option != "Default (Tokenizer's built-in)" and system_prompt is not None:
                                if system_prompt:  # Only add if not empty
                                    messages.append({"role": "system", "content": system_prompt})
                            messages.append({"role": "user", "content": test_prompt})
                            
                            # Apply chat template
                            formatted_prompt = st.session_state.test_engine.apply_chat_template(messages)
                            
                            # Generate using the test engine with all sampling parameters
                            generation_params = {
                                'prompt': formatted_prompt,
                                'max_tokens': test_sampling_config.max_tokens,
                                'temperature': test_sampling_config.temperature,
                                'top_p': test_sampling_config.top_p,
                            }
                            # Add other sampling parameters
                            generation_params.update({k: v for k, v in test_sampling_config.to_dict().items() 
                                                    if k not in ['max_tokens', 'temperature', 'top_p']})
                            
                            # Check if engine has async generation
                            if hasattr(st.session_state.test_engine, 'generate') and callable(st.session_state.test_engine.generate):
                                import asyncio
                                
                                # Create async wrapper for the generation
                                async def async_generate():
                                    return await st.session_state.test_engine.generate(**generation_params)
                                
                                # Run the async generation
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                try:
                                    response = loop.run_until_complete(async_generate())
                                finally:
                                    loop.close()
                            else:
                                # Fallback for sync engines
                                response = "Error: Test engine does not support generation"
                        except Exception as e:
                            st.error(f"❌ Test engine generation failed: {e}")
                            response = f"Error: {e}"
                    else:
                        # Fall back to inference manager
                        response = st.session_state.inference_manager.generate_response(
                            selected_model,
                            test_prompt,
                                max_new_tokens=test_sampling_config.max_tokens,
                                temperature=test_sampling_config.temperature,
                                top_p=test_sampling_config.top_p,
                                repetition_penalty=test_sampling_config.repetition_penalty,
                            system_prompt=system_prompt
                        )
                
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
            st.info("Compare responses from different model checkpoints to evaluate training progress.")
            
            if st.button("📊 Compare Models"):
                st.info("Model comparison feature coming soon!")
        else:
            st.info("Train multiple checkpoints to enable model comparison.")

# Dataset explorer page
def page_dataset_explorer():
    """Dataset explorer page"""
    st.markdown('<h2 class="gradient-text">📚 Dataset Explorer</h2>', unsafe_allow_html=True)
    
    if not st.session_state.dataset_preview:
        st.warning("⚠️ Please generate a dataset first.")
        return
    
    dataset = st.session_state.dataset_preview
    dataset_size = len(dataset)
    
    if dataset_size == 0:
        st.info("Dataset is empty. Generate samples first.")
        return
    
    # Pagination
    page_size = st.selectbox("Samples per page", [10, 25, 50, 100], index=1)
    page_number = st.session_state.get('dataset_page', 1)
    total_pages = (dataset_size + page_size - 1) // page_size
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⏮️ First"):
            page_number = 1
    with col2:
        page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=page_number)
    page_number = int(page_number)
    with col3:
        if st.button("⏭️ Last"):
            page_number = total_pages
    
    # Update session state
    st.session_state.dataset_page = page_number
    
    # Display dataset stats
    st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin: 0 0 1rem 0;">📊 Dataset Stats</h4>
            <p><strong>Total Samples:</strong> {dataset_size}</p>
            <p><strong>Page Size:</strong> {page_size}</p>
            <p><strong>Total Pages:</strong> {total_pages}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display current page of dataset
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    page_data = dataset[start_idx:end_idx]
    
    delete_selection = []
    for local_i, sample in enumerate(page_data):
        global_idx = start_idx + local_i
        with st.expander(f"Sample {global_idx + 1}"):
            st.markdown(f"**👤 User:** {sample['messages'][1]['content']}")
            st.markdown(f"**🎭 Assistant:** {sample['messages'][2]['content']}")
            if st.checkbox("Mark for deletion", key=f"del_{global_idx}"):
                delete_selection.append(global_idx)
    
    if delete_selection and st.button("🗑️ Delete Selected Samples"):
        if st.session_state.dataset_manager.delete_samples(st.session_state.current_character, delete_selection):
            # Reload dataset preview
            st.session_state.dataset_preview = st.session_state.dataset_manager.load_dataset(st.session_state.current_character)
            st.success(f"Deleted {len(delete_selection)} samples.")
            st.rerun()
    
    # Quality analysis block
    if st.checkbox("Show quality analysis", value=False):
        stats = st.session_state.dataset_manager.analyze_dataset_quality(dataset)
        if stats:
            st.json(stats)

# Main app function
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
    elif selected_page == "📚 Dataset Explorer":
        page_dataset_explorer()
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