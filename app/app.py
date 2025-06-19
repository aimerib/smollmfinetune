import asyncio
from asyncio.log import logger
import os

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
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
            options=["📁 Character Upload", "🔍 Dataset Preview", "📚 Dataset Explorer", "⚙️ Training Config", "📊 Training Dashboard", "🧪 Model Testing", "⚔️ Model Comparison", "🔧 Model Management"],
            icons=["upload", "search", "table", "gear", "graph-up", "flask", "shuffle", "tools"],
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
        
        # Training status (ensure it's always up to date)
        if hasattr(st.session_state, 'training_manager'):
            current_manager_status = st.session_state.training_manager.get_training_status()
            if current_manager_status != st.session_state.training_status:
                st.session_state.training_status = current_manager_status
        
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
        
        # CharacterCore conversion section
        if st.session_state.current_character:
            st.markdown("---")
            st.markdown("### 🔄 Convert to CharacterCore Format")
            
            col_convert1, col_convert2 = st.columns([2, 1])
            
            with col_convert1:
                st.markdown("""
                **Convert your SillyTavern card to the new CharacterCore format with:**
                - 🧠 Big Five personality traits (auto-estimated)
                - 📝 Structured appearance and backstory
                - 🎯 Extracted goals and relationships
                - 🏷️ Automatic tagging
                """)
            
            with col_convert2:
                if st.button("🔄 Convert to CharacterCore", type="primary", use_container_width=True):
                    with st.spinner("Converting character card..."):
                        try:
                            # Use existing character manager from session state
                            char_manager = st.session_state.character_manager
                            
                            # For now, show a preview of the conversion process
                            st.info("🔄 Character conversion functionality is available. This will:")
                            st.markdown("""
                            1. 🧠 Analyze character with LLM to estimate Big Five traits
                            2. 📝 Extract appearance and backstory elements
                            3. 🎯 Identify goals and relationships
                            4. 💾 Save in new folder structure:
                               ```
                               characters/[name]/
                               ├── character_core.json
                               ├── mes_example.txt
                               └── assets/
                               ```
                            """)
                            
                            # Show a preview of what would be converted
                            st.markdown("**Preview of converted structure:**")
                            char_name = st.session_state.current_character.get('name', 'Unknown')
                            char_desc = st.session_state.current_character.get('description', '')
                            char_personality = st.session_state.current_character.get('personality', '')
                            
                            preview_data = {
                                "name": char_name,
                                "description": char_desc[:100] + "..." if len(char_desc) > 100 else char_desc,
                                "personality_traits": {
                                    "openness": "0.7 (estimated from description)",
                                    "conscientiousness": "0.6 (estimated from personality)",
                                    "extraversion": "0.8 (estimated from examples)",
                                    "agreeableness": "0.7 (estimated from traits)",
                                    "neuroticism": "0.3 (estimated from behavior)"
                                },
                                "goals": ["[Auto-extracted from description and personality]"],
                                "relationships": ["[Auto-extracted from character context]"],
                                "appearance": "[Extracted appearance details]",
                                "backstory": "[Derived backstory elements]",
                                "tags": ["[Auto-generated based on content]"],
                                "world": char_manager.get_current_world() or "Default World"
                            }
                            
                            st.json(preview_data)
                            
                            # Note about full implementation
                            st.warning("""
                            **Note**: Full conversion with LLM analysis requires an async context. 
                            The complete functionality is implemented in the CharacterManager.import_sillytavern_card() method.
                            This preview shows the structure that would be created.
                            """)
                            
                        except Exception as e:
                            st.error(f"❌ Error during conversion preview: {str(e)}")
                            # Add some debug info
                            st.error(f"Debug info: {type(e).__name__}: {str(e)}")
    
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
        
        # Add CharacterCore info
        if st.session_state.current_character:
            st.markdown("""
                <div style="background: rgba(34, 197, 94, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(34, 197, 94, 0.2); margin-top: 1rem;">
                    <h4 style="color: #22c55e; margin-top: 0;">🚀 CharacterCore Format</h4>
                    <ul style="color: #cbd5e1; font-size: 0.9rem;">
                        <li>Structured Big Five personality traits</li>
                        <li>Organized character data for better training</li>
                        <li>World-based character organization</li>
                        <li>Enhanced prompt generation</li>
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
    
    # Streamlined generation mode tabs
    generation_tab, interactive_tab, fast_tab, slow_tab, quality_tab = st.tabs([
        "📊 Overview", 
        "🤝 Interactive Generation (Primary)",
        "⚡ Fast Mode (Templated)",
        "🔬 Slow Mode (AI-Curated)",
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
        ### 🎯 Streamlined Generation Methods
        
        🚀 **New simplified approach with 3 focused modes:**
        """)
        
        st.success("""
        **🤝 Interactive Generation (Primary)**: Collaborative batch-by-batch generation with real-time feedback
        - ✨ Best for: Perfect quality control, learning what works for your character
        - ⏱️ User-guided (as fast or slow as you want)
        
        **⚡ Fast Mode (Templated)**: Template-based questions + LLM paraphrasing + character responses
        - ✨ Best for: Parameter tuning, quick experiments, baseline datasets
        - ⏱️ Fast (5-15 minutes)
        
        **🔬 Slow Mode (AI-Curated)**: LLM creates questions → Judge filters → Character responds → Quality control
        - ✨ Best for: Hands-off exploration, discovering optimal parameters  
        - ⏱️ Thorough (20-60 minutes)
        """)
        
        st.info("""
        🎯 **All modes maintain temporal (past/present/future) and categorical (personal/emotional/casual/worldbuilding/nsfw) distributions**
        
        📈 **Recommended workflow:**
        1. Start with **Interactive Mode** to understand your character and find the right parameters
        2. Use **Fast Mode** for quick iteration and parameter testing
        3. Use **Slow Mode** for hands-off generation once you know what works
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
    
    with fast_tab:
        st.markdown("### ⚡ Fast Mode (Templated Generation)")
        
        st.info("""
        🎯 **Fast Mode - Template-Based**:
        - Uses predefined templates + LLM paraphrasing for clean questions
        - Character responds directly to paraphrased questions
        - Maintains temporal and categorical distributions
        - Good for: Parameter tuning, quick experiments, baseline datasets
        """)
        
        # Fast mode settings
        st.markdown("#### ⚙️ Fast Mode Configuration")
        
        with st.form("fast_generation"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                fast_num_samples = st.slider(
                    "Number of samples",
                    min_value=20,
                    max_value=500,
                    value=100,
                    step=20,
                    help="Total samples to generate using template approach"
                )
                
                fast_temperature = st.slider(
                    "Temperature",
                    min_value=0.3,
                    max_value=1.2,
                    value=0.7,
                    step=0.1,
                    help="Controls randomness in paraphrasing and responses"
                )
                
                fast_paraphrase_strength = st.slider(
                    "Paraphrasing Strength",
                    min_value=0.5,
                    max_value=1.5,
                    value=0.8,
                    step=0.1,
                    help="How much to vary the template questions (higher = more variation)"
                )
            
            with col_b:
                fast_max_tokens = st.slider(
                    "Max tokens per response",
                    min_value=100,
                    max_value=800,
                    value=300,
                    step=50,
                    help="Maximum length of character responses"
                )
                
                fast_use_custom_system = st.checkbox(
                    "Apply custom system prompt", 
                    value=False, 
                    help="Override temporal prompts with custom system prompt"
                )
                
                fast_distribution_enforcement = st.checkbox(
                    "Enforce Distribution Balance",
                    value=True,
                    help="Ensure equal representation across temporal and categorical buckets"
                )
            
            if fast_use_custom_system:
                fast_system_prompt = st.text_area(
                    "Custom System Prompt",
                    placeholder="You are a helpful assistant...",
                    height=100,
                    key="fast_system_prompt"
                )
            else:
                fast_system_prompt = None
                st.info("Using temporal distribution system prompts")
            
            # Generate button
            fast_generate_button = st.form_submit_button(
                "⚡ Generate Fast Dataset", 
                use_container_width=True,
                type="primary"
            )
        
        if fast_generate_button:
            st.session_state._generating_dataset = True
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    dataset = loop.run_until_complete(
                        st.session_state.dataset_manager.generate_fast_templated_dataset(
                            st.session_state.current_character,
                            num_samples=fast_num_samples,
                            temperature=fast_temperature,
                            max_tokens=fast_max_tokens,
                            paraphrase_strength=fast_paraphrase_strength,
                            custom_system_prompt=fast_system_prompt if fast_use_custom_system else None,
                            enforce_distribution=fast_distribution_enforcement,
                            progress_callback=lambda p: progress_bar.progress(p),
                            append_to_existing=True
                        )
                    )
                finally:
                    loop.close()
                
                st.session_state.dataset_preview = dataset
                st.session_state.dataset_metadata = {
                    'generation_method': 'fast_templated',
                    'paraphrase_strength': fast_paraphrase_strength,
                    'distribution_enforced': fast_distribution_enforcement,
                    'system_prompt_config': {
                        'type': 'custom' if fast_use_custom_system else 'temporal',
                        'prompt': fast_system_prompt if fast_use_custom_system else None
                    }
                }
                
                progress_bar.progress(1.0)
                status_text.text("Fast templated generation complete!")
                st.success(f"⚡ Generated {len(dataset)} samples using template approach!")
                
                st.session_state._generating_dataset = False
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error in fast generation: {str(e)}")
                st.session_state._generating_dataset = False
    
    with slow_tab:
        st.markdown("### 🔬 Slow Mode (AI-Curated Generation)")
        
        st.info("""
        🎯 **Slow Mode - AI-Curated Pipeline**:
        - LLM creates custom questions based on character analysis
        - Judge LLM filters out poor quality questions
        - Character generates responses to approved questions
        - Judge LLM evaluates response quality and regenerates if needed
        - Maintains strict temporal and categorical distributions
        - Good for: Parameter exploration, high-quality discovery, research
        """)
        
        # Slow mode settings
        st.markdown("#### 🔬 AI-Curated Configuration")
        
        with st.form("slow_generation"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                slow_num_samples = st.slider(
                    "Target final samples",
                    min_value=20,
                    max_value=300,
                    value=60,
                    step=10,
                    help="Final dataset size after all filtering and quality checks"
                )
                
                slow_generation_multiplier = st.slider(
                    "Generation Multiplier",
                    min_value=2.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.5,
                    help="Generate N×target questions for filtering (higher = more selective)"
                )
                
                slow_quality_threshold = st.slider(
                    "Quality Threshold",
                    min_value=0.6,
                    max_value=0.95,
                    value=0.75,
                    step=0.05,
                    help="Minimum quality score for question and response acceptance"
                )
            
            with col_b:
                slow_max_regenerations = st.slider(
                    "Max Regeneration Attempts",
                    min_value=1,
                    max_value=5,
                    value=2,
                    help="How many times to retry generating better responses"
                )
                
                slow_distribution_strictness = st.slider(
                    "Distribution Strictness",
                    min_value=0.7,
                    max_value=1.0,
                    value=0.85,
                    step=0.05,
                    help="How strictly to enforce temporal/categorical balance (1.0 = perfect balance)"
                )
                
                slow_use_custom_system = st.checkbox(
                    "Apply custom system prompt", 
                    value=False, 
                    help="Override temporal prompts with custom system prompt"
                )
            
            if slow_use_custom_system:
                slow_system_prompt = st.text_area(
                    "Custom System Prompt",
                    placeholder="You are a helpful assistant...",
                    height=100,
                    key="slow_system_prompt"
                )
            else:
                slow_system_prompt = None
                st.info("Using adaptive temporal system prompts")
            
            # Estimated processing
            estimated_questions = int(slow_num_samples * slow_generation_multiplier)
            estimated_time = estimated_questions / 5 / 60  # Rough estimate for slow processing
            
            st.markdown(f"""
            **📊 AI-Curated Generation Plan:**
            - Generate: ~{estimated_questions} diverse questions
            - Filter: Down to ~{slow_num_samples} high-quality questions
            - Response generation: {slow_num_samples} character responses
            - Quality evaluation: Judge each response (regenerate if needed)
            - Estimated time: ~{estimated_time:.1f} minutes
            """)
            
            # Generate button
            slow_generate_button = st.form_submit_button(
                "🔬 Generate AI-Curated Dataset", 
                use_container_width=True,
                type="primary"
            )
        
        if slow_generate_button:
            st.session_state._generating_dataset = True
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    dataset = loop.run_until_complete(
                        st.session_state.dataset_manager.generate_slow_curated_dataset(
                            st.session_state.current_character,
                            target_samples=slow_num_samples,
                            generation_multiplier=slow_generation_multiplier,
                            quality_threshold=slow_quality_threshold,
                            max_regenerations=slow_max_regenerations,
                            distribution_strictness=slow_distribution_strictness,
                            custom_system_prompt=slow_system_prompt if slow_use_custom_system else None,
                            progress_callback=lambda p: progress_bar.progress(p),
                            stage_callback=lambda stage: status_text.text(stage),
                            append_to_existing=True
                        )
                    )
                finally:
                    loop.close()
                
                st.session_state.dataset_preview = dataset
                st.session_state.dataset_metadata = {
                    'generation_method': 'slow_curated',
                    'generation_multiplier': slow_generation_multiplier,
                    'quality_threshold': slow_quality_threshold,
                    'max_regenerations': slow_max_regenerations,
                    'distribution_strictness': slow_distribution_strictness,
                    'system_prompt_config': {
                        'type': 'custom' if slow_use_custom_system else 'temporal',
                        'prompt': slow_system_prompt if slow_use_custom_system else None
                    }
                }
                
                progress_bar.progress(1.0)
                status_text.text("AI-curated generation complete!")
                st.success(f"🔬 Generated {len(dataset)} AI-curated samples with rigorous quality control!")
                
                # Show curation statistics if available
                if hasattr(st.session_state.dataset_manager, 'curation_stats'):
                    stats = st.session_state.dataset_manager.curation_stats
                    st.info(f"""
                    📊 **Curation Statistics**:
                    - Questions generated: {stats.get('questions_generated', 0)}
                    - Questions approved: {stats.get('questions_approved', 0)}
                    - Responses regenerated: {stats.get('responses_regenerated', 0)}
                    - Final quality score: {stats.get('final_avg_quality', 0):.2f}
                    """)
                
                st.session_state._generating_dataset = False
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error in slow mode generation: {str(e)}")
                st.session_state._generating_dataset = False
    
    with interactive_tab:
        st.markdown("### 🤝 Interactive Dataset Generation")
        
        st.info("""
        🎯 **Interactive Mode**:
        - Generate in small batches (20 samples at a time)
        - Rate and provide feedback on each batch
        - AI learns from your feedback to improve subsequent generations
        - Perfect collaboration between human creativity and AI efficiency
        """)
        
        # Interactive generation state management
        interactive_key = f"interactive_state_{st.session_state.current_character.get('name', 'unknown')}"
        if interactive_key not in st.session_state:
            st.session_state[interactive_key] = {
                'current_batch': [],
                'approved_samples': [],
                'rejected_samples': [],
                'feedback_tags': {},
                'generation_round': 0,
                'target_total': 80,
                'batch_size': 20,
                'is_generating': False,
                'few_shot_examples': [],
                'negative_patterns': []
            }
        
        interactive_state = st.session_state[interactive_key]
        
        # Progress overview
        st.markdown("#### 📊 Interactive Generation Progress")
        
        current_total = len(interactive_state['approved_samples'])
        target_total = interactive_state['target_total']
        progress_pct = min(100, (current_total / target_total) * 100) if target_total > 0 else 0
        
        col_prog1, col_prog2, col_prog3, col_prog4 = st.columns(4)
        
        with col_prog1:
            st.metric("Approved Samples", current_total)
        with col_prog2:
            st.metric("Target Total", target_total)
        with col_prog3:
            st.metric("Progress", f"{progress_pct:.1f}%")
        with col_prog4:
            st.metric("Generation Round", interactive_state['generation_round'])
        
        # Progress bar
        st.progress(progress_pct / 100, text=f"Dataset Progress: {current_total}/{target_total} samples")
        
        # Configuration (only show if not started)
        interactive_sampling_config = None
        if interactive_state['generation_round'] == 0:
            st.markdown("#### ⚙️ Interactive Generation Settings")
            
            col_cfg1, col_cfg2 = st.columns(2)
            
            with col_cfg1:
                target_total = st.slider(
                    "Target Dataset Size",
                    min_value=20,
                    max_value=500,
                    value=interactive_state['target_total'],
                    step=20,
                    help="Total number of approved samples you want to end up with"
                )
                interactive_state['target_total'] = target_total
                
                batch_size = st.slider(
                    "Batch Size",
                    min_value=10,
                    max_value=30,
                    value=interactive_state['batch_size'],
                    step=5,
                    help="Number of samples to generate and review at once"
                )
                interactive_state['batch_size'] = batch_size
            
            with col_cfg2:
                # Import sampling config
                from utils.sampling_config import render_sampling_config_ui, SamplingConfig
                
                interactive_default_config = SamplingConfig(
                    temperature=0.9,  # Higher for more creativity in interactive mode
                    top_p=0.95,
                    max_tokens=300,
                    repetition_penalty=1.05,
                )
                
                interactive_sampling_config = render_sampling_config_ui(
                    current_config=interactive_default_config,
                    key_prefix="interactive_gen",
                    use_expander=False
                )
        
        # Current batch review interface
        if interactive_state['current_batch']:
            st.markdown("---")
            st.markdown("#### 📝 Review Current Batch")
            st.info(f"Rate each sample below. Your feedback will improve the next generation batch.")
            
            # Batch review interface
            for i, sample in enumerate(interactive_state['current_batch']):
                sample_key = f"sample_{interactive_state['generation_round']}_{i}"
                
                st.markdown(f"##### Sample {i+1}")
                
                # Display the conversation
                messages = sample['messages']
                for msg in messages:
                    if msg['role'] == 'system' and msg['content']:
                        st.markdown(f"**🔧 System:** {msg['content'][:100]}...")
                    elif msg['role'] == 'user':
                        st.markdown(f"**👤 User:** {msg['content']}")
                    elif msg['role'] == 'assistant':
                        st.markdown(f"**🎭 Assistant:** {msg['content']}")
                
                # Rating interface
                col_rate1, col_rate2, col_rate3, col_rate4 = st.columns([1, 1, 1, 2])
                
                with col_rate1:
                    if st.button("👍 Good", key=f"approve_{sample_key}", use_container_width=True):
                        interactive_state['approved_samples'].append(sample)
                        interactive_state['few_shot_examples'].append({
                            'user': messages[1]['content'],
                            'assistant': messages[2]['content']
                        })
                        # Keep only best 10 few-shot examples
                        if len(interactive_state['few_shot_examples']) > 10:
                            interactive_state['few_shot_examples'] = interactive_state['few_shot_examples'][-10:]
                        
                        # Remove from current batch
                        interactive_state['current_batch'] = [s for j, s in enumerate(interactive_state['current_batch']) if j != i]
                        st.rerun()
                
                with col_rate2:
                    if st.button("👎 Bad", key=f"reject_{sample_key}", use_container_width=True):
                        interactive_state['rejected_samples'].append(sample)
                        # Add to negative patterns
                        response_text = messages[2]['content']
                        interactive_state['negative_patterns'].append(response_text[:200])
                        
                        # Remove from current batch
                        interactive_state['current_batch'] = [s for j, s in enumerate(interactive_state['current_batch']) if j != i]
                        st.rerun()
                
                with col_rate3:
                    # Quick feedback tags
                    tag_options = ["🎭 Out of Character", "😴 Boring", "🤖 Too AI-like", "📝 Poor Quality", "🔄 Repetitive"]
                    selected_tag = st.selectbox(
                        "Flag issue",
                        ["None"] + tag_options,
                        key=f"tag_{sample_key}",
                        label_visibility="collapsed"
                    )
                    
                    if selected_tag != "None":
                        if sample_key not in interactive_state['feedback_tags']:
                            interactive_state['feedback_tags'][sample_key] = []
                        if selected_tag not in interactive_state['feedback_tags'][sample_key]:
                            interactive_state['feedback_tags'][sample_key].append(selected_tag)
                
                with col_rate4:
                    # Custom feedback
                    custom_feedback = st.text_input(
                        "Custom feedback (optional)",
                        key=f"feedback_{sample_key}",
                        placeholder="What's wrong with this sample?",
                        label_visibility="collapsed"
                    )
                    
                    if custom_feedback:
                        interactive_state['feedback_tags'][sample_key] = interactive_state['feedback_tags'].get(sample_key, []) + [f"Custom: {custom_feedback}"]
                
                st.markdown("---")
            
            # Batch actions
            col_batch1, col_batch2, col_batch3 = st.columns(3)
            
            with col_batch1:
                if st.button("✅ Approve All Remaining", use_container_width=True):
                    for sample in interactive_state['current_batch']:
                        interactive_state['approved_samples'].append(sample)
                        messages = sample['messages']
                        interactive_state['few_shot_examples'].append({
                            'user': messages[1]['content'],
                            'assistant': messages[2]['content']
                        })
                    interactive_state['current_batch'] = []
                    st.rerun()
            
            with col_batch2:
                if st.button("❌ Reject All Remaining", use_container_width=True):
                    for sample in interactive_state['current_batch']:
                        interactive_state['rejected_samples'].append(sample)
                    interactive_state['current_batch'] = []
                    st.rerun()
            
            with col_batch3:
                if st.button("🔄 Regenerate Batch", use_container_width=True):
                    interactive_state['current_batch'] = []
                    interactive_state['is_generating'] = True
                    st.rerun()
        
        # Generation controls
        st.markdown("---")
        st.markdown("#### 🎮 Generation Controls")
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        
        # Check if we need more samples
        remaining_needed = max(0, interactive_state['target_total'] - len(interactive_state['approved_samples']))
        
        with col_ctrl1:
            # Generate next batch button
            disabled = interactive_state['is_generating'] or (remaining_needed == 0)
            button_text = "🚀 Start Interactive Generation" if interactive_state['generation_round'] == 0 else f"➡️ Generate Next Batch ({min(interactive_state['batch_size'], remaining_needed)} samples)"
            
            if st.button(button_text, disabled=disabled or bool(interactive_state['current_batch']), use_container_width=True):
                interactive_state['is_generating'] = True
                st.rerun()
        
        with col_ctrl2:
            # Auto-complete button (only show if we have some approved samples)
            if len(interactive_state['approved_samples']) >= 20:
                remaining = interactive_state['target_total'] - len(interactive_state['approved_samples'])
                if remaining > 0 and st.button(f"🤖 Auto-Complete ({remaining} samples)", use_container_width=True):
                    st.session_state[f"{interactive_key}_auto_complete"] = True
                    st.rerun()
        
        with col_ctrl3:
            # Finish early button
            if len(interactive_state['approved_samples']) > 0:
                if st.button("🏁 Finish with Current Samples", use_container_width=True):
                    # Save the approved samples as the dataset
                    st.session_state.dataset_preview = interactive_state['approved_samples']
                    st.session_state.dataset_metadata = {
                        'generation_method': 'interactive',
                        'interactive_rounds': interactive_state['generation_round'],
                        'system_prompt_config': {'type': 'temporal'}
                    }
                    st.success(f"✅ Interactive generation complete! Saved {len(interactive_state['approved_samples'])} samples.")
                    # Reset interactive state
                    del st.session_state[interactive_key]
                    st.rerun()
        
        # Handle generation
        if interactive_state['is_generating'] and not interactive_state['current_batch']:
            # ✅ FIX: Set generation state to prevent UI interference
            st.session_state._generating_dataset = True
            
            with st.spinner(f"Generating batch {interactive_state['generation_round'] + 1}..."):
                try:
                    samples_to_generate = min(interactive_state['batch_size'], remaining_needed)
                    
                    # Prepare generation parameters with feedback
                    if interactive_sampling_config is not None:
                        sampling_kwargs = interactive_sampling_config.to_dict()
                    else:
                        sampling_kwargs = {}
                    generation_params = {
                        'num_samples': samples_to_generate,
                        'progress_callback': lambda p: None,  # No progress bar for small batches
                        'append_to_existing': False,  # Generate fresh batch
                        'extra_quality': True,  # Always use quality for interactive
                        'few_shot_examples': interactive_state['few_shot_examples'][-5:],  # Use recent good examples
                        'negative_patterns': interactive_state['negative_patterns'][-10:],  # Use recent bad patterns
                        **sampling_kwargs
                    }
                    
                    # Generate the batch
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        batch = loop.run_until_complete(
                            st.session_state.dataset_manager.generate_interactive_batch(
                                st.session_state.current_character,
                                **generation_params
                            )
                        )
                    finally:
                        loop.close()
                    
                    # Update state
                    interactive_state['current_batch'] = batch
                    interactive_state['generation_round'] += 1
                    interactive_state['is_generating'] = False
                    
                    # ✅ FIX: Clear generation state
                    st.session_state._generating_dataset = False
                    
                    st.success(f"✅ Generated batch {interactive_state['generation_round']} with {len(batch)} samples!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating batch: {str(e)}")
                    interactive_state['is_generating'] = False
                    # ✅ FIX: Always clear generation state on error
                    st.session_state._generating_dataset = False
        
        # Handle auto-completion
        auto_complete_key = f"{interactive_key}_auto_complete"
        if st.session_state.get(auto_complete_key, False):
            remaining = interactive_state['target_total'] - len(interactive_state['approved_samples'])
            
            with st.spinner(f"Auto-completing remaining {remaining} samples using writer-judge loop..."):
                try:
                    # Use the enhanced generation with all accumulated feedback
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        remaining_samples = loop.run_until_complete(
                            st.session_state.dataset_manager.generate_dataset(
                                st.session_state.current_character,
                                num_samples=remaining,
                                progress_callback=lambda p: None,
                                append_to_existing=False,
                                extra_quality=True,
                                few_shot_examples=interactive_state['few_shot_examples'],
                                negative_patterns=interactive_state['negative_patterns'],
                                **interactive_sampling_config.to_dict()
                            )
                        )
                    finally:
                        loop.close()
                    
                    # Combine with approved samples
                    final_dataset = interactive_state['approved_samples'] + remaining_samples
                    
                    # Save the complete dataset
                    st.session_state.dataset_preview = final_dataset
                    st.session_state.dataset_metadata = {
                        'generation_method': 'interactive_auto_complete',
                        'interactive_rounds': interactive_state['generation_round'],
                        'human_approved_samples': len(interactive_state['approved_samples']),
                        'auto_generated_samples': len(remaining_samples),
                        'system_prompt_config': {'type': 'temporal'}
                    }
                    
                    st.success(f"🎉 Interactive generation complete! Final dataset: {len(final_dataset)} samples ({len(interactive_state['approved_samples'])} human-approved + {len(remaining_samples)} auto-generated)")
                    
                    # Reset state
                    del st.session_state[interactive_key]
                    del st.session_state[auto_complete_key]
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error in auto-completion: {str(e)}")
                    del st.session_state[auto_complete_key]
        
        # Show feedback summary
        if interactive_state['generation_round'] > 0:
            with st.expander("📊 Feedback Summary", expanded=False):
                col_sum1, col_sum2 = st.columns(2)
                
                with col_sum1:
                    st.markdown("**Positive Examples (Few-shot):**")
                    if interactive_state['few_shot_examples']:
                        for i, example in enumerate(interactive_state['few_shot_examples'][-3:], 1):
                            st.markdown(f"*{i}.* {example['user'][:50]}... → {example['assistant'][:50]}...")
                    else:
                        st.info("No positive examples yet")
                
                with col_sum2:
                    st.markdown("**Negative Patterns to Avoid:**")
                    if interactive_state['negative_patterns']:
                        for i, pattern in enumerate(interactive_state['negative_patterns'][-3:], 1):
                            st.markdown(f"*{i}.* {pattern[:100]}...")
                    else:
                        st.info("No negative patterns identified yet")
                
                # Tag summary
                if interactive_state['feedback_tags']:
                    st.markdown("**Common Issues Flagged:**")
                    all_tags = []
                    for tags_list in interactive_state['feedback_tags'].values():
                        all_tags.extend(tags_list)
                    
                    from collections import Counter
                    tag_counts = Counter(all_tags)
                    for tag, count in tag_counts.most_common(5):
                        st.write(f"• {tag}: {count} times")
        
        # Reset generation button
        if interactive_state['generation_round'] > 0:
            st.markdown("---")
            if st.button("🔄 Reset Interactive Generation", use_container_width=True):
                if st.checkbox("I understand this will lose all progress"):
                    del st.session_state[interactive_key]
                    st.rerun()
    
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
    
    finetune_method = st.session_state.get('finetune_method', 'lora')
    # Check if a profile needs to be applied
    if 'profile_to_apply' in st.session_state and st.session_state.profile_to_apply:
        profile = st.session_state.profile_to_apply
        
        # Store profile values in a persistent session state key (don't delete immediately)
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
    # Note: Don't delete training_form_defaults here - keep it for next page load

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
        
        # st.markdown("### Logging Configuration")
        # Note: Main logging frequency is set in Advanced Settings below
        
        # early_stopping_patience = st.slider(
        #     "Early Stopping Patience",
        #     min_value=1,
        #     max_value=10,
        #     value=defaults.get("early_stopping_patience", 3),
        #     help="Number of evaluation steps without improvement before stopping"
        # ) if enable_validation else 3
        
        # # Force GPU option - not supported yet
        # force_gpu = st.checkbox(
        #     "Force GPU Usage",
        #     value=defaults.get("force_gpu", False),
        #     help="Override device selection to use GPU (if available)"
        # )
        
        # Store advanced config in session state
        st.session_state.advanced_training_config = {
            'enable_validation': enable_validation,
            'adaptive_lora': adaptive_lora,
            'enhanced_quality_filtering': enhanced_filtering,
            'enable_tensorboard': enable_tensorboard,
            'enable_wandb': enable_wandb,
            'force_gpu': False # Force GPU usage is not supported yet
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
        
        # Select fine-tuning method outside the form to allow UI updates
        finetune_method = st.radio(
            "Fine-tuning Method",
            ("LoRA", "RSLoRA", "DoRA"),
            horizontal=True,
            index=["lora", "rslora", "dora"].index(defaults.get("finetune_method", "lora")),
            help="Choose between LoRA, RSLoRA, and DoRA. RSLoRA uses rank-stabilized scaling. DoRA offers more precise training."
        ).lower()
        
        # Store the selected method in session state
        st.session_state.finetune_method = finetune_method
        
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
            
            # Store the selected method in the form
            st.text(f"Selected method: {finetune_method.upper()}")
            
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
            
            # Method-specific options
            if finetune_method == "dora":
                st.markdown("#### DoRA Settings")
                ephemeral_gpu_offload = st.checkbox(
                    "Enable Ephemeral GPU Offload",
                    value=defaults.get("ephemeral_gpu_offload", False),
                    help="Speed up DoRA training with temporary VRAM overhead (CUDA only)"
                )
                st.info("💡 DoRA works best with low dropout (0.0-0.05) and is optimized for eval mode")
            else:
                ephemeral_gpu_offload = False
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
                logging_steps = st.slider("Log Every N Steps", 1, 100, defaults.get("logging_steps", 5), step=1, help="Controls how often training metrics are logged and displayed")
                eval_steps = st.slider("Evaluation Steps", 1, 100, defaults.get("eval_steps", 10), step=1)
                early_stopping_patience = st.slider(
                    "Early Stopping Patience",
                    min_value=1,
                    max_value=10,
                    value=defaults.get("early_stopping_patience", 3),
                    help="Number of evaluation steps without improvement before stopping"
                ) if enable_validation else 3
                max_steps_override = st.number_input(
                    "Override Total Training Steps (0 = auto)",
                    min_value=0,
                    max_value=50000,
                    value=defaults.get("max_steps_override", 0),
                    step=100,
                    help="Manually set the total number of optimisation steps if you need finer control. Leave at 0 to use the computed value."
                )
            
            # Form buttons
            col_form1, col_form2 = st.columns(2)
            with col_form1:
                start_training = st.form_submit_button("🚀 Start Training", use_container_width=True)
            with col_form2:
                save_profile = st.form_submit_button("💾 Save Profile", use_container_width=True, help="Save current settings as a profile")
    
    with col2:
        st.markdown("### Training Recommendations")
        
        # Profile I/O section
        st.markdown("#### Profile Management")
        st.info("💡 Use the '💾 Save Profile' button in the training form below to save current settings.")
        
        # Profile controls
        profile_cols = st.columns(2)
        with profile_cols[0]:
            uploaded_profile = st.file_uploader(
                "Load Profile", 
                type=['json'], 
                help="Upload a saved training profile"
            )
        with profile_cols[1]:
            if st.button("🔄 Reset to Defaults", use_container_width=True, help="Clear any applied profile and reset to default values"):
                if 'training_form_defaults' in st.session_state:
                    del st.session_state['training_form_defaults']
                st.success("✅ Reset to default values!")

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
                # Store profile to be applied on next render
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
    
    # Handle profile saving
    if 'save_profile' in locals() and save_profile:
        try:
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
                    'early_stopping_patience': early_stopping_patience,
                    'logging_steps': logging_steps,
                    'eval_steps': eval_steps,
                    'max_steps_override': int(max_steps_override) if max_steps_override else 0,
                    'ephemeral_gpu_offload': ephemeral_gpu_offload,
                }
            }
            
            # Create profiles directory if it doesn't exist
            profiles_dir = Path("profiles")
            profiles_dir.mkdir(exist_ok=True)
            
            # Save the profile
            save_path = profiles_dir / f"{char_name}_profile.json"
            with open(save_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            
            st.success(f"✅ Profile saved: {save_path.name}")
            
        except Exception as e:
            st.error(f"❌ Error saving profile: {e}")
    
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
            # Method-specific parameters
            'use_rslora': finetune_method == 'rslora',
            'use_dora': finetune_method == 'dora',
            'ephemeral_gpu_offload': ephemeral_gpu_offload if finetune_method == 'dora' else False,
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
            
            # ✅ NEW: Display actual configuration values being used
            with st.expander("🔍 **Actual Training Configuration Used**", expanded=True):
                st.info("💡 **Tip:** This shows the exact values the training process will use, which may differ from UI defaults due to fallbacks or advanced settings.")
                
                config_col1, config_col2, config_col3 = st.columns(3)
                
                with config_col1:
                    st.markdown("##### 📊 **Core Training**")
                    # Show the actual logging frequency that will be used
                    actual_log_freq = config.get('logging_steps', advanced_config.get('logging_steps', 10))
                    st.write(f"**Log Every N Steps:** `{actual_log_freq}`")
                    st.write(f"**Learning Rate:** `{config.get('learning_rate', 2e-4)}`")
                    st.write(f"**Batch Size:** `{config.get('batch_size', 2)}`")
                    st.write(f"**Gradient Accumulation:** `{config.get('gradient_accumulation_steps', 2)}`")
                    st.write(f"**Max Steps Override:** `{config.get('max_steps_override', 'None (use calculated)')}`")
                    
                with config_col2:
                    st.markdown("##### ⚙️ **Method & Parameters**")
                    st.write(f"**Method:** `{config.get('finetune_method', 'lora').upper()}`")
                    st.write(f"**LoRA Rank (r):** `{config.get('lora_r', 16)}`")
                    st.write(f"**LoRA Alpha:** `{config.get('lora_alpha', config.get('lora_r', 16))}`")
                    st.write(f"**LoRA Dropout:** `{config.get('lora_dropout', 0.1)}`")
                    st.write(f"**Use RSLoRA:** `{config.get('use_rslora', False)}`")
                    st.write(f"**Use DoRA:** `{config.get('use_dora', False)}`")
                    
                with config_col3:
                    st.markdown("##### 🎯 **Advanced Settings**")
                    st.write(f"**Save Every N Steps:** `{config.get('save_steps', 50)}`")
                    st.write(f"**Max Samples:** `{config.get('max_samples', 'All')}`")
                    st.write(f"**Include System Prompts:** `{config.get('include_system_prompts', False)}`")
                    st.write(f"**FP16:** `{config.get('fp16', False)}`")
                    
                # Show any discrepancies as warnings
                ui_log_steps = config.get('logging_steps')
                advanced_log_steps = advanced_config.get('logging_steps', 10)
                if ui_log_steps and ui_log_steps != actual_log_freq:
                    st.warning(f"⚠️ **Logging Frequency Discrepancy:** UI shows `{ui_log_steps}` but training will use `{actual_log_freq}` (from advanced config)")
            
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
        
        # Show model compatibility info
        if not selected_model.startswith("Base:"):
            metadata = st.session_state.inference_manager.get_model_metadata(selected_model)
            if metadata:
                required_base_model = metadata.get('base_model')
                training_method = metadata.get('training_method', 'lora')
                use_dora = metadata.get('use_dora', False)
                use_rslora = metadata.get('use_rslora', False)
                
                # Display model info
                method_display = "DoRA" if use_dora else "RSLoRA" if use_rslora else "LoRA"
                
                st.info(f"🔍 **Model Info**: {method_display} trained on `{required_base_model}`")
                
                # Check compatibility
                current_base = st.session_state.inference_manager.base_model
                if required_base_model != current_base:
                    st.warning(f"⚠️ Model trained on `{required_base_model}` but current inference base is `{current_base}`. Will auto-switch for compatibility.")
            else:
                st.warning("⚠️ No metadata available for this model. It may be from an older training run.")
        
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
    # total_samples = sum(len(v) for v in buckets.values())
    
    overview_cols = st.columns(4)
    overview_cols[0].metric("Total Active Samples", sum(len(v) for k, v in buckets.items() if k != 'quarantined' and v is not None))
    overview_cols[1].metric("Quarantined Samples", len(buckets.get('quarantined', [])))
    overview_cols[2].metric("Total Buckets", len(buckets))

    with overview_cols[3]:
        # Consolidate all non-quarantined data for export
        active_data = [sample for b_name, b_list in buckets.items() if b_name != 'quarantined' and b_list is not None for sample in b_list]
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
            if buckets.get(b_name) is not None:
                bucket_length = len(buckets.get(b_name, []))
            else:
                bucket_length = 0
            return f"{b_name.replace('_', ' ').title()} ({bucket_length})"

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
                    # Escape HTML and use text_area for safer display
                    user_content = sample['messages'][1]['content'][:200] + ("..." if len(sample['messages'][1]['content']) > 200 else "")
                    st.text_area(" ", value=user_content, height=100, disabled=True, label_visibility="collapsed", key=f"user_{item_key}")
                with row_cols[2]:
                    # Escape HTML and use text_area for safer display  
                    assistant_content = sample['messages'][2]['content'][:200] + ("..." if len(sample['messages'][2]['content']) > 200 else "")
                    st.text_area(" ", value=assistant_content, height=100, disabled=True, label_visibility="collapsed", key=f"asst_{item_key}")
                
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
        if bucket_name != 'quarantined' and bucket_list is not None
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
    
    # Filter for models related to the current character
    character_models = [m for m in available_models if char_name.lower().replace(' ', '_') in m.lower()]
    
    # Auto-detect base models needed for comparison
    base_models_needed = set()
    for model in character_models:
        if not model.startswith("Base:"):
            metadata = st.session_state.inference_manager.get_model_metadata(model)
            if metadata and 'base_model' in metadata:
                base_models_needed.add(metadata['base_model'])
    
    # Add detected base models to available options
    for base_model in base_models_needed:
        base_option = f"Base: {base_model}"
        if base_option not in character_models:
            character_models.append(base_option)
    
    # Fallback: add current base model if no models detected
    if not character_models:
        character_models.append(f"Base: {st.session_state.inference_manager.base_model}")
    
    if len(character_models) < 2:
        st.info("ℹ️ You need at least two trained models/checkpoints for this character to compare them. Train more models or checkpoints to enable comparison.")
        return
    
    # Show detected base models info
    if base_models_needed:
        st.info(f"🔍 **Auto-detected base models**: {', '.join(sorted(base_models_needed))}")
    
    selected_models = st.multiselect(
        "Select models to compare (2 or more)",
        options=list(base_models_needed) + character_models,
        default=list(base_models_needed) + character_models[:2] if len(character_models) >= 2 else list(base_models_needed) + character_models,
        help="🎯 **Best practice**: Include base model for sanity check, then select checkpoints from the same training run to see progression."
    )

    if len(selected_models) < 2:
        st.warning("⚠️ Please select at least two models to compare.")
        return

    # Test prompt
    prompt = st.text_area(
        "Enter a test prompt",
        "Who are you and what do you want?",
        height=100,
        key="comparison_prompt"
    )

    # Generation settings
    with st.expander("⚙️ Generation Settings"):
        from utils.sampling_config import render_sampling_config_ui, SamplingConfig
        default_test_config = SamplingConfig(
            temperature=0.9,
            top_p=0.95,
            max_tokens=200,
            repetition_penalty=1.0,
        )
        test_sampling_config = render_sampling_config_ui(
            current_config=default_test_config,
            key_prefix="model_comparison_gen"
        )
        
        st.markdown("#### 🎲 Reproducibility Settings")
        col_seed1, col_seed2 = st.columns([2, 1])
        with col_seed1:
            use_custom_seed = st.checkbox(
                "Use custom seed for reproducible comparison",
                value=False,
                help="Set a specific seed to get identical results across multiple comparison runs"
            )
        with col_seed2:
            if use_custom_seed:
                custom_seed = st.number_input(
                    "Seed value",
                    min_value=0,
                    max_value=2**32-1,
                    value=42,
                    step=1,
                    help="Same seed = identical randomness for fair comparison"
                )
            else:
                custom_seed = None
        
        if not use_custom_seed:
            st.info("🎲 **Auto-seed**: A random seed will be generated and used consistently across all models for fair comparison")
        else:
            st.info(f"🔒 **Fixed seed {custom_seed}**: All models will use this seed for identical randomness")
    
    if st.button("🚀 Compare Responses", use_container_width=True, type="primary"):
        if not prompt.strip():
            st.error("❌ Please enter a prompt.")
            return

        # Store results in session state to persist them
        with st.spinner("Generating responses and fetching metrics..."):
            sp_config = test_sampling_config.to_dict()
            if 'min_tokens' in sp_config:
                min_tokens = sp_config.pop('min_tokens')
            if 'max_tokens' in sp_config:
                max_tokens = sp_config.pop('max_tokens')
            comparison_results = st.session_state.comparison_manager.compare_models_side_by_side(
                model_identifiers=selected_models,
                prompt=prompt,
                max_tokens=max_tokens,
                generation_config=sp_config,
                seed=custom_seed
            )
            
            # Enhanced metrics with personality engine judge evaluation
            with st.spinner("Evaluating character consistency with AI judge..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    base_metrics, training_summary, variable_metrics = loop.run_until_complete(
                        st.session_state.comparison_manager.get_enhanced_comparison_metrics(
                            selected_models,
                            st.session_state.current_character,
                            prompt,
                            comparison_results,
                            st.session_state.dataset_manager
                        )
                    )
                finally:
                    loop.close()
            
            st.session_state.comparison_data = {
                "responses": comparison_results,
                "base_metrics": base_metrics,
                "training_summary": training_summary,
                "variable_metrics": variable_metrics
            }
        st.success("✅ Comparison complete!")

    # Display comparison if data exists
    if 'comparison_data' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Comparison Results")
        st.info("🎯 **Fair Comparison**: All models used the same random seed, so differences in responses are due to model differences, not randomness.")

        responses = st.session_state.comparison_data['responses']
        base_metrics = st.session_state.comparison_data['base_metrics']
        training_summary = st.session_state.comparison_data['training_summary']
        variable_metrics = st.session_state.comparison_data['variable_metrics']

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
                
                # Promotion button - ONLY for checkpoints
                if model_id.startswith("Checkpoint:"):
                    checkpoint_name = model_id.split(': ')[1].split('/')[-1] if '/' in model_id else model_id.split(': ')[1]
                    if st.button(f"🏆 Promote {checkpoint_name}", key=f"promote_{i}", use_container_width=True):
                        checkpoint_id = model_id.split(': ')[1]
                        success = st.session_state.comparison_manager.promote_checkpoint(
                            character_name=char_name,
                            checkpoint_id=checkpoint_id,
                            reason=f"Promoted after comparing with prompt: '{prompt[:50]}...'"
                        )
                        if success:
                            st.success(f"✅ Promoted {checkpoint_name} as the best version!")
                        else:
                            st.error("❌ Failed to promote checkpoint.")
                elif model_id.startswith("LoRA:"):
                    st.info("Final LoRA model (already complete)")
                elif model_id.startswith("Base:"):
                    st.info("Base model (no promotion needed)")
        
        # Display promoted checkpoint info
        promoted_checkpoint = st.session_state.comparison_manager.get_promoted_checkpoint(char_name)
        if promoted_checkpoint:
            st.success(f"🏆 **Promoted Model:** `{promoted_checkpoint}` is currently selected as the best version for this character.")


        # Training Run Summary - Show shared configuration
        if training_summary:
            st.markdown("#### 📋 Training Run Summary")
            st.info("**Shared Configuration** (identical across all checkpoints from this training run)")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.markdown("**Model Configuration:**")
                st.write(f"• Base Model: `{training_summary.get('base_model', 'Unknown')}`")
                st.write(f"• Method: **{training_summary.get('training_method', 'Unknown')}**")
                if training_summary.get('use_dora'):
                    st.write("• DoRA: ✅ Enabled")
                if training_summary.get('use_rslora'):
                    st.write("• RSLoRA: ✅ Enabled")
                
            with summary_col2:
                st.markdown("**LoRA Parameters:**")
                st.write(f"• Rank (r): `{training_summary.get('lora_rank', 0)}`")
                st.write(f"• Alpha: `{training_summary.get('lora_alpha', 0)}`")
                st.write(f"• Dropout: `{training_summary.get('lora_dropout', 0.1):.2f}`")
                
            with summary_col3:
                st.markdown("**Dataset & Training:**")
                st.write(f"• Dataset Size: **{training_summary.get('dataset_size', 0)} samples**")
                st.write(f"• Configured Total Steps: `{training_summary.get('total_configured_steps', 0)}`")
                st.write(f"• Character: **{training_summary.get('character_name', 'Unknown')}**")
            
            st.markdown("---")

        # Variable Metrics Analysis - Show metrics that actually differ
        st.markdown("#### 📊 Checkpoint Progression Analysis")
        st.info("**Variable Metrics** (showing how performance changes throughout training)")
        
        # Create tabs for different types of analysis
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["📈 Training Progress", "🎯 Character Consistency", "📋 Detailed Comparison"])
        
        with analysis_tab1:
            st.markdown("##### Training Loss & Learning Progress")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Training progress line chart
                fig_progress = go.Figure()
                
                # Extract progression data - organize by training progression
                progression_data = []
                checkpoint_data = []
                final_data = []
                
                for model_id in selected_models:
                    metrics = variable_metrics.get(model_id, {})
                    if not metrics.get('is_base_model', False):
                        data_point = {
                            'model': model_id.split(': ')[-1] if ': ' in model_id else model_id,
                            'full_model_id': model_id,
                            'steps': metrics.get('actual_steps_completed', 0),
                            'loss': metrics.get('training_loss', 0),
                            'validation_loss': metrics.get('validation_loss', 0),
                            'is_final': metrics.get('is_final_model', False),
                            'is_checkpoint': metrics.get('is_checkpoint', False)
                        }
                        
                        if data_point['is_final']:
                            final_data.append(data_point)
                        elif data_point['is_checkpoint']:
                            checkpoint_data.append(data_point)
                        
                        progression_data.append(data_point)
                
                # Sort by steps for proper line progression
                progression_data.sort(key=lambda x: x['steps'])
                checkpoint_data.sort(key=lambda x: x['steps'])
                
                # Helpful info for users
                base_count = len([m for m in selected_models if m.startswith("Base:")])
                if len(checkpoint_data) == 0:
                    st.info("💡 **No checkpoints found.** To see training progression, you need multiple checkpoints from the same training run. Base model provides the starting baseline.")
                elif len(checkpoint_data) == 1:
                    st.info(f"📊 Found {base_count} base model(s), 1 checkpoint and {len(final_data)} final model. For progression analysis, multiple checkpoints work best.")
                else:
                    st.success(f"📊 Found {base_count} base model(s), {len(checkpoint_data)} checkpoints and {len(final_data)} final model(s) - perfect for progression analysis!")
                
                if progression_data:
                    # Add base model as starting point if selected
                    base_models_in_selection = [m for m in selected_models if m.startswith("Base:")]
                    
                    # Show training progression through checkpoints
                    if checkpoint_data:
                        checkpoint_steps = [d['steps'] for d in checkpoint_data if d['loss'] and d['loss'] > 0]
                        checkpoint_losses = [d['loss'] for d in checkpoint_data if d['loss'] and d['loss'] > 0]
                        checkpoint_names = [d['model'] for d in checkpoint_data if d['loss'] and d['loss'] > 0]
                        
                        # If we have a base model selected, add it as the starting point (step 0)
                        if base_models_in_selection and checkpoint_steps:
                            # Add base model as step 0 with a reasonable starting loss estimate
                            estimated_start_loss = max(checkpoint_losses) * 1.2 if checkpoint_losses else 2.0
                            checkpoint_steps = [0] + checkpoint_steps
                            checkpoint_losses = [estimated_start_loss] + checkpoint_losses
                            base_name = base_models_in_selection[0].replace('Base: ', '').split('/')[-1]
                            checkpoint_names = [f"{base_name} (base)"] + checkpoint_names
                        
                        if checkpoint_steps and checkpoint_losses:
                            fig_progress.add_trace(go.Scatter(
                                x=checkpoint_steps, y=checkpoint_losses,
                                mode='lines+markers',
                                name='Training Progression',
                                line=dict(color='#6366f1', width=3),
                                marker=dict(size=8),
                                text=checkpoint_names,
                                hovertemplate='<b>%{text}</b><br>Step: %{x}<br>Loss: %{y:.4f}<extra></extra>'
                            ))
                    
                    # Add validation loss if available
                    val_checkpoint_data = [d for d in checkpoint_data if d['validation_loss'] and d['validation_loss'] > 0]
                    if val_checkpoint_data:
                        val_steps = [d['steps'] for d in val_checkpoint_data]
                        val_losses = [d['validation_loss'] for d in val_checkpoint_data]
                        val_names = [d['model'] for d in val_checkpoint_data]
                        
                        fig_progress.add_trace(go.Scatter(
                            x=val_steps, y=val_losses,
                            mode='lines+markers',
                            name='Validation Loss',
                            line=dict(color='#ef4444', width=3),
                            marker=dict(size=8, color='orange'),
                            text=val_names,
                            hovertemplate='<b>%{text}</b><br>Step: %{x}<br>Val Loss: %{y:.4f}<extra></extra>'
                        ))
                    
                    # Mark final model
                    if final_data:
                        final_model = final_data[0]
                        if final_model['loss'] and final_model['loss'] > 0:
                            fig_progress.add_trace(go.Scatter(
                                x=[final_model['steps']], y=[final_model['loss']],
                                mode='markers',
                                name='Final LoRA Model',
                                marker=dict(size=15, color='#10b981', symbol='star'),
                                text=[final_model['model']],
                                hovertemplate='<b>%{text}</b><br>Step: %{x}<br>Loss: %{y:.4f}<extra></extra>'
                            ))
                    
                    # If no checkpoints, show explanation
                    if not checkpoint_data and not final_data:
                        st.info("💡 **No checkpoint progression data available.** This typically means you're comparing base models or single models without training checkpoints.")
                
                fig_progress.update_layout(
                    title="Training Progression: Base Model → Checkpoints → Final LoRA",
                    xaxis_title="Training Steps (0 = Base Model)",
                    yaxis_title="Loss",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig_progress, use_container_width=True)
            
            with col2:
                # Model Performance Summary Chart
                fig_performance = go.Figure()
                
                # Create a comprehensive performance chart including base models
                all_model_data = []
                
                # Add base models to the analysis
                for model_id in selected_models:
                    metrics = variable_metrics.get(model_id, {})
                    if metrics.get('is_base_model', False):
                        all_model_data.append({
                            'model': model_id.split(': ')[-1] if ': ' in model_id else model_id.replace('Base: ', ''),
                            'full_model_id': model_id,
                            'steps': 0,  # Base model is step 0
                            'loss': 0,   # Base models don't have training loss
                            'validation_loss': 0,
                            'is_final': False,
                            'is_checkpoint': False,
                            'is_base_model': True
                        })
                
                # Add trained models
                all_model_data.extend(checkpoint_data + final_data)
                
                if all_model_data:
                    # Performance comparison chart - Training Loss vs Character Consistency
                    model_names = []
                    training_losses = []
                    consistency_scores = []
                    model_types = []
                    
                    for d in all_model_data:
                        model_id = d['full_model_id']
                        metrics = variable_metrics.get(model_id, {})
                        
                        # For base models, we don't have training loss, so we'll use consistency only
                        if d.get('is_base_model', False):
                            consistency = metrics.get('character_consistency', 0)
                            if consistency > 0:  # Only include if we have consistency data
                                model_names.append(d['model'])
                                training_losses.append(0)  # Base model has no training loss
                                consistency_scores.append(consistency)
                                model_types.append('base')
                        elif d['loss'] and d['loss'] > 0:
                            model_names.append(d['model'])
                            training_losses.append(d['loss'])
                            consistency_scores.append(metrics.get('character_consistency', 0))
                            if d['is_checkpoint']:
                                model_types.append('checkpoint')
                            else:
                                model_types.append('final')
                    
                    if model_names and len(model_names) > 0:
                        # Color code by model type: Base=Gray, Checkpoint=Red, Final=Green
                        color_map = {'base': '#94a3b8', 'checkpoint': '#ef4444', 'final': '#10b981'}
                        colors = [color_map[t] for t in model_types]
                        
                        fig_performance.add_trace(go.Scatter(
                            x=training_losses,
                            y=consistency_scores,
                            mode='markers+text',
                            text=model_names,
                            textposition='top center',
                            marker=dict(
                                size=12,
                                color=colors,
                                line=dict(width=2, color='white')
                            ),
                            hovertemplate='<b>%{text}</b><br>Training Loss: %{x:.4f}<br>Consistency: %{y:.3f}<extra></extra>',
                            name='Models'
                        ))
                        
                        fig_performance.update_layout(
                            title="Model Performance: Loss vs Consistency",
                            xaxis_title="Training Loss (lower is better, base model at 0)",
                            yaxis_title="Character Consistency (higher is better)",
                            template="plotly_dark",
                            height=400,
                            showlegend=False
                        )
                        
                        # Add legend/annotations to explain the colors and chart
                        fig_performance.add_annotation(
                            text="🔵 Gray: Base Model (sanity check)<br/>🔴 Red: Checkpoints<br/>🟢 Green: Final LoRA<br/><br/>🎯 Good models: high consistency<br/>Base model shows pre-training behavior",
                            xref="paper", yref="paper",
                            x=0.02, y=0.98,
                            xanchor="left", yanchor="top",
                            bgcolor="rgba(0,0,0,0.7)",
                            bordercolor="white",
                            borderwidth=1,
                            font=dict(size=9)
                        )
                else:
                    # Show empty state
                    st.info("💡 **Performance comparison requires trained models.** Train some checkpoints or LoRA models to see this analysis.")
                
                if checkpoint_data or final_data:
                    st.plotly_chart(fig_performance, use_container_width=True)
                
        with analysis_tab2:
            st.markdown("##### Character Consistency Evolution")
            
            # Character consistency radar chart
            fig_consistency = go.Figure()
            
            consistency_metrics = {
                'Overall Consistency': 'character_consistency',
                'Personality': 'personality_consistency', 
                'Speech Style': 'speech_style',
                'Emotional Auth.': 'emotional_authenticity',
                'Scenario Fit': 'scenario_appropriateness'
            }
            
            colors = ['#6366f1', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
            for i, model_id in enumerate(selected_models):
                if not variable_metrics.get(model_id, {}).get('is_base_model', False):
                    metrics = variable_metrics.get(model_id, {})
                    values = []
                    for metric_name, metric_key in consistency_metrics.items():
                        values.append(metrics.get(metric_key, 0))
                    
                    # Only add if we have non-zero values
                    if any(v > 0 for v in values):
                        fig_consistency.add_trace(go.Scatterpolar(
                            r=values + [values[0]],  # Close the polygon
                            theta=list(consistency_metrics.keys()) + [list(consistency_metrics.keys())[0]],
                            fill='toself',
                            name=model_id.split(': ')[-1] if ': ' in model_id else model_id,
                            line_color=colors[i % len(colors)]
                        ))
            
            fig_consistency.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickmode='linear',
                        tick0=0,
                        dtick=0.2
                    )),
                showlegend=True,
                title="Character Consistency by Aspect",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig_consistency, use_container_width=True)
            
        with analysis_tab3:
            # Detailed metrics table focusing on variable metrics
            st.markdown("##### Checkpoint Comparison Table")
            st.info("💡 **Focus on differences:** This table shows metrics that vary between checkpoints")
            
            metrics_df = []
            for model_id in selected_models:
                metrics = variable_metrics.get(model_id, {})
                
                # Create model name
                model_name = model_id.split(': ')[-1] if ': ' in model_id else model_id
                if metrics.get('is_base_model'):
                    model_type = "BASE"
                elif metrics.get('is_final_model'):
                    model_type = "FINAL"
                elif metrics.get('is_checkpoint'):
                    model_type = "CHECKPOINT"
                else:
                    model_type = "UNKNOWN"
                
                # Format values, showing only meaningful differences
                row = {
                    'Model': model_name,
                    'Type': model_type,
                    'Steps Completed': f"{int(metrics.get('actual_steps_completed', 0))}" if metrics.get('actual_steps_completed', 0) > 0 else '—',
                    'Training Loss': f"{metrics.get('training_loss', 0):.4f}" if metrics.get('training_loss', 0) > 0 else '—',
                    'Validation Loss': f"{metrics.get('validation_loss', 0):.4f}" if metrics.get('validation_loss', 0) > 0 else '—',
                    'Character Consistency': f"{metrics.get('character_consistency', 0):.3f}" if metrics.get('character_consistency', 0) > 0 else '—',
                    'Training Time (min)': f"{metrics.get('training_time_elapsed', 0):.1f}" if metrics.get('training_time_elapsed', 0) > 0 else '—',
                    'Learning Rate': f"{metrics.get('learning_rate_at_checkpoint', 0):.2e}" if metrics.get('learning_rate_at_checkpoint', 0) > 0 else '—'
                }
                metrics_df.append(row)
            
            import pandas as pd
            df = pd.DataFrame(metrics_df)
            st.dataframe(df, use_container_width=True)
            
            # Key insights
            trained_models = [m for m in variable_metrics.keys() if not variable_metrics[m].get('is_base_model', False)]
            if len(trained_models) > 1:
                st.markdown("**📈 Key Insights:**")
                
                # Find best performing checkpoint
                best_consistency = max(variable_metrics[m].get('character_consistency', 0) for m in trained_models)
                best_model = [m for m in trained_models if variable_metrics[m].get('character_consistency', 0) == best_consistency][0]
                
                lowest_loss = min(variable_metrics[m].get('training_loss', float('inf')) for m in trained_models if variable_metrics[m].get('training_loss', 0) > 0)
                best_loss_model = [m for m in trained_models if variable_metrics[m].get('training_loss', float('inf')) == lowest_loss][0]
                
                if best_consistency > 0:
                    st.success(f"🏆 **Best Character Consistency:** `{best_model.split(': ')[-1] if ': ' in best_model else best_model}` ({best_consistency:.3f})")
                
                if lowest_loss < float('inf'):
                    st.success(f"📉 **Lowest Training Loss:** `{best_loss_model.split(': ')[-1] if ': ' in best_loss_model else best_loss_model}` ({lowest_loss:.4f})")
                
                # Check for overfitting
                final_models = [m for m in trained_models if variable_metrics[m].get('is_final_model', False)]
                if final_models:
                    final_model = final_models[0]
                    final_loss = variable_metrics[final_model].get('training_loss', 0)
                    final_val_loss = variable_metrics[final_model].get('validation_loss', 0)
                    
                    if final_loss > 0 and final_val_loss > 0 and final_val_loss > final_loss * 1.2:
                        st.warning("⚠️ **Potential Overfitting Detected:** Validation loss is significantly higher than training loss in final model")
                    elif final_loss > 0 and final_val_loss > 0 and abs(final_val_loss - final_loss) < 0.01:
                        st.success("✅ **Good Generalization:** Training and validation losses are well-aligned")


def page_model_management():
    """Model management page for merging, managing trained models"""
    st.markdown('<h2 class="gradient-text">🔧 Model Management</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return
    
    # Get available models and character info
    available_models = st.session_state.inference_manager.get_available_models()
    character_name = st.session_state.current_character.get('name', 'Unknown')
    
    tab1, tab2, tab3 = st.tabs(["🔀 Model Merging", "📊 Model Overview", "🗃️ Model Assets"])
    
    with tab1:
        st.markdown("### Merge LoRA/DoRA with Base Model")
        st.info("💡 Merging creates a complete model that doesn't require PEFT adapters. This is recommended for DoRA models and final deployment.")
        
        # Filter to only show LoRA/DoRA models for the current character
        lora_models = [m for m in available_models if not m.startswith("Base:")]
        
        if not lora_models:
            st.info("ℹ️ No LoRA/DoRA models available for merging. Train a model first.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_model = st.selectbox("Select Model to Merge", lora_models)
                
                # Show model metadata
                if selected_model:
                    metadata = st.session_state.inference_manager.get_model_metadata(selected_model)
                    if metadata:
                        base_model = metadata.get('base_model', 'Unknown')
                        use_dora = metadata.get('use_dora', False)
                        use_rslora = metadata.get('use_rslora', False)
                        
                        method_display = "DoRA" if use_dora else "RSLoRA" if use_rslora else "LoRA"
                        
                        st.markdown(f"""
                        **Model Information:**
                        - **Method**: {method_display}
                        - **Base Model**: `{base_model}`
                        - **Training Date**: {metadata.get('training_date', 'Unknown')}
                        - **Dataset Size**: {metadata.get('dataset_size', 'Unknown')} samples
                        """)
                        
                        if use_dora:
                            st.info("💡 DoRA models benefit significantly from merging for optimal inference performance.")
                    else:
                        st.warning("⚠️ No metadata available for this model.")
                
                include_checkpoints = st.checkbox("Include checkpoint selection", help="Also show intermediate checkpoints for merging")
                
                if include_checkpoints:
                    checkpoints = st.session_state.training_manager.get_available_checkpoints(character_name)
                    if checkpoints:
                        checkpoint_option = st.selectbox("Or select checkpoint to merge", ["None (use final model)"] + checkpoints)
                        checkpoint_path = None if checkpoint_option == "None (use final model)" else checkpoint_option
                    else:
                        checkpoint_path = None
                        st.info("No checkpoints available")
                else:
                    checkpoint_path = None
            
            with col2:
                st.markdown("### Merge Benefits")
                st.markdown("""
                - **Faster Inference**: No PEFT overhead
                - **Easier Deployment**: Single model file
                - **DoRA Optimization**: Better performance for DoRA
                - **Portability**: Compatible with any transformers setup
                """)
            
            if st.button("🔀 Merge Model", use_container_width=True):
                with st.spinner("Merging model... This may take several minutes."):
                    try:
                        merged_path = st.session_state.training_manager.merge_and_export_model(
                            character_name, checkpoint_path
                        )
                        st.success(f"✅ Model merged successfully! Exported to: {merged_path}")
                        
                        # Show file size
                        file_size_mb = merged_path.stat().st_size / (1024 * 1024)
                        st.info(f"📦 Export size: {file_size_mb:.1f} MB")
                        
                    except Exception as e:
                        st.error(f"❌ Merge failed: {str(e)}")
    
    with tab2:
        st.markdown("### Model Overview")
        
        if not lora_models:
            st.info("ℹ️ No trained models available.")
        else:
            for model in lora_models:
                with st.expander(f"📊 {model}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Model metadata
                        metadata = st.session_state.inference_manager.get_model_metadata(model)
                        if metadata:
                            st.markdown("**Training Information:**")
                            st.write(f"- Base Model: `{metadata.get('base_model', 'Unknown')}`")
                            st.write(f"- Method: {metadata.get('training_method', 'Unknown').upper()}")
                            st.write(f"- Rank (r): {metadata.get('lora_r', 'Unknown')}")
                            st.write(f"- Alpha: {metadata.get('lora_alpha', 'Unknown')}")
                            st.write(f"- Dropout: {metadata.get('lora_dropout', 'Unknown')}")
                            st.write(f"- Total Steps: {metadata.get('total_steps', 'Unknown')}")
                            st.write(f"- Dataset Size: {metadata.get('dataset_size', 'Unknown')} samples")
                    
                    with col2:
                        # Training metrics
                        metrics = st.session_state.inference_manager.get_model_metrics(model)
                        if metrics:
                            st.markdown("**Training Metrics:**")
                            final_loss = metrics.get('current_loss', metrics.get('loss'))
                            if final_loss:
                                st.write(f"- Final Loss: {final_loss:.4f}")
                            
                            if 'character_consistency' in metrics:
                                consistency = metrics['character_consistency']
                                st.write(f"- Character Consistency: {consistency:.2f}")
                            
                            training_time = metrics.get('elapsed_time')
                            if training_time:
                                minutes = int(training_time // 60)
                                seconds = int(training_time % 60)
                                st.write(f"- Training Time: {minutes}m {seconds}s")
    
    with tab3:
        st.markdown("### Model Asset Management")
        
        # Fix legacy models section
        st.markdown("#### Fix Legacy Models")
        st.info("💡 If you have models trained before the metadata system, add compatibility info here.")
        
        legacy_models = [m for m in available_models if not m.startswith("Base:")]
        if legacy_models:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                legacy_model = st.selectbox("Select legacy model to fix", legacy_models, key="legacy_model_fix")
                
                # Check if it already has metadata
                if legacy_model:
                    metadata = st.session_state.inference_manager.get_model_metadata(legacy_model)
                    if metadata:
                        st.success("✅ This model already has metadata!")
                    else:
                        st.warning("⚠️ This model needs metadata to work properly.")
                        
                        base_model_options = [
                            "HuggingFaceTB/SmolLM2-135M-Instruct",
                            "HuggingFaceTB/SmolLM2-360M-Instruct", 
                            "HuggingFaceTB/SmolLM2-1.7B-Instruct"
                        ]
                        
                        base_model_fix = st.selectbox("Which base model was this trained on?", base_model_options, index=1)
                        method_fix = st.selectbox("Training method", ["lora", "dora", "rslora"])
            
            with col_b:
                if st.button("🔧 Add Metadata", use_container_width=True):
                    if legacy_model and base_model_fix:
                        success = st.session_state.training_manager.add_metadata_to_existing_model(
                            character_name, base_model_fix, method_fix
                        )
                        if success:
                            st.success("✅ Metadata added! Model should work now.")
                            st.rerun()
                        else:
                            st.error("❌ Failed to add metadata.")
        else:
            st.info("No models found that need fixing.")
        
        st.markdown("---")
        
        # ✅ NEW: Fix checkpoint metadata section
        st.markdown("#### Fix Missing Checkpoint Metadata")
        st.info("💡 Add missing training_metadata.json files to checkpoints for better comparison charts.")
        
        # Get all checkpoints without metadata
        checkpoints_without_metadata = []
        for model in available_models:
            if "Checkpoint:" in model and not model.startswith("Base:"):
                metadata = st.session_state.inference_manager.get_model_metadata(model)
                if not metadata:
                    checkpoints_without_metadata.append(model)
        
        if checkpoints_without_metadata:
            st.warning(f"Found {len(checkpoints_without_metadata)} checkpoints missing metadata")
            
            # Get base model from final model
            final_cricket_metadata = st.session_state.inference_manager.get_model_metadata("LoRA: cricket")
            if final_cricket_metadata:
                detected_base_model = final_cricket_metadata.get('base_model', 'HuggingFaceTB/SmolLM2-360M-Instruct')
                detected_method = final_cricket_metadata.get('training_method', 'dora')
                
                st.info(f"🔍 **Auto-detected from final model**: {detected_method.upper()} on `{detected_base_model}`")
                
                if st.button("🔧 Fix All Checkpoint Metadata", use_container_width=True):
                    success_count = 0
                    for checkpoint_model in checkpoints_without_metadata:
                        # Extract character name and checkpoint path
                        parts = checkpoint_model.split(": ")[1].split("/")
                        char_name = parts[0]
                        checkpoint_name = parts[1]
                        
                        # Add metadata to this specific checkpoint
                        success = st.session_state.training_manager.add_metadata_to_checkpoint(
                            char_name, checkpoint_name, detected_base_model, detected_method
                        )
                        if success:
                            success_count += 1
                    
                    if success_count > 0:
                        st.success(f"✅ Added metadata to {success_count} checkpoints!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add metadata to checkpoints.")
            else:
                st.warning("⚠️ Could not auto-detect settings from final model. Please add manually.")
        else:
            st.success("✅ All checkpoints have metadata!")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear All Training Assets", use_container_width=True):
                if st.session_state.training_manager.clear_training_assets(character_name):
                    st.success("Training assets cleared!")
                    st.rerun()
                else:
                    st.info("No training assets to remove.")
        
        with col2:
            if st.button("⬇️ Export LoRA", use_container_width=True):
                try:
                    zip_path = st.session_state.training_manager.export_lora(character_name)
                    st.success(f"LoRA exported to {zip_path}")
                except Exception as e:
                    st.error(str(e))
        
        with col3:
            if st.button("⬇️ Export Latest Checkpoint", use_container_width=True):
                zip_path = st.session_state.training_manager.export_latest_checkpoint(character_name)
                if zip_path:
                    st.success(f"Checkpoint exported to {zip_path}")
                else:
                    st.info("No checkpoints found to export.")
        
        # Show disk usage
        st.markdown("### Disk Usage")
        
        adapter_dir = Path("training_output/adapters")
        exports_dir = Path("training_output/exports")
        
        def get_dir_size(path):
            if not path.exists():
                return 0
            total = 0
            for file in path.rglob('*'):
                if file.is_file():
                    total += file.stat().st_size
            return total / (1024 * 1024)  # MB
        
        adapters_size = get_dir_size(adapter_dir)
        exports_size = get_dir_size(exports_dir)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Assets", f"{adapters_size:.1f} MB")
        with col2:
            st.metric("Exports", f"{exports_size:.1f} MB")
        with col3:
            st.metric("Total", f"{adapters_size + exports_size:.1f} MB")


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
    elif selected_page == "🔧 Model Management":
        page_model_management()
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: #64748b; border-top: 1px solid rgba(100, 116, 139, 0.2); margin-top: 3rem;">
            <p>🎭 Character AI Training Studio • Built with ❤️ and Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 