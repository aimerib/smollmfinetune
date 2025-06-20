from asyncio.log import logger
import os

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
from streamlit_option_menu import option_menu
import torch

torch.classes.__path__ = []

import sys
from pathlib import Path
from typing import Optional

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.character import CharacterManager
from utils.dataset import DatasetManager
from utils.training import TrainingManager
from utils.inference import InferenceManager
from utils.comparison import ComparisonManager
from utils.world import WorldManager

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
    
    if 'world_manager' not in st.session_state:
        st.session_state.world_manager = WorldManager()
    
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
            options=["📁 Character Upload", "🌍 World Management", "🎨 Dataset Studio", "⚙️ Training Config", "📊 Training Dashboard", "🧪 Model Testing", "⚔️ Model Comparison", "🔧 Model Management"],
            icons=["upload", "globe", "palette", "gear", "graph-up", "flask", "shuffle", "tools"],
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

# Original page_character_upload() function has been extracted to pages/character_upload.py

# Original page_dataset_preview() function has been extracted to pages/dataset_studio.py

# Original page_training_config() function has been extracted to pages/training_config.py
# Original page_training_dashboard() function has been extracted to pages/training_dashboard.py

# Model testing and inference page

# Original page_dataset_explorer_v2() function has been extracted to pages/dataset_studio.py

# Original page_model_comparison() function has been extracted to pages/model_comparison.py

# Original page_model_management() function has been extracted to pages/model_management.py

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
        from pages.character_upload import page_character_upload
        page_character_upload()
    elif selected_page == "🌍 World Management":
        from pages.world_management import page_world_management
        page_world_management()
    elif selected_page == "🎨 Dataset Studio":
        from pages.dataset_studio import page_dataset_studio
        page_dataset_studio()
    elif selected_page == "⚙️ Training Config":
        from pages.training_config import page_training_config
        page_training_config()
    elif selected_page == "📊 Training Dashboard":
        from pages.training_dashboard import page_training_dashboard
        page_training_dashboard()
    elif selected_page == "🧪 Model Testing":
        from pages.model_testing import page_model_testing
        page_model_testing()
    elif selected_page == "⚔️ Model Comparison":
        from pages.model_comparison import page_model_comparison
        page_model_comparison()
    elif selected_page == "🔧 Model Management":
        from pages.model_management import page_model_management
        page_model_management()
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: #64748b; border-top: 1px solid rgba(100, 116, 139, 0.2); margin-top: 3rem;">
            <p>🎭 Character AI Training Studio • Built with ❤️ and Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 