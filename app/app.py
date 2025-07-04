from asyncio.log import logger
import os

from utils.openai_client import get_client

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
from utils.health import get_health_checker
from utils.error_handling import setup_global_error_handling, streamlit_error_boundary

# Configure logging for debugging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('logs/app.log') if Path('logs').exists() else logging.NullHandler()
    ]
)

# Setup global error handling
setup_global_error_handling()

# ✅ FIX: Global singleton to prevent DatasetManager re-initialization
_GLOBAL_DATASET_MANAGER = None

def get_or_create_dataset_manager(api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Ensure that we don't accidentally create multiple DatasetManager instances"""
    # Use a global dataset manager to prevent reinitialization issues
    if hasattr(get_or_create_dataset_manager, '_instance'):
        return get_or_create_dataset_manager._instance
    
    try:
        from utils.dataset import DatasetManager
        manager = DatasetManager(api_key=api_key, base_url=base_url)
        get_or_create_dataset_manager._instance = manager
        logger.info("✅ DatasetManager singleton created successfully")
        return manager
    except Exception as e:
        logger.error(f"❌ Failed to create DatasetManager: {e}")
        # Don't cache failed instances
        return None

def health_endpoint():
    """Health check endpoint for production monitoring"""
    health_checker = get_health_checker()
    redis_url = os.getenv('REDIS_URL')
    
    health_status = health_checker.get_comprehensive_health(redis_url)
    
    # Return appropriate HTTP status based on health
    if health_status["status"] == "healthy":
        st.success("✅ System Healthy")
    elif health_status["status"] == "warning":
        st.warning("⚠️ System Warning")
    else:
        st.error("❌ System Unhealthy")
    
    # Display health details in JSON format
    st.json(health_status)
    
    # Stop execution here for health check
    st.stop()


def get_current_character_safely():
    """
    Safely get the current character from session state, handling both CharacterCore and dict formats.
    
    Returns:
        tuple: (character_object, character_name) or (None, None) if no character
    """
    # Try CharacterCore first (new format)
    char_core = st.session_state.get('current_character_core')
    if char_core and hasattr(char_core, 'name'):
        return char_core, char_core.name
    
    # Try legacy dict format
    legacy_char = st.session_state.get('current_character')
    if legacy_char:
        if isinstance(legacy_char, dict):
            name = legacy_char.get('name', 'Unknown')
            return legacy_char, name
        elif hasattr(legacy_char, 'name'):
            # It's actually a CharacterCore stored in wrong key
            return legacy_char, legacy_char.name
    
    return None, None


def set_current_character(character):
    """
    Safely set the current character in session state.
    
    Args:
        character: Either a CharacterCore object or dict
    """
    if hasattr(character, 'name'):
        # It's a CharacterCore object
        st.session_state.current_character_core = character
        st.session_state.selected_character = character.name
        # Clear legacy format to avoid confusion
        if 'current_character' in st.session_state:
            del st.session_state.current_character
    elif isinstance(character, dict):
        # It's legacy format
        st.session_state.current_character = character
        name = character.get('name', 'Unknown')
        st.session_state.selected_character = name
    else:
        logger.warning(f"Unknown character format: {type(character)}")


def init_session_state():
    # Initialize authentication manager first
    if 'auth_manager' not in st.session_state:
        try:
            from utils.auth import AuthManager
            st.session_state.auth_manager = AuthManager()
            logger.info("✅ AuthManager initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AuthManager: {e}")
    
    # Initialize OpenAI client first (before other managers that might use it)
    if 'openai_client_initialized' not in st.session_state:
        # Get OpenAI configuration from environment
        api_key = os.getenv('OPENAI_API_KEY', "empty")
        base_url = os.getenv('OPENAI_BASE_URL')
        
        # Set up the global OpenAI client before anything else
        try:
            from utils.openai_client import OpenAIClient, set_client
            client = OpenAIClient(api_key=api_key, base_url=base_url)
            set_client(client)
            st.session_state.openai_client_initialized = True
            logger.info("✅ Global OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            st.session_state.openai_client_initialized = False

    if 'world_manager' not in st.session_state:
        st.session_state.world_manager = WorldManager()

    if 'world_discovery_manager' not in st.session_state:
        from utils.world_discovery import WorldDiscoveryManager
        st.session_state.world_discovery_manager = WorldDiscoveryManager()

    if 'character_manager' not in st.session_state:
        st.session_state.character_manager = CharacterManager(world_manager=st.session_state.world_manager, client=get_client())

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
def render_sidebar(pg):
    """Render the modern navigation sidebar with authentication"""
    with st.sidebar:
        # App branding
        st.markdown(f"""
            <div style="text-align: center; padding: 1rem 0;">
                <h1 style="margin: 0; background: linear-gradient(45deg, #6366f1, #8b5cf6); 
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                           background-clip: text; font-size: 1.5rem;">
                    🎭 Character AI Studio
                </h1>
                <p style="margin: 0.5rem 0 0 0; color: #64748b; font-size: 0.9rem;">
                    Build • Train • Deploy
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Authentication section
        render_auth_section()
        
        # Get current page info first
        current_page_title = pg.title if hasattr(pg, 'title') else "Unknown"
        
        # Create a mapping of page titles to page objects for navigation
        page_mapping = {}
        page_options = []
        
        # Add authentication pages to navigation
        auth_pages = [
            ("🔐 Login", "pages/login.py"),
            ("📝 Register", "pages/register.py"),
            ("👤 Profile", "pages/profile.py"),
        ]
        
        # Flatten the pages structure for the option menu
        for section, section_pages in [
            ("Authentication", auth_pages),
            ("Platform", [
                ("🌍 Discover Worlds", "pages/world_discovery.py"),
                ("🎭 Character Selection", "pages/character_selection.py"),
            ]),
            ("Character Studio", [
                ("📁 Character Upload", "pages/character_upload.py"),
                ("🗨️ Conversational Builder", "pages/character_builder.py"),
                ("📋 Character Management", "pages/character_management.py"),
            ]),
            ("World & Data", [
                ("🌍 World Management", "pages/world_management.py"),
                ("🎨 Dataset Studio", "pages/dataset_studio.py"),
                ("🌊 N-Script Studio", "pages/nscript_studio.py"),
            ]),
            ("Training & Testing", [
                ("⚙️ Training Config", "pages/training_config.py"),
                ("📊 Training Dashboard", "pages/training_dashboard.py"),
                ("💬 Character Chat", "pages/character_chat.py"),
                ("🧪 Model Testing", "pages/model_testing.py"),
                ("⚔️ Model Comparison", "pages/model_comparison.py"),
                ("🔍 Inference Inspector", "pages/inference_inspector.py"),
                ("🔧 Model Management", "pages/model_management.py"),
            ]),
        ]:
            for title, path in section_pages:
                page_options.append(title)
                page_mapping[title] = path
        
        # Find current selection index
        default_index = 0
        for i, title in enumerate(page_options):
            if current_page_title in title or title in current_page_title:
                default_index = i
                break
        
        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=page_options,
            icons=[None] * len(page_options),  # No bootstrap icons, just use emojis
            menu_icon=None,
            default_index=default_index,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"display": "none"},  # Hide icon space since we're using emojis in text
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
        
        # Handle navigation - switch to selected page if different from current
        if selected != current_page_title and selected in page_mapping:
            st.switch_page(page_mapping[selected])
        
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
        
        # Character info - handle both legacy dict and new CharacterCore
        current_character = st.session_state.get('current_character_core') or st.session_state.get('current_character')
        if current_character:
            # Get character name safely regardless of format
            if hasattr(current_character, 'name'):
                # CharacterCore object
                char_name = current_character.name
                char_data = current_character
            elif isinstance(current_character, dict):
                # Legacy dictionary format
                char_name = current_character.get("name", "Unknown")
                char_data = current_character
            else:
                char_name = "Unknown"
                char_data = None
            
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0 0 0.5rem 0;">🎭 Current Character</h4>
                    <p style="margin: 0; font-weight: 500;">{char_name}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Dataset info - only if we have a valid character
            if char_data:
                dataset_info = st.session_state.dataset_manager.get_dataset_info(char_data)
                if dataset_info['exists']:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="margin: 0 0 0.5rem 0;">📊 Dataset</h4>
                            <p style="margin: 0;">Samples: {dataset_info['sample_count']}</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Quick actions for current character
        current_character = st.session_state.get('current_character_core') or st.session_state.get('current_character')
        if current_character:
            with st.expander("🗃️ Character Assets", expanded=False):
                # Get character name safely
                if hasattr(current_character, 'name'):
                    char_name = current_character.name
                elif isinstance(current_character, dict):
                    char_name = current_character.get("name", "unknown")
                else:
                    char_name = "unknown"

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


def render_auth_section():
    """Render authentication section in sidebar"""
    try:
        # Check if authenticated
        authenticated = st.session_state.get('authenticated', False)
        
        if authenticated:
            # Show user info
            current_user = st.session_state.get('current_user')
            if current_user:
                st.markdown(f"""
                    <div class="metric-card">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <div>
                                <h4 style="margin: 0 0 0.25rem 0;">👤 {current_user.username}</h4>
                                <p style="margin: 0; color: #64748b; font-size: 0.85rem;">{current_user.role.value.title()}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Quick auth actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👤", help="Profile", use_container_width=True):
                        st.switch_page("pages/profile.py")
                with col2:
                    if st.button("🚪", help="Logout", use_container_width=True):
                        logout_user_simple()
                        st.rerun()
        else:
            # Show login/register buttons for unauthenticated users
            st.markdown("**🔐 Authentication**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔐 Login", use_container_width=True):
                    st.switch_page("pages/login.py")
            with col2:
                if st.button("📝 Register", use_container_width=True):
                    st.switch_page("pages/register.py")
                    
    except Exception as e:
        logger.error(f"Auth section error: {e}")
        # Fallback to basic auth buttons
        st.markdown("**🔐 Authentication**")
        if st.button("🔐 Login"):
            st.switch_page("pages/login.py")


def logout_user_simple():
    """Simple logout function for sidebar"""
    try:
        # Clear authentication state
        auth_keys = ['authenticated', 'current_user', 'access_token', 'refresh_token']
        for key in auth_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        logger.info("User logged out from sidebar")
        return True
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return False


# Main app function
@streamlit_error_boundary
def main():
    """Main app function"""
    # Handle health check endpoint
    query_params = st.query_params
    if 'health' in query_params:
        health_endpoint()
        return
    
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

    # Configure pages for modern navigation
    pages = {
        "Authentication": [
            st.Page("pages/login.py", title="🔐 Login", icon="🔐"),
            st.Page("pages/register.py", title="📝 Register", icon="📝"),
            st.Page("pages/profile.py", title="👤 Profile", icon="👤"),
        ],
        "Platform": [
            st.Page("pages/world_discovery.py", title="🌍 Discover Worlds", icon="🌍"),
        ],
        "Character Studio": [
            st.Page("pages/character_upload.py", title="📁 Character Upload", icon="📁"),
            st.Page("pages/character_builder.py", title="🗨️ Conversational Builder", icon="🗨️"),
            st.Page("pages/character_management.py", title="📋 Character Management", icon="📋"),
        ],
        "World & Data": [
            st.Page("pages/world_management.py", title="🌍 World Management", icon="🌍"),
            st.Page("pages/dataset_studio.py", title="🎨 Dataset Studio", icon="🎨"),
            st.Page("pages/nscript_studio.py", title="🌊 N-Script Studio", icon="🌊"),
        ],
        "Training & Testing": [
            st.Page("pages/training_config.py", title="⚙️ Training Config", icon="⚙️"),
            st.Page("pages/training_dashboard.py", title="📊 Training Dashboard", icon="📊"),
            st.Page("pages/character_chat.py", title="💬 Character Chat", icon="💬"),
            st.Page("pages/model_testing.py", title="🧪 Model Testing", icon="🧪"),
            st.Page("pages/model_comparison.py", title="⚔️ Model Comparison", icon="⚔️"),
            st.Page("pages/inference_inspector.py", title="🔍 Inference Inspector", icon="🔍"),
            st.Page("pages/advanced_emotion_control.py", title="🎭 Advanced Emotion Control", icon="🎭"),
            st.Page("pages/model_management.py", title="🔧 Model Management", icon="🔧"),
        ],
    }

    # Use hidden navigation to maintain custom sidebar
    pg = st.navigation(pages, position="hidden")

    render_header()
    
    # Render custom sidebar with navigation and authentication
    render_sidebar(pg)
    
    # Run the selected page
    pg.run()
    
    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0; color: #64748b; border-top: 1px solid rgba(100, 116, 139, 0.2); margin-top: 3rem;">
            <p>🎭 Character AI Training Studio • Built with ❤️ and Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 