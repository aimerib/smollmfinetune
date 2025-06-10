import asyncio
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
from utils.inference_engines import InferenceEngineFactory

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
    if 'generated_questions' not in st.session_state:
        st.session_state.generated_questions = None

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
        
        # ------------------------------------------------------------------
        # 🔧 Asset-management shortcuts
        # ------------------------------------------------------------------

        if st.session_state.current_character:
            char_name = st.session_state.current_character.get("name", "unknown")

            with st.expander("🗑️ Clear / Export / Checkpoints"):
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    if st.button("🗑️ Clear Training", key="clear_training_btn"):
                        if st.session_state.training_manager.clear_training_assets(char_name):
                            st.success("Training assets cleared!")
                        else:
                            st.info("No training assets to remove.")

                with col_b:
                    if st.button("⬇️ Export LoRA", key="export_lora_btn"):
                        try:
                            zip_path = st.session_state.training_manager.export_lora(char_name)
                            st.success(f"LoRA exported to {zip_path}")
                        except Exception as e:
                            st.error(str(e))

                with col_c:
                    if st.button("⬇️ Export Latest Checkpoint", key="export_ckpt_btn"):
                        zip_path = st.session_state.training_manager.export_latest_checkpoint(char_name)
                        if zip_path:
                            st.success(f"Checkpoint exported to {zip_path}")
                        else:
                            st.info("No checkpoints found to export.")
        
        # ------------------------------------------------------------------
        # Inference-engine selector
        # ------------------------------------------------------------------

        # Build a map of available engines (name ➜ internal key)
        _engine_key_map = {
            "vLLM": "vllm",
            "LM Studio": "lmstudio",
        }

        # Cache available engines to avoid repeated testing
        if 'available_engines_cache' not in st.session_state:
            available_engine_names = []
            for friendly_name, key in _engine_key_map.items():
                try:
                    if InferenceEngineFactory.create_engine(key).is_available():
                        available_engine_names.append(friendly_name)
                except Exception:
                    continue
            
            if not available_engine_names:
                available_engine_names = ["LM Studio"]
            
            st.session_state.available_engines_cache = available_engine_names
        else:
            available_engine_names = st.session_state.available_engines_cache

        selected_engine_friendly = st.selectbox(
            "🛠️ Inference Engine",
            available_engine_names,
            index=available_engine_names.index(st.session_state.selected_engine)
            if st.session_state.selected_engine in available_engine_names else 0,
            help="Select which backend to use for text generation"
        )

        # Debug info
        current_actual_engine = st.session_state.dataset_manager.inference_engine.name
        
        # If the user picked a different engine we recreate the DatasetManager
        if selected_engine_friendly != st.session_state.selected_engine:
            internal_key = _engine_key_map[selected_engine_friendly]
            st.info(f"🔄 Switching from {st.session_state.selected_engine} to {selected_engine_friendly}...")
            st.session_state.selected_engine = selected_engine_friendly
            st.session_state.dataset_manager = DatasetManager(preferred_engine=internal_key)
            st.rerun()
        
        # Also check if the actual engine doesn't match what's selected
        elif current_actual_engine != selected_engine_friendly:
            internal_key = _engine_key_map[selected_engine_friendly]
            st.warning(f"⚠️ Engine mismatch detected! Selected: {selected_engine_friendly}, Actual: {current_actual_engine}. Fixing...")
            st.session_state.selected_engine = selected_engine_friendly
            st.session_state.dataset_manager = DatasetManager(preferred_engine=internal_key)
            st.rerun()
        
        # Status indicator
        status_colors = {
            'idle': '#94a3b8',
            'training': '#f59e0b',
            'paused': '#f97316',  # orange
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
                
                # Auto-load existing dataset if available
                existing_dataset = st.session_state.dataset_manager.load_dataset(character_data)
                if existing_dataset:
                    st.session_state.dataset_preview = existing_dataset
                    st.success(f"✅ Character card loaded with existing dataset ({len(existing_dataset)} samples)!")
                else:
                    st.session_state.dataset_preview = None
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
    
    # Check for existing dataset
    dataset_info = st.session_state.dataset_manager.get_dataset_info(st.session_state.current_character)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show existing dataset info if available
        if dataset_info['exists']:
            st.markdown("### 📂 Existing Dataset Found")
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Existing Samples", dataset_info['sample_count'])
            with col_info2:
                if st.button("📂 Load Existing", use_container_width=True):
                    existing_dataset = st.session_state.dataset_manager.load_dataset(st.session_state.current_character)
                    if existing_dataset:
                        st.session_state.dataset_preview = existing_dataset
                        st.success(f"✅ Loaded {len(existing_dataset)} existing samples!")
                        st.rerun()
            with col_info3:
                if st.button("🗑️ Reset Dataset", use_container_width=True):
                    if st.session_state.dataset_manager.delete_dataset(st.session_state.current_character):
                        st.session_state.dataset_preview = None
                        st.success("✅ Dataset reset! Generate a new one below.")
                        st.rerun()
            
            st.markdown("---")
        
        # ----------------------------
        # ✨ Augment Baseline Questions
        # ----------------------------
        st.markdown("### ✨ Augment Baseline Questions")
        with st.expander("Generate questions with AI"):
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
                                existing_dataset=st.session_state.dataset_preview
                            )
                        )
                        st.session_state.generated_questions = qs
                    finally:
                        loop.close()
                st.success(f"Generated {len(st.session_state.generated_questions)} questions!")

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
                        st.rerun()
                    else:
                        st.error("Failed to import dataset. Check file format.")

        # ------------------------------------------------------------
        
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
                # Allow generating much larger synthetic datasets (up to 20k samples)
                num_samples = st.slider(
                    "Total samples target",
                    min_value=20,
                    max_value=20000,
                    value=80,
                    step=20,
                    help="Desired total size of the synthetic dataset. Research shows 20-100 samples is optimal for character LoRAs, with 200-500 for more complex characters. Larger datasets risk overfitting."
                )
                temperature = st.slider("Temperature", 0.5, 1.2, 0.8, step=0.1)
            
            with col_b:
                top_p = st.slider("Top-p", 0.7, 1.0, 0.9, step=0.05)
            
            # Show incremental info
            if dataset_info['exists']:
                current_count = dataset_info['sample_count']
                if current_count < num_samples:
                    st.info(f"📈 Will generate {num_samples - current_count} new samples to reach {num_samples} total")
                else:
                    st.info(f"✅ Target already reached ({current_count} samples)")
            
            generate_button = st.form_submit_button(
                "🚀 Generate/Add to Dataset" if dataset_info['exists'] else "🚀 Generate Dataset", 
                use_container_width=True
            )
        
        if generate_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress callback (called by DatasetManager)
            def update_progress(p: float):
                """Update status text for current chunk progress."""
                status_text.text(
                    f"Chunk {chunk_idx+1}: {p*100:.1f}% • Total so far: {current_total}/{target_total} samples"
                )
            
            try:
                # Run generation in smaller logical chunks (>100 → 50-sample blocks)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    existing_count = dataset_info['sample_count'] if dataset_info['exists'] else 0
                    target_total = num_samples
                    current_total = existing_count
                    dataset = None
                    chunk_idx = 0
                    
                    # For vLLM, use larger chunks
                    chunk_size = 100 if st.session_state.selected_engine == "vLLM" else 50
                    
                    while current_total < target_total:
                        step_target = min(
                            current_total + chunk_size,
                            target_total,
                        )
                        dataset = loop.run_until_complete(
                            st.session_state.dataset_manager.generate_dataset(
                                st.session_state.current_character,
                                num_samples=step_target,
                                temperature=temperature,
                                top_p=top_p,
                                progress_callback=update_progress,
                                append_to_existing=True
                            )
                        )
                        current_total = len(dataset)
                        global_fraction = min(current_total / target_total, 1.0)  # Clamp to [0,1]
                        progress_bar.progress(global_fraction)
                        status_text.text(f"Total progress: {min(current_total, target_total)}/{target_total} samples")
                        chunk_idx += 1
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
        
        # Calculate training recommendations
        total_steps = (dataset_size * epochs) // (batch_size * gradient_accumulation)
        effective_batch_size = batch_size * gradient_accumulation
        
        # More nuanced overfitting risk calculation
        if dataset_size < 50:
            if total_steps > 300:
                overfitting_risk = "Very High"
            elif total_steps > 200:
                overfitting_risk = "High"
            else:
                overfitting_risk = "Medium"
        elif dataset_size < 100:
            if total_steps > 500:
                overfitting_risk = "High"
            elif total_steps > 300:
                overfitting_risk = "Medium"
            else:
                overfitting_risk = "Low"
        else:  # dataset_size >= 100
            if total_steps > 1000:
                overfitting_risk = "Medium"
            else:
                overfitting_risk = "Low"
        
        # Display recommendations
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0;">📊 Training Analysis</h4>
                <p><strong>Dataset Size:</strong> {dataset_size} samples</p>
                <p><strong>Total Steps:</strong> {total_steps}</p>
                <p><strong>Effective Batch Size:</strong> {effective_batch_size}</p>
                <p><strong>Overfitting Risk:</strong> <span style="color: {'#dc2626' if overfitting_risk == 'Very High' else '#ef4444' if overfitting_risk == 'High' else '#f59e0b' if overfitting_risk == 'Medium' else '#10b981'}">{overfitting_risk}</span></p>
                <p><strong>Est. Time:</strong> ~{max(1, total_steps * 2 // 60)} minutes</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Warnings based on configuration
        if overfitting_risk in ["High", "Very High"]:
            st.warning(f"⚠️ {overfitting_risk} overfitting risk! Consider: reducing epochs to {max(1, epochs-2)}, increasing dataset size, or lowering learning rate.")
        
        if dataset_size > 300:
            st.info("💡 Large dataset detected. Consider using rank 32 for more model capacity.")
        
        if learning_rate >= 5e-4:
            st.warning("⚠️ High learning rate may cause training instability. Consider 2e-4 for safer training.")
            
        if epochs > 8:
            st.warning("⚠️ High epoch count increases overfitting risk. 5-6 epochs is usually sufficient.")
        
        # Tips
        st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
                <h4 style="color: #10b981; margin-top: 0;">💡 Character LoRA Best Practices</h4>
                <ul style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0;">
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
            'resume_from_checkpoint': resume_ckpt
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
        
        selected_model = st.selectbox("Select Model", available_models)

        system_prompt_option = st.radio(
            "Choose system prompt strategy:",
            [
                "Default (Tokenizer's built-in)",
                "Empty (No system prompt)",
                "Roleplay Director",
                "Custom"
            ],
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
        st.markdown("### System Prompt Configuration")
        
        system_prompt = None
        if system_prompt_option == "Empty (No system prompt)":
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
        
        if system_prompt_option != "Default (Tokenizer's built-in)" and system_prompt_option != "Custom":
            st.code(f"System: {system_prompt}", language="text")
        elif system_prompt_option == "Custom" and system_prompt:
            st.code(f"System: {system_prompt}", language="text")

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
                        repetition_penalty=repetition_penalty,
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
        
        # Model comparison
        st.markdown("### Model Comparison")
        
        if len(available_models) > 1:
            st.info("Compare responses from different model checkpoints to evaluate training progress.")
            
            if st.button("📊 Compare Models"):
                st.info("Model comparison feature coming soon!")
        else:
            st.info("Train multiple checkpoints to enable model comparison.")

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