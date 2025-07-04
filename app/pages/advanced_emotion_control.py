"""
Advanced Emotion Control Page

This page provides access to the advanced emotion control and narrative context integration
system. It allows users to test and configure sophisticated emotion blending, prosody
control, and temporal consistency features.
"""

import streamlit as st
import sys
from pathlib import Path

# Add the app directory to the path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from components.advanced_emotion_control import render_advanced_emotion_control

# Page configuration
st.set_page_config(
    page_title="Advanced Emotion Control",
    page_icon="🎭",
    layout="wide"
)

# Main page content
def main():
    """Main function for the advanced emotion control page."""
    
    # Page header
    st.title("🎭 Advanced Emotion Control")
    st.markdown("""
    **Sophisticated emotion control system with narrative context integration**
    
    This advanced system provides:
    - **Emotion Blending**: Combine multiple emotions for complex emotional states
    - **Narrative Context**: Analyze story state and adjust emotions accordingly  
    - **Prosody Control**: Fine-tune speaking rate, pitch, and emphasis
    - **Temporal Consistency**: Track emotional progression over time
    - **Living Interface**: Integration with tri-head architecture
    """)
    
    # Check authentication
    if not st.session_state.get('authenticated', False):
        st.warning("🔐 Please log in to access the Advanced Emotion Control system.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 Go to Login"):
                st.switch_page("pages/login.py")
        with col2:
            if st.button("📝 Register"):
                st.switch_page("pages/register.py")
        return
    
    # Render the advanced emotion control interface
    try:
        render_advanced_emotion_control()
    except Exception as e:
        st.error(f"Error loading emotion control interface: {e}")
        st.info("Please ensure all required services are properly configured.")

if __name__ == "__main__":
    main() 