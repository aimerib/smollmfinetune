"""
🗨️ Conversational Character Builder

Transform character creation from tabbed form-filling into a guided conversational flow 
where AI interviews the user to discover the character's "soul."
"""

import streamlit as st
import asyncio
from pathlib import Path
from typing import Optional
import logging

from utils.character import CharacterManager
from utils.character.models import CharacterCore
from utils.character.character_intelligence import CharacterIntelligenceService
from utils.world import WorldManager
from components.character_creation.conversational_builder import render_conversational_builder, reset_conversation
from components.character_creation.character_synthesis_preview import render_character_synthesis_preview

logger = logging.getLogger(__name__)


def page_character_builder():
    """Main Conversational Character Builder page"""
    
    st.markdown('<h2 class="gradient-text">🗨️ Conversational Character Builder</h2>', unsafe_allow_html=True)
    
    # Initialize managers if not available
    if 'character_manager' not in st.session_state:
        st.session_state.character_manager = CharacterManager()
    
    if 'world_manager' not in st.session_state:
        st.session_state.world_manager = WorldManager()
    
    # Initialize intelligence service
    if 'character_intelligence' not in st.session_state:
        st.session_state.character_intelligence = CharacterIntelligenceService(
            world_manager=st.session_state.world_manager
        )
    
    # Show introduction
    st.markdown("""
        <div class="custom-card">
            <h3 style="color: white; margin-top: 0;">🎭 AI-Guided Character Creation</h3>
            <p style="color: rgba(255,255,255,0.8);">
                Let our AI guide you through creating a rich, compelling character through natural conversation. 
                No more blank forms - just tell us about your character and watch them come to life!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Check if we have a current world
    char_manager = st.session_state.character_manager
    world_manager = st.session_state.world_manager
    
    current_world = char_manager.get_current_world()
    if not current_world:
        st.warning("⚠️ No world selected. Please select a world first.")
        if st.button("🌍 Go to World Management", type="primary"):
            st.switch_page("pages/world_management.py")
        return
    
    st.info(f"🌍 Creating character in world: **{current_world}**")
    
    # Main conversation interface
    completed_character = render_conversational_builder(
        world_manager=world_manager,
        character_name=st.session_state.get('new_character_name', '')
    )
    
    # Handle completed character
    if completed_character:
        st.success("🎉 Character creation completed!")
        
        # Save the character
        if st.button("💾 Save Character", type="primary", use_container_width=True):
            if char_manager.save_character(completed_character):
                st.success(f"✅ Character '{completed_character.name}' saved successfully!")
                
                # Set as current character
                st.session_state.current_character_core = completed_character
                st.session_state.selected_character = completed_character.name
                
                # Clear conversation state
                reset_conversation()
                
                # Offer next steps
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📋 Edit in Management Studio", use_container_width=True):
                        st.switch_page("pages/character_management.py")
                
                with col2:
                    if st.button("🎨 Generate Dataset", use_container_width=True):
                        st.switch_page("pages/dataset_studio.py")
                
                with col3:
                    if st.button("🗨️ Create Another Character", use_container_width=True):
                        reset_conversation()
                        st.rerun()
            else:
                st.error("❌ Failed to save character")
    
    # Show help and tips in sidebar
    with st.sidebar:
        st.markdown("### 💡 Conversation Tips")
        st.markdown("""
            **Be Natural**: Just describe your character as you would to a friend
            
            **Share Details**: The more you tell us, the richer your character becomes
            
            **Ask Questions**: Feel free to ask the AI for suggestions or clarification
            
            **Take Your Time**: There's no rush - develop your character at your own pace
        """)
        
        st.markdown("### 🎯 What We'll Discover")
        st.markdown("""
            • **Personality**: Core traits and behaviors
            • **Background**: History and formative experiences  
            • **Goals**: What drives your character
            • **Relationships**: Important connections
            • **Voice**: How they speak and express themselves
        """)
        
        # Reset conversation button
        if st.button("🔄 Start Over", help="Reset the conversation and start fresh"):
            reset_conversation()
            st.rerun()


# Run the page function if this file is executed directly (for testing)
if __name__ == "__main__":
    page_character_builder() 