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

from utils.openai_client import get_client
from utils.character import CharacterManager
from utils.character.models import CharacterCore
from utils.character.character_intelligence import CharacterIntelligenceService
from utils.world import WorldManager
from components.character_creation.conversational_builder import render_conversational_builder, reset_conversation
from components.character_creation.character_synthesis_preview import render_character_synthesis_preview

logger = logging.getLogger(__name__)


def page_character_builder():
    """Main Conversational Character Builder page"""
    
    # Check if we're enhancing an existing character from import
    enhance_existing = st.session_state.get('enhance_existing_character')
    
    if enhance_existing:
        st.markdown('<h2 class="gradient-text">🗨️ Character Enhancement Interview</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 class="gradient-text">🗨️ Conversational Character Builder</h2>', unsafe_allow_html=True)
    
    # Initialize managers if not available
    if 'world_manager' not in st.session_state:
        st.session_state.world_manager = WorldManager()
    
    if 'character_manager' not in st.session_state:
        st.session_state.character_manager = CharacterManager(world_manager=st.session_state.world_manager, client=get_client())
    
    # Initialize intelligence service
    if 'character_intelligence' not in st.session_state:
        st.session_state.character_intelligence = CharacterIntelligenceService(
            world_manager=st.session_state.world_manager
        )
    
    # Show different introductions based on mode
    if enhance_existing:
        st.markdown(f"""
            <div class="custom-card">
                <h3 style="color: white; margin-top: 0;">🎭 Character Enhancement Interview</h3>
                <p style="color: rgba(255,255,255,0.8);">
                    Let's dive deeper into <strong>{enhance_existing.name}</strong>'s personality! We've created a foundation 
                    from your imported character, now let's discover their unique voice, motivations, and inner world 
                    through guided conversation.
                </p>
                <div style="background: rgba(168, 85, 247, 0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                    <strong>✨ Current Character Foundation:</strong><br>
                    📝 Description: {len(enhance_existing.description)} characters<br>
                    🎯 Goals: {len(enhance_existing.goals)} defined<br>
                    🤝 Relationships: {len(enhance_existing.relationships)} connections<br>
                    🧠 Personality: Big Five traits analyzed
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
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
    
    if enhance_existing:
        st.info(f"🌍 Enhancing character in world: **{current_world}** | Character: **{enhance_existing.name}**")
    else:
        st.info(f"🌍 Creating character in world: **{current_world}**")
    
    # Main conversation interface
    # Pass the existing character if enhancing
    initial_character = enhance_existing if enhance_existing else None
    character_name = enhance_existing.name if enhance_existing else st.session_state.get('new_character_name', '')
    
    completed_character = render_conversational_builder(
        world_manager=world_manager,
        character_name=character_name,
        existing_character=initial_character
    )
    
    # Handle completed character
    if completed_character:
        if enhance_existing:
            st.success("🎉 Character enhancement completed!")
        else:
            st.success("🎉 Character creation completed!")
        
        # Save the character
        save_button_text = "💾 Save Enhanced Character" if enhance_existing else "💾 Save Character"
        if st.button(save_button_text, type="primary", use_container_width=True):
            if char_manager.save_character(completed_character):
                success_msg = f"✅ Character '{completed_character.name}' enhanced successfully!" if enhance_existing else f"✅ Character '{completed_character.name}' saved successfully!"
                st.success(success_msg)
                
                # Set as current character
                st.session_state.current_character_core = completed_character
                st.session_state.selected_character = completed_character.name
                
                # Clear conversation state and enhancement flag
                reset_conversation()
                if 'enhance_existing_character' in st.session_state:
                    del st.session_state.enhance_existing_character
                
                # Show enhancement comparison if this was an enhancement
                if enhance_existing:
                    st.markdown("### 🔄 Enhancement Summary")
                    col_before, col_after = st.columns(2)
                    
                    with col_before:
                        st.markdown("**Before Enhancement:**")
                        st.markdown(f"• Description: {len(enhance_existing.description)} chars")
                        st.markdown(f"• Goals: {len(enhance_existing.goals)}")
                        st.markdown(f"• Relationships: {len(enhance_existing.relationships)}")
                    
                    with col_after:
                        st.markdown("**After Enhancement:**")
                        st.markdown(f"• Description: {len(completed_character.description)} chars")
                        st.markdown(f"• Goals: {len(completed_character.goals)}")
                        st.markdown(f"• Relationships: {len(completed_character.relationships)}")
                
                # Offer next steps
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📋 Edit in Management Studio", use_container_width=True):
                        st.switch_page("pages/character_management.py")
                
                with col2:
                    if st.button("🎨 Generate Dataset", use_container_width=True):
                        st.switch_page("pages/dataset_studio.py")
                
                with col3:
                    if enhance_existing:
                        if st.button("🗨️ Enhance Another Character", use_container_width=True):
                            reset_conversation()
                            if 'enhance_existing_character' in st.session_state:
                                del st.session_state.enhance_existing_character
                            st.switch_page("pages/character_upload.py")
                    else:
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