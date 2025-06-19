"""
🌍 World Management Page

This page allows creators to:
1. Create and select worlds
2. Edit world lore (facts, timeline, places) 
3. Use AI helpers for content generation

Following the TDD approach, this starts as a minimal implementation.
"""

import streamlit as st
from typing import Dict, Any
import pandas as pd

def page_world_management():
    """World Management page - create and edit worlds"""
    
    st.markdown('<h2 class="gradient-text">🌍 World Management</h2>', unsafe_allow_html=True)
    
    # Ensure WorldManager is available
    if 'world_manager' not in st.session_state:
        st.error("WorldManager not initialized. Please restart the application.")
        return
    
    wm = st.session_state.world_manager
    
    # Sidebar world selection
    with st.sidebar:
        st.markdown("### World Selection")
        
        # Get available worlds
        worlds = wm.list_worlds()
        
        if worlds:
            selected_world = st.selectbox("Select World", worlds, key="world_selector")
            
            # Load selected world
            if selected_world:
                current_lore = wm.load_world(selected_world)
                if current_lore:
                    st.success(f"✅ Loaded: {selected_world}")
                else:
                    st.error(f"❌ Failed to load: {selected_world}")
        else:
            st.info("No worlds found. Create your first world!")
            selected_world = None
            current_lore = None
        
        # New World button
        if st.button("➕ New World", use_container_width=True):
            st.session_state.show_new_world_dialog = True
    
    # New World Dialog
    if st.session_state.get('show_new_world_dialog', False):
        with st.container():
            st.markdown("### Create New World")
            new_world_name = st.text_input("World Name", placeholder="Enter world name...")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create World"):
                    if new_world_name and new_world_name.strip():
                        success = wm.create_world(new_world_name.strip())
                        if success:
                            st.success(f"✅ Created world: {new_world_name}")
                            st.session_state.show_new_world_dialog = False
                            st.rerun()
                        else:
                            st.error("❌ Failed to create world")
                    else:
                        st.error("Please enter a valid world name")
            
            with col2:
                if st.button("Cancel"):
                    st.session_state.show_new_world_dialog = False
                    st.rerun()
    
    # Main content area - tabs
    if worlds and selected_world and current_lore:
        
        # Create tabs for Facts, Timeline, Places
        tab1, tab2, tab3 = st.tabs(["📊 Facts", "📅 Timeline", "🏰 Places"])
        
        with tab1:
            st.markdown("### World Facts")
            
            # Convert facts dict to DataFrame for editing
            if current_lore.facts:
                facts_df = pd.DataFrame(list(current_lore.facts.items()), columns=["Key", "Value"])
            else:
                facts_df = pd.DataFrame(columns=["Key", "Value"])
            
            # Data editor for facts
            edited_facts = st.data_editor(
                facts_df,
                num_rows="dynamic",
                use_container_width=True,
                key="facts_editor"
            )
            
            # AI Helper buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🤖 Draft Back-story"):
                    st.info("AI back-story generation coming soon!")
            
            with col2:
                if st.button("🤖 Fill Empty Fields"):
                    st.info("AI field filling coming soon!")
        
        with tab2:
            st.markdown("### World Timeline")
            
            # Convert timeline to DataFrame
            if current_lore.timeline:
                timeline_data = [(event.year, event.event) for event in current_lore.timeline]
                timeline_df = pd.DataFrame(timeline_data, columns=["Year", "Event"])
            else:
                timeline_df = pd.DataFrame(columns=["Year", "Event"])
            
            # Data editor for timeline
            edited_timeline = st.data_editor(
                timeline_df,
                num_rows="dynamic",
                use_container_width=True,
                key="timeline_editor"
            )
            
            # AI Helper button
            if st.button("🤖 Generate 5 Historical Events"):
                st.info("AI historical events generation coming soon!")
        
        with tab3:
            st.markdown("### World Places")
            
            # List places
            if current_lore.places:
                for i, place in enumerate(current_lore.places):
                    with st.expander(f"🏰 {place.name}"):
                        st.write(f"**Description:** {place.description}")
                        if place.npcs:
                            st.write("**NPCs:**")
                            for npc in place.npcs:
                                st.write(f"- {npc.name}: {npc.description}")
                        if place.events:
                            st.write("**Events:**")
                            for event in place.events:
                                st.write(f"- {event.name}: {event.description}")
            else:
                st.info("No places defined yet. Add some places to bring your world to life!")
            
            # AI Helper button
            if st.button("🤖 Suggest NPCs"):
                st.info("AI NPC suggestions coming soon!")
    
    else:
        # No world selected
        st.info("Select a world from the sidebar to start editing, or create a new world!")


# Run the page function if this file is executed directly (for testing)
if __name__ == "__main__":
    page_world_management() 