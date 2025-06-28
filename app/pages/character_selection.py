"""
🎭 Character Selection - Where Adventures Begin

This immersive character selection interface makes players feel like they're already
stepping into the story world. Characters aren't just cards to select - they're
living beings you encounter in the world's natural setting.
"""

import streamlit as st
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from app.utils.world_discovery import WorldDiscoveryManager
from app.utils.character.character import CharacterManager
from app.utils.character.models import CharacterCore
from app.utils.world import WorldManager
from app.utils.auth.models import UserRole


def init_managers():
    """Initialize all required managers"""
    if 'world_discovery_manager' not in st.session_state:
        st.session_state.world_discovery_manager = WorldDiscoveryManager()
    
    if 'character_manager' not in st.session_state:
        st.session_state.character_manager = CharacterManager()
    
    if 'world_manager' not in st.session_state:
        st.session_state.world_manager = WorldManager()
    
    return (
        st.session_state.world_discovery_manager,
        st.session_state.character_manager,
        st.session_state.world_manager
    )


def get_current_user():
    """Get current authenticated user"""
    try:
        if not st.session_state.get('authenticated', False):
            return None
        return st.session_state.get('user') or st.session_state.get('current_user')
    except Exception:
        return None


def get_session_info():
    """Get current session information from URL parameters or session state"""
    # In a real app, this would come from URL parameters
    # For now, we'll use session state or create a demo session
    
    if 'current_session_id' in st.session_state:
        return st.session_state.current_session_id
    
    # Create a demo session for testing
    user = get_current_user()
    if not user:
        return None
    
    discovery_manager, _, _ = init_managers()
    
    # Get a published world for demo
    worlds = discovery_manager.get_published_worlds(limit=1)
    if not worlds:
        return None
    
    world = worlds[0]
    
    # Create demo session
    session_result = discovery_manager.create_play_session(
        world_id=world['id'],
        user_id=user.id,
        session_name=f"Adventure in {world['name']}",
        privacy_setting="private"
    )
    
    if session_result.success:
        st.session_state.current_session_id = session_result.session_id
        st.session_state.current_world_info = world
        return session_result.session_id
    
    return None


def get_narrative_setting(world_info: Dict[str, Any]) -> Dict[str, str]:
    """Generate narrative framing based on world type and tags"""
    world_tags = world_info.get('tags', [])
    world_name = world_info['name']
    
    # Determine setting based on tags
    if 'fantasy' in world_tags:
        return {
            'location': 'The Wandering Griffin Tavern',
            'atmosphere': 'warm firelight dances across weathered wooden tables, casting long shadows',
            'entry_text': f"As twilight falls over {world_name}, you push open the heavy oak door of the tavern. The familiar creak announces your arrival, and several heads turn to regard you with curious eyes.",
            'instruction': "Look around the tavern. Each person here has their own story, their own skills. Choose wisely who will join your adventure...",
            'ambient_sound': '🔥 The fire crackles softly, punctuated by distant laughter and the clink of ale mugs.'
        }
    elif 'sci-fi' in world_tags:
        return {
            'location': 'Nexus Station Cantina',
            'atmosphere': 'holographic displays flicker with news from across the galaxy, casting blue light on diverse alien faces',
            'entry_text': f"The airlock hisses open as you enter the bustling hub of {world_name}. Beings from a dozen worlds mingle here, each with their own agenda.",
            'instruction': "Scan the cantina. Your mission requires the right team. Each individual here brings unique capabilities to the table...",
            'ambient_sound': '🌌 The hum of station life mingles with alien conversations and the soft beep of communication devices.'
        }
    elif 'modern' in world_tags:
        return {
            'location': 'The Underground Coffee Shop',
            'atmosphere': 'exposed brick walls lined with old books, while the aroma of fresh coffee mingles with whispered conversations',
            'entry_text': f"You descend the worn steps into the basement café that serves as an unofficial meeting ground for {world_name}'s most interesting residents.",
            'instruction': "Survey the room. Each person nursing their coffee might be exactly who you need for what's coming...",
            'ambient_sound': '☕ The espresso machine hisses occasionally, while jazz music plays softly in the background.'
        }
    else:
        # Generic setting
        return {
            'location': 'The Crossroads',
            'atmosphere': 'paths converge from all directions, bringing together travelers from distant lands',
            'entry_text': f"You arrive at the ancient crossroads at the heart of {world_name}, where fate brings together those meant to share adventures.",
            'instruction': "Observe those gathered here. Destiny has brought each soul to this moment for a reason...",
            'ambient_sound': '🌬️ A gentle breeze carries the sounds of distant lands and whispered promises of adventure.'
        }


def render_narrative_introduction(world_info: Dict[str, Any]):
    """Render the immersive narrative introduction"""
    setting = get_narrative_setting(world_info)
    
    # Hero section with atmospheric introduction
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(26,35,126,0.1) 0%, rgba(76,81,191,0.1) 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(76,81,191,0.2);
    ">
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="
                font-size: 2.5rem;
                font-weight: bold;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            ">
                📍 {setting['location']}
            </h1>
            <h2 style="
                font-size: 1.5rem;
                color: #666;
                font-weight: normal;
                margin-bottom: 2rem;
            ">
                {world_info['name']}
            </h2>
        </div>
        
        <div style="
            background: rgba(255,255,255,0.9);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #667eea;
            margin-bottom: 1.5rem;
        ">
            <p style="
                font-size: 1.1rem;
                line-height: 1.7;
                margin: 0;
                font-style: italic;
                color: #2c3e50;
            ">
                {setting['entry_text']}
            </p>
        </div>
        
        <div style="
            background: rgba(102,126,234,0.1);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        ">
            <p style="
                margin: 0;
                color: #4c51bf;
                font-weight: 500;
            ">
                {setting['instruction']}
            </p>
        </div>
        
        <div style="
            text-align: center;
            color: #718096;
            font-style: italic;
            font-size: 0.9rem;
        ">
            {setting['ambient_sound']}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_character_preview_modal(character: Dict[str, Any], world_info: Dict[str, Any]):
    """Render an interactive character preview modal"""
    modal_key = f"preview_{character['id']}"
    
    if st.session_state.get(modal_key, False):
        with st.container():
            st.markdown("---")
            st.markdown(f"### 🗣️ A Brief Encounter with {character['name']}")
            
            # Character speaks to you
            setting = get_narrative_setting(world_info)
            
            if 'fantasy' in world_info.get('tags', []):
                intro_line = f"*{character['name']} looks up from their ale as you approach*"
            elif 'sci-fi' in world_info.get('tags', []):
                intro_line = f"*{character['name']} adjusts their comm device as you approach*"
            else:
                intro_line = f"*{character['name']} glances up from their coffee as you approach*"
            
            st.markdown(f"*{intro_line}*")
            
            # Generate a character-appropriate greeting
            personality_tags = character.get('tags', [])
            
            if 'friendly' in personality_tags:
                greeting = f"\"Well hello there! Looking for some company on your adventures? I'd be happy to lend a hand.\""
            elif 'mysterious' in personality_tags:
                greeting = f"\"Hmm... you have the look of someone with interesting plans. Perhaps our paths should cross.\""
            elif 'warrior' in personality_tags:
                greeting = f"\"You look like you could use someone who knows their way around a fight. I'm interested.\""
            elif 'wise' in personality_tags:
                greeting = f"\"I sense you're embarking on something significant. My knowledge might prove useful.\""
            else:
                greeting = f"\"I don't often meet fellow travelers here. What brings you to seek companions?\""
            
            st.markdown(f"""
            <div style="
                background: #f7fafc;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #4299e1;
                margin: 1rem 0;
            ">
                <p style="margin: 0; font-style: italic; color: #2d3748;">
                    {greeting}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Character details in context
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**What you observe:**")
                st.write(f"• {character['description']}")
                
                if character.get('tags'):
                    st.markdown("**Notable traits:**")
                    trait_text = ", ".join([f"*{tag}*" for tag in character['tags'][:4]])
                    st.write(f"• {trait_text}")
            
            with col2:
                st.markdown("**Community reputation:**")
                st.metric("⭐ Rating", f"{character.get('avg_rating', 0):.1f}")
                st.metric("🎭 Adventures", character.get('session_count', 0))
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💬 Continue Talking", key=f"chat_{character['id']}"):
                    st.info("💭 This would open a full character preview chat...")
            
            with col2:
                if st.button("✅ Invite to Party", key=f"invite_{character['id']}", type="primary"):
                    add_character_to_party(character)
                    st.session_state[modal_key] = False
                    st.rerun()
            
            with col3:
                if st.button("🚶 Continue Looking", key=f"close_{character['id']}"):
                    st.session_state[modal_key] = False
                    st.rerun()


def render_immersive_character_card(character: Dict[str, Any], world_info: Dict[str, Any]):
    """Render a character as a living being in the world setting"""
    setting = get_narrative_setting(world_info)
    
    # Generate contextual appearance based on setting
    if 'fantasy' in world_info.get('tags', []):
        location_context = "seated at a corner table"
    elif 'sci-fi' in world_info.get('tags', []):
        location_context = "leaning against the bar"
    else:
        location_context = "by the window"
    
    with st.container():
        # Character card with immersive styling
        st.markdown(f"""
        <div style="
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
            cursor: pointer;
            margin-bottom: 1rem;
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 24px rgba(0,0,0,0.15)'"
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.1)'">
            
            <div style="display: flex; align-items: flex-start; gap: 1rem;">
                <div style="
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 1.5rem;
                    font-weight: bold;
                    flex-shrink: 0;
                ">
                    {character['name'][0]}
                </div>
                
                <div style="flex: 1;">
                    <h4 style="
                        margin: 0 0 0.5rem 0;
                        font-size: 1.2rem;
                        color: #2d3748;
                    ">
                        {character['name']}
                    </h4>
                    
                    <p style="
                        margin: 0 0 0.5rem 0;
                        color: #718096;
                        font-style: italic;
                        font-size: 0.9rem;
                    ">
                        *{location_context}, {character['description'][:80]}...*
                    </p>
                    
                    <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
        """ + "".join([
            f"""<span style="
                background: #edf2f7;
                color: #4a5568;
                padding: 0.2rem 0.5rem;
                border-radius: 12px;
                font-size: 0.8rem;
            ">{tag}</span>"""
            for tag in character.get('tags', [])[:3]
        ]) + f"""
                    </div>
                    
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-top: 0.5rem;
                    ">
                        <small style="color: #a0aec0;">
                            ⭐ {character.get('avg_rating', 0):.1f} • 
                            🎭 {character.get('session_count', 0)} adventures
                        </small>
                        
                        <small style="color: #718096;">
                            by {character.get('creator_username', 'Unknown')}
                        </small>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if st.button(f"👁️ Observe", key=f"observe_{character['id']}", use_container_width=True):
                modal_key = f"preview_{character['id']}"
                st.session_state[modal_key] = True
                st.rerun()
        
        with col2:
            if st.button(f"🤝 Approach", key=f"approach_{character['id']}", use_container_width=True, type="primary"):
                add_character_to_party(character)
                st.rerun()
        
        with col3:
            party_characters = st.session_state.get('selected_characters', [])
            is_selected = any(c['id'] == character['id'] for c in party_characters)
            
            if is_selected:
                st.markdown("✅", help="In party")
            else:
                if st.button("❤️", key=f"fav_{character['id']}", help="Favorite"):
                    st.success("Added to favorites!")


def add_character_to_party(character: Dict[str, Any]):
    """Add a character to the current party"""
    if 'selected_characters' not in st.session_state:
        st.session_state.selected_characters = []
    
    # Check if already selected
    if any(c['id'] == character['id'] for c in st.session_state.selected_characters):
        st.warning(f"{character['name']} is already in your party!")
        return
    
    # Check party limit
    if len(st.session_state.selected_characters) >= 4:
        st.warning("Your party is full! (Maximum 4 characters)")
        return
    
    st.session_state.selected_characters.append(character)
    st.success(f"🎉 {character['name']} joins your party!")


def render_party_panel():
    """Render the current party selection panel"""
    party_characters = st.session_state.get('selected_characters', [])
    
    st.markdown("### 🎭 Your Adventuring Party")
    
    if not party_characters:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 2rem;
            background: #f7fafc;
            border-radius: 8px;
            border: 2px dashed #cbd5e0;
        ">
            <p style="color: #718096; margin: 0;">
                <strong>No companions chosen yet</strong><br>
                <small>Select characters from the tavern to join your adventure</small>
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Display party members
    for i, character in enumerate(party_characters):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**{character['name']}**")
            tags_display = " • ".join(character.get('tags', [])[:2])
            st.markdown(f"*{tags_display}*")
        
        with col2:
            st.metric("⭐", f"{character.get('avg_rating', 0):.1f}")
        
        with col3:
            if st.button("❌", key=f"remove_{character['id']}", help="Remove from party"):
                st.session_state.selected_characters.pop(i)
                st.rerun()
    
    # Party analysis
    if len(party_characters) >= 2:
        discovery_manager, _, _ = init_managers()
        character_ids = [c['id'] for c in party_characters]
        compatibility = discovery_manager.analyze_character_compatibility(character_ids)
        
        color = "green" if compatibility.score > 0.6 else "orange" if compatibility.score > 0.4 else "red"
        
        st.markdown("**Party Dynamics:**")
        st.markdown(f"""
        <div style="
            background: #{color}20;
            border-left: 4px solid {color};
            padding: 0.5rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        ">
            <strong style="color: {color};">
                {compatibility.relationship_type.title()} ({compatibility.score:.1f}/1.0)
            </strong><br>
            <small>{compatibility.reasoning}</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Action buttons
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Begin Adventure", disabled=len(party_characters) == 0, use_container_width=True, type="primary"):
            launch_adventure()
    
    with col2:
        if st.button("🔄 Clear Party", disabled=len(party_characters) == 0, use_container_width=True):
            st.session_state.selected_characters = []
            st.rerun()


def launch_adventure():
    """Launch the adventure with selected characters"""
    party_characters = st.session_state.get('selected_characters', [])
    session_id = st.session_state.get('current_session_id')
    
    if not party_characters or not session_id:
        st.error("Cannot launch adventure: missing party or session")
        return
    
    discovery_manager, _, _ = init_managers()
    
    # Add characters to session
    for character in party_characters:
        success = discovery_manager.add_character_to_session(
            session_id=session_id,
            character_name=character['name'],
            character_data=character
        )
        
        if not success:
            st.error(f"Failed to add {character['name']} to session")
            return
    
    # Update session status to active
    discovery_manager.update_session_status(session_id, "active")
    
    # Success!
    st.success("🎉 Adventure begins!")
    st.balloons()
    
    st.markdown("""
    ### 🌟 Your Adventure Awaits!
    
    Your carefully chosen companions are ready to face whatever challenges lie ahead.
    The bonds you forge and the stories you create together will be remembered forever.
    
    *Redirecting to your adventure...*
    """)
    
    # In a real app, redirect to the runtime interface
    st.info("🎮 This would redirect to the platform runtime interface (R3-1.0)")


def render_character_filters():
    """Render character filtering controls"""
    st.sidebar.markdown("### 🔍 Find Your Companions")
    
    # Search
    search_query = st.sidebar.text_input(
        "Search characters",
        placeholder="Name or description...",
        key="character_search"
    )
    
    # Personality filters
    personality_tags = [
        "friendly", "wise", "brave", "mysterious", "cheerful",
        "serious", "magical", "warrior", "healer", "scholar"
    ]
    
    selected_traits = st.sidebar.multiselect(
        "Personality traits",
        personality_tags,
        key="trait_filter"
    )
    
    # Sort options
    sort_options = {
        "popular": "🔥 Most Popular",
        "rating": "⭐ Highest Rated",
        "recent": "🕒 Recently Added"
    }
    
    sort_by = st.sidebar.selectbox(
        "Sort by",
        options=list(sort_options.keys()),
        format_func=lambda x: sort_options[x],
        key="character_sort"
    )
    
    return search_query, selected_traits, sort_by


def page_character_selection():
    """Main character selection page"""
    
    # Check authentication
    user = get_current_user()
    if not user:
        st.error("🚪 Please log in to select characters")
        if st.button("Go to Login"):
            st.switch_page("app/pages/login.py")
        return
    
    # Get session info
    session_id = get_session_info()
    if not session_id:
        st.error("❌ No valid session found. Please start from world discovery.")
        if st.button("🌍 Go to World Discovery"):
            st.switch_page("app/pages/world_discovery.py")
        return
    
    # Initialize managers
    discovery_manager, char_manager, world_manager = init_managers()
    
    # Get world info
    world_info = st.session_state.get('current_world_info')
    if not world_info:
        st.error("❌ World information not found")
        return
    
    # Render narrative introduction
    render_narrative_introduction(world_info)
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Render character filters
        search_query, selected_traits, sort_by = render_character_filters()
        
        # Get available characters
        characters = discovery_manager.get_published_characters(
            world_id=world_info['id'],
            search_query=search_query,
            tags=selected_traits,
            sort_by=sort_by
        )
        
        if not characters:
            st.markdown("""
            <div style="
                text-align: center;
                padding: 3rem 2rem;
                color: #666;
                background: #f7fafc;
                border-radius: 12px;
                border: 2px dashed #cbd5e0;
            ">
                <h3>🌅 The tavern is quiet tonight...</h3>
                <p>No companions match your search. Try adjusting your filters, or perhaps check back later when more travelers arrive.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"### 👥 Available Companions ({len(characters)} found)")
            
            # Render character cards
            for character in characters:
                render_immersive_character_card(character, world_info)
                
                # Render preview modal if activated
                modal_key = f"preview_{character['id']}"
                if st.session_state.get(modal_key, False):
                    render_character_preview_modal(character, world_info)
    
    with col2:
        # Party panel
        render_party_panel()
        
        # Tips and atmosphere
        st.markdown("---")
        st.markdown("### 💡 Adventurer's Tips")
        
        tips = [
            "🎭 Different personalities create unique group dynamics",
            "⚔️ Balance your party with diverse skills",
            "🌟 Higher-rated characters are community favorites",
            "💬 Use 'Observe' to get a feel for each character",
            "🤝 Great adventures come from unexpected partnerships"
        ]
        
        for tip in tips:
            st.markdown(f"- {tip}")


# CSS styling for immersive experience
st.markdown("""
<style>
    /* Global immersive styling */
    .main {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Character card hover effects */
    .character-card {
        transition: all 0.3s ease;
    }
    
    .character-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.15);
    }
    
    /* Narrative text styling */
    .narrative-text {
        font-family: 'Georgia', serif;
        line-height: 1.7;
    }
    
    /* Hide default Streamlit elements for immersion */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
</style>
""", unsafe_allow_html=True)


# Run the page
if __name__ == "__main__":
    page_character_selection() 