"""
🌍 World Discovery Platform

The beautiful world discovery interface - players' first glimpse into the platform.
Features Netflix-style browsing, Steam-like world cards, and elegant filtering.
"""

import streamlit as st
import json
from datetime import datetime
from typing import List, Dict, Any
from app.utils.auth.models import UserRole
from app.utils.world_discovery import WorldDiscoveryManager


def init_discovery_manager():
    """Initialize the world discovery manager"""
    # Handle both testing mode (dict) and real session_state
    try:
        # Check if we're in testing mode and session_state is a dict
        if isinstance(st.session_state, dict) and 'world_discovery_manager' in st.session_state:
            return st.session_state['world_discovery_manager']
        # Normal session_state object
        elif hasattr(st.session_state, 'world_discovery_manager'):
            return st.session_state.world_discovery_manager
        elif 'world_discovery_manager' not in st.session_state:
            st.session_state.world_discovery_manager = WorldDiscoveryManager()
        return st.session_state.world_discovery_manager
    except Exception:
        # Fallback - create a new manager for testing
        return WorldDiscoveryManager()


def get_current_user():
    """Get current authenticated user"""
    try:
        if not st.session_state.get('authenticated', False):
            return None
        # Check for both 'user' (testing) and 'current_user' (real app)
        return st.session_state.get('user') or st.session_state.get('current_user')
    except Exception:
        return None


def render_world_card(world: Dict[str, Any], col):
    """Render a beautiful world card"""
    with col:
        # Card container with custom styling
        card_container = st.container()
        
        with card_container:
            # Thumbnail or placeholder
            if world.get('thumbnail_url'):
                st.image(world['thumbnail_url'], use_container_width=True)
            else:
                # Beautiful placeholder with gradient
                st.markdown(f"""
                <div style="
                    width: 100%;
                    height: 200px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 1rem;
                ">
                    🌍 {world['name'][0] if world['name'] else '?'}
                </div>
                """, unsafe_allow_html=True)
            
            # World title with featured badge
            title_html = f"### {world['name']}"
            if world.get('featured'):
                title_html = f"🌟 ### {world['name']}"
            st.markdown(title_html)
            
            # Description
            st.write(world['description'][:100] + "..." if len(world['description']) > 100 else world['description'])
            
            # Tags with styled chips
            if world['tags']:
                tags_html = " ".join([
                    f'<span style="background-color: #f0f2f6; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-right: 4px;">{tag}</span>'
                    for tag in world['tags'][:3]  # Show max 3 tags
                ])
                st.markdown(tags_html, unsafe_allow_html=True)
            
            # Stats row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rating", f"⭐ {world.get('avg_rating', 0):.1f}", f"({world.get('total_ratings', 0)})")
            with col2:
                st.metric("Sessions", world.get('total_sessions', 0))
            with col3:
                st.write(f"**By:** {world.get('creator_username', 'Unknown')}")
            
            # Action buttons
            enter_col, fav_col = st.columns([2, 1])
            with enter_col:
                if st.button(f"🚀 Enter World", key=f"enter_{world['id']}", use_container_width=True):
                    handle_enter_world(world)
            with fav_col:
                if st.button("❤️", key=f"fav_{world['id']}", help="Add to favorites"):
                    st.success("Added to favorites!")


def handle_enter_world(world: Dict[str, Any]):
    """Handle entering a world - create session and redirect"""
    user = get_current_user()
    if not user:
        st.error("Please log in to enter worlds")
        return
    
    discovery_manager = init_discovery_manager()
    
    # Create play session
    session_name = f"Adventure in {world['name']}"
    result = discovery_manager.create_play_session(
        world_id=world['id'],
        user_id=user.id,
        session_name=session_name,
        privacy_setting="private"
    )
    
    if result.success:
        st.success(f"🎉 Welcome to {world['name']}!")
        st.info("Redirecting to character selection...")
        # In a real app, this would redirect to character selection page
        # For now, we'll show session details
        st.json({
            "session_id": result.session_id,
            "world_name": world['name'],
            "session_name": session_name,
            "next_step": "Character Selection"
        })
    else:
        st.error(f"Failed to create session: {result.error_message}")


def render_publish_dialog():
    """Render the publish world dialog for creators"""
    if not st.session_state.get('show_publish_dialog', False):
        return
    
    user = get_current_user()
    if not user or user.role not in [UserRole.CREATOR, UserRole.ADMIN]:
        return
    
    st.markdown("### 📢 Publish World")
    
    # Get user's unpublished worlds
    world_manager = st.session_state.get('world_manager')
    if not world_manager:
        st.error("World manager not available")
        return
    
    user_worlds = world_manager.list_worlds()
    discovery_manager = init_discovery_manager()
    published_worlds = discovery_manager.get_published_worlds()
    published_names = {w['name'] for w in published_worlds}
    
    unpublished_worlds = [w for w in user_worlds if w not in published_names]
    
    if not unpublished_worlds:
        st.info("All your worlds are already published!")
        if st.button("Close"):
            st.session_state.show_publish_dialog = False
            st.rerun()
        return
    
    with st.form("publish_world_form"):
        selected_world = st.selectbox("Select World", unpublished_worlds)
        description = st.text_area(
            "Description",
            placeholder="Describe your world to attract players...",
            help="A compelling description helps players discover your world"
        )
        
        # Tag selection with common tags
        common_tags = [
            "fantasy", "sci-fi", "modern", "historical", "horror", "romance",
            "adventure", "mystery", "comedy", "drama", "action", "magic",
            "space", "cyberpunk", "steampunk", "medieval", "post-apocalyptic"
        ]
        
        selected_tags = st.multiselect(
            "Tags (select up to 5)",
            common_tags,
            help="Tags help players find worlds they're interested in"
        )
        
        # Custom tags
        custom_tags = st.text_input(
            "Custom Tags",
            placeholder="Add custom tags separated by commas",
            help="Add specific tags not in the list above"
        )
        
        thumbnail_url = st.text_input(
            "Thumbnail URL (optional)",
            placeholder="https://example.com/world-image.jpg",
            help="A beautiful thumbnail makes your world stand out"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            publish_btn = st.form_submit_button("🌟 Publish World", use_container_width=True)
        with col2:
            cancel_btn = st.form_submit_button("Cancel", use_container_width=True)
        
        if cancel_btn:
            st.session_state.show_publish_dialog = False
            st.rerun()
        
        if publish_btn:
            if not selected_world or not description:
                st.error("Please select a world and provide a description")
            elif len(selected_tags) > 5:
                st.error("Please select maximum 5 tags")
            else:
                # Process custom tags
                all_tags = selected_tags.copy()
                if custom_tags:
                    custom_tag_list = [tag.strip().lower() for tag in custom_tags.split(",") if tag.strip()]
                    all_tags.extend(custom_tag_list)
                
                # Publish the world
                result = discovery_manager.publish_world(
                    world_name=selected_world,
                    user_id=user.id,
                    description=description,
                    tags=all_tags,
                    thumbnail_url=thumbnail_url if thumbnail_url else None
                )
                
                if result.success:
                    st.success(f"🎉 Successfully published '{selected_world}'!")
                    st.session_state.show_publish_dialog = False
                    st.rerun()
                else:
                    st.error(f"Failed to publish: {result.error_message}")


def render_hero_section(featured_worlds: List[Dict[str, Any]]):
    """Render the hero section with featured worlds"""
    if not featured_worlds:
        return
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 3rem; font-weight: bold; margin-bottom: 1rem;">
            🌍 Discover Worlds
        </h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
            Step into infinite adventures. Choose your world, select your character, create your story.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Featured worlds carousel (simplified for now)
    if len(featured_worlds) > 0:
        st.markdown("### 🌟 Featured Worlds")
        
        # Show featured worlds in larger cards
        for i in range(0, len(featured_worlds), 2):  # 2 per row
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(featured_worlds):
                    render_world_card(featured_worlds[i + j], col)


def render_filter_controls():
    """Render search and filter controls"""
    st.markdown("### 🔍 Explore Worlds")
    
    # Search and filter row
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        search_query = st.text_input(
            "Search worlds",
            placeholder="Search by name or description...",
            key="world_search"
        )
    
    with col2:
        tag_options = [
            "fantasy", "sci-fi", "modern", "historical", "horror", "romance",
            "adventure", "mystery", "comedy", "drama", "action", "magic"
        ]
        selected_tags = st.multiselect(
            "Filter by tags",
            tag_options,
            key="tag_filter"
        )
    
    with col3:
        sort_options = {
            "popular": "🔥 Popular",
            "recent": "🕒 Recent", 
            "rating": "⭐ Top Rated"
        }
        sort_by = st.selectbox(
            "Sort by",
            options=list(sort_options.keys()),
            format_func=lambda x: sort_options[x],
            key="sort_by"
        )
    
    return search_query, selected_tags, sort_by


def page_world_discovery():
    """Main world discovery page"""
    
    # Check authentication
    user = get_current_user()
    if not user:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h1>🌍 Welcome to World Discovery</h1>
            <p style="font-size: 1.2rem; margin: 2rem 0;">
                Please log in to discover and explore amazing worlds created by our community.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚪 Go to Login", use_container_width=True):
                st.switch_page("app/pages/login.py")
        return
    
    # Initialize managers
    discovery_manager = init_discovery_manager()
    
    # Get published worlds
    all_worlds = discovery_manager.get_published_worlds()
    featured_worlds = [w for w in all_worlds if w.get('featured', False)]
    
    # Hero section
    render_hero_section(featured_worlds)
    
    # Publish button for creators
    if user.role in [UserRole.CREATOR, UserRole.ADMIN]:
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col4:
            if st.button("📢 Publish World", use_container_width=True):
                st.session_state.show_publish_dialog = True
                st.rerun()
    
    # Publish dialog
    render_publish_dialog()
    
    # Search and filter controls
    search_query, selected_tags, sort_by = render_filter_controls()
    
    # Get filtered worlds
    filtered_worlds = discovery_manager.get_published_worlds(
        search_query=search_query,
        tags=selected_tags,
        featured_only=False,
        sort_by=sort_by
    )
    
    # Show results count
    if search_query or selected_tags:
        st.write(f"Found {len(filtered_worlds)} worlds")
    
    # Render world grid
    if filtered_worlds:
        st.markdown("---")
        
        # World grid (3 per row)
        for i in range(0, len(filtered_worlds), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(filtered_worlds):
                    render_world_card(filtered_worlds[i + j], col)
    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #666;">
            <h3>🔍 No worlds found</h3>
            <p>Try adjusting your search or filters to discover more worlds.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recently played sessions (if user has any)
    if user:
        recent_sessions = discovery_manager.get_user_sessions(user.id)
        if recent_sessions:
            st.markdown("---")
            st.markdown("### 🎮 Continue Your Adventures")
            
            for session in recent_sessions[:3]:  # Show last 3 sessions
                with st.expander(f"📖 {session['session_name']} - {session['world_name']}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**World:** {session['world_description']}")
                        st.write(f"**Last played:** {session['last_active'][:10]}")
                    with col2:
                        if st.button("Continue", key=f"continue_{session['id']}"):
                            st.info("Resuming session...")


# CSS styling
st.markdown("""
<style>
    /* Global styles for the discovery page */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Card styling */
    .element-container {
        border-radius: 12px;
    }
    
    /* Metric styling */
    .metric-container {
        background: #f8f9fa;
        padding: 8px;
        border-radius: 6px;
        margin: 4px 0;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Run the page
if __name__ == "__main__":
    page_world_discovery() 