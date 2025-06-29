"""
🎮 Platform Runtime Interface - Where Adventures Come Alive!

This is the immersive gameplay interface where players interact with their 
selected characters in the chosen world. The heart of the platform experience!
"""

import streamlit as st
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from utils.runtime.platform_engine import PlatformRuntimeEngine

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Adventure Runtime",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for immersive experience
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    
    .character-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .world-context {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .chat-message {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 10px;
    }
    
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .character-message {
        background: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    
    .mood-indicator {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    
    .mood-happy { background: #4caf50; color: white; }
    .mood-sad { background: #f44336; color: white; }
    .mood-neutral { background: #9e9e9e; color: white; }
    .mood-excited { background: #ff9800; color: white; }
    .mood-contemplative { background: #3f51b5; color: white; }
    .mood-cheerful { background: #ffeb3b; color: black; }
</style>
""", unsafe_allow_html=True)


def get_mood_emoji(mood: str) -> str:
    """Get emoji representation for character mood"""
    mood_emojis = {
        'happy': '😊',
        'sad': '😢', 
        'neutral': '😐',
        'excited': '🤩',
        'contemplative': '🤔',
        'cheerful': '😄',
        'angry': '😠',
        'surprised': '😲',
        'confused': '🤷',
        'confident': '😎'
    }
    return mood_emojis.get(mood.lower(), '🎭')


def render_character_panel(runtime_engine: PlatformRuntimeEngine):
    """Render the character panel with live character states"""
    st.markdown("### 🎭 Your Companions")
    
    active_characters = runtime_engine.get_active_character_ids()
    
    for char_id in active_characters:
        character_state = runtime_engine.get_character_state(char_id)
        
        # Character card
        mood_emoji = get_mood_emoji(character_state.current_mood)
        mood_class = f"mood-{character_state.current_mood.lower()}"
        
        st.markdown(f"""
        <div class="character-card">
            <h4>{mood_emoji} {character_state.character_name}</h4>
            <span class="mood-indicator {mood_class}">
                {character_state.current_mood.title()}
            </span>
            <br><br>
            <small>💬 Ready to chat</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Character controls
        with st.expander(f"⚙️ {character_state.character_name} Controls", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Active toggle
                is_active = st.toggle(
                    "Active", 
                    value=character_state.is_active,
                    key=f"active_{char_id}",
                    help="Character will respond to messages"
                )
                
                if is_active != character_state.is_active:
                    runtime_engine.set_character_active(char_id, is_active)
            
            with col2:
                # Mood selector
                mood_options = ['neutral', 'happy', 'sad', 'excited', 'contemplative', 'cheerful']
                current_mood = st.selectbox(
                    "Mood",
                    options=mood_options,
                    index=mood_options.index(character_state.current_mood) if character_state.current_mood in mood_options else 0,
                    key=f"mood_{char_id}",
                    help="Adjust character's emotional state"
                )
                
                if current_mood != character_state.current_mood:
                    runtime_engine.update_character_mood(char_id, current_mood)


def render_world_context(runtime_engine: PlatformRuntimeEngine):
    """Render the world context strip"""
    session_data = runtime_engine.session_data
    
    # World context header
    st.markdown(f"""
    <div class="world-context">
        <h3>🌍 {session_data.get('session_name', 'Adventure')}</h3>
        <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
            <div><strong>📍 Location:</strong> {session_data.get('current_location', 'Unknown')}</div>
            <div><strong>🕐 Time:</strong> {session_data.get('world_time', 'Unknown')}</div>
            <div><strong>🌤️ Weather:</strong> {session_data.get('weather', 'Unknown')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Recent world events
    recent_events = session_data.get('recent_world_events', [])
    if recent_events:
        st.markdown("**📰 Recent Events:**")
        for event in recent_events[-3:]:  # Show last 3 events
            st.markdown(f"• {event}")


def render_conversation_history(runtime_engine: PlatformRuntimeEngine):
    """Render the conversation history with character attribution"""
    st.markdown("### 💬 Conversation")
    
    conversation = runtime_engine.get_conversation_history()
    
    if not conversation:
        st.info("🌟 Start your adventure by sending a message to your companions!")
        return
    
    # Display messages
    for message in conversation[-20:]:  # Show last 20 messages
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        if role == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong> {content}
            </div>
            """, unsafe_allow_html=True)
        
        elif role == 'assistant':
            character_name = message.get('character_name', 'Character')
            character_id = message.get('character_id')
            
            # Get character mood for styling
            mood_emoji = '🎭'
            if character_id and character_id in runtime_engine.character_states:
                char_state = runtime_engine.character_states[character_id]
                mood_emoji = get_mood_emoji(char_state.current_mood)
            
            st.markdown(f"""
            <div class="chat-message character-message">
                <strong>{mood_emoji} {character_name}:</strong> {content}
            </div>
            """, unsafe_allow_html=True)


async def handle_user_message(runtime_engine: PlatformRuntimeEngine, user_message: str):
    """Handle user message and generate character responses"""
    try:
        # Add user message to history
        runtime_engine.add_message_to_history({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now()
        })
        
        # Determine which characters should respond
        active_characters = runtime_engine.get_active_character_ids()
        responding_characters = runtime_engine.determine_responding_characters(
            message=user_message,
            message_type="user"
        )
        
        # Filter to only active characters
        responding_characters = [char_id for char_id in responding_characters 
                               if char_id in active_characters]
        
        if not responding_characters:
            st.warning("No characters are available to respond. Try activating some characters!")
            return
        
        # Generate responses
        with st.spinner("🤔 Your companions are thinking..."):
            responses = await runtime_engine.generate_multi_character_responses(
                user_message=user_message,
                responding_characters=responding_characters
            )
        
        # Add character responses to history
        for char_id, response in responses.items():
            if char_id in runtime_engine.character_states:
                char_state = runtime_engine.character_states[char_id]
                runtime_engine.add_message_to_history({
                    'role': 'assistant',
                    'content': response,
                    'character_id': char_id,
                    'character_name': char_state.character_name,
                    'timestamp': datetime.now()
                })
        
        # Auto-save session
        runtime_engine.save_session_state()
        
        # Rerun to show new messages
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error handling user message: {e}")
        st.error(f"❌ Something went wrong: {str(e)}")


def render_chat_input(runtime_engine: PlatformRuntimeEngine):
    """Render the chat input interface"""
    
    # Chat input
    user_message = st.chat_input(
        "Send a message to your companions...",
        key="main_chat_input"
    )
    
    if user_message:
        # Run async message handling
        asyncio.run(handle_user_message(runtime_engine, user_message))
    
    # Quick action buttons
    st.markdown("#### 🎮 Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("👋 Greet Everyone", use_container_width=True):
            asyncio.run(handle_user_message(runtime_engine, "Hello everyone! How are you all doing?"))
    
    with col2:
        if st.button("❓ Ask for Advice", use_container_width=True):
            asyncio.run(handle_user_message(runtime_engine, "I could use some advice. What do you think I should do?"))
    
    with col3:
        if st.button("🌟 Share Excitement", use_container_width=True):
            asyncio.run(handle_user_message(runtime_engine, "I'm so excited about this adventure! What should we explore first?"))
    
    with col4:
        if st.button("🤔 Check Status", use_container_width=True):
            asyncio.run(handle_user_message(runtime_engine, "How is everyone feeling? What's our current situation?"))


def render_session_controls(runtime_engine: PlatformRuntimeEngine):
    """Render session management controls"""
    with st.expander("⚙️ Session Controls", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Session", use_container_width=True):
                runtime_engine.save_session_state()
                st.success("Session saved!")
        
        with col2:
            if st.button("📊 Session Stats", use_container_width=True):
                stats = runtime_engine.db_service.get_session_statistics(runtime_engine.session_id)
                if stats:
                    st.json(stats)
        
        with col3:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()


def main():
    """Main platform runtime interface"""
    
    # Get session ID from query params
    session_id = st.query_params.get("session_id")
    
    if not session_id:
        st.error("❌ No session ID provided. Please select a session from the Character Selection page.")
        st.markdown("[← Back to Character Selection](character_selection)")
        return
    
    try:
        # Initialize runtime engine
        if 'runtime_engine' not in st.session_state or st.session_state.get('current_session_id') != session_id:
            with st.spinner("🎮 Loading your adventure..."):
                runtime_engine = PlatformRuntimeEngine(session_id=session_id)
                st.session_state.runtime_engine = runtime_engine
                st.session_state.current_session_id = session_id
        else:
            runtime_engine = st.session_state.runtime_engine
        
        # Main layout
        st.markdown(f"# 🎮 {runtime_engine.session_data.get('session_name', 'Adventure Runtime')}")
        
        # Three-column layout
        left_col, main_col, right_col = st.columns([1, 2, 1])
        
        with left_col:
            # Character panel
            render_character_panel(runtime_engine)
            
            # Session controls
            render_session_controls(runtime_engine)
        
        with main_col:
            # World context
            render_world_context(runtime_engine)
            
            # Conversation history
            render_conversation_history(runtime_engine)
            
            # Chat input
            render_chat_input(runtime_engine)
        
        with right_col:
            # Additional controls or info can go here
            st.markdown("### 🌟 Adventure Tips")
            st.info("""
            **💡 How to Play:**
            - Chat naturally with your companions
            - Use Quick Actions for common interactions
            - Adjust character moods to influence responses
            - Toggle characters active/inactive to control who responds
            """)
            
            # World events (future feature)
            st.markdown("### 🌍 World Events")
            st.info("World events coming soon! Your choices will shape the story.")
    
    except ValueError as e:
        if "not found" in str(e):
            st.error(f"❌ Session not found: {session_id}")
            st.markdown("[← Back to Character Selection](character_selection)")
        else:
            st.error(f"❌ Error loading session: {str(e)}")
    
    except Exception as e:
        logger.error(f"Platform runtime error: {e}")
        st.error(f"❌ Something went wrong: {str(e)}")
        st.markdown("[← Back to Character Selection](character_selection)")


if __name__ == "__main__":
    main() 