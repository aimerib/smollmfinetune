"""
Conversational Character Builder Component

This component provides a guided conversational interface for character creation,
leveraging the CharacterIntelligenceService for an engaging and intuitive experience.
"""

import streamlit as st
import asyncio
from typing import Optional, Dict, Any
import logging

try:
    from utils.character.character_intelligence import CharacterIntelligenceService, ConversationSuggestion
    from utils.character.models import CharacterCore
    from utils.world import WorldManager
except ImportError:
    from app.utils.character.character_intelligence import CharacterIntelligenceService, ConversationSuggestion
    from app.utils.character.models import CharacterCore
    from app.utils.world import WorldManager

logger = logging.getLogger(__name__)


def render_conversational_builder(world_manager: WorldManager, character_name: str = "", existing_character: Optional[CharacterCore] = None) -> Optional[CharacterCore]:
    """
    Render the conversational character builder interface.
    
    Returns the completed CharacterCore if character creation is finished,
    None if still in progress.
    """
    st.markdown('<h3 class="gradient-text">🗨️ Conversational Character Builder</h3>', unsafe_allow_html=True)
    
    # Initialize intelligence service
    if 'char_intelligence' not in st.session_state:
        st.session_state.char_intelligence = CharacterIntelligenceService(world_manager)
    
    intelligence = st.session_state.char_intelligence
    
    # Get current world context
    current_world = world_manager.list_worlds()[0] if world_manager.list_worlds() else None
    
    # Initialize conversation state
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
        # Use existing character if provided (enhancement mode) or create new one
        if existing_character:
            st.session_state.current_character = existing_character
            st.session_state.enhancement_mode = True
            # Store original character for personality-aware prompting
            st.session_state.personality_context = get_personality_context(existing_character)
        else:
            st.session_state.current_character = CharacterCore(name=character_name, description="")
            st.session_state.enhancement_mode = False
            st.session_state.personality_context = None
        st.session_state.conversation_complete = False
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_conversation_interface(intelligence, current_world)
    
    with col2:
        render_character_preview_panel(intelligence)
    
    # Return character if creation is complete
    if st.session_state.get('conversation_complete', False):
        return st.session_state.current_character
    
    return None


def render_conversation_interface(intelligence: CharacterIntelligenceService, world_context: Optional[str]):
    """Render the main conversation interface."""
    
    # Conversation container
    conversation_container = st.container()
    
    with conversation_container:
        # Start conversation button or continue
        if not st.session_state.conversation_started:
            # Different intro for enhancement vs creation
            if st.session_state.get('enhancement_mode', False):
                char_name = st.session_state.current_character.name
                personality_insights = st.session_state.get('personality_context', '')
                
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%); 
                                padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 2rem;">
                        <h3 style="color: white; margin: 0;">🎭 Let's Enhance {char_name}</h3>
                        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;">
                            Based on their personality analysis, I'll guide you through deepening their unique voice and inner world
                        </p>
                        {f'<div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem;"><small>🧠 Personality Focus: {personality_insights}</small></div>' if personality_insights else ''}
                    </div>
                """, unsafe_allow_html=True)
                button_text = "🚀 Start Personality-Aware Enhancement"
            else:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); 
                                padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 2rem;">
                        <h3 style="color: white; margin: 0;">✨ Let's Create Your Character Together</h3>
                        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0;">
                            I'll guide you through a conversation to discover your character's soul
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                button_text = "🚀 Start Character Creation"
            
            if st.button(button_text, type="primary", use_container_width=True):
                # Start the conversation
                suggestion = asyncio.run(intelligence.start_conversational_creation(
                    initial_name=st.session_state.current_character.name,
                    world_context=world_context
                ))
                st.session_state.current_suggestion = suggestion
                st.session_state.conversation_started = True
                st.rerun()
        
        else:
            # Display conversation history
            render_conversation_history(intelligence)
            
            # Current AI question
            if hasattr(st.session_state, 'current_suggestion'):
                render_current_question(intelligence)
            
            # User input area
            render_user_input_area(intelligence)


def render_conversation_history(intelligence: CharacterIntelligenceService):
    """Render the conversation history in a chat-like format."""
    
    if not intelligence.conversation_history:
        return
    
    st.markdown("### 💬 Our Conversation")
    
    # Create a scrollable conversation area
    with st.container():
        for message in intelligence.conversation_history:
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            focus_area = message.get('focus_area', '')
            
            if role == 'assistant':
                # AI message
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)
                    if focus_area and focus_area != 'introduction':
                        st.caption(f"🎯 Exploring: {focus_area.title()}")
            
            elif role == 'user':
                # User message
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)


def render_current_question(intelligence: CharacterIntelligenceService):
    """Render the current AI question with context."""
    
    suggestion = st.session_state.current_suggestion
    
    # Question container with styling
    st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.1); 
                    border-left: 4px solid #6366f1; 
                    padding: 1.5rem; 
                    border-radius: 8px; 
                    margin: 1rem 0;">
    """, unsafe_allow_html=True)
    
    # Priority indicator
    priority_emoji = "🔥" if suggestion.priority >= 4 else "⭐" if suggestion.priority >= 3 else "💡"
    
    st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 1.2rem; margin-right: 0.5rem;">{priority_emoji}</span>
            <strong style="color: #6366f1;">AI Question ({suggestion.focus_area.title()})</strong>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**{suggestion.question}**")
    
    # Show reasoning in expandable section
    with st.expander("💭 Why I'm asking this"):
        st.write(suggestion.reasoning)
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_user_input_area(intelligence: CharacterIntelligenceService):
    """Render the user input area for responses."""
    
    st.markdown("### 💬 Your Response")
    
    # Text area for user response
    user_response = st.text_area(
        "Tell me about your character:",
        height=100,
        placeholder="Share whatever comes to mind about your character...",
        key="user_response_input"
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📝 Submit Response", type="primary", disabled=not user_response.strip()):
            if user_response.strip():
                process_user_response(intelligence, user_response)
    
    with col2:
        if st.button("🎲 Need Inspiration"):
            show_inspiration_prompts(intelligence)
    
    with col3:
        if st.button("✅ Finish Character"):
            if confirm_character_completion(intelligence):
                st.session_state.conversation_complete = True
                st.rerun()


def process_user_response(intelligence: CharacterIntelligenceService, user_response: str):
    """Process user's response and update character."""
    
    try:
        with st.spinner("🧠 Understanding your character..."):
            # Process the response
            updated_character, next_suggestion, synthesis = asyncio.run(
                intelligence.process_conversation_response(
                    user_response, 
                    st.session_state.current_character
                )
            )
            
            # Update session state
            st.session_state.current_character = updated_character
            st.session_state.current_suggestion = next_suggestion
            st.session_state.current_synthesis = synthesis
            
            # Clear input
            st.session_state.user_response_input = ""
            
            st.rerun()
            
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        logger.error(f"Error in process_user_response: {e}")


def show_inspiration_prompts(intelligence: CharacterIntelligenceService):
    """Show inspiration prompts to help users respond."""
    
    current_focus = st.session_state.current_suggestion.focus_area
    
    inspiration_prompts = {
        "personality": [
            "Think about someone you know with an interesting personality - what makes them unique?",
            "What's a personality trait that often gets misunderstood?",
            "How does your character react when they're stressed or excited?"
        ],
        "backstory": [
            "What's a moment that changed everything for your character?",
            "Who was the most important person in their childhood?",
            "What's something they've never told anyone?"
        ],
        "goals": [
            "What would make your character feel like their life was meaningful?",
            "What are they trying to prove to themselves or others?",
            "What do they dream about at night?"
        ],
        "relationships": [
            "Who does your character trust completely?",
            "Who brings out the worst in them?",
            "What kind of person would they fall in love with?"
        ]
    }
    
    prompts = inspiration_prompts.get(current_focus, inspiration_prompts["personality"])
    
    st.markdown("**💡 Need some inspiration? Try answering one of these:**")
    for prompt in prompts:
        if st.button(f"• {prompt}", key=f"inspiration_{hash(prompt)}"):
            st.session_state.user_response_input = prompt
            st.rerun()


def confirm_character_completion(intelligence: CharacterIntelligenceService) -> bool:
    """Check if character is ready and confirm completion."""
    
    # Validate character
    validation = asyncio.run(intelligence.validate_character_for_training(st.session_state.current_character))
    
    if validation.is_ready_for_training:
        st.success("✅ Your character looks great and is ready for training!")
        return True
    else:
        st.warning("⚠️ Your character could use a bit more development:")
        for issue in validation.issues[:3]:  # Show top 3 issues
            st.write(f"• {issue}")
        
        st.info("You can finish now or continue to improve your character.")
        
        # Force completion option
        if st.button("🚀 Finish Anyway", key="force_complete"):
            return True
    
    return False


def render_character_preview_panel(intelligence: CharacterIntelligenceService):
    """Render the live character preview panel."""
    
    st.markdown("### 🎭 Character Preview")
    
    character = st.session_state.current_character
    
    if not character.name:
        st.info("Your character will appear here as we build them together...")
        return
    
    # Character header
    st.markdown(f"""
        <div style="background: rgba(168, 85, 247, 0.1); 
                    padding: 1rem; 
                    border-radius: 8px; 
                    text-align: center; 
                    margin-bottom: 1rem;">
            <h4 style="color: #a855f7; margin: 0;">{character.name}</h4>
            <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0;">Evolving Character</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Show synthesis if available
    if hasattr(st.session_state, 'current_synthesis'):
        render_character_synthesis_preview(st.session_state.current_synthesis)
    else:
        # Basic character info
        if character.description:
            st.markdown("**Description:**")
            st.write(character.description[:150] + "..." if len(character.description) > 150 else character.description)
        
        if character.goals:
            st.markdown("**Goals:**")
            for goal in character.goals[:3]:
                st.write(f"• {goal}")
        
        # Progress indicator
        progress = calculate_character_progress(character)
        st.progress(progress, text=f"Character Development: {int(progress * 100)}%")


def render_character_synthesis_preview(synthesis):
    """Render detailed character synthesis preview."""
    
    # Personality summary
    if synthesis.personality_summary:
        st.markdown("**🧠 Personality:**")
        st.write(synthesis.personality_summary)
    
    # Character archetype
    if synthesis.character_archetype:
        st.markdown("**🎭 Archetype:**")
        st.info(f"{synthesis.character_archetype}")
    
    # Sample dialogue
    if synthesis.sample_dialogue:
        with st.expander("💬 Character Voice Preview"):
            for dialogue in synthesis.sample_dialogue[:1]:  # Show just one
                st.markdown(f'```\n{dialogue}\n```')
    
    # Training readiness
    readiness = synthesis.training_readiness
    readiness_color = "#22c55e" if readiness >= 0.7 else "#f59e0b" if readiness >= 0.5 else "#ef4444"
    
    st.markdown(f"""
        <div style="background: rgba(34, 197, 94, 0.1); 
                    padding: 0.75rem; 
                    border-radius: 6px; 
                    border-left: 4px solid {readiness_color};">
            <strong>🎯 Training Readiness: {int(readiness * 100)}%</strong>
        </div>
    """, unsafe_allow_html=True)
    
    # Development suggestions
    if synthesis.development_suggestions:
        with st.expander("💡 Development Suggestions"):
            for suggestion in synthesis.development_suggestions[:3]:
                st.write(f"• {suggestion}")


def calculate_character_progress(character: CharacterCore) -> float:
    """Calculate character completion progress."""
    
    progress_factors = []
    
    # Name
    progress_factors.append(1.0 if character.name else 0.0)
    
    # Description
    progress_factors.append(1.0 if character.description and len(character.description) > 50 else 0.5 if character.description else 0.0)
    
    # Goals
    progress_factors.append(min(1.0, len(character.goals) / 2))
    
    # Backstory
    progress_factors.append(1.0 if character.backstory else 0.0)
    
    # Relationships
    progress_factors.append(min(1.0, len(character.relationships) / 2))
    
    return sum(progress_factors) / len(progress_factors)


def reset_conversation():
    """Reset the conversation state."""
    keys_to_clear = [
        'conversation_started',
        'current_character', 
        'conversation_complete',
        'current_suggestion',
        'current_synthesis',
        'char_intelligence',
        'enhancement_mode',
        'personality_context'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def get_personality_context(character: CharacterCore) -> str:
    """Generate personality context string for enhanced conversation guidance."""
    if not character.personality_traits:
        return ""
    
    traits = character.personality_traits
    high_traits = []
    low_traits = []
    
    trait_map = {
        'openness': ('Creative/Open-minded', 'Traditional/Practical'),
        'conscientiousness': ('Organized/Disciplined', 'Spontaneous/Flexible'),
        'extraversion': ('Outgoing/Social', 'Reserved/Introspective'),
        'agreeableness': ('Cooperative/Trusting', 'Competitive/Skeptical'),
        'neuroticism': ('Emotionally Sensitive', 'Emotionally Stable')
    }
    
    for trait_key, (high_desc, low_desc) in trait_map.items():
        value = getattr(traits, trait_key)
        if value >= 0.7:
            high_traits.append(high_desc)
        elif value <= 0.3:
            low_traits.append(low_desc)
    
    context_parts = []
    if high_traits:
        context_parts.append(f"High: {', '.join(high_traits)}")
    if low_traits:
        context_parts.append(f"Low: {', '.join(low_traits)}")
    
    return "; ".join(context_parts) if context_parts else "Balanced personality" 