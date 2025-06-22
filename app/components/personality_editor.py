"""
Personality Editor Component - Simplified implementation for R1-4
Provides Big Five sliders and basic radar chart visualization
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Optional
from utils.character.models import CharacterCore, Personality


def log_preference_event(event_type: str, old_value, new_value, context: Optional[str] = None):
    """
    Basic preference logging function
    TODO: Implement proper preference logging system from R1-3
    """
    # For now, just log to console - will be enhanced later
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Preference Event: {event_type} | Old: {old_value} | New: {new_value} | Context: {context}")


def create_personality_radar(personality: Personality, key_prefix: str = "radar_") -> go.Figure:
    """Create a radar chart for Big Five personality traits"""
    
    labels = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    values = [
        personality.openness,
        personality.conscientiousness, 
        personality.extraversion,
        personality.agreeableness,
        personality.neuroticism
    ]
    
    # Close the radar chart by adding first value at the end
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill='toself',
        name='Personality',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='rgba(99, 102, 241, 1)', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title="Big Five Personality Traits",
        height=400
    )
    
    return fig


def render_personality_editor(core: CharacterCore, key_prefix: str = "pers_", show_ai_btn: bool = True) -> Personality:
    """
    Render personality editor with sliders and radar chart
    
    Args:
        core: CharacterCore object to edit
        key_prefix: Key prefix for Streamlit widgets
        show_ai_btn: Whether to show AI estimation button
        
    Returns:
        Updated Personality object
    """
    
    st.markdown("### 🧠 Personality Traits (Big Five)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Adjust Personality Traits:**")
        
        # Store original values for comparison
        original_traits = core.personality_traits
        
        # Create sliders for each trait
        openness = st.slider(
            "🌟 Openness to Experience",
            min_value=0.0,
            max_value=1.0,
            value=original_traits.openness,
            step=0.05,
            key=f"{key_prefix}openness",
            help="Curiosity, creativity, willingness to try new things"
        )
        
        conscientiousness = st.slider(
            "📋 Conscientiousness", 
            min_value=0.0,
            max_value=1.0,
            value=original_traits.conscientiousness,
            step=0.05,
            key=f"{key_prefix}conscientiousness",
            help="Organization, discipline, reliability"
        )
        
        extraversion = st.slider(
            "🎉 Extraversion",
            min_value=0.0,
            max_value=1.0, 
            value=original_traits.extraversion,
            step=0.05,
            key=f"{key_prefix}extraversion",
            help="Sociability, assertiveness, energy level"
        )
        
        agreeableness = st.slider(
            "🤝 Agreeableness",
            min_value=0.0,
            max_value=1.0,
            value=original_traits.agreeableness,
            step=0.05,
            key=f"{key_prefix}agreeableness",
            help="Cooperation, trust, empathy"
        )
        
        neuroticism = st.slider(
            "😰 Neuroticism",
            min_value=0.0,
            max_value=1.0,
            value=original_traits.neuroticism,
            step=0.05,
            key=f"{key_prefix}neuroticism",
            help="Emotional instability, anxiety, moodiness"
        )
        
        # AI estimation button
        if show_ai_btn:
            if st.button("✨ Estimate from Examples", key=f"{key_prefix}ai_estimate"):
                # Placeholder for AI estimation
                st.info("🔄 AI estimation would analyze character description and examples to suggest Big Five traits")
                st.warning("Note: AI estimation requires implementation of `llm_estimate_big5` function")
    
    with col2:
        # Create updated personality object
        updated_personality = Personality(
            openness=openness,
            conscientiousness=conscientiousness,
            extraversion=extraversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism
        )
        
        # Create and display radar chart
        fig = create_personality_radar(updated_personality, key_prefix)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}chart")
        
        # Show trait descriptions
        with st.expander("📖 Trait Descriptions"):
            st.markdown("""
            **Openness**: Imagination, curiosity, artistic interests
            **Conscientiousness**: Self-discipline, orderliness, achievement-striving  
            **Extraversion**: Sociability, assertiveness, positive emotions
            **Agreeableness**: Trust, altruism, cooperation
            **Neuroticism**: Anxiety, angry hostility, emotional instability
            """)
    
    # Update the core object
    core.personality_traits = updated_personality
    
    # Log changes if traits changed
    if updated_personality != original_traits:
        log_preference_event(
            "personality_edit",
            original_traits.model_dump(),
            updated_personality.model_dump(),
            f"Character: {core.name}"
        )
    
    return updated_personality 