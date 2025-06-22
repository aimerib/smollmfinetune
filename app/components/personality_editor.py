"""
🧠 Personality Editor Component - Complete R1-5 Implementation

Interactive Big Five personality editor with:
- Five sliders (0.05 step precision)
- Live-updating Plotly radar chart
- AI estimation from character examples with diff preview
- Preference logging when AI estimates are accepted
- Detailed tooltips explaining each trait
"""

import streamlit as st
import plotly.graph_objects as go
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from app.utils.character.models import CharacterCore, Personality, llm_estimate_big5

logger = logging.getLogger(__name__)


def log_preference_event(event_type: str, old_value: Any, new_value: Any, context: Optional[str] = None):
    """
    Enhanced preference logging function for R1-3 integration
    """
    # Initialize preference tracking in session state if not exists
    if 'preference_events' not in st.session_state:
        st.session_state.preference_events = []
    
    # Create preference event
    event = {
        'type': event_type,
        'old': old_value,
        'new': new_value,
        'context': context,
        'timestamp': datetime.now().isoformat()
    }
    
    # Store event
    st.session_state.preference_events.append(event)
    
    # Also log for debugging
    logger.info(f"Preference Event: {event_type} | Context: {context}")
    
    # Track in intelligence service if available
    if 'character_intelligence' in st.session_state:
        try:
            intelligence_service = st.session_state.character_intelligence
            if hasattr(intelligence_service, 'track_user_preference'):
                intelligence_service.track_user_preference(
                    context=event_type,
                    options=["old_value", "new_value"],
                    chosen="new_value",
                    character_context=context
                )
        except Exception as e:
            logger.debug(f"Failed to track preference in intelligence service: {e}")


def create_personality_radar(personality: Personality, key_prefix: str = "radar_") -> go.Figure:
    """Create an enhanced radar chart for Big Five personality traits"""
    
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
    
    # Add main personality trace
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill='toself',
        name='Current Personality',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='rgba(99, 102, 241, 1)', width=3),
        marker=dict(size=8, color='rgba(99, 102, 241, 1)')
    ))
    
    # Add comparison trace if available in session state
    if f'{key_prefix}comparison_personality' in st.session_state:
        comparison = st.session_state[f'{key_prefix}comparison_personality']
        comp_values = [
            comparison.openness,
            comparison.conscientiousness,
            comparison.extraversion,
            comparison.agreeableness,
            comparison.neuroticism
        ] + [comparison.openness]
        
        fig.add_trace(go.Scatterpolar(
            r=comp_values,
            theta=labels_closed,
            fill='toself',
            name='AI Suggested',
            fillcolor='rgba(34, 197, 94, 0.1)',
            line=dict(color='rgba(34, 197, 94, 1)', width=2, dash='dash'),
            marker=dict(size=6, color='rgba(34, 197, 94, 1)')
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2,
                showticklabels=True,
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            )
        ),
        showlegend=True,
        title=dict(
            text="Big Five Personality Profile",
            x=0.5,
            font=dict(size=16)
        ),
        height=450,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    
    return fig


def render_trait_tooltip(trait_name: str, trait_value: float) -> str:
    """Generate detailed tooltip text for personality traits"""
    
    trait_descriptions = {
        "Openness": {
            "high": "Creative, curious, imaginative, artistic, open to new experiences and ideas",
            "low": "Conventional, practical, traditional, prefers routine and familiar approaches",
            "affects": "How the character approaches new situations, creativity, and intellectual curiosity"
        },
        "Conscientiousness": {
            "high": "Organized, disciplined, reliable, goal-oriented, plans ahead",
            "low": "Spontaneous, flexible, casual, may procrastinate or be disorganized",
            "affects": "Work ethic, reliability, attention to detail, and goal achievement"
        },
        "Extraversion": {
            "high": "Outgoing, sociable, energetic, assertive, enjoys social interaction",
            "low": "Reserved, introspective, quiet, prefers solitude or small groups",
            "affects": "Social behavior, energy levels, and comfort in group settings"
        },
        "Agreeableness": {
            "high": "Cooperative, trusting, empathetic, helpful, avoids conflict",
            "low": "Competitive, skeptical, direct, values honesty over harmony",
            "affects": "Interpersonal relationships, conflict resolution, and cooperation"
        },
        "Neuroticism": {
            "high": "Emotionally sensitive, anxious, prone to stress and mood swings",
            "low": "Emotionally stable, calm, resilient, handles stress well",
            "affects": "Emotional reactions, stress management, and mood stability"
        }
    }
    
    desc = trait_descriptions.get(trait_name, {})
    level = "high" if trait_value >= 0.6 else "low" if trait_value <= 0.4 else "moderate"
    
    if level == "moderate":
        tooltip = f"**{trait_name}** (Moderate): Balanced between high and low traits. "
    else:
        tooltip = f"**{trait_name}** ({level.title()}): {desc.get(level, '')}. "
    
    tooltip += f"\n\n**Character Impact**: {desc.get('affects', 'Influences character behavior')}"
    
    return tooltip


async def estimate_personality_from_ai(core: CharacterCore) -> Optional[Personality]:
    """Use AI to estimate personality traits from character information"""
    try:
        # Combine character information for analysis
        description = core.description or ""
        personality_text = ""
        
        # Add existing personality info if available
        if hasattr(core, 'imports') and core.imports.get('original_personality'):
            personality_text = core.imports['original_personality']
        
        # Add example dialogue if available
        mes_example = ""
        if hasattr(core, 'imports') and core.imports.get('original_mes_example'):
            mes_example = core.imports['original_mes_example']
        
        # Use the existing LLM estimation function
        estimated_personality = await llm_estimate_big5(
            description + " " + personality_text, 
            mes_example
        )
        
        return estimated_personality
        
    except Exception as e:
        logger.error(f"AI personality estimation failed: {e}")
        st.error(f"AI estimation failed: {str(e)}")
        return None


def show_personality_comparison(current: Personality, suggested: Personality) -> Dict[str, float]:
    """Show a comparison between current and suggested personality traits"""
    
    st.markdown("### 🔍 AI Suggestion Comparison")
    
    traits = [
        ("Openness", current.openness, suggested.openness),
        ("Conscientiousness", current.conscientiousness, suggested.conscientiousness),
        ("Extraversion", current.extraversion, suggested.extraversion),
        ("Agreeableness", current.agreeableness, suggested.agreeableness),
        ("Neuroticism", current.neuroticism, suggested.neuroticism)
    ]
    
    changes = {}
    
    for trait_name, current_val, suggested_val in traits:
        diff = suggested_val - current_val
        changes[trait_name.lower()] = diff
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.write(f"**{trait_name}**")
        
        with col2:
            st.write(f"{current_val:.2f}")
        
        with col3:
            arrow = "→"
            st.write(f"{arrow} {suggested_val:.2f}")
        
        with col4:
            if abs(diff) < 0.05:
                st.write("🟡 Same")
            elif diff > 0:
                st.write(f"🔼 +{diff:.2f}")
            else:
                st.write(f"🔽 {diff:.2f}")
    
    return changes


def render_personality_editor(core: CharacterCore, key_prefix: str = "pers_", show_ai_btn: bool = True) -> Personality:
    """
    Render enhanced personality editor with sliders, radar chart, and AI estimation
    
    Args:
        core: CharacterCore object to edit (modified in-place)
        key_prefix: Key prefix for Streamlit widgets
        show_ai_btn: Whether to show AI estimation button
        
    Returns:
        Updated Personality object
    """
    
    st.markdown("### 🧠 Personality Traits (Big Five)")
    
    # Store original values for change detection
    original_traits = core.personality_traits
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Adjust Personality Traits:**")
        
        # Create sliders for each trait with enhanced tooltips
        openness = st.slider(
            "🌟 Openness to Experience",
            min_value=0.0,
            max_value=1.0,
            value=original_traits.openness,
            step=0.05,
            key=f"{key_prefix}openness",
            help=render_trait_tooltip("Openness", original_traits.openness)
        )
        
        conscientiousness = st.slider(
            "📋 Conscientiousness", 
            min_value=0.0,
            max_value=1.0,
            value=original_traits.conscientiousness,
            step=0.05,
            key=f"{key_prefix}conscientiousness",
            help=render_trait_tooltip("Conscientiousness", original_traits.conscientiousness)
        )
        
        extraversion = st.slider(
            "🎉 Extraversion",
            min_value=0.0,
            max_value=1.0, 
            value=original_traits.extraversion,
            step=0.05,
            key=f"{key_prefix}extraversion",
            help=render_trait_tooltip("Extraversion", original_traits.extraversion)
        )
        
        agreeableness = st.slider(
            "🤝 Agreeableness",
            min_value=0.0,
            max_value=1.0,
            value=original_traits.agreeableness,
            step=0.05,
            key=f"{key_prefix}agreeableness",
            help=render_trait_tooltip("Agreeableness", original_traits.agreeableness)
        )
        
        neuroticism = st.slider(
            "😰 Neuroticism",
            min_value=0.0,
            max_value=1.0,
            value=original_traits.neuroticism,
            step=0.05,
            key=f"{key_prefix}neuroticism",
            help=render_trait_tooltip("Neuroticism", original_traits.neuroticism)
        )
        
        # AI estimation button with enhanced functionality
        if show_ai_btn and core.name:
            ai_col1, ai_col2 = st.columns([2, 1])
            
            with ai_col1:
                if st.button("✨ Estimate from Examples", key=f"{key_prefix}ai_estimate", 
                           help="Use AI to analyze character description and suggest personality traits"):
                    with st.spinner("🤖 Analyzing character with AI..."):
                        estimated_personality = asyncio.run(estimate_personality_from_ai(core))
                        
                        if estimated_personality:
                            # Store for comparison
                            st.session_state[f'{key_prefix}ai_suggestion'] = estimated_personality
                            st.session_state[f'{key_prefix}comparison_personality'] = estimated_personality
                            st.session_state[f'{key_prefix}show_ai_comparison'] = True
                            st.rerun()
            
            with ai_col2:
                if st.session_state.get(f'{key_prefix}show_ai_comparison', False):
                    if st.button("❌", key=f"{key_prefix}close_ai", help="Close AI suggestions"):
                        st.session_state[f'{key_prefix}show_ai_comparison'] = False
                        st.session_state.pop(f'{key_prefix}comparison_personality', None)
                        st.rerun()
        
        # Show AI comparison if available
        if st.session_state.get(f'{key_prefix}show_ai_comparison', False):
            suggested = st.session_state.get(f'{key_prefix}ai_suggestion')
            if suggested:
                current_personality = Personality(
                    openness=openness,
                    conscientiousness=conscientiousness,
                    extraversion=extraversion,
                    agreeableness=agreeableness,
                    neuroticism=neuroticism
                )
                
                changes = show_personality_comparison(current_personality, suggested)
                
                # Accept/Reject buttons
                accept_col, reject_col = st.columns(2)
                
                with accept_col:
                    if st.button("✅ Accept AI Suggestion", key=f"{key_prefix}accept_ai", type="primary"):
                        # Log preference event
                        log_preference_event(
                            "big5_estimate_accept",
                            current_personality.model_dump(),
                            suggested.model_dump(),
                            f"Character: {core.name}"
                        )
                        
                        # Update sliders by setting session state values
                        st.session_state[f"{key_prefix}openness"] = suggested.openness
                        st.session_state[f"{key_prefix}conscientiousness"] = suggested.conscientiousness
                        st.session_state[f"{key_prefix}extraversion"] = suggested.extraversion
                        st.session_state[f"{key_prefix}agreeableness"] = suggested.agreeableness
                        st.session_state[f"{key_prefix}neuroticism"] = suggested.neuroticism
                        
                        # Clear comparison
                        st.session_state[f'{key_prefix}show_ai_comparison'] = False
                        st.session_state.pop(f'{key_prefix}comparison_personality', None)
                        
                        st.success("✅ AI suggestions applied!")
                        st.rerun()
                
                with reject_col:
                    if st.button("❌ Reject Suggestion", key=f"{key_prefix}reject_ai"):
                        # Log rejection
                        log_preference_event(
                            "big5_estimate_reject",
                            current_personality.model_dump(),
                            suggested.model_dump(),
                            f"Character: {core.name}"
                        )
                        
                        # Clear comparison
                        st.session_state[f'{key_prefix}show_ai_comparison'] = False
                        st.session_state.pop(f'{key_prefix}comparison_personality', None)
                        st.rerun()
        
        elif show_ai_btn and not core.name:
            st.info("💡 Add character name and description to enable AI personality estimation")
    
    with col2:
        # Create updated personality object
        updated_personality = Personality(
            openness=openness,
            conscientiousness=conscientiousness,
            extraversion=extraversion,
            agreeableness=agreeableness,
            neuroticism=neuroticism
        )
        
        # Create and display enhanced radar chart
        fig = create_personality_radar(updated_personality, key_prefix)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}chart")
        
        # Show personality summary
        st.markdown("**Personality Summary:**")
        trait_summaries = []
        
        if updated_personality.openness >= 0.7:
            trait_summaries.append("🌟 Highly creative")
        elif updated_personality.openness <= 0.3:
            trait_summaries.append("🏛️ Traditional")
        
        if updated_personality.conscientiousness >= 0.7:
            trait_summaries.append("📋 Very organized")
        elif updated_personality.conscientiousness <= 0.3:
            trait_summaries.append("🌪️ Spontaneous")
        
        if updated_personality.extraversion >= 0.7:
            trait_summaries.append("🎉 Highly social")
        elif updated_personality.extraversion <= 0.3:
            trait_summaries.append("🤫 Introverted")
        
        if updated_personality.agreeableness >= 0.7:
            trait_summaries.append("🤝 Very cooperative")
        elif updated_personality.agreeableness <= 0.3:
            trait_summaries.append("⚔️ Competitive")
        
        if updated_personality.neuroticism >= 0.7:
            trait_summaries.append("😰 Emotionally sensitive")
        elif updated_personality.neuroticism <= 0.3:
            trait_summaries.append("😌 Emotionally stable")
        
        if trait_summaries:
            for summary in trait_summaries:
                st.markdown(f"• {summary}")
        else:
            st.markdown("• 🎯 Balanced personality")
        
        # Show detailed trait descriptions in expandable section
        with st.expander("📖 Trait Descriptions & Impact"):
            st.markdown("""
            ### Big Five Personality Traits
            
            **🌟 Openness to Experience**
            - **High (0.7+)**: Creative, curious, imaginative, enjoys art and new ideas
            - **Low (0.3-)**: Conventional, practical, prefers routine and tradition
            - **Impact**: Affects creativity, learning style, and openness to change
            
            **📋 Conscientiousness**
            - **High (0.7+)**: Organized, disciplined, reliable, goal-oriented
            - **Low (0.3-)**: Spontaneous, flexible, may be disorganized
            - **Impact**: Work ethic, reliability, and achievement orientation
            
            **🎉 Extraversion**
            - **High (0.7+)**: Outgoing, energetic, assertive, enjoys social interaction
            - **Low (0.3-)**: Reserved, quiet, prefers solitude or small groups
            - **Impact**: Social behavior, leadership style, and energy sources
            
            **🤝 Agreeableness**
            - **High (0.7+)**: Cooperative, trusting, empathetic, avoids conflict
            - **Low (0.3-)**: Competitive, skeptical, direct, values honesty
            - **Impact**: Interpersonal relationships and conflict resolution
            
            **😰 Neuroticism**
            - **High (0.7+)**: Emotionally sensitive, anxious, prone to stress
            - **Low (0.3-)**: Emotionally stable, calm, resilient
            - **Impact**: Emotional reactions, stress management, and mood stability
            """)
    
    # Update the core object with new personality
    core.personality_traits = updated_personality
    
    # Log personality changes (but not on every slider move to avoid spam)
    if updated_personality != original_traits:
        # Only log significant changes (> 0.1 difference in any trait)
        significant_change = any([
            abs(updated_personality.openness - original_traits.openness) > 0.1,
            abs(updated_personality.conscientiousness - original_traits.conscientiousness) > 0.1,
            abs(updated_personality.extraversion - original_traits.extraversion) > 0.1,
            abs(updated_personality.agreeableness - original_traits.agreeableness) > 0.1,
            abs(updated_personality.neuroticism - original_traits.neuroticism) > 0.1,
        ])
        
        if significant_change:
            log_preference_event(
                "personality_manual_edit",
                original_traits.model_dump(),
                updated_personality.model_dump(),
                f"Character: {core.name}"
            )
    
    return updated_personality 