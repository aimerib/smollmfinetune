"""
Character Synthesis Preview Component

Real-time character preview that shows the character "coming alive" as users build them,
with dynamic personality visualization and auto-generated sample dialogue.
"""

import streamlit as st
import asyncio
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional
import logging

try:
    from utils.character.character_intelligence import CharacterIntelligenceService, CharacterSynthesis
    from utils.character.models import CharacterCore, Personality
except ImportError:
    from app.utils.character.character_intelligence import CharacterIntelligenceService, CharacterSynthesis
    from app.utils.character.models import CharacterCore, Personality

logger = logging.getLogger(__name__)


def render_character_synthesis_preview(character: CharacterCore, 
                                     intelligence_service: CharacterIntelligenceService,
                                     key_prefix: str = "synthesis") -> Optional[CharacterSynthesis]:
    """
    Render a comprehensive real-time character synthesis preview.
    
    Args:
        character: The character to synthesize
        intelligence_service: The intelligence service for analysis
        key_prefix: Unique key prefix for widgets
        
    Returns:
        CharacterSynthesis object with analysis results
    """
    
    if not character.name:
        render_empty_synthesis_state()
        return None
    
    # Generate synthesis
    try:
        with st.spinner("🧠 Synthesizing character..."):
            synthesis = asyncio.run(intelligence_service.synthesize_character(character))
    except Exception as e:
        st.error(f"Error synthesizing character: {str(e)}")
        logger.error(f"Synthesis error: {e}")
        return None
    
    # Render synthesis components
    render_character_header(character, synthesis)
    render_personality_visualization(character.personality_traits, key_prefix)
    render_character_voice_preview(synthesis)
    render_training_readiness_panel(synthesis)
    render_development_insights(synthesis)
    
    # NSFW assessment if relevant
    if synthesis.nsfw_assessment.get('has_nsfw_content', False):
        render_nsfw_assessment(synthesis.nsfw_assessment)
    
    return synthesis


def render_empty_synthesis_state():
    """Render placeholder when no character is available."""
    st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.05); 
                    border: 2px dashed rgba(99, 102, 241, 0.3); 
                    padding: 3rem; 
                    border-radius: 12px; 
                    text-align: center;">
            <h4 style="color: #6366f1; margin: 0;">🎭 Character Preview</h4>
            <p style="color: #94a3b8; margin: 0.5rem 0;">
                Your character will come alive here as you develop them
            </p>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0;">
                ✨ Real-time personality analysis<br>
                🗣️ Voice consistency scoring<br>
                🎯 Training readiness assessment
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_character_header(character: CharacterCore, synthesis: CharacterSynthesis):
    """Render the character header with basic info and archetype."""
    
    # Character name and archetype
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; 
                    border-radius: 12px; 
                    text-align: center; 
                    margin-bottom: 1.5rem;
                    color: white;">
            <h2 style="margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{character.name}</h2>
            <div style="background: rgba(255,255,255,0.2); 
                        border-radius: 20px; 
                        padding: 0.5rem 1rem; 
                        margin: 1rem auto 0; 
                        display: inline-block;">
                <strong>🎭 {synthesis.character_archetype}</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Personality summary
    if synthesis.personality_summary:
        st.markdown("### 🧠 Personality Profile")
        st.info(synthesis.personality_summary)


def render_personality_visualization(personality: Personality, key_prefix: str):
    """Render interactive personality trait visualization."""
    
    st.markdown("### 📊 Personality Traits (Big Five)")
    
    # Create radar chart
    traits = {
        'Openness': personality.openness,
        'Conscientiousness': personality.conscientiousness,
        'Extraversion': personality.extraversion,
        'Agreeableness': personality.agreeableness,
        'Neuroticism': personality.neuroticism
    }
    
    # Radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(traits.values()),
        theta=list(traits.keys()),
        fill='toself',
        name='Personality',
        line_color='rgba(99, 102, 241, 0.8)',
        fillcolor='rgba(99, 102, 241, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=True,
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['Low', '', 'Moderate', '', 'High']
            )),
        showlegend=False,
        title="",
        height=400,
        margin=dict(l=80, r=80, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_radar")
    
    # Trait details in columns
    cols = st.columns(5)
    trait_descriptions = {
        'Openness': ('🎨', 'Creativity & curiosity'),
        'Conscientiousness': ('⚡', 'Organization & discipline'),
        'Extraversion': ('👥', 'Social energy'),
        'Agreeableness': ('🤝', 'Cooperation & trust'),
        'Neuroticism': ('😰', 'Emotional sensitivity')
    }
    
    for i, (trait, value) in enumerate(traits.items()):
        with cols[i]:
            emoji, desc = trait_descriptions[trait]
            
            # Color coding
            if value >= 0.7:
                color = "#22c55e"
                level = "High"
            elif value >= 0.3:
                color = "#f59e0b"
                level = "Moderate"
            else:
                color = "#64748b"
                level = "Low"
            
            st.markdown(f"""
                <div style="text-align: center; padding: 0.5rem;">
                    <div style="font-size: 1.5rem;">{emoji}</div>
                    <div style="font-size: 0.8rem; font-weight: bold; color: {color};">
                        {level} ({value:.1f})
                    </div>
                    <div style="font-size: 0.7rem; color: #64748b;">
                        {desc}
                    </div>
                </div>
            """, unsafe_allow_html=True)


def render_character_voice_preview(synthesis: CharacterSynthesis):
    """Render character voice preview with sample dialogue."""
    
    st.markdown("### 🗣️ Character Voice Preview")
    
    if synthesis.sample_dialogue:
        # Voice consistency score
        consistency_color = "#22c55e" if synthesis.voice_consistency_score >= 0.7 else "#f59e0b" if synthesis.voice_consistency_score >= 0.5 else "#ef4444"
        
        st.markdown(f"""
            <div style="background: rgba(34, 197, 94, 0.1); 
                        padding: 1rem; 
                        border-radius: 8px; 
                        border-left: 4px solid {consistency_color}; 
                        margin-bottom: 1rem;">
                <strong>🎯 Voice Consistency: {int(synthesis.voice_consistency_score * 100)}%</strong>
                <br>
                <span style="color: #64748b; font-size: 0.9rem;">
                    How consistently the character speaks and responds
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        # Sample dialogues
        for i, dialogue in enumerate(synthesis.sample_dialogue[:2]):  # Show max 2
            with st.expander(f"💬 Sample Conversation {i + 1}", expanded=i == 0):
                st.markdown(f"```\n{dialogue}\n```")
    
    else:
        st.info("🔄 Add more character details to see voice preview")


def render_training_readiness_panel(synthesis: CharacterSynthesis):
    """Render training readiness assessment panel."""
    
    st.markdown("### 🎯 Training Readiness")
    
    readiness = synthesis.training_readiness
    
    # Color coding for readiness
    if readiness >= 0.8:
        color = "#22c55e"
        status = "Excellent"
        icon = "🟢"
    elif readiness >= 0.6:
        color = "#f59e0b" 
        status = "Good"
        icon = "🟡"
    elif readiness >= 0.4:
        color = "#f97316"
        status = "Needs Work"
        icon = "🟠"
    else:
        color = "#ef4444"
        status = "Not Ready"
        icon = "🔴"
    
    # Readiness display
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Progress bar
        st.progress(readiness, text=f"Training Readiness: {int(readiness * 100)}%")
        
        # Status
        st.markdown(f"""
            <div style="background: rgba(34, 197, 94, 0.1); 
                        padding: 1rem; 
                        border-radius: 8px; 
                        border-left: 4px solid {color};">
                {icon} <strong>Status: {status}</strong>
                <br>
                <span style="color: #64748b; font-size: 0.9rem;">
                    Character is {'ready' if readiness >= 0.6 else 'not ready'} for dataset generation
                </span>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Quick actions
        if readiness >= 0.6:
            if st.button("🚀 Generate Dataset", type="primary", use_container_width=True):
                st.info("Dataset generation would start here")
        
        if st.button("📋 Full Validation", use_container_width=True):
            show_detailed_validation(synthesis)


def render_development_insights(synthesis: CharacterSynthesis):
    """Render character development insights and suggestions."""
    
    st.markdown("### 💡 Development Insights")
    
    # World connections
    if synthesis.world_connections:
        st.markdown("**🌍 World Integration:**")
        for connection in synthesis.world_connections:
            st.success(f"✅ {connection}")
    
    # Development suggestions
    if synthesis.development_suggestions:
        st.markdown("**🔧 Improvement Suggestions:**")
        for suggestion in synthesis.development_suggestions:
            st.warning(f"💡 {suggestion}")
    
    # Character strengths
    strengths = identify_character_strengths(synthesis)
    if strengths:
        st.markdown("**⭐ Character Strengths:**")
        for strength in strengths:
            st.info(f"🎯 {strength}")


def render_nsfw_assessment(nsfw_assessment: Dict[str, Any]):
    """Render NSFW content assessment if applicable."""
    
    st.markdown("### 🔞 Content Assessment")
    
    with st.expander("NSFW Content Analysis", expanded=False):
        if nsfw_assessment.get('nsfw_style'):
            st.write(f"**Style:** {nsfw_assessment['nsfw_style']}")
        
        if nsfw_assessment.get('intimacy_style'):
            intimacy = nsfw_assessment['intimacy_style']
            st.write("**Intimacy Characteristics:**")
            for key, value in intimacy.items():
                st.write(f"- {key.title()}: {value}")
        
        if nsfw_assessment.get('requires_careful_dataset_generation'):
            st.warning("⚠️ This character requires careful dataset generation with appropriate content filtering.")


def show_detailed_validation(synthesis: CharacterSynthesis):
    """Show detailed character validation in modal."""
    
    st.markdown("### 📋 Detailed Character Validation")
    
    # This would show more detailed validation results
    # For now, showing placeholder
    st.info("Detailed validation panel would appear here with:")
    st.write("• Consistency checks across all character elements")
    st.write("• Potential contradictions or gaps")
    st.write("• Specific recommendations for improvement")
    st.write("• Estimated dataset quality and training time")


def identify_character_strengths(synthesis: CharacterSynthesis) -> list:
    """Identify character strengths based on synthesis."""
    
    strengths = []
    
    if synthesis.voice_consistency_score >= 0.8:
        strengths.append("Strong voice consistency")
    
    if synthesis.training_readiness >= 0.8:
        strengths.append("Well-developed character profile")
    
    if synthesis.world_connections:
        strengths.append("Good world integration")
    
    if synthesis.character_archetype != "The Everyperson":
        strengths.append(f"Clear archetype: {synthesis.character_archetype}")
    
    return strengths


def render_character_comparison(characters: list, key_prefix: str = "comparison"):
    """Render comparison between multiple characters (bonus feature)."""
    
    if len(characters) < 2:
        return
    
    st.markdown("### 🔄 Character Comparison")
    
    # This would show side-by-side personality comparisons
    # Implementation would depend on specific needs
    st.info("Character comparison feature - shows personality trait differences between characters")


def export_synthesis_report(synthesis: CharacterSynthesis, character: CharacterCore) -> str:
    """Export synthesis as a formatted report."""
    
    report = f"""
# Character Synthesis Report: {character.name}

## Character Archetype
{synthesis.character_archetype}

## Personality Summary
{synthesis.personality_summary}

## Training Readiness: {int(synthesis.training_readiness * 100)}%

## Voice Consistency: {int(synthesis.voice_consistency_score * 100)}%

## Development Suggestions
{chr(10).join(f"• {s}" for s in synthesis.development_suggestions)}

## World Connections
{chr(10).join(f"• {c}" for c in synthesis.world_connections)}
    """
    
    return report.strip() 