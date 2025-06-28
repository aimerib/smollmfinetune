"""
World Context Panel Component

Displays relevant world lore and integration suggestions during character creation.
"""

import streamlit as st
import asyncio
from typing import Dict, Any, List, Optional
import logging

try:
    from utils.character.models import CharacterCore
    from utils.character.world_character_integration import WorldCharacterIntegrator, WorldIntegrationSuggestions
    from utils.world import WorldManager
except ImportError:
    from app.utils.character.models import CharacterCore
    from app.utils.character.world_character_integration import WorldCharacterIntegrator, WorldIntegrationSuggestions
    from app.utils.world import WorldManager

logger = logging.getLogger(__name__)


def render_world_context_panel(character: CharacterCore, world_name: str, 
                             world_manager: WorldManager, 
                             key_prefix: str = "world_ctx") -> Optional[WorldIntegrationSuggestions]:
    """
    Render world context panel showing relevant lore and integration suggestions.
    
    Args:
        character: Character being created/edited
        world_name: Name of the current world
        world_manager: World manager instance
        key_prefix: Unique key prefix for Streamlit components
        
    Returns:
        World integration suggestions if analysis was performed
    """
    if not character.name:
        st.info("👈 Add character name to see world integration suggestions")
        return None
    
    with st.container():
        st.markdown("### 🌍 World Integration")
        
        # Load world lore
        world_lore = world_manager.load_world(world_name)
        if not world_lore:
            st.warning(f"No world lore found for '{world_name}'")
            return None
        
        # Show relevant world context
        with st.expander("📖 Relevant World Lore", expanded=False):
            _render_relevant_world_lore(world_lore, character)
        
        # Generate integration suggestions
        if st.button("🔍 Analyze World Integration", key=f"{key_prefix}_analyze"):
            with st.spinner("Analyzing character's fit within world lore..."):
                try:
                    integrator = WorldCharacterIntegrator(world_manager)
                    suggestions = asyncio.run(
                        integrator.analyze_character_world_fit(character, world_name)
                    )
                    
                    # Store in session state for persistence
                    st.session_state[f"{key_prefix}_suggestions"] = suggestions
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    logger.error(f"World integration analysis failed: {e}")
                    return None
        
        # Display suggestions if available
        suggestions_key = f"{key_prefix}_suggestions"
        if suggestions_key in st.session_state:
            suggestions = st.session_state[suggestions_key]
            _render_integration_suggestions(suggestions, key_prefix)
            return suggestions
        
        return None


def _render_relevant_world_lore(world_lore, character: CharacterCore):
    """Render relevant world lore based on character details"""
    
    # Show world facts
    if world_lore.facts:
        st.markdown("**World Facts:**")
        for key, value in list(world_lore.facts.items())[:5]:  # Show top 5
            st.markdown(f"• **{key}**: {value}")
    
    # Show recent timeline events
    if world_lore.timeline:
        st.markdown("**Recent Timeline:**")
        for event in world_lore.timeline[-3:]:  # Last 3 events
            st.markdown(f"• **{event.year}**: {event.event}")
    
    # Show factions
    if world_lore.factions:
        st.markdown("**Active Factions:**")
        for faction in world_lore.factions[:3]:  # Top 3 factions
            st.markdown(f"• **{faction.name}**")
    
    # Show places with NPCs
    places_with_npcs = [place for place in world_lore.places if place.npcs]
    if places_with_npcs:
        st.markdown("**Key Locations:**")
        for place in places_with_npcs[:3]:  # Top 3 places
            npc_count = len(place.npcs)
            st.markdown(f"• **{place.name}** ({npc_count} NPCs)")


def _render_integration_suggestions(suggestions: WorldIntegrationSuggestions, key_prefix: str):
    """Render the world integration suggestions"""
    
    # Integration score
    score_color = "🟢" if suggestions.integration_score >= 0.7 else "🟡" if suggestions.integration_score >= 0.4 else "🔴"
    st.markdown(f"**Integration Score:** {score_color} {suggestions.integration_score:.1%}")
    
    # Timeline connections
    if suggestions.timeline_connections:
        with st.expander(f"⏰ Timeline Connections ({len(suggestions.timeline_connections)})", expanded=True):
            for i, conn in enumerate(suggestions.timeline_connections):
                relevance_bar = "🔥" if conn.relevance_score >= 0.8 else "⭐" if conn.relevance_score >= 0.6 else "💡"
                
                st.markdown(f"**{conn.event_year}: {conn.event_description}**")
                st.markdown(f"{relevance_bar} *{conn.character_involvement}*")
                if conn.suggested_age:
                    st.markdown(f"📅 Character would be ~{conn.suggested_age} years old")
                
                # Add relationship button
                if st.button(f"Add to Character History", key=f"{key_prefix}_timeline_{i}"):
                    st.success("Timeline connection noted! You can incorporate this into the character's backstory.")
                
                st.markdown("---")
    
    # Faction recommendations
    if suggestions.faction_recommendations:
        with st.expander(f"⚔️ Faction Compatibility ({len(suggestions.faction_recommendations)})", expanded=True):
            for i, rec in enumerate(suggestions.faction_recommendations):
                compatibility_emoji = "🎯" if rec.compatibility_score >= 0.8 else "👍" if rec.compatibility_score >= 0.6 else "🤔"
                
                st.markdown(f"**{rec.faction_name}** {compatibility_emoji} {rec.compatibility_score:.1%}")
                st.markdown(f"*{rec.reasoning}*")
                st.markdown(f"**Suggested Role:** {rec.suggested_role}")
                
                if rec.potential_conflicts:
                    st.markdown("**Potential Challenges:**")
                    for conflict in rec.potential_conflicts:
                        st.markdown(f"• {conflict}")
                
                # Add relationship button
                if st.button(f"Join {rec.faction_name}", key=f"{key_prefix}_faction_{i}"):
                    st.success(f"Faction membership noted! Consider adding this to character relationships.")
                
                st.markdown("---")
    
    # NPC relationships
    if suggestions.npc_relationships:
        with st.expander(f"👥 NPC Relationships ({len(suggestions.npc_relationships)})", expanded=True):
            for i, rel in enumerate(suggestions.npc_relationships):
                affinity_emoji = "❤️" if rel.affinity_score >= 5 else "👍" if rel.affinity_score > 0 else "👎" if rel.affinity_score < 0 else "😐"
                
                st.markdown(f"**{rel.npc_name}** ({rel.relationship_type}) {affinity_emoji} {rel.affinity_score:+d}")
                st.markdown(f"*{rel.backstory}*")
                if rel.interaction_history:
                    st.markdown(f"**History:** {rel.interaction_history}")
                
                # Add relationship button
                if st.button(f"Add Relationship with {rel.npc_name}", key=f"{key_prefix}_npc_{i}"):
                    st.success("Relationship noted! You can add this to the character's relationships tab.")
                
                st.markdown("---")
    
    # World lore proposals
    if suggestions.proposed_world_updates:
        with st.expander(f"📝 Proposed World Lore ({len(suggestions.proposed_world_updates)})", expanded=False):
            st.markdown("*These are new lore elements this character's existence suggests:*")
            
            for i, proposal in enumerate(suggestions.proposed_world_updates):
                category_emoji = {"timeline": "⏰", "faction": "⚔️", "place": "🏛️", "fact": "📋"}.get(proposal.category, "📝")
                
                st.markdown(f"**{category_emoji} {proposal.title}**")
                st.markdown(f"*{proposal.description}*")
                st.markdown(f"**Why:** {proposal.justification}")
                st.markdown(f"**Impact:** {proposal.impact_assessment}")
                
                if st.button(f"Propose to World Lore", key=f"{key_prefix}_lore_{i}"):
                    st.info("Lore proposal noted! Consider adding this to your world's timeline or facts.")
                
                st.markdown("---")
    
    # Consistency warnings
    if suggestions.consistency_warnings:
        with st.expander("⚠️ Consistency Warnings", expanded=True):
            for warning in suggestions.consistency_warnings:
                st.warning(warning)


def render_ecosystem_insights(world_name: str, intelligence_service, key_prefix: str = "ecosystem"):
    """
    Render character ecosystem insights for the current world.
    """
    st.markdown("### 🎭 Character Ecosystem")
    
    if st.button("🔍 Analyze Character Ecosystem", key=f"{key_prefix}_analyze"):
        with st.spinner("Analyzing character ecosystem..."):
            try:
                ecosystem = asyncio.run(
                    intelligence_service.analyze_character_ecosystem(world_name)
                )
                
                # Store in session state
                st.session_state[f"{key_prefix}_analysis"] = ecosystem
                st.rerun()
                
            except Exception as e:
                st.error(f"Ecosystem analysis failed: {str(e)}")
                return
    
    # Display ecosystem analysis if available
    analysis_key = f"{key_prefix}_analysis"
    if analysis_key in st.session_state:
        ecosystem = st.session_state[analysis_key]
        _render_ecosystem_analysis(ecosystem, key_prefix)


def _render_ecosystem_analysis(ecosystem, key_prefix: str):
    """Render the ecosystem analysis results"""
    
    # World integration score
    score_color = "🟢" if ecosystem.world_integration_score >= 0.7 else "🟡" if ecosystem.world_integration_score >= 0.4 else "🔴"
    st.markdown(f"**World Integration:** {score_color} {ecosystem.world_integration_score:.1%}")
    
    # Character roles distribution
    if ecosystem.character_roles:
        with st.expander("🎭 Character Archetypes", expanded=True):
            for archetype, characters in ecosystem.character_roles.items():
                st.markdown(f"**{archetype.title()}**: {', '.join(characters)}")
    
    # Personality distribution
    if ecosystem.character_distribution:
        with st.expander("🧠 Personality Distribution", expanded=False):
            for trait, count in ecosystem.character_distribution.items():
                st.markdown(f"**{trait}**: {count} characters")
    
    # Narrative opportunities
    if ecosystem.narrative_opportunities:
        with st.expander("📖 Story Opportunities", expanded=True):
            for opportunity in ecosystem.narrative_opportunities:
                st.markdown(f"• {opportunity}")
    
    # Ecosystem gaps
    if ecosystem.ecosystem_gaps:
        with st.expander("🎯 Character Suggestions", expanded=True):
            st.markdown("*Consider creating characters to fill these gaps:*")
            for gap in ecosystem.ecosystem_gaps:
                st.markdown(f"• {gap}")
    
    # Relationship dynamics
    if ecosystem.relationship_dynamics:
        with st.expander("🤝 Relationship Network", expanded=False):
            for char_name, relationships in ecosystem.relationship_dynamics.items():
                if relationships:
                    rel_summary = ", ".join([f"{name} ({affinity:+d})" for name, affinity in relationships.items()])
                    st.markdown(f"**{char_name}**: {rel_summary}")


def render_preference_insights(intelligence_service, key_prefix: str = "prefs"):
    """
    Render user preference insights if available.
    """
    insights = intelligence_service.get_user_preference_insights()
    
    if insights["status"] == "insufficient_data":
        st.info(f"🎯 Learning your preferences... ({insights['interactions']}/5 interactions)")
        return
    
    with st.expander("🎯 Your Character Creation Style", expanded=False):
        st.markdown(f"**Confidence:** {insights['confidence']:.1%} ({insights['total_interactions']} interactions)")
        st.markdown(f"**Narrative Style:** {insights['narrative_style'].title()}")
        
        if insights['preferred_archetypes']:
            st.markdown(f"**Favorite Archetypes:** {', '.join(insights['preferred_archetypes'])}")
        
        if insights['personality_tendencies']:
            st.markdown("**Personality Preferences:**")
            for trait, tendency in insights['personality_tendencies'].items():
                st.markdown(f"• {trait.title()}: {tendency}") 