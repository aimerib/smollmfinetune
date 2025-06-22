"""
📋 Enhanced Character Management Studio

This enhanced version integrates the CharacterIntelligenceService with visual components
for a more engaging and intelligent character creation experience.
"""

import streamlit as st
import pandas as pd
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from utils.character import CharacterManager
from utils.character.models import CharacterCore, Personality, Relationship
from utils.character.character_intelligence import CharacterIntelligenceService
from utils.world import WorldManager
from utils.openai_client import get_client
from components.personality_editor import render_personality_editor, log_preference_event
from components.character_creation.conversational_builder import render_conversational_builder, reset_conversation
from components.character_creation.character_synthesis_preview import render_character_synthesis_preview

logger = logging.getLogger(__name__)


def create_character_selector():
    """Create enhanced character selection interface in sidebar"""
    st.sidebar.markdown("### 👤 Character Selection")
    
    char_manager = st.session_state.character_manager
    world_manager = st.session_state.world_manager
    
    # Get current world
    current_world = char_manager.get_current_world()
    if not current_world:
        st.sidebar.warning("⚠️ No world selected. Please select a world first.")
        return None
    
    st.sidebar.markdown(f"**World:** {current_world}")
    
    # List characters in current world
    characters = char_manager.list_characters_in_world()
    
    # Character creation options
    with st.sidebar.expander("✨ Create New Character", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗨️ Conversational", use_container_width=True):
                st.session_state.creation_mode = "conversational"
                reset_conversation()
                st.rerun()
        
        with col2:
            if st.button("📝 Traditional", use_container_width=True):
                st.session_state.creation_mode = "traditional"
                # Create new empty character
                st.session_state.current_character_core = CharacterCore(name="New Character")
                st.session_state.selected_character = "New Character"
                st.rerun()
    
    if not characters:
        st.sidebar.info("No characters found in this world.")
        if st.sidebar.button("📤 Import from Upload Page"):
            st.switch_page("pages/character_upload.py")
        return None
    
    # Character selector
    selected_char = st.sidebar.selectbox(
        "Select Character:",
        options=[""] + characters,
        index=0,
        key="character_selector"
    )
    
    if selected_char:
        # Load character if selection changed
        if st.session_state.get('selected_character') != selected_char:
            st.session_state.selected_character = selected_char
            st.session_state.creation_mode = "traditional"  # Switch to traditional mode
            
            # Load character data
            world_path = world_manager.get_world_path(current_world)
            char_path = world_path / "characters" / selected_char
            
            character_core = char_manager.load_character_core(char_path)
            if character_core:
                st.session_state.current_character_core = character_core
                st.sidebar.success(f"✅ Loaded {selected_char}")
            else:
                st.sidebar.error(f"❌ Failed to load {selected_char}")
                return None
    
    return selected_char


def render_enhanced_toolbar(core: CharacterCore, intelligence_service: CharacterIntelligenceService):
    """Render enhanced toolbar with AI-powered actions"""
    st.markdown("---")
    
    # Quick actions row
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        if st.button("💾 Save", type="primary", use_container_width=True):
            save_character(core)
    
    with col2:
        if st.button("🧠 AI Enhance", use_container_width=True):
            show_ai_enhancement_suggestions(core, intelligence_service)
    
    with col3:
        if st.button("🎭 Preview Voice", use_container_width=True):
            show_character_voice_preview(core, intelligence_service)
    
    with col4:
        if st.button("📋 Validate", use_container_width=True):
            show_character_validation(core, intelligence_service)
    
    with col5:
        if st.button("🚀 Generate Dataset", use_container_width=True):
            show_dataset_generation_options(core, intelligence_service)
    
    # Advanced actions row
    col1, col2, col3, col4, spacer = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        if st.button("↩️ Duplicate", use_container_width=True):
            duplicate_character(core)
    
    with col2:
        if st.button("📤 Export to ST", use_container_width=True):
            export_to_sillytavern(core, intelligence_service)
    
    with col3:
        if st.button("📊 Analytics", use_container_width=True):
            show_character_analytics(core, intelligence_service)
    
    with col4:
        if st.button("🗑️ Delete", use_container_width=True):
            delete_character_confirmation(core)


def show_ai_enhancement_suggestions(core: CharacterCore, intelligence_service: CharacterIntelligenceService):
    """Show AI-powered enhancement suggestions"""
    
    with st.expander("🧠 AI Enhancement Suggestions", expanded=True):
        try:
            suggestions = asyncio.run(intelligence_service.suggest_character_enhancements(core))
            
            if suggestions:
                st.markdown("**🔧 Recommended Improvements:**")
                for suggestion in suggestions:
                    priority_color = "#ef4444" if suggestion.priority >= 4 else "#f59e0b" if suggestion.priority >= 3 else "#22c55e"
                    
                    st.markdown(f"""
                        <div style="background: rgba(99, 102, 241, 0.1); 
                                    padding: 1rem; 
                                    border-radius: 8px; 
                                    border-left: 4px solid {priority_color}; 
                                    margin: 0.5rem 0;">
                            <strong>{suggestion.question}</strong><br>
                            <small style="color: #64748b;">Focus: {suggestion.focus_area} | {suggestion.reasoning}</small>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ Your character looks well-developed!")
                
        except Exception as e:
            st.error(f"Error generating suggestions: {str(e)}")


def show_character_voice_preview(core: CharacterCore, intelligence_service: CharacterIntelligenceService):
    """Show character voice preview"""
    
    with st.expander("🗣️ Character Voice Preview", expanded=True):
        try:
            preview_samples = asyncio.run(intelligence_service.generate_character_dataset_preview(core, num_samples=3))
            
            if preview_samples:
                st.markdown("**Sample Conversations:**")
                for i, sample in enumerate(preview_samples):
                    if 'messages' in sample and len(sample['messages']) >= 3:
                        user_msg = sample['messages'][1]['content']
                        char_msg = sample['messages'][2]['content']
                        
                        with st.expander(f"Conversation {i + 1}", expanded=i == 0):
                            st.markdown(f"**User:** {user_msg}")
                            st.markdown(f"**{core.name}:** {char_msg}")
            else:
                st.info("Add more character details to generate voice preview")
                
        except Exception as e:
            st.error(f"Error generating voice preview: {str(e)}")


def show_character_validation(core: CharacterCore, intelligence_service: CharacterIntelligenceService):
    """Show comprehensive character validation"""
    
    with st.expander("📋 Character Validation", expanded=True):
        try:
            validation = asyncio.run(intelligence_service.validate_character_for_training(core))
            
            # Overall status
            status_color = "#22c55e" if validation.is_ready_for_training else "#f59e0b"
            status_text = "Ready for Training" if validation.is_ready_for_training else "Needs Development"
            
            st.markdown(f"""
                <div style="background: rgba(34, 197, 94, 0.1); 
                            padding: 1rem; 
                            border-radius: 8px; 
                            border-left: 4px solid {status_color}; 
                            text-align: center;">
                    <strong>🎯 Status: {status_text}</strong>
                </div>
            """, unsafe_allow_html=True)
            
            # Scores
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Completeness", f"{int(validation.completeness_score * 100)}%")
            with col2:
                st.metric("Consistency", f"{int(validation.consistency_score * 100)}%")
            with col3:
                st.metric("Dataset Quality", f"{int(validation.estimated_dataset_quality * 100)}%")
            
            # Issues and recommendations
            if validation.issues:
                st.markdown("**⚠️ Issues to Address:**")
                for issue in validation.issues:
                    st.warning(f"• {issue}")
            
            if validation.recommendations:
                st.markdown("**💡 Recommendations:**")
                for rec in validation.recommendations:
                    st.info(f"• {rec}")
                    
        except Exception as e:
            st.error(f"Error validating character: {str(e)}")


def show_dataset_generation_options(core: CharacterCore, intelligence_service: CharacterIntelligenceService):
    """Show dataset generation options"""
    
    with st.expander("🚀 Dataset Generation", expanded=True):
        st.markdown("**Configure Dataset Generation:**")
        
        col1, col2 = st.columns(2)
        with col1:
            num_samples = st.number_input("Number of Samples", min_value=50, max_value=2000, value=500, step=50)
            temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
        
        with col2:
            quality_level = st.selectbox("Quality Level", ["Fast", "Standard", "Premium"])
            include_nsfw = st.checkbox("Include NSFW Content", value=False)
        
        if st.button("🚀 Start Generation", type="primary"):
            # This would start dataset generation
            st.info(f"Would start generating {num_samples} samples with {quality_level} quality")
            st.info("This would integrate with your existing DatasetManager!")


def export_to_sillytavern(core: CharacterCore, intelligence_service: CharacterIntelligenceService):
    """Export character to SillyTavern format"""
    
    try:
        sillytavern_card = asyncio.run(intelligence_service.export_for_sillytavern(core))
        
        st.download_button(
            label="📤 Download SillyTavern Card",
            data=str(sillytavern_card),
            file_name=f"{core.name}_sillytavern.json",
            mime="application/json"
        )
        
        st.success("✅ SillyTavern card ready for download!")
        
    except Exception as e:
        st.error(f"Error exporting to SillyTavern: {str(e)}")


def show_character_analytics(core: CharacterCore, intelligence_service: CharacterIntelligenceService):
    """Show character analytics and insights"""
    
    with st.expander("📊 Character Analytics", expanded=True):
        st.markdown("**Character Development Timeline:**")
        
        # This would show character evolution over time
        # For now, showing placeholder metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Trait Variance", f"{calculate_trait_variance(core.personality_traits):.2f}")
        
        with col2:
            st.metric("Goal Count", len(core.goals))
        
        with col3:
            st.metric("Relationship Count", len(core.relationships))
        
        with col4:
            content_richness = len(core.description) + len(core.backstory or "")
            st.metric("Content Richness", content_richness)


def calculate_trait_variance(personality: Personality) -> float:
    """Calculate personality trait variance"""
    traits = [
        personality.openness,
        personality.conscientiousness,
        personality.extraversion,
        personality.agreeableness,
        personality.neuroticism
    ]
    
    mean_trait = sum(traits) / len(traits)
    variance = sum((t - mean_trait) ** 2 for t in traits) / len(traits)
    return variance


def save_character(core: CharacterCore):
    """Save character with enhanced feedback"""
    char_manager = st.session_state.character_manager
    
    if char_manager.save_character(core):
        st.success("✅ Character saved successfully!")
        
        # Update world version
        world_manager = st.session_state.world_manager
        current_world = char_manager.get_current_world()
        if current_world:
            world_lore = world_manager.load_world(current_world)
            if world_lore:
                world_lore.meta.version += 1
                world_manager.write_lore(current_world, world_lore)
    else:
        st.error("❌ Failed to save character")


def duplicate_character(core: CharacterCore):
    """Duplicate character with new name"""
    new_name = f"{core.name}_Copy"
    new_core = CharacterCore(
        name=new_name,
        description=core.description,
        scenario=core.scenario,
        backstory=core.backstory,
        appearance=core.appearance,
        personality_traits=core.personality_traits,
        goals=core.goals.copy(),
        relationships=core.relationships.copy(),
        tags=core.tags.copy()
    )
    
    char_manager = st.session_state.character_manager
    if char_manager.save_character(new_core):
        st.success(f"✅ Character duplicated as {new_name}")
    else:
        st.error("❌ Failed to duplicate character")


def delete_character_confirmation(core: CharacterCore):
    """Show delete confirmation"""
    if st.session_state.get('confirm_delete', False):
        st.error("⚠️ Are you sure? This cannot be undone!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, Delete", type="primary"):
                # Actually delete the character
                char_manager = st.session_state.character_manager
                world_manager = st.session_state.world_manager
                current_world = char_manager.get_current_world()
                
                if current_world:
                    world_path = world_manager.get_world_path(current_world)
                    char_path = world_path / "characters" / core.name
                    
                    import shutil
                    try:
                        shutil.rmtree(char_path)
                        st.success("✅ Character deleted")
                        
                        # Clear session state
                        st.session_state.current_character_core = None
                        st.session_state.selected_character = None
                        st.session_state.confirm_delete = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting character: {e}")
        
        with col2:
            if st.button("Cancel"):
                st.session_state.confirm_delete = False
                st.rerun()
    else:
        st.session_state.confirm_delete = True
        st.rerun()


def page_character_management_enhanced():
    """Enhanced Character Management Studio page"""
    st.markdown('<h2 class="gradient-text">📋 Character Management Studio (Enhanced)</h2>', unsafe_allow_html=True)
    
    # Initialize managers
    if 'character_manager' not in st.session_state:
        st.session_state.character_manager = CharacterManager()
    
    if 'world_manager' not in st.session_state:
        st.session_state.world_manager = WorldManager()
    
    if 'char_intelligence' not in st.session_state:
        st.session_state.char_intelligence = CharacterIntelligenceService(st.session_state.world_manager)
    
    intelligence_service = st.session_state.char_intelligence
    
    # Character selection
    selected_character = create_character_selector()
    
    # Check creation mode
    creation_mode = st.session_state.get('creation_mode', 'traditional')
    
    if creation_mode == 'conversational':
        # Conversational character creation mode
        completed_character = render_conversational_builder(
            st.session_state.world_manager,
            character_name=""
        )
        
        if completed_character:
            # Character creation completed, save and switch to traditional mode
            char_manager = st.session_state.character_manager
            if char_manager.save_character(completed_character):
                st.session_state.current_character_core = completed_character
                st.session_state.selected_character = completed_character.name
                st.session_state.creation_mode = 'traditional'
                st.success(f"✅ Character '{completed_character.name}' created successfully!")
                st.rerun()
        
        return  # Skip traditional interface in conversational mode
    
    # Traditional tabbed interface
    if not selected_character or 'current_character_core' not in st.session_state:
        st.info("👈 Select a character from the sidebar to begin editing, or create a new one")
        
        # Show introduction to new features
        render_features_showcase()
        return
    
    # Get current character
    core = st.session_state.current_character_core
    
    # Main interface layout
    main_col, synthesis_col = st.columns([2, 1])
    
    with main_col:
        # Display character info header
        st.markdown(f"### Editing: **{core.name}**")
        
        # Create tabs for traditional editing
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Profile", "🧠 Personality", "🎯 Goals & Relationships", "💬 Examples"])
        
        with tab1:
            render_profile_tab_enhanced(core, intelligence_service)
        
        with tab2:
            render_personality_editor(core, key_prefix="enhanced_pers_", show_ai_btn=True)
        
        with tab3:
            render_goals_relationships_tab(core)
        
        with tab4:
            render_examples_tab(core)
        
        # Enhanced toolbar
        render_enhanced_toolbar(core, intelligence_service)
    
    with synthesis_col:
        # Live character synthesis preview
        st.markdown("### 🎭 Live Character Analysis")
        render_character_synthesis_preview(core, intelligence_service, key_prefix="mgmt_synthesis")


def render_features_showcase():
    """Render showcase of new enhanced features"""
    
    st.markdown("### ✨ Enhanced Character Studio Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="background: rgba(99, 102, 241, 0.1); 
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        margin: 1rem 0;">
                <h4 style="color: #6366f1; margin-top: 0;">🗨️ Conversational Creation</h4>
                <p style="color: #64748b; font-size: 0.9rem;">
                    AI-guided character discovery through natural conversation. 
                    No more empty form fields - just tell us about your character!
                </p>
                <ul style="color: #64748b; font-size: 0.9rem;">
                    <li>Intelligent question flow</li>
                    <li>Real-time character evolution</li>
                    <li>Smart insight extraction</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background: rgba(168, 85, 247, 0.1); 
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        margin: 1rem 0;">
                <h4 style="color: #a855f7; margin-top: 0;">🧠 AI Intelligence</h4>
                <p style="color: #64748b; font-size: 0.9rem;">
                    Advanced character analysis using your sophisticated dataset pipeline.
                </p>
                <ul style="color: #64748b; font-size: 0.9rem;">
                    <li>Voice consistency scoring</li>
                    <li>Training readiness assessment</li>
                    <li>Character archetype detection</li>
                    <li>NSFW content analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background: rgba(34, 197, 94, 0.1); 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    margin: 1rem 0;">
            <h4 style="color: #22c55e; margin-top: 0;">🚀 Integrated Dataset Pipeline</h4>
            <p style="color: #64748b; font-size: 0.9rem;">
                Seamless integration with your existing dataset generation and training pipeline.
                Character validation, preview generation, and quality assessment built right in.
            </p>
        </div>
    """, unsafe_allow_html=True)


def render_profile_tab_enhanced(core: CharacterCore, intelligence_service: CharacterIntelligenceService):
    """Enhanced profile tab with AI assistance"""
    
    st.markdown("### 📝 Character Profile")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Basic information (similar to original but with AI hints)
        core.name = st.text_input(
            "Character Name",
            value=core.name,
            key="enhanced_profile_name",
            help="The character's name"
        )
        
        # Description with enhanced AI helper
        core.description = st.text_area(
            "Description",
            value=core.description,
            height=120,
            key="enhanced_profile_description",
            help="Core character description and traits"
        )
        
        if st.button("✨ AI Enhance Description", key="enhance_desc"):
            # This would use the intelligence service to enhance description
            st.info("AI enhancement would improve the description based on character analysis")
        
        # Other fields...
        core.scenario = st.text_area(
            "Scenario",
            value=core.scenario,
            height=100,
            key="enhanced_profile_scenario",
            help="The setting or context where the character exists"
        )
        
        core.backstory = st.text_area(
            "Backstory", 
            value=core.backstory,
            height=120,
            key="enhanced_profile_backstory",
            help="Character's history and background"
        )
    
    with col2:
        # AI insights panel
        st.markdown("**🧠 AI Insights**")
        
        if core.description:
            try:
                # Quick character analysis
                char_dict = {
                    'name': core.name,
                    'description': core.description,
                    'personality': '',
                    'scenario': core.scenario or '',
                    'backstory': core.backstory or ''
                }
                
                from utils.dataset.character_analysis import extract_character_knowledge
                knowledge = extract_character_knowledge(char_dict)
                
                if knowledge.get('traits', []):
                    st.markdown("**Detected Traits:**")
                    for trait in knowledge['traits'][:3]:
                        st.write(f"• {trait}")
                
                if knowledge.get('goals', []):
                    st.markdown("**Potential Goals:**")
                    for goal in knowledge['goals'][:2]:
                        st.write(f"• {goal}")
                        
            except Exception as e:
                st.info("Add character details to see AI insights")
        
        # Appearance
        core.appearance = st.text_area(
            "Physical Description",
            value=core.appearance,
            height=100,
            key="enhanced_profile_appearance",
            help="Character's physical appearance"
        )
        
        # Tags
        tags_str = ", ".join(core.tags) if core.tags else ""
        new_tags_str = st.text_input(
            "Tags (comma-separated)",
            value=tags_str,
            key="enhanced_profile_tags",
            help="Keywords that describe the character"
        )
        
        if new_tags_str.strip():
            core.tags = [tag.strip() for tag in new_tags_str.split(",") if tag.strip()]
        else:
            core.tags = []


def render_goals_relationships_tab(core: CharacterCore):
    """Enhanced goals and relationships tab"""
    
    st.markdown("### 🎯 Goals & Relationships")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Goals")
        
        # Goals editor (same as original)
        goals_df = pd.DataFrame({
            "Goal": core.goals + [""] * max(0, 3 - len(core.goals))
        })
        
        edited_goals = st.data_editor(
            goals_df,
            num_rows="dynamic",
            key="enhanced_goals_editor",
            use_container_width=True
        )
        
        core.goals = [goal for goal in edited_goals["Goal"].tolist() if goal.strip()]
        
        # AI goal suggestions
        if st.button("🧠 AI Suggest Goals", key="ai_goals_enhanced"):
            st.info("AI would suggest goals based on character personality and background")
    
    with col2:
        st.markdown("#### Relationships")
        
        # Relationships editor (same as original)
        if core.relationships:
            rel_data = {
                "Character": [rel.name for rel in core.relationships],
                "Affinity": [rel.affinity for rel in core.relationships]
            }
        else:
            rel_data = {"Character": [""], "Affinity": [0]}
        
        rel_df = pd.DataFrame(rel_data)
        
        edited_relationships = st.data_editor(
            rel_df,
            num_rows="dynamic",
            key="enhanced_relationships_editor",
            use_container_width=True,
            column_config={
                "Affinity": st.column_config.NumberColumn(
                    "Affinity",
                    help="Relationship affinity (-10 to +10)",
                    min_value=-10,
                    max_value=10,
                    step=1
                )
            }
        )
        
        core.relationships = [
            Relationship(name=name, affinity=affinity)
            for name, affinity in zip(edited_relationships["Character"], edited_relationships["Affinity"])
            if name.strip()
        ]


def render_examples_tab(core: CharacterCore):
    """Enhanced examples tab with AI assistance"""
    
    st.markdown("### 💬 Dialogue Examples")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader and editor (same as original)
        uploaded_file = st.file_uploader(
            "Upload Example File",
            type=['txt'],
            key="enhanced_examples_uploader",
            help="Upload a mes_example.txt file"
        )
        
        # Load existing example
        example_text = ""
        if uploaded_file is not None:
            try:
                example_text = str(uploaded_file.read(), "utf-8")
                st.success("✅ Example file uploaded!")
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
        
        # Text editor
        updated_example = st.text_area(
            "Dialogue Examples",
            value=example_text,
            height=400,
            key="enhanced_examples_editor",
            help="Example conversations showing the character's voice and style"
        )
        
        # Token count
        token_count = len(updated_example.split())
        st.caption(f"📊 Estimated tokens: ~{token_count}")
        
        st.session_state.current_example_text = updated_example
    
    with col2:
        # AI assistance panel
        st.markdown("**🤖 AI Assistant**")
        
        if st.button("✨ Generate Examples", key="ai_examples_enhanced"):
            st.info("AI would generate character-appropriate dialogue examples")
        
        if st.button("🔍 Analyze Voice", key="analyze_voice_enhanced"):
            if updated_example:
                st.info("AI would analyze the character's voice consistency and style")
            else:
                st.warning("Add some examples first")
        
        # Style guide
        with st.expander("📖 Style Guide"):
            st.markdown("""
                **Good Examples Include:**
                - Character-specific mannerisms
                - Consistent speech patterns
                - Emotional range demonstration
                - World-appropriate knowledge
                """)


# Run the enhanced page
if __name__ == "__main__":
    page_character_management_enhanced() 