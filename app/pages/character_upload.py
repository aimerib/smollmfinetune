"""
📤 Character Upload & Import Studio

Unified import flow: Upload → AI Analysis (always) → Optional Conversational Enhancement → Character Ready
"""

import streamlit as st
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from utils.openai_client import get_client
from utils.character import CharacterManager
from utils.character.models import CharacterCore
from utils.character.character_intelligence import CharacterIntelligenceService
from utils.world import WorldManager
from utils.dataset.character_analysis import extract_character_knowledge
from utils.dataset.content_evaluation import is_nsfw_content, categorize_nsfw_style

logger = logging.getLogger(__name__)


def page_character_upload():
    """Main Character Upload page with unified import flow"""
    
    st.markdown('<h2 class="gradient-text">📤 Character Import Studio</h2>', unsafe_allow_html=True)
    
    # Initialize managers

    if 'world_manager' not in st.session_state:
        st.session_state.world_manager = WorldManager()
    
    if 'character_manager' not in st.session_state:
        st.session_state.character_manager = CharacterManager(world_manager=st.session_state.world_manager, client=get_client())
    
    if 'character_intelligence' not in st.session_state:
        st.session_state.character_intelligence = CharacterIntelligenceService(
            world_manager=st.session_state.world_manager
        )
    
    # Check current world
    char_manager = st.session_state.character_manager
    current_world = char_manager.get_current_world()
    
    if not current_world:
        st.warning("⚠️ No world selected. Please select a world first.")
        if st.button("🌍 Go to World Management", type="primary"):
            st.switch_page("pages/world_management.py")
        return
    
    st.info(f"🌍 Importing to world: **{current_world}**")
    
    # Check import flow state
    import_state = st.session_state.get('import_flow_state', 'upload')
    
    if import_state == 'upload':
        render_upload_phase()
    elif import_state == 'analysis':
        render_analysis_phase()
    elif import_state == 'enhancement_choice':
        render_enhancement_choice_phase()
    elif import_state == 'conversational':
        render_conversational_enhancement_phase()
    elif import_state == 'complete':
        render_completion_phase()


def render_upload_phase():
    """Phase 1: Upload and validate SillyTavern card"""
    
    st.markdown("### 📂 Upload Character Card")
    
    # Create a simple column layout for the upload interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="custom-card">
                <p style="color: rgba(255,255,255,0.8);">
                    Upload a SillyTavern character card (JSON format) to begin the import process.
                    Our AI will analyze the character and help you enhance them.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose character card file",
            type=['json'],
            help="Upload a SillyTavern character card in JSON format"
        )
    
    with col2:
        st.markdown("#### 📋 Quick Guide")
        st.markdown("""
        1. Upload JSON card
        2. AI analyzes character  
        3. Optional enhancement
        4. Character ready!
        """)
        st.info("💡 Tip: NSFW content is automatically detected and categorized")
    
    if uploaded_file is not None:
        try:
            # Load and validate the character card
            card_data = json.loads(uploaded_file.read())
            
            # Basic validation
            if not isinstance(card_data, dict):
                st.error("❌ Invalid file: Expected JSON object")
                return
            
            if 'name' not in card_data or not card_data['name'].strip():
                st.error("❌ Invalid character card: Missing character name")
                return
            
            # Check for minimum content
            has_content = any([
                card_data.get('description', '').strip(),
                card_data.get('personality', '').strip(),
                card_data.get('mes_example', '').strip()
            ])
            
            if not has_content:
                st.error("❌ Character card appears to be empty (no description, personality, or examples)")
                return
            
            # Store the card data and move to analysis phase
            st.session_state.uploaded_card_data = card_data
            st.session_state.import_flow_state = 'analysis'
            st.success(f"✅ Successfully uploaded character: **{card_data['name']}**")
            st.rerun()
            
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file. Please check the file format.")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Help section
    with st.expander("📖 Need Help?"):
        st.markdown("""
        **Supported Format:** SillyTavern character cards (`.json` files)
        
        **Required Fields:**
        - `name`: Character name
        - At least one of: `description`, `personality`, or `mes_example`
        
        **Optional Fields:**
        - `scenario`: Setting/context
        - `first_mes`: Opening message
        - `avatar`: Character image (preserved)
        """)


def render_analysis_phase():
    """Phase 2: AI analyzes the uploaded character"""
    
    card_data = st.session_state.get('uploaded_card_data')
    if not card_data:
        st.error("❌ No character data found. Please upload a character card first.")
        st.session_state.import_flow_state = 'upload'
        st.rerun()
        return
    
    st.markdown(f"### 🧠 Analyzing Character: **{card_data['name']}**")
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Analysis container
    analysis_container = st.container()
    
    with analysis_container:
        if 'analysis_complete' not in st.session_state:
            # Perform analysis
            with st.spinner("🤖 AI is analyzing your character..."):
                try:
                    async def perform_analysis():
                        # Step 1: Convert to CharacterCore (this does the heavy lifting)
                        status_text.text("🔍 Extracting character knowledge...")
                        progress_bar.progress(20)
                        
                        char_manager = st.session_state.character_manager
                        character_core = await char_manager.import_sillytavern_card(card_data)
                        
                        # Step 2: Advanced analysis
                        status_text.text("🧠 Analyzing personality traits...")
                        progress_bar.progress(50)
                        
                        knowledge = extract_character_knowledge(card_data)
                        
                        # Step 3: Content evaluation
                        status_text.text("🔍 Evaluating content style...")
                        progress_bar.progress(80)
                        
                        content_text = f"{card_data.get('description', '')} {card_data.get('personality', '')}"
                        # Create a temporary client for NSFW detection
                        from utils.openai_client import get_client
                        client = get_client()
                        nsfw_content = await is_nsfw_content(client, content_text)
                        nsfw_style = await categorize_nsfw_style(client, content_text) if nsfw_content else None
                        
                        # Step 4: Enhancement recommendations
                        status_text.text("💡 Generating enhancement recommendations...")
                        progress_bar.progress(100)
                        
                        intelligence_service = st.session_state.character_intelligence
                        enhancement_suggestions = await intelligence_service.suggest_character_enhancements(character_core)
                        
                        return character_core, knowledge, nsfw_content, nsfw_style, enhancement_suggestions
                    
                    # Run the async analysis
                    character_core, knowledge, nsfw_content, nsfw_style, enhancement_suggestions = asyncio.run(perform_analysis())
                    
                    # Store analysis results
                    st.session_state.character_core = character_core
                    st.session_state.character_knowledge = knowledge
                    st.session_state.nsfw_assessment = {
                        'has_nsfw': nsfw_content,
                        'style': nsfw_style
                    }
                    st.session_state.enhancement_suggestions = enhancement_suggestions
                    st.session_state.analysis_complete = True
                    
                    status_text.text("✅ Analysis complete!")
                    st.success("🎉 Character analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    status_text.text("❌ Analysis failed")
                    return
        
        # Show analysis results
        if st.session_state.get('analysis_complete'):
            character_core = st.session_state.character_core
            knowledge = st.session_state.character_knowledge
            nsfw_assessment = st.session_state.nsfw_assessment
            enhancement_suggestions = st.session_state.enhancement_suggestions
            
            st.markdown("---")
            st.markdown("### 📊 Analysis Results")
            
            # Character overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Content Richness", f"{len(character_core.description)} chars")
                st.metric("Goals Identified", len(character_core.goals))
            
            with col2:
                st.metric("Knowledge Domains", len(knowledge.get('knowledge_domains', [])))
                st.metric("Speech Patterns", len(knowledge.get('speech_patterns', [])))
            
            with col3:
                trait_variance = calculate_personality_variance(character_core.personality_traits)
                st.metric("Personality Depth", f"{trait_variance:.2f}")
                st.metric("Relationships", len(character_core.relationships))
            
            # Detailed breakdown
            with st.expander("🔍 Detailed Analysis", expanded=True):
                
                tab1, tab2, tab3 = st.tabs(["🧠 Personality", "📖 Knowledge", "💡 Suggestions"])
                
                with tab1:
                    st.markdown("**Big Five Personality Traits:**")
                    traits = character_core.personality_traits
                    
                    for trait_name, value in [
                        ("Openness", traits.openness),
                        ("Conscientiousness", traits.conscientiousness), 
                        ("Extraversion", traits.extraversion),
                        ("Agreeableness", traits.agreeableness),
                        ("Neuroticism", traits.neuroticism)
                    ]:
                        # Create a visual bar
                        bar_width = int(value * 20)  # 0-20 chars
                        bar = "█" * bar_width + "░" * (20 - bar_width)
                        st.markdown(f"**{trait_name}:** `{bar}` {value:.2f}")
                
                with tab2:
                    if knowledge.get('traits'):
                        st.markdown("**Identified Traits:**")
                        for trait in knowledge['traits'][:5]:
                            st.markdown(f"• {trait}")
                    
                    if knowledge.get('goals'):
                        st.markdown("**Extracted Goals:**")
                        for goal in knowledge['goals'][:3]:
                            st.markdown(f"• {goal}")
                    
                    if knowledge.get('knowledge_domains'):
                        st.markdown("**Knowledge Areas:**")
                        for domain in knowledge['knowledge_domains'][:3]:
                            st.markdown(f"• {domain}")
                
                with tab3:
                    if enhancement_suggestions:
                        st.markdown("**AI Enhancement Recommendations:**")
                        for suggestion in enhancement_suggestions[:3]:
                            priority_emoji = "🔴" if suggestion.priority >= 4 else "🟡" if suggestion.priority >= 3 else "🟢"
                            st.markdown(f"{priority_emoji} **{suggestion.focus_area.title()}:** {suggestion.question}")
                    else:
                        st.info("Your character is well-developed! No critical enhancements needed.")
            
            # NSFW assessment if relevant
            if nsfw_assessment['has_nsfw']:
                st.warning("⚠️ NSFW content detected. Dataset generation will use appropriate filtering.")
            
            # Continue button
            if st.button("➡️ Continue to Enhancement Options", type="primary", use_container_width=True):
                st.session_state.import_flow_state = 'enhancement_choice'
                st.rerun()


def render_enhancement_choice_phase():
    """Phase 3: Choose enhancement path"""
    
    character_core = st.session_state.get('character_core')
    enhancement_suggestions = st.session_state.get('enhancement_suggestions', [])
    
    if not character_core:
        st.error("❌ Character data not found. Starting over...")
        reset_import_flow()
        return
    
    st.markdown(f"### 🎯 Choose Enhancement Path for **{character_core.name}**")
    
    # Determine recommendation based on analysis
    needs_enhancement = len(enhancement_suggestions) > 0
    complexity_score = calculate_character_complexity(character_core)
    
    # Show character preview
    with st.expander("📋 Character Preview", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Description:** {character_core.description[:200]}{'...' if len(character_core.description) > 200 else ''}")
            st.markdown(f"**Goals:** {', '.join(character_core.goals[:3])}")
        with col2:
            st.markdown(f"**Tags:** {', '.join(character_core.tags[:5])}")
            st.markdown(f"**Relationships:** {len(character_core.relationships)} defined")
    
    st.markdown("---")
    
    # Enhancement options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="background: rgba(34, 197, 94, 0.1); 
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        margin: 1rem 0;
                        border: 2px solid rgba(34, 197, 94, 0.3);">
                <h4 style="color: #22c55e; margin-top: 0;">✅ Complete Import</h4>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0.5rem 0;">
                    Save character as-is with AI analysis complete. Ready for dataset generation.
                </p>
                <ul style="color: #64748b; font-size: 0.85rem; margin: 0.5rem 0;">
                    <li>Big Five personality analyzed</li>
                    <li>Knowledge domains extracted</li>
                    <li>Goals and traits identified</li>
                    <li>Ready for training immediately</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("✅ Complete Import", type="primary", use_container_width=True):
            complete_import()
    
    with col2:
        # Recommendation logic
        if needs_enhancement:
            border_color = "rgba(168, 85, 247, 0.4)"
            recommendation_text = "🎯 **Recommended** - Your character could benefit from deeper development"
        elif complexity_score < 0.6:
            border_color = "rgba(245, 158, 11, 0.4)" 
            recommendation_text = "💡 **Suggested** - Add more personality depth and detail"
        else:
            border_color = "rgba(99, 102, 241, 0.3)"
            recommendation_text = "🔮 **Optional** - Further develop your already rich character"
        
        st.markdown(f"""
            <div style="background: rgba(168, 85, 247, 0.1); 
                        padding: 1.5rem; 
                        border-radius: 12px; 
                        margin: 1rem 0;
                        border: 2px solid {border_color};">
                <h4 style="color: #a855f7; margin-top: 0;">🗨️ Conversational Enhancement</h4>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0.5rem 0;">
                    AI-guided interview to discover deeper personality, voice, and relationships.
                </p>
                <p style="color: #64748b; font-size: 0.85rem; margin: 0.5rem 0;">
                    {recommendation_text}
                </p>
                <ul style="color: #64748b; font-size: 0.85rem; margin: 0.5rem 0;">
                    <li>Personality-aware conversation flow</li>
                    <li>Voice consistency development</li>
                    <li>Relationship depth building</li>
                    <li>5-10 minute guided interview</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗨️ Enhance Conversationally", use_container_width=True):
            st.session_state.import_flow_state = 'conversational'
            st.session_state.enhance_existing_character = character_core
            st.rerun()
    
    st.markdown("---")
    
    # Show what enhancement would focus on
    if enhancement_suggestions:
        st.markdown("### 💡 Enhancement Focus Areas")
        focus_areas = {}
        for suggestion in enhancement_suggestions:
            area = suggestion.focus_area
            if area not in focus_areas:
                focus_areas[area] = []
            focus_areas[area].append(suggestion.question)
        
        cols = st.columns(min(3, len(focus_areas)))
        for i, (area, questions) in enumerate(focus_areas.items()):
            with cols[i % len(cols)]:
                st.markdown(f"**{area.title()}**")
                for question in questions[:2]:
                    st.markdown(f"• {question[:60]}{'...' if len(question) > 60 else ''}")


def render_conversational_enhancement_phase():
    """Phase 4: Conversational enhancement using the character builder"""
    
    character_core = st.session_state.get('character_core')
    if not character_core:
        st.error("❌ Character data not found. Starting over...")
        reset_import_flow()
        return
    
    # Import and use the conversational builder
    from components.character_creation.conversational_builder import render_conversational_builder
    
    st.markdown(f"### 🗨️ Enhancing: **{character_core.name}**")
    
    st.markdown(f"""
        <div class="custom-card">
            <h4 style="color: white; margin-top: 0;">🎭 AI-Guided Character Enhancement</h4>
            <p style="color: rgba(255,255,255,0.8);">
                Based on our analysis, let's dive deeper into <strong>{character_core.name}</strong>'s personality, 
                voice, and inner world. The AI will ask targeted questions to help develop areas that need more depth.
            </p>
            <div style="background: rgba(168, 85, 247, 0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <strong>🧠 Using Personality Analysis:</strong><br>
                We'll use {character_core.name}'s Big Five traits to guide the conversation and suggest 
                dialogue patterns that match their personality.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Use the conversational builder with the existing character
    world_manager = st.session_state.world_manager
    completed_character = render_conversational_builder(
        world_manager=world_manager,
        character_name=character_core.name,
        existing_character=character_core
    )
    
    # Handle completion
    if completed_character:
        st.session_state.enhanced_character = completed_character
        st.session_state.import_flow_state = 'complete'
        st.rerun()


def render_completion_phase():
    """Phase 5: Import completion with character comparison"""
    
    original_core = st.session_state.get('character_core')
    enhanced_character = st.session_state.get('enhanced_character')
    
    # Use enhanced character if available, otherwise original
    final_character = enhanced_character or original_core
    
    if not final_character:
        st.error("❌ Character data not found. Starting over...")
        reset_import_flow()
        return
    
    st.markdown("### 🎉 Import Complete!")
    
    if enhanced_character:
        st.success(f"✨ **{final_character.name}** has been enhanced and is ready to save!")
        
        # Show before/after comparison
        st.markdown("### 📊 Enhancement Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📥 Original Import:**")
            st.metric("Description", f"{len(original_core.description)} chars")
            st.metric("Goals", len(original_core.goals))
            st.metric("Relationships", len(original_core.relationships))
            st.metric("Backstory", f"{len(original_core.backstory)} chars")
        
        with col2:
            st.markdown("**✨ After Enhancement:**")
            st.metric("Description", f"{len(enhanced_character.description)} chars", 
                     delta=len(enhanced_character.description) - len(original_core.description))
            st.metric("Goals", len(enhanced_character.goals),
                     delta=len(enhanced_character.goals) - len(original_core.goals))
            st.metric("Relationships", len(enhanced_character.relationships),
                     delta=len(enhanced_character.relationships) - len(original_core.relationships))
            st.metric("Backstory", f"{len(enhanced_character.backstory)} chars",
                     delta=len(enhanced_character.backstory) - len(original_core.backstory))
    else:
        st.success(f"✅ **{final_character.name}** is ready to save!")
    
    # Save character button
    char_manager = st.session_state.character_manager
    if st.button("💾 Save Character", type="primary", use_container_width=True):
        if char_manager.save_character(final_character):
            st.balloons()
            st.success(f"🎉 Character '{final_character.name}' saved successfully!")
            
            # Set as current character
            st.session_state.current_character_core = final_character
            st.session_state.selected_character = final_character.name
            
            # Clear import flow state
            reset_import_flow()
            
            # Show next steps
            st.markdown("### 🚀 What's Next?")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Edit Character", use_container_width=True):
                    st.switch_page("pages/character_management.py")
            
            with col2:
                if st.button("🎨 Generate Dataset", use_container_width=True):
                    st.switch_page("pages/dataset_studio.py")
            
            with col3:
                if st.button("📤 Import Another", use_container_width=True):
                    reset_import_flow()
                    st.rerun()
        else:
            st.error("❌ Failed to save character")
    
    # Additional options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Start Over", use_container_width=True):
            reset_import_flow()
            st.rerun()
    
    with col2:
        if enhanced_character and st.button("📤 Export Enhanced Card", use_container_width=True):
            # Export to SillyTavern format
            intelligence_service = st.session_state.character_intelligence
            try:
                export_card = asyncio.run(intelligence_service.export_for_sillytavern(enhanced_character))
                st.download_button(
                    label="📥 Download Enhanced SillyTavern Card",
                    data=json.dumps(export_card, indent=2),
                    file_name=f"{enhanced_character.name}_enhanced.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")


def reset_import_flow():
    """Reset the import flow state"""
    keys_to_clear = [
        'import_flow_state',
        'uploaded_card_data',
        'character_core', 
        'enhanced_character',
        'character_knowledge',
        'nsfw_assessment',
        'enhancement_suggestions',
        'analysis_complete',
        'enhance_existing_character'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.session_state.import_flow_state = 'upload'


def complete_import():
    """Complete import without enhancement"""
    character_core = st.session_state.get('character_core')
    if character_core:
        char_manager = st.session_state.character_manager
        if char_manager.save_character(character_core):
            st.balloons()
            st.success(f"🎉 Character '{character_core.name}' imported successfully!")
            
            # Set as current character
            st.session_state.current_character_core = character_core
            st.session_state.selected_character = character_core.name
            
            # Clear import flow
            reset_import_flow()
            
            # Immediate next steps
            st.markdown("### 🚀 Character Imported!")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Manage Character", type="primary", use_container_width=True):
                    st.switch_page("pages/character_management.py")
            
            with col2:
                if st.button("🎨 Generate Dataset", use_container_width=True):
                    st.switch_page("pages/dataset_studio.py")
        else:
            st.error("❌ Failed to save character")


def calculate_personality_variance(personality) -> float:
    """Calculate variance in personality traits"""
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


def calculate_character_complexity(character_core: CharacterCore) -> float:
    """Calculate overall character complexity score 0-1"""
    factors = []
    
    # Description richness
    factors.append(min(1.0, len(character_core.description) / 300))
    
    # Goal count
    factors.append(min(1.0, len(character_core.goals) / 3))
    
    # Relationship count
    factors.append(min(1.0, len(character_core.relationships) / 3))
    
    # Backstory richness
    factors.append(min(1.0, len(character_core.backstory) / 200))
    
    # Personality variance
    trait_variance = calculate_personality_variance(character_core.personality_traits)
    factors.append(min(1.0, trait_variance * 10))
    
    return sum(factors) / len(factors)


if __name__ == "__main__":
    page_character_upload() 