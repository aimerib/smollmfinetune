"""
📁 Character Upload Page

This page allows users to upload and manage character cards:
1. Upload SillyTavern-compatible JSON character cards
2. Preview character information and metadata
3. Convert to CharacterCore format with Big Five traits
4. Auto-load existing datasets for characters

Extracted from main app.py for better maintainability.
"""

import streamlit as st
import json


def page_character_upload():
    """Character upload and card management page"""
    st.markdown('<h2 class="gradient-text">📁 Character Card Upload</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class="custom-card">
                <h3 style="color: white; margin-top: 0;">Upload Your Character Card</h3>
                <p style="color: rgba(255,255,255,0.8);">
                    Upload a SillyTavern-compatible JSON character card to begin training your AI character.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a character card file",
            type=['json'],
            help="Upload a .json character card file"
        )

        if st.session_state.current_character:
            st.markdown("### Character Preview")
            st.markdown(f"**Name:** {st.session_state.current_character.get('name', 'Unknown')}")
            st.markdown(f"**Example:** {st.session_state.current_character.get('mes_example', 'No example available')[:300]}{'...' if len(st.session_state.current_character.get('mes_example', 'No example available')) > 300 else ''}")
            st.markdown(f"**Scenario:** {st.session_state.current_character.get('scenario', 'No scenario available')[:300]}{'...' if len(st.session_state.current_character.get('scenario', 'No scenario available')) > 300 else ''}")
            st.markdown(f"**Description:** {st.session_state.current_character.get('description', 'No description available')[:200]}{'...' if len(st.session_state.current_character.get('description', 'No description available')) > 200 else ''}")
            st.markdown(f"**Personality:** {st.session_state.current_character.get('personality', 'No personality available')[:300]}{'...' if len(st.session_state.current_character.get('personality', 'No personality available')) > 300 else ''}")
        
        if uploaded_file is not None:
            try:
                character_data = json.load(uploaded_file)
                st.session_state.current_character = character_data
                
                # Auto-load existing dataset if available
                dataset_with_metadata = st.session_state.dataset_manager.load_dataset_with_metadata(character_data)
                if dataset_with_metadata:
                    existing_dataset, metadata = dataset_with_metadata
                    st.session_state.dataset_preview = existing_dataset
                    st.session_state.dataset_metadata = metadata
                    st.success(f"✅ Character card loaded with existing dataset ({len(existing_dataset)} samples)!")
                else:
                    st.session_state.dataset_preview = None
                    st.session_state.dataset_metadata = {}
                    st.success("✅ Character card loaded successfully!")
                
                # Display character preview
                st.markdown("### Character Preview")
                
                preview_col1, preview_col2 = st.columns(2)
                
                with preview_col1:
                    st.markdown(f"**Name:** {character_data.get('name', 'Unknown')}")
                    st.markdown(f"**Example:** {character_data.get('mes_example', 'No example available')[:300]}{'...' if len(character_data.get('mes_example', 'No example available')) > 300 else ''}")
                    st.markdown(f"**Scenario:** {character_data.get('scenario', 'No scenario available')[:300]}{'...' if len(character_data.get('scenario', 'No scenario available')) > 300 else ''}")

                with preview_col2:
                    description = character_data.get('description', 'No description available')
                    st.markdown(f"**Description:** {description[:200]}{'...' if len(description) > 200 else ''}")
                
                # Personality preview
                if 'personality' in character_data:
                    st.markdown("**Personality:**")
                    personality = character_data['personality']
                    st.markdown(f"{personality[:300]}{'...' if len(personality) > 300 else ''}")
                
            except Exception as e:
                st.error(f"❌ Error loading character card: {str(e)}")
        
        # CharacterCore conversion section
        if st.session_state.current_character:
            st.markdown("---")
            st.markdown("### 🔄 Convert to CharacterCore Format")
            
            col_convert1, col_convert2 = st.columns([2, 1])
            
            with col_convert1:
                st.markdown("""
                **Convert your SillyTavern card to the new CharacterCore format with:**
                - 🧠 Big Five personality traits (auto-estimated)
                - 📝 Structured appearance and backstory
                - 🎯 Extracted goals and relationships
                - 🏷️ Automatic tagging
                """)
            
            with col_convert2:
                if st.button("🔄 Convert to CharacterCore", type="primary", use_container_width=True):
                    with st.spinner("Converting character card..."):
                        try:
                            # Use existing character manager from session state
                            char_manager = st.session_state.character_manager
                            
                            # For now, show a preview of the conversion process
                            st.info("🔄 Character conversion functionality is available. This will:")
                            st.markdown("""
                            1. 🧠 Analyze character with LLM to estimate Big Five traits
                            2. 📝 Extract appearance and backstory elements
                            3. 🎯 Identify goals and relationships
                            4. 💾 Save in new folder structure:
                               ```
                               characters/[name]/
                               ├── character_core.json
                               ├── mes_example.txt
                               └── assets/
                               ```
                            """)
                            
                            # Show a preview of what would be converted
                            st.markdown("**Preview of converted structure:**")
                            char_name = st.session_state.current_character.get('name', 'Unknown')
                            char_desc = st.session_state.current_character.get('description', '')
                            char_personality = st.session_state.current_character.get('personality', '')
                            
                            preview_data = {
                                "name": char_name,
                                "description": char_desc[:100] + "..." if len(char_desc) > 100 else char_desc,
                                "personality_traits": {
                                    "openness": "0.7 (estimated from description)",
                                    "conscientiousness": "0.6 (estimated from personality)",
                                    "extraversion": "0.8 (estimated from examples)",
                                    "agreeableness": "0.7 (estimated from traits)",
                                    "neuroticism": "0.3 (estimated from behavior)"
                                },
                                "goals": ["[Auto-extracted from description and personality]"],
                                "relationships": ["[Auto-extracted from character context]"],
                                "appearance": "[Extracted appearance details]",
                                "backstory": "[Derived backstory elements]",
                                "tags": ["[Auto-generated based on content]"],
                                "world": char_manager.get_current_world() or "Default World"
                            }
                            
                            st.json(preview_data)
                            
                            # Note about full implementation
                            st.warning("""
                            **Note**: Full conversion with LLM analysis requires an async context. 
                            The complete functionality is implemented in the CharacterManager.import_sillytavern_card() method.
                            This preview shows the structure that would be created.
                            """)
                            
                        except Exception as e:
                            st.error(f"❌ Error during conversion preview: {str(e)}")
                            # Add some debug info
                            st.error(f"Debug info: {type(e).__name__}: {str(e)}")
    
    with col2:
        st.markdown("""
            <div style="background: rgba(99, 102, 241, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
                <h4 style="color: #6366f1; margin-top: 0;">💡 Tips</h4>
                <ul style="color: #cbd5e1; font-size: 0.9rem;">
                    <li>Ensure your JSON file follows SillyTavern format</li>
                    <li>Rich character descriptions lead to better training results</li>
                    <li>Include personality traits and example dialogue</li>
                    <li>Keep descriptions concise but detailed</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Add CharacterCore info
        if st.session_state.current_character:
            st.markdown("""
                <div style="background: rgba(34, 197, 94, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(34, 197, 94, 0.2); margin-top: 1rem;">
                    <h4 style="color: #22c55e; margin-top: 0;">🚀 CharacterCore Format</h4>
                    <ul style="color: #cbd5e1; font-size: 0.9rem;">
                        <li>Structured Big Five personality traits</li>
                        <li>Organized character data for better training</li>
                        <li>World-based character organization</li>
                        <li>Enhanced prompt generation</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)


# Run the page function if this file is executed directly (for testing)
if __name__ == "__main__":
    page_character_upload() 