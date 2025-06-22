"""
📁 Character Upload Page

Streamlined character import flow:
1. Upload SillyTavern-compatible JSON character cards
2. Automatic conversion to CharacterCore format  
3. Immediate redirect to Character Management for editing
4. No manual conversion step needed

This creates a clean user journey from import → edit → dataset generation.
"""

import streamlit as st
import json
import asyncio
import os
from pathlib import Path


def page_character_upload():
    """Streamlined character upload page"""
    st.markdown('<h2 class="gradient-text">📁 Import Character Card</h2>', unsafe_allow_html=True)
    
    # Info about the streamlined process
    st.markdown("""
        <div class="custom-card">
            <h3 style="color: white; margin-top: 0;">🚀 Quick Character Import</h3>
            <p style="color: rgba(255,255,255,0.8);">
                Upload a SillyTavern JSON card and we'll automatically convert it to our enhanced CharacterCore format 
                with Big Five personality analysis, then take you straight to the Character Management Studio.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a character card file",
            type=['json'],
            help="Upload a .json character card file"
        )

        if uploaded_file is not None:
            try:
                # Load and validate the character data
                character_data = json.load(uploaded_file)
                
                # Basic validation
                char_manager = st.session_state.character_manager
                is_valid, error_msg = char_manager.validate_character_card(character_data)
                
                if not is_valid:
                    st.error(f"❌ Invalid character card: {error_msg}")
                    return
                
                # Show character preview
                st.markdown("### ✅ Character Card Loaded")
                
                char_name = character_data.get('name', 'Unknown')
                char_desc = character_data.get('description', 'No description')
                char_personality = character_data.get('personality', 'No personality info')
                
                # Quick preview
                with st.expander("📖 Character Preview", expanded=True):
                    st.markdown(f"**Name:** {char_name}")
                    st.markdown(f"**Description:** {char_desc[:200]}{'...' if len(char_desc) > 200 else ''}")
                    st.markdown(f"**Personality:** {char_personality[:200]}{'...' if len(char_personality) > 200 else ''}")
                    
                    if 'mes_example' in character_data and character_data['mes_example']:
                        example = character_data['mes_example']
                        st.markdown(f"**Example Dialogue:** {example[:150]}{'...' if len(example) > 150 else ''}")
                
                # Auto-conversion and redirect
                st.markdown("### 🔄 Converting to CharacterCore Format...")
                st.info("🧠 **Real LLM Analysis**: Using AI to analyze personality traits and extract structured information.")
                
                with st.spinner("Analyzing character and estimating personality traits using LLM..."):
                    try:
                        # Store the uploaded character data temporarily
                        st.session_state.uploaded_character_data = character_data
                        
                        # Check if OpenAI client is properly initialized
                        if not st.session_state.get('openai_client_initialized', False):
                            st.error("❌ OpenAI client not properly initialized. Please check your API configuration.")
                            return
                        
                        # Convert to CharacterCore format using real LLM analysis
                        char_core = asyncio.run(char_manager.import_sillytavern_card(character_data))
                        
                        # Save the character
                        if char_manager.save_character(char_core):
                            st.success(f"✅ Character '{char_name}' converted and saved!")
                            
                            # Set as current character for management
                            st.session_state.current_character_core = char_core
                            st.session_state.selected_character = char_name
                            
                            # Show what was created
                            st.markdown("**✨ Enhanced with:**")
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("• 🧠 Big Five personality traits")
                                st.markdown("• 📝 Structured appearance/backstory")
                                st.markdown("• 🎯 Extracted goals")
                            with col_b:
                                st.markdown("• 🤝 Identified relationships")
                                st.markdown("• 🏷️ Auto-generated tags")
                                st.markdown("• 🌍 World integration")
                            
                            # Automatic redirect to Character Management
                            st.markdown("**🚀 Redirecting to Character Management Studio...**")
                            
                            if st.button("📋 Continue to Character Management", type="primary", use_container_width=True):
                                # Use modern navigation
                                st.switch_page("pages/character_management.py")
                            
                            # Also provide auto-redirect with countdown
                            if 'redirect_countdown' not in st.session_state:
                                st.session_state.redirect_countdown = 5
                            
                            countdown_placeholder = st.empty()
                            
                            if st.session_state.redirect_countdown > 0:
                                countdown_placeholder.info(f"⏰ Auto-redirecting in {st.session_state.redirect_countdown} seconds... (Click above to go now)")
                                st.session_state.redirect_countdown -= 1
                                # Use modern navigation for auto-redirect
                                if st.session_state.redirect_countdown == 0:
                                    st.switch_page("pages/character_management.py")
                                # Use JavaScript to refresh every second
                                st.markdown("""
                                    <script>
                                    setTimeout(function() {
                                        window.location.reload();
                                    }, 1000);
                                    </script>
                                """, unsafe_allow_html=True)
                            else:
                                # Auto redirect
                                st.switch_page("pages/character_management.py")
                        else:
                            st.error("❌ Failed to save converted character")
                            
                    except Exception as e:
                        st.error(f"❌ Error during LLM-powered character conversion: {str(e)}")
                        
                        # Show debug information
                        import traceback
                        with st.expander("🔍 Debug Information"):
                            st.code(traceback.format_exc())
                            
                            # Check environment variables
                            st.write("**Environment Check:**")
                            st.write(f"- OPENAI_API_KEY set: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
                            st.write(f"- OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'Not set (using default)')}")
                            st.write(f"- OpenAI client initialized: {'✅' if st.session_state.get('openai_client_initialized') else '❌'}")
                        
                        st.error("The character conversion failed. Please check your OpenAI API configuration and try again.")
                        
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please upload a valid character card.")
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col2:
        # Help and tips
        st.markdown("""
            <div style="background: rgba(99, 102, 241, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
                <h4 style="color: #6366f1; margin-top: 0;">💡 What Happens Next</h4>
                <ul style="color: #cbd5e1; font-size: 0.9rem;">
                    <li><strong>Upload:</strong> Choose your JSON card</li>
                    <li><strong>Auto-Convert:</strong> We enhance it with Big Five traits</li>
                    <li><strong>Management:</strong> Edit in the Character Studio</li>
                    <li><strong>Dataset:</strong> Generate training data</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="background: rgba(34, 197, 94, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(34, 197, 94, 0.2); margin-top: 1rem;">
                <h4 style="color: #22c55e; margin-top: 0;">🎯 CharacterCore Benefits</h4>
                <ul style="color: #cbd5e1; font-size: 0.9rem;">
                    <li>Scientific personality modeling</li>
                    <li>Enhanced prompt generation</li>
                    <li>Better training data quality</li>
                    <li>Consistent character behavior</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Character format requirements
        st.markdown("""
            <div style="background: rgba(168, 85, 247, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(168, 85, 247, 0.2); margin-top: 1rem;">
                <h4 style="color: #a855f7; margin-top: 0;">📋 Required Fields</h4>
                <ul style="color: #cbd5e1; font-size: 0.9rem;">
                    <li><strong>name:</strong> Character name</li>
                    <li><strong>description:</strong> Character description</li>
                    <li><em>Optional:</em> personality, mes_example, scenario</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


# Simulation function removed - now using real LLM analysis via char_manager.import_sillytavern_card()


# Run the page function if this file is executed directly (for testing)
if __name__ == "__main__":
    page_character_upload() 