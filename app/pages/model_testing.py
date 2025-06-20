"""
🧪 Model Testing Page

This page allows users to test their trained models with various configurations:
1. Select and test trained models
2. Configure system prompts and generation settings
3. Run quick tests and see responses
4. Debug and analyze model outputs

Extracted from main app.py for better maintainability.
"""

import streamlit as st
import asyncio
from typing import Dict, Any


def page_model_testing():
    """Model testing and inference page"""
    st.markdown('<h2 class="gradient-text">🧪 Model Testing</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Test Your Trained Model")
        
        st.info("🧪 **Pure LoRA Testing**: No character context is injected. Testing how well the LoRA learned character behavior during training.")
        
        
        # Model selection
        available_models = st.session_state.inference_manager.get_available_models()
        
        if not available_models:
            st.info("ℹ️ No trained models available. Complete training first.")
            return
        
        selected_model = st.selectbox("Select Trained Model", available_models, key="test_model_select")
        
        # Show model compatibility info
        if not selected_model.startswith("Base:"):
            metadata = st.session_state.inference_manager.get_model_metadata(selected_model)
            if metadata:
                required_base_model = metadata.get('base_model')
                training_method = metadata.get('training_method', 'lora')
                use_dora = metadata.get('use_dora', False)
                use_rslora = metadata.get('use_rslora', False)
                
                # Display model info
                method_display = "DoRA" if use_dora else "RSLoRA" if use_rslora else "LoRA"
                
                st.info(f"🔍 **Model Info**: {method_display} trained on `{required_base_model}`")
                
                # Check compatibility
                current_base = st.session_state.inference_manager.base_model
                if required_base_model != current_base:
                    st.warning(f"⚠️ Model trained on `{required_base_model}` but current inference base is `{current_base}`. Will auto-switch for compatibility.")
            else:
                st.warning("⚠️ No metadata available for this model. It may be from an older training run.")
        
        # Check if we have dataset metadata with system prompt info
        dataset_metadata = st.session_state.get('dataset_metadata', {})
        system_prompt_config = dataset_metadata.get('system_prompt_config', {})
        
        # Set default option based on dataset
        if system_prompt_config.get('type') == 'custom':
            default_option = "Dataset System Prompt"
            options = ["Dataset System Prompt", "Default (Tokenizer's built-in)", "Empty (No system prompt)", "Roleplay Director", "Custom"]
        elif system_prompt_config.get('type') == 'none':
            default_option = "Empty (No system prompt)"
            options = ["Empty (No system prompt)", "Default (Tokenizer's built-in)", "Roleplay Director", "Custom"]
        else:
            default_option = "Default (Tokenizer's built-in)"
            options = ["Default (Tokenizer's built-in)", "Empty (No system prompt)", "Roleplay Director", "Custom"]

        system_prompt_option = st.radio(
            "Choose system prompt strategy:",
            options,
            help="Test how the LoRA responds to different system prompts"
        )
        
        # Show debug info about what's being tested
        with st.expander("🔧 Test Configuration"):
            st.write(f"**Selected Model:** {selected_model}")
            st.write(f"**System Prompt Strategy:** {system_prompt_option}")
            st.write(f"**Character Context Injection:** No (Pure LoRA Test)")
            if st.session_state.current_character and not selected_model.startswith("Base:"):
                char = st.session_state.current_character
                st.write(f"**Testing Character:** {char.get('name', 'Unknown')}")
                st.info("💡 Testing how well the LoRA learned the character behavior")
            elif selected_model.startswith("Base:"):
                st.write("**Mode:** Base model testing")
            else:
                st.warning("⚠️ No character uploaded for LoRA comparison")
        
        # System prompt selection
        system_prompt = None
        if system_prompt_option == "Dataset System Prompt":
            system_prompt = system_prompt_config.get('prompt', '')
            if system_prompt:
                st.info(f"📊 Using system prompt from dataset generation")
            else:
                st.info(f"📊 Dataset was generated with no system prompt")
        elif system_prompt_option == "Empty (No system prompt)":
            system_prompt = ""
            st.info("🧪 Testing pure LoRA behavior without any system guidance")
        elif system_prompt_option == "Roleplay Director":
            system_prompt = "You are a scene director playing the role of a character in a never ending chat"
            st.info("🎭 Testing LoRA with roleplay-oriented system prompt")
        elif system_prompt_option == "Custom":
            system_prompt = st.text_area(
                "Enter custom system prompt:",
                placeholder="You are...",
                height=80
            )
            st.info("✏️ Testing LoRA with your custom system prompt")
        else:
            st.info("🤖 Using tokenizer's default system prompt (SmolLM assistant)")
        
        if system_prompt is not None and system_prompt_option not in ["Default (Tokenizer's built-in)", "Custom"]:
            st.code(f"System: {system_prompt if system_prompt else '[No system prompt]'}", language="text")
        elif system_prompt_option == "Custom" and system_prompt:
            st.code(f"System: {system_prompt}", language="text")

        # Test prompt
        test_prompt = st.text_area(
            "Enter your test prompt:",
            value=st.session_state.get('quick_test_prompt', ''),
            placeholder="Tell me about yourself...",
            height=100,
            key="main_test_prompt"
        )
        
        # Generation settings
        with st.expander("⚙️ Generation Settings"):
            # Import and use sampling configuration
            from utils.sampling_config import render_sampling_config_ui, SamplingConfig, get_model_preset
            
            # Try to get current model name for testing
            test_model = selected_model
            if test_model.startswith("Base:"):
                # For base models, try to get from inference manager
                test_model = getattr(st.session_state.inference_manager, 'base_model', None)
            
            # Check if we have a model-specific preset
            model_preset = get_model_preset(test_model) if test_model else None
            if model_preset:
                st.info(f"🎯 **{model_preset['name']}** preset available for this model")
            
            # Create testing-specific default config
            default_test_config = SamplingConfig(
                temperature=0.8,
                top_p=0.9,
                max_tokens=150,
                repetition_penalty=1.1,
            )
            
            # Render compact sampling configuration
            test_sampling_config = render_sampling_config_ui(
                current_config=default_test_config,
                model_name=test_model,
                key_prefix="model_test"
            )
        
        if st.button("🚀 Generate Response", use_container_width=True):
            if test_prompt.strip():
                with st.spinner("Generating response..."):
                    try:
                        # Build messages for OpenAI API
                        messages = []
                        if system_prompt_option != "Default (Tokenizer's built-in)" and system_prompt is not None:
                            if system_prompt:  # Only add if not empty
                                messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": test_prompt})
                        
                        # Use OpenAI API for testing
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            response = loop.run_until_complete(
                                st.session_state.dataset_manager.client.chat_complete(
                                    messages=messages,
                                    max_tokens=test_sampling_config.max_tokens,
                                    temperature=test_sampling_config.temperature,
                                    top_p=test_sampling_config.top_p
                                )
                            )
                        finally:
                            loop.close()
                    except Exception as e:
                        st.error(f"❌ Generation failed: {e}")
                        response = f"Error: {e}"
                
                st.markdown("### Response")
                st.markdown(f"""
                    <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #6366f1;">
                        <p style="margin: 0; color: #f8fafc;">{response}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show recent logs for debugging
                with st.expander("🔍 Debug Logs (Last 20 lines)"):
                    try:
                        with open('app.log', 'r') as f:
                            lines = f.readlines()
                            recent_logs = ''.join(lines[-20:])
                            st.code(recent_logs, language='text')
                    except FileNotFoundError:
                        st.info("No log file found yet.")
            else:
                st.warning("⚠️ Please enter a test prompt.")
    
    with col2:
        st.markdown("### System Prompt Guide")
        
        st.markdown("""
        **🧪 Empty System Prompt**
        - Tests pure LoRA learned behavior
        - No guidance from system prompt
        - Best for seeing raw character adaptation
        
        **🎭 Roleplay Director**
        - Encourages character roleplay
        - Tests how LoRA responds to roleplay cues
        - Good for interactive character testing
        
        **✏️ Custom System Prompt**
        - Test specific scenarios
        - Control system behavior precisely
        - Useful for targeted evaluation
        
        **🤖 Default (SmolLM)**
        - Uses built-in assistant prompt
        - May conflict with character training
        - Good for comparison baseline
        """)
        
        st.markdown("### Quick Tests")
        
        quick_tests = [
            "Who are you?",
            "What drives you in life?",
            "Describe your greatest fear.", 
            "Tell me about your past.",
            "What's your personality like?",
            "How do you speak to others?"
        ]
        
        for i, prompt in enumerate(quick_tests):
            if st.button(f"🎯 {prompt}", key=f"quick_test_{i}"):
                # Auto-fill the test prompt
                st.session_state.quick_test_prompt = prompt
                st.rerun()
        
        # Model comparison
        st.markdown("### Model Comparison")
        
        if len(available_models) > 1:
            st.info("Go to the '⚔️ Model Comparison' page from the sidebar to compare models side-by-side.")
        else:
            st.info("Train multiple checkpoints to enable model comparison.")


# Run the page function if this file is executed directly (for testing)
if __name__ == "__main__":
    page_model_testing() 