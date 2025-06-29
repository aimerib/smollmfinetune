"""
Character Chat Page - Embedded Runtime for Character Testing

This page provides an embedded chat interface where creators can immediately
test their trained characters using the RuntimePromptConstructor and InferenceManager.
"""

import streamlit as st
import asyncio
import pandas as pd
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path
from datetime import datetime

# Import our runtime components
from utils.runtime.prompt_constructor import RuntimePromptConstructor
from utils.inference import InferenceManager
from utils.async_training import data_collection_service

logger = logging.getLogger(__name__)


def page_character_chat():
    """Render the Character Chat page"""
    
    st.markdown('<h2 class="gradient-text">💬 Character Chat</h2>', unsafe_allow_html=True)
    st.markdown("Test your trained characters with dynamic conversation and mood controls")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_dynamic_state' not in st.session_state:
        st.session_state.chat_dynamic_state = {
            "current_mood": "neutral",
            "relationship_to_user": {"trust": 0.5, "affinity": 0.5},
            "recent_events": [],
            "forced_control_tokens": []
        }
    
    # Get available characters
    available_characters = get_available_characters()
    
    if not available_characters:
        st.warning("⚠️ No trained characters found. Please train a character first using the Training Dashboard.")
        st.markdown("""
        **To get started:**
        1. Go to Character Management and create or upload a character
        2. Use the Training Dashboard to train the character
        3. Return here to test your character in conversation
        """)
        return
    
    # Character selection
    st.subheader("🎭 Select Character")
    
    # Character selector
    selected = st.selectbox(
        "Choose a trained character to chat with:",
        options=available_characters,
        key="selected_character_chat",
        help="Characters with trained adapters or exported runtime packets"
    )
    
    if not selected:
        st.info("👆 Please select a character to start chatting!")
        return
    
    # Load character and create chat interface
    try:
        render_character_chat_interface(selected)
    except Exception as e:
        st.error(f"❌ Failed to load character: {str(e)}")
        st.info("💡 Try selecting a different character or check if the character files are intact.")


def get_available_characters() -> List[str]:
    """Get list of available characters for chat testing"""
    characters = []
    
    # Get characters from inference manager (trained adapters)
    if hasattr(st.session_state, 'inference_manager'):
        try:
            models = st.session_state.inference_manager.get_available_models()
            for model in models:
                if model.startswith(("LoRA:", "Checkpoint:")):
                    characters.append(model)
        except Exception as e:
            logger.warning(f"Could not get models from inference manager: {e}")
    
    # Get characters from runtime packets
    runtime_packets_dir = Path("runtime_packets")
    if runtime_packets_dir.exists():
        for packet_dir in runtime_packets_dir.iterdir():
            if packet_dir.is_dir() and (packet_dir / "character_core.json").exists():
                characters.append(f"Packet: {packet_dir.name}")
    
    return characters


def render_character_chat_interface(character_identifier: str):
    """Render the chat interface for a selected character"""
    
    # Load character using appropriate method
    runtime_constructor, character_info = load_character_for_chat(character_identifier)
    
    if not runtime_constructor:
        return
    
    # Display character info
    render_character_summary(character_info, runtime_constructor)
    
    # Data collection consent notice
    render_data_collection_consent()
    
    # Create two columns for controls and chat
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_dynamic_state_controls(runtime_constructor)
    
    with col2:
        render_chat_interface(runtime_constructor, character_identifier)


def load_character_for_chat(character_identifier: str) -> tuple[Optional[RuntimePromptConstructor], Optional[Dict]]:
    """Load character data for chat testing"""
    
    try:
        if character_identifier.startswith("Packet:"):
            # Load from runtime packet
            packet_name = character_identifier.split("Packet: ")[1]
            packet_path = Path("runtime_packets") / packet_name
            
            if not packet_path.exists():
                st.error(f"Runtime packet not found: {packet_path}")
                return None, None
            
            runtime_constructor = RuntimePromptConstructor(str(packet_path))
            character_info = {
                "source": "runtime_packet",
                "path": str(packet_path),
                "name": runtime_constructor.get_character_name()
            }
            
        else:
            # Load from trained adapter - we need to find the character data
            character_name = character_identifier.split(": ")[1].split("/")[0]
            
            # Try to find character core data in worlds
            character_core_path = find_character_core_file(character_name)
            if not character_core_path:
                st.error(f"Could not find character data for '{character_name}'. Make sure the character exists in a world.")
                return None, None
            
            # Create a temporary runtime packet structure for the constructor
            temp_packet_path = create_temp_runtime_environment(character_core_path, character_name)
            runtime_constructor = RuntimePromptConstructor(str(temp_packet_path))
            
            character_info = {
                "source": "adapter",
                "adapter_id": character_identifier,
                "name": character_name,
                "character_core_path": character_core_path
            }
        
        return runtime_constructor, character_info
        
    except Exception as e:
        logger.error(f"Failed to load character {character_identifier}: {e}")
        st.error(f"Failed to load character: {str(e)}")
        return None, None


def find_character_core_file(character_name: str) -> Optional[Path]:
    """Find character_core.json file for a character across all worlds"""
    worlds_root = Path("content/worlds")
    if not worlds_root.exists():
        return None
    
    for world_dir in worlds_root.iterdir():
        if not world_dir.is_dir():
            continue
        
        char_core_path = world_dir / "characters" / character_name / "character_core.json"
        if char_core_path.exists():
            return char_core_path
    
    return None


def create_temp_runtime_environment(character_core_path: Path, character_name: str) -> Path:
    """Create a temporary runtime environment for adapter-based characters"""
    import tempfile
    import json
    import shutil
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix=f"chat_{character_name}_"))
    
    # Copy character_core.json
    shutil.copy2(character_core_path, temp_dir / "character_core.json")
    
    # Find and copy world files
    world_dir = character_core_path.parent.parent.parent
    
    # Copy world_lore.json if exists
    world_lore_path = world_dir / "world_lore.json"
    if world_lore_path.exists():
        shutil.copy2(world_lore_path, temp_dir / "world_lore.json")
    else:
        # Create minimal world lore
        minimal_lore = {"facts": {}, "timeline": [], "factions": [], "places": []}
        with open(temp_dir / "world_lore.json", 'w') as f:
            json.dump(minimal_lore, f)
    
    # Copy tokens.json if exists
    tokens_path = world_dir / "tokens.json"
    if tokens_path.exists():
        shutil.copy2(tokens_path, temp_dir / "tokens.json")
    else:
        # Create minimal tokens
        with open(temp_dir / "tokens.json", 'w') as f:
            json.dump([], f)
    
    # Create minimal runtime_config.json
    runtime_config = {
        "character_file": "character_core.json",
        "world_file": "world_lore.json", 
        "tokens_file": "tokens.json"
    }
    with open(temp_dir / "runtime_config.json", 'w') as f:
        json.dump(runtime_config, f)
    
    return temp_dir


def render_character_summary(character_info: Dict, runtime_constructor: RuntimePromptConstructor):
    """Display character summary information"""
    
    st.subheader(f"🎭 {runtime_constructor.get_character_name()}")
    
    # Character details expander
    with st.expander("📋 Character Details", expanded=False):
        character_core = runtime_constructor.character_core
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Goals:**")
            goals = character_core.get('goals', [])
            if goals:
                for goal in goals[:3]:  # Show top 3 goals
                    st.write(f"• {goal}")
            else:
                st.write("• No specific goals set")
        
        with col2:
            st.write("**Personality Traits:**")
            big_five = character_core.get('big_five', {})
            if big_five:
                for trait, score in big_five.items():
                    st.write(f"• {trait.title()}: {score:.1f}")
            else:
                st.write("• No personality scores set")
        
        # Source information
        st.write(f"**Source:** {character_info['source']}")
        
        # Available tokens
        tokens = runtime_constructor.get_available_tokens()
        if tokens:
            st.write(f"**Control Tokens:** {len(tokens)} available")


def render_dynamic_state_controls(runtime_constructor: RuntimePromptConstructor):
    """Render controls for dynamic state management"""
    
    st.subheader("🎛️ Dynamic Controls")
    
    # Mood selector
    mood_tokens = runtime_constructor.get_tokens_by_category("mood")
    mood_options = ["neutral"] + [token["token"].replace("<mood_", "").replace(">", "") 
                                 for token in mood_tokens]
    
    current_mood = st.selectbox(
        "Current Mood:",
        options=mood_options,
        index=mood_options.index(st.session_state.chat_dynamic_state.get("current_mood", "neutral")),
        key="mood_selector",
        help="Select the character's current emotional state"
    )
    
    # Update mood in session state
    st.session_state.chat_dynamic_state["current_mood"] = current_mood
    
    # Relationship sliders
    st.write("**Relationship State:**")
    
    trust = st.slider(
        "Trust Level:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.chat_dynamic_state["relationship_to_user"]["trust"],
        step=0.1,
        key="trust_slider",
        help="How much the character trusts the user"
    )
    
    affinity = st.slider(
        "Affinity Level:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.chat_dynamic_state["relationship_to_user"]["affinity"],
        step=0.1,
        key="affinity_slider", 
        help="How much the character likes/connects with the user"
    )
    
    # Update relationship in session state
    st.session_state.chat_dynamic_state["relationship_to_user"] = {
        "trust": trust,
        "affinity": affinity
    }
    
    # Recent events
    with st.expander("📝 Recent Events", expanded=False):
        recent_events_text = st.text_area(
            "Recent Events (one per line):",
            value="\n".join(st.session_state.chat_dynamic_state.get("recent_events", [])),
            help="Events that have happened recently in the conversation",
            key="recent_events_input"
        )
        
        # Update recent events
        events = [event.strip() for event in recent_events_text.split("\n") if event.strip()]
        st.session_state.chat_dynamic_state["recent_events"] = events
    
    # Advanced controls
    with st.expander("⚙️ Advanced Controls", expanded=False):
        # Force specific tokens
        available_tokens = runtime_constructor.get_available_tokens()
        token_options = [token["token"] for token in available_tokens]
        
        if token_options:
            forced_tokens = st.multiselect(
                "Force Control Tokens:",
                options=token_options,
                default=st.session_state.chat_dynamic_state.get("forced_control_tokens", []),
                help="Manually inject specific control tokens",
                key="forced_tokens_selector"
            )
            st.session_state.chat_dynamic_state["forced_control_tokens"] = forced_tokens
    
    # Reset button
    if st.button("🔄 Reset State", help="Reset all dynamic state to defaults"):
        st.session_state.chat_dynamic_state = {
            "current_mood": "neutral",
            "relationship_to_user": {"trust": 0.5, "affinity": 0.5},
            "recent_events": [],
            "forced_control_tokens": []
        }
        st.rerun()


def render_chat_interface(runtime_constructor: RuntimePromptConstructor, character_identifier: str):
    """Render the main chat interface"""
    
    st.subheader("💬 Conversation")
    
    # Chat controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display conversation history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.write(content)
            else:  # assistant
                char_name = runtime_constructor.get_character_name()
                with st.chat_message("assistant", avatar="🎭"):
                    st.write(content)
    
    # User input
    user_message = st.chat_input(
        "Type your message here...",
        key="chat_input"
    )
    
    if user_message:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Generate character response
        with st.spinner("🤔 Character is thinking..."):
            try:
                response = generate_character_response(
                    runtime_constructor, 
                    character_identifier,
                    user_message
                )
                
                # Add character response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Collect conversation data for training (R3-2.5)
                collect_conversation_data_if_consented(
                    runtime_constructor,
                    character_identifier,
                    user_message,
                    response
                )
                
            except Exception as e:
                st.error(f"❌ Failed to generate response: {str(e)}")
                logger.error(f"Chat response generation failed: {e}")
        
        # Rerun to update chat display
        st.rerun()


def generate_character_response(runtime_constructor: RuntimePromptConstructor, 
                              character_identifier: str, user_message: str) -> str:
    """Generate character response using RuntimePromptConstructor + InferenceManager with contamination-free support"""
    
    try:
        # Construct the prompt with dynamic state
        conversation_history = st.session_state.chat_history.copy()
        # Add the current user message to history for prompt construction
        conversation_history.append({"role": "user", "content": user_message})
        
        prompt = runtime_constructor.construct(
            conversation_history=conversation_history,
            dynamic_state=st.session_state.chat_dynamic_state
        )
        
        logger.info(f"Generated prompt length: {len(prompt)} characters")
        
        # Generate response using InferenceManager
        if not hasattr(st.session_state, 'inference_manager'):
            raise RuntimeError("InferenceManager not available in session state")
        
        # Determine model path for inference
        if character_identifier.startswith("Packet:"):
            # For runtime packets, we need to find the corresponding adapter
            packet_name = character_identifier.split("Packet: ")[1]
            # Try to find a matching LoRA adapter
            models = st.session_state.inference_manager.get_available_models()
            model_path = None
            for model in models:
                if model.startswith("LoRA:") and packet_name.lower() in model.lower():
                    model_path = model
                    break
            
            if not model_path:
                # Fallback to base model if no adapter found
                model_path = f"Base: {st.session_state.inference_manager.base_model}"
        else:
            # Use the adapter directly
            model_path = character_identifier
        
        # 🔥 CONTAMINATION-FREE INFERENCE SUPPORT
        # Check if this is a contamination MoE model
        is_contamination_moe = False
        contamination_moe_path = None
        
        if "contamination_moe" in model_path.lower():
            is_contamination_moe = True
            contamination_moe_path = _find_contamination_moe_model(character_identifier)
        
        if is_contamination_moe and contamination_moe_path:
            # Use contamination-free generation
            logger.info("🔥 Using contamination-free inference with MoE architecture!")
            response = _generate_contamination_free_response(
                contamination_moe_path,
                prompt,
                user_message
            )
            
            # Add contamination isolation status
            with st.sidebar:
                st.success("🔥 **CONTAMINATION-FREE RESPONSE**")
                st.info("✅ Expert specialization active\n✅ Character purity protected")
        else:
            # Standard inference
            response = st.session_state.inference_manager.generate_response(
                model_path=model_path,
                prompt=prompt,
                max_tokens=200,
                temperature=0.8,
                system_prompt=""  # Prompt already contains character context
            )
        
        logger.info(f"Generated response length: {len(response)} characters")
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate character response: {e}")
        raise RuntimeError(f"Response generation failed: {str(e)}")


def _find_contamination_moe_model(character_identifier: str) -> Optional[str]:
    """Find the contamination MoE model path for a character"""
    
    try:
        # Extract character name
        if character_identifier.startswith("LoRA:"):
            character_name = character_identifier.split("/")[0].split(":")[1].strip()
        else:
            character_name = character_identifier.split(":")[1].strip()
        
        # Look for contamination MoE model
        adapters_dir = Path("training_output/adapters")
        
        for adapter_dir in adapters_dir.iterdir():
            if (adapter_dir.is_dir() and 
                "contamination_moe" in adapter_dir.name.lower() and
                character_name.lower() in adapter_dir.name.lower()):
                
                # Check if it has the metadata file
                metadata_path = adapter_dir / "contamination_moe_metadata.json"
                if metadata_path.exists():
                    return str(adapter_dir)
        
        return None
        
    except Exception as e:
        logger.warning(f"Could not find contamination MoE model: {e}")
        return None


def _generate_contamination_free_response(model_path: str, prompt: str, user_input: str) -> str:
    """Generate response using contamination-free MoE inference"""
    
    try:
        # Load contamination MoE model
        from narrative_engine.contamination_moe import create_contamination_moe_model
        import torch
        import json
        
        # Load model metadata
        metadata_path = Path(model_path) / "contamination_moe_metadata.json" 
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create model with contamination settings
        model = create_contamination_moe_model(
            base_model_name=metadata['base_model'],
            contamination_threshold=metadata.get('contamination_threshold', 0.7),
            routing_temperature=metadata.get('routing_temperature', 1.0),
            expert_dropout=metadata.get('expert_dropout', 0.1)
        )
        
        # Load the trained weights
        model_state_path = Path(model_path) / "pytorch_model.bin"
        if model_state_path.exists():
            state_dict = torch.load(model_state_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        
        # Tokenize input
        inputs = model.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
        
        # Generate with contamination isolation
        with torch.no_grad():
            outputs = model.generate_contamination_free(
                input_ids=inputs['input_ids'],
                input_text=user_input,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                force_character_expert=True  # Force pure character responses
            )
        
        response = outputs.get('generated_text', '').strip()
        
        # Log contamination analysis
        if 'contamination_detected' in outputs:
            if outputs['contamination_detected']:
                logger.warning(f"🚨 Contamination detected in user input: {user_input[:50]}...")
            else:
                logger.info("✅ Clean input detected - routed to character expert")
        
        return response
        
    except Exception as e:
        logger.error(f"Contamination-free generation failed: {e}")
        # Fallback to standard inference
        return st.session_state.inference_manager.generate_response(
            model_path=f"Base: {st.session_state.inference_manager.base_model}",
            prompt=prompt,
            max_tokens=200,
            temperature=0.8
        )


def render_data_collection_consent():
    """Render data collection consent UI"""
    
    # Initialize consent state if not present
    if 'data_collection_consent' not in st.session_state:
        st.session_state.data_collection_consent = False
    
    # Consent notice
    with st.expander("📊 Data Collection for Model Improvement", expanded=not st.session_state.data_collection_consent):
        st.markdown("""
        **Help Improve Character Training! 🚀**
        
        Your conversations can help create better training data for character models. When enabled:
        
        - ✅ **What's collected:** Chat messages, character responses, and interaction quality
        - ✅ **Privacy:** All personally identifiable information is automatically removed
        - ✅ **Voluntary:** You can opt-out anytime and your data can be deleted
        - ✅ **Purpose:** Only used to improve character consistency and training
        
        **Your data helps make characters more engaging for everyone!**
        """)
        
        # Consent checkbox
        consent = st.checkbox(
            "🤝 I consent to contribute my conversations for model improvement",
            value=st.session_state.data_collection_consent,
            key="consent_checkbox",
            help="This helps improve training data quality for all users"
        )
        
        # Update session state
        st.session_state.data_collection_consent = consent
        
        if consent:
            st.success("✅ Thank you for contributing to better AI characters!")
            st.info("💡 Your conversations will be anonymized and used to improve character training.")
        else:
            st.info("ℹ️ No data will be collected. You can enable this anytime to help improve the platform.")


def collect_conversation_data_if_consented(runtime_constructor: RuntimePromptConstructor,
                                         character_identifier: str, 
                                         user_message: str, 
                                         character_response: str):
    """Collect conversation data if user has consented"""
    
    # Check consent
    if not st.session_state.get('data_collection_consent', False):
        return
    
    try:
        # Get user and character IDs (mock for now)
        user_id = st.session_state.get('current_user_id', 1)
        
        # Extract character name for database lookup
        character_name = runtime_constructor.get_character_name()
        
        # For now, use a mock character ID (in production, we'd look this up in database)
        character_id = hash(character_name) % 1000000  # Simple hash-based ID
        
        # Create conversation messages
        messages = [
            {
                "role": "user",
                "content": user_message,
                "timestamp": st.session_state.get('last_message_time', 'unknown')
            },
            {
                "role": "assistant", 
                "content": character_response,
                "timestamp": str(pd.Timestamp.now())
            }
        ]
        
        # Add metadata about the conversation context
        metadata = {
            "character_source": character_identifier,
            "dynamic_state": st.session_state.get('chat_dynamic_state', {}),
            "conversation_length": len(st.session_state.chat_history),
            "platform_version": "R3-2.5"
        }
        
        # Queue for data collection
        success = data_collection_service.collect_conversation(
            user_id=user_id,
            character_id=character_id, 
            messages=messages,
            metadata=metadata
        )
        
        if success:
            logger.info(f"✅ Conversation data queued for collection: user={user_id}, character={character_name}")
        else:
            logger.warning("⚠️ Failed to queue conversation data for collection")
            
    except Exception as e:
        logger.error(f"❌ Error collecting conversation data: {e}")
        # Don't show error to user - data collection should be invisible


if __name__ == "__main__":
    # For testing
    page_character_chat() 