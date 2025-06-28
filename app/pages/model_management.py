"""
🔧 Model Management Page

This page allows users to manage their trained models:
1. Merge LoRA/DoRA with base models
2. View model overview and metadata
3. Fix legacy models and manage assets
4. Export models and manage disk usage

Extracted from main app.py for better maintainability.
"""

import streamlit as st
from pathlib import Path


def page_model_management():
    """Model management page for merging, managing trained models"""
    st.markdown('<h2 class="gradient-text">🔧 Model Management</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return
    
    # Get available models and character info
    available_models = st.session_state.inference_manager.get_available_models()
    # Get character name safely - handle both CharacterCore and dict
    current_character = st.session_state.get('current_character_core') or st.session_state.get('current_character')
    if hasattr(current_character, 'name'):
        character_name = current_character.name
    elif isinstance(current_character, dict):
        character_name = current_character.get("name", "Unknown")
    else:
        character_name = "Unknown"
    
    tab1, tab2, tab3 = st.tabs(["🔀 Model Merging", "📊 Model Overview", "🗃️ Model Assets"])
    
    with tab1:
        st.markdown("### Merge LoRA/DoRA with Base Model")
        st.info("💡 Merging creates a complete model that doesn't require PEFT adapters. This is recommended for DoRA models and final deployment.")
        
        # Filter to only show LoRA/DoRA models for the current character
        lora_models = [m for m in available_models if not m.startswith("Base:")]
        
        if not lora_models:
            st.info("ℹ️ No LoRA/DoRA models available for merging. Train a model first.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_model = st.selectbox("Select Model to Merge", lora_models)
                
                # Show model metadata
                if selected_model:
                    metadata = st.session_state.inference_manager.get_model_metadata(selected_model)
                    if metadata:
                        base_model = metadata.get('base_model', 'Unknown')
                        use_dora = metadata.get('use_dora', False)
                        use_rslora = metadata.get('use_rslora', False)
                        
                        method_display = "DoRA" if use_dora else "RSLoRA" if use_rslora else "LoRA"
                        
                        st.markdown(f"""
                        **Model Information:**
                        - **Method**: {method_display}
                        - **Base Model**: `{base_model}`
                        - **Training Date**: {metadata.get('training_date', 'Unknown')}
                        - **Dataset Size**: {metadata.get('dataset_size', 'Unknown')} samples
                        """)
                        
                        if use_dora:
                            st.info("💡 DoRA models benefit significantly from merging for optimal inference performance.")
                    else:
                        st.warning("⚠️ No metadata available for this model.")
                
                include_checkpoints = st.checkbox("Include checkpoint selection", help="Also show intermediate checkpoints for merging")
                
                if include_checkpoints:
                    checkpoints = st.session_state.training_manager.get_available_checkpoints(character_name)
                    if checkpoints:
                        checkpoint_option = st.selectbox("Or select checkpoint to merge", ["None (use final model)"] + checkpoints)
                        checkpoint_path = None if checkpoint_option == "None (use final model)" else checkpoint_option
                    else:
                        checkpoint_path = None
                        st.info("No checkpoints available")
                else:
                    checkpoint_path = None
            
            with col2:
                st.markdown("### Merge Benefits")
                st.markdown("""
                - **Faster Inference**: No PEFT overhead
                - **Easier Deployment**: Single model file
                - **DoRA Optimization**: Better performance for DoRA
                - **Portability**: Compatible with any transformers setup
                """)
            
            if st.button("🔀 Merge Model", use_container_width=True):
                with st.spinner("Merging model... This may take several minutes."):
                    try:
                        merged_path = st.session_state.training_manager.merge_and_export_model(
                            character_name, checkpoint_path
                        )
                        st.success(f"✅ Model merged successfully! Exported to: {merged_path}")
                        
                        # Show file size
                        file_size_mb = merged_path.stat().st_size / (1024 * 1024)
                        st.info(f"📦 Export size: {file_size_mb:.1f} MB")
                        
                    except Exception as e:
                        st.error(f"❌ Merge failed: {str(e)}")
    
    with tab2:
        st.markdown("### Model Overview")
        
        if not lora_models:
            st.info("ℹ️ No trained models available.")
        else:
            for model in lora_models:
                with st.expander(f"📊 {model}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Model metadata
                        metadata = st.session_state.inference_manager.get_model_metadata(model)
                        if metadata:
                            st.markdown("**Training Information:**")
                            st.write(f"- Base Model: `{metadata.get('base_model', 'Unknown')}`")
                            st.write(f"- Method: {metadata.get('training_method', 'Unknown').upper()}")
                            st.write(f"- Rank (r): {metadata.get('lora_r', 'Unknown')}")
                            st.write(f"- Alpha: {metadata.get('lora_alpha', 'Unknown')}")
                            st.write(f"- Dropout: {metadata.get('lora_dropout', 'Unknown')}")
                            st.write(f"- Total Steps: {metadata.get('total_steps', 'Unknown')}")
                            st.write(f"- Dataset Size: {metadata.get('dataset_size', 'Unknown')} samples")
                    
                    with col2:
                        # Training metrics
                        metrics = st.session_state.inference_manager.get_model_metrics(model)
                        if metrics:
                            st.markdown("**Training Metrics:**")
                            final_loss = metrics.get('current_loss', metrics.get('loss'))
                            if final_loss:
                                st.write(f"- Final Loss: {final_loss:.4f}")
                            
                            if 'character_consistency' in metrics:
                                consistency = metrics['character_consistency']
                                st.write(f"- Character Consistency: {consistency:.2f}")
                            
                            training_time = metrics.get('elapsed_time')
                            if training_time:
                                minutes = int(training_time // 60)
                                seconds = int(training_time % 60)
                                st.write(f"- Training Time: {minutes}m {seconds}s")
    
    with tab3:
        st.markdown("### Model Asset Management")
        
        # Fix legacy models section
        st.markdown("#### Fix Legacy Models")
        st.info("💡 If you have models trained before the metadata system, add compatibility info here.")
        
        legacy_models = [m for m in available_models if not m.startswith("Base:")]
        if legacy_models:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                legacy_model = st.selectbox("Select legacy model to fix", legacy_models, key="legacy_model_fix")
                
                # Check if it already has metadata
                if legacy_model:
                    metadata = st.session_state.inference_manager.get_model_metadata(legacy_model)
                    if metadata:
                        st.success("✅ This model already has metadata!")
                    else:
                        st.warning("⚠️ This model needs metadata to work properly.")
                        
                        base_model_options = [
                            "HuggingFaceTB/SmolLM2-135M-Instruct",
                            "HuggingFaceTB/SmolLM2-360M-Instruct", 
                            "HuggingFaceTB/SmolLM2-1.7B-Instruct"
                        ]
                        
                        base_model_fix = st.selectbox("Which base model was this trained on?", base_model_options, index=1)
                        method_fix = st.selectbox("Training method", ["lora", "dora", "rslora"])
            
            with col_b:
                if st.button("🔧 Add Metadata", use_container_width=True):
                    if legacy_model and base_model_fix:
                        success = st.session_state.training_manager.add_metadata_to_existing_model(
                            character_name, base_model_fix, method_fix
                        )
                        if success:
                            st.success("✅ Metadata added! Model should work now.")
                            st.rerun()
                        else:
                            st.error("❌ Failed to add metadata.")
        else:
            st.info("No models found that need fixing.")
        
        st.markdown("---")
        
        # ✅ NEW: Fix checkpoint metadata section
        st.markdown("#### Fix Missing Checkpoint Metadata")
        st.info("💡 Add missing training_metadata.json files to checkpoints for better comparison charts.")
        
        # Get all checkpoints without metadata
        checkpoints_without_metadata = []
        for model in available_models:
            if "Checkpoint:" in model and not model.startswith("Base:"):
                metadata = st.session_state.inference_manager.get_model_metadata(model)
                if not metadata:
                    checkpoints_without_metadata.append(model)
        
        if checkpoints_without_metadata:
            st.warning(f"Found {len(checkpoints_without_metadata)} checkpoints missing metadata")
            
            # Get base model from final model
            final_cricket_metadata = st.session_state.inference_manager.get_model_metadata("LoRA: cricket")
            if final_cricket_metadata:
                detected_base_model = final_cricket_metadata.get('base_model', 'HuggingFaceTB/SmolLM2-360M-Instruct')
                detected_method = final_cricket_metadata.get('training_method', 'dora')
                
                st.info(f"🔍 **Auto-detected from final model**: {detected_method.upper()} on `{detected_base_model}`")
                
                if st.button("🔧 Fix All Checkpoint Metadata", use_container_width=True):
                    success_count = 0
                    for checkpoint_model in checkpoints_without_metadata:
                        # Extract character name and checkpoint path
                        parts = checkpoint_model.split(": ")[1].split("/")
                        char_name = parts[0]
                        checkpoint_name = parts[1]
                        
                        # Add metadata to this specific checkpoint
                        success = st.session_state.training_manager.add_metadata_to_checkpoint(
                            char_name, checkpoint_name, detected_base_model, detected_method
                        )
                        if success:
                            success_count += 1
                    
                    if success_count > 0:
                        st.success(f"✅ Added metadata to {success_count} checkpoints!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add metadata to checkpoints.")
            else:
                st.warning("⚠️ Could not auto-detect settings from final model. Please add manually.")
        else:
            st.success("✅ All checkpoints have metadata!")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear All Training Assets", use_container_width=True):
                if st.session_state.training_manager.clear_training_assets(character_name):
                    st.success("Training assets cleared!")
                    st.rerun()
                else:
                    st.info("No training assets to remove.")
        
        with col2:
            if st.button("⬇️ Export LoRA", use_container_width=True):
                try:
                    zip_path = st.session_state.training_manager.export_lora(character_name)
                    st.success(f"LoRA exported to {zip_path}")
                except Exception as e:
                    st.error(str(e))
        
        with col3:
            if st.button("⬇️ Export Latest Checkpoint", use_container_width=True):
                zip_path = st.session_state.training_manager.export_latest_checkpoint(character_name)
                if zip_path:
                    st.success(f"Checkpoint exported to {zip_path}")
                else:
                    st.info("No checkpoints found to export.")
        
        # ✅ NEW: Runtime Packet Export
        st.markdown("---")
        st.markdown("### 🎮 Runtime Packet Export")
        st.info("💡 **Runtime Packets** are self-contained cartridges ready for deployment in game engines or runtime environments. They include everything needed to run your character!")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            **Runtime Packet Contents:**
            - 🧠 Trained adapter (RLHF preferred, or SFT)
            - 👤 Character core data (personality, goals, relationships)
            - 🌍 World lore and context
            - 🎛️ Control tokens for runtime control
            - ⚙️ Runtime configuration for deployment
            """)
        
        with col2:
            if st.button("🎮 Export Runtime Packet", use_container_width=True, type="primary"):
                try:
                    with st.spinner(f"Creating runtime packet for {character_name}..."):
                        packet_path = st.session_state.training_manager.export_runtime_packet(character_name)
                    
                    st.balloons()
                    st.success(f"🎉 Runtime packet created successfully!")
                    st.info(f"📦 **Location**: `{packet_path}`")
                    
                    # Show packet contents
                    packet_dir = Path(packet_path)
                    if packet_dir.exists():
                        files = [f.name for f in packet_dir.iterdir() if f.is_file()]
                        st.markdown("**Packet Contents:**")
                        for file in sorted(files):
                            st.markdown(f"- ✅ `{file}`")
                    
                    st.markdown("---")
                    st.markdown("**🚀 Next Steps:**")
                    st.markdown("- Copy the runtime packet to your game engine or runtime environment")
                    st.markdown("- Load the packet using the runtime configuration")
                    st.markdown("- Your character is ready to interact with players!")
                    
                except FileNotFoundError as e:
                    st.error(f"❌ Export failed: {str(e)}")
                    st.info("💡 Make sure you have trained a model for this character first.")
                except Exception as e:
                    st.error(f"❌ Export failed: {str(e)}")
        
        # Show disk usage
        st.markdown("### Disk Usage")
        
        adapter_dir = Path("training_output/adapters")
        exports_dir = Path("training_output/exports")
        
        def get_dir_size(path):
            if not path.exists():
                return 0
            total = 0
            for file in path.rglob('*'):
                if file.is_file():
                    total += file.stat().st_size
            return total / (1024 * 1024)  # MB
        
        adapters_size = get_dir_size(adapter_dir)
        exports_size = get_dir_size(exports_dir)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Assets", f"{adapters_size:.1f} MB")
        with col2:
            st.metric("Exports", f"{exports_size:.1f} MB")
        with col3:
            st.metric("Total", f"{adapters_size + exports_size:.1f} MB")


# Run the page function if this file is executed directly (for testing)
if __name__ == "__main__":
    page_model_management() 