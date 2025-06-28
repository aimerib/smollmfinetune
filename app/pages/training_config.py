"""
📄 Training Configuration Page

Enhanced training configuration page with advanced features including:
- Profile management (save/load configurations)
- Advanced training features (validation, tensorboard, wandb)
- Comprehensive hyperparameter tuning
- Model selection and PEFT configuration
- Real-time training analysis and recommendations

Extracted from main app.py for better maintainability.
"""

import streamlit as st
import json
import traceback
from pathlib import Path

def page_training_config():
    """Enhanced training configuration page with advanced features"""
    st.markdown('<h2 class="gradient-text">⚙️ Training Configuration</h2>', unsafe_allow_html=True)
    
    # Ensure required managers are available
    if 'training_manager' not in st.session_state:
        st.error("Training manager not initialized. Please restart the application.")
        return
    
    if 'inference_manager' not in st.session_state:
        st.error("Inference manager not initialized. Please restart the application.")
        return
    
    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return
    
    if not st.session_state.dataset_preview:
        st.warning("⚠️ Please generate a dataset first.")
        return
    
    finetune_method = st.session_state.get('finetune_method', 'lora')
    # Check if a profile needs to be applied
    if 'profile_to_apply' in st.session_state and st.session_state.profile_to_apply:
        profile = st.session_state.profile_to_apply
        
        # Store profile values in a persistent session state key (don't delete immediately)
        st.session_state.training_form_defaults = {
            **profile.get("advanced_training_config", {}),
            **profile.get("hyperparameters", {}),
            "base_model": profile.get("base_model")
        }
        
        # Clear the flag
        del st.session_state['profile_to_apply']
        st.toast("✅ Profile applied to configuration below!", icon="✨")

    # Get defaults from session state, or set to empty dict if not present
    defaults = st.session_state.get('training_form_defaults', {})
    # Note: Don't delete training_form_defaults here - keep it for next page load

    # If training is already running or paused, encourage user to switch to Dashboard
    if st.session_state.get('training_status') in ['training', 'paused']:
        st.info("🚧 Training is in progress. Please use the Training Dashboard to monitor or control the run.")
        return
    
    dataset_size = len(st.session_state.dataset_preview)
    
    # Advanced Features Configuration
    with st.expander("🚀 Advanced Training Features", expanded=False):
        st.markdown("### Core Improvements")
        
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            enable_validation = st.checkbox(
                "Enable Validation Split & Early Stopping",
                value=defaults.get("enable_validation", True),
                help="Split dataset for validation and enable early stopping to prevent overfitting"
            )
            
            adaptive_lora = st.checkbox(
                "Adaptive LoRA Parameters",
                value=defaults.get("adaptive_lora", False),
                help="Automatically adjust LoRA rank and alpha based on character complexity"
            )
            
            enhanced_filtering = st.checkbox(
                "Enhanced Quality Filtering",
                value=defaults.get("enhanced_quality_filtering", False),
                help="Apply character-specific quality filters to training data"
            )
        
        with col_adv2:
            enable_tensorboard = st.checkbox(
                "Enable TensorBoard Monitoring",
                value=defaults.get("enable_tensorboard", False),
                help="Enable detailed TensorBoard logging for training analysis"
            )
            
            enable_wandb = st.checkbox(
                "Weights & Biases Integration",
                value=defaults.get("enable_wandb", False),
                help="Log training to Wandb for advanced experiment tracking"
            )
        
        # Store advanced config in session state
        st.session_state.advanced_training_config = {
            'enable_validation': enable_validation,
            'adaptive_lora': adaptive_lora,
            'enhanced_quality_filtering': enhanced_filtering,
            'enable_tensorboard': enable_tensorboard,
            'enable_wandb': enable_wandb,
            'force_gpu': False # Force GPU usage is not supported yet
        }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Training Configuration")
        
        # Base model selection
        with st.expander("🤖 Base Model Selection", expanded=True):
            st.markdown("**Select the base model for LoRA training**")
            
            # Popular small models for LoRA
            base_model_options = [
                "HuggingFaceTB/SmolLM2-135M-Instruct",  # Default
                "HuggingFaceTB/SmolLM2-360M-Instruct",
                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "microsoft/Phi-3.5-mini-instruct",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "Custom (enter HF ID below)"
            ]
            
            # Get the index of the default base model
            try:
                default_model_index = base_model_options.index(defaults.get("base_model"))
            except ValueError:
                # If the model from profile is not in the standard list, select "Custom"
                default_model_index = base_model_options.index("Custom (enter HF ID below)")

            base_model_choice = st.selectbox(
                "Select base model",
                base_model_options,
                index=default_model_index,
                help="Smaller models train faster and work better for character LoRAs"
            )
            
            if base_model_choice == "Custom (enter HF ID below)":
                custom_base_model = st.text_input(
                    "HuggingFace Model ID",
                    value=defaults.get("base_model", ""), # Pre-fill custom model
                    placeholder="e.g., teknium/OpenHermes-2.5-Mistral-7B",
                    help="Enter any HuggingFace model ID compatible with PEFT/LoRA"
                )
                selected_base_model = custom_base_model if custom_base_model else base_model_options[0]
            else:
                selected_base_model = base_model_choice
            
            # Update the training manager's base model
            if st.session_state.training_manager.base_model != selected_base_model:
                st.session_state.training_manager.set_base_model(selected_base_model)
                st.session_state.inference_manager.set_base_model(selected_base_model)
                
            # Model size info
            model_size_info = {
                "HuggingFaceTB/SmolLM2-135M-Instruct": "135M params - Very fast, good for testing",
                "HuggingFaceTB/SmolLM2-360M-Instruct": "360M params - Fast, better quality",
                "HuggingFaceTB/SmolLM2-1.7B-Instruct": "1.7B params - Balanced speed/quality",
                "mistralai/Mistral-7B-Instruct-v0.2": "7B params - High quality, slower",
                "meta-llama/Llama-3.2-1B-Instruct": "1B params - Good balance",
                "meta-llama/Llama-3.2-3B-Instruct": "3B params - Better quality",
                "microsoft/Phi-3.5-mini-instruct": "3.8B params - Efficient & capable",
                "Qwen/Qwen2.5-0.5B-Instruct": "0.5B params - Very fast",
                "Qwen/Qwen2.5-1.5B-Instruct": "1.5B params - Good balance",
                "Qwen/Qwen2.5-3B-Instruct": "3B params - Better quality"
            }
            
            if selected_base_model in model_size_info:
                st.info(f"ℹ️ {model_size_info[selected_base_model]}")
            
            st.success(f"✅ Base model: {selected_base_model}")
        
        # RLHF Section - R1-11 Implementation
        st.markdown("### 🧠 Reinforcement Learning Fine-Tuning")
        
        # Check if character has sufficient preference data
        current_character = st.session_state.get('current_character_core') or st.session_state.get('current_character')
        char_name = "unknown"
        if hasattr(current_character, 'name'):
            char_name = current_character.name
        elif isinstance(current_character, dict):
            char_name = current_character.get("name", "unknown")
        
        # Check for preference data
        has_preferences = False
        preference_count = 0
        try:
            from utils.rlhf_trainer import has_sufficient_preferences
            has_preferences = st.session_state.training_manager.has_preference_data(char_name, min_preferences=10)
            
            # Count actual preferences for display
            from pathlib import Path
            pref_path = Path(f"content/worlds/Default World/characters/{char_name}/preference_logs.ndjson")
            if pref_path.exists():
                with open(pref_path, 'r') as f:
                    preference_count = sum(1 for line in f if line.strip())
        except Exception as e:
            st.debug(f"Error checking preferences: {e}")
        
        # RLHF UI based on preference availability
        with st.expander("🎯 Preference-Based Fine-Tuning (GRPO/PPO)", expanded=has_preferences):
            if has_preferences:
                st.success(f"✅ Found {preference_count} preference pairs - RLHF training available!")
                
                # Enable RLHF checkbox
                enable_rlhf = st.checkbox(
                    "Enable RL Fine-Tuning",
                    value=defaults.get("enable_rlhf", False),
                    help="Run GRPO or PPO training after SFT using collected preference data"
                )
                
                if enable_rlhf:
                    # Algorithm selection
                    rlhf_algorithm = st.selectbox(
                        "Algorithm",
                        ["GRPO", "PPO"],
                        index=0,  # GRPO default
                        help="GRPO is more sample-efficient and stable for preference data"
                    )
                    
                    # Store RLHF config
                    rlhf_config = {
                        "enable_rlhf": enable_rlhf,
                        "algorithm": rlhf_algorithm.lower(),
                        "preference_count": preference_count
                    }
                    
                    # Advanced RLHF settings
                    with st.expander("⚙️ Advanced RLHF Settings"):
                        col_rlhf1, col_rlhf2 = st.columns(2)
                        
                        with col_rlhf1:
                            rlhf_learning_rate = st.number_input(
                                "RLHF Learning Rate",
                                min_value=1e-6,
                                max_value=1e-4,
                                value=defaults.get("rlhf_learning_rate", 5e-6),
                                format="%.0e",
                                help="Conservative rate for RLHF to avoid divergence"
                            )
                            
                            rlhf_max_steps = st.slider(
                                "Max RLHF Steps",
                                min_value=100,
                                max_value=2000,
                                value=defaults.get("rlhf_max_steps", 500),
                                help="Number of RLHF training steps"
                            )
                        
                        with col_rlhf2:
                            if rlhf_algorithm == "GRPO":
                                beta_kl = st.slider(
                                    "Beta (KL Penalty)",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=defaults.get("rlhf_beta", 0.0),
                                    step=0.1,
                                    help="0.0 = no reference model (memory efficient)"
                                )
                                
                                num_generations = st.slider(
                                    "Number of Generations",
                                    min_value=2,
                                    max_value=12,
                                    value=defaults.get("rlhf_num_generations", 6),
                                    help="Generations for group-relative scoring"
                                )
                            else:  # PPO
                                beta_kl = st.slider(
                                    "KL Penalty",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=defaults.get("rlhf_kl_penalty", 0.1),
                                    step=0.01,
                                    help="KL divergence penalty for PPO"
                                )
                                
                                num_generations = 1  # PPO doesn't use multiple generations
                        
                        # Update RLHF config with advanced settings
                        rlhf_config.update({
                            "learning_rate": rlhf_learning_rate,
                            "max_steps": rlhf_max_steps,
                            "beta": beta_kl,
                            "num_generations": num_generations,
                        })
                    
                    st.info(f"💡 RLHF will run automatically after SFT training completes using {rlhf_algorithm}")
                else:
                    rlhf_config = {"enable_rlhf": False}
                    
            else:
                st.warning(f"⚠️ No preference data found ({preference_count} pairs available)")
                st.info("💡 To enable RLHF training, collect preference data by:")
                st.markdown("""
                - Using the Dataset Studio with multiple assistant options
                - Selecting preferred responses during generation
                - Building up preference pairs in your character directory
                """)
                rlhf_config = {"enable_rlhf": False}
            
            # Store RLHF config in session state
            st.session_state.rlhf_config = rlhf_config
        
        st.markdown("### Hyperparameter Configuration")
        
        # Select fine-tuning method outside the form to allow UI updates
        finetune_method = st.radio(
            "Fine-tuning Method",
            ("LoRA", "RSLoRA", "DoRA"),
            horizontal=True,
            index=["lora", "rslora", "dora"].index(defaults.get("finetune_method", "lora")),
            help="Choose between LoRA, RSLoRA, and DoRA. RSLoRA uses rank-stabilized scaling. DoRA offers more precise training."
        ).lower()
        
        # Store the selected method in session state
        st.session_state.finetune_method = finetune_method
        
        with st.form("training_config"):
            # Basic settings
            st.markdown("#### Basic Settings")
            
            # Dynamically pick a sensible default epoch count based on dataset size
            if dataset_size >= 200:
                optimal_epochs = 3  # Reduced for larger datasets
            else:
                optimal_epochs = min(6, max(3, 10000 // dataset_size))  # 3-6 epochs for smaller datasets
            
            # Epochs with explanation
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                st.markdown("**Epochs**")
            with c2:
                with st.popover("ℹ️", help="Explain Epochs"):
                    st.markdown("""
                    An **epoch** is one full pass through the entire training dataset.
                    
                    - **Too few epochs:** The model might not learn enough about the character (underfitting).
                    - **Too many epochs:** The model might memorize the training data and lose its ability to be creative (overfitting).
                    
                    **Recommendation:** 5-6 epochs for small datasets (<100 samples), and 3-4 for larger ones is a good starting point.
                    """)
            epochs = st.slider("Epochs", 1, 1000, defaults.get("epochs", optimal_epochs), label_visibility="collapsed", help="How many times the model sees the entire dataset.")

            # Learning Rate with explanation
            lr_options = [1e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1e-4, 2e-4, 3e-4, 5e-4]
            default_lr = defaults.get("learning_rate", 2e-4)
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                st.markdown("**Learning Rate**")
            with c2:
                with st.popover("ℹ️", help="Explain Learning Rate"):
                    st.markdown("""
                    The **Learning Rate** controls how much the model's parameters are adjusted during each training step.
                    
                    - **Too high:** The model might learn too fast and become unstable, with loss jumping around wildly.
                    - **Too low:** Training will be very slow, and the model might get stuck.
                    
                    **Recommendation:** `2e-4` is a safe and effective starting point for most characters.
                    """)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=lr_options,
                value=default_lr,
                format_func=lambda x: f"{x:.0e}",
                help="5e-5 to 5e-4 recommended for character LoRA training",
                label_visibility="collapsed"
            )

            # Batch Size with explanation
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                st.markdown("**Batch Size**")
            with c2:
                with st.popover("ℹ️", help="Explain Batch Size"):
                    st.markdown("""
                    The **Batch Size** is the number of training samples processed before the model's internal parameters are updated.
                    - It's limited by your GPU memory (VRAM).
                    - A larger batch size can lead to more stable training, but uses more memory.
                    - If you run out of memory, lower this value. You can compensate for a small batch size by increasing **Gradient Accumulation Steps**.
                    
                    **Recommendation:** Start with 2 or 4 and adjust based on your hardware.
                    """)
            batch_size = st.selectbox("Batch Size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], 
                                      index=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].index(defaults.get("batch_size", 2)), 
                                      label_visibility="collapsed")
            
            gradient_accumulation = st.selectbox("Gradient Accumulation Steps", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], 
                                                 index=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024].index(defaults.get("gradient_accumulation_steps", 2)))
            warmup_steps = st.slider("Warmup Steps", 0, 100, defaults.get("warmup_steps", 10), help="10-20 steps usually sufficient")
            max_grad_norm = st.slider("Max Gradient Norm", 0.5, 8.0, defaults.get("max_grad_norm", 1.0), step=0.1, help="1.0 is standard")
            
            # Sample selection for dataset
            st.markdown("**Dataset Sampling**")
            if dataset_size > 0:
                use_all_samples = st.checkbox("Use All Samples", value=not ("max_samples" in defaults and defaults["max_samples"] != dataset_size), help="Use the entire dataset for training")
                if not use_all_samples:
                    max_samples = st.slider(
                        "Number of Samples", 
                        min_value=1, 
                        max_value=dataset_size, 
                        value=defaults.get("max_samples", min(dataset_size, 100)),
                        help=f"Select subset from {dataset_size} total samples (randomized selection)"
                    )
                else:
                    max_samples = dataset_size
            else:
                max_samples = dataset_size
            
            # LoRA settings optimized for character training
            st.markdown("#### PEFT Configuration (Character-Optimized)")
            
            # Store the selected method in the form
            st.text(f"Selected method: {finetune_method.upper()}")
            
            default_r = defaults.get("lora_r", 16)  # Optimal for character LoRA per research
            # LoRA Rank with explanation
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                st.markdown("**Rank (r)**")
            with c2:
                with st.popover("ℹ️", help="Explain Rank (r)"):
                    st.markdown("""
                    The **Rank (r)** determines the number of trainable parameters in the PEFT adapter. It controls the 'capacity' of the adapter.
                    - **Higher Rank:** More parameters, allowing the model to learn more complex details. This also increases training time and VRAM usage.
                    - **Lower Rank:** Fewer parameters, faster training, less VRAM.
                    
                    **Recommendation:** `8` or `16` is highly effective for most characters. Use `32` for very complex characters with large datasets.
                    """)
            lora_r = st.slider("Rank (r)", 4, 256, default_r, step=4,
                               label_visibility="collapsed",
                               help="8-16 optimal for character LoRAs. Higher rank = more capacity but slower.")

            # LoRA Alpha with explanation
            c1, c2 = st.columns([0.9, 0.1])
            with c1:
                st.markdown("**Alpha**")
            with c2:
                with st.popover("ℹ️", help="Explain Alpha"):
                    st.markdown("""
                    **Alpha** is a scaling factor for the PEFT adjustments. Think of it as controlling the 'intensity' of the training.
                    - By setting **Alpha equal to Rank (α = r)**, you are using a standard configuration that works very well for character training. This helps balance the learning process.
                    - Deviating from this (e.g., alpha = 2 * rank) is an advanced technique and not typically recommended for characters.
                    
                    **Recommendation:** Keep this value the same as your Rank.
                    """)
            lora_alpha = st.slider("Alpha", 8, 1024, defaults.get("lora_alpha", default_r), step=8,
                                   label_visibility="collapsed",
                                   help="Set equal to rank (α = r) for character training")
            
            lora_dropout = st.slider("Dropout", 0.0, 0.2, defaults.get("lora_dropout", 0.1), step=0.01, 
                                     help="0.05-0.1 for regularization")
            target_modules = st.multiselect(
                "Target Modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                default=defaults.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),  # Focus on attention layers
                help="Attention layers (q,k,v,o) are most important for character behavior"
            )
            
            # Method-specific options
            if finetune_method == "dora":
                st.markdown("#### DoRA Settings")
                ephemeral_gpu_offload = st.checkbox(
                    "Enable Ephemeral GPU Offload",
                    value=defaults.get("ephemeral_gpu_offload", False),
                    help="Speed up DoRA training with temporary VRAM overhead (CUDA only)"
                )
                st.info("💡 DoRA works best with low dropout (0.0-0.05) and is optimized for eval mode")
            else:
                ephemeral_gpu_offload = False
            # --------------------------------------------------------------
            # Resume-from-checkpoint selection
            # --------------------------------------------------------------
            # Get character name safely - handle both CharacterCore and dict
            current_character = st.session_state.get('current_character_core') or st.session_state.get('current_character')
            if hasattr(current_character, 'name'):
                char_name = current_character.name
            elif isinstance(current_character, dict):
                char_name = current_character.get("name", "unknown")
            else:
                char_name = "unknown"
            
            available_ckpts = st.session_state.training_manager.get_available_checkpoints(char_name)

            if available_ckpts:
                resume_ckpt_option = st.selectbox(
                    "Resume from checkpoint (optional)",
                    ["None"] + available_ckpts
                )
                resume_ckpt = None if resume_ckpt_option == "None" else resume_ckpt_option
            else:
                resume_ckpt = None
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                # Check if dataset has system prompts
                dataset_has_system = False
                if st.session_state.dataset_preview and len(st.session_state.dataset_preview) > 0:
                    first_sample = st.session_state.dataset_preview[0]
                    if 'messages' in first_sample and len(first_sample['messages']) > 0:
                        dataset_has_system = first_sample['messages'][0].get('role') == 'system'
                
                include_system_prompts = st.checkbox(
                    "Include System Prompts in Training",
                    value=defaults.get("include_system_prompts", False),
                    help="If checked, system prompts will be included in training data. Usually better to let LoRA internalize character behavior without system prompts.",
                    disabled=not dataset_has_system
                )
                
                if dataset_has_system and not include_system_prompts:
                    st.info("💡 System prompts will be removed during training (recommended for character LoRAs)")
                elif dataset_has_system and include_system_prompts:
                    st.warning("⚠️ Including system prompts in training - make sure this is intentional")
                elif not dataset_has_system:
                    st.info("ℹ️ Dataset has no system prompts")
                
                fp16 = st.checkbox("Enable FP16", value=defaults.get("fp16", True), help="Enables mixed precision training for better performance")
                save_steps = st.slider("Save Every N Steps", 1, 200, defaults.get("save_steps", 50), step=1)
                logging_steps = st.slider("Log Every N Steps", 1, 100, defaults.get("logging_steps", 5), step=1, help="Controls how often training metrics are logged and displayed")
                eval_steps = st.slider("Evaluation Steps", 1, 100, defaults.get("eval_steps", 10), step=1)
                early_stopping_patience = st.slider(
                    "Early Stopping Patience",
                    min_value=1,
                    max_value=10,
                    value=defaults.get("early_stopping_patience", 3),
                    help="Number of evaluation steps without improvement before stopping"
                ) if enable_validation else 3
                max_steps_override = st.number_input(
                    "Override Total Training Steps (0 = auto)",
                    min_value=0,
                    max_value=50000,
                    value=defaults.get("max_steps_override", 0),
                    step=100,
                    help="Manually set the total number of optimisation steps if you need finer control. Leave at 0 to use the computed value."
                )
            
            # Form buttons
            col_form1, col_form2 = st.columns(2)
            with col_form1:
                start_training = st.form_submit_button("🚀 Start Training", use_container_width=True)
            with col_form2:
                save_profile = st.form_submit_button("💾 Save Profile", use_container_width=True, help="Save current settings as a profile")
    
    with col2:
        st.markdown("### Training Recommendations")
        
        # Profile I/O section
        st.markdown("#### Profile Management")
        st.info("💡 Use the '💾 Save Profile' button in the training form below to save current settings.")
        
        # Profile controls
        profile_cols = st.columns(2)
        with profile_cols[0]:
            uploaded_profile = st.file_uploader(
                "Load Profile", 
                type=['json'], 
                help="Upload a saved training profile"
            )
        with profile_cols[1]:
            if st.button("🔄 Reset to Defaults", use_container_width=True, help="Clear any applied profile and reset to default values"):
                if 'training_form_defaults' in st.session_state:
                    del st.session_state['training_form_defaults']
                st.success("✅ Reset to default values!")

        if uploaded_profile:
            try:
                profile_data = json.load(uploaded_profile)
                
                # Store loaded profile in session state to be applied
                st.session_state.loaded_profile = profile_data
                
                st.success(f"📂 Profile '{uploaded_profile.name}' loaded!")
                st.info("Click 'Apply Profile' to update the configuration below.")

            except Exception as e:
                st.error(f"❌ Error loading profile: {e}")
        
        # Button to apply the loaded profile
        if 'loaded_profile' in st.session_state and st.session_state.loaded_profile:
            if st.button("✨ Apply Profile", use_container_width=True):
                # Store profile to be applied on next render
                st.session_state.profile_to_apply = st.session_state.loaded_profile
                del st.session_state['loaded_profile'] # Clear after flagging
                st.rerun() # Rerun to apply the settings

        # Calculate training recommendations based on selected samples
        effective_dataset_size = max_samples if 'max_samples' in locals() else dataset_size
        total_steps = (effective_dataset_size * epochs) // (batch_size * gradient_accumulation)
        effective_batch_size = batch_size * gradient_accumulation
        
        # More nuanced overfitting risk calculation
        if effective_dataset_size < 50:
            if total_steps > 300:
                overfitting_risk = "Very High"
            elif total_steps > 200:
                overfitting_risk = "High"
            else:
                overfitting_risk = "Medium"
        elif effective_dataset_size < 100:
            if total_steps > 500:
                overfitting_risk = "High"
            elif total_steps > 300:
                overfitting_risk = "Medium"
            else:
                overfitting_risk = "Low"
        else:  # effective_dataset_size >= 100
            if total_steps > 1000:
                overfitting_risk = "Medium"
            else:
                overfitting_risk = "Low"
        
        # Display recommendations
        # Show both total and selected dataset info
        dataset_info = f"{effective_dataset_size} samples"
        if effective_dataset_size != dataset_size:
            dataset_info += f" (from {dataset_size} total)"
        
        st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin: 0 0 1rem 0;">📊 Training Analysis</h4>
                <p><strong>Training Dataset:</strong> {dataset_info}</p>
                <p><strong>Total Steps:</strong> {total_steps}</p>
                <p><strong>Effective Batch Size:</strong> {effective_batch_size}</p>
                <p><strong>Overfitting Risk:</strong> <span style="color: {'#dc2626' if overfitting_risk == 'Very High' else '#ef4444' if overfitting_risk == 'High' else '#f59e0b' if overfitting_risk == 'Medium' else '#10b981'}">{overfitting_risk}</span></p>
                <p><strong>Est. Time:</strong> ~{max(1, total_steps * 2 // 60)} minutes</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Warnings based on configuration
        if overfitting_risk in ["High", "Very High"]:
            st.warning(f"⚠️ {overfitting_risk} overfitting risk! Consider: reducing epochs to {max(1, epochs-2)}, increasing dataset size, or lowering learning rate.")
        
        if effective_dataset_size > 300:
            st.info("💡 Large dataset detected. Consider using rank 32 for more model capacity.")
        
        if learning_rate >= 5e-4:
            st.warning("⚠️ High learning rate may cause training instability. Consider 2e-4 for safer training.")
            
        if epochs > 8:
            st.warning("⚠️ High epoch count increases overfitting risk. 5-6 epochs is usually sufficient.")
        
        # Tips
        st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #10b981;">
                <h4 style="color: #10b981; margin-top: 0;">💡 Character LoRA Best Practices</h4>
                <ul style="color: #cbd5e1; font-size: 0.9rem;">
                    <li><strong>Dataset:</strong> 50-100 samples optimal, 200-300 max</li>
                    <li><strong>Learning Rate:</strong> Start with 2e-4, use 1e-4 if unstable</li>
                    <li><strong>LoRA Rank:</strong> 8-16 for most characters, 32 for complex ones</li>
                    <li><strong>LoRA Alpha:</strong> Set equal to rank (α = r)</li>
                    <li><strong>Epochs:</strong> 5-6 for small datasets, 3-4 for larger ones</li>
                    <li><strong>Monitor:</strong> Stop if loss plateaus or increases</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Handle profile saving
    if 'save_profile' in locals() and save_profile:
        try:
            # Get character name safely - handle both CharacterCore and dict
            current_character = st.session_state.get('current_character_core') or st.session_state.get('current_character')
            if hasattr(current_character, 'name'):
                char_name = current_character.name
            elif isinstance(current_character, dict):
                char_name = current_character.get("name", "untitled")
            else:
                char_name = "untitled"
            
            config_to_save = {
                "character_name": char_name,
                "base_model": selected_base_model,
                "dataset_file": st.session_state.dataset_metadata.get('path'),
                "advanced_training_config": st.session_state.get('advanced_training_config', {}),
                "hyperparameters": {
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'gradient_accumulation_steps': gradient_accumulation,
                    'warmup_steps': warmup_steps,
                    'max_grad_norm': max_grad_norm,
                    'max_samples': max_samples,
                    'finetune_method': finetune_method,
                    'lora_r': lora_r,
                    'lora_alpha': lora_alpha,
                    'lora_dropout': lora_dropout,
                    'target_modules': target_modules,
                    'include_system_prompts': include_system_prompts,
                    'fp16': fp16,
                    'save_steps': save_steps,
                    'early_stopping_patience': early_stopping_patience,
                    'logging_steps': logging_steps,
                    'eval_steps': eval_steps,
                    'max_steps_override': int(max_steps_override) if max_steps_override else 0,
                    'ephemeral_gpu_offload': ephemeral_gpu_offload,
                }
            }
            
            # Create profiles directory if it doesn't exist
            profiles_dir = Path("profiles")
            profiles_dir.mkdir(exist_ok=True)
            
            # Save the profile
            save_path = profiles_dir / f"{char_name}_profile.json"
            with open(save_path, 'w') as f:
                json.dump(config_to_save, f, indent=4)
            
            st.success(f"✅ Profile saved: {save_path.name}")
            
        except Exception as e:
            st.error(f"❌ Error saving profile: {e}")
    
    if start_training:
        # Enhanced training config with advanced features
        config = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation,
            'warmup_steps': warmup_steps,
            'max_grad_norm': max_grad_norm,
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'target_modules': target_modules,
            'fp16': fp16,
            'save_steps': save_steps,
            'logging_steps': logging_steps,
            'eval_steps': eval_steps,
            'max_steps_override': int(max_steps_override) if max_steps_override else 0,
            'resume_from_checkpoint': resume_ckpt,
            'include_system_prompts': include_system_prompts,
            'max_samples': max_samples,
            'finetune_method': finetune_method,
            # Method-specific parameters
            'use_rslora': finetune_method == 'rslora',
            'use_dora': finetune_method == 'dora',
            'ephemeral_gpu_offload': ephemeral_gpu_offload if finetune_method == 'dora' else False,
            # Enhanced scheduling options
            'lr_scheduler_type': 'cosine',
            'warmup_ratio': 0.05
        }
        
        # Add RLHF configuration if available
        rlhf_config = st.session_state.get('rlhf_config', {})
        if rlhf_config.get('enable_rlhf', False):
            config.update(rlhf_config)
            st.info(f"🧠 RLHF training will run after SFT using {rlhf_config.get('algorithm', 'grpo').upper()}")
        
        try:
            # Configure advanced features
            advanced_config = st.session_state.get('advanced_training_config', {})
            st.session_state.training_manager.configure_advanced_features(advanced_config)
            
            # Initialize with force_gpu if specified
            if advanced_config.get('force_gpu', False):
                from utils.training import TrainingManager
                st.session_state.training_manager = TrainingManager(
                    base_model=st.session_state.training_manager.base_model,
                    force_gpu=True
                )
                st.session_state.training_manager.configure_advanced_features(advanced_config)
            
            # Use different key to avoid widget conflict
            st.session_state.active_training_config = config
            
            # Show configuration summary
            st.success("✅ Training configuration complete!")
            
            # ✅ NEW: Display actual configuration values being used
            with st.expander("🔍 **Actual Training Configuration Used**", expanded=True):
                st.info("💡 **Tip:** This shows the exact values the training process will use, which may differ from UI defaults due to fallbacks or advanced settings.")
                
                config_col1, config_col2, config_col3 = st.columns(3)
                
                with config_col1:
                    st.markdown("##### 📊 **Core Training**")
                    # Show the actual logging frequency that will be used
                    actual_log_freq = config.get('logging_steps', advanced_config.get('logging_steps', 10))
                    st.write(f"**Log Every N Steps:** `{actual_log_freq}`")
                    st.write(f"**Learning Rate:** `{config.get('learning_rate', 2e-4)}`")
                    st.write(f"**Batch Size:** `{config.get('batch_size', 2)}`")
                    st.write(f"**Gradient Accumulation:** `{config.get('gradient_accumulation_steps', 2)}`")
                    st.write(f"**Max Steps Override:** `{config.get('max_steps_override', 'None (use calculated)')}`")
                    
                with config_col2:
                    st.markdown("##### ⚙️ **Method & Parameters**")
                    st.write(f"**Method:** `{config.get('finetune_method', 'lora').upper()}`")
                    st.write(f"**LoRA Rank (r):** `{config.get('lora_r', 16)}`")
                    st.write(f"**LoRA Alpha:** `{config.get('lora_alpha', config.get('lora_r', 16))}`")
                    st.write(f"**LoRA Dropout:** `{config.get('lora_dropout', 0.1)}`")
                    st.write(f"**Use RSLoRA:** `{config.get('use_rslora', False)}`")
                    st.write(f"**Use DoRA:** `{config.get('use_dora', False)}`")
                    
                with config_col3:
                    st.markdown("##### 🎯 **Advanced Settings**")
                    st.write(f"**Save Every N Steps:** `{config.get('save_steps', 50)}`")
                    st.write(f"**Max Samples:** `{config.get('max_samples', 'All')}`")
                    st.write(f"**Include System Prompts:** `{config.get('include_system_prompts', False)}`")
                    st.write(f"**FP16:** `{config.get('fp16', False)}`")
                    
                # Show any discrepancies as warnings
                ui_log_steps = config.get('logging_steps')
                advanced_log_steps = advanced_config.get('logging_steps', 10)
                if ui_log_steps and ui_log_steps != actual_log_freq:
                    st.warning(f"⚠️ **Logging Frequency Discrepancy:** UI shows `{ui_log_steps}` but training will use `{actual_log_freq}` (from advanced config)")
            
            # Display what features are enabled
            enabled_features = []
            if advanced_config.get('enable_validation', True):
                enabled_features.append("🔍 Validation & Early Stopping")
            if advanced_config.get('adaptive_lora', False):
                enabled_features.append("🎯 Adaptive LoRA Parameters")
            if advanced_config.get('enhanced_quality_filtering', False):
                enabled_features.append("✨ Enhanced Quality Filtering")
            if advanced_config.get('enable_tensorboard', False):
                enabled_features.append("📊 TensorBoard Monitoring")
            if advanced_config.get('enable_wandb', False):
                enabled_features.append("🌐 Wandb Integration")
            
            if enabled_features:
                st.info("🚀 **Enhanced Features Active:**\n\n" + "\n".join([f"• {feature}" for feature in enabled_features]))
            
            # Start enhanced training
            st.session_state.training_manager.start_training(
                st.session_state.current_character,
                st.session_state.dataset_preview,
                config
            )
            
            # Update status
            st.session_state.training_status = 'training'
            
            st.success("🚀 Enhanced training started! Switch to the Training Dashboard to monitor progress.")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to start training: {str(e)}")
            st.session_state.training_status = 'error'
            st.error(f"Debug info: {traceback.format_exc()}")

# Run the page function if this file is executed directly (for testing)
if __name__ == "__main__":
    page_training_config() 