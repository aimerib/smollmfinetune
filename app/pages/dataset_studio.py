"""
🎨 Dataset Studio Page

Unified dataset management interface that combines:
- Interactive dataset generation (primary workflow)  
- Dataset review and editing (simplified from explorer)
- Experimental generation methods (fast, slow, factual Q&A)

This replaces both page_dataset_preview and page_dataset_explorer_v2 with a cleaner,
more focused interface designed around the primary interactive generation workflow.

Extracted from main app.py for better maintainability.
"""

import streamlit as st
import asyncio
import json

def page_dataset_studio():
    """Unified dataset management interface"""
    st.markdown('<h2 class="gradient-text">🎨 Dataset Studio</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return
    
    # Get character name safely - handle both CharacterCore and dict
    current_character = st.session_state.get('current_character_core') or st.session_state.get('current_character')
    if hasattr(current_character, 'name'):
        char_name = current_character.name
    elif isinstance(current_character, dict):
        char_name = current_character.get("name", "Unknown")
    else:
        char_name = "Unknown"
    st.markdown(f"### Creating dataset for: **{char_name}**")
    
    # Check for existing dataset
    dataset_info = st.session_state.dataset_manager.get_dataset_info(st.session_state.current_character)
    
    # Main tabs - simplified structure
    interactive_tab, review_tab, experimental_tab = st.tabs([
        "🤝 Interactive Generation",
        "📋 Dataset Review & Edit", 
        "🧪 Experimental Methods"
    ])
    
    # ============================================================================
    # TAB 1: INTERACTIVE GENERATION (Primary workflow)
    # ============================================================================
    with interactive_tab:
        st.markdown("### 🤝 Interactive Dataset Generation")
        
        st.info("""
        **Primary Workflow**: Generate datasets through collaborative batch-by-batch creation
        - ✨ Best for: Perfect quality control, learning what works for your character
        - 🎯 Interactive feedback loop with AI learning from your preferences
        - ⏱️ User-guided (as fast or slow as you want)
        """)
        
        # Show existing dataset overview if available
        if dataset_info['exists']:
            st.markdown("#### 📊 Current Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Existing Samples", dataset_info['sample_count'])
            with col2:
                if st.button("📂 Load Existing", use_container_width=True):
                    if st.session_state.get('_generating_dataset', False):
                        st.warning("⚠️ Dataset generation in progress. Please wait.")
                    else:
                        dataset_with_metadata = st.session_state.dataset_manager.load_dataset_with_metadata(st.session_state.current_character)
                        if dataset_with_metadata:
                            existing_dataset, metadata = dataset_with_metadata
                            st.session_state.dataset_preview = existing_dataset
                            st.session_state.dataset_metadata = metadata
                            st.success(f"✅ Loaded {len(existing_dataset)} existing samples!")
                            st.rerun()
            with col3:
                if st.button("🗑️ Reset Dataset", use_container_width=True):
                    if st.session_state.get('_generating_dataset', False):
                        st.warning("⚠️ Dataset generation in progress. Please wait.")
                    else:
                        if st.session_state.dataset_manager.delete_dataset(st.session_state.current_character):
                            st.session_state.dataset_preview = None
                            st.session_state.dataset_metadata = {}
                            st.success("✅ Dataset reset! Generate a new one below.")
                            st.rerun()
            with col4:
                # Export current dataset
                if st.session_state.dataset_preview:
                    json_data = json.dumps(st.session_state.dataset_preview, indent=2)
                    st.download_button(
                        label="⬇️ Export Dataset",
                        data=json_data,
                        file_name=f"{char_name}_dataset.json",
                        mime="application/json",
                        use_container_width=True
                    )
            
            st.markdown("---")
        
        # Interactive generation state management
        interactive_key = f"interactive_state_{char_name}"
        if interactive_key not in st.session_state:
            st.session_state[interactive_key] = {
                'current_batch': [],
                'approved_samples': [],
                'rejected_samples': [],
                'feedback_tags': {},
                'generation_round': 0,
                'target_total': 80,
                'batch_size': 20,
                'is_generating': False,
                'few_shot_examples': [],
                'negative_patterns': []
            }
        
        interactive_state = st.session_state[interactive_key]
        
        # Progress overview
        current_total = len(interactive_state['approved_samples'])
        target_total = interactive_state['target_total']
        progress_pct = min(100, (current_total / target_total) * 100) if target_total > 0 else 0
        
        col_prog1, col_prog2, col_prog3, col_prog4 = st.columns(4)
        
        with col_prog1:
            st.metric("Approved Samples", current_total)
        with col_prog2:
            st.metric("Target Total", target_total)
        with col_prog3:
            st.metric("Progress", f"{progress_pct:.1f}%")
        with col_prog4:
            st.metric("Generation Round", interactive_state['generation_round'])
        
        # Progress bar
        st.progress(progress_pct / 100, text=f"Dataset Progress: {current_total}/{target_total} samples")
        
        # Configuration (only show if not started)
        if interactive_state['generation_round'] == 0:
            st.markdown("#### ⚙️ Interactive Generation Settings")
            
            col_cfg1, col_cfg2 = st.columns(2)
            
            with col_cfg1:
                target_total = st.slider(
                    "Target Dataset Size",
                    min_value=20,
                    max_value=500,
                    value=interactive_state['target_total'],
                    step=20,
                    help="Total number of approved samples you want to end up with"
                )
                interactive_state['target_total'] = target_total
                
                batch_size = st.slider(
                    "Batch Size",
                    min_value=10,
                    max_value=30,
                    value=interactive_state['batch_size'],
                    step=5,
                    help="Number of samples to generate and review at once"
                )
                interactive_state['batch_size'] = batch_size
            
            with col_cfg2:
                # Import sampling config
                from utils.sampling_config import render_sampling_config_ui, SamplingConfig
                
                interactive_default_config = SamplingConfig(
                    temperature=0.9,
                    top_p=0.95,
                    max_tokens=300,
                    repetition_penalty=1.05,
                )
                
                interactive_sampling_config = render_sampling_config_ui(
                    current_config=interactive_default_config,
                    key_prefix="interactive_gen",
                    use_expander=False
                )
        
        # Current batch review interface
        if interactive_state['current_batch']:
            st.markdown("---")
            st.markdown("#### 📝 Review Current Batch")
            st.info(f"Rate each sample below. Your feedback will improve the next generation batch.")
            
            # Batch review interface
            for i, sample in enumerate(interactive_state['current_batch']):
                sample_key = f"sample_{interactive_state['generation_round']}_{i}"
                
                st.markdown(f"##### Sample {i+1}")
                
                # Display the conversation
                messages = sample['messages']
                for msg in messages:
                    if msg['role'] == 'system' and msg['content']:
                        st.markdown(f"**🔧 System:** {msg['content'][:100]}...")
                    elif msg['role'] == 'user':
                        st.markdown(f"**👤 User:** {msg['content']}")
                    elif msg['role'] == 'assistant':
                        st.markdown(f"**🎭 Assistant:** {msg['content']}")
                
                # Rating interface
                col_rate1, col_rate2, col_rate3 = st.columns([1, 1, 2])
                
                with col_rate1:
                    if st.button("👍 Good", key=f"approve_{sample_key}", use_container_width=True):
                        interactive_state['approved_samples'].append(sample)
                        interactive_state['few_shot_examples'].append({
                            'user': messages[1]['content'],
                            'assistant': messages[2]['content']
                        })
                        # Keep only best 10 few-shot examples
                        if len(interactive_state['few_shot_examples']) > 10:
                            interactive_state['few_shot_examples'] = interactive_state['few_shot_examples'][-10:]
                        
                        # Remove from current batch
                        interactive_state['current_batch'] = [s for j, s in enumerate(interactive_state['current_batch']) if j != i]
                        st.rerun()
                
                with col_rate2:
                    if st.button("👎 Bad", key=f"reject_{sample_key}", use_container_width=True):
                        interactive_state['rejected_samples'].append(sample)
                        # Add to negative patterns
                        response_text = messages[2]['content']
                        interactive_state['negative_patterns'].append(response_text[:200])
                        
                        # Remove from current batch
                        interactive_state['current_batch'] = [s for j, s in enumerate(interactive_state['current_batch']) if j != i]
                        st.rerun()
                
                with col_rate3:
                    # Custom feedback
                    custom_feedback = st.text_input(
                        "Custom feedback (optional)",
                        key=f"feedback_{sample_key}",
                        placeholder="What's wrong with this sample?",
                        label_visibility="collapsed"
                    )
                    
                    if custom_feedback:
                        interactive_state['feedback_tags'][sample_key] = interactive_state['feedback_tags'].get(sample_key, []) + [f"Custom: {custom_feedback}"]
                
                st.markdown("---")
            
            # Batch actions
            col_batch1, col_batch2, col_batch3 = st.columns(3)
            
            with col_batch1:
                if st.button("✅ Approve All Remaining", use_container_width=True):
                    for sample in interactive_state['current_batch']:
                        interactive_state['approved_samples'].append(sample)
                        messages = sample['messages']
                        interactive_state['few_shot_examples'].append({
                            'user': messages[1]['content'],
                            'assistant': messages[2]['content']
                        })
                    interactive_state['current_batch'] = []
                    st.rerun()
            
            with col_batch2:
                if st.button("❌ Reject All Remaining", use_container_width=True):
                    for sample in interactive_state['current_batch']:
                        interactive_state['rejected_samples'].append(sample)
                    interactive_state['current_batch'] = []
                    st.rerun()
            
            with col_batch3:
                if st.button("🔄 Regenerate Batch", use_container_width=True):
                    interactive_state['current_batch'] = []
                    interactive_state['is_generating'] = True
                    st.rerun()
        
        # Generation controls
        st.markdown("---")
        st.markdown("#### 🎮 Generation Controls")
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        
        # Check if we need more samples
        remaining_needed = max(0, interactive_state['target_total'] - len(interactive_state['approved_samples']))
        
        with col_ctrl1:
            # Generate next batch button
            disabled = interactive_state['is_generating'] or (remaining_needed == 0)
            button_text = "🚀 Start Interactive Generation" if interactive_state['generation_round'] == 0 else f"➡️ Generate Next Batch ({min(interactive_state['batch_size'], remaining_needed)} samples)"
            
            if st.button(button_text, disabled=disabled or bool(interactive_state['current_batch']), use_container_width=True):
                interactive_state['is_generating'] = True
                st.rerun()
        
        with col_ctrl2:
            # Auto-complete button (only show if we have some approved samples)
            if len(interactive_state['approved_samples']) >= 20:
                remaining = interactive_state['target_total'] - len(interactive_state['approved_samples'])
                if remaining > 0 and st.button(f"🤖 Auto-Complete ({remaining} samples)", use_container_width=True):
                    st.session_state[f"{interactive_key}_auto_complete"] = True
                    st.rerun()
        
        with col_ctrl3:
            # Finish early button
            if len(interactive_state['approved_samples']) > 0:
                if st.button("🏁 Finish with Current Samples", use_container_width=True):
                    # Save the approved samples as the dataset
                    st.session_state.dataset_preview = interactive_state['approved_samples']
                    st.session_state.dataset_metadata = {
                        'generation_method': 'interactive',
                        'interactive_rounds': interactive_state['generation_round'],
                        'system_prompt_config': {'type': 'temporal'}
                    }
                    st.success(f"✅ Interactive generation complete! Saved {len(interactive_state['approved_samples'])} samples.")
                    # Reset interactive state
                    del st.session_state[interactive_key]
                    st.rerun()
        
        # Handle generation
        if interactive_state['is_generating'] and not interactive_state['current_batch']:
            st.session_state._generating_dataset = True
            
            with st.spinner(f"Generating batch {interactive_state['generation_round'] + 1}..."):
                try:
                    samples_to_generate = min(interactive_state['batch_size'], remaining_needed)
                    
                    # Prepare generation parameters with feedback
                    if 'interactive_sampling_config' in locals():
                        sampling_kwargs = interactive_sampling_config.to_dict()
                    else:
                        sampling_kwargs = {}
                    generation_params = {
                        'num_samples': samples_to_generate,
                        'progress_callback': lambda p: None,
                        'append_to_existing': False,
                        'extra_quality': True,
                        'few_shot_examples': interactive_state['few_shot_examples'][-5:],
                        'negative_patterns': interactive_state['negative_patterns'][-10:],
                        **sampling_kwargs
                    }
                    
                    # Generate the batch
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        batch = loop.run_until_complete(
                            st.session_state.dataset_manager.generate_interactive_batch(
                                st.session_state.current_character,
                                **generation_params
                            )
                        )
                    finally:
                        loop.close()
                    
                    # Update state
                    interactive_state['current_batch'] = batch
                    interactive_state['generation_round'] += 1
                    interactive_state['is_generating'] = False
                    
                    st.session_state._generating_dataset = False
                    
                    st.success(f"✅ Generated batch {interactive_state['generation_round']} with {len(batch)} samples!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating batch: {str(e)}")
                    interactive_state['is_generating'] = False
                    st.session_state._generating_dataset = False
        
        # Handle auto-completion
        auto_complete_key = f"{interactive_key}_auto_complete"
        if st.session_state.get(auto_complete_key, False):
            remaining = interactive_state['target_total'] - len(interactive_state['approved_samples'])
            
            with st.spinner(f"Auto-completing remaining {remaining} samples..."):
                try:
                    # Use the enhanced generation with all accumulated feedback
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        remaining_samples = loop.run_until_complete(
                            st.session_state.dataset_manager.generate_dataset(
                                st.session_state.current_character,
                                num_samples=remaining,
                                progress_callback=lambda p: None,
                                append_to_existing=False,
                                extra_quality=True,
                                few_shot_examples=interactive_state['few_shot_examples'],
                                negative_patterns=interactive_state['negative_patterns']
                            )
                        )
                    finally:
                        loop.close()
                    
                    # Combine with approved samples
                    final_dataset = interactive_state['approved_samples'] + remaining_samples
                    
                    # Save the complete dataset
                    st.session_state.dataset_preview = final_dataset
                    st.session_state.dataset_metadata = {
                        'generation_method': 'interactive_auto_complete',
                        'interactive_rounds': interactive_state['generation_round'],
                        'human_approved_samples': len(interactive_state['approved_samples']),
                        'auto_generated_samples': len(remaining_samples),
                        'system_prompt_config': {'type': 'temporal'}
                    }
                    
                    st.success(f"🎉 Interactive generation complete! Final dataset: {len(final_dataset)} samples ({len(interactive_state['approved_samples'])} human-approved + {len(remaining_samples)} auto-generated)")
                    
                    # Reset state
                    del st.session_state[interactive_key]
                    del st.session_state[auto_complete_key]
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error in auto-completion: {str(e)}")
                    del st.session_state[auto_complete_key]
    
    # ============================================================================
    # TAB 2: DATASET REVIEW & EDIT (Simplified from explorer)
    # ============================================================================
    with review_tab:
        st.markdown("### 📋 Dataset Review & Edit")
        
        if not st.session_state.dataset_preview:
            st.info("No dataset loaded. Generate one in the Interactive Generation tab or load an existing dataset.")
            return
        
        dataset = st.session_state.dataset_preview
        
        # Dataset statistics
        st.markdown("#### 📊 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Stats
        avg_length = sum(len(sample['messages'][2]['content'].split()) for sample in dataset) / len(dataset)
        unique_responses = len(set(sample['messages'][2]['content'] for sample in dataset))
        
        with col1:
            st.metric("Total Samples", len(dataset))
        with col2:
            st.metric("Avg Response Length", f"{avg_length:.1f} words")
        with col3:
            st.metric("Unique Responses", f"{unique_responses}/{len(dataset)}")
        with col4:
            # Quality score
            quality_score = min(100, (unique_responses / len(dataset)) * 100)
            st.metric("Quality Score", f"{quality_score:.1f}%")
        
        st.markdown("---")
        
        # Simple dataset browser with editing
        st.markdown("#### 🔍 Browse & Edit Samples")
        
        # Pagination
        page_size = st.selectbox("Items per page", [10, 25, 50], value=25)
        page_count = (len(dataset) + page_size - 1) // page_size
        page_num = 1
        if page_count > 1:
            page_num = st.number_input("Page", 1, page_count, 1)
        
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(dataset))
        page_data = dataset[start_idx:end_idx]
        
        # Display samples
        for i, sample in enumerate(page_data):
            global_idx = start_idx + i
            
            with st.expander(f"Sample {global_idx + 1}: {sample['messages'][1]['content'][:50]}..."):
                # Display current content
                st.markdown("**User Prompt:**")
                user_content = st.text_area(
                    "User Prompt",
                    value=sample['messages'][1]['content'],
                    height=100,
                    key=f"user_{global_idx}",
                    label_visibility="collapsed"
                )
                
                st.markdown("**Assistant Response:**")
                assistant_content = st.text_area(
                    "Assistant Response", 
                    value=sample['messages'][2]['content'],
                    height=150,
                    key=f"assistant_{global_idx}",
                    label_visibility="collapsed"
                )
                
                # Action buttons
                col_edit1, col_edit2, col_edit3 = st.columns(3)
                
                with col_edit1:
                    if st.button("💾 Save Changes", key=f"save_{global_idx}", use_container_width=True):
                        # Update the sample
                        st.session_state.dataset_preview[global_idx]['messages'][1]['content'] = user_content
                        st.session_state.dataset_preview[global_idx]['messages'][2]['content'] = assistant_content
                        st.success("✅ Sample updated!")
                        st.rerun()
                
                with col_edit2:
                    if st.button("🗑️ Delete Sample", key=f"delete_{global_idx}", use_container_width=True):
                        # Remove the sample
                        del st.session_state.dataset_preview[global_idx]
                        st.success("✅ Sample deleted!")
                        st.rerun()
                
                with col_edit3:
                    # Show system prompt if it exists
                    if sample['messages'][0]['content']:
                        if st.button("🔧 View System", key=f"system_{global_idx}", use_container_width=True):
                            st.info(f"**System Prompt:** {sample['messages'][0]['content']}")
        
        # Add new sample
        st.markdown("---")
        st.markdown("#### ➕ Add New Sample")
        
        with st.form("new_sample_form", clear_on_submit=True):
            new_user_prompt = st.text_area("User Prompt", height=100)
            new_assistant_response = st.text_area("Assistant Response", height=150)
            
            if st.form_submit_button("Add Sample", use_container_width=True):
                if new_user_prompt and new_assistant_response:
                    system_prompt = st.session_state.dataset_metadata.get('system_prompt_config', {}).get('prompt', '')
                    new_sample = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": new_user_prompt},
                            {"role": "assistant", "content": new_assistant_response}
                        ]
                    }
                    st.session_state.dataset_preview.append(new_sample)
                    st.success("✅ Sample added!")
                    st.rerun()
                else:
                    st.warning("Both fields are required.")
    
    # ============================================================================
    # TAB 3: EXPERIMENTAL METHODS (Hidden/collapsed by default)
    # ============================================================================
    with experimental_tab:
        st.markdown("### 🧪 Experimental Generation Methods")
        
        st.warning("""
        **Experimental Methods**: These are alternative generation approaches for specific use cases.
        For most users, the **Interactive Generation** tab provides the best results.
        """)
        
        # Sub-tabs for different experimental methods
        fast_tab, slow_tab, factual_tab = st.tabs([
            "⚡ Fast Mode (Templated)",
            "🔬 Slow Mode (AI-Curated)", 
            "🎯 Factual Q&A Generation"
        ])
        
        with fast_tab:
            st.markdown("#### ⚡ Fast Mode - Template-Based Generation")
            st.info("Uses predefined templates + LLM paraphrasing for rapid dataset creation")
            
            # Simplified fast mode interface
            with st.form("fast_generation"):
                fast_num_samples = st.slider("Number of samples", 20, 500, 100, 20)
                fast_temperature = st.slider("Temperature", 0.3, 1.2, 0.7, 0.1)
                
                if st.form_submit_button("⚡ Generate Fast Dataset", use_container_width=True):
                    st.info("Fast mode generation would be implemented here")
        
        with slow_tab:
            st.markdown("#### 🔬 Slow Mode - AI-Curated Generation")
            st.info("LLM creates questions → Judge filters → Character responds → Quality control")
            
            # Simplified slow mode interface
            with st.form("slow_generation"):
                slow_num_samples = st.slider("Target final samples", 20, 300, 60, 10)
                slow_quality_threshold = st.slider("Quality Threshold", 0.6, 0.95, 0.75, 0.05)
                
                if st.form_submit_button("🔬 Generate AI-Curated Dataset", use_container_width=True):
                    st.info("Slow mode generation would be implemented here")
        
        with factual_tab:
            st.markdown("#### 🎯 Factual Q&A Generation")
            st.info("Extracts core facts from character card and creates reinforcement Q&A pairs")
            
            # Simplified factual mode interface
            with st.form("factual_generation"):
                num_facts = st.slider("Number of Core Facts", 5, 50, 15, 1)
                variations_per_fact = st.slider("Q&A Variations per Fact", 1, 10, 3, 1)
                
                if st.form_submit_button("🎯 Generate Factual Dataset", use_container_width=True):
                    st.info("Factual Q&A generation would be implemented here")

# Run the page function if this file is executed directly (for testing)
if __name__ == "__main__":
    page_dataset_studio() 