"""
⚔️ Model Comparison Page

Advanced model comparison dashboard with side-by-side evaluation featuring:
- Multi-model response comparison with reproducible seeds
- Training progression analysis and checkpoint evaluation
- Character consistency radar charts and metrics
- Checkpoint promotion system for best model selection
- Interactive charts showing training loss vs consistency
- Detailed comparison tables with key insights

Extracted from main app.py for better maintainability.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go

def page_model_comparison():
    """Page for comparing different models side-by-side."""
    st.markdown('<h2 class="gradient-text">⚔️ Model Comparison Dashboard</h2>', unsafe_allow_html=True)

    if not st.session_state.current_character:
        st.warning("⚠️ Please upload a character card first.")
        return

    # Get character name safely - handle both CharacterCore and dict
    current_character = getattr(st.session_state, 'current_character_core', None) or getattr(st.session_state, 'current_character', None)
    if hasattr(current_character, 'name'):
        char_name = current_character.name
    elif isinstance(current_character, dict):
        char_name = current_character.get("name", "Unknown")
    else:
        char_name = "Unknown"
    st.markdown(f"### Comparing models for: **{char_name}**")

            # Get available models
    available_models = st.session_state.inference_manager.get_available_models()
    
    # Filter for models related to the current character
    character_models = [m for m in available_models if char_name.lower().replace(' ', '_') in m.lower()]
    
    # Auto-detect base models needed for comparison
    base_models_needed = set()
    for model in character_models:
        if not model.startswith("Base:"):
            metadata = st.session_state.inference_manager.get_model_metadata(model)
            if metadata and 'base_model' in metadata:
                base_models_needed.add(metadata['base_model'])
    
    # Add detected base models to available options
    for base_model in base_models_needed:
        base_option = f"Base: {base_model}"
        if base_option not in character_models:
            character_models.append(base_option)
    
    # Fallback: add current base model if no models detected
    if not character_models:
        character_models.append(f"Base: {st.session_state.inference_manager.base_model}")
    
    if len(character_models) < 2:
        st.info("ℹ️ You need at least two trained models/checkpoints for this character to compare them. Train more models or checkpoints to enable comparison.")
        return
    
    # Show detected base models info
    if base_models_needed:
        st.info(f"🔍 **Auto-detected base models**: {', '.join(sorted(base_models_needed))}")
    
    selected_models = st.multiselect(
        "Select models to compare (2 or more)",
        options=list(base_models_needed) + character_models,
        default=list(base_models_needed) + character_models[:2] if len(character_models) >= 2 else list(base_models_needed) + character_models,
        help="🎯 **Best practice**: Include base model for sanity check, then select checkpoints from the same training run to see progression."
    )

    if len(selected_models) < 2:
        st.warning("⚠️ Please select at least two models to compare.")
        return

    # Test prompt
    prompt = st.text_area(
        "Enter a test prompt",
        "Who are you and what do you want?",
        height=100,
        key="comparison_prompt"
    )

    # Generation settings
    with st.expander("⚙️ Generation Settings"):
        from utils.sampling_config import render_sampling_config_ui, SamplingConfig
        default_test_config = SamplingConfig(
            temperature=0.9,
            top_p=0.95,
            max_tokens=200,
            repetition_penalty=1.0,
        )
        test_sampling_config = render_sampling_config_ui(
            current_config=default_test_config,
            key_prefix="model_comparison_gen"
        )
        
        st.markdown("#### 🎲 Reproducibility Settings")
        col_seed1, col_seed2 = st.columns([2, 1])
        with col_seed1:
            use_custom_seed = st.checkbox(
                "Use custom seed for reproducible comparison",
                value=False,
                help="Set a specific seed to get identical results across multiple comparison runs"
            )
        with col_seed2:
            if use_custom_seed:
                custom_seed = st.number_input(
                    "Seed value",
                    min_value=0,
                    max_value=2**32-1,
                    value=42,
                    step=1,
                    help="Same seed = identical randomness for fair comparison"
                )
            else:
                custom_seed = None
        
        if not use_custom_seed:
            st.info("🎲 **Auto-seed**: A random seed will be generated and used consistently across all models for fair comparison")
        else:
            st.info(f"🔒 **Fixed seed {custom_seed}**: All models will use this seed for identical randomness")
    
    if st.button("🚀 Compare Responses", use_container_width=True, type="primary"):
        if not prompt.strip():
            st.error("❌ Please enter a prompt.")
            return

        # Store results in session state to persist them
        with st.spinner("Generating responses and fetching metrics..."):
            sp_config = test_sampling_config.to_dict()
            if 'min_tokens' in sp_config:
                min_tokens = sp_config.pop('min_tokens')
            if 'max_tokens' in sp_config:
                max_tokens = sp_config.pop('max_tokens')
            comparison_results = st.session_state.comparison_manager.compare_models_side_by_side(
                model_identifiers=selected_models,
                prompt=prompt,
                max_tokens=max_tokens,
                generation_config=sp_config,
                seed=custom_seed
            )
            
            # Enhanced metrics with personality engine judge evaluation
            with st.spinner("Evaluating character consistency with AI judge..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    base_metrics, training_summary, variable_metrics = loop.run_until_complete(
                        st.session_state.comparison_manager.get_enhanced_comparison_metrics(
                            selected_models,
                            st.session_state.current_character,
                            prompt,
                            comparison_results,
                            st.session_state.dataset_manager
                        )
                    )
                finally:
                    loop.close()
            
            st.session_state.comparison_data = {
                "responses": comparison_results,
                "base_metrics": base_metrics,
                "training_summary": training_summary,
                "variable_metrics": variable_metrics
            }
        st.success("✅ Comparison complete!")

    # Display comparison if data exists
    if 'comparison_data' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Comparison Results")
        st.info("🎯 **Fair Comparison**: All models used the same random seed, so differences in responses are due to model differences, not randomness.")

        responses = st.session_state.comparison_data['responses']
        base_metrics = st.session_state.comparison_data['base_metrics']
        training_summary = st.session_state.comparison_data['training_summary']
        variable_metrics = st.session_state.comparison_data['variable_metrics']

        # Side-by-side responses
        st.markdown("#### Side-by-Side Responses")
        cols = st.columns(len(selected_models))
        for i, model_id in enumerate(selected_models):
            with cols[i]:
                st.markdown(f"##### {model_id}")
                st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 1rem; border-radius: 8px; height: 300px; overflow-y: auto;">
                        {responses.get(model_id, "N/A")}
                    </div>
                """, unsafe_allow_html=True)
                
                # Promotion button - ONLY for checkpoints
                if model_id.startswith("Checkpoint:"):
                    checkpoint_name = model_id.split(': ')[1].split('/')[-1] if '/' in model_id else model_id.split(': ')[1]
                    if st.button(f"🏆 Promote {checkpoint_name}", key=f"promote_{i}", use_container_width=True):
                        checkpoint_id = model_id.split(': ')[1]
                        success = st.session_state.comparison_manager.promote_checkpoint(
                            character_name=char_name,
                            checkpoint_id=checkpoint_id,
                            reason=f"Promoted after comparing with prompt: '{prompt[:50]}...'"
                        )
                        if success:
                            st.success(f"✅ Promoted {checkpoint_name} as the best version!")
                        else:
                            st.error("❌ Failed to promote checkpoint.")
                elif model_id.startswith("LoRA:"):
                    st.info("Final LoRA model (already complete)")
                elif model_id.startswith("Base:"):
                    st.info("Base model (no promotion needed)")
        
        # Display promoted checkpoint info
        promoted_checkpoint = st.session_state.comparison_manager.get_promoted_checkpoint(char_name)
        if promoted_checkpoint:
            st.success(f"🏆 **Promoted Model:** `{promoted_checkpoint}` is currently selected as the best version for this character.")


        # Training Run Summary - Show shared configuration
        if training_summary:
            st.markdown("#### 📋 Training Run Summary")
            st.info("**Shared Configuration** (identical across all checkpoints from this training run)")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.markdown("**Model Configuration:**")
                st.write(f"• Base Model: `{training_summary.get('base_model', 'Unknown')}`")
                st.write(f"• Method: **{training_summary.get('training_method', 'Unknown')}**")
                if training_summary.get('use_dora'):
                    st.write("• DoRA: ✅ Enabled")
                if training_summary.get('use_rslora'):
                    st.write("• RSLoRA: ✅ Enabled")
                
            with summary_col2:
                st.markdown("**LoRA Parameters:**")
                st.write(f"• Rank (r): `{training_summary.get('lora_rank', 0)}`")
                st.write(f"• Alpha: `{training_summary.get('lora_alpha', 0)}`")
                st.write(f"• Dropout: `{training_summary.get('lora_dropout', 0.1):.2f}`")
                
            with summary_col3:
                st.markdown("**Dataset & Training:**")
                st.write(f"• Dataset Size: **{training_summary.get('dataset_size', 0)} samples**")
                st.write(f"• Configured Total Steps: `{training_summary.get('total_configured_steps', 0)}`")
                st.write(f"• Character: **{training_summary.get('character_name', 'Unknown')}**")
            
            st.markdown("---")

        # Variable Metrics Analysis - Show metrics that actually differ
        st.markdown("#### 📊 Checkpoint Progression Analysis")
        st.info("**Variable Metrics** (showing how performance changes throughout training)")
        
        # Create tabs for different types of analysis
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs(["📈 Training Progress", "🎯 Character Consistency", "🎭 Personality Drift", "📋 Detailed Comparison"])
        
        with analysis_tab1:
            st.markdown("##### Training Loss & Learning Progress")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Training progress line chart
                fig_progress = go.Figure()
                
                # Extract progression data - organize by training progression
                progression_data = []
                checkpoint_data = []
                final_data = []
                
                for model_id in selected_models:
                    metrics = variable_metrics.get(model_id, {})
                    if not metrics.get('is_base_model', False):
                        data_point = {
                            'model': model_id.split(': ')[-1] if ': ' in model_id else model_id,
                            'full_model_id': model_id,
                            'steps': metrics.get('actual_steps_completed', 0),
                            'loss': metrics.get('training_loss', 0),
                            'validation_loss': metrics.get('validation_loss', 0),
                            'is_final': metrics.get('is_final_model', False),
                            'is_checkpoint': metrics.get('is_checkpoint', False)
                        }
                        
                        if data_point['is_final']:
                            final_data.append(data_point)
                        elif data_point['is_checkpoint']:
                            checkpoint_data.append(data_point)
                        
                        progression_data.append(data_point)
                
                # Sort by steps for proper line progression
                progression_data.sort(key=lambda x: x['steps'])
                checkpoint_data.sort(key=lambda x: x['steps'])
                
                # Helpful info for users
                base_count = len([m for m in selected_models if m.startswith("Base:")])
                if len(checkpoint_data) == 0:
                    st.info("💡 **No checkpoints found.** To see training progression, you need multiple checkpoints from the same training run. Base model provides the starting baseline.")
                elif len(checkpoint_data) == 1:
                    st.info(f"📊 Found {base_count} base model(s), 1 checkpoint and {len(final_data)} final model. For progression analysis, multiple checkpoints work best.")
                else:
                    st.success(f"📊 Found {base_count} base model(s), {len(checkpoint_data)} checkpoints and {len(final_data)} final model(s) - perfect for progression analysis!")
                
                if progression_data:
                    # Add base model as starting point if selected
                    base_models_in_selection = [m for m in selected_models if m.startswith("Base:")]
                    
                    # Show training progression through checkpoints
                    if checkpoint_data:
                        checkpoint_steps = [d['steps'] for d in checkpoint_data if d['loss'] and d['loss'] > 0]
                        checkpoint_losses = [d['loss'] for d in checkpoint_data if d['loss'] and d['loss'] > 0]
                        checkpoint_names = [d['model'] for d in checkpoint_data if d['loss'] and d['loss'] > 0]
                        
                        # If we have a base model selected, add it as the starting point (step 0)
                        if base_models_in_selection and checkpoint_steps:
                            # Add base model as step 0 with a reasonable starting loss estimate
                            estimated_start_loss = max(checkpoint_losses) * 1.2 if checkpoint_losses else 2.0
                            checkpoint_steps = [0] + checkpoint_steps
                            checkpoint_losses = [estimated_start_loss] + checkpoint_losses
                            base_name = base_models_in_selection[0].replace('Base: ', '').split('/')[-1]
                            checkpoint_names = [f"{base_name} (base)"] + checkpoint_names
                        
                        if checkpoint_steps and checkpoint_losses:
                            fig_progress.add_trace(go.Scatter(
                                x=checkpoint_steps, y=checkpoint_losses,
                                mode='lines+markers',
                                name='Training Progression',
                                line=dict(color='#6366f1', width=3),
                                marker=dict(size=8),
                                text=checkpoint_names,
                                hovertemplate='<b>%{text}</b><br>Step: %{x}<br>Loss: %{y:.4f}<extra></extra>'
                            ))
                    
                    # Add validation loss if available
                    val_checkpoint_data = [d for d in checkpoint_data if d['validation_loss'] and d['validation_loss'] > 0]
                    if val_checkpoint_data:
                        val_steps = [d['steps'] for d in val_checkpoint_data]
                        val_losses = [d['validation_loss'] for d in val_checkpoint_data]
                        val_names = [d['model'] for d in val_checkpoint_data]
                        
                        fig_progress.add_trace(go.Scatter(
                            x=val_steps, y=val_losses,
                            mode='lines+markers',
                            name='Validation Loss',
                            line=dict(color='#ef4444', width=3),
                            marker=dict(size=8, color='orange'),
                            text=val_names,
                            hovertemplate='<b>%{text}</b><br>Step: %{x}<br>Val Loss: %{y:.4f}<extra></extra>'
                        ))
                    
                    # Mark final model
                    if final_data:
                        final_model = final_data[0]
                        if final_model['loss'] and final_model['loss'] > 0:
                            fig_progress.add_trace(go.Scatter(
                                x=[final_model['steps']], y=[final_model['loss']],
                                mode='markers',
                                name='Final LoRA Model',
                                marker=dict(size=15, color='#10b981', symbol='star'),
                                text=[final_model['model']],
                                hovertemplate='<b>%{text}</b><br>Step: %{x}<br>Loss: %{y:.4f}<extra></extra>'
                            ))
                    
                    # If no checkpoints, show explanation
                    if not checkpoint_data and not final_data:
                        st.info("💡 **No checkpoint progression data available.** This typically means you're comparing base models or single models without training checkpoints.")
                
                fig_progress.update_layout(
                    title="Training Progression: Base Model → Checkpoints → Final LoRA",
                    xaxis_title="Training Steps (0 = Base Model)",
                    yaxis_title="Loss",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig_progress, use_container_width=True)
            
            with col2:
                # Model Performance Summary Chart
                fig_performance = go.Figure()
                
                # Create a comprehensive performance chart including base models
                all_model_data = []
                
                # Add base models to the analysis
                for model_id in selected_models:
                    metrics = variable_metrics.get(model_id, {})
                    if metrics.get('is_base_model', False):
                        all_model_data.append({
                            'model': model_id.split(': ')[-1] if ': ' in model_id else model_id.replace('Base: ', ''),
                            'full_model_id': model_id,
                            'steps': 0,  # Base model is step 0
                            'loss': 0,   # Base models don't have training loss
                            'validation_loss': 0,
                            'is_final': False,
                            'is_checkpoint': False,
                            'is_base_model': True
                        })
                
                # Add trained models
                all_model_data.extend(checkpoint_data + final_data)
                
                if all_model_data:
                    # Performance comparison chart - Training Loss vs Character Consistency
                    model_names = []
                    training_losses = []
                    consistency_scores = []
                    model_types = []
                    
                    for d in all_model_data:
                        model_id = d['full_model_id']
                        metrics = variable_metrics.get(model_id, {})
                        
                        # For base models, we don't have training loss, so we'll use consistency only
                        if d.get('is_base_model', False):
                            consistency = metrics.get('character_consistency', 0)
                            if consistency > 0:  # Only include if we have consistency data
                                model_names.append(d['model'])
                                training_losses.append(0)  # Base model has no training loss
                                consistency_scores.append(consistency)
                                model_types.append('base')
                        elif d['loss'] and d['loss'] > 0:
                            model_names.append(d['model'])
                            training_losses.append(d['loss'])
                            consistency_scores.append(metrics.get('character_consistency', 0))
                            if d['is_checkpoint']:
                                model_types.append('checkpoint')
                            else:
                                model_types.append('final')
                    
                    if model_names and len(model_names) > 0:
                        # Color code by model type: Base=Gray, Checkpoint=Red, Final=Green
                        color_map = {'base': '#94a3b8', 'checkpoint': '#ef4444', 'final': '#10b981'}
                        colors = [color_map[t] for t in model_types]
                        
                        fig_performance.add_trace(go.Scatter(
                            x=training_losses,
                            y=consistency_scores,
                            mode='markers+text',
                            text=model_names,
                            textposition='top center',
                            marker=dict(
                                size=12,
                                color=colors,
                                line=dict(width=2, color='white')
                            ),
                            hovertemplate='<b>%{text}</b><br>Training Loss: %{x:.4f}<br>Consistency: %{y:.3f}<extra></extra>',
                            name='Models'
                        ))
                        
                        fig_performance.update_layout(
                            title="Model Performance: Loss vs Consistency",
                            xaxis_title="Training Loss (lower is better, base model at 0)",
                            yaxis_title="Character Consistency (higher is better)",
                            template="plotly_dark",
                            height=400,
                            showlegend=False
                        )
                        
                        # Add legend/annotations to explain the colors and chart
                        fig_performance.add_annotation(
                            text="🔵 Gray: Base Model (sanity check)<br/>🔴 Red: Checkpoints<br/>🟢 Green: Final LoRA<br/><br/>🎯 Good models: high consistency<br/>Base model shows pre-training behavior",
                            xref="paper", yref="paper",
                            x=0.02, y=0.98,
                            xanchor="left", yanchor="top",
                            bgcolor="rgba(0,0,0,0.7)",
                            bordercolor="white",
                            borderwidth=1,
                            font=dict(size=9)
                        )
                else:
                    # Show empty state
                    st.info("💡 **Performance comparison requires trained models.** Train some checkpoints or LoRA models to see this analysis.")
                
                if checkpoint_data or final_data:
                    st.plotly_chart(fig_performance, use_container_width=True)
                
        with analysis_tab2:
            st.markdown("##### Character Consistency Evolution")
            
            # Character consistency radar chart
            fig_consistency = go.Figure()
            
            consistency_metrics = {
                'Overall Consistency': 'character_consistency',
                'Personality': 'personality_consistency', 
                'Speech Style': 'speech_style',
                'Emotional Auth.': 'emotional_authenticity',
                'Scenario Fit': 'scenario_appropriateness'
            }
            
            colors = ['#6366f1', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']
            for i, model_id in enumerate(selected_models):
                if not variable_metrics.get(model_id, {}).get('is_base_model', False):
                    metrics = variable_metrics.get(model_id, {})
                    values = []
                    for metric_name, metric_key in consistency_metrics.items():
                        values.append(metrics.get(metric_key, 0))
                    
                    # Only add if we have non-zero values
                    if any(v > 0 for v in values):
                        fig_consistency.add_trace(go.Scatterpolar(
                            r=values + [values[0]],  # Close the polygon
                            theta=list(consistency_metrics.keys()) + [list(consistency_metrics.keys())[0]],
                            fill='toself',
                            name=model_id.split(': ')[-1] if ': ' in model_id else model_id,
                            line_color=colors[i % len(colors)]
                        ))
            
            fig_consistency.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickmode='linear',
                        tick0=0,
                        dtick=0.2
                    )),
                showlegend=True,
                title="Character Consistency by Aspect",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig_consistency, use_container_width=True)
            
        with analysis_tab3:
            st.markdown("##### 🎭 Personality Drift Analysis")
            st.info("🌟 **New Feature**: Compare how well your trained models maintain the authored personality profile.")
            
            # Model selection for drift analysis
            st.markdown("**Select a model for personality drift analysis:**")
            
            # Filter to only trained models (not base models)
            trained_models_for_drift = [m for m in selected_models if not m.startswith("Base:")]
            
            if not trained_models_for_drift:
                st.warning("⚠️ **No trained models selected.** Select some trained models (checkpoints or final models) to analyze personality drift.")
            else:
                # Select model for analysis
                selected_drift_model = st.selectbox(
                    "Choose model to analyze:",
                    trained_models_for_drift,
                    key="drift_analysis_model"
                )
                
                col_btn, col_samples = st.columns([1, 1])
                
                with col_samples:
                    num_samples = st.slider(
                        "Analysis samples:",
                        min_value=10,
                        max_value=100,
                        value=50,
                        step=10,
                        help="More samples = higher accuracy but slower analysis"
                    )
                
                with col_btn:
                    if st.button("🚀 Run Drift Analysis", type="primary", use_container_width=True):
                        if selected_drift_model and st.session_state.get('current_character_core'):
                            
                            with st.spinner(f"🔄 Analyzing personality drift for {selected_drift_model}..."):
                                try:
                                    # Import here to avoid circular imports
                                    from ..components.personality_drift_analyzer import PersonalityDriftAnalyzer
                                    
                                    # Get character data safely
                                    current_character = st.session_state.current_character_core
                                    if hasattr(current_character, 'personality_traits'):
                                        character_dict = {
                                            'name': current_character.name,
                                            'personality_traits': {
                                                'openness': current_character.personality_traits.openness,
                                                'conscientiousness': current_character.personality_traits.conscientiousness,
                                                'extraversion': current_character.personality_traits.extraversion,
                                                'agreeableness': current_character.personality_traits.agreeableness,
                                                'neuroticism': current_character.personality_traits.neuroticism
                                            }
                                        }
                                    else:
                                        character_dict = current_character
                                    
                                    # Create analyzer and run analysis
                                    analyzer = PersonalityDriftAnalyzer(selected_drift_model, character_dict)
                                    
                                    # Use asyncio to run the async function
                                    import asyncio
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    
                                    try:
                                        drift_result = loop.run_until_complete(
                                            analyzer.analyze_personality_drift(num_samples=num_samples)
                                        )
                                        
                                        # Store result in session state
                                        st.session_state[f'drift_result_{selected_drift_model}'] = drift_result
                                        
                                    finally:
                                        loop.close()
                                    
                                    st.success(f"✅ **Analysis complete!** Analyzed {drift_result.samples_analyzed} samples.")
                                    
                                except Exception as e:
                                    st.error(f"❌ **Analysis failed:** {str(e)}")
                        else:
                            st.error("❌ Please select a model and ensure a character is loaded.")
                
                # Display results if available
                if selected_drift_model:
                    drift_result_key = f'drift_result_{selected_drift_model}'
                    if drift_result_key in st.session_state:
                        drift_result = st.session_state[drift_result_key]
                        
                        st.markdown("---")
                        st.markdown("#### 📊 Analysis Results")
                        
                        # Import chart renderer
                        try:
                            from ..components.personality_drift_analyzer import render_personality_drift_chart
                            
                            # Create two columns for chart and insights
                            chart_col, insights_col = st.columns([2, 1])
                            
                            with chart_col:
                                # Render the beautiful radar chart
                                fig = render_personality_drift_chart(
                                    drift_result, 
                                    f"Personality Drift: {selected_drift_model.split(': ')[-1]}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with insights_col:
                                st.markdown("##### 🔍 Analysis Summary")
                                
                                # Drift magnitude
                                drift_magnitude = drift_result.drift_magnitude
                                if drift_magnitude < 0.1:
                                    drift_status = "🟢 Excellent"
                                    drift_color = "#10b981"
                                elif drift_magnitude < 0.2:
                                    drift_status = "🟡 Good"
                                    drift_color = "#f59e0b"
                                else:
                                    drift_status = "🔴 Needs Attention"
                                    drift_color = "#ef4444"
                                
                                st.markdown(f"""
                                    <div style="padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 4px solid {drift_color}; margin-bottom: 1rem;">
                                        <h4 style="margin: 0; color: {drift_color};">Overall Drift</h4>
                                        <h2 style="margin: 0.5rem 0; color: {drift_color};">{drift_status}</h2>
                                        <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1;">Magnitude: {drift_magnitude:.3f}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Per-trait breakdown
                                if drift_result.drift_breakdown:
                                    st.markdown("**Per-Trait Drift:**")
                                    trait_names = {
                                        'openness': '🎨 Openness',
                                        'conscientiousness': '📋 Conscientiousness',
                                        'extraversion': '🎉 Extraversion', 
                                        'agreeableness': '🤝 Agreeableness',
                                        'neuroticism': '😰 Neuroticism'
                                    }
                                    
                                    for trait, drift_amount in drift_result.drift_breakdown.items():
                                        trait_display = trait_names.get(trait, trait.title())
                                        drift_pct = drift_amount * 100
                                        
                                        if drift_amount < 0.1:
                                            trait_color = "#10b981"
                                        elif drift_amount < 0.2:
                                            trait_color = "#f59e0b"
                                        else:
                                            trait_color = "#ef4444"
                                        
                                        st.markdown(f"""
                                            <div style="display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                                                <span>{trait_display}</span>
                                                <span style="color: {trait_color}; font-weight: bold;">{drift_pct:.1f}%</span>
                                            </div>
                                        """, unsafe_allow_html=True)
                                
                                # Analysis details
                                st.markdown(f"**Samples Analyzed:** {drift_result.samples_analyzed}")
                                st.markdown(f"**Confidence:** {drift_result.confidence_score:.0%}")
                                
                                # Recommendations
                                if drift_magnitude > 0.2:
                                    st.warning("💡 **Recommendation:** Consider fine-tuning with more personality-consistent training data.")
                                elif drift_magnitude > 0.1:
                                    st.info("💡 **Recommendation:** Monitor personality consistency as training progresses.")
                                else:
                                    st.success("🎉 **Great job!** Your model maintains excellent personality consistency.")
                        
                        except ImportError as e:
                            st.error(f"Could not load drift analysis components: {e}")
            
        with analysis_tab4:
            # Detailed metrics table focusing on variable metrics
            st.markdown("##### Checkpoint Comparison Table")
            st.info("💡 **Focus on differences:** This table shows metrics that vary between checkpoints")
            
            metrics_df = []
            for model_id in selected_models:
                metrics = variable_metrics.get(model_id, {})
                
                # Create model name
                model_name = model_id.split(': ')[-1] if ': ' in model_id else model_id
                if metrics.get('is_base_model'):
                    model_type = "BASE"
                elif metrics.get('is_final_model'):
                    model_type = "FINAL"
                elif metrics.get('is_checkpoint'):
                    model_type = "CHECKPOINT"
                else:
                    model_type = "UNKNOWN"
                
                # Format values, showing only meaningful differences
                row = {
                    'Model': model_name,
                    'Type': model_type,
                    'Steps Completed': f"{int(metrics.get('actual_steps_completed', 0))}" if metrics.get('actual_steps_completed', 0) > 0 else '—',
                    'Training Loss': f"{metrics.get('training_loss', 0):.4f}" if metrics.get('training_loss', 0) > 0 else '—',
                    'Validation Loss': f"{metrics.get('validation_loss', 0):.4f}" if metrics.get('validation_loss', 0) > 0 else '—',
                    'Character Consistency': f"{metrics.get('character_consistency', 0):.3f}" if metrics.get('character_consistency', 0) > 0 else '—',
                    'Training Time (min)': f"{metrics.get('training_time_elapsed', 0):.1f}" if metrics.get('training_time_elapsed', 0) > 0 else '—',
                    'Learning Rate': f"{metrics.get('learning_rate_at_checkpoint', 0):.2e}" if metrics.get('learning_rate_at_checkpoint', 0) > 0 else '—'
                }
                metrics_df.append(row)
            
            df = pd.DataFrame(metrics_df)
            st.dataframe(df, use_container_width=True)
            
            # Key insights
            trained_models = [m for m in variable_metrics.keys() if not variable_metrics[m].get('is_base_model', False)]
            if len(trained_models) > 1:
                st.markdown("**📈 Key Insights:**")
                
                # Find best performing checkpoint
                best_consistency = max(variable_metrics[m].get('character_consistency', 0) for m in trained_models)
                best_model = [m for m in trained_models if variable_metrics[m].get('character_consistency', 0) == best_consistency][0]
                
                lowest_loss = min(variable_metrics[m].get('training_loss', float('inf')) for m in trained_models if variable_metrics[m].get('training_loss', 0) > 0)
                best_loss_model = [m for m in trained_models if variable_metrics[m].get('training_loss', float('inf')) == lowest_loss][0]
                
                if best_consistency > 0:
                    st.success(f"🏆 **Best Character Consistency:** `{best_model.split(': ')[-1] if ': ' in best_model else best_model}` ({best_consistency:.3f})")
                
                if lowest_loss < float('inf'):
                    st.success(f"📉 **Lowest Training Loss:** `{best_loss_model.split(': ')[-1] if ': ' in best_loss_model else best_loss_model}` ({lowest_loss:.4f})")
                
                # Check for overfitting
                final_models = [m for m in trained_models if variable_metrics[m].get('is_final_model', False)]
                if final_models:
                    final_model = final_models[0]
                    final_loss = variable_metrics[final_model].get('training_loss', 0)
                    final_val_loss = variable_metrics[final_model].get('validation_loss', 0)
                    
                    if final_loss > 0 and final_val_loss > 0 and final_val_loss > final_loss * 1.2:
                        st.warning("⚠️ **Potential Overfitting Detected:** Validation loss is significantly higher than training loss in final model")
                    elif final_loss > 0 and final_val_loss > 0 and abs(final_val_loss - final_loss) < 0.01:
                        st.success("✅ **Good Generalization:** Training and validation losses are well-aligned")

# Run the page function if this file is executed directly (for testing)
if __name__ == "__main__":
    page_model_comparison() 