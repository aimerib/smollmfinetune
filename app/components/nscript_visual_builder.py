"""
🌊 N-Script Visual Builder - Where Narrative Magic Happens

A beautiful, intuitive visual interface for creating N-Scripts that makes
narrative designers feel like digital wizards. Transforms complex scripting
into an engaging, visual experience perfect for Type B creators.

Features:
- 🌊 Trigger Flow Canvas - Visual flowchart for conditions
- ⚡ Chain Reactions - Cascading trigger visualization  
- 🎲 Probability Sliders - Visual chance controls
- 🎭 Multi-Character Orchestration - Conduct characters like a symphony
- 📊 Dynamic Variables - Visual containers for story state
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid

# Import N-Script components
from narrative_engine.nscript import (
    Script, NScriptTrigger, NScriptAction, 
    TriggerType, ActionType, ScriptManager
)


class VisualNScriptBuilder:
    """Visual N-Script builder with drag-and-drop interface"""
    
    def __init__(self):
        self.current_script = None
        self.available_characters = []
        self.available_relationships = []
        self.canvas_elements = []
        self.dynamic_variables = {}
        
    def render(self):
        """Render the complete visual N-Script builder interface"""
        st.markdown('<h2 class="gradient-text">🌊 Visual N-Script Studio</h2>', unsafe_allow_html=True)
        
        # Custom CSS for visual builder
        st.markdown("""
        <style>
            .script-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            
            .trigger-node {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 0.5rem;
                text-align: center;
                cursor: pointer;
                transition: transform 0.3s ease;
            }
            
            .trigger-node:hover {
                transform: scale(1.05);
            }
            
            .action-node {
                background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 0.5rem;
                text-align: center;
                cursor: pointer;
                transition: transform 0.3s ease;
            }
            
            .action-node:hover {
                transform: scale(1.05);
            }
            
            .flow-connection {
                border: 2px dashed #6366f1;
                border-radius: 5px;
                padding: 0.5rem;
                margin: 0.5rem 0;
                background: rgba(99, 102, 241, 0.1);
            }
            
            .probability-slider {
                background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 100%);
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
            }
            
            .variable-container {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border: 2px solid #4ecdc4;
            }
            
            .character-conductor {
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Main tabs for the visual builder
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🌊 Trigger Flow Canvas",
            "⚡ Chain Reactions", 
            "🎲 Probability Weighting",
            "🎭 Character Orchestration",
            "📊 Dynamic Variables"
        ])
        
        with tab1:
            self._render_trigger_flow_canvas()
            
        with tab2:
            self._render_chain_reactions()
            
        with tab3:
            self._render_probability_weighting()
            
        with tab4:
            self._render_character_orchestration()
            
        with tab5:
            self._render_dynamic_variables()
        
        # Script preview and export
        st.markdown("---")
        self._render_script_preview()
    
    def _render_trigger_flow_canvas(self):
        """Render the visual trigger flow canvas"""
        st.markdown("### 🌊 Trigger Flow Canvas")
        st.markdown("*Create narrative triggers with visual flow - like building a story circuit!*")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Available Triggers")
            
            # Relationship-based triggers
            st.markdown("**🤝 Relationship Triggers**")
            
            if st.button("💔 Affinity Threshold", key="affinity_trigger"):
                self._add_trigger_to_canvas("ON_AFFINITY_THRESHOLD")
            
            if st.button("😍 Emotional Pattern", key="emotion_pattern_trigger"):
                self._add_trigger_to_canvas("ON_EMOTIONAL_PATTERN")
            
            if st.button("🧠 Memory Significance", key="memory_trigger"):
                self._add_trigger_to_canvas("ON_MEMORY_SIGNIFICANCE")
            
            if st.button("🔄 Relationship Change", key="relationship_change_trigger"):
                self._add_trigger_to_canvas("ON_RELATIONSHIP_CHANGE")
            
            # Classic triggers
            st.markdown("**📍 Classic Triggers**")
            
            if st.button("🚪 Enter Location", key="location_trigger"):
                self._add_trigger_to_canvas("ON_ENTER_LOCATION")
            
            if st.button("😊 Emotional State", key="emotional_state_trigger"):
                self._add_trigger_to_canvas("ON_EMOTIONAL_STATE")
        
        with col2:
            st.markdown("#### Your Story Flow")
            
            # Render the visual canvas
            self._render_flow_canvas()
    
    def _render_flow_canvas(self):
        """Render the interactive flow canvas"""
        if not hasattr(st.session_state, 'nscript_canvas_elements'):
            st.session_state.nscript_canvas_elements = []
        
        if not st.session_state.nscript_canvas_elements:
            st.info("🎨 Your story canvas is empty. Add triggers from the left to start building!")
            return
        
        # Create a flow diagram
        fig = go.Figure()
        
        # Add nodes for each element
        for i, element in enumerate(st.session_state.nscript_canvas_elements):
            trigger_type = element.get('type', 'Unknown')
            
            # Different colors for different trigger types
            color_map = {
                'ON_AFFINITY_THRESHOLD': '#ff6b6b',
                'ON_EMOTIONAL_PATTERN': '#4ecdc4', 
                'ON_MEMORY_SIGNIFICANCE': '#45b7d1',
                'ON_RELATIONSHIP_CHANGE': '#96ceb4',
                'ON_ENTER_LOCATION': '#ffeaa7',
                'ON_EMOTIONAL_STATE': '#fd79a8'
            }
            
            color = color_map.get(trigger_type, '#6c5ce7')
            
            fig.add_trace(go.Scatter(
                x=[i], y=[0],
                mode='markers+text',
                marker=dict(size=50, color=color),
                text=[trigger_type.replace('ON_', '').replace('_', '<br>')],
                textposition="middle center",
                textfont=dict(color='white', size=10),
                hovertemplate=f"<b>{trigger_type}</b><br>Click to configure<extra></extra>",
                name=trigger_type
            ))
        
        # Add flow connections
        if len(st.session_state.nscript_canvas_elements) > 1:
            x_coords = list(range(len(st.session_state.nscript_canvas_elements)))
            y_coords = [0] * len(st.session_state.nscript_canvas_elements)
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines',
                line=dict(color='rgba(99, 102, 241, 0.3)', width=3, dash='dash'),
                hoverinfo='skip',
                showlegend=False
            ))
        
        fig.update_layout(
            title="Your N-Script Flow",
            showlegend=False,
            height=200,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Element configuration
        if st.session_state.nscript_canvas_elements:
            st.markdown("#### Configure Elements")
            
            selected_element = st.selectbox(
                "Select element to configure:",
                range(len(st.session_state.nscript_canvas_elements)),
                format_func=lambda x: f"{st.session_state.nscript_canvas_elements[x]['type']} ({x+1})"
            )
            
            if selected_element is not None:
                self._configure_trigger_element(selected_element)
    
    def _add_trigger_to_canvas(self, trigger_type: str):
        """Add a trigger to the visual canvas"""
        if 'nscript_canvas_elements' not in st.session_state:
            st.session_state.nscript_canvas_elements = []
        
        element = {
            'id': str(uuid.uuid4()),
            'type': trigger_type,
            'conditions': {},
            'configured': False
        }
        
        st.session_state.nscript_canvas_elements.append(element)
        st.rerun()
    
    def _configure_trigger_element(self, element_index: int):
        """Configure a trigger element"""
        element = st.session_state.nscript_canvas_elements[element_index]
        trigger_type = element['type']
        
        st.markdown(f"**Configuring: {trigger_type}**")
        
        if trigger_type == "ON_AFFINITY_THRESHOLD":
            self._configure_affinity_trigger(element, element_index)
        elif trigger_type == "ON_EMOTIONAL_PATTERN":
            self._configure_emotional_pattern_trigger(element, element_index)
        elif trigger_type == "ON_MEMORY_SIGNIFICANCE":
            self._configure_memory_trigger(element, element_index)
        elif trigger_type == "ON_RELATIONSHIP_CHANGE":
            self._configure_relationship_change_trigger(element, element_index)
        elif trigger_type == "ON_ENTER_LOCATION":
            self._configure_location_trigger(element, element_index)
        elif trigger_type == "ON_EMOTIONAL_STATE":
            self._configure_emotional_state_trigger(element, element_index)
    
    def _configure_affinity_trigger(self, element: Dict, element_index: int):
        """Configure affinity threshold trigger"""
        st.markdown("*Trigger when relationship affinity crosses a threshold*")
        
        # Character selection
        char1 = st.selectbox("First Character:", ["npc_tom", "npc_clara", "player"], key=f"char1_{element_index}")
        char2 = st.selectbox("Second Character:", ["npc_tom", "npc_clara", "player"], key=f"char2_{element_index}")
        
        # Threshold configuration
        threshold = st.slider(
            "Affinity Threshold:", 
            min_value=-1.0, max_value=1.0, value=0.5, step=0.1,
            key=f"threshold_{element_index}",
            help="Trigger when affinity crosses this value"
        )
        
        direction = st.selectbox(
            "Direction:", 
            ["above", "below"], 
            key=f"direction_{element_index}",
            help="Trigger when affinity goes above or below threshold"
        )
        
        # Once option
        once = st.checkbox("Trigger only once", key=f"once_{element_index}")
        
        # Update element
        element['conditions'] = {
            'relationship_pair': [char1, char2],
            'affinity_threshold': threshold,
            'direction': direction
        }
        element['once'] = once
        element['configured'] = True
        
        # Visual feedback
        st.success(f"✅ Configured: When {char1} and {char2}'s affinity goes {direction} {threshold}")
    
    def _configure_emotional_pattern_trigger(self, element: Dict, element_index: int):
        """Configure emotional pattern trigger"""
        st.markdown("*Trigger when characters show specific emotional sequences*")
        
        # Character selection
        char1 = st.selectbox("First Character:", ["npc_tom", "npc_clara", "player"], key=f"char1_ep_{element_index}")
        char2 = st.selectbox("Second Character:", ["npc_tom", "npc_clara", "player"], key=f"char2_ep_{element_index}")
        
        # Emotional sequence
        st.markdown("**Emotional Sequence:**")
        emotions = ["happy", "sad", "angry", "fearful", "surprised", "disgusted", "grateful", "betrayed"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            emotion1 = st.selectbox("First Emotion:", emotions, key=f"emotion1_{element_index}")
        with col2:
            emotion2 = st.selectbox("Second Emotion:", emotions, key=f"emotion2_{element_index}")
        with col3:
            emotion3 = st.selectbox("Third Emotion:", emotions, key=f"emotion3_{element_index}")
        
        # Interaction window
        within_interactions = st.number_input(
            "Within interactions:", 
            min_value=1, max_value=10, value=3,
            key=f"within_{element_index}",
            help="Look for this pattern within N recent interactions"
        )
        
        # Update element
        element['conditions'] = {
            'relationship_pair': [char1, char2],
            'emotion_sequence': [emotion1, emotion2, emotion3],
            'within_interactions': within_interactions
        }
        element['configured'] = True
        
        # Visual feedback
        st.success(f"✅ Configured: When {char1} and {char2} show pattern: {emotion1} → {emotion2} → {emotion3}")
    
    def _configure_memory_trigger(self, element: Dict, element_index: int):
        """Configure memory significance trigger"""
        st.markdown("*Trigger when shared memories reach significance threshold*")
        
        # Character selection
        char1 = st.selectbox("First Character:", ["npc_tom", "npc_clara", "player"], key=f"char1_mem_{element_index}")
        char2 = st.selectbox("Second Character:", ["npc_tom", "npc_clara", "player"], key=f"char2_mem_{element_index}")
        
        # Memory significance
        significance = st.slider(
            "Memory Significance Threshold:", 
            min_value=0.0, max_value=1.0, value=0.8, step=0.1,
            key=f"significance_{element_index}",
            help="Trigger when memory significance exceeds this value"
        )
        
        # Memory count
        memory_count = st.number_input(
            "Minimum shared memories:", 
            min_value=1, max_value=20, value=5,
            key=f"mem_count_{element_index}",
            help="Number of significant shared memories required"
        )
        
        # Update element
        element['conditions'] = {
            'relationship_pair': [char1, char2],
            'memory_significance_above': significance,
            'memory_count': memory_count
        }
        element['configured'] = True
        
        # Visual feedback
        st.success(f"✅ Configured: When {char1} and {char2} have {memory_count}+ memories with significance > {significance}")
    
    def _configure_relationship_change_trigger(self, element: Dict, element_index: int):
        """Configure relationship change trigger"""
        st.markdown("*Trigger when relationship status changes*")
        
        # Character selection
        char1 = st.selectbox("First Character:", ["npc_tom", "npc_clara", "player"], key=f"char1_rc_{element_index}")
        char2 = st.selectbox("Second Character:", ["npc_tom", "npc_clara", "player"], key=f"char2_rc_{element_index}")
        
        # Change magnitude
        change_magnitude = st.slider(
            "Minimum change magnitude:", 
            min_value=0.1, max_value=1.0, value=0.3, step=0.1,
            key=f"change_mag_{element_index}",
            help="Minimum affinity change to trigger"
        )
        
        # Change direction
        change_direction = st.selectbox(
            "Change direction:", 
            ["any", "positive", "negative"], 
            key=f"change_dir_{element_index}"
        )
        
        # Update element
        element['conditions'] = {
            'relationship_pair': [char1, char2],
            'affinity_change': f"> {change_magnitude}",
            'change_direction': change_direction
        }
        element['configured'] = True
        
        # Visual feedback
        st.success(f"✅ Configured: When {char1} and {char2}'s relationship changes by {change_magnitude}+ ({change_direction})")
    
    def _configure_location_trigger(self, element: Dict, element_index: int):
        """Configure location trigger"""
        st.markdown("*Trigger when character enters a location*")
        
        location = st.text_input("Location ID:", key=f"location_{element_index}")
        actor = st.selectbox("Character:", ["player", "npc_tom", "npc_clara", "any"], key=f"actor_{element_index}")
        
        element['conditions'] = {
            'location_id': location,
            'actor_filter': actor if actor != "any" else None
        }
        element['configured'] = True
        
        st.success(f"✅ Configured: When {actor} enters {location}")
    
    def _configure_emotional_state_trigger(self, element: Dict, element_index: int):
        """Configure emotional state trigger"""
        st.markdown("*Trigger when character's emotional state changes*")
        
        target_agent = st.selectbox("Target Character:", ["player", "npc_tom", "npc_clara"], key=f"target_{element_index}")
        emotional_state = st.selectbox("Emotional State:", ["happy", "sad", "angry", "fearful"], key=f"emotion_{element_index}")
        intensity = st.slider("Minimum Intensity:", 0.0, 1.0, 0.6, key=f"intensity_{element_index}")
        
        element['conditions'] = {
            'target_agent_id': target_agent,
            'emotional_state': emotional_state,
            'mood_intensity': f"> {intensity}"
        }
        element['configured'] = True
        
        st.success(f"✅ Configured: When {target_agent} feels {emotional_state} with intensity > {intensity}")
    
    def _render_chain_reactions(self):
        """Render chain reaction builder"""
        st.markdown("### ⚡ Chain Reactions")
        st.markdown("*Create cascading triggers that light up in sequence!*")
        
        # Chain reaction visualization
        fig = go.Figure()
        
        # Example chain
        chain_steps = ["Trust Broken", "Confrontation", "Reconciliation", "Deeper Bond"]
        x_coords = list(range(len(chain_steps)))
        y_coords = [0] * len(chain_steps)
        
        # Add nodes
        colors = ['#ff4757', '#ff6348', '#2ed573', '#3742fa']
        for i, (step, color) in enumerate(zip(chain_steps, colors)):
            fig.add_trace(go.Scatter(
                x=[i], y=[0],
                mode='markers+text',
                marker=dict(size=60, color=color),
                text=[step],
                textposition="middle center",
                textfont=dict(color='white', size=10),
                showlegend=False
            ))
        
        # Add arrows
        for i in range(len(chain_steps) - 1):
            fig.add_annotation(
                x=i + 0.5, y=0,
                ax=i + 0.4, ay=0,
                axref='x', ayref='y',
                xref='x', yref='y',
                arrowhead=2,
                arrowsize=2,
                arrowwidth=3,
                arrowcolor='#6c5ce7'
            )
        
        fig.update_layout(
            title="Example Chain Reaction: Trust → Confrontation → Reconciliation",
            showlegend=False,
            height=200,
            xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, len(chain_steps) - 0.5]),
            yaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, 0.5]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Chain builder
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Build Your Chain")
            
            if st.button("🔗 Add Chain Step"):
                if 'chain_steps' not in st.session_state:
                    st.session_state.chain_steps = []
                st.session_state.chain_steps.append(f"Step {len(st.session_state.chain_steps) + 1}")
                st.rerun()
            
            if st.button("🗑️ Clear Chain"):
                st.session_state.chain_steps = []
                st.rerun()
        
        with col2:
            st.markdown("#### Chain Preview")
            
            if hasattr(st.session_state, 'chain_steps') and st.session_state.chain_steps:
                for i, step in enumerate(st.session_state.chain_steps):
                    st.markdown(f"**{i+1}.** {step}")
            else:
                st.info("No chain steps yet. Add some steps to see the preview!")
    
    def _render_probability_weighting(self):
        """Render probability weighting interface"""
        st.markdown("### 🎲 Probability Weighting")
        st.markdown("*Add chance and randomness to your narratives - because life is unpredictable!*")
        
        # Probability visualization
        st.markdown("#### Probability Sliders")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Outcome A: Friendship**")
            prob_a = st.slider("Probability:", 0.0, 1.0, 0.7, key="prob_a")
            st.markdown(f"<div class='probability-slider'>Chance: {prob_a*100:.0f}%</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Outcome B: Rivalry**")
            prob_b = st.slider("Probability:", 0.0, 1.0, 0.3, key="prob_b")
            st.markdown(f"<div class='probability-slider'>Chance: {prob_b*100:.0f}%</div>", unsafe_allow_html=True)
        
        # Probability pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Friendship', 'Rivalry'], 
            values=[prob_a, prob_b],
            marker_colors=['#4ecdc4', '#ff6b6b']
        )])
        
        fig.update_layout(
            title="Probability Distribution",
            showlegend=True,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Probability conditions
        st.markdown("#### Conditional Probabilities")
        
        condition_type = st.selectbox(
            "Condition Type:",
            ["Character Personality", "Relationship Status", "Recent Events", "Time of Day"]
        )
        
        if condition_type == "Character Personality":
            trait = st.selectbox("Personality Trait:", ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"])
            threshold = st.slider("Threshold:", 0.0, 1.0, 0.5)
            
            st.info(f"If {trait} > {threshold}, increase probability by 20%")
        
        elif condition_type == "Relationship Status":
            status = st.selectbox("Status:", ["Stranger", "Friend", "Rival", "Romantic"])
            
            st.info(f"If relationship is {status}, modify probabilities accordingly")
    
    def _render_character_orchestration(self):
        """Render multi-character orchestration interface"""
        st.markdown("### 🎭 Character Orchestration")
        st.markdown("*Conduct multiple characters like a symphony conductor!*")
        
        # Character selection
        st.markdown("#### Available Characters")
        
        characters = ["npc_tom", "npc_clara", "npc_sarah", "npc_alex"]
        selected_chars = st.multiselect(
            "Select characters to orchestrate:",
            characters,
            default=["npc_tom", "npc_clara"]
        )
        
        if selected_chars:
            # Timeline visualization
            st.markdown("#### Orchestration Timeline")
            
            fig = go.Figure()
            
            # Create timeline for each character
            for i, char in enumerate(selected_chars):
                # Add character timeline
                fig.add_trace(go.Scatter(
                    x=[0, 1, 2, 3, 4],
                    y=[i, i, i, i, i],
                    mode='lines+markers',
                    name=char,
                    line=dict(width=6),
                    marker=dict(size=12)
                ))
                
                # Add action annotations
                actions = ["React", "Speak", "Move", "Think", "Emote"]
                for j, action in enumerate(actions):
                    fig.add_annotation(
                        x=j, y=i,
                        text=action,
                        showarrow=False,
                        yshift=15,
                        font=dict(size=10)
                    )
            
            fig.update_layout(
                title="Character Action Timeline",
                xaxis_title="Time Steps",
                yaxis_title="Characters",
                yaxis=dict(
                    tickvals=list(range(len(selected_chars))),
                    ticktext=selected_chars
                ),
                height=200 + len(selected_chars) * 50,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Character-specific controls
            st.markdown("#### Individual Character Controls")
            
            for char in selected_chars:
                with st.expander(f"🎭 {char.replace('npc_', '').title()}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        action = st.selectbox(f"Action:", ["Speak", "Move", "Emote", "Think"], key=f"action_{char}")
                    
                    with col2:
                        timing = st.slider(f"Timing:", 0.0, 5.0, 1.0, key=f"timing_{char}")
                    
                    with col3:
                        intensity = st.slider(f"Intensity:", 0.0, 1.0, 0.5, key=f"intensity_{char}")
                    
                    st.markdown(f"<div class='character-conductor'>**{char}** will **{action}** at time **{timing}** with intensity **{intensity}**</div>", unsafe_allow_html=True)
    
    def _render_dynamic_variables(self):
        """Render dynamic variables interface"""
        st.markdown("### 📊 Dynamic Variables")
        st.markdown("*Visual containers that change based on story events - like trust meters and mood rings!*")
        
        # Variable types
        st.markdown("#### Available Variable Types")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💝 Trust Meter"):
                self._add_dynamic_variable("trust_meter", "Trust Level", 0.0, 1.0, 0.5)
        
        with col2:
            if st.button("🌡️ Tension Gauge"):
                self._add_dynamic_variable("tension", "Tension Level", 0.0, 1.0, 0.2)
        
        with col3:
            if st.button("🎭 Mood Ring"):
                self._add_dynamic_variable("mood", "Mood State", 0, 10, 5)
        
        # Display current variables
        if hasattr(st.session_state, 'dynamic_variables') and st.session_state.dynamic_variables:
            st.markdown("#### Current Variables")
            
            for var_id, var_data in st.session_state.dynamic_variables.items():
                self._render_variable_widget(var_id, var_data)
        
        # Variable rules
        st.markdown("#### Variable Rules")
        
        with st.expander("📝 Add Variable Rule"):
            rule_trigger = st.selectbox("When:", ["Affinity increases", "Conflict occurs", "Memory formed"])
            rule_action = st.selectbox("Then:", ["Increase variable", "Decrease variable", "Set variable"])
            rule_amount = st.slider("Amount:", 0.0, 1.0, 0.1)
            
            if st.button("Add Rule"):
                st.success(f"Rule added: {rule_trigger} → {rule_action} by {rule_amount}")
    
    def _add_dynamic_variable(self, var_type: str, name: str, min_val: float, max_val: float, current_val: float):
        """Add a dynamic variable"""
        if 'dynamic_variables' not in st.session_state:
            st.session_state.dynamic_variables = {}
        
        var_id = f"{var_type}_{len(st.session_state.dynamic_variables)}"
        st.session_state.dynamic_variables[var_id] = {
            'type': var_type,
            'name': name,
            'min_value': min_val,
            'max_value': max_val,
            'current_value': current_val
        }
        st.rerun()
    
    def _render_variable_widget(self, var_id: str, var_data: Dict):
        """Render a dynamic variable widget"""
        name = var_data['name']
        current = var_data['current_value']
        min_val = var_data['min_value']
        max_val = var_data['max_value']
        
        # Calculate percentage for visual display
        percentage = (current - min_val) / (max_val - min_val) * 100
        
        # Color based on value
        if percentage > 70:
            color = "#10b981"  # Green
        elif percentage > 40:
            color = "#f59e0b"  # Yellow
        else:
            color = "#ef4444"  # Red
        
        # Create visual container
        st.markdown(f"""
        <div class='variable-container'>
            <h4>{name}</h4>
            <div style='background: #e5e7eb; border-radius: 10px; padding: 5px;'>
                <div style='background: {color}; width: {percentage}%; height: 20px; border-radius: 8px; transition: all 0.3s ease;'></div>
            </div>
            <p style='text-align: center; margin-top: 10px;'>{current:.2f} / {max_val:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_script_preview(self):
        """Render script preview and export options"""
        st.markdown("### 📝 Script Preview")
        
        if hasattr(st.session_state, 'nscript_canvas_elements') and st.session_state.nscript_canvas_elements:
            # Generate script preview
            script_data = self._generate_script_from_canvas()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### YAML Preview")
                st.code(script_data, language="yaml")
            
            with col2:
                st.markdown("#### Visual Summary")
                
                configured_count = sum(1 for elem in st.session_state.nscript_canvas_elements if elem.get('configured', False))
                total_count = len(st.session_state.nscript_canvas_elements)
                
                st.metric("Configured Triggers", f"{configured_count}/{total_count}")
                
                if configured_count == total_count:
                    st.success("🎉 All triggers configured! Script is ready to use.")
                    
                    if st.button("💾 Save Script", type="primary"):
                        st.success("Script saved to N-Script library!")
                else:
                    st.warning(f"⚠️ {total_count - configured_count} triggers need configuration")
        else:
            st.info("🎨 Add triggers to your canvas to see the script preview!")
    
    def _generate_script_from_canvas(self) -> str:
        """Generate YAML script from canvas elements"""
        if not hasattr(st.session_state, 'nscript_canvas_elements') or not st.session_state.nscript_canvas_elements:
            return ""
        
        # For now, generate a simple preview
        script_preview = f"""script_id: visual_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}
trigger:
  type: {st.session_state.nscript_canvas_elements[0]['type']}
  conditions:
    # Configuration from visual builder
    {json.dumps(st.session_state.nscript_canvas_elements[0].get('conditions', {}), indent=4)}
actions:
  - type: TRIPLE_HEAD_ACTION
    target_agent_id: "npc_tom"
    generation_params:
      style: "responsive"
      tone: "emotional"
    control_params:
      emotions: ["surprised", "concerned"]
    memory_params:
      importance: 0.8
      tags: ["relationship_milestone"]
"""
        return script_preview


def render_visual_nscript_builder():
    """Main function to render the visual N-Script builder"""
    builder = VisualNScriptBuilder()
    builder.render()


if __name__ == "__main__":
    render_visual_nscript_builder() 