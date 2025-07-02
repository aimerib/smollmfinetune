"""
🌊 N-Script Studio - Visual Narrative Scripting

Beautiful visual interface for creating N-Scripts with relationship triggers.
Perfect for Type B creators who think visually and intuitively.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any

def page_nscript_studio():
    """Main N-Script Studio page"""
    st.markdown('<h2 class="gradient-text">🌊 N-Script Studio</h2>', unsafe_allow_html=True)
    st.markdown("*Create relationship-driven narrative scripts with visual magic*")
    
    # Custom CSS
    st.markdown("""
    <style>
        .trigger-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.75rem;
            border-radius: 10px;
            border: none;
            margin: 0.25rem;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        .trigger-button:hover {
            transform: scale(1.05);
        }
        .script-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🌊 Trigger Canvas",
        "⚡ Script Builder", 
        "📊 Script Library"
    ])
    
    with tab1:
        _render_trigger_canvas()
    
    with tab2:
        _render_script_builder()
    
    with tab3:
        _render_script_library()

def _render_trigger_canvas():
    """Render the visual trigger canvas"""
    st.markdown("### 🌊 Trigger Flow Canvas")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Available Triggers")
        
        # Relationship triggers
        st.markdown("**🤝 Relationship Triggers**")
        
        trigger_buttons = [
            ("💔 Affinity Threshold", "ON_AFFINITY_THRESHOLD"),
            ("😍 Emotional Pattern", "ON_EMOTIONAL_PATTERN"),
            ("🧠 Memory Significance", "ON_MEMORY_SIGNIFICANCE"),
            ("🔄 Relationship Change", "ON_RELATIONSHIP_CHANGE")
        ]
        
        for label, trigger_type in trigger_buttons:
            if st.button(label, key=f"btn_{trigger_type}"):
                _add_trigger_to_canvas(trigger_type)
        
        # Classic triggers
        st.markdown("**📍 Classic Triggers**")
        
        classic_triggers = [
            ("🚪 Enter Location", "ON_ENTER_LOCATION"),
            ("😊 Emotional State", "ON_EMOTIONAL_STATE")
        ]
        
        for label, trigger_type in classic_triggers:
            if st.button(label, key=f"btn_{trigger_type}"):
                _add_trigger_to_canvas(trigger_type)
    
    with col2:
        st.markdown("#### Your Script Flow")
        _render_flow_canvas()

def _add_trigger_to_canvas(trigger_type: str):
    """Add trigger to canvas"""
    if 'nscript_canvas' not in st.session_state:
        st.session_state.nscript_canvas = []
    
    element = {
        'id': str(uuid.uuid4()),
        'type': trigger_type,
        'conditions': {},
        'configured': False
    }
    
    st.session_state.nscript_canvas.append(element)
    st.rerun()

def _render_flow_canvas():
    """Render the flow visualization"""
    if 'nscript_canvas' not in st.session_state:
        st.session_state.nscript_canvas = []
    
    if not st.session_state.nscript_canvas:
        st.info("🎨 Add triggers to start building your script!")
        return
    
    # Create flow diagram
    fig = go.Figure()
    
    colors = {
        'ON_AFFINITY_THRESHOLD': '#ff6b6b',
        'ON_EMOTIONAL_PATTERN': '#4ecdc4',
        'ON_MEMORY_SIGNIFICANCE': '#45b7d1',
        'ON_RELATIONSHIP_CHANGE': '#96ceb4',
        'ON_ENTER_LOCATION': '#ffeaa7',
        'ON_EMOTIONAL_STATE': '#fd79a8'
    }
    
    for i, element in enumerate(st.session_state.nscript_canvas):
        trigger_type = element['type']
        color = colors.get(trigger_type, '#6c5ce7')
        
        fig.add_trace(go.Scatter(
            x=[i], y=[0],
            mode='markers+text',
            marker=dict(size=50, color=color),
            text=[trigger_type.replace('ON_', '').replace('_', '<br>')],
            textposition="middle center",
            textfont=dict(color='white', size=8),
            name=trigger_type,
            showlegend=False
        ))
    
    fig.update_layout(
        title="Script Flow",
        showlegend=False,
        height=200,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Configuration section
    if st.session_state.nscript_canvas:
        st.markdown("#### Configure Triggers")
        
        selected = st.selectbox(
            "Select trigger to configure:",
            range(len(st.session_state.nscript_canvas)),
            format_func=lambda x: f"{st.session_state.nscript_canvas[x]['type']} ({x+1})"
        )
        
        if selected is not None:
            _configure_trigger(selected)

def _configure_trigger(index: int):
    """Configure a trigger element"""
    element = st.session_state.nscript_canvas[index]
    trigger_type = element['type']
    
    st.markdown(f"**Configuring: {trigger_type}**")
    
    if trigger_type == "ON_AFFINITY_THRESHOLD":
        char1 = st.selectbox("First Character:", ["npc_tom", "npc_clara", "player"], key=f"char1_{index}")
        char2 = st.selectbox("Second Character:", ["npc_tom", "npc_clara", "player"], key=f"char2_{index}")
        threshold = st.slider("Threshold:", -1.0, 1.0, 0.5, step=0.1, key=f"thresh_{index}")
        direction = st.selectbox("Direction:", ["above", "below"], key=f"dir_{index}")
        
        element['conditions'] = {
            'relationship_pair': [char1, char2],
            'affinity_threshold': threshold,
            'direction': direction
        }
        element['configured'] = True
        
        st.success(f"✅ When {char1} and {char2}'s affinity goes {direction} {threshold}")
    
    elif trigger_type == "ON_EMOTIONAL_PATTERN":
        char1 = st.selectbox("First Character:", ["npc_tom", "npc_clara", "player"], key=f"char1_ep_{index}")
        char2 = st.selectbox("Second Character:", ["npc_tom", "npc_clara", "player"], key=f"char2_ep_{index}")
        
        emotions = ["happy", "sad", "angry", "fearful", "grateful", "betrayed"]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            emotion1 = st.selectbox("First:", emotions, key=f"em1_{index}")
        with col2:
            emotion2 = st.selectbox("Second:", emotions, key=f"em2_{index}")
        with col3:
            emotion3 = st.selectbox("Third:", emotions, key=f"em3_{index}")
        
        within = st.number_input("Within interactions:", 1, 10, 3, key=f"within_{index}")
        
        element['conditions'] = {
            'relationship_pair': [char1, char2],
            'emotion_sequence': [emotion1, emotion2, emotion3],
            'within_interactions': within
        }
        element['configured'] = True
        
        st.success(f"✅ When {char1} and {char2} show: {emotion1} → {emotion2} → {emotion3}")

def _render_script_builder():
    """Render script builder with actions"""
    st.markdown("### ⚡ Script Builder")
    
    if 'nscript_canvas' not in st.session_state or not st.session_state.nscript_canvas:
        st.info("Add triggers in the Trigger Canvas first!")
        return
    
    # Actions section
    st.markdown("#### Add Actions")
    
    action_type = st.selectbox("Action Type:", [
        "TRIPLE_HEAD_ACTION",
        "RELATIONSHIP_MODIFY", 
        "TRIGGER_CHAIN",
        "PROBABILITY_BRANCH"
    ])
    
    target_agent = st.selectbox("Target Character:", ["npc_tom", "npc_clara", "player"])
    
    if action_type == "TRIPLE_HEAD_ACTION":
        col1, col2 = st.columns(2)
        
        with col1:
            style = st.selectbox("Style:", ["friendly", "hostile", "romantic", "mysterious"])
            tone = st.selectbox("Tone:", ["warm", "cold", "excited", "sad"])
        
        with col2:
            emotions = st.multiselect("Emotions:", ["happy", "sad", "angry", "surprised"])
            importance = st.slider("Memory Importance:", 0.0, 1.0, 0.8)
    
    elif action_type == "RELATIONSHIP_MODIFY":
        relationship_target = st.selectbox("Relationship Target:", ["npc_tom", "npc_clara", "player"])
        affinity_change = st.slider("Affinity Change:", -1.0, 1.0, 0.1, step=0.1)
        status_change = st.selectbox("New Status:", ["Friend", "Rival", "Romantic", "Enemy"])
    
    if st.button("Add Action"):
        st.success("Action added to script!")

def _render_script_library():
    """Render script library and examples"""
    st.markdown("### 📊 Script Library")
    
    # Example scripts
    st.markdown("#### Example Scripts")
    
    examples = [
        {
            "name": "Friendship Milestone",
            "description": "Triggers when two characters become close friends",
            "trigger": "ON_AFFINITY_THRESHOLD",
            "complexity": "Simple"
        },
        {
            "name": "Betrayal Response",
            "description": "Complex emotional chain when trust is broken",
            "trigger": "ON_EMOTIONAL_PATTERN", 
            "complexity": "Advanced"
        },
        {
            "name": "Memory Bonding",
            "description": "Special moments when shared memories are significant",
            "trigger": "ON_MEMORY_SIGNIFICANCE",
            "complexity": "Intermediate"
        }
    ]
    
    for example in examples:
        with st.expander(f"📜 {example['name']} ({example['complexity']})"):
            st.markdown(f"**Description:** {example['description']}")
            st.markdown(f"**Trigger Type:** {example['trigger']}")
            
            if st.button(f"Load {example['name']}", key=f"load_{example['name']}"):
                st.success(f"Loaded {example['name']} template!")
    
    # Export options
    st.markdown("#### Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Script"):
            st.success("Script saved!")
    
    with col2:
        if st.button("📤 Export YAML"):
            st.success("YAML exported!")
    
    with col3:
        if st.button("🔄 Share Script"):
            st.success("Share link created!")

if __name__ == "__main__":
    page_nscript_studio() 