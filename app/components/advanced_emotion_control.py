"""
Advanced Emotion Control Component

This Streamlit component provides a comprehensive interface for the advanced emotion control
and narrative context integration system. It allows users to visualize and control emotion
blending, prosody parameters, and temporal consistency.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import json

from utils.narrative_context import (
    NarrativeContext, NarrativeContextService, EmotionBlendingService,
    TemporalConsistencyTracker, EmotionalState, EmotionBlend, ProsodyControl
)
from utils.living_interface import (
    LivingInterfaceOrchestrator, UserInteraction, InteractionType,
    VoiceAdaptationConfig
)


def render_emotion_blend_controls(emotion_service: EmotionBlendingService) -> EmotionBlend:
    """
    Render controls for creating and adjusting emotion blends.
    
    Args:
        emotion_service: The emotion blending service
        
    Returns:
        EmotionBlend object based on user inputs
    """
    st.subheader("🎭 Emotion Blend Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Primary emotion selection
        primary_emotions = [
            "joy", "sadness", "anger", "fear", "surprise", "disgust",
            "calm", "excitement", "determination", "anxiety", "confusion",
            "hope", "desperation", "confidence", "vulnerability"
        ]
        
        primary_emotion = st.selectbox(
            "Primary Emotion",
            primary_emotions,
            index=primary_emotions.index("calm"),
            help="The dominant emotion for the character"
        )
        
        # Narrative tension slider
        narrative_tension = st.slider(
            "Narrative Tension",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Current tension level in the narrative (affects intensity)"
        )
    
    with col2:
        # Secondary emotions
        st.write("**Secondary Emotions**")
        secondary_emotions = []
        
        for i in range(3):
            emotion_col, weight_col = st.columns([2, 1])
            
            with emotion_col:
                emotion = st.selectbox(
                    f"Secondary {i+1}",
                    ["None"] + primary_emotions,
                    key=f"secondary_emotion_{i}",
                    help=f"Optional secondary emotion {i+1}"
                )
            
            with weight_col:
                if emotion != "None":
                    weight = st.slider(
                        "Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.3,
                        step=0.1,
                        key=f"secondary_weight_{i}",
                        help="Intensity of this secondary emotion"
                    )
                    secondary_emotions.append(emotion)
                else:
                    st.write("—")
    
    # Create emotion blend
    emotion_blend = emotion_service.create_emotion_blend(
        primary_emotion=primary_emotion,
        secondary_emotions=secondary_emotions,
        narrative_tension=narrative_tension
    )
    
    # Display emotion tag
    emotion_tag = emotion_service.generate_emotion_tag(emotion_blend)
    st.info(f"**Generated Emotion Tag:** `{emotion_tag}`")
    
    return emotion_blend


def render_prosody_controls(
    emotion_blend: EmotionBlend,
    narrative_context: NarrativeContext,
    emotion_service: EmotionBlendingService
) -> ProsodyControl:
    """
    Render controls for prosody parameters.
    
    Args:
        emotion_blend: Current emotion blend
        narrative_context: Current narrative context
        emotion_service: The emotion blending service
        
    Returns:
        ProsodyControl object
    """
    st.subheader("🎵 Prosody Controls")
    
    # Calculate base prosody from emotion and context
    base_prosody = emotion_service.calculate_prosody_from_context(
        emotion_blend,
        narrative_context
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        speaking_rate = st.slider(
            "Speaking Rate",
            min_value=0.5,
            max_value=2.0,
            value=base_prosody.speaking_rate,
            step=0.1,
            help="Speed of speech (1.0 = normal)"
        )
        
        pitch_variation = st.slider(
            "Pitch Variation",
            min_value=0.5,
            max_value=2.0,
            value=base_prosody.pitch_variation,
            step=0.1,
            help="Amount of pitch variation (1.0 = normal)"
        )
    
    with col2:
        pause_duration = st.slider(
            "Pause Duration",
            min_value=0.0,
            max_value=2.0,
            value=base_prosody.pause_duration,
            step=0.1,
            help="Additional pause duration in seconds"
        )
        
        # Emphasis words
        emphasis_words = st.text_input(
            "Emphasis Words",
            value=", ".join(base_prosody.emphasis_words),
            help="Words to emphasize (comma-separated)"
        )
        
        emphasis_list = [word.strip() for word in emphasis_words.split(",") if word.strip()]
    
    return ProsodyControl(
        speaking_rate=speaking_rate,
        pause_duration=pause_duration,
        emphasis_words=emphasis_list,
        pitch_variation=pitch_variation
    )


def render_narrative_context_form() -> NarrativeContext:
    """
    Render form for manually setting narrative context.
    
    Returns:
        NarrativeContext object based on user inputs
    """
    st.subheader("📖 Narrative Context")
    
    col1, col2 = st.columns(2)
    
    with col1:
        narrative_tension = st.slider(
            "Narrative Tension",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Overall tension level in the story"
        )
        
        character_arc_stage = st.selectbox(
            "Character Arc Stage",
            [
                "exposition",
                "inciting_incident", 
                "rising_action",
                "climax",
                "falling_action",
                "resolution"
            ],
            index=2,
            help="Current stage in the character's narrative arc"
        )
    
    with col2:
        scene_atmosphere = st.selectbox(
            "Scene Atmosphere",
            [
                "calm", "tense", "comedic", "somber", "chaotic",
                "mysterious", "romantic", "energetic", "peaceful"
            ],
            index=1,
            help="Overall mood and atmosphere of the current scene"
        )
        
        dialogue_context = st.selectbox(
            "Dialogue Context",
            [
                "friendly_banter", "heated_argument", "solemn_confession",
                "desperate_plea", "confrontation", "celebration",
                "introduction", "farewell", "planning", "reminiscing"
            ],
            index=4,
            help="Immediate context of the conversation"
        )
    
    primary_emotion = st.selectbox(
        "Primary Emotion",
        [
            "calm", "determination", "anxiety", "joy", "sadness",
            "anger", "fear", "excitement", "confusion", "hope"
        ],
        index=1,
        help="Dominant emotion the character should be feeling"
    )
    
    secondary_emotions_text = st.text_input(
        "Secondary Emotions",
        value="anxiety, hope",
        help="Secondary emotions (comma-separated)"
    )
    
    secondary_emotions = [
        emotion.strip() for emotion in secondary_emotions_text.split(",")
        if emotion.strip()
    ]
    
    return NarrativeContext(
        narrative_tension=narrative_tension,
        character_arc_stage=character_arc_stage,
        scene_atmosphere=scene_atmosphere,
        dialogue_context=dialogue_context,
        primary_emotion=primary_emotion,
        secondary_emotions=secondary_emotions
    )


def render_emotional_timeline(tracker: TemporalConsistencyTracker):
    """
    Render a timeline visualization of emotional states.
    
    Args:
        tracker: The temporal consistency tracker
    """
    st.subheader("📈 Emotional Timeline")
    
    emotional_arc = tracker.get_emotional_arc(lookback_minutes=60)
    
    if not emotional_arc:
        st.info("No emotional history available yet.")
        return
    
    # Prepare data for plotting
    timestamps = [state.timestamp for state in emotional_arc]
    intensities = [state.intensity for state in emotional_arc]
    tensions = [state.narrative_tension for state in emotional_arc]
    emotions = [state.primary_emotion for state in emotional_arc]
    
    # Create subplot with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Emotional Intensity & Narrative Tension", "Primary Emotions"),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Intensity and tension lines
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=intensities,
            mode='lines+markers',
            name='Emotional Intensity',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=tensions,
            mode='lines+markers',
            name='Narrative Tension',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Emotion categories
    emotion_colors = {
        'joy': '#FFD700', 'sadness': '#4169E1', 'anger': '#DC143C',
        'fear': '#800080', 'surprise': '#FF69B4', 'disgust': '#228B22',
        'calm': '#87CEEB', 'excitement': '#FF4500', 'determination': '#2E8B57',
        'anxiety': '#DDA0DD', 'confusion': '#D2691E', 'hope': '#32CD32'
    }
    
    # Create emotion bars
    emotion_y = [1] * len(emotions)  # All at same height
    colors = [emotion_colors.get(emotion, '#808080') for emotion in emotions]
    
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=emotion_y,
            name='Primary Emotions',
            marker=dict(color=colors),
            text=emotions,
            textposition='inside',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        title="Character Emotional Progression",
        xaxis_title="Time",
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Intensity/Tension (0-1)", row=1, col=1)
    fig.update_yaxes(title_text="Emotions", row=2, col=1, showticklabels=False)
    
    st.plotly_chart(fig, use_container_width=True)


def render_emotion_radar_chart(emotion_blend: EmotionBlend):
    """
    Render a radar chart showing the emotion blend composition.
    
    Args:
        emotion_blend: The emotion blend to visualize
    """
    st.subheader("🕸️ Emotion Composition")
    
    # Prepare data
    emotions = [emotion_blend.primary_emotion]
    values = [1.0]  # Primary emotion gets full weight
    
    # Add secondary emotions
    for emotion, weight in emotion_blend.secondary_emotions.items():
        emotions.append(emotion)
        values.append(weight)
    
    # Ensure we have at least 3 points for a proper radar chart
    while len(emotions) < 3:
        emotions.append("neutral")
        values.append(0.0)
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=emotions,
        fill='toself',
        name='Emotion Blend',
        line=dict(color='rgba(0, 100, 255, 0.8)'),
        fillcolor='rgba(0, 100, 255, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title=f"Overall Intensity: {emotion_blend.overall_intensity:.2f}",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_advanced_emotion_control():
    """
    Main function to render the complete advanced emotion control interface.
    """
    st.title("🎭 Advanced Emotion Control & Narrative Context")
    
    # Initialize session state
    if "emotion_service" not in st.session_state:
        st.session_state.emotion_service = EmotionBlendingService()
    
    if "temporal_tracker" not in st.session_state:
        st.session_state.temporal_tracker = TemporalConsistencyTracker("demo_character")
    
    if "living_interface" not in st.session_state:
        config = VoiceAdaptationConfig()
        st.session_state.living_interface = LivingInterfaceOrchestrator(
            "demo_character",
            config=config
        )
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Voice adaptation settings
        st.subheader("Voice Adaptation")
        adaptation_speed = st.slider("Adaptation Speed", 0.1, 1.0, 0.7, 0.1)
        memory_influence = st.slider("Memory Influence", 0.0, 1.0, 0.3, 0.1)
        story_influence = st.slider("Story Influence", 0.0, 1.0, 0.5, 0.1)
        control_influence = st.slider("Control Influence", 0.0, 1.0, 0.2, 0.1)
        
        # Update configuration
        st.session_state.living_interface.config = VoiceAdaptationConfig(
            adaptation_speed=adaptation_speed,
            memory_influence=memory_influence,
            story_influence=story_influence,
            control_influence=control_influence
        )
        
        # Demo controls
        st.subheader("Demo Controls")
        if st.button("Reset Emotional History"):
            st.session_state.temporal_tracker = TemporalConsistencyTracker("demo_character")
            st.rerun()
        
        if st.button("Add Sample Emotional States"):
            # Add some sample states for demo
            sample_states = [
                EmotionalState(
                    primary_emotion="calm",
                    secondary_emotions={"contentment": 0.3},
                    intensity=0.4,
                    narrative_tension=0.2,
                    timestamp=datetime.now() - timedelta(minutes=30)
                ),
                EmotionalState(
                    primary_emotion="curiosity",
                    secondary_emotions={"excitement": 0.4},
                    intensity=0.6,
                    narrative_tension=0.4,
                    timestamp=datetime.now() - timedelta(minutes=20)
                ),
                EmotionalState(
                    primary_emotion="determination",
                    secondary_emotions={"anxiety": 0.3, "hope": 0.2},
                    intensity=0.8,
                    narrative_tension=0.7,
                    timestamp=datetime.now() - timedelta(minutes=10)
                )
            ]
            
            for state in sample_states:
                st.session_state.temporal_tracker.add_emotional_state(state)
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎭 Emotion Control",
        "📖 Narrative Context", 
        "📈 Emotional Timeline",
        "🔧 Live Integration"
    ])
    
    with tab1:
        # Emotion blend controls
        emotion_blend = render_emotion_blend_controls(st.session_state.emotion_service)
        
        col1, col2 = st.columns(2)
        
        with col1:
            render_emotion_radar_chart(emotion_blend)
        
        with col2:
            # Narrative context for prosody calculation
            narrative_context = render_narrative_context_form()
            
            # Prosody controls
            prosody_control = render_prosody_controls(
                emotion_blend,
                narrative_context,
                st.session_state.emotion_service
            )
            
            # Display prosody parameters
            st.subheader("🎵 Prosody Parameters")
            st.json({
                "speaking_rate": prosody_control.speaking_rate,
                "pitch_variation": prosody_control.pitch_variation,
                "pause_duration": prosody_control.pause_duration,
                "emphasis_words": prosody_control.emphasis_words
            })
    
    with tab2:
        st.subheader("📖 Narrative Context Analysis")
        
        # Text input for dialogue analysis
        dialogue_input = st.text_area(
            "Enter dialogue for analysis:",
            value="I'm really worried about what might happen tomorrow. This whole situation feels overwhelming.",
            height=100,
            help="Enter dialogue text to analyze narrative context"
        )
        
        if st.button("Analyze Context", type="primary"):
            if dialogue_input.strip():
                # Create mock dialogue history
                dialogue_history = [
                    {"role": "user", "content": dialogue_input},
                    {"role": "assistant", "content": "I understand your concerns. Let's work through this together."}
                ]
                
                # This would normally use the LLM, but for demo we'll create a mock context
                mock_context = NarrativeContext(
                    narrative_tension=0.7,
                    character_arc_stage="rising_action",
                    scene_atmosphere="tense",
                    dialogue_context="emotional_support",
                    primary_emotion="anxiety",
                    secondary_emotions=["vulnerability", "hope"]
                )
                
                st.success("Context analyzed successfully!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.json(mock_context.model_dump())
                
                with col2:
                    # Create emotion blend from context
                    context_blend = st.session_state.emotion_service.create_emotion_blend(
                        primary_emotion=mock_context.primary_emotion,
                        secondary_emotions=mock_context.secondary_emotions or [],
                        narrative_tension=mock_context.narrative_tension
                    )
                    
                    emotion_tag = st.session_state.emotion_service.generate_emotion_tag(context_blend)
                    st.info(f"**Generated Emotion Tag:** `{emotion_tag}`")
                    
                    # Add to temporal tracker
                    emotional_state = EmotionalState(
                        primary_emotion=mock_context.primary_emotion,
                        secondary_emotions={
                            emotion: 0.3 for emotion in mock_context.secondary_emotions or []
                        },
                        intensity=mock_context.narrative_tension,
                        narrative_tension=mock_context.narrative_tension,
                        timestamp=datetime.now()
                    )
                    
                    st.session_state.temporal_tracker.add_emotional_state(emotional_state)
    
    with tab3:
        render_emotional_timeline(st.session_state.temporal_tracker)
        
        # Emotional momentum
        momentum = st.session_state.temporal_tracker.calculate_emotional_momentum()
        if momentum:
            st.subheader("🌊 Emotional Momentum")
            st.json(momentum)
        
        # Character summary
        summary = st.session_state.living_interface.get_character_emotional_summary()
        st.subheader("📊 Character Summary")
        st.json(summary)
    
    with tab4:
        st.subheader("🔧 Living Interface Integration")
        
        st.info("""
        This tab demonstrates the integration with the tri-head architecture.
        In a real implementation, this would connect to the memory, story generation,
        and control heads to provide dynamic voice adaptation.
        """)
        
        # Simulate user interaction
        interaction_text = st.text_input(
            "Simulate User Interaction:",
            value="I'm excited about our upcoming adventure!",
            help="Enter text to simulate a user interaction"
        )
        
        if st.button("Process Interaction"):
            if interaction_text.strip():
                # Create user interaction
                interaction = UserInteraction(
                    interaction_type=InteractionType.DIALOGUE,
                    content=interaction_text,
                    timestamp=datetime.now()
                )
                
                dialogue_history = [
                    {"role": "user", "content": interaction_text},
                    {"role": "assistant", "content": "That sounds wonderful! I'm excited too!"}
                ]
                
                # Process with living interface (this would be async in real implementation)
                # For demo, we'll show the structure
                st.success("Interaction processed!")
                
                st.subheader("🎯 Response Parameters")
                mock_response = {
                    "narrative_context": {
                        "narrative_tension": 0.6,
                        "primary_emotion": "excitement",
                        "scene_atmosphere": "energetic"
                    },
                    "emotion_blend": {
                        "primary_emotion": "excitement",
                        "secondary_emotions": {"joy": 0.4, "anticipation": 0.3},
                        "overall_intensity": 0.8
                    },
                    "prosody_control": {
                        "speaking_rate": 1.2,
                        "pitch_variation": 1.3,
                        "pause_duration": 0.1
                    },
                    "emotion_tag": "<excitement with joy (0.4), anticipation (0.3)>",
                    "generation_parameters": {
                        "temperature": 0.85,
                        "top_p": 0.9
                    }
                }
                
                st.json(mock_response)


if __name__ == "__main__":
    render_advanced_emotion_control() 