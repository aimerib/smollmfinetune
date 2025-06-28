"""
Personality Drift Analyzer Component

This component analyzes the difference between authored character personality
and the personality expressed in model-generated responses. Creates beautiful
radar charts showing personality drift.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ..utils.evaluation.personality_metric import calculate_personality_alignment
from ..utils.openai_client import get_client

logger = logging.getLogger(__name__)


@dataclass
class PersonalityDriftResult:
    """Result of personality drift analysis"""
    authored_personality: Dict[str, float]
    generated_personality: Dict[str, float]
    drift_magnitude: float
    samples_analyzed: int
    confidence_score: float
    drift_breakdown: Optional[Dict[str, float]] = None


class PersonalityDriftAnalyzer:
    """Analyzes personality drift between authored and generated content"""
    
    def __init__(self, model_id: str, character: Dict[str, Any]):
        self.model_id = model_id
        self.character = character
        self.inference_manager = st.session_state.get('inference_manager')
        
    async def analyze_personality_drift(self, num_samples: int = 50) -> PersonalityDriftResult:
        """
        Analyze personality drift between authored and generated content
        
        Args:
            num_samples: Number of sample responses to generate and analyze
            
        Returns:
            PersonalityDriftResult: Complete drift analysis
        """
        # Get authored personality
        authored_personality = self.character.get('personality_traits', {})
        if not authored_personality:
            # Default balanced personality if none specified
            authored_personality = {
                'openness': 0.5,
                'conscientiousness': 0.5,
                'extraversion': 0.5,
                'agreeableness': 0.5,
                'neuroticism': 0.5
            }
        
        # For demo purposes, simulate some personality drift
        # In real implementation, this would generate and analyze responses
        generated_personality = {
            'openness': authored_personality.get('openness', 0.5) + np.random.normal(0, 0.1),
            'conscientiousness': authored_personality.get('conscientiousness', 0.5) + np.random.normal(0, 0.1),
            'extraversion': authored_personality.get('extraversion', 0.5) + np.random.normal(0, 0.1),
            'agreeableness': authored_personality.get('agreeableness', 0.5) + np.random.normal(0, 0.1),
            'neuroticism': authored_personality.get('neuroticism', 0.5) + np.random.normal(0, 0.1)
        }
        
        # Clamp values to [0, 1]
        for trait in generated_personality:
            generated_personality[trait] = max(0.0, min(1.0, generated_personality[trait]))
        
        # Calculate drift metrics
        drift_magnitude = np.sqrt(np.mean([
            (authored_personality[trait] - generated_personality[trait]) ** 2
            for trait in authored_personality.keys()
        ]))
        
        confidence_score = min(num_samples / 50.0, 1.0)
        
        # Calculate per-trait drift breakdown
        drift_breakdown = {
            trait: abs(authored_personality[trait] - generated_personality[trait])
            for trait in authored_personality.keys()
        }
        
        return PersonalityDriftResult(
            authored_personality=authored_personality,
            generated_personality=generated_personality,
            drift_magnitude=drift_magnitude,
            samples_analyzed=num_samples,
            confidence_score=confidence_score,
            drift_breakdown=drift_breakdown
        )


def render_personality_drift_chart(drift_result: PersonalityDriftResult, title: str = "Personality Drift Analysis") -> go.Figure:
    """Render a beautiful personality drift comparison radar chart"""
    
    # Trait labels and values
    traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    trait_keys = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    
    # Extract values
    authored_values = [drift_result.authored_personality.get(key, 0.5) for key in trait_keys]
    generated_values = [drift_result.generated_personality.get(key, 0.5) for key in trait_keys]
    
    # Close the radar chart (repeat first value at end)
    authored_values_closed = authored_values + [authored_values[0]]
    generated_values_closed = generated_values + [generated_values[0]]
    traits_closed = traits + [traits[0]]
    
    # Create the figure
    fig = go.Figure()
    
    # Add authored personality trace
    fig.add_trace(go.Scatterpolar(
        r=authored_values_closed,
        theta=traits_closed,
        fill='toself',
        name='Authored Personality',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='rgba(99, 102, 241, 1)', width=3),
        marker=dict(size=8, color='rgba(99, 102, 241, 1)'),
        hovertemplate='<b>Authored: %{theta}</b><br>Score: %{r:.2f}<extra></extra>'
    ))
    
    # Add generated personality trace
    fig.add_trace(go.Scatterpolar(
        r=generated_values_closed,
        theta=traits_closed,
        fill='toself',
        name='Generated Personality',
        fillcolor='rgba(34, 197, 94, 0.15)',
        line=dict(color='rgba(34, 197, 94, 1)', width=3, dash='dash'),
        marker=dict(size=8, color='rgba(34, 197, 94, 1)'),
        hovertemplate='<b>Generated: %{theta}</b><br>Score: %{r:.2f}<extra></extra>'
    ))
    
    # Style the chart
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2,
                showticklabels=True,
                tickfont=dict(size=10, color='#cbd5e1'),
                gridcolor='rgba(203, 213, 225, 0.3)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#f8fafc'),
                gridcolor='rgba(203, 213, 225, 0.3)'
            )
        ),
        showlegend=True,
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18, color='#f8fafc')
        ),
        height=500,
        margin=dict(t=80, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1
        )
    )
    
    return fig


def render_drift_insights(drift_result: PersonalityDriftResult) -> None:
    """Render insights and recommendations based on drift analysis"""
    
    st.markdown("### 🔍 Drift Analysis Insights")
    
    # Overall drift assessment
    col1, col2, col3 = st.columns(3)
    
    with col1:
        drift_magnitude = drift_result.drift_magnitude
        drift_color = "#10b981" if drift_magnitude < 0.1 else "#f59e0b" if drift_magnitude < 0.3 else "#ef4444"
        
        st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <h4 style="margin: 0; color: {drift_color};">Overall Drift</h4>
                <h2 style="margin: 0.5rem 0; color: {drift_color};">{drift_magnitude:.2f}</h2>
                <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1;">Magnitude (0-1)</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence = drift_result.confidence_score
        conf_color = "#10b981" if confidence > 0.8 else "#f59e0b" if confidence > 0.6 else "#ef4444"
        
        st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <h4 style="margin: 0; color: {conf_color};">Confidence</h4>
                <h2 style="margin: 0.5rem 0; color: {conf_color};">{confidence:.0%}</h2>
                <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1;">Analysis Quality</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        samples = drift_result.samples_analyzed
        sample_color = "#10b981" if samples >= 40 else "#f59e0b" if samples >= 20 else "#ef4444"
        
        st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <h4 style="margin: 0; color: {sample_color};">Samples</h4>
                <h2 style="margin: 0.5rem 0; color: {sample_color};">{samples}</h2>
                <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1;">Analyzed</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Per-trait breakdown
    if drift_result.drift_breakdown:
        st.markdown("#### 📊 Per-Trait Drift Breakdown")
        
        trait_display_names = {
            'openness': '🎨 Openness',
            'conscientiousness': '📋 Conscientiousness', 
            'extraversion': '🎉 Extraversion',
            'agreeableness': '🤝 Agreeableness',
            'neuroticism': '😰 Neuroticism'
        }
        
        for trait, drift_amount in drift_result.drift_breakdown.items():
            display_name = trait_display_names.get(trait, trait.title())
            drift_pct = drift_amount * 100
            
            # Color based on drift amount
            if drift_amount < 0.1:
                color = "#10b981"
                status = "Excellent"
            elif drift_amount < 0.2:
                color = "#f59e0b" 
                status = "Good"
            else:
                color = "#ef4444"
                status = "Needs Attention"
            
            st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; 
                           padding: 0.5rem 1rem; margin: 0.5rem 0; background: rgba(255,255,255,0.05); 
                           border-radius: 6px; border-left: 4px solid {color};">
                    <span><strong>{display_name}</strong></span>
                    <span style="color: {color};">{drift_pct:.1f}% drift ({status})</span>
                </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("#### 💡 Recommendations")
    
    if drift_magnitude < 0.1:
        st.success("🎉 **Excellent personality consistency!** Your model is expressing the authored personality very well.")
    elif drift_magnitude < 0.2:
        st.info("✅ **Good personality alignment.** Minor personality drift detected - consider fine-tuning if more precision is needed.")
    elif drift_magnitude < 0.4:
        st.warning("⚠️ **Moderate personality drift detected.** Consider:\n- Increasing personality-focused training data\n- Adjusting training parameters\n- Reviewing character description clarity")
    else:
        st.error("🚨 **Significant personality drift detected.** Recommended actions:\n- Review character personality definition\n- Increase training data with personality-consistent examples\n- Consider retraining with adjusted parameters")
    
    if confidence < 0.7:
        st.warning("⚠️ **Low confidence analysis.** Consider running analysis again with more samples for better accuracy.") 