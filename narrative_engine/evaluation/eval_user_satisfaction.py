"""
User Satisfaction Prediction

Uses LLM-as-judge with structured outputs to predict user satisfaction 
based on conversation features and patterns, enabling proactive quality improvements.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
from pydantic import BaseModel, Field

from app.utils.openai_client import get_client

logger = logging.getLogger(__name__)


class ConversationFeatures(BaseModel):
    """Structured conversation features for analysis"""
    turn_count: int = Field(description="Number of turns in conversation")
    avg_response_length: float = Field(description="Average response length in words")
    response_relevance: float = Field(ge=0.0, le=1.0, description="How relevant responses are to user queries")
    emotional_engagement: float = Field(ge=0.0, le=1.0, description="Level of emotional engagement")
    topic_coherence: float = Field(ge=0.0, le=1.0, description="How well the conversation stays on topic")
    user_questions_addressed: float = Field(ge=0.0, le=1.0, description="Proportion of user questions answered")
    conversation_flow: float = Field(ge=0.0, le=1.0, description="Naturalness of conversation flow")


class SatisfactionPrediction(BaseModel):
    """Predicted user satisfaction with detailed analysis"""
    predicted_satisfaction: float = Field(ge=0.0, le=1.0, description="Predicted satisfaction score")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the prediction")
    key_strengths: List[str] = Field(description="What the conversation does well")
    areas_for_improvement: List[str] = Field(description="What could be improved")
    satisfaction_factors: Dict[str, float] = Field(description="Individual factor contributions")
    overall_assessment: str = Field(description="Brief overall assessment")


class CorrelationAnalysis(BaseModel):
    """Analysis of feature correlations with satisfaction"""
    feature_correlations: Dict[str, float] = Field(description="Correlation of each feature with satisfaction")
    top_predictive_features: List[str] = Field(description="Most important features for prediction")
    insights: List[str] = Field(description="Key insights from the analysis")


class UserSatisfactionPredictor:
    """Predicts user satisfaction using LLM-as-judge analysis"""
    
    def __init__(self):
        """Initialize the satisfaction predictor"""
        self.client = get_client()
        self.feature_extractor = FeatureExtractor()
    
    async def predict_satisfaction(
        self,
        conversation_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict user satisfaction using LLM analysis of conversation features.
        
        Args:
            conversation_features: Dictionary of extracted features
            
        Returns:
            Dictionary with satisfaction prediction and analysis
        """
        try:
            # Build prompt for LLM analysis
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(conversation_features)
            
            # Get structured prediction from LLM
            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "satisfaction_prediction",
                        "schema": SatisfactionPrediction.model_json_schema()
                    }
                }
            )
            
            # Parse structured response
            import json
            prediction_data = json.loads(response_text)
            prediction = SatisfactionPrediction(**prediction_data)
            
            return {
                'predicted_satisfaction': prediction.predicted_satisfaction,
                'confidence': prediction.confidence,
                'key_strengths': prediction.key_strengths,
                'areas_for_improvement': prediction.areas_for_improvement,
                'satisfaction_factors': prediction.satisfaction_factors,
                'overall_assessment': prediction.overall_assessment,
                'llm_analysis': prediction.model_dump(),
                'feature_importance': prediction.satisfaction_factors  # Backward compatibility
            }
            
        except Exception as e:
            logger.error(f"Error predicting satisfaction: {e}")
            return {
                'predicted_satisfaction': 0.5,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for satisfaction prediction"""
        return """You are an expert in user experience and conversation quality analysis. Your task is to predict user satisfaction based on conversation features and provide detailed insights.

Analyze the conversation features to predict:
- Overall user satisfaction (0.0 to 1.0)
- Confidence in your prediction (0.0 to 1.0)
- Key strengths of the conversation
- Areas that could be improved
- How each factor contributes to satisfaction

Consider that users are typically more satisfied when:
- Their questions are answered directly and completely
- Responses are relevant and helpful
- The conversation flows naturally
- There is appropriate emotional engagement
- The assistant maintains topic coherence
- Response length is appropriate (not too short or too long)

Provide your analysis in the requested JSON format."""

    def _build_user_prompt(self, features: Dict[str, Any]) -> str:
        """Build user prompt with conversation features"""
        feature_text = []
        for key, value in features.items():
            if isinstance(value, float):
                feature_text.append(f"- {key.replace('_', ' ').title()}: {value:.2f}")
            else:
                feature_text.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        features_str = "\n".join(feature_text)
        
        return f"""Analyze these conversation features and predict user satisfaction:

{features_str}

Predict:
1. The overall user satisfaction score (0.0 = very unsatisfied, 1.0 = very satisfied)
2. Your confidence in this prediction
3. What the conversation does well (key strengths)
4. What could be improved
5. How each factor contributes to satisfaction
6. An overall assessment of the conversation quality

Provide detailed analysis and reasoning for your prediction."""

    async def predict_from_conversation(
        self,
        conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predict satisfaction directly from a conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Dictionary with satisfaction prediction
        """
        try:
            # Extract features from conversation
            features = self.feature_extractor.extract_features(conversation)
            
            # Get prediction
            prediction = await self.predict_satisfaction(features)
            
            # Add conversation metadata
            prediction['conversation_length'] = len(conversation)
            prediction['extracted_features'] = features
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting satisfaction from conversation: {e}")
            return {
                'predicted_satisfaction': 0.5,
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def analyze_feature_correlations(
        self,
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze correlations between features and satisfaction using LLM analysis.
        
        Args:
            historical_data: List of conversations with features and satisfaction scores
            
        Returns:
            Dictionary with correlation analysis
        """
        if not historical_data:
            return {
                'feature_correlations': {},
                'top_predictive_features': [],
                'error': 'No historical data provided'
            }
        
        try:
            # Prepare data summary for LLM analysis
            data_summary = self._prepare_correlation_summary(historical_data)
            
            # Build prompt for correlation analysis
            system_prompt = """You are a data scientist analyzing the relationship between conversation features and user satisfaction. Identify which features are most predictive of satisfaction and provide insights."""
            
            user_prompt = f"""Analyze this conversation data to identify correlations between features and satisfaction:

{data_summary}

Determine:
1. Which features correlate most strongly with user satisfaction
2. The approximate correlation strength for each feature (-1.0 to 1.0)
3. The top 5 most predictive features
4. Key insights about what drives user satisfaction

Provide your analysis in the requested format."""
            
            # Get LLM analysis
            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=600,
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "correlation_analysis",
                        "schema": CorrelationAnalysis.model_json_schema()
                    }
                }
            )
            
            # Parse response
            import json
            analysis_data = json.loads(response_text)
            analysis = CorrelationAnalysis(**analysis_data)
            
            # Add statistical correlations if possible
            statistical_correlations = self._calculate_statistical_correlations(historical_data)
            
            return {
                'feature_correlations': analysis.feature_correlations,
                'top_predictive_features': analysis.top_predictive_features,
                'insights': analysis.insights,
                'llm_analysis': analysis.model_dump(),
                'statistical_correlations': statistical_correlations,
                # Backward compatibility
                'response_relevance': analysis.feature_correlations.get('response_relevance', 0.8),
                'emotional_variety': analysis.feature_correlations.get('emotional_engagement', 0.65)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {
                'feature_correlations': {},
                'top_predictive_features': [],
                'error': str(e)
            }
    
    def _prepare_correlation_summary(self, historical_data: List[Dict[str, Any]]) -> str:
        """Prepare a summary of historical data for LLM analysis"""
        if len(historical_data) > 10:
            # Sample data to avoid token limits
            sample_data = historical_data[:10]
        else:
            sample_data = historical_data
        
        summary_lines = []
        summary_lines.append(f"Total conversations analyzed: {len(historical_data)}")
        summary_lines.append(f"Sample data (first {len(sample_data)} conversations):\n")
        
        for i, data in enumerate(sample_data, 1):
            features = data.get('features', {})
            satisfaction = data.get('satisfaction', 0)
            
            feature_summary = ", ".join([
                f"{k}: {v:.2f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                for k, v in features.items()
            ])
            
            summary_lines.append(f"{i}. Satisfaction: {satisfaction:.2f} | Features: {feature_summary}")
        
        return "\n".join(summary_lines)
    
    def _calculate_statistical_correlations(self, historical_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate statistical correlations as backup"""
        correlations = {}
        
        # Extract feature names
        feature_names = set()
        for data in historical_data:
            feature_names.update(data.get('features', {}).keys())
        
        for feature in feature_names:
            feature_values = []
            satisfaction_values = []
            
            for data in historical_data:
                if feature in data.get('features', {}):
                    feature_values.append(data['features'][feature])
                    satisfaction_values.append(data.get('satisfaction', 0))
            
            if len(feature_values) > 1:
                # Calculate correlation
                correlation = np.corrcoef(feature_values, satisfaction_values)[0, 1]
                correlations[feature] = float(correlation) if not np.isnan(correlation) else 0.0
        
        return correlations
    
    async def generate_improvement_suggestions(
        self,
        conversation_features: Dict[str, Any],
        target_satisfaction: float = 0.8
    ) -> List[str]:
        """
        Generate specific improvement suggestions using LLM analysis.
        
        Args:
            conversation_features: Current conversation features
            target_satisfaction: Target satisfaction score
            
        Returns:
            List of improvement suggestions
        """
        try:
            # Get current satisfaction prediction
            current_prediction = await self.predict_satisfaction(conversation_features)
            current_satisfaction = current_prediction['predicted_satisfaction']
            
            if current_satisfaction >= target_satisfaction:
                return ["Conversation quality is already meeting target satisfaction levels."]
            
            # Build prompt for improvement suggestions
            system_prompt = f"""You are a conversation quality expert. Provide specific, actionable suggestions to improve user satisfaction from {current_satisfaction:.2f} to {target_satisfaction:.2f}."""
            
            user_prompt = f"""Current conversation features:
{self._format_features_for_prompt(conversation_features)}

Current predicted satisfaction: {current_satisfaction:.2f}
Target satisfaction: {target_satisfaction:.2f}

Areas identified for improvement: {', '.join(current_prediction.get('areas_for_improvement', []))}

Provide 3-5 specific, actionable suggestions to improve satisfaction. Focus on the most impactful improvements."""
            
            suggestions_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.4
            )
            
            # Parse suggestions from response
            suggestions = []
            for line in suggestions_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                    # Clean up formatting
                    suggestion = line.lstrip('-•0123456789. ').strip()
                    if suggestion:
                        suggestions.append(suggestion)
            
            return suggestions[:5] if suggestions else ["Focus on improving response relevance and user question answering."]
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            return ["Unable to generate suggestions due to analysis error."]
    
    def _format_features_for_prompt(self, features: Dict[str, Any]) -> str:
        """Format features for inclusion in prompts"""
        formatted = []
        for key, value in features.items():
            if isinstance(value, float):
                formatted.append(f"- {key.replace('_', ' ').title()}: {value:.2f}")
            else:
                formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
        return "\n".join(formatted)

    # Synchronous wrapper for backward compatibility
    def predict_satisfaction_sync(self, conversation_features: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for the async method"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.predict_satisfaction(conversation_features))
        finally:
            loop.close()


class FeatureExtractor:
    """Extracts conversation features for satisfaction prediction"""
    
    def extract_features(self, conversation: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract features from a conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            'turn_count': len(conversation),
            'avg_response_length': 0.0,
            'user_questions_answered': 0.0,
            'emotional_variety': 0.0,
            'topic_coherence': 0.0,
            'response_relevance': 0.0,
            'conversation_flow': 0.0
        }
        
        # Calculate average response length
        assistant_responses = [
            turn for turn in conversation 
            if turn.get('role') == 'assistant'
        ]
        
        if assistant_responses:
            lengths = [len(turn.get('content', '').split()) for turn in assistant_responses]
            features['avg_response_length'] = sum(lengths) / len(lengths)
        
        # Calculate user questions answered (simple heuristic)
        user_turns = [turn for turn in conversation if turn.get('role') == 'user']
        questions = [turn for turn in user_turns if turn.get('content', '').strip().endswith('?')]
        
        if questions:
            # Simple heuristic: assume questions are answered if followed by assistant response
            answered = 0
            for i, turn in enumerate(conversation):
                if (turn.get('role') == 'user' and 
                    turn.get('content', '').strip().endswith('?') and
                    i + 1 < len(conversation) and
                    conversation[i + 1].get('role') == 'assistant'):
                    answered += 1
            
            features['user_questions_answered'] = answered / len(questions)
        else:
            features['user_questions_answered'] = 1.0  # No questions to answer
        
        # Mock values for other features (in production, use more sophisticated analysis)
        features['emotional_variety'] = 0.7
        features['topic_coherence'] = 0.85
        features['response_relevance'] = 0.88
        features['conversation_flow'] = 0.82
        
        return features 