"""
Basic Generation Evaluation

Uses LLM-as-judge with structured outputs to test that the model can generate coherent text in response to prompts.
Focuses on catching obvious failures like empty outputs or repetitive text, but uses LLM judgment for quality assessment.
"""

import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import asyncio
import json
from pydantic import BaseModel, Field

from backend.app.core.openai_client import get_client

logger = logging.getLogger(__name__)


class ResponseQualityAnalysis(BaseModel):
    """Structured output for response quality analysis"""
    coherence_score: float = Field(ge=0.0, le=1.0, description="How coherent and logical the response is")
    relevance_score: float = Field(ge=0.0, le=1.0, description="How relevant the response is to the prompt")
    creativity_score: float = Field(ge=0.0, le=1.0, description="Level of creativity and originality")
    language_quality: float = Field(ge=0.0, le=1.0, description="Grammar, vocabulary, and language quality")
    completion_score: float = Field(ge=0.0, le=1.0, description="How complete and satisfying the response is")
    
    quality_issues: List[str] = Field(description="Specific quality issues identified")
    strengths: List[str] = Field(description="Notable strengths of the response")
    overall_assessment: str = Field(description="Overall assessment of response quality")


class RepetitionAnalysis(BaseModel):
    """Structured output for repetition detection"""
    repetition_score: float = Field(ge=0.0, le=1.0, description="Level of repetition (0=none, 1=highly repetitive)")
    repetitive_patterns: List[str] = Field(description="Specific repetitive patterns found")
    variety_score: float = Field(ge=0.0, le=1.0, description="Lexical and structural variety score")
    assessment: str = Field(description="Assessment of repetition and variety")


class BasicGenerationEvaluator:
    """Evaluates basic text generation capabilities using LLM-as-judge"""
    
    def __init__(self):
        self.client = get_client()
        self.test_prompts = [
            "Tell me a story about a brave knight.",
            "What is your favorite color and why?",
            "Describe a peaceful day in the countryside.",
            "Explain how to make a simple sandwich.",
            "Write a haiku about the ocean.",
        ]
        
    def evaluate(
        self, 
        model: Any, 
        tokenizer: Any,
        test_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate basic generation capabilities using LLM-as-judge.
        
        Args:
            model: The model to evaluate
            tokenizer: Tokenizer for the model
            test_prompts: Optional custom test prompts
            
        Returns:
            Dictionary with evaluation metrics
        """
        if test_prompts is None:
            test_prompts = self.test_prompts
            
        results = {
            'coherence_score': 0.0,
            'generation_success_rate': 0.0,
            'avg_response_length': 0.0,
            'individual_scores': []
        }
        
        try:
            successful_generations = 0
            total_length = 0
            coherence_scores = []
            
            for prompt in test_prompts:
                # Generate response
                response = self._generate_response(model, tokenizer, prompt)
                
                # Evaluate response using LLM-as-judge
                eval_result = asyncio.run(self._evaluate_single_response_async(response, prompt))
                results['individual_scores'].append(eval_result)
                
                if eval_result['success']:
                    successful_generations += 1
                    total_length += eval_result['length']
                    coherence_scores.append(eval_result['coherence'])
                    
            # Calculate aggregate metrics
            num_prompts = len(test_prompts)
            results['generation_success_rate'] = successful_generations / num_prompts if num_prompts > 0 else 0
            results['avg_response_length'] = total_length / successful_generations if successful_generations > 0 else 0
            results['coherence_score'] = np.mean(coherence_scores) if coherence_scores else 0.0
            
            logger.info(f"Basic generation evaluation complete: {successful_generations}/{num_prompts} successful")
            
        except Exception as e:
            logger.error(f"Error in basic generation evaluation: {e}")
            results['error'] = str(e)
            
        return results
    
    def _generate_response(self, model: Any, tokenizer: Any, prompt: str) -> str:
        """Generate a response for a single prompt"""
        try:
            # This is simplified - actual implementation would handle batching, device placement, etc.
            if hasattr(model, 'generate'):
                # Encode prompt
                inputs = tokenizer.encode(prompt, return_tensors='pt') if hasattr(tokenizer, 'encode') else torch.tensor([[1, 2, 3]])
                
                # Generate with reasonable parameters
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
                    )
                
                # Decode response
                if hasattr(tokenizer, 'decode'):
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Remove the prompt from response
                    response = response[len(prompt):].strip()
                else:
                    response = "This is a coherent response to the prompt."
                    
                return response
            else:
                # Mock response for testing
                return "This is a coherent response to the prompt."
                
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""
    
    async def _evaluate_single_response_async(self, response: str, prompt: str) -> Dict[str, Any]:
        """Evaluate a single generated response using LLM-as-judge"""
        result = {
            'prompt': prompt,
            'response': response,
            'success': False,
            'length': 0,
            'coherence': 0.0,
            'issues': []
        }
        
        # Check if response is empty
        if not response or len(response.strip()) == 0:
            result['issues'].append('Empty response')
            return result
            
        # Check response length
        result['length'] = len(response.split())
        if result['length'] < 3:
            result['issues'].append('Response too short')
            return result
        
        try:
            # Use LLM-as-judge for quality assessment
            quality_analysis = await self._judge_response_quality(response, prompt)
            
            # Use LLM-as-judge for repetition analysis
            repetition_analysis = await self._judge_repetition(response)
            
            # Combine assessments
            result['coherence'] = quality_analysis.coherence_score
            result['relevance'] = quality_analysis.relevance_score
            result['creativity'] = quality_analysis.creativity_score
            result['language_quality'] = quality_analysis.language_quality
            result['completion_score'] = quality_analysis.completion_score
            result['quality_issues'] = quality_analysis.quality_issues
            result['strengths'] = quality_analysis.strengths
            result['overall_assessment'] = quality_analysis.overall_assessment
            
            result['repetition_score'] = repetition_analysis.repetition_score
            result['repetitive_patterns'] = repetition_analysis.repetitive_patterns
            result['variety_score'] = repetition_analysis.variety_score
            result['repetition_assessment'] = repetition_analysis.assessment
            
            # Check for common failure patterns
            failure_patterns = [
                '',  # Unicode errors
                '<|endoftext|>',  # Exposed special tokens
                'None',  # Null responses
                'Error:',  # Error messages
            ]
            
            for pattern in failure_patterns:
                if pattern in response:
                    result['issues'].append(f'Contains failure pattern: {pattern}')
                    result['coherence'] *= 0.5
            
            # Mark as successful if no critical issues and good quality
            if (result['length'] >= 3 and 
                result['coherence'] > 0.3 and 
                result['repetition_score'] < 0.7 and
                not any('failure pattern' in issue for issue in result['issues'])):
                result['success'] = True
                
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            # Fallback to heuristic analysis
            fallback_result = self._evaluate_single_response_fallback(response, prompt)
            result.update(fallback_result)
            
        return result
    
    async def _judge_response_quality(self, response: str, prompt: str) -> ResponseQualityAnalysis:
        """Use LLM-as-judge to assess response quality"""
        try:
            system_prompt = """You are an expert in text generation quality assessment. Evaluate the quality of AI-generated responses across multiple dimensions.

Assess the response for:
1. **Coherence**: Logical flow and internal consistency
2. **Relevance**: How well it addresses the prompt
3. **Creativity**: Originality and creative expression
4. **Language Quality**: Grammar, vocabulary, style
5. **Completion**: How complete and satisfying the response is

**SCORING GUIDELINES:**
- 0.9-1.0: Excellent quality, professional level
- 0.7-0.8: Good quality with minor issues
- 0.5-0.6: Acceptable quality but noticeable problems
- 0.3-0.4: Poor quality with significant issues
- 0.0-0.2: Very poor quality, barely functional

**CRITICAL**: All numeric scores must be decimal values between 0.0 and 1.0 (inclusive).

Provide specific feedback on strengths and issues."""

            user_prompt = f"""Evaluate this AI-generated response:

**Prompt:** {prompt}

**Response:** {response}

Assess the response across all quality dimensions and provide:
1. Scores for coherence, relevance, creativity, language quality, and completion
2. Specific quality issues (if any)
3. Notable strengths
4. Overall assessment

Be specific and constructive in your feedback."""

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
                        "name": "response_quality_analysis",
                        "schema": ResponseQualityAnalysis.model_json_schema()
                    }
                }
            )
            
            analysis_data = json.loads(response_text)
            return ResponseQualityAnalysis(**analysis_data)
            
        except Exception as e:
            logger.error(f"Error in quality judgment: {e}")
            # Return fallback assessment
            return ResponseQualityAnalysis(
                coherence_score=0.5,
                relevance_score=0.5,
                creativity_score=0.5,
                language_quality=0.5,
                completion_score=0.5,
                quality_issues=["Could not assess with LLM judge"],
                strengths=["Basic response generated"],
                overall_assessment="Fallback assessment due to LLM error"
            )
    
    async def _judge_repetition(self, response: str) -> RepetitionAnalysis:
        """Use LLM-as-judge to assess repetition and variety"""
        try:
            system_prompt = """You are an expert in text analysis, specifically detecting repetition and assessing linguistic variety. Analyze the given text for repetitive patterns and lexical diversity.

Evaluate:
1. **Repetition**: Repeated words, phrases, or structural patterns
2. **Variety**: Lexical diversity and structural variation
3. **Pattern Detection**: Specific repetitive elements

**SCORING GUIDELINES:**
- Repetition Score: 0.0 = no repetition, 1.0 = highly repetitive
- Variety Score: 0.0 = no variety, 1.0 = high variety

**CRITICAL**: All numeric scores must be decimal values between 0.0 and 1.0 (inclusive).

Identify specific repetitive patterns with examples."""

            user_prompt = f"""Analyze this text for repetition and variety:

**Text:** {response}

Assess:
1. Level of repetition (words, phrases, structures)
2. Lexical and structural variety
3. Specific repetitive patterns found
4. Overall assessment of text diversity

Provide specific examples of any repetitive patterns you find."""

            response_text = await self.client.chat_complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400,
                temperature=0.3,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "repetition_analysis",
                        "schema": RepetitionAnalysis.model_json_schema()
                    }
                }
            )
            
            analysis_data = json.loads(response_text)
            return RepetitionAnalysis(**analysis_data)
            
        except Exception as e:
            logger.error(f"Error in repetition judgment: {e}")
            # Return fallback assessment
            fallback_score = self._check_repetition_fallback(response)
            return RepetitionAnalysis(
                repetition_score=fallback_score,
                repetitive_patterns=["Could not assess with LLM judge"],
                variety_score=1.0 - fallback_score,
                assessment="Fallback assessment due to LLM error"
            )
    
    def _evaluate_single_response_fallback(self, response: str, prompt: str) -> Dict[str, Any]:
        """Fallback heuristic evaluation when LLM calls fail"""
        result = {}
        
        # Simple coherence check based on length and basic patterns
        repetition_score = self._check_repetition_fallback(response)
        result['coherence'] = 1.0 - repetition_score
        result['repetition_score'] = repetition_score
        
        # Basic relevance check (very simple)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words)
        result['relevance'] = min(1.0, overlap / len(prompt_words)) if prompt_words else 0.5
        
        # Default scores for other metrics
        result['creativity'] = 0.5
        result['language_quality'] = 0.7 if len(response) > 10 else 0.3
        result['completion_score'] = 0.6
        result['quality_issues'] = ["Assessed using fallback heuristics"]
        result['strengths'] = ["Response generated successfully"]
        result['overall_assessment'] = "Fallback heuristic assessment"
        result['variety_score'] = 1.0 - repetition_score
        result['repetitive_patterns'] = []
        result['repetition_assessment'] = "Heuristic repetition check"
        
        return result
    
    def _check_repetition_fallback(self, text: str) -> float:
        """
        Check for repetitive patterns in text using heuristics.
        Returns a score from 0 (no repetition) to 1 (highly repetitive).
        """
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
            
        # Check for repeated sequences
        repetition_count = 0
        for i in range(len(words) - 3):
            sequence = tuple(words[i:i+3])
            # Count how many times this 3-word sequence appears
            occurrences = sum(1 for j in range(len(words) - 2) 
                            if tuple(words[j:j+3]) == sequence)
            if occurrences > 2:
                repetition_count += occurrences - 1
                
        # Normalize by text length
        repetition_score = repetition_count / len(words)
        return min(repetition_score, 1.0) 