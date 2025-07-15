import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ProgressiveRefiner:
    """Progressively refines dataset samples using LLM feedback."""
    
    def __init__(self, client, character_profile):
        self.client = client
        self.character_profile = character_profile
    
    async def refine_sample(self, sample: Dict[str, Any], 
                          iteration: int = 1, 
                          max_iterations: int = 2) -> Dict[str, Any]:
        """Progressively refine a sample through multiple iterations."""
        if iteration > max_iterations:
            return sample
            
        try:
            # Extract the assistant's response
            assistant_message = None
            for msg in sample.get('messages', []):
                if msg.get('role') == 'assistant':
                    assistant_message = msg.get('content', '')
                    break
            
            if not assistant_message:
                logger.warning("No assistant message found in sample")
                return sample
            
            # Create refinement prompt
            refinement_prompt = f"""
            You are helping to improve a roleplay response. Here's the current response:
            
            "{assistant_message}"
            
            Character Profile: {self.character_profile}
            
            Please improve this response by:
            1. Making it more character-consistent
            2. Improving the writing quality
            3. Adding more personality and depth
            4. Ensuring it flows naturally
            
            Return only the improved response, nothing else.
            """
            
            refined_response = await self.client.generate(
                refinement_prompt,
                temperature=0.7,
                max_tokens=1024
            )
            
            # Update the sample with refined response
            refined_sample = sample.copy()
            for msg in refined_sample.get('messages', []):
                if msg.get('role') == 'assistant':
                    msg['content'] = refined_response.strip()
                    break
            
            # Recursively refine if more iterations needed
            if iteration < max_iterations:
                return await self.refine_sample(refined_sample, iteration + 1, max_iterations)
            
            return refined_sample
            
        except Exception as e:
            logger.error(f"Error refining sample: {e}")
            return sample


class EnhancedQualityFilter:
    """Enhanced quality filtering for dataset samples."""
    
    def __init__(self, character_profile):
        self.character_profile = character_profile
    
    def filter_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Filter and score a sample based on quality metrics."""
        try:
            # Extract messages
            messages = sample.get('messages', [])
            if len(messages) < 2:
                return {
                    'sample': sample,
                    'quality_score': 0.0,
                    'passed': False,
                    'reasons': ['Insufficient messages']
                }
            
            user_msg = None
            assistant_msg = None
            
            for msg in messages:
                if msg.get('role') == 'user':
                    user_msg = msg.get('content', '')
                elif msg.get('role') == 'assistant':
                    assistant_msg = msg.get('content', '')
            
            if not user_msg or not assistant_msg:
                return {
                    'sample': sample,
                    'quality_score': 0.0,
                    'passed': False,
                    'reasons': ['Missing user or assistant message']
                }
            
            # Quality scoring
            scores = []
            reasons = []
            
            # Length check
            if len(assistant_msg.strip()) < 10:
                reasons.append('Response too short')
                scores.append(0.2)
            elif len(assistant_msg.strip()) > 2000:
                reasons.append('Response too long')
                scores.append(0.7)
            else:
                scores.append(1.0)
            
            # Basic content quality
            if assistant_msg.strip().lower() in ['i don\'t know', 'i can\'t help', 'sorry']:
                reasons.append('Generic/unhelpful response')
                scores.append(0.1)
            else:
                scores.append(0.8)
            
            # Check for repetition
            words = assistant_msg.lower().split()
            if len(words) > 10:
                unique_words = set(words)
                repetition_ratio = len(unique_words) / len(words)
                if repetition_ratio < 0.5:
                    reasons.append('High repetition detected')
                    scores.append(0.3)
                else:
                    scores.append(0.9)
            else:
                scores.append(0.8)
            
            # Check for meta commentary
            meta_phrases = ['as an ai', 'i am an ai', 'i cannot', 'i\'m not able to']
            has_meta = any(phrase in assistant_msg.lower() for phrase in meta_phrases)
            if has_meta:
                reasons.append('Contains meta commentary')
                scores.append(0.2)
            else:
                scores.append(1.0)
            
            # Calculate overall quality score
            quality_score = sum(scores) / len(scores) if scores else 0.0
            passed = quality_score >= 0.6 and len(reasons) == 0
            
            return {
                'sample': sample,
                'quality_score': quality_score,
                'passed': passed,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Error filtering sample: {e}")
            return {
                'sample': sample,
                'quality_score': 0.0,
                'passed': False,
                'reasons': [f'Filter error: {str(e)}']
            }