import json
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from .content_evaluation import is_nsfw_content, categorize_nsfw_style, evaluate_nsfw_quality
from .prompt_generators import calculate_prompt_similarity

logger = logging.getLogger(__name__)


async def generate_with_quality_curation(
    client,
    character: Dict[str, Any],
    save_dataset_func: Callable,
    final_dataset_size: int = 1000,
    raw_samples_target: int = 3000,
    quality_threshold: float = 6.5,
    diversity_weight: float = 0.3,
    custom_system_prompt: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    stage_callback: Optional[Callable] = None
) -> List[Dict[str, Any]]:
    """
    Orchestrates a process of mass generation followed by quality evaluation using LLM-as-judge and curation.
    
    This method generates a large number of samples, evaluates them for quality,
    and then curates the best ones while maintaining diversity.
    
    Args:
        client: The OpenAI client for generation
        character: Character dictionary
        save_dataset_func: Function to save the dataset
        final_dataset_size: Target size for final curated dataset
        raw_samples_target: Number of raw samples to generate before curation
        quality_threshold: Minimum quality score for samples to be included
        diversity_weight: Weight for diversity in curation (0.0 = quality only, 1.0 = diversity only)
        custom_system_prompt: Optional custom system prompt to replace temporal prompts
        progress_callback: Optional callback for progress updates
        stage_callback: Optional callback for stage updates
    
    Returns:
        List of curated high-quality samples
    """
    char_name = character.get('name', 'Assistant')
    logger.info(f"🎯 Starting quality curation for {char_name}: {raw_samples_target} → {final_dataset_size} samples")
    
    # Create temporary directory for large dataset chunks
    temp_dir = tempfile.mkdtemp(prefix=f"dataset_curation_{char_name}_")
    logger.info(f"📁 Using temporary directory: {temp_dir}")
    
    try:
        # ================== STAGE 1: Mass Generation ==================
        if stage_callback:
            stage_callback({'stage': 'generation', 'message': f'Generating {raw_samples_target} raw samples...'})
        
        # Generate raw samples in chunks to manage memory
        chunk_size = 500
        all_samples = []
        
        for chunk_start in range(0, raw_samples_target, chunk_size):
            chunk_end = min(chunk_start + chunk_size, raw_samples_target)
            chunk_target = chunk_end - chunk_start
            
            logger.info(f"📝 Generating chunk {chunk_start//chunk_size + 1}: samples {chunk_start+1}-{chunk_end}")
            
            # Generate chunk (this would call the main generation method)
            # For now, we'll assume this is handled by the calling code
            # chunk_samples = await self.generate_samples(character, chunk_target)
            # all_samples.extend(chunk_samples)
            
            # Save chunk to temporary file
            chunk_file = os.path.join(temp_dir, f"chunk_{chunk_start//chunk_size}.json")
            # with open(chunk_file, 'w') as f:
            #     json.dump(chunk_samples, f)
            
            if progress_callback:
                progress_callback((chunk_end / raw_samples_target) * 0.5)  # First 50% for generation
        
        logger.info(f"✅ Generated {len(all_samples)} raw samples")
        
        # ================== STAGE 2: Quality Evaluation ==================
        if stage_callback:
            stage_callback({'stage': 'evaluation', 'message': f'Evaluating {len(all_samples)} samples for quality...'})
        
        evaluated_samples = []
        batch_size = 20  # Process in smaller batches for evaluation
        total_evaluated = 0
        
        # Process samples in batches
        for i in range(0, len(all_samples), batch_size):
            batch = all_samples[i:i + batch_size]
            
            # Judge this batch
            batch_scores = await judge_sample_batch(
                client=client,
                samples=batch,
                character=character,
            )
            
            # Store samples with their scores
            for sample, scores in zip(batch, batch_scores):
                if scores['overall_score'] >= quality_threshold:
                    evaluated_samples.append({
                        'sample': sample,
                        'scores': scores,
                        'user_prompt': sample['messages'][1]['content'],
                        'response': sample['messages'][2]['content']
                    })
            
            total_evaluated += len(batch)
            if progress_callback:
                progress_callback(0.5 + (total_evaluated / len(all_samples)) * 0.3)  # 50-80% for evaluation
            
            # Log acceptance rate
            if total_evaluated % 500 == 0:
                acceptance_rate = len(evaluated_samples) / total_evaluated * 100
                logger.info(f"📈 Evaluated: {total_evaluated}, Accepted: {len(evaluated_samples)} ({acceptance_rate:.1f}%)")
        
        logger.info(f"✅ Evaluation complete: {len(evaluated_samples)}/{len(all_samples)} samples passed quality threshold")
        
        # ================== STAGE 3: Diversity-Aware Curation ==================
        if stage_callback:
            stage_callback({'stage': 'curation', 'message': f'Curating {final_dataset_size} best samples with diversity...'})
        
        # If we have fewer samples than target, just return what we have
        if len(evaluated_samples) <= final_dataset_size:
            logger.warning(f"⚠️ Only {len(evaluated_samples)} samples passed quality threshold")
            curated_samples = [s['sample'] for s in evaluated_samples]
        else:
            # Curate with diversity
            curated_samples = curate_diverse_samples(
                evaluated_samples=evaluated_samples,
                target_size=final_dataset_size,
                diversity_weight=diversity_weight
            )
        
        logger.info(f"🎉 Curation complete: {len(curated_samples)} high-quality diverse samples")
        
        # If custom system prompt is provided, replace all temporal prompts with it
        if custom_system_prompt is not None:
            if custom_system_prompt == "":
                # Empty string means remove system prompts entirely
                logger.info(f"🔄 Removing all system prompts from dataset (empty custom prompt)")
                for sample in curated_samples:
                    if 'messages' in sample and len(sample['messages']) > 0 and sample['messages'][0].get('role') == 'system':
                        # Remove the system message
                        sample['messages'].pop(0)
            else:
                # Replace with the custom prompt
                logger.info(f"🔄 Replacing temporal system prompts with custom prompt for training consistency")
                for sample in curated_samples:
                    if 'messages' in sample and len(sample['messages']) > 0:
                        # Replace the system prompt with the custom one
                        sample['messages'][0]['content'] = custom_system_prompt
        
        # Save the curated dataset with metadata
        metadata = {}
        if custom_system_prompt is not None:
            if custom_system_prompt == "":
                metadata['system_prompt_config'] = {
                    'type': 'none',
                    'prompt': ''
                }
            else:
                metadata['system_prompt_config'] = {
                    'type': 'custom',
                    'prompt': custom_system_prompt
                }
        else:
            metadata['system_prompt_config'] = {
                'type': 'temporal',
                'prompt': None
            }
        
        save_dataset_func(character, curated_samples, metadata)
        
        if progress_callback:
            progress_callback(1.0)  # 100% complete
        
        return curated_samples
        
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info("🧹 Cleaned up temporary files")


async def judge_sample_batch(
    client,
    samples: List[Dict[str, Any]],
    character: Dict[str, Any],
) -> List[Dict[str, float]]:
    """Judge a batch of samples for quality using LLM-as-judge.
    
    Returns a list of score dictionaries for each sample.
    """
    char_name = character.get('name', 'Assistant')
    
    # Prepare judgment prompts
    judgment_prompts = []
    nsfw_indices = []  # Track which samples are NSFW
    batch_scores = [None] * len(samples)  # Pre-allocate list for scores

    for i, sample in enumerate(samples):
        user_msg = sample['messages'][1]['content']
        assistant_msg = sample['messages'][2]['content']
        
        # Check if this is NSFW content
        if is_nsfw_content(sample):
            nsfw_indices.append(i)
            # Use specialized NSFW evaluation
            scores = await evaluate_nsfw_quality(
                client, assistant_msg, character, user_msg
            )
            # Store the scores directly
            batch_scores[i] = scores
        else:
            # Create standard judgment prompt
            judgment_prompt = f"""You are evaluating roleplay responses for quality and character consistency.

Character Name: {char_name}
Character Description: {character.get('description', 'No description')[:500]}
Character Personality: {character.get('personality', 'No personality')[:300]}

User Question: {user_msg}
Character Response: {assistant_msg}

Evaluate this response on the following criteria (0-10 scale):

1. Character Consistency: Does this response match the character's established personality, mannerisms, and knowledge?
2. Narrative Coherence: Does this response make sense within the character's story and maintain internal consistency?
3. Response Quality: Is this engaging, detailed, and appropriately addresses the user's question?
4. Uniqueness: Does this response feel specific to this character rather than generic?
5. Emotional Authenticity: Does the emotional tone match what we'd expect from this character in this situation?

Respond with ONLY a JSON object with numeric scores:
{{"character_consistency": 8, "narrative_coherence": 9, "response_quality": 7, "uniqueness": 8, "emotional_authenticity": 9}}"""
            
            judgment_prompts.append(judgment_prompt)
    
    # Process non-NSFW samples in batch
    prompts = [p for p in judgment_prompts if p is not None]
    non_nsfw_indices = [i for i, sample in enumerate(samples) if not is_nsfw_content(sample)]
    
    if prompts:
        try:
            responses = await client.generate_batch(
                prompts=prompts,
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistent judging
                top_p=0.95
            )
            
            # Parse scores from responses
            parsed_scores = []
            for response in responses:
                scores = parse_judgment_scores(response)
                parsed_scores.append(scores)

            # Place scores in correct positions
            for original_idx, scores in zip(non_nsfw_indices, parsed_scores):
                batch_scores[original_idx] = scores

        except Exception as e:
            logger.error(f"Error in batch judgment: {e}")
            parsed_scores = [get_default_scores() for _ in prompts]
            for original_idx, scores in zip(non_nsfw_indices, parsed_scores):
                batch_scores[original_idx] = scores
    
    # Combine results, inserting NSFW evaluations where appropriate
    return [score or get_default_scores() for score in batch_scores]


def parse_judgment_scores(response: str) -> Dict[str, float]:
    """Parse quality scores from judge response."""
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            scores_dict = json.loads(json_match.group())
            
            # Calculate overall score
            score_values = [
                scores_dict.get('character_consistency', 5),
                scores_dict.get('narrative_coherence', 5),
                scores_dict.get('response_quality', 5),
                scores_dict.get('uniqueness', 5),
                scores_dict.get('emotional_authenticity', 5)
            ]
            
            # Weighted average (character consistency and narrative coherence weighted higher)
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]
            overall_score = sum(s * w for s, w in zip(score_values, weights))
            
            return {
                'character_consistency': scores_dict.get('character_consistency', 5),
                'narrative_coherence': scores_dict.get('narrative_coherence', 5),
                'response_quality': scores_dict.get('response_quality', 5),
                'uniqueness': scores_dict.get('uniqueness', 5),
                'emotional_authenticity': scores_dict.get('emotional_authenticity', 5),
                'overall_score': overall_score
            }
        else:
            return get_default_scores()
            
    except Exception as e:
        logger.debug(f"Failed to parse judgment scores: {e}")
        return get_default_scores()


def get_default_scores() -> Dict[str, float]:
    """Return neutral default scores."""
    return {
        'character_consistency': 5.0,
        'narrative_coherence': 5.0,
        'response_quality': 5.0,
        'uniqueness': 5.0,
        'emotional_authenticity': 5.0,
        'overall_score': 5.0
    }


def ensure_nsfw_diversity(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure variety in intimate content styles"""
    
    nsfw_categories = {
        'romantic': [],
        'playful': [],
        'passionate': [],
        'emotional': [],
        'sensual': [],
        'non_nsfw': []
    }
    
    # Categorize all samples
    for sample in samples:
        category = categorize_nsfw_style(sample['sample'])
        nsfw_categories[category].append(sample)
    
    # Calculate target distribution
    total_nsfw = sum(len(nsfw_categories[cat]) for cat in nsfw_categories if cat != 'non_nsfw')
    total_samples = len(samples)
    nsfw_ratio = total_nsfw / total_samples if total_samples > 0 else 0
    
    # Build balanced selection
    balanced_samples = []
    
    # First add non-NSFW samples
    non_nsfw_count = int((1 - nsfw_ratio) * len(samples))
    balanced_samples.extend(nsfw_categories['non_nsfw'][:non_nsfw_count])
    
    # Then distribute NSFW samples across categories
    nsfw_per_category = max(1, (len(samples) - len(balanced_samples)) // 5)
    
    for category in ['romantic', 'playful', 'passionate', 'emotional', 'sensual']:
        category_samples = nsfw_categories[category]
        if category_samples:
            # Sort by quality within category
            category_samples.sort(key=lambda x: x['scores']['overall_score'], reverse=True)
            # Take top samples from category
            balanced_samples.extend(category_samples[:nsfw_per_category])
    
    # Fill any remaining slots with highest quality samples
    remaining_needed = len(samples) - len(balanced_samples)
    if remaining_needed > 0:
        all_remaining = []
        for cat, cat_samples in nsfw_categories.items():
            all_remaining.extend(s for s in cat_samples if s not in balanced_samples)
        
        all_remaining.sort(key=lambda x: x['scores']['overall_score'], reverse=True)
        balanced_samples.extend(all_remaining[:remaining_needed])
    
    logger.info(f"🌈 NSFW diversity: {len(balanced_samples)} samples across {sum(1 for cat in nsfw_categories.values() if cat)} categories")
    
    return balanced_samples


def curate_diverse_samples(
    evaluated_samples: List[Dict[str, Any]],
    target_size: int,
    diversity_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """Curate a diverse set of high-quality samples.
    
    Uses a combination of quality scores and diversity metrics to select
    the best samples while maintaining variety.
    """
    # First ensure NSFW diversity if applicable
    if evaluated_samples and 'sample' in evaluated_samples[0]:
        first_sample = evaluated_samples[0]['sample']
        if 'messages' in first_sample and len(first_sample['messages']) > 0:
            # Try to detect if this character has romantic/intimate traits
            system_msg = first_sample['messages'][0].get('content', '').lower()
            if any(word in system_msg for word in ['romantic', 'lover', 'passionate', 'sensual', 'intimate']):
                logger.info("💕 Applying NSFW diversity curation")
                evaluated_samples = ensure_nsfw_diversity(evaluated_samples)
    
    # Sort by quality score first
    sorted_samples = sorted(evaluated_samples, key=lambda x: x['scores']['overall_score'], reverse=True)
    
    # If diversity weight is 0, just return top samples
    if diversity_weight == 0:
        return [s['sample'] for s in sorted_samples[:target_size]]
    
    # Otherwise, use diversity-aware selection
    selected_samples = []
    selected_prompts = set()
    selected_responses = set()
    
    # Group samples by prompt similarity
    prompt_groups = defaultdict(list)
    for sample in sorted_samples:
        # Simple grouping by first few words of prompt
        prompt_key = ' '.join(sample['user_prompt'].lower().split()[:3])
        prompt_groups[prompt_key].append(sample)
    
    # First pass: Take best from each prompt group
    for prompt_key, group_samples in prompt_groups.items():
        if len(selected_samples) >= target_size:
            break
        
        # Take the best sample from this group
        best_sample = group_samples[0]
        selected_samples.append(best_sample['sample'])
        selected_prompts.add(best_sample['user_prompt'])
        selected_responses.add(best_sample['response'][:100])  # First 100 chars for similarity
    
    # Second pass: Fill remaining slots with high-quality diverse samples
    for sample in sorted_samples:
        if len(selected_samples) >= target_size:
            break
        
        # Skip if already selected
        if sample['sample'] in selected_samples:
            continue
        
        # Check diversity
        prompt_similarity = any(
            calculate_prompt_similarity(sample['user_prompt'], p) > 0.8 
            for p in selected_prompts
        )
        
        response_similarity = any(
            sample['response'][:100] == r 
            for r in selected_responses
        )
        
        # Add if diverse enough
        if not (prompt_similarity and response_similarity):
            selected_samples.append(sample['sample'])
            selected_prompts.add(sample['user_prompt'])
            selected_responses.add(sample['response'][:100])
    
    logger.info(f"📊 Curated {len(selected_samples)} samples with diversity weight {diversity_weight}")
    
    return selected_samples