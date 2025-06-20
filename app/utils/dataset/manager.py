"""DatasetManager façade that maintains backward compatibility.

This module provides a DatasetManager class that inherits from NSFWGenerationManager
to maintain backward compatibility while utilizing the new generation package structure.
"""

import asyncio
import logging
import os
import random
import re
import warnings
import gc
from typing import Any, Callable, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None

from ..generation import NSFWGenerationManager
from .models import GenerationConfig, QualityLevel
from . import character_analysis
from . import prompt_generators
from . import quality_curation

logger = logging.getLogger(__name__)


class DatasetManager(NSFWGenerationManager):
    """Manages synthetic dataset generation and processing using OpenAI API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize DatasetManager with enhanced client and configuration

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (defaults to OpenAI, but can be changed for compatible endpoints)
            generation_config: Enhanced generation configuration
        """
        # Initialize parent NSFWGenerationManager which handles client setup
        super().__init__(api_key=api_key, base_url=base_url, generation_config=generation_config)
        logger.info(f"DatasetManager façade initialized with model: {os.getenv('MODEL_NAME')}")

        # Initialize default prompts for backwards compatibility
        self.default_user_prompts = [
            "Tell me about yourself.",
            "What's your story?",
            "How are you feeling today?",
            "What's on your mind?",
            "What do you think about that?",
            "Can you help me with something?",
            "What's your opinion on this?",
            "How would you handle this situation?",
            "What's your perspective?",
            "Tell me more about that.",
        ]

        # Templates for backwards compatibility
        self.templates = [
            "short_qa",
            "narration", 
            "monologue",
            "dialogue_turn",
            "internal_thought",
            "character_response"
        ]

    async def generate_fast_templated_dataset(
        self,
        character: Dict[str, Any],
        num_samples: int = 50,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset quickly using character-aware prompt templates.

        This 'fast mode' is designed for rapid dataset creation without the
        more advanced (and slower) quality curation and refinement steps.
        """
        logger.info(f"🚀 Starting fast-templated dataset generation for {character.get('name', 'Unknown')}")
        self._setup_character_components(character)

        # 1. Generate a diverse set of prompts tailored to the character
        try:
            exploration_prompts = await prompt_generators.generate_exploration_prompts(
                self.client, character, num_prompts=num_samples * 2  # Generate more to ensure variety
            )
            unique_prompts = prompt_generators.deduplicate_prompts(exploration_prompts)
            if len(unique_prompts) < num_samples:
                logger.warning(f"Generated {len(unique_prompts)} unique prompts, less than the {num_samples} requested.")
                if not unique_prompts:
                    return [] # Return empty if no prompts could be generated
            
            prompts_to_use = random.sample(unique_prompts, min(num_samples, len(unique_prompts)))
            logger.info(f"🧠 Generated {len(prompts_to_use)} unique prompts for generation.")

        except Exception as e:
            logger.error(f"Failed to generate prompts: {e}", exc_info=True)
            return []

        # 2. Create system prompt
        system_prompt = self._create_system_prompt(character)

        # Get generation params from kwargs or use defaults
        temperature = kwargs.get('temperature', 0.8)
        max_tokens = kwargs.get('max_tokens', 800)

        # 3. Generate samples in parallel
        dataset = []
        tasks = []
        for user_prompt in prompts_to_use:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            task = self.client.chat_complete(messages=messages, max_tokens=max_tokens, temperature=temperature, top_p=0.9)
            tasks.append((task, messages)) # Keep messages to pair with result

        for i, (task, messages) in enumerate(tasks):
            try:
                response_content = await task
                if response_content:
                    final_messages = messages + [{"role": "assistant", "content": response_content}]
                    dataset.append({"messages": final_messages})
                else:
                    logger.warning("Generation resulted in an empty response.")
                
                if progress_callback:
                    progress_callback((i + 1) / len(prompts_to_use))

            except Exception as e:
                logger.error(f"Error during sample generation: {e}", exc_info=True)

        logger.info(f"✅ Fast-templated generation complete. Produced {len(dataset)} samples.")
        return dataset

    async def generate_dataset(self, character: Dict[str, Any], num_samples: int = 80,
                               max_tokens: Optional[int] = None, temperature: float = 0.8,
                               top_p: float = 0.9, progress_callback: Optional[Callable] = None,
                               append_to_existing: bool = True, custom_system_prompt: Optional[str] = None,
                               extra_quality: bool = False, quality_level: QualityLevel = QualityLevel.COMPREHENSIVE,
                               **sampling_kwargs) -> List[Dict[str, Any]]:
        """Generate synthetic dataset for character using efficient batching"""
        # ✅ FIX: Better error handling to prevent silent crashes
        try:
            # Suppress coroutine warnings in Streamlit environment
            warnings.filterwarnings(
                "ignore", message="coroutine.*was never awaited")

            # Setup character-specific components
            self._setup_character_components(character)

            # Extract max_tokens from sampling_kwargs if provided there instead
            if max_tokens is None and 'max_tokens' in sampling_kwargs:
                max_tokens = sampling_kwargs.pop('max_tokens')
            
            # Use default if still None
            if max_tokens is None:
                max_tokens = 400  # Reasonable default

            # Force garbage collection to clean up memory
            gc.collect()

            # Clear CUDA cache if available
            try:
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as cuda_error:
                logger.warning(f"⚠️ CUDA cache clear failed: {cuda_error}")

            card_block = self._make_card_block(character)
            
            # Enhanced quality-based generation strategy
            if quality_level == QualityLevel.COMPREHENSIVE:
                logger.info("🌟 Using COMPREHENSIVE quality generation with progressive refinement")
                return await self._generate_premium_dataset(
                    character, num_samples, max_tokens, temperature, top_p,
                    progress_callback, append_to_existing, custom_system_prompt,
                    extra_quality, **sampling_kwargs
                )
            elif quality_level == QualityLevel.ITERATIVE:
                logger.info("⭐ Using ITERATIVE quality generation with real-time filtering")
                # Continue with enhanced standard generation (below)
            else:
                logger.info("📝 Using FAST quality generation")
                # Disable advanced features for fast mode
                self.generation_config.enable_real_time_filtering = False

            # Load existing dataset if append_to_existing is True
            existing_samples = []
            if append_to_existing:
                existing_dataset = self.load_dataset(character)
                if existing_dataset:
                    existing_samples = existing_dataset
                    logger.info(
                        f"📂 Found existing dataset with {len(existing_samples)} samples")

            # Calculate how many new samples to generate
            existing_count = len(existing_samples)
            if existing_count >= num_samples:
                logger.info(
                    f"✅ Dataset already has {existing_count} samples (requested: {num_samples})")
                return existing_samples[:num_samples]  # Return requested amount

            new_samples_needed = num_samples - existing_count
            logger.info(
                f"🎯 Generating {new_samples_needed} new samples to reach {num_samples} total")

            samples = existing_samples.copy()

            # Extract existing prompts to avoid duplication
            seen_user_prompts: set[str] = {
                sample['messages'][1]['content'] for sample in existing_samples
                if isinstance(sample, dict) and 'messages' in sample and len(sample['messages']) > 1
            }

            # Generate diverse prompts using multiple strategies
            all_prompts = []
            
            # 1. Baseline prompts (ensure coverage) - convert to dict format
            baseline_prompts = [q for q in self.default_user_prompts if q not in seen_user_prompts]
            for prompt in baseline_prompts:
                all_prompts.append({
                    'prompt': prompt,
                    'type': 'baseline'
                })
            
            # 2. LLM-generated character-specific prompts
            try:
                logger.info("🧠 Generating LLM-tailored questions...")
                num_llm_questions = min(new_samples_needed * 2, 100)
                
                llm_questions = await self.suggest_user_questions(
                    character=character,
                    num_questions=num_llm_questions,
                    temperature=0.9,
                    top_p=0.95,
                    existing_dataset=existing_samples,
                    context_samples=8
                )
                
                logger.info(f"✅ Generated {len(llm_questions)} LLM-tailored questions")
                
                # Add these high-quality questions to the prompt pool
                for question_data in llm_questions:
                    question_text = question_data['question']
                    if question_text not in seen_user_prompts:
                        all_prompts.append(question_text)
                        
            except Exception as e:
                logger.warning(f"⚠️ LLM question generation failed: {e}")

            character_knowledge = character_analysis.extract_character_knowledge(character)
            # Add a small selection of algorithmic prompts for diversity
            scenarios = await self.generate_scenario_based_prompts(character, character_knowledge, num_scenarios=3)
            for scenario in scenarios[:2]:  # Only use 2 scenarios
                for prompt in scenario['prompts'][:2]:  # Only 2 prompts per scenario
                    if prompt not in seen_user_prompts:
                        all_prompts.append({
                            'prompt': prompt,
                            'context': scenario['context'],
                            'type': 'scenario_supplement'
                        })
            # 3. Random prompts from buckets
            while len(all_prompts) < new_samples_needed * 2:
                prompt = await self._build_user_prompt()
                if prompt not in seen_user_prompts and prompt not in all_prompts:
                    all_prompts.append(prompt)

           # Generate multi-turn conversations for all modes
            multi_turn_convos = []
            # For LLM mode, create a few simple scenarios for multi-turn
            logger.info("💬 Generating multi-turn conversation flows...")
            character_knowledge = character_analysis.extract_character_knowledge(character)
            simple_scenarios = await self.generate_scenario_based_prompts(character, character_knowledge, num_scenarios=2)
            for scenario in simple_scenarios:
                convos = await self.generate_multi_turn_conversation(character, scenario, turns=2)  # 2 turns = 4 total messages
                multi_turn_convos.extend(convos)
            
            # Deduplicate prompts
            unique_prompts_map = {}
            for item in all_prompts:
                if isinstance(item, dict):
                    prompt_text = item.get('prompt')
                    if prompt_text and prompt_text not in unique_prompts_map:
                        unique_prompts_map[prompt_text] = item
                elif isinstance(item, str):
                    if item not in unique_prompts_map:
                        unique_prompts_map[item] = {'prompt': item, 'type': 'simple'}

            unique_prompts = list(unique_prompts_map.values())
            logger.info(f"📊 Generated {len(unique_prompts)} unique prompts")
            
            # Apply EXTRA QUALITY paraphrasing if requested
            if extra_quality:
                logger.info("🌟 EXTRA QUALITY enabled - paraphrasing prompts for enhanced variety...")
                paraphrased_prompts = []
                
                for i, prompt_item in enumerate(unique_prompts):
                    try:
                        prompt_text = prompt_item['prompt']
                        # Use simple paraphrasing prompt with OpenAI
                        paraphrase_prompt = f"Rewrite this question to mean the same thing but with different words. Keep the same tone and meaning, just vary the phrasing:\n\nOriginal: {prompt_text}\n\nRewritten:"
                        
                        paraphrased = await self._generate_single_response(
                            paraphrase_prompt,
                            max_tokens=1000,
                            temperature=0.6
                        )
                        
                        paraphrased_prompts.append(paraphrased.strip())
                        
                        if i % 10 == 0:
                            logger.info(f"   Paraphrased {i+1}/{len(unique_prompts)} prompts...")
                        
                    except Exception as e:
                        logger.debug(f"   Failed to paraphrase prompt {i+1}: {e}")
                        paraphrased_prompts.append(prompt_item['prompt'])
                
                # After paraphrasing, we have a list of strings. Convert back to dicts.
                unique_prompts = [{'prompt': p, 'type': 'paraphrased'} for p in paraphrased_prompts]
                logger.info(f"✅ EXTRA QUALITY complete - paraphrased {len(unique_prompts)} prompts")
            
            # Length buckets following best practices
            length_buckets = [
                ("short", 0.40, 200),
                ("medium", 0.45, 500),
                ("long", 0.15, 800),
            ]

            def _sample_max_tokens() -> int:
                names, probs, toks = zip(*[(n, p, t) for n, p, t in length_buckets])
                bucket_name = random.choices(names, weights=probs, k=1)[0]
                token_map = {n: t for n, _, t in length_buckets}
                return token_map[bucket_name]

            # Enhanced batch processing with vLLM optimization
            if self.generation_config.use_vllm_optimization and hasattr(self.client, 'generate_batch_optimized'):
                logger.info("🚀 Using vLLM-optimized batch processing")
                batch_size = 16  # Larger batch size for vLLM
            else:
                batch_size = 10  # Conservative batch size for standard API
                
            processed_count = 0
            filtered_count = 0
            
            logger.info(f"📊 Starting enhanced batch processing: {len(unique_prompts)} prompts prepared")

            for batch_start in range(0, len(unique_prompts), batch_size):
                if len(samples) >= num_samples:
                    break
                    
                batch_end = min(batch_start + batch_size, len(unique_prompts))
                batch_prompts = unique_prompts[batch_start:batch_end]

                # Create messages for each prompt
                batch_messages = []
                for prompt_item in batch_prompts:
                    # Always use temporal system prompt during generation
                    temporal_context = self._choose_temporal_bucket()
                    system_prompt = self._generate_temporal_system_prompt(character, temporal_context)
                    
                    # Ensure prompt is a string
                    user_prompt = prompt_item.get('prompt', '') if isinstance(prompt_item, dict) else prompt_item
                    if not isinstance(user_prompt, str):
                        logger.warning(f"Invalid prompt type: {type(user_prompt)}, skipping.")
                        continue

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    batch_messages.append(messages)

                try:
                    # Use optimized batch generation if available
                    if hasattr(self.client, 'generate_batch_optimized'):
                        replies = await self.client.generate_batch_optimized(
                            prompts=batch_messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            priority=2 if quality_level == QualityLevel.COMPREHENSIVE else 0,
                            **sampling_kwargs
                        )
                    else:
                        # Fallback to standard batch generation
                        replies = await self.client.generate_batch(
                            prompts=batch_messages,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            **sampling_kwargs
                        )

                    # Enhanced sample processing with real-time filtering
                    for i, (messages, reply) in enumerate(zip(batch_messages, replies)):
                        try:
                            reply_str = str(reply).strip()

                            if not reply_str or "Error:" in reply_str:
                                continue

                            sample = {
                                "messages": [
                                    messages[0],  # system
                                    messages[1],  # user
                                    {"role": "assistant", "content": reply_str},
                                ]
                            }
                            
                            # Apply real-time quality filtering if enabled
                            if (self.generation_config.enable_real_time_filtering and 
                                self.quality_filter and 
                                quality_level != QualityLevel.FAST):
                                
                                filter_result = self.quality_filter.filter_sample(sample)
                                
                                if not filter_result['accept']:
                                    filtered_count += 1
                                    logger.debug(f"Filtered sample: {filter_result['issues']}")
                                    continue
                                    
                                # Update quality statistics
                                self.generation_stats['avg_quality_score'] = (
                                    self.generation_stats['avg_quality_score'] * 0.9 + 
                                    filter_result['score'] * 0.1
                                )
                            
                            samples.append(sample)
                            processed_count += 1

                            # Update progress
                            if progress_callback:
                                total_current = len(samples)
                                progress_callback(min(total_current / num_samples, 1.0))

                            # Stop if we have enough samples
                            if len(samples) >= num_samples:
                                break

                        except Exception as e:
                            logger.debug(f"Error processing sample: {e}")
                            continue

                    if len(samples) >= num_samples:
                        break

                    await asyncio.sleep(0.1)  # Small delay between batches
                        
                except Exception as e:
                    logger.warning(f"⚠️ Batch processing error: {e}")
                    continue

            if progress_callback:
                progress_callback(1.0)

            new_generated = len(samples) - existing_count
            logger.info(f"🎯 ENHANCED DATASET GENERATION COMPLETE:")
            logger.info(f"   Existing samples: {existing_count}")
            logger.info(f"   New samples generated: {new_generated}")
            logger.info(f"   Total samples: {len(samples)}")
            logger.info(f"   Filtered out: {filtered_count}")
            if len(unique_prompts) > 0:
                logger.info(f"   Success rate: {new_generated/len(unique_prompts)*100:.1f}%")
                logger.info(f"   Filter efficiency: {(1 - filtered_count/max(len(unique_prompts), 1))*100:.1f}%")
            
            # Log vLLM performance if available
            if hasattr(self.client, 'get_batch_stats'):
                batch_stats = self.client.get_batch_stats()
                logger.info(f"   vLLM batch utilization: {batch_stats.get('batch_utilization', 0)*100:.1f}%")
                logger.info(f"   Average batch size: {batch_stats.get('avg_batch_size', 0):.1f}")
            
            # Update generation statistics
            self.generation_stats.update({
                'total_generated': self.generation_stats['total_generated'] + new_generated,
                'filtered_out': self.generation_stats['filtered_out'] + filtered_count,
            })

            # If custom system prompt is provided, replace all temporal prompts with it
            if custom_system_prompt is not None:
                if custom_system_prompt == "":
                    # Empty string means remove system prompts entirely
                    logger.info(f"🔄 Removing all system prompts from dataset (empty custom prompt)")
                    for sample in samples:
                        if 'messages' in sample and len(sample['messages']) > 0 and sample['messages'][0].get('role') == 'system':
                            # Remove the system message
                            sample['messages'].pop(0)
                else:
                    # Replace with the custom prompt
                    logger.info(f"🔄 Replacing temporal system prompts with custom prompt for training consistency")
                    for sample in samples:
                        if 'messages' in sample and len(sample['messages']) > 0:
                            # Replace the system prompt with the custom one
                            sample['messages'][0]['content'] = custom_system_prompt
            
            # 💾 Auto-save dataset with metadata
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
            
            self.save_dataset(character, samples, metadata)

            logger.info(f"Generated {len(samples)} total samples ({new_generated} new) using OpenAI API")
            return samples

        except Exception as e:
            logger.error(f"❌ Failed to generate dataset: {e}")
            logger.exception("Full traceback:")
            return []

    async def generate_interactive_batch(
        self,
        character: Dict[str, Any],
        num_samples: int = 20,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of samples for interactive refinement.
        
        This method uses quality curation to generate and filter samples interactively.
        """
        logger.info("🔬 Starting interactive batch generation with quality curation")
        
        # Use quality curation for interactive generation
        return await quality_curation.generate_with_quality_curation(
            self.client,
            character=character,
            raw_samples_target=num_samples * 3,  # Generate 3x for selection
            final_dataset_size=num_samples,
            quality_threshold=7.0,
            diversity_weight=0.3,
            judgment_batch_size=20,
            **kwargs
        )

    def _create_system_prompt(self, character: Dict[str, Any]) -> str:
        """Creates the system prompt for the character."""
        char_name = character.get("name", "Character")
        description = character.get("description", "A character.")
        personality = character.get("personality", "A personality.")
        scenario = character.get("scenario", "")
        
        # Using a format similar to established conventions for good performance
        prompt = (
            f"You are {char_name}. Act as {char_name}. Do not break character.\n\n"
            f"### Personality\n{personality}\n\n"
            f"### Description\n{description}\n\n"
            f"### Scenario\n{scenario}"
        )
        return prompt

    async def _generate_premium_dataset(self, character: Dict[str, Any], num_samples: int,
                                       max_tokens: int, temperature: float, top_p: float,
                                       progress_callback: Optional[Callable] = None,
                                       append_to_existing: bool = True,
                                       custom_system_prompt: Optional[str] = None,
                                       extra_quality: bool = False,
                                       **sampling_kwargs) -> List[Dict[str, Any]]:
        """Generate premium quality dataset with progressive refinement and specialized judging"""
        
        logger.info("🌟 Starting PREMIUM dataset generation with progressive refinement")
        
        # Load existing dataset
        existing_samples = []
        if append_to_existing:
            existing_dataset = self.load_dataset(character)
            if existing_dataset:
                existing_samples = existing_dataset
                logger.info(f"📂 Found existing dataset with {len(existing_samples)} samples")

        existing_count = len(existing_samples)
        new_samples_needed = num_samples - existing_count
        
        if existing_count >= num_samples:
            logger.info(f"✅ Dataset already has sufficient samples ({existing_count} >= {num_samples})")
            return existing_samples[:num_samples]
        
        # Phase 1: Generate initial samples with higher diversity
        logger.info(f"📊 Phase 1: Generating {new_samples_needed * 3} diverse samples for selection")
        
        initial_samples = await self.generate_dataset(
            character=character,
            num_samples=new_samples_needed * 3,  # Generate 3x for selection
            max_tokens=max_tokens,
            temperature=min(temperature + 0.1, 1.0),  # Slightly higher temperature for diversity
            top_p=top_p,
            progress_callback=lambda p: progress_callback(p * 0.6) if progress_callback else None,
            append_to_existing=False,  # Don't save intermediate results
            custom_system_prompt=custom_system_prompt,
            extra_quality=extra_quality,
            quality_level=QualityLevel.ITERATIVE,  # Use iterative for initial generation
            **sampling_kwargs
        )
        
        if not initial_samples:
            logger.warning("No initial samples generated for premium processing")
            return existing_samples
        
        # Phase 2: Use quality curation to select the best samples
        logger.info(f"📊 Phase 2: Curating {new_samples_needed} best samples from {len(initial_samples)} candidates")
        
        try:
            curated_samples = await quality_curation.generate_with_quality_curation(
                self.client,
                character=character,
                raw_samples_target=len(initial_samples),
                final_dataset_size=new_samples_needed,
                quality_threshold=8.0,  # Higher threshold for premium
                diversity_weight=0.4,
                judgment_batch_size=20,
                progress_callback=lambda p: progress_callback(0.6 + p * 0.4) if progress_callback else None,
                **sampling_kwargs
            )
            
            # Combine with existing samples
            final_samples = existing_samples + curated_samples
            
            # Save the final dataset
            self.save_dataset(character, final_samples)
            
            logger.info(f"🌟 PREMIUM generation complete: {len(final_samples)} total samples")
            return final_samples
            
        except Exception as e:
            logger.error(f"Premium curation failed: {e}")
            # Fallback to regular selection
            final_samples = existing_samples + initial_samples[:new_samples_needed]
            return final_samples

    # Helper methods from the legacy implementation

    async def suggest_user_questions(
        self,
        character: Dict[str, Any],
        num_questions: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        existing_dataset: Optional[List[Dict[str, Any]]] = None,
        context_samples: int = 12,
        **sampling_kwargs
    ) -> List[Dict[str, Any]]:
        """Generate character-specific questions using LLM"""
        
        char_name = character.get('name', 'Character')
        char_description = character.get('description', '')
        char_personality = character.get('personality', '')
        
        # Build context from existing dataset
        context_examples = ""
        if existing_dataset and len(existing_dataset) > 0:
            sample_size = min(context_samples, len(existing_dataset))
            context_samples_list = random.sample(existing_dataset, sample_size)
            
            context_examples = "\n\nExisting conversation examples:\n"
            for i, sample in enumerate(context_samples_list):
                if 'messages' in sample and len(sample['messages']) >= 2:
                    user_msg = sample['messages'][-2]['content']
                    context_examples += f"{i+1}. {user_msg}\n"
        
        prompt = f"""Generate {num_questions} diverse, engaging questions that would be interesting to ask {char_name}.

Character: {char_name}
Description: {char_description}
Personality: {char_personality}

{context_examples}

Generate questions that:
- Explore different aspects of the character's personality and background
- Are natural and conversational
- Avoid repetition with existing examples
- Range from casual to deeper topics
- Would lead to interesting character responses

Format each question on a new line, numbered 1-{num_questions}:"""

        try:
            response = await self._generate_single_response(
                prompt,
                max_tokens=2000,
                temperature=temperature
            )
            
            # Parse the response into individual questions
            questions = []
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering and clean up
                    question = re.sub(r'^\d+\.?\s*', '', line)
                    question = re.sub(r'^-\s*', '', question)
                    question = question.strip()
                    
                    if question and len(question) > 10:  # Basic quality check
                        questions.append({'question': question})
            
            logger.info(f"Generated {len(questions)} questions for {char_name}")
            return questions[:num_questions]  # Ensure we don't exceed requested amount
            
        except Exception as e:
            logger.error(f"Failed to generate questions: {e}")
            return []

    async def generate_scenario_based_prompts(self, character: Dict[str, Any], 
                                        knowledge: Dict[str, Any], num_scenarios: int = 20) -> List[Dict[str, Any]]:
        """Generate scenario-based prompts using character knowledge"""
        
        char_name = character.get('name', 'Character')
        scenarios = []
        
        # Create scenarios based on character knowledge
        if knowledge.get('traits'):
            for trait in knowledge['traits'][:3]:
                scenario = {
                    'context': f"Exploring {char_name}'s {trait} trait",
                    'prompts': [
                        f"How does your {trait} nature affect your daily life?",
                        f"Tell me about a time when being {trait} helped you.",
                        f"What challenges do you face because you're {trait}?"
                    ]
                }
                scenarios.append(scenario)
        
        if knowledge.get('skills'):
            for skill in knowledge['skills'][:2]:
                scenario = {
                    'context': f"Discussing {char_name}'s {skill} skills",
                    'prompts': [
                        f"How did you develop your {skill} skills?",
                        f"What's the most challenging aspect of {skill}?",
                        f"Can you teach me something about {skill}?"
                    ]
                }
                scenarios.append(scenario)
        
        # Add some general scenarios
        general_scenarios = [
            {
                'context': 'Personal reflection',
                'prompts': [
                    "What's something you've learned about yourself recently?",
                    "How do you handle difficult situations?",
                    "What motivates you to keep going?"
                ]
            },
            {
                'context': 'Relationships and social',
                'prompts': [
                    "How do you prefer to spend time with others?",
                    "What do you value most in friendships?",
                    "How do you handle conflicts with people?"
                ]
            }
        ]
        
        scenarios.extend(general_scenarios)
        return scenarios[:num_scenarios]

    async def generate_multi_turn_conversation(self, character: Dict[str, Any], 
                                         scenario: Dict[str, Any], turns: int = 3) -> List[Dict[str, str]]:
        """Generate multi-turn conversation flows"""
        
        conversations = []
        base_prompts = scenario.get('prompts', [])
        
        for base_prompt in base_prompts[:2]:  # Limit to 2 base prompts
            conversation = [base_prompt]
            
            # Generate follow-up questions
            for turn in range(turns - 1):
                follow_up_prompt = f"Based on the conversation about '{base_prompt}', what would be a natural follow-up question?"
                
                try:
                    follow_up = await self._generate_single_response(
                        follow_up_prompt,
                        max_tokens=100,
                        temperature=0.7
                    )
                    
                    if follow_up and len(follow_up.strip()) > 5:
                        conversation.append(follow_up.strip())
                    
                except Exception as e:
                    logger.debug(f"Failed to generate follow-up: {e}")
                    break
            
            if len(conversation) > 1:
                conversations.extend(conversation)
        
        return [{'prompt': conv} for conv in conversations]

    def _generate_temporal_system_prompt(self, character: Dict[str, Any],
                                       temporal_context: str,
                                       relationship_context: Optional[str] = None) -> str:
        """Generate temporal system prompt for character"""
        
        char_name = character.get('name', 'Character')
        description = character.get('description', '')
        personality = character.get('personality', '')
        scenario = character.get('scenario', '')
        
        # Basic temporal system prompt
        base_prompt = f"""You are {char_name}. Stay in character at all times.

Character Description: {description}

Personality: {personality}

Current Scenario: {scenario}

Temporal Context: {temporal_context}"""

        if relationship_context:
            base_prompt += f"\n\nRelationship Context: {relationship_context}"
        
        return base_prompt

    def _choose_temporal_bucket(self) -> str:
        """Choose a temporal bucket for context"""
        buckets = [
            "present_moment",
            "recent_past", 
            "distant_past",
            "near_future",
            "hypothetical"
        ]
        return random.choice(buckets)

    async def _build_user_prompt(self) -> str:
        """Build a user prompt using default prompts"""
        return random.choice(self.default_user_prompts)

    # DatasetManager-specific methods that are not in the base classes

    def _make_card_block(self, card: Dict[str, str]) -> str:
        """Create a formatted character card block for few-shot examples"""
        return f"""Character Card:
Name: {card.get('name', 'Unknown')}
Description: {card.get('description', '')}
Personality: {card.get('personality', '')}
Scenario: {card.get('scenario', '')}"""
