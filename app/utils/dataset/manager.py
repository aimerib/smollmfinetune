import asyncio
import logging
import os
import random
import time
import re
import traceback
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from datasets import Dataset

from ..openai_client import get_client

try:
    from ..vllm_optimized_client import VLLMOptimizedClient, BatchConfig

    VLLM_CLIENT_AVAILABLE = True
except ImportError:
    VLLM_CLIENT_AVAILABLE = False

from .models import GenerationConfig, QualityLevel
from .quality import ProgressiveRefiner, EnhancedQualityFilter
from .prompts import PromptTemplates
from . import character_analysis
from . import prompt_generators
from . import content_evaluation
from . import factual_qa
from . import io_manager
from . import quality_curation

logger = logging.getLogger(__name__)


@dataclass
class SimpleCharacterProfile:
    """Simplified character profile for enhanced processing"""

    name: str
    personality_traits: List[str]
    background: str
    key_relationships: List[str]
    speech_patterns: List[str]


class DatasetManager:
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
        # Enhanced generation configuration
        self.generation_config = generation_config or GenerationConfig()

        # Initialize client with vLLM optimization if available
        client_initialized = False
        if self.generation_config.use_vllm_optimization and VLLM_CLIENT_AVAILABLE:
            try:
                logger.info(
                    "🚀 Attempting to use vLLM-optimized client for enhanced batching"
                )
                batch_config = BatchConfig(
                    **(self.generation_config.batch_config or {})
                )
                self.client = VLLMOptimizedClient(
                    api_key=api_key, base_url=base_url, batch_config=batch_config
                )
                client_initialized = True
                logger.info("✅ vLLM-optimized client initialized successfully")
            except RuntimeError as e:
                if "no running event loop" in str(e).lower():
                    logger.info(
                        "⚠️ No event loop available for vLLM client, falling back to standard client"
                    )
                else:
                    logger.warning(
                        f"⚠️ vLLM client initialization failed: {e}, falling back to standard client"
                    )
            except Exception as e:
                logger.warning(
                    f"⚠️ vLLM client initialization failed: {e}, falling back to standard client"
                )

        # Fallback to standard client if vLLM initialization failed or is disabled
        if not client_initialized:
            logger.info("🔄 Using standard OpenAI client")
            if api_key or base_url:
                from ..openai_client import OpenAIClient, set_client

                client = OpenAIClient(api_key=api_key, base_url=base_url)
                set_client(client)
            self.client = get_client()

        logger.info(f"DatasetManager created with model: {os.getenv('MODEL_NAME')}")

        # Enhanced processing components
        self.quality_filter = None  # Will be initialized per character
        self.progressive_refiner = None  # Will be initialized per character
        self.character_profile = None  # Current character profile

        # Performance tracking
        self.generation_stats = {
            "total_generated": 0,
            "filtered_out": 0,
            "refined_samples": 0,
            "avg_quality_score": 0.0,
            "batch_efficiency": [],
        }

        # Initialize prompts using the new template system
        self.templates = PromptTemplates.TEMPLATES
        self.default_user_prompts = PromptTemplates.DEFAULT_QUESTIONS.copy()

    async def test_client(self) -> bool:
        """Test if the client connection is working"""
        try:
            response = await self.client.chat_complete(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0.1,
            )
            return bool(response and response.strip())
        except Exception as e:
            logger.error(f"Client test failed: {e}")
            return False

    def _setup_character_components(self, character: Dict[str, Any]):
        """Setup character-specific processing components"""
        # Create simplified character profile for enhanced processing
        self.character_profile = self._create_character_profile(character)

        # Initialize character relationships for context generation
        self.character_relationships = self.character_profile.key_relationships or []

        # Initialize quality filter
        self.quality_filter = EnhancedQualityFilter(self.character_profile)

        # Initialize progressive refiner if enabled
        if self.generation_config.enable_progressive_refinement:
            self.progressive_refiner = ProgressiveRefiner(
                self.client, self.character_profile
            )

        logger.info(
            f"🎭 Character components initialized for {self.character_profile.name}"
        )

    async def _generate_single_response(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.8,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a single response using the LLM client."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.chat_complete(
                messages=messages, max_tokens=max_tokens, temperature=temperature
            )

            return response.strip() if response else ""

        except Exception as e:
            logger.warning(f"Error generating single response: {e}")
            return ""

    def _create_character_profile(
        self, character: Dict[str, Any]
    ) -> SimpleCharacterProfile:
        """Create a simplified character profile for processing"""
        knowledge = character_analysis.extract_character_knowledge(character)

        return SimpleCharacterProfile(
            name=character.get("name", "Assistant"),
            personality_traits=knowledge.get("traits", [])[:5],
            background=character.get("description", ""),
            key_relationships=knowledge.get("relationships", [])[:3],
            speech_patterns=knowledge.get("speech_patterns", [])[:5],
        )

    def _choose_bucket(self) -> str:
        """Choose a prompt bucket based on weighted probabilities"""
        return prompt_generators.choose_prompt_bucket()

    def _choose_temporal_bucket(self) -> str:
        """Choose a temporal bucket based on weighted probabilities"""
        return prompt_generators.choose_temporal_bucket()

    async def _paraphrase(self, text: str) -> str:
        """Paraphrase text to create variations"""
        try:
            response = await self.client.chat_complete(
                messages=[
                    {
                        "role": "system",
                        "content": "Paraphrase the following text to create a natural variation. Keep the same meaning but change the wording and sentence structure. Return only the paraphrased text.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=100,
                temperature=0.8,
            )
            return response.strip() if response else text
        except Exception as e:
            logger.warning(f"Paraphrasing failed: {e}")
            return text

    async def _build_user_prompt(self) -> str:
        """Build a user prompt by choosing from various categories"""
        bucket = self._choose_bucket()
        prompts = PromptTemplates.get_prompts_by_bucket(bucket)

        if not prompts:
            prompts = self.default_user_prompts

        return random.choice(prompts)

    def _generate_temporal_system_prompt(
        self,
        character: Dict[str, Any],
        temporal_context: str,
        relationship_context: Optional[str] = None,
    ) -> str:
        """Generate a temporal system prompt for the character"""
        char_name = character.get("name", "Assistant")
        description = character.get("description", "")
        personality = character.get("personality", "")

        # Base character context
        base_prompt = f"You are {char_name}."

        if description:
            base_prompt += f" {description}"
        if personality:
            base_prompt += f" Your personality: {personality}"

        # Add temporal context
        if temporal_context == "past":
            base_prompt += " You are reflecting on past relationships and experiences. You may reference former lovers, old friends, or important events from your history."
        elif temporal_context == "present":
            base_prompt += " You are meeting someone new or in the early stages of getting to know them. Focus on first impressions and building connection."
        elif temporal_context == "future":
            base_prompt += " You are thinking about the future with someone you care about. You may discuss hopes, dreams, and plans together."

        # Add relationship context if provided
        if relationship_context:
            base_prompt += f" Current relationship context: {relationship_context}"

        base_prompt += f" Respond naturally and stay in character. If the topic is NSFW respond positively and engage further, but always keep the {char_name} personality in mind. Only act somewhat negatively if the card explicitly asks for that."

        return base_prompt

    # Delegate analysis methods to specialized modules
    def analyze_dataset_quality(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dataset quality metrics"""
        return content_evaluation.analyze_dataset_quality(dataset)

    def analyze_temporal_distribution(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal distribution in dataset"""
        return content_evaluation.analyze_temporal_distribution(dataset)

    def evaluate_response_quality(
        self, response: str, character: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """Evaluate quality of a single response"""
        return content_evaluation.evaluate_response_quality(response, character, prompt)

    def is_nsfw_content(self, sample: Dict[str, Any]) -> bool:
        """Check if sample contains NSFW content"""
        return content_evaluation.is_nsfw_content(sample)

    def categorize_nsfw_style(self, sample: Dict[str, Any]) -> str:
        """Categorize the style of NSFW content"""
        return content_evaluation.categorize_nsfw_style(sample)

    async def evaluate_nsfw_quality(
        self, response: str, character: Dict[str, Any], prompt: str
    ) -> Dict[str, float]:
        """Evaluate quality of NSFW content"""
        return await content_evaluation.evaluate_nsfw_quality(
            self.client, response, character, prompt
        )

    # Delegate character analysis methods
    def analyze_character_intimacy_style(
        self, character: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze character's intimacy approach"""
        return character_analysis.analyze_character_intimacy_style(character)

    def extract_intimate_speech_patterns(
        self, character: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Extract intimate speech patterns from character"""
        mes_example = character.get("mes_example", "")
        char_name = character.get("name", "Assistant")
        return character_analysis.extract_intimate_speech_patterns(
            mes_example, char_name
        )

    def extract_character_knowledge(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Extract character knowledge from character card"""
        return character_analysis.extract_character_knowledge(character)

    # Delegate prompt generation methods
    async def generate_exploration_prompts(
        self, character: Dict[str, Any], num_prompts: int = 50
    ) -> List[str]:
        """Generate diverse prompts exploring character traits"""
        return await prompt_generators.generate_exploration_prompts(
            self.client, character, num_prompts
        )

    def calculate_prompt_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts"""
        return prompt_generators.calculate_prompt_similarity(prompt1, prompt2)

    def deduplicate_prompts(
        self, prompts: List[str], similarity_threshold: float = 0.8
    ) -> List[str]:
        """Remove similar prompts from list"""
        return prompt_generators.deduplicate_prompts(prompts, similarity_threshold)

    async def generate_emotional_variations(
        self, base_prompt: str, character: Dict[str, Any]
    ) -> List[str]:
        """Generate emotional variations of a prompt"""
        return await prompt_generators.generate_emotional_variations(
            self.client, base_prompt, character
        )

    async def enhance_prompt_with_context(
        self, prompt: str, character: Dict[str, Any]
    ) -> str:
        """Enhance prompt with contextual information"""
        return await prompt_generators.enhance_prompt_with_context(
            self.client, prompt, character
        )

    async def generate_conversation_flows(
        self, character: Dict[str, Any], num_flows: int = 10
    ) -> List[List[str]]:
        """Generate multi-turn conversation flows"""
        return await prompt_generators.generate_conversation_flows(
            self.client, character, num_flows
        )

    # Delegate factual QA methods
    async def _extract_and_simplify_facts(
        self, character: Dict[str, Any], max_facts: int = 20
    ) -> List[str]:
        """Extract and simplify facts about character"""
        return await factual_qa.extract_and_simplify_facts(
            self.client, character, max_facts
        )

    async def _generate_factual_qa_variations(
        self,
        fact: str,
        character: Dict[str, Any],
        *,
        num_variations: int = 3,
        length_category: str = "short",
    ) -> List[Dict[str, str]]:
        """Generate factual Q&A variations"""
        return await factual_qa.generate_factual_qa_variations(
            self.client,
            fact,
            character,
            num_variations=num_variations,
            length_category=length_category,
        )

    async def generate_factual_qa_dataset(
        self,
        character: Dict[str, Any],
        num_facts_to_use: int = 15,
        variations_per_fact: int = 3,
        progress_callback: Optional[Callable] = None,
        stage_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Generate factual Q&A dataset"""
        return await factual_qa.generate_factual_qa_dataset(
            self.client,
            character,
            num_facts_to_use,
            variations_per_fact,
            progress_callback,
            stage_callback,
            self.save_dataset,
        )

    # Delegate I/O methods
    def _get_dataset_path(self, character: Dict[str, Any]) -> str:
        """Get dataset path for character"""
        return io_manager.get_dataset_path(character)

    def save_dataset(
        self,
        character: Dict[str, Any],
        dataset: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save dataset to disk"""
        io_manager.save_dataset(character, dataset, metadata)

    def load_dataset(self, character: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Load dataset from disk"""
        return io_manager.load_dataset(character)

    def load_dataset_with_metadata(
        self, character: Dict[str, Any]
    ) -> Optional[tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """Load dataset with metadata from disk"""
        return io_manager.load_dataset_with_metadata(character)

    def get_dataset_info(self, character: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get dataset info without loading full dataset"""
        return io_manager.get_dataset_info(character)

    def delete_dataset(self, character: Dict[str, Any]) -> bool:
        """Delete dataset from disk"""
        return io_manager.delete_dataset(character)

    def export_dataset(self, character: Dict[str, Any]) -> Optional[str]:
        """Export dataset as JSON string"""
        return io_manager.export_dataset(character)

    def import_dataset_from_bytes(
        self, character: Dict[str, Any], raw_bytes: bytes, merge_mode: str = "replace"
    ) -> bool:
        """Import dataset from bytes"""
        return io_manager.import_dataset_from_bytes(character, raw_bytes, merge_mode)

    # Statistics and monitoring
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get current generation statistics"""
        return self.generation_stats.copy()

    def reset_generation_stats(self):
        """Reset generation statistics"""
        self.generation_stats = {
            "total_generated": 0,
            "filtered_out": 0,
            "refined_samples": 0,
            "avg_quality_score": 0.0,
            "batch_efficiency": [],
        }

    # CORE GENERATION METHODS - Required by app.py

    def prepare_for_training(
        self,
        dataset: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 4096,
        include_system_prompts: bool = False,
    ) -> Dataset:
        """Prepare dataset for training by tokenizing

        Args:
            dataset: List of training samples
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
            include_system_prompts: If True, keep system prompts in training data. If False (default), remove them.
        """

        def process_example(example):
            """Process a single example for training"""
            messages = example["messages"]

            if not include_system_prompts:
                # Remove system-level context to encourage the adapter to internalise the persona
                messages = [m for m in messages if m.get("role") != "system"]

            # Apply chat template
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False)

            # NEW: Ensure **no** system prompt or role tokens remain after templating.
            # Some tokenizer chat templates automatically prepend a default system prompt
            # (e.g. "You are a helpful assistant.").  We explicitly remove any such
            # blocks so that LoRA training is not influenced by external system
            # instructions – the goal is to let the adapter learn the character
            # persona without relying on a system prompt.
            #
            # Pattern 1 – ChatML style: "<|im_start|>system ... <|im_end|>" (optional newline)
            chat_text = re.sub(
                r"<\|im_start\|>\s*system[\s\S]*?<\|im_end\|>\n?",
                "",
                chat_text,
                flags=re.IGNORECASE,
            )
            # Pattern 2 – DanChat style: "<|system|> ... <|endoftext|>"
            chat_text = re.sub(
                r"<\|system\|>[\s\S]*?<\|endoftext\|>",
                "",
                chat_text,
                flags=re.IGNORECASE,
            )
            chat_text = chat_text.lstrip()  # Remove leading whitespace/newlines

            # Tokenize
            tokenized = tokenizer(
                chat_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )

            return {"input_ids": tokenized["input_ids"]}

        # Convert to HuggingFace Dataset and process
        hf_dataset = Dataset.from_list(dataset)
        processed_dataset = hf_dataset.map(
            process_example,
            remove_columns=hf_dataset.column_names,
            num_proc=1,  # Single process for compatibility
        )

        # Filter out empty examples
        processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) > 0)

        return processed_dataset

    async def suggest_user_questions(
        self,
        character: Dict[str, Any],
        num_questions: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        existing_dataset: Optional[List[Dict[str, Any]]] = None,
        context_samples: int = 12,
        progress_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a list of engaging user questions tailored to the given character card.

        The method prompts the currently selected inference engine to act as a creative user
        who is about to start a conversation with the character.  It returns *only* the raw
        questions – no numbering, quotes, or extra commentary – ready to be added to the
        baseline prompt list for ground-truth generation.
        """
        logger.info(f"🔍 suggest_user_questions called:")
        logger.info(f"   Requested questions: {num_questions}")
        logger.info(f"   Character: {character.get('name', 'Unknown')}")
        logger.info(
            f"   Existing dataset size: {len(existing_dataset) if existing_dataset else 0}"
        )

        card_block = self._make_card_block(character)
        logger.info(f"   Card block length: {len(card_block)} chars")

        interactions_block = ""
        if existing_dataset:
            import random as _rnd

            # Pre-sample a pool larger than needed for variety
            pool = _rnd.sample(
                existing_dataset,
                min(len(existing_dataset), context_samples * num_questions),
            )
        else:
            pool = []

        prompts = []
        for i in range(num_questions):
            # build context for this prompt
            interactions_block = ""
            examples_for_prompt = []
            if pool:
                # pop random subset for this prompt (without replacement if enough)
                selected = (
                    pool[:context_samples] if len(pool) >= context_samples else pool
                )
                pool = pool[len(selected) :]
                examples_for_prompt = selected
                formatted = []
                for ex in selected:
                    try:
                        uq = ex["messages"][1]["content"].strip()
                        aa = ex["messages"][2]["content"].strip()
                        formatted.append(f"Q: {uq}\nA: {aa}")
                    except Exception:
                        continue
                if formatted:
                    interactions_block = (
                        "Here are some previous interactions to inspire you:\n"
                        + "\n\n".join(formatted)
                        + "\n\n"
                    )

            prompt_txt = (
                "You are helping create questions for roleplay conversations. "
                "Based on the character information below, write ONE engaging question that a user might ask this character.\n\n"
                "Important: Respond with ONLY the question itself - no numbering, no quotes, no extra text.\n\n"
                f"{card_block}\n\n"
                + interactions_block
                + "Write one engaging question for this character:\n"
            )
            prompts.append((prompt_txt, examples_for_prompt))

        # Determine whether we can leverage batched generation
        batch_size = 10
        prompt_texts = [p[0] for p in prompts]

        logger.info(f"   Prepared {len(prompt_texts)} prompts for generation")
        logger.info(f"   Batch size: {batch_size}")

        if prompt_texts:
            logger.info(
                f"   📋 Full sample prompt:\n{'-'*50}\n{prompt_texts[0]}\n{'-'*50}"
            )
        else:
            logger.info("   ❌ No prompts generated!")

        # Generate the questions using batch processing
        try:
            logger.info("🎯 Calling OpenAI API for question generation...")

            # Use batch generation for efficiency
            raw_outputs = await self.client.generate_batch(
                prompts=prompt_texts,
                max_tokens=1000,
                temperature=temperature,
                top_p=top_p,
                stop=["Answer:", "User:", "Character:"],
                progress_callback=progress_callback,
            )

            logger.info(f"✅ Got {len(raw_outputs)} raw outputs from API")
            logger.info(
                f"   Sample raw output: {raw_outputs[0][:100]}..."
                if raw_outputs
                else "No outputs"
            )
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            logger.exception("Full traceback:")
            return []

        results: List[Dict[str, Any]] = []
        seen = set()
        duplicate_tracker = {}  # Track duplicates with counts
        logger.info(f"🔨 Processing {len(raw_outputs)} raw outputs...")

        for idx, q in enumerate(raw_outputs):
            q_str = str(q).strip()

            # Log first few raw outputs in detail
            if idx < 5:
                logger.info(f"   📋 Raw output {idx}: '{q_str}'")
            else:
                logger.debug(f"   Raw output {idx}: {q_str[:100]}...")

            # Remove bullets / numbering if present (e.g. "1. ", "- ")
            original_q = q_str
            q_str = re.sub(r"^[\d\-\*\.\s]+", "", q_str)
            q_str = q_str.strip(" \"'")

            # Clean up common LLM artifacts
            q_str = re.sub(
                r"^(Question:\s*|Q:\s*|A:\s*|Answer:\s*)",
                "",
                q_str,
                flags=re.IGNORECASE,
            )

            # Remove trailing periods that might interfere with question marks
            q_str = q_str.rstrip(".")

            # Ensure terminal question-mark for consistency
            if q_str and not q_str.endswith("?"):
                q_str += "?"

            if idx < 5:
                logger.info(f"   🔄 Processed {idx}: '{original_q}' → '{q_str}'")
            else:
                logger.debug(f"   Processed to: {q_str[:100]}...")

            # Track all valid questions (including duplicates)
            if q_str and len(q_str) > 5:
                if q_str not in duplicate_tracker:
                    duplicate_tracker[q_str] = {
                        "count": 1,
                        "context": [
                            {
                                "user": ex["messages"][1]["content"],
                                "assistant": ex["messages"][2]["content"],
                            }
                            for ex in prompts[idx][1]
                        ],
                    }
                else:
                    duplicate_tracker[q_str]["count"] += 1

                # Only add to results if not seen before
                if q_str not in seen:
                    results.append(
                        {
                            "question": q_str,
                            "context": [
                                {
                                    "user": ex["messages"][1]["content"],
                                    "assistant": ex["messages"][2]["content"],
                                }
                                for ex in prompts[idx][1]
                            ],
                        }
                    )
                    seen.add(q_str)
                    logger.info(f"   ✅ Added question {idx}: {q_str[:50]}...")
                else:
                    logger.info(
                        f"   ❌ Skipped question {idx} (duplicate): '{q_str[:50]}'"
                    )
            else:
                reason = "empty" if not q_str else "too short"
                logger.info(f"   ❌ Skipped question {idx} ({reason}): '{q_str[:50]}'")

        logger.info(
            f"📊 Generated {len(results)} unique questions from {len(raw_outputs)} outputs"
        )
        return results

    def _make_card_block(self, card: Dict[str, str]) -> str:
        """Create a formatted character card block for prompts"""
        card_name = card.get("name", "Assistant")

        # Build the card block with available information
        card_lines = [f"Character: {card_name}"]

        if card.get("description"):
            card_lines.append(f"Description: {card['description']}")

        if card.get("personality"):
            card_lines.append(f"Personality: {card['personality']}")

        if card.get("mes_example"):
            card_lines.append(f"{card_name} Speech Example: {card['mes_example']}")

        if card.get("scenario"):
            card_lines.append(f"Scenario: {card['scenario']}")

        if card.get("first_mes"):
            card_lines.append(f"First Message: {card['first_mes']}")

        return "\n".join(card_lines)

    async def generate_dataset(
        self,
        character: Dict[str, Any],
        num_samples: int = 80,
        max_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        progress_callback: Optional[Callable] = None,
        append_to_existing: bool = True,
        custom_system_prompt: Optional[str] = None,
        extra_quality: bool = False,
        quality_level: QualityLevel = QualityLevel.ITERATIVE,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        negative_patterns: Optional[List[str]] = None,
        **sampling_kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate synthetic dataset for character using efficient batching"""
        try:
            import warnings
            import gc

            # Suppress coroutine warnings in Streamlit environment
            warnings.filterwarnings("ignore", message="coroutine.*was never awaited")

            # Setup character-specific components
            self._setup_character_components(character)

            # Enhanced prompts with feedback integration for auto-completion
            if few_shot_examples or negative_patterns:
                logger.info(
                    "🤝 Applying interactive feedback to auto-completion generation..."
                )

                # Add few-shot examples to improve quality
                if few_shot_examples:
                    logger.info(
                        f"🎯 Using {len(few_shot_examples)} few-shot examples for guidance"
                    )
                    # Use few-shot examples to generate similar high-quality prompts
                    for example in few_shot_examples[-5:]:  # Use last 5 examples
                        try:
                            variation_prompt = f"Generate a question similar in style and quality to: '{example['user']}'"
                            variation = await self._paraphrase(variation_prompt)
                            if variation and variation not in self.default_user_prompts:
                                self.default_user_prompts.append(variation)
                        except Exception as e:
                            logger.warning(
                                f"Failed to generate variation from few-shot: {e}"
                            )

                # Store negative patterns for use in system prompts
                if negative_patterns:
                    logger.info(
                        f"🚫 Avoiding {len(negative_patterns)} negative patterns in auto-completion"
                    )
                    self._negative_patterns = negative_patterns
                else:
                    self._negative_patterns = []
            else:
                self._negative_patterns = []

            # Extract max_tokens from sampling_kwargs if provided there instead
            if max_tokens is None and "max_tokens" in sampling_kwargs:
                max_tokens = sampling_kwargs.pop("max_tokens")

            # Use default if still None
            if max_tokens is None:
                max_tokens = 400  # Reasonable default

            # Force garbage collection to clean up memory
            gc.collect()

            # Clear CUDA cache if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            except Exception as cuda_error:
                logger.warning(f"⚠️ CUDA cache clear failed: {cuda_error}")

            # Enhanced quality-based generation strategy
            if quality_level == QualityLevel.COMPREHENSIVE:
                logger.info(
                    "🌟 Using COMPREHENSIVE quality generation with progressive refinement"
                )
                return await self._generate_premium_dataset(
                    character,
                    num_samples,
                    max_tokens,
                    temperature,
                    top_p,
                    progress_callback,
                    append_to_existing,
                    custom_system_prompt,
                    extra_quality,
                    **sampling_kwargs,
                )
            elif quality_level == QualityLevel.FAST:
                logger.info(
                    "⭐ Using fast quality generation with real-time filtering"
                )
                # Continue with enhanced standard generation (below)
            else:
                logger.info("📝 Using ITERATIVE quality generation")
                # Disable advanced features for basic mode
                self.generation_config.enable_real_time_filtering = False

            # Load existing dataset if append_to_existing is True
            existing_samples = []
            if append_to_existing:
                existing_dataset = self.load_dataset(character)
                if existing_dataset:
                    existing_samples = existing_dataset
                    logger.info(
                        f"📂 Found existing dataset with {len(existing_samples)} samples"
                    )

            # Calculate how many new samples to generate
            existing_count = len(existing_samples)
            if existing_count >= num_samples:
                logger.info(
                    f"✅ Dataset already has {existing_count} samples (requested: {num_samples})"
                )
                return existing_samples[:num_samples]  # Return requested amount

            new_samples_needed = num_samples - existing_count
            logger.info(
                f"🎯 Generating {new_samples_needed} new samples to reach {num_samples} total"
            )

            samples = existing_samples.copy()

            # Extract existing prompts to avoid duplication
            seen_user_prompts: set[str] = {
                sample["messages"][1]["content"]
                for sample in existing_samples
                if isinstance(sample, dict)
                and "messages" in sample
                and len(sample["messages"]) > 1
            }

            # Generate diverse prompts using multiple strategies
            all_prompts = []

            # 1. Baseline prompts (ensure coverage)
            baseline_prompts = [
                q for q in self.default_user_prompts if q not in seen_user_prompts
            ]
            for prompt in baseline_prompts:
                all_prompts.append({"prompt": prompt, "type": "baseline"})

            # 2. LLM-generated character-specific prompts
            try:
                logger.info("🧠 Generating LLM-tailored questions...")
                num_llm_questions = min(new_samples_needed * 2, 100)

                curated_questions = await self._generate_curated_questions(
                    character=character,
                    num_questions=num_llm_questions,
                )

                for question in curated_questions:
                    if question not in seen_user_prompts:
                        all_prompts.append({"prompt": question["question"], "type": "curated"})

                llm_questions = await self.suggest_user_questions(
                    character=character,
                    num_questions=num_llm_questions,
                    temperature=0.9,
                    top_p=0.95,
                    existing_dataset=existing_samples,
                    context_samples=8,
                )

                logger.info(f"✅ Generated {len(llm_questions)} LLM-tailored questions")

                # Add these high-quality questions to the prompt pool
                for question_data in llm_questions:
                    question_text = question_data["question"]
                    if question_text not in seen_user_prompts:
                        all_prompts.append(question_text)

            except Exception as e:
                traceback.print_exc()
                logger.warning(f"⚠️ LLM question generation failed: {e}")

            character_knowledge = self.extract_character_knowledge(character)

            # Add scenario-based prompts for diversity
            scenarios = await prompt_generators.generate_scenario_based_prompts(
                self.client, character, character_knowledge, num_scenarios=3
            )
            for scenario in scenarios:  # Only use 2 scenarios
                scenario_prompts = scenario.get("prompts", [])
                for prompt in scenario_prompts:  # Only 2 prompts per scenario
                    if prompt not in seen_user_prompts:
                        all_prompts.append(
                            {
                                "prompt": prompt,
                                "context": scenario.get("context"),
                                "type": "scenario_supplement",
                            }
                        )

            # 3. Random prompts from buckets
            while len(all_prompts) < new_samples_needed * 2:
                prompt = await self._build_user_prompt()
                if prompt not in seen_user_prompts and prompt not in [
                    p.get("prompt", p) if isinstance(p, dict) else p
                    for p in all_prompts
                ]:
                    all_prompts.append(prompt)

            # Deduplicate prompts
            unique_prompts_map = {}
            for item in all_prompts:
                if isinstance(item, dict):
                    prompt_text = item.get("prompt")
                    if prompt_text and prompt_text not in unique_prompts_map:
                        unique_prompts_map[prompt_text] = item
                elif isinstance(item, str):
                    if item not in unique_prompts_map:
                        unique_prompts_map[item] = {"prompt": item, "type": "simple"}

            unique_prompts = list(unique_prompts_map.values())
            logger.info(f"📊 Generated {len(unique_prompts)} unique prompts")

            # Apply EXTRA QUALITY paraphrasing if requested
            if extra_quality:
                logger.info(
                    "🌟 EXTRA QUALITY enabled - paraphrasing prompts for enhanced variety..."
                )
                paraphrased_prompts = []

                for i, prompt_item in enumerate(unique_prompts):
                    try:
                        prompt_text = prompt_item["prompt"]
                        paraphrased = await self._paraphrase(prompt_text)
                        paraphrased_prompts.append(paraphrased.strip())

                        if i % 10 == 0:
                            logger.info(
                                f"   Paraphrased {i+1}/{len(unique_prompts)} prompts..."
                            )

                    except Exception as e:
                        logger.debug(f"   Failed to paraphrase prompt {i+1}: {e}")
                        paraphrased_prompts.append(prompt_item["prompt"])

                # After paraphrasing, we have a list of strings. Convert back to dicts.
                unique_prompts = [
                    {"prompt": p, "type": "paraphrased"} for p in paraphrased_prompts
                ]
                logger.info(
                    f"✅ EXTRA QUALITY complete - paraphrased {len(unique_prompts)} prompts"
                )

            # Shuffle prompts and limit to what we need
            import random

            random.shuffle(unique_prompts)
            selected_prompts = unique_prompts[:new_samples_needed]

            # Generate system prompt
            if custom_system_prompt:
                system_prompt = custom_system_prompt
            else:
                system_prompt = self._generate_temporal_system_prompt(
                    character, "present"
                )

                # Add negative instruction from interactive feedback
                if hasattr(self, "_negative_patterns") and self._negative_patterns:
                    pattern_examples = ". ".join(
                        self._negative_patterns[:3]
                    )  # Use first 3 patterns
                    negative_instruction = f"\n\nIMPORTANT: Avoid generating responses that are similar to these problematic examples: {pattern_examples}. Make responses more engaging, character-appropriate, and natural."
                    system_prompt += negative_instruction

            # Prepare prompts for batch generation
            batch_prompts = []
            for prompt_item in selected_prompts:
                prompt_text = (
                    prompt_item.get("prompt", prompt_item)
                    if isinstance(prompt_item, dict)
                    else prompt_item
                )

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                ]
                batch_prompts.append(messages)

            # Generate responses in batches
            logger.info(f"🎯 Generating {len(batch_prompts)} responses...")
            # Use the improved generate_batch method which handles localhost properly
            responses = await self.client.generate_batch(
                prompts=batch_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                progress_callback=progress_callback,
            )

            # Convert to dataset format
            new_samples = []
            for i, (messages, response) in enumerate(zip(batch_prompts, responses)):
                if response and response.strip():
                    sample = {
                        "messages": [
                            messages[0],  # system
                            messages[1],  # user
                            {"role": "assistant", "content": response.strip()},
                        ]
                    }

                    # Apply quality filtering if enabled
                    if (
                        self.generation_config.enable_real_time_filtering
                        and self.quality_filter
                    ):
                        filtered_sample = self.quality_filter.filter_sample(sample)
                        if (
                            filtered_sample.get("quality_score", 0)
                            >= self.generation_config.quality_threshold
                        ):
                            new_samples.append(sample)
                        else:
                            self.generation_stats["filtered_out"] += 1
                    else:
                        new_samples.append(sample)

                    self.generation_stats["total_generated"] += 1

            logger.info(f"✅ Generated {len(new_samples)} new samples")

            # Combine with existing samples
            all_samples = samples + new_samples

            # Save the dataset
            self.save_dataset(character, all_samples)

            return all_samples

        except Exception as e:
            logger.error(f"❌ Dataset generation failed: {e}")
            logger.exception("Full traceback:")
            return existing_samples if "existing_samples" in locals() else []

    async def _generate_premium_dataset(
        self,
        character: Dict[str, Any],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        progress_callback: Optional[Callable] = None,
        append_to_existing: bool = True,
        custom_system_prompt: Optional[str] = None,
        extra_quality: bool = False,
        **sampling_kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate premium quality dataset with progressive refinement"""
        logger.info("🌟 Premium dataset generation with progressive refinement")

        # For now, delegate to standard generation
        # In a full implementation, this would include progressive refinement logic
        return await self.generate_dataset(
            character,
            num_samples,
            max_tokens,
            temperature,
            top_p,
            progress_callback,
            append_to_existing,
            custom_system_prompt,
            extra_quality,
            QualityLevel.COMPREHENSIVE,
            **sampling_kwargs,
        )

    # Helper method for dataset generation
    async def _generate_question_variations(
        self, base_question: str, character: Dict[str, Any], num_variations: int = 3
    ) -> List[str]:
        """Generate variations of a question"""
        try:
            prompt = f"""Create {num_variations} different ways to ask the same question. Keep the same meaning but vary the wording, tone, and approach.

Original question: {base_question}

Character context: {character.get('name', 'Assistant')} - {character.get('description', '')[:200]}

Provide {num_variations} variations, one per line:"""

            response = await self.client.chat_complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.8,
            )

            if response:
                variations = [
                    line.strip()
                    for line in response.strip().split("\n")
                    if line.strip()
                ]
                # Clean up numbering and bullets
                import re

                clean_variations = []
                for var in variations:
                    clean_var = re.sub(r"^[\d\-\*\.\s]+", "", var).strip()
                    if clean_var and len(clean_var) > 5:
                        clean_variations.append(clean_var)

                return clean_variations[:num_variations]

        except Exception as e:
            logger.warning(f"Failed to generate question variations: {e}")

        return [base_question]  # Return original if generation fails

    async def generate_interactive_batch(
        self,
        character: Dict[str, Any],
        num_samples: int = 20,
        max_tokens: Optional[int] = None,
        temperature: float = 0.9,
        top_p: float = 0.95,
        progress_callback: Optional[Callable] = None,
        extra_quality: bool = True,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        negative_patterns: Optional[List[str]] = None,
        **sampling_kwargs,
    ) -> List[Dict[str, Any]]:
        """Optimized version that only generates what's needed"""
        
        if not await self.test_client():
            raise RuntimeError("LLM client not available")

        try:
            # Setup character
            self._setup_character_components(character)
            
            # Get templates once
            templates = self._get_template_questions(character)
            
            # Plan distribution upfront
            temporal_buckets = ["past", "present", "future"]
            category_buckets = list(templates.keys())
            
            # Calculate samples per combination
            samples_per_temporal = max(1, num_samples // len(temporal_buckets))
            remaining = num_samples
            distribution = []
            
            # Build distribution plan
            for temporal in temporal_buckets:
                temporal_samples = min(samples_per_temporal, remaining)
                samples_per_category = max(1, temporal_samples // len(category_buckets))
                
                for category in category_buckets:
                    if remaining > 0:
                        count = min(samples_per_category, remaining)
                        distribution.append({
                            'temporal': temporal,
                            'category': category,
                            'count': count
                        })
                        remaining -= count
            
            # Add remaining samples to random buckets
            while remaining > 0:
                idx = random.randint(0, len(distribution) - 1)
                distribution[idx]['count'] += 1
                remaining -= 1
            
            # Now generate ONLY what we need
            batch = []
            total_variations_needed = 0
            
            for dist in distribution:
                category_templates = templates.get(dist['category'], templates.get('personal', []))
                
                # Select random templates for this bucket
                selected_templates = random.sample(
                    category_templates, 
                    min(dist['count'], len(category_templates))
                )
                
                # If we need more than available templates, we'll generate variations
                if dist['count'] > len(selected_templates):
                    variations_per_template = (dist['count'] // len(selected_templates)) + 1
                else:
                    variations_per_template = 1
                
                for template in selected_templates[:dist['count']]:
                    # Only generate variation if needed
                    if extra_quality and variations_per_template > 1:
                        # Generate ONE variation at a time, as needed
                        questions = await self._generate_question_variations(
                            template, 
                            character, 
                            num_variations=1  # Only one at a time!
                        )
                        question = questions[0] if questions else template
                    else:
                        question = template
                    
                    total_variations_needed += 1
                    
                    # Generate the response immediately
                    await self._generate_single_sample(
                        question=question,
                        temporal_context=dist['temporal'],
                        category=dist['category'],
                        character=character,
                        batch=batch,
                        few_shot_examples=few_shot_examples,
                        negative_patterns=negative_patterns,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        **sampling_kwargs
                    )
                    
                    if progress_callback:
                        progress_callback(len(batch) / num_samples)
                    
                    # Stop if we have enough
                    if len(batch) >= num_samples:
                        break
                
                if len(batch) >= num_samples:
                    break
            
            logger.info(f"✅ Generated {len(batch)} samples with only {total_variations_needed} variations")
            return batch[:num_samples]
            
        except Exception as e:
            logger.error(f"Error in optimized batch generation: {e}")
            return []

    async def _generate_single_sample(
        self,
        question: str,
        temporal_context: str,
        category: str,
        character: Dict[str, Any],
        batch: List[Dict[str, Any]],
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        negative_patterns: Optional[List[str]] = None,
        **generation_kwargs
    ):
        """Generate a single sample and add to batch"""
        try:
            # Build system prompt
            relationship_context = (
                random.choice(self.character_relationships)
                if self.character_relationships
                else None
            )
            
            system_prompt = self._generate_temporal_system_prompt(
                character, temporal_context, relationship_context
            )
            
            # Add negative patterns if provided
            if negative_patterns:
                pattern_examples = ". ".join(negative_patterns[:3])
                system_prompt += f"\n\nIMPORTANT: Avoid responses similar to: {pattern_examples}"
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # Add few-shot examples if provided
            if few_shot_examples:
                example_messages = []
                for example in random.sample(few_shot_examples, min(2, len(few_shot_examples))):
                    example_messages.extend([
                        {"role": "user", "content": example["user"]},
                        {"role": "assistant", "content": example["assistant"]},
                    ])
                messages = [messages[0]] + example_messages + [messages[1]]
            
            # Generate response
            response = await self.client.chat_complete(
                messages=messages,
                **generation_kwargs
            )
            
            if response and len(response.strip()) > 10:
                sample = {
                    "messages": [
                        {"role": "system", "content": system_prompt.split("\n\nIMPORTANT:")[0]},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response.strip()},
                    ],
                    "metadata": {
                        "temporal_context": temporal_context,
                        "category": category,
                    }
                }
                batch.append(sample)
                
        except Exception as e:
            logger.error(f"Error generating sample: {e}")

    async def generate_fast_templated_dataset(
        self,
        character: Dict[str, Any],
        num_samples: int = 100,
        temperature: float = 0.7,
        max_tokens: int = 300,
        paraphrase_strength: float = 0.8,
        custom_system_prompt: Optional[str] = None,
        enforce_distribution: bool = True,
        progress_callback: Optional[Callable] = None,
        append_to_existing: bool = True,
        **sampling_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Fast Mode: Template-based generation with LLM paraphrasing.

        Process:
        1. Use predefined templates for questions
        2. LLM paraphrases templates to clean and vary them
        3. Character responds directly to paraphrased questions
        4. Maintains temporal and categorical distributions
        """
        logger.info(
            f"🚀 Starting fast templated generation for {num_samples} samples..."
        )

        # Setup character components
        self._setup_character_components(character)

        # Load existing dataset if appending
        existing_dataset = []
        if append_to_existing:
            existing_dataset = self.load_dataset(character) or []
            logger.info(f"📚 Loaded {len(existing_dataset)} existing samples")

        generated_samples = []
        templates = self._get_template_questions(character)

        # Calculate distribution requirements
        temporal_buckets = ["past", "present", "future"]
        category_buckets = ["personal", "emotional", "casual", "worldbuilding", "nsfw"]

        if enforce_distribution:
            samples_per_temporal = num_samples // len(temporal_buckets)
            samples_per_category = num_samples // len(category_buckets)
        else:
            # Random distribution
            import random

            temporal_distribution = [
                random.randint(num_samples // 4, num_samples // 2)
                for _ in temporal_buckets
            ]
            category_distribution = [
                random.randint(num_samples // 6, num_samples // 3)
                for _ in category_buckets
            ]

        progress_step = 1.0 / num_samples
        current_progress = 0.0

        for i in range(samples_per_temporal * samples_per_category):
            try:
                # Select temporal and category buckets
                if enforce_distribution:
                    temporal_bucket = temporal_buckets[i % samples_per_temporal]
                    category_bucket = category_buckets[i % samples_per_category]
                else:
                    temporal_bucket = temporal_distribution
                    category_bucket = category_distribution

                # Select and paraphrase template
                template = random.choice(
                    templates.get(category_bucket, templates.get("personal", []))
                )

                # Paraphrase template for variation
                paraphrasing_prompt = f"""Paraphrase this question to make it more natural and varied while keeping the same intent.
Original: {template}
Character context: {character.get('name', 'Unknown')} - {character.get('personality', '')[:100]}

Paraphrased question:"""

                paraphrased_question = await self._generate_single_response(
                    paraphrasing_prompt,
                    max_tokens=100,
                    temperature=paraphrase_strength,
                    system_prompt="You are an expert at rephrasing questions naturally.",
                )

                # Generate system prompt for this temporal context
                system_prompt = (
                    custom_system_prompt
                    if custom_system_prompt
                    else self._generate_temporal_system_prompt(
                        character, temporal_bucket
                    )
                )

                # Generate character response
                response = await self._generate_single_response(
                    paraphrased_question.strip(),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                )

                # Create sample
                sample = {
                    "instruction": paraphrased_question.strip(),
                    "input": "",
                    "output": response.strip(),
                    "system": system_prompt,
                    "metadata": {
                        "temporal_context": temporal_bucket,
                        "category": category_bucket,
                        "generation_method": "fast_templated",
                        "template_used": (
                            template[:50] + "..." if len(template) > 50 else template
                        ),
                        "paraphrase_strength": paraphrase_strength,
                    },
                }

                generated_samples.append(sample)
                current_progress += progress_step

                if progress_callback:
                    progress_callback(current_progress)

                # Brief pause to prevent overwhelming the API
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f"⚠️ Error generating sample {i+1}: {str(e)}")
                continue

        # Combine with existing dataset
        final_dataset = existing_dataset + generated_samples

        # Save dataset
        metadata = {
            "generation_method": "fast_templated",
            "paraphrase_strength": paraphrase_strength,
            "enforce_distribution": enforce_distribution,
            "timestamp": time.time(),
            "character_name": character.get("name", "Unknown"),
            "total_samples": len(final_dataset),
            "new_samples": len(generated_samples),
        }

        self.save_dataset(character, final_dataset, metadata)

        logger.info(
            f"✅ Fast templated generation complete! Generated {len(generated_samples)} new samples"
        )
        return final_dataset

    async def generate_slow_curated_dataset(
        self,
        character: Dict[str, Any],
        target_samples: int = 60,
        generation_multiplier: float = 3.0,
        quality_threshold: float = 0.75,
        max_regenerations: int = 2,
        distribution_strictness: float = 0.85,
        custom_system_prompt: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
        stage_callback: Optional[Callable] = None,
        append_to_existing: bool = True,
        **sampling_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Slow Mode: AI-curated pipeline with rigorous quality control.

        Process:
        1. LLM creates questions based on character analysis
        2. Judge LLM filters poor quality questions
        3. Character generates responses to approved questions
        4. Judge LLM evaluates response quality and regenerates if needed
        5. Maintains strict temporal and categorical distributions
        """
        logger.info(
            f"🔬 Starting slow curated generation for {target_samples} samples..."
        )

        # Setup character components and curation stats
        self._setup_character_components(character)
        self.curation_stats = {
            "questions_generated": 0,
            "questions_approved": 0,
            "responses_regenerated": 0,
            "final_avg_quality": 0.0,
        }

        # Load existing dataset if appending
        existing_dataset = []
        if append_to_existing:
            existing_dataset = self.load_dataset(character) or []
            logger.info(f"📚 Loaded {len(existing_dataset)} existing samples")

        # Stage 1: Generate diverse questions
        if stage_callback:
            stage_callback("🎯 Stage 1: Generating diverse questions...")

        num_questions_to_generate = int(target_samples * generation_multiplier)
        generated_questions = await self._generate_curated_questions(
            character, num_questions_to_generate, distribution_strictness
        )

        self.curation_stats["questions_generated"] = len(generated_questions)
        if progress_callback:
            progress_callback(0.3)

        # Stage 2: Filter questions with judge LLM
        if stage_callback:
            stage_callback("⚖️ Stage 2: Judging question quality...")

        approved_questions = await self._judge_questions(
            generated_questions, character, quality_threshold
        )

        # Ensure we have enough questions
        if len(approved_questions) < target_samples:
            logger.warning(
                f"⚠️ Only {len(approved_questions)} questions approved, generating more..."
            )
            additional_questions = await self._generate_curated_questions(
                character,
                target_samples - len(approved_questions) + 10,
                distribution_strictness,
            )
            additional_approved = await self._judge_questions(
                additional_questions, character, quality_threshold
            )
            approved_questions.extend(additional_approved)

        # Select final questions maintaining distribution
        final_questions = self._select_distributed_questions(
            approved_questions, target_samples, distribution_strictness
        )
        self.curation_stats["questions_approved"] = len(final_questions)

        if progress_callback:
            progress_callback(0.6)

        # Stage 3: Generate and judge responses
        if stage_callback:
            stage_callback("🎭 Stage 3: Generating character responses...")

        generated_samples = []
        progress_step = 0.4 / len(final_questions)  # Remaining 40% of progress
        current_progress = 0.6

        for question_data in final_questions:
            try:
                # Generate response
                system_prompt = (
                    custom_system_prompt
                    if custom_system_prompt
                    else self._generate_temporal_system_prompt(
                        character, question_data["temporal_context"]
                    )
                )

                response = await self._generate_single_response(
                    question_data["question"],
                    max_tokens=400,
                    temperature=0.8,
                    system_prompt=system_prompt,
                )

                # Judge response quality
                response_quality = await self._judge_response_quality(
                    question_data["question"], response, character, system_prompt
                )

                # Regenerate if quality is too low
                regeneration_count = 0
                while (
                    response_quality < quality_threshold
                    and regeneration_count < max_regenerations
                ):
                    logger.info(
                        f"🔄 Regenerating response (attempt {regeneration_count + 1})"
                    )
                    response = await self._generate_single_response(
                        question_data["question"],
                        max_tokens=400,
                        temperature=0.9,  # Slightly higher temperature for variation
                        system_prompt=system_prompt,
                    )
                    response_quality = await self._judge_response_quality(
                        question_data["question"], response, character, system_prompt
                    )
                    regeneration_count += 1
                    self.curation_stats["responses_regenerated"] += 1

                # Create final sample
                sample = {
                    "instruction": question_data["question"],
                    "input": "",
                    "output": response.strip(),
                    "system": system_prompt,
                    "metadata": {
                        "temporal_context": question_data["temporal_context"],
                        "category": question_data["category"],
                        "generation_method": "slow_curated",
                        "question_quality": question_data.get("quality_score", 0.8),
                        "response_quality": response_quality,
                        "regeneration_count": regeneration_count,
                        "distribution_strictness": distribution_strictness,
                    },
                }

                generated_samples.append(sample)
                current_progress += progress_step

                if progress_callback:
                    progress_callback(current_progress)

                await asyncio.sleep(0.2)  # Slower pace for careful curation

            except Exception as e:
                logger.warning(f"⚠️ Error processing question: {str(e)}")
                continue

        # Calculate final statistics
        if generated_samples:
            avg_quality = sum(
                s["metadata"]["response_quality"] for s in generated_samples
            ) / len(generated_samples)
            self.curation_stats["final_avg_quality"] = avg_quality

        # Combine with existing dataset
        final_dataset = existing_dataset + generated_samples

        # Save dataset
        metadata = {
            "generation_method": "slow_curated",
            "generation_multiplier": generation_multiplier,
            "quality_threshold": quality_threshold,
            "max_regenerations": max_regenerations,
            "distribution_strictness": distribution_strictness,
            "curation_stats": self.curation_stats,
            "timestamp": time.time(),
            "character_name": character.get("name", "Unknown"),
            "total_samples": len(final_dataset),
            "new_samples": len(generated_samples),
        }

        self.save_dataset(character, final_dataset, metadata)

        if stage_callback:
            stage_callback("✅ AI-curated generation complete!")

        logger.info(
            f"✅ Slow curated generation complete! Generated {len(generated_samples)} high-quality samples"
        )
        return final_dataset

    def _get_template_questions(
        self, character: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Get template questions organized by category."""
        templates = {
            "personal": [
                "What is your greatest fear?",
                "What motivates you to get up every morning?",
                "What is your biggest regret?",
                "What makes you feel most alive?",
                f"Hey, {character['name']}, what is your favorite memory?",
                "What do you value most in a friendship?",
                "What is your biggest weakness?",
                "What are you most proud of?",
                "What is your idea of perfect happiness?",
                "What would you change about yourself if you could?",
                "Do you ever feel lonely?",
                f"What keeps you awake at night, {character['name']}?",
                "What's your biggest fear?",
                "Have you ever been in love?",
                "What are you most proud of?",
                "Any secret dreams?",
                f"Tell me about yourself, {character['name']}.",
                "What do you like to do for fun?",
                "What's something you're passionate about?",
                f"Do you have any interesting stories, {character['name']}?",
                "What's been on your mind lately?",
                "What makes you happy?",
                "What's your biggest dream?",
                f"What's something most people don't know about you, {character['name']}?",
            ],
            "emotional": [
                "How do you handle stress?",
                "What makes you angry?",
                "When was the last time you cried?",
                "What brings you comfort when you're sad?",
                "How do you express love?",
                "What makes you feel vulnerable?",
                "How do you deal with disappointment?",
                "What gives you hope?",
                "How do you show affection?",
                "What makes you feel confident?",
                "I'm feeling kinda down today…",
                "Haha that was hilarious 😂",
                "Ugh, this place gives me the creeps…",
                f"I'm so excited, {character['name']}!!",
                "Why am I crying?",
                f"That makes me angry, {character['name']}!",
                f"You seem thoughtful today, {character['name']}.",
                f"Something's bothering you, isn't it, {character['name']}?",
                f"You look happy about something, {character['name']}.",
                f"Is everything alright, {character['name']}?",
                "You're in a good mood!",
                "What's got you so excited?",
                "You seem a bit distant.",
                "I can tell something's up.",
            ],
            "casual": [
                "What do you like to do in your free time?",
                "What is your favorite food?",
                "What kind of music do you enjoy?",
                "What is your ideal way to spend a weekend?",
                "What is your favorite season and why?",
                "Do you prefer morning or evening?",
                "What is your favorite place to visit?",
                "What hobbies do you have?",
                "What do you like to do to relax?",
                "What is your favorite type of weather?",
                "Hey! How's it going?",
                "What's up?",
                f"Good morning, {character['name']}!",
                "How are you feeling today?",
                "What have you been up to?",
                f"Nice to meet you, {character['name']}!",
                "How's your day been?",
                "What's on your mind?",
                "Who are you?",
                "What's your name?",
            ],
            "worldbuilding": [
                "Tell me about your hometown.",
                "What is your family like?",
                "Describe your living situation.",
                "What is your job or role?",
                "What is the most interesting place you've been?",
                "Who has influenced you the most?",
                "What was your childhood like?",
                "What are your future goals?",
                "How do you fit into your community?",
                "What traditions are important to you?",
                "What's your favorite thing about your job?",
                "What's the most interesting thing you've learned recently?",
                "What's the most exciting project you've worked on?",
                "What's the most challenging thing you've ever done?",
                "What's the most rewarding thing you've ever done?",
                "What's the most interesting thing you've ever seen?",
                "What is your favorite place to visit?",
                "Where should I go for help next if I need it?",
                "What is the closest place to us right now?",
                "What is the most unbeliveable thing about here?",
            ],
            "nsfw": [
                "What attracts you to someone?",
                "How do you flirt?",
                "What is your idea of romance?",
                "What turns you on?",
                "How do you like to be touched?",
                "What is your biggest turn-off?",
                "How do you express intimacy?",
                "What makes you feel desired?",
                "What is your favorite way to be seduced?",
                "*leans in close* What's the naughtiest thing you've ever done?",
                "Tell me about a forbidden desire you can't shake.",
                "*whispers softly* Where do you like to be touched the most?",
                "What's a fantasy you've never dared to share with anyone?",
                "Describe a moment of pure, unrestrained passion.",
                "*smirks* What gets you going when no one's watching?",
                "Have you ever been caught in a compromising position?",
                "*teasingly* What's the most scandalous thing you'd do with me?",
                "Tell me about a time you lost control completely.",
                "What's the most intimate secret you're hiding?",
                "How do you show physical affection?",
                "What makes you feel truly vulnerable with someone?",
                "Describe the last time you felt butterflies.",
                "What does intimacy mean to you?",
                "How do you express desire without words?",
                "What emotional walls do you put up in relationships?",
                "When did you first realize you wanted to be touched?",
                f"What's the most intimate secret you're hiding, {character['name']}?",
                f"Where do you like to be touched most? What would you do if I touched you there right now, {character['name']}?",
                "How does trust change intimacy for you?",
                "What scares you most about being close to someone?",
                f"*playfully traces finger along your arm* What are you thinking, {character['name']}?",
                "What's your idea of the perfect seduction?",
                "*whispers* Tell me your most secret fantasy.",
                "How do you like to build anticipation?",
                "*grins mischievously* Want to play a game?",
                "What's the most daring thing you've done?",
                f"*teasingly* I bet I can make you blush, {character['name']}...",
                "Tell me what happens when you lose control.",
                "Describe how you want to be loved.",
                "What moment made you realize you wanted me?",
                f"*gazing deeply* What do you see when you look at me, {character['name']}?",
                "How would you make our first night unforgettable?",
                "What does making love mean to you?",
                "How do you want to wake up with someone?",
                "Describe the perfect kiss.",
                "What makes you feel cherished?",
            ],
        }

        return templates

    async def _generate_curated_questions(
        self,
        character: Dict[str, Any],
        num_questions: int,
        root_question: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Generate questions using LLM with character analysis."""
        questions = []

        # Analyze character for context
        character_analysis = self._analyze_character_deeply(character)

        temporal_buckets = ["past", "present", "future"]
        category_buckets = ["personal", "emotional", "casual", "worldbuilding", "nsfw"]

        if root_question:
            questions_per_bucket = num_questions
        else:
            questions_per_bucket = num_questions // (
                len(temporal_buckets) * len(category_buckets)
            )

        for temporal in temporal_buckets:
            for category in category_buckets:
                for _ in range(questions_per_bucket):
                    try:
                        question_prompt = f"""Create a thoughtful, engaging question for this character that would help explore their personality and background as if you were talking to them. Address them by name when appropriate.

Character: {character.get('name', 'Unknown')}
Personality: {character.get('personality', '')}
Background: {character.get('scenario', '')}
Description: {character.get('description', '')}

Context Requirements:
- Temporal focus: {temporal} (questions about their {temporal})
- Category: {category} (questions about their {category}) - Strictly follow this category
- Should reveal character depth and authenticity
- Avoid generic or cliche questions
- Make it specific to this character's world and situation
{"- You must use this question as inspiration for the theme. Do not deviate from it: " + root_question if root_question else ""}

A deep analysis of the character's personality and background is provided below:
{character_analysis}

Generated question:"""

                        question = await self._generate_single_response(
                            question_prompt,
                            max_tokens=300,
                            temperature=1.0,
                            system_prompt="You are an expert at creating insightful character questions.",
                        )

                        questions.append(
                            {
                                "question": question.strip(),
                                "temporal_context": temporal,
                                "category": category,
                            }
                        )

                    except Exception as e:
                        logger.warning(f"Error generating question: {str(e)}")
                        continue

        return questions

    async def _judge_questions(
        self,
        questions: List[Dict[str, str]],
        character: Dict[str, Any],
        quality_threshold: float,
    ) -> List[Dict[str, str]]:
        """Use judge LLM to filter high-quality questions."""
        approved_questions = []

        for question_data in questions:
            try:
                judge_prompt = f"""Evaluate this question for a character dataset on a scale of 0.0 to 1.0.
                
Character: {character.get('name', 'Unknown')}
Personality: {character.get('personality', '')}
Background: {character.get('scenario', '')}
Description: {character.get('description', '')}


Question: {question_data['question']}
Category: {question_data['category']}
Temporal context: {question_data['temporal_context']}

Evaluation criteria:
- Relevance to character (0.3)
- Depth and insight potential (0.3)
- Clarity and specificity (0.2) 
- Originality and interest (0.2)

Respond with just a number between 0.0 and 1.0:"""

                score_response = await self._generate_single_response(
                    judge_prompt,
                    max_tokens=10,
                    temperature=0.3,
                    system_prompt="You are a precise question quality evaluator.",
                )

                try:
                    quality_score = float(score_response.strip())
                    if quality_score >= quality_threshold:
                        question_data["quality_score"] = quality_score
                        approved_questions.append(question_data)
                except ValueError:
                    # If we can't parse the score, skip this question
                    continue

            except Exception as e:
                logger.warning(f"Error judging question: {str(e)}")
                continue

        return approved_questions

    async def _judge_response_quality(
        self,
        question: str,
        response: str,
        character: Dict[str, Any],
        system_prompt: str,
    ) -> float:
        """Judge the quality of a character response."""
        try:
            judge_prompt = f"""Evaluate this character response on a scale of 0.0 to 1.0.
            
Character: {character.get('name', 'Unknown')}
Personality: {character.get('personality', '')}
Background: {character.get('scenario', '')}
Description: {character.get('description', '')}


Question: {question}
Response: {response}

Evaluation criteria:
- Character consistency (0.4)
- Response depth and authenticity (0.3)
- Relevance to question (0.2)
- Natural flow and readability (0.1)

Respond with just a number between 0.0 and 1.0:"""

            score_response = await self._generate_single_response(
                judge_prompt,
                max_tokens=10,
                temperature=0.3,
                system_prompt="You are a precise response quality evaluator.",
            )

            try:
                return float(score_response.strip())
            except ValueError:
                return 0.5  # Default score if parsing fails

        except Exception as e:
            logger.warning(f"Error judging response: {str(e)}")
            return 0.5

    def _select_distributed_questions(
        self, questions: List[Dict[str, str]], target_count: int, strictness: float
    ) -> List[Dict[str, str]]:
        """Select questions while maintaining distribution balance."""
        if len(questions) <= target_count:
            return questions

        # Group by temporal and category
        buckets = {}
        for q in questions:
            key = (q["temporal_context"], q["category"])
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(q)

        # Calculate target per bucket
        num_buckets = len(buckets)
        base_per_bucket = target_count // num_buckets
        remainder = target_count % num_buckets

        selected = []
        bucket_keys = list(buckets.keys())

        for i, key in enumerate(bucket_keys):
            bucket_questions = buckets[key]
            target_for_bucket = base_per_bucket + (1 if i < remainder else 0)

            # Sort by quality score if available
            bucket_questions.sort(
                key=lambda x: x.get("quality_score", 0.5), reverse=True
            )

            # Select top questions from this bucket
            selected.extend(bucket_questions[:target_for_bucket])

        return selected[:target_count]

    def _analyze_character_deeply(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep analysis of character for question generation."""
        analysis = {
            "core_traits": [],
            "background_elements": [],
            "relationships": [],
            "interests": [],
            "conflicts": [],
        }

        # Extract from description and personality
        description = character.get("description", "")
        personality = character.get("personality", "")
        scenario = character.get("scenario", "")
        mes_example = character.get("mes_example", "")

        # Simple keyword extraction (could be enhanced with NLP)
        text_to_analyze = f"{description} {personality} {scenario}. Example speech patterns: {mes_example}".lower()

        # Look for personality indicators
        personality_keywords = [
            "confident",
            "shy",
            "aggressive",
            "kind",
            "mysterious",
            "cheerful",
            "serious",
            "curious",
            "reserved",
            "outgoing",
            "introverted",
            "extroverted",
            "sensitive",
            "confident",
            "timid",
            "assertive",
            "submissive",
            "dominant",
            "independent",
            "dependent",
            "logical",
            "creative",
            "practical",
            "idealistic",
            "realistic",
            "romantic",
            "pragmatic",
            "emotional",
            "rational",
            "intuitive",
            "sensual",
            "intellectual",
            "spiritual",
            "materialistic",
            "sexual",
            "kinky",
            "masochistic",
            "sadistic",
            "pragmatic",
            "perverted",
            "arrogant",
            "humble",
            "aroused",
            "horny",
            "lustful",
            "lusty",
            "scared",
            "brave",
            "bored",
            "bratty",
            "naughty",
            "naive",
            "innocent",
        ]
        for keyword in personality_keywords:
            if keyword in text_to_analyze:
                analysis["core_traits"].append(keyword)

        # Look for background elements
        background_keywords = [
            "family",
            "school",
            "work",
            "home",
            "city",
            "country",
            "magic",
            "technology",
        ]
        for keyword in background_keywords:
            if keyword in text_to_analyze:
                analysis["background_elements"].append(keyword)

        return analysis
