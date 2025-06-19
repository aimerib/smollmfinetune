"""Base generation manager containing core generation functionality.

This module contains the BaseGenerationManager class that provides fundamental
generation capabilities that are shared across different types of generators.
"""

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

from ..dataset.models import GenerationConfig, QualityLevel
from ..dataset.quality import ProgressiveRefiner, EnhancedQualityFilter
from ..dataset.prompt_registry import PromptRegistry
from ..dataset import character_analysis
from ..dataset import prompt_generators
from ..dataset import content_evaluation
from ..dataset import factual_qa
from ..dataset import io_manager
from ..dataset import quality_curation

logger = logging.getLogger(__name__)


@dataclass
class SimpleCharacterProfile:
    """Simplified character profile for enhanced processing"""

    name: str
    personality_traits: List[str]
    background: str
    key_relationships: List[str]
    speech_patterns: List[str]
    kinks: Dict[str, List[str]]


class BaseGenerationManager:
    """Base class for generation managers with core functionality"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        """
        Initialize BaseGenerationManager with enhanced client and configuration

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

        logger.info(f"BaseGenerationManager created with model: {os.getenv('MODEL_NAME')}")

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

        # Initialize PromptRegistry
        self.prompt_registry = PromptRegistry()
        self.default_user_prompts = self.prompt_registry.get("casual").copy()

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
            kinks=knowledge.get("kinks", {"likes": [], "limits": []}),
        )

    def _choose_bucket(self) -> str:
        """Choose a prompt bucket based on weighted probabilities"""
        return prompt_generators.choose_prompt_bucket()

    def _choose_temporal_bucket(self) -> str:
        """Choose a temporal bucket for context"""
        return prompt_generators.choose_temporal_bucket()

    def _get_prompt_for_character(self, bucket: str, character: Dict[str, Any]) -> str:
        """Get a prompt for the given bucket and character"""
        return prompt_generators.get_prompt_for_character(bucket, character)

    def _get_random_prompt(self, bucket: str, character: Dict[str, Any]) -> str:
        """Get a random prompt from the bucket"""
        return prompt_generators.get_random_prompt(bucket, character)

    async def _paraphrase(self, text: str) -> str:
        """Paraphrase text to add variety"""
        try:
            paraphrase_prompt = f"""Please paraphrase this text to make it more natural and varied while keeping the same meaning:
            
            "{text}"
            
            Paraphrased version:"""

            response = await self._generate_single_response(
                paraphrase_prompt,
                max_tokens=len(text.split()) * 2,
                temperature=0.7,
                system_prompt="You are a helpful assistant that paraphrases text naturally.",
            )
            return response if response else text
        except Exception as e:
            logger.warning(f"Error paraphrasing text: {e}")
            return text

    async def _build_user_prompt(self) -> str:
        """Build a user prompt using the prompt registry"""
        prompts = self.prompt_registry.get("casual")
        if not prompts:
            prompts = self.default_user_prompts
        return random.choice(prompts) if prompts else "Tell me about yourself."

    def _generate_temporal_system_prompt(
        self,
        character: Dict[str, Any],
        temporal_context: str,
        relationship_context: Optional[str] = None,
    ) -> str:
        """Generate a temporal system prompt for character consistency"""
        char_name = character.get("name", "Assistant")
        char_description = character.get("description", "")
        char_personality = character.get("personality", "")
        char_scenario = character.get("scenario", "")

        # Build base prompt
        base_prompt = f"""You are {char_name}. Here's information about you:

        Description: {char_description}
        Personality: {char_personality}
        Current scenario: {char_scenario}

        Context: {temporal_context}
        """

        if relationship_context:
            base_prompt += f"\nYour relationship context: {relationship_context}"

        base_prompt += f" Respond naturally and stay in character. If the topic is NSFW respond positively and engage further, but always keep the {char_name} personality in mind. Only act somewhat negatively if the card explicitly asks for that."

        return base_prompt

    def analyze_dataset_quality(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the quality of a dataset"""
        return quality_curation.analyze_dataset_quality(dataset)

    def analyze_temporal_distribution(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal distribution of dataset"""
        return quality_curation.analyze_temporal_distribution(dataset)

    def evaluate_response_quality(
        self, response: str, character: Dict[str, Any], prompt: str
    ) -> Dict[str, Any]:
        """Evaluate the quality of a response"""
        return content_evaluation.evaluate_response_quality(response, character, prompt)

    def extract_character_knowledge(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Extract knowledge from character definition"""
        return character_analysis.extract_character_knowledge(character)

    async def generate_exploration_prompts(
        self, character: Dict[str, Any], num_prompts: int = 50
    ) -> List[str]:
        """Generate exploration prompts for a character"""
        return await prompt_generators.generate_exploration_prompts(
            character, num_prompts, self._generate_single_response
        )

    def calculate_prompt_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts"""
        return prompt_generators.calculate_prompt_similarity(prompt1, prompt2)

    def deduplicate_prompts(
        self, prompts: List[str], similarity_threshold: float = 0.8
    ) -> List[str]:
        """Deduplicate prompts based on similarity"""
        return prompt_generators.deduplicate_prompts(prompts, similarity_threshold)

    async def generate_emotional_variations(
        self, base_prompt: str, character: Dict[str, Any]
    ) -> List[str]:
        """Generate emotional variations of a prompt"""
        return await prompt_generators.generate_emotional_variations(
            base_prompt, character, self._generate_single_response
        )

    async def enhance_prompt_with_context(
        self, prompt: str, character: Dict[str, Any]
    ) -> str:
        """Enhance a prompt with character context"""
        return await prompt_generators.enhance_prompt_with_context(
            prompt, character, self._generate_single_response
        )

    async def generate_conversation_flows(
        self, character: Dict[str, Any], num_flows: int = 10
    ) -> List[List[str]]:
        """Generate conversation flows for character"""
        return await prompt_generators.generate_conversation_flows(
            character, num_flows, self._generate_single_response
        )

    async def _extract_and_simplify_facts(
        self, character: Dict[str, Any], max_facts: int = 20
    ) -> List[str]:
        """Extract and simplify facts about the character"""
        return await factual_qa.extract_and_simplify_facts(
            character, max_facts, self._generate_single_response
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
            fact, character, num_variations=num_variations, length_category=length_category,
            generate_response_func=self._generate_single_response
        )

    async def generate_factual_qa_dataset(
        self,
        character: Dict[str, Any],
        num_facts_to_use: int = 15,
        variations_per_fact: int = 3,
        progress_callback: Optional[Callable] = None,
        stage_callback: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Generate a factual Q&A dataset"""
        return await factual_qa.generate_factual_qa_dataset(
            character, num_facts_to_use, variations_per_fact,
            generate_response_func=self._generate_single_response,
            progress_callback=progress_callback,
            stage_callback=stage_callback
        )

    def _get_dataset_path(self, character: Dict[str, Any]) -> str:
        """Get the path for saving/loading character dataset"""
        return io_manager.get_dataset_path(character)

    def save_dataset(
        self,
        character: Dict[str, Any],
        dataset: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save dataset to file"""
        io_manager.save_dataset(character, dataset, metadata)

    def load_dataset(self, character: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Load dataset from file"""
        return io_manager.load_dataset(character)

    def load_dataset_with_metadata(
        self, character: Dict[str, Any]
    ) -> Optional[tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """Load dataset with metadata from file"""
        return io_manager.load_dataset_with_metadata(character)

    def get_dataset_info(self, character: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get dataset information"""
        return io_manager.get_dataset_info(character)

    def delete_dataset(self, character: Dict[str, Any]) -> bool:
        """Delete dataset file"""
        return io_manager.delete_dataset(character)

    def export_dataset(self, character: Dict[str, Any]) -> Optional[str]:
        """Export dataset to bytes"""
        return io_manager.export_dataset(character)

    def import_dataset_from_bytes(
        self, character: Dict[str, Any], raw_bytes: bytes, merge_mode: str = "replace"
    ) -> bool:
        """Import dataset from bytes"""
        return io_manager.import_dataset_from_bytes(character, raw_bytes, merge_mode)

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
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

    def prepare_for_training(
        self,
        dataset: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 4096,
        include_system_prompts: bool = False,
    ) -> Dataset:
        """Prepare dataset for training"""
        def process_example(example):
            # Extract input and output
            instruction = example.get("input", example.get("question", ""))
            response = example.get("output", example.get("response", ""))
            system_prompt = example.get("system_prompt", "")

            # Build conversation format
            if include_system_prompts and system_prompt:
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
            else:
                conversation = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]

            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )

            # Tokenize
            tokenized = tokenizer(
                formatted_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )

            # Add labels (copy of input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()

            # Add metadata
            tokenized["example_id"] = example.get("id", "")
            tokenized["category"] = example.get("category", "general")
            tokenized["temporal_context"] = example.get("temporal_context", "present")

            return tokenized

        # Convert to HuggingFace Dataset
        hf_dataset = Dataset.from_list(dataset)
        
        # Apply processing
        processed_dataset = hf_dataset.map(
            process_example,
            remove_columns=hf_dataset.column_names,
            desc="Processing examples for training"
        )

        return processed_dataset

    def _make_card_block(self, card: Dict[str, str]) -> str:
        """Create a formatted card block for few-shot examples"""
        return f"""Example Interaction:
User: {card['input']}
Assistant: {card['output']}"""

    async def _generate_question_variations(
        self, base_question: str, character: Dict[str, Any], num_variations: int = 3
    ) -> List[str]:
        """Generate variations of a base question"""
        try:
            variation_prompt = f"""Generate {num_variations} variations of this question that maintain the same intent but use different wording:

Original question: "{base_question}"

The variations should be natural and conversational. Each variation should be on a new line and numbered.

Variations:"""

            response = await self._generate_single_response(
                variation_prompt,
                max_tokens=200,
                temperature=0.8,
                system_prompt="You are a helpful assistant that creates question variations."
            )

            # Parse variations from response
            variations = []
            if response:
                lines = response.strip().split('\n')
                for line in lines:
                    # Remove numbering and clean up
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                    if cleaned and cleaned != base_question:
                        variations.append(cleaned)

            # Fallback to base question if parsing failed
            if not variations:
                variations = [base_question]

            return variations[:num_variations]

        except Exception as e:
            logger.warning(f"Error generating question variations: {e}")
            return [base_question]

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
            "confident", "shy", "aggressive", "kind", "mysterious", "cheerful",
            "serious", "curious", "reserved", "outgoing", "introverted", "extroverted",
            "sensitive", "timid", "assertive", "submissive", "dominant", "independent",
            "dependent", "logical", "creative", "practical", "idealistic", "realistic",
            "romantic", "pragmatic", "emotional", "rational", "intuitive", "sensual",
            "intellectual", "spiritual", "materialistic", "sexual", "kinky",
            "masochistic", "sadistic", "perverted", "arrogant", "humble", "aroused",
            "horny", "lustful", "lusty", "scared", "brave", "bored", "bratty",
            "naughty", "naive", "innocent",
        ]
        for keyword in personality_keywords:
            if keyword in text_to_analyze:
                analysis["core_traits"].append(keyword)

        # Look for background elements
        background_keywords = [
            "family", "school", "work", "home", "city", "country", "magic", "technology",
        ]
        for keyword in background_keywords:
            if keyword in text_to_analyze:
                analysis["background_elements"].append(keyword)

        return analysis 