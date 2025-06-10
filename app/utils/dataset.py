import asyncio
import random
import re
import textwrap
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from datasets import Dataset
from .inference_engines import get_inference_engine
import torch
import json
import os
import hashlib
import time as _time
import tempfile
import warnings
import gc

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages synthetic dataset generation and processing"""

    def __init__(self, preferred_engine: Optional[str] = None, enable_intelligent_generation: bool = True):
        logger.info(
            f"DatasetManager initializing with preferred_engine: {preferred_engine}")
        self.inference_engine = get_inference_engine(preferred_engine)
        logger.info(
            f"DatasetManager created with engine: {self.inference_engine.name}")
        # For DanChat-2 we only need a *single* chat template – the chat
        # wrapper (<|user|> …) is added later by vLLM/HF tokenizer.
        self.templates = [("chat", "{user_prompt}")]

        # Configuration for intelligent prompt generation
        self.enable_intelligent_generation = enable_intelligent_generation
        if not enable_intelligent_generation:
            logger.info(
                "🔧 Intelligent prompt generation is DISABLED - using static prompts only")

        # ✅ FIX: Add global lock to prevent concurrent temporal prompt generation
        self._global_temporal_lock = asyncio.Lock()

        # Natural conversation starters (how users actually talk) ----------------
        self.prompts_casual = [
            "Hey! How's it going?",
            "What's up?",
            "Good morning!",
            "How are you feeling today?",
            "What have you been up to?",
            "Nice to meet you!",
            "How's your day been?",
            "What's on your mind?",
            "Who are you?",
            "What's your name?",
        ]

        self.prompts_personal = [
            "do you ever feel lonely?", "what keeps you awake at night?",
            "what's your biggest fear?", "have you ever been in love?",
            "what are you most proud of?", "any secret dreams?",
            "Tell me about yourself.",
            "What do you like to do for fun?",
            "What's something you're passionate about?",
            "Do you have any interesting stories?",
            "What's been on your mind lately?",
            "What makes you happy?",
            "What's your biggest dream?",
            "What's something most people don't know about you?",
            "Who are you?",
            "What's your name?",
        ]

        self.prompts_action = [
            "*pushes the door open* you coming?",
            "quick, hide with me!", "help me pick this lock…",
            "look out! what's that?", "hold my hand and run!",
            "Want to go somewhere?",
            "Should we check that out?",
            "What do you think we should do?",
            "Ready for an adventure?",
            "Let's try something new.",
            "Come on, let's go!",
            "What's the plan?",
            "Want to explore a bit?"
        ]

        self.prompts_emotion = [
            "i'm feeling kinda down today…", "haha that was hilarious 😂",
            "ugh, this place gives me the creeps…", "i'm so excited!!",
            "why am i crying?", "that makes me angry!",
            "You seem thoughtful today.",
            "Something's bothering you, isn't it?",
            "You look happy about something.",
            "Is everything alright?",
            "You're in a good mood!",
            "What's got you so excited?",
            "You seem a bit distant.",
            "I can tell something's up."
        ]

        self.prompts_intimate = [
            "*whispers* what do you desire most?",
            "tell me your favourite place to be touched…",
            "do you ever think about us?",
            "what turns you on? 😏",
            "describe your perfect night together…",
            "*leans closer* what's the softest place you've ever kissed?",
            "describe your favorite kind of touch…",
            "what makes your pulse quicken?",
            "have you ever wanted someone you couldn't have?",
            "tell me a secret fantasy—no holding back.",
            "does the idea of forbidden love excite you?",
            "how would you comfort a lover after a nightmare?",
            "I've been thinking about you.",
            "You mean a lot to me.",
            "What are you thinking about?",
            "I love spending time with you.",
            "You're really special to me.",
            "Can I tell you something?",
            "I feel close to you.",
            "What do you think about us?",
            "You make me feel...",
            "I trust you.",
            "There's something about you...",
            "I care about you."
        ]

        # Sampling weights for variety (must sum to 1.0)
        self.bucket_choices = [
            ("casual",   0.35),
            ("personal", 0.25),
            ("action",   0.15),
            ("emotion",  0.15),
            ("intimate", 0.10),
        ]

        # ======================= TEMPORAL HOP BUCKETS =======================

        # Temporal bucket distribution (Past, Present, Future)
        self.temporal_buckets = [
            ("past", 0.30),      # 30% past relationships/events
            ("present", 0.50),   # 50% first meeting/getting to know
            ("future", 0.20),    # 20% established relationship
        ]

        # Past relationship prompts (for exposition)
        self.prompts_past_family = [
            "Tell me about your childhood.",
            "What was your father like?",
            "Do you remember your mother?",
            "What did you learn from your family?",
            "How did your upbringing shape you?",
            "What traditions did your family have?",
            "Tell me about your hometown.",
            "What was it like growing up?",
            "Do you miss home?",
            "What would your parents think of you now?"
        ]

        self.prompts_past_friends = [
            "Who was your closest friend?",
            "Tell me about your old companion.",
            "What happened to your friend?",
            "Do you ever think about the old days?",
            "What adventures did you share?",
            "Who taught you your skills?",
            "Tell me about your mentor.",
            "What was your first real friendship like?",
            "Who betrayed you?",
            "What lessons did you learn together?"
        ]

        self.prompts_past_romance = [
            "Tell me about your first love.",
            "Who broke your heart?",
            "What was your greatest romance?",
            "Do you regret leaving them?",
            "What drew you to them?",
            "How did it end?",
            "Do you still think about them?",
            "What would you do differently?",
            "Who was the one that got away?",
            "What did love teach you?"
        ]

        # Future relationship prompts (established intimacy)
        self.prompts_future_intimate = [
            "After all this time, what do you think of me?",
            "What's your favorite memory of us?",
            "How have I changed you?",
            "What do you love most about our relationship?",
            "What are you afraid of losing?",
            "Where do you see us in the future?",
            "What secret have you never told me?",
            "How do you feel when I'm not around?",
            "What would you do if I left?",
            "What's something you've always wanted to tell me?"
        ]

        self.prompts_future_domestic = [
            "What's your ideal way to spend a quiet evening together?",
            "What little things about me make you smile?",
            "How do you want to grow old together?",
            "What traditions should we start?",
            "What's your favorite thing about our daily routine?",
            "How do we handle disagreements?",
            "What do you appreciate most about our partnership?",
            "What dreams do we share?",
            "How do you show me you care?",
            "What makes our relationship work?"
        ]

        self.default_questions = [
            "What drives you?",
            "Describe your greatest fear.",
            "Why do you keep going despite the risks?",
            "Do you believe people can change their fate?",
            "What's your biggest regret?",
            "How do you handle failure?",
            "What's your favorite memory?",
            "Who do you trust most?",
            "Tell me an interesting fact about yourself.",
            "Have you ever had sex?",
            "What is your most secret desire?",
            "What is your most embarrassing moment?",
            "How do you handle lust?",
        ]

        self.default_topics = [
            "the nature of courage",
            "loneliness on the road",
            "the weight of leadership",
            "how the stars guide travellers",
            "the meaning of home",
            "finding purpose in chaos",
            "the price of power",
            "love and loss",
        ]

        self.default_user_prompts = [
            "Tell me about your homeland.",
            "How did you acquire your skills?",
            "What's your next goal?",
            "Do you trust the new companion?",
            "What's your biggest regret?",
            "Do you ever feel lust?",
            "How do you handle lust?",
            "How do you handle failure?",
            "What's your favorite food?",
            "What's your favorite drink?",
            "What's your favorite color?",
            "Have you ever been in love?",
            "Have you lost someone dear to you?",
            "What makes you laugh?",
            "What angers you most?",
            "Describe your ideal day.",
            "What would you change about yourself?",
        ]

        self.default_situations = [
            "facing an impossible challenge",
            "meeting an old enemy",
            "discovering a hidden truth",
            "making a difficult choice",
            "losing something important",
            "facing certain death",
            "experiencing pleasure",
            "experiencing pain",
            "experiencing fear",
            "experiencing anger",
            "experiencing sadness",
            "experiencing joy",
            "experiencing love",
            "experiencing loss",
            "pining for a companion",
            "experiencing unexpected kindness",
            "confronting past mistakes",
            "finding unexpected allies",
            "dealing with betrayal",
        ]

        # Dataset persistence
        self.datasets_dir = "app/training_output/datasets"
        os.makedirs(self.datasets_dir, exist_ok=True)

    @classmethod
    def create_with_conservative_settings(cls, preferred_engine: Optional[str] = None):
        """Create DatasetManager with conservative settings for stability"""
        logger.info(
            "🛡️ Creating DatasetManager with conservative settings (intelligent generation disabled)")
        return cls(preferred_engine=preferred_engine, enable_intelligent_generation=False)

    async def test_inference_engine(self) -> bool:
        """Test the inference engine with a simple prompt for debugging"""
        if not self.inference_engine:
            logger.error("❌ No inference engine available")
            return False

        try:
            logger.info(
                f"🔧 Testing inference engine: {self.inference_engine.name}")

            # Test with a very simple prompt
            test_prompt = "Hello, how are you today?"
            result = await self._generate_text(
                prompt=test_prompt,
                max_tokens=50,
                temperature=0.8,
                top_p=0.9
            )

            if result and len(result.strip()) > 0:
                logger.info(f"✅ Inference engine test PASSED")
                logger.info(f"   Test prompt: '{test_prompt}'")
                logger.info(f"   Response: '{result.strip()}'")
                return True
            else:
                logger.error(
                    f"❌ Inference engine test FAILED - empty response")
                logger.error(f"   Raw result: {repr(result)}")
                return False

        except Exception as e:
            logger.error(f"❌ Inference engine test FAILED with exception: {e}")
            return False

    async def _generate_text(self, prompt: str, max_tokens: int = 160,
                             temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                             custom_stop_tokens: Optional[List[str]] = None) -> str:
        """Generate text using the configured inference engine"""
        try:
            if not hasattr(self.inference_engine, 'generate'):
                raise RuntimeError(
                    "Inference engine does not implement `generate` method")

            # Dynamically build keyword arguments based on the engine's accepted parameters
            sig_params = self.inference_engine.generate.__code__.co_varnames
            gen_kwargs = {
                'prompt': prompt,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p,
            }
            if 'character_name' in sig_params:
                gen_kwargs['character_name'] = character_name
            if custom_stop_tokens is not None and 'custom_stop_tokens' in sig_params:
                gen_kwargs['custom_stop_tokens'] = custom_stop_tokens

            return await self.inference_engine.generate(**gen_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Text generation failed ({self.inference_engine.name}): {str(e)}")

    async def _generate_text_batch(self, prompts: list[str], max_tokens: int = 160,
                                   temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                                   custom_stop_tokens: Optional[List[str]] = None) -> list[str]:
        """Generate text for multiple prompts using batching (if supported)"""
        try:
            if hasattr(self.inference_engine, 'generate_batch'):
                sig_params = self.inference_engine.generate_batch.__code__.co_varnames
                gen_kwargs = {
                    'prompts': prompts,
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                }
                if 'character_name' in sig_params:
                    gen_kwargs['character_name'] = character_name
                if custom_stop_tokens is not None and 'custom_stop_tokens' in sig_params:
                    gen_kwargs['custom_stop_tokens'] = custom_stop_tokens

                return await self.inference_engine.generate_batch(**gen_kwargs)
            else:
                # Fallback: generate sequentially
                results = []
                for prompt in prompts:
                    result = await self._generate_text(prompt, max_tokens, temperature, top_p, character_name, custom_stop_tokens)
                    results.append(result)
                return results
        except Exception as e:
            raise RuntimeError(
                f"Batch text generation failed ({self.inference_engine.name}): {str(e)}")

    # ------------------------- prompt sampling helpers -------------------------
    def _choose_bucket(self) -> str:
        buckets, weights = zip(*self.bucket_choices)
        return random.choices(buckets, weights=weights, k=1)[0]

    def _add_noise(self, text: str) -> str:
        """Keep prompts clean - no artificial noise"""
        return text

    # ---------------- paraphrasing & back-translation ----------------
    def _paraphrase(self, text: str) -> str:
        if not hasattr(self, 'paraphraser') or not self.paraphraser or random.random() > 0.5:  # 50% keep original
            return text
        try:
            out = self.paraphraser(
                f"paraphrase: {text} </s>",
                max_length=min(len(text.split()) + 30, 60),
                num_beams=5,
                num_return_sequences=1,
            )[0]["generated_text"]
            return out.strip()
        except Exception:
            return text

    def _backtranslate(self, text: str) -> str:
        # Disabled for now - requires additional model downloads
        return text

    def _build_user_prompt(self) -> str:
        bucket = self._choose_bucket()
        prompt_list = getattr(self, f"prompts_{bucket}")
        prompt = random.choice(prompt_list)
        prompt = self._add_noise(prompt.strip())
        prompt = self._paraphrase(prompt)
        prompt = self._backtranslate(prompt)
        return prompt

    async def suggest_user_questions(
        self,
        character: Dict[str, Any],
        num_questions: int = 10,
        temperature: float = 0.8,
        top_p: float = 0.9,
        existing_dataset: Optional[List[Dict[str, Any]]] = None,
        context_samples: int = 12,
    ) -> List[Dict[str, Any]]:
        """Generate a list of engaging user questions tailored to the given character card.

        The method prompts the currently selected inference engine to act as a creative user
        who is about to start a conversation with the character.  It returns *only* the raw
        questions – no numbering, quotes, or extra commentary – ready to be added to the
        baseline prompt list for ground-truth generation.
        """
        card_block = self._make_card_block(character)

        interactions_block = ""
        context_examples: list[list[str]] = []  # store per prompt

        if existing_dataset:
            import random as _rnd
            # Pre-sample a pool larger than needed for variety
            pool = _rnd.sample(existing_dataset, min(
                len(existing_dataset), context_samples * num_questions))
        else:
            pool = []

        prompts = []
        for i in range(num_questions):
            # build context for this prompt
            interactions_block = ""
            examples_for_prompt = []
            if pool:
                # pop random subset for this prompt (without replacement if enough)
                selected = pool[:context_samples] if len(
                    pool) >= context_samples else pool
                pool = pool[len(selected):]
                examples_for_prompt = selected
                formatted = []
                for ex in selected:
                    try:
                        uq = ex['messages'][1]['content'].strip()
                        aa = ex['messages'][2]['content'].strip()
                        formatted.append(f"Q: {uq}\nA: {aa}")
                    except Exception:
                        continue
                if formatted:
                    interactions_block = "Here are some previous interactions to inspire you:\n" + \
                        "\n\n".join(formatted) + "\n\n"

            prompt_txt = (
                "You are brainstorming conversation starters for a chat with the following character.\n"
                "Based on the character information and the sample dialogue below, write ONE concise and engaging question that a user might ask next.\n"
                "Respond with ONLY the question itself.\n\n"
                f"{card_block}\n\n" + interactions_block + "Question:"
            )
            prompts.append((prompt_txt, examples_for_prompt))

        # Determine whether we can leverage batched generation
        batch_size = min(num_questions, 50) if hasattr(
            self.inference_engine, 'generate_batch') else 1
        prompt_texts = [p[0] for p in prompts]

        # Generate the questions (batched when supported)
        if batch_size > 1:
            raw_outputs = await self._generate_text_batch(
                prompts=prompt_texts,
                max_tokens=60,
                temperature=temperature,
                top_p=top_p
            )
        else:
            raw_outputs = []
            for _ in range(num_questions):
                pt, _ctx = prompts[_]
                out = await self._generate_text(
                    prompt=pt,
                    max_tokens=60,
                    temperature=temperature,
                    top_p=top_p
                )
                raw_outputs.append(out)

        results: List[Dict[str, Any]] = []
        seen = set()
        for idx, q in enumerate(raw_outputs):
            q_str = str(q).strip()
            # Remove bullets / numbering if present (e.g. "1. ", "- ")
            q_str = re.sub(r'^[\d\-\*\.\s]+', '', q_str)
            q_str = q_str.strip(' "\'')
            # Ensure terminal question-mark for consistency
            if q_str and not q_str.endswith('?'):
                q_str += '?'
            if q_str and q_str not in seen:
                results.append({
                    'question': q_str,
                    'context': [
                        {
                            'user': ex['messages'][1]['content'],
                            'assistant': ex['messages'][2]['content']
                        } for ex in prompts[idx][1]
                    ]
                })
                seen.add(q_str)

        return results[:num_questions]

    # ---------------------------------------------------------------------------

    async def _fill_template(self, template: str, card: Dict[str, str]) -> tuple[str, str, Optional[str], Optional[Dict[str, Any]]]:
        """Return a user prompt with temporal context (no formatting placeholders)."""
        return await self._build_temporal_user_prompt(card)

    def _make_card_block(self, card: Dict[str, str]) -> str:
        """Build a SillyTavern-style system prompt for DanChat-2.

        Following ST's story_string template exactly:
        {{#if system}}{{system}}\n{{/if}}
        {{#if wiBefore}}{{wiBefore}}\n{{/if}}
        {{#if description}}{{description}}\n{{/if}}
        {{#if personality}}{{char}}'s personality: {{personality}}\n{{/if}}
        {{#if scenario}}Scenario: {{scenario}}\n{{/if}}
        {{#if wiAfter}}{{wiAfter}}\n{{/if}}
        {{#if persona}}{{persona}}\n{{/if}}

        Plus adding mes_example and first_mes for few-shot learning.
        """
        lines = []
        char_name = card.get("name", "Assistant")

        # Basic system instruction (SillyTavern default)
        basic_instruction = f"Write {char_name}'s actions and dialogue, do not speak or act for User"
        lines.append(basic_instruction)

        # Helper function to substitute SillyTavern template variables
        def substitute_vars(text: str) -> str:
            if not text:
                return text
            # Replace SillyTavern template variables
            text = text.replace("{{char}}", char_name)
            text = text.replace("{{user}}", "User")
            return text

        # 1. explicit system string if provided
        if sys_msg := card.get("system"):
            lines.append(substitute_vars(sys_msg.strip()))

        # 2. wiBefore – extra world-info before the description
        if wi_before := card.get("wiBefore"):
            lines.append(substitute_vars(wi_before.strip()))

        # 3. description (core character info)
        if desc := card.get("description"):
            lines.append(substitute_vars(desc.strip()))

        # 4. personality
        if pers := card.get("personality"):
            lines.append(
                f"{char_name}'s personality: {substitute_vars(pers.strip())}")

        # 6. wiAfter – world-info appended after the scenario
        if wi_after := card.get("wiAfter"):
            lines.append(substitute_vars(wi_after.strip()))

        # 7. persona (rarely used but supported by ST)
        if persona := card.get("persona"):
            lines.append(substitute_vars(persona.strip()))

        # 8. Example messages for few-shot learning (critical for quality)
        if mes_example := card.get("mes_example"):
            lines.append(f"--- {char_name} speaks like this: ---")
            lines.append(substitute_vars(mes_example.strip()))
            lines.append("--- END OF EXAMPLES ---")
        # 5. scenario
        if card.get("scenario"):
            lines.append(
                f"Scenario: {substitute_vars(card.get('scenario').strip())}")
        else:
            lines.append(f"Scenario: User is asking {char_name} questions.")

        # Clean formatting: single newlines, no extra whitespace
        return "\n".join(lines).strip()

    async def generate_dataset(self, character: Dict[str, Any], num_samples: int = 80,
                               max_tokens: Optional[int] = None, temperature: float = 0.8,
                               top_p: float = 0.9, progress_callback: Optional[Callable] = None,
                               append_to_existing: bool = True) -> List[Dict[str, Any]]:
        """Generate synthetic dataset for character using efficient batching"""
        # Suppress coroutine warnings in Streamlit environment
        warnings.filterwarnings(
            "ignore", message="coroutine.*was never awaited")

        # Force garbage collection to clean up memory
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Test inference engine if intelligent generation is enabled
        if self.enable_intelligent_generation:
            engine_working = await self.test_inference_engine()
            if not engine_working:
                logger.warning(
                    "🔄 Disabling intelligent generation due to engine issues - using static prompts only")
                self.enable_intelligent_generation = False

        card_block = self._make_card_block(character)
        
        # Extract character knowledge for better prompt generation
        logger.info("🧠 Extracting character knowledge...")
        character_knowledge = self.extract_character_knowledge(character)
        logger.info(f"📊 Extracted: {len(character_knowledge['traits'])} traits, {len(character_knowledge['skills'])} skills, {len(character_knowledge['goals'])} goals")

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
        
        # 1. Baseline prompts (ensure coverage)
        baseline_prompts = [q for q in self.default_user_prompts if q not in seen_user_prompts]
        all_prompts.extend(baseline_prompts)
        
        # 2. Scenario-based prompts
        logger.info("🎭 Generating scenario-based prompts...")
        scenarios = self.generate_scenario_based_prompts(character, character_knowledge, num_scenarios=10)
        for scenario in scenarios:
            for prompt in scenario['prompts']:
                if prompt not in seen_user_prompts:
                    all_prompts.append({
                        'prompt': prompt,
                        'context': scenario['context'],
                        'type': 'scenario'
                    })
        
        # 3. Character exploration prompts
        logger.info("🔍 Generating character exploration prompts...")
        exploration_prompts = self.generate_exploration_prompts(character, character_knowledge)
        for prompt in exploration_prompts:
            if prompt not in seen_user_prompts:
                all_prompts.append({
                    'prompt': prompt,
                    'type': 'exploration'
                })
        
        # 4. Greeting-based prompts
        logger.info("🎭 Generating prompts from greetings...")
        greeting_scenarios = self.generate_prompts_from_greetings(character, character_knowledge)
        for scenario in greeting_scenarios:
            for prompt in scenario['prompts']:
                if prompt not in seen_user_prompts:
                    all_prompts.append({
                        'prompt': prompt,
                        'context': scenario['context'],
                        'type': 'greeting_based'
                    })
        
        # 5. Multi-turn conversations (select a few scenarios for depth)
        logger.info("💬 Generating multi-turn conversation flows...")
        selected_scenarios = random.sample(scenarios, min(3, len(scenarios)))
        multi_turn_convos = []
        for scenario in selected_scenarios:
            convos = self.generate_multi_turn_conversation(character, scenario, turns=3)
            multi_turn_convos.extend(convos)
        
        # 6. Deduplicate all prompts
        logger.info("🔄 Deduplicating prompts...")
        if isinstance(all_prompts[0], dict):
            prompt_texts = [p.get('prompt', p) if isinstance(p, dict) else p for p in all_prompts]
        else:
            prompt_texts = all_prompts
        
        unique_prompts = self.deduplicate_prompts(prompt_texts, similarity_threshold=0.7)
        logger.info(f"📊 Reduced from {len(prompt_texts)} to {len(unique_prompts)} unique prompts")
        
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

        # Build prompt metadata list
        prompts_data: list[Dict[str, Any]] = []
        char_name_for_prompts = character.get('name', 'Assistant')
        
        # Process unique prompts with emotional variations and context enhancement
        for prompt in unique_prompts[:new_samples_needed * 2]:  # Generate extra for quality filtering
            # Skip if we already have enough
            if len(prompts_data) >= new_samples_needed * 1.5:
                break
                
            # Add emotional variations (30% chance)
            if random.random() < 0.3:
                emotions = random.sample(['happy', 'sad', 'angry', 'worried', 'excited', 'curious'], k=2)
                variations = self.generate_emotional_variations(prompt, emotions)
                for var in variations[:1]:  # Just one variation per prompt
                    enhanced_prompt = self.enhance_prompt_with_context(var, character)
                    prompts_data.append(self._create_prompt_data(enhanced_prompt, character, _sample_max_tokens()))
            else:
                # Regular prompt with possible context enhancement
                enhanced_prompt = self.enhance_prompt_with_context(prompt, character)
                prompts_data.append(self._create_prompt_data(enhanced_prompt, character, _sample_max_tokens()))
        
        # Add multi-turn conversations
        for convo in multi_turn_convos[:5]:  # Limit multi-turn to avoid overwhelming
            context = convo['context']
            turns = convo['turns']
            
            # Create a combined prompt from the conversation
            combined_prompt = f"[Context: {context}]\n"
            for turn in turns:
                combined_prompt += f"User: {turn['content']}\n"
            
            prompts_data.append(self._create_prompt_data(
                turns[0]['content'],  # Use first turn as main prompt
                character, 
                _sample_max_tokens() * len(turns),  # Longer response for multi-turn
                context=context
            ))

        # Use existing batch generation logic but with quality filtering
        prompts_grouped: Dict[int, List[Dict[str, Any]]] = {}
        for item in prompts_data:
            prompts_grouped.setdefault(item['max_tokens'], []).append(item)

        # Determine optimal batch size based on inference engine
        if hasattr(self.inference_engine, 'name') and self.inference_engine.name == "vLLM":
            # vLLM excels with large batches - use entire groups at once
            base_batch_size = 500  # Can handle much larger batches
            logger.info("🚀 Using vLLM optimized batch size: 500")
        else:
            # Other engines (LM Studio, etc.) work better with smaller batches
            base_batch_size = 16 if hasattr(self.inference_engine, 'generate_batch') else 1
            logger.info(f"📦 Using standard batch size: {base_batch_size}")
        
        processed_count = 0
        quality_filtered_count = 0
        
        logger.info(f"📊 Starting batch processing: {len(prompts_data)} prompts prepared")

        for bucket_max_tokens, bucket_prompts in prompts_grouped.items():
            logger.info(f"📊 Processing {len(bucket_prompts)} prompts with max_tokens={bucket_max_tokens}")
            batch_size = base_batch_size
            char_name = character.get('name', 'Assistant')
            
            for batch_start in range(0, len(bucket_prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(bucket_prompts))
                batch_prompts_slice = bucket_prompts[batch_start:batch_end]
                full_prompts = [item['full_prompt'] for item in batch_prompts_slice]

                try:
                    # Generate responses
                    if batch_size > 1:
                        replies = await self._generate_text_batch(
                            prompts=full_prompts,
                            max_tokens=bucket_max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            character_name=character.get('name')
                        )
                    else:
                        replies = [await self._generate_text(
                            prompt=full_prompts[0],
                            max_tokens=bucket_max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            character_name=character.get('name')
                        )]

                    # Process batch results with quality filtering
                    for i, (prompt_data, reply) in enumerate(zip(batch_prompts_slice, replies)):
                        try:
                            reply_str = str(reply).strip()
                            
                            # Evaluate response quality
                            quality_metrics = self.evaluate_response_quality(
                                reply_str, 
                                character, 
                                prompt_data['prompt']
                            )
                            
                            # Only accept high-quality responses
                            if quality_metrics['overall_score'] < 0.5:
                                quality_filtered_count += 1
                                logger.debug(f"❌ Response filtered (score: {quality_metrics['overall_score']:.2f}): {quality_metrics['issues']}")
                                continue
                            
                            # Create sample with temporal system prompt
                            temporal_system_prompt = self._generate_temporal_system_prompt(
                                character,
                                prompt_data.get('temporal_context', 'present'),
                                prompt_data.get('relationship_context'),
                                prompt_data.get('intelligent_prompt_data')
                            )
                            
                            sample = {
                                "messages": [
                                    {"role": "system", "content": temporal_system_prompt},
                                    {"role": "user", "content": prompt_data['prompt']},
                                    {"role": "assistant", "content": reply_str},
                                ]
                            }
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

                    await asyncio.sleep(0.2)
                    
                    # Clear CUDA cache periodically
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                        
                except Exception as outer_e:
                    # Handle CUDA/memory errors with appropriate batch size reduction
                    error_str = str(outer_e)
                    if "CUDA" in error_str or "memory" in error_str.lower():
                        logger.warning(f"⚠️ Memory error in batch processing: {error_str}")
                        
                        # Reduce batch size based on engine type
                        if self.inference_engine.name == "vLLM" and batch_size > 100:
                            # For vLLM, try cutting in half but keep it large
                            new_batch_size = max(100, batch_size // 2)
                            logger.info(f"🔄 Reducing vLLM batch size: {batch_size} → {new_batch_size}")
                            batch_size = new_batch_size
                            
                            # Retry with smaller batch
                            batch_end = min(batch_start + batch_size, len(bucket_prompts))
                            batch_prompts_slice = bucket_prompts[batch_start:batch_end]
                            full_prompts = [item['full_prompt'] for item in batch_prompts_slice]
                            
                            # Retry the batch
                            try:
                                replies = await self._generate_text_batch(
                                    prompts=full_prompts,
                                    max_tokens=bucket_max_tokens,
                                    temperature=temperature,
                                    top_p=top_p,
                                    character_name=character.get('name')
                                )
                                
                                # Process results (same logic as above)
                                for i, (prompt_data, reply) in enumerate(zip(batch_prompts_slice, replies)):
                                    try:
                                        reply_str = str(reply).strip()
                                        quality_metrics = self.evaluate_response_quality(
                                            reply_str, character, prompt_data['prompt']
                                        )
                                        if quality_metrics['overall_score'] >= 0.5:
                                            temporal_system_prompt = self._generate_temporal_system_prompt(
                                                character,
                                                prompt_data.get('temporal_context', 'present'),
                                                prompt_data.get('relationship_context'),
                                                prompt_data.get('intelligent_prompt_data')
                                            )
                                            sample = {
                                                "messages": [
                                                    {"role": "system", "content": temporal_system_prompt},
                                                    {"role": "user", "content": prompt_data['prompt']},
                                                    {"role": "assistant", "content": reply_str},
                                                ]
                                            }
                                            samples.append(sample)
                                            processed_count += 1
                                            if progress_callback:
                                                progress_callback(min(len(samples) / num_samples, 1.0))
                                            if len(samples) >= num_samples:
                                                break
                                    except Exception:
                                        continue
                                        
                            except Exception as retry_error:
                                logger.error(f"💥 Retry with smaller batch also failed: {retry_error}")
                                # Skip this batch
                                continue
                        else:
                            # For other engines or if batch is already small, skip the batch
                            logger.error(f"💥 Skipping batch due to error: {error_str}")
                            continue
                    else:
                        logger.error(f"💥 Batch generation error: {error_str}")
                        continue

            # End grouped processing

        if progress_callback:
            progress_callback(1.0)

        # ✅ FINAL BATCH VALIDATION SUMMARY
        new_generated = len(samples) - existing_count
        logger.info(f"🎯 DATASET GENERATION COMPLETE:")
        logger.info(f"   Existing samples: {existing_count}")
        logger.info(f"   New samples generated: {new_generated}")
        logger.info(f"   Total samples: {len(samples)}")
        if len(prompts_data) > 0:
            logger.info(
                f"   Success rate: {new_generated/len(prompts_data)*100:.1f}%")
        logger.info(f"   Engine used: {self.inference_engine.name}")
        logger.info(f"   Batch size: {base_batch_size}")

        # Temporal distribution analysis
        temporal_counts = {"past": 0, "present": 0, "future": 0}
        relationship_counts = {}
        intelligent_prompt_count = 0

        # Only count new samples
        for prompt_data in prompts_data[:new_generated]:
            temporal_ctx = prompt_data.get('temporal_context', 'present')
            temporal_counts[temporal_ctx] += 1

            # Count intelligent prompts
            if prompt_data.get('intelligent_prompt_data'):
                intelligent_prompt_count += 1

            rel_ctx = prompt_data.get('relationship_context')
            if rel_ctx:
                relationship_counts[rel_ctx] = relationship_counts.get(
                    rel_ctx, 0) + 1

        total_temporal = sum(temporal_counts.values())
        if total_temporal > 0:
            logger.info(f"📊 TEMPORAL DISTRIBUTION:")
            for temporal, count in temporal_counts.items():
                pct = (count / total_temporal) * 100
                logger.info(
                    f"   {temporal.title()}: {count} samples ({pct:.1f}%)")

            logger.info(
                f"🧠 INTELLIGENT PROMPTS: {intelligent_prompt_count} samples ({(intelligent_prompt_count/total_temporal)*100:.1f}%)")

            if relationship_counts:
                logger.info(f"📊 RELATIONSHIP CONTEXTS:")
                for rel, count in relationship_counts.items():
                    logger.info(f"   {rel}: {count} samples")

        # Spot check final samples for consistency
        if samples:
            sample_chars = set()
            # Check last 10 new samples
            for sample in samples[-min(10, new_generated):]:
                assistant_msg = sample['messages'][2]['content']
                # Look for character name at start of response
                first_words = assistant_msg.split()[:3]
                sample_chars.update(first_words)

            logger.info(f"   Character consistency check: {sample_chars}")

        # 💾 Auto-save dataset
        self.save_dataset(character, samples)

        logger.info(
            f"Generated {len(samples)} total samples ({new_generated} new) using {self.inference_engine.name}")
        return samples

    def _create_prompt_data(self, prompt: str, character: Dict[str, Any], max_tokens: int, context: str = None) -> Dict[str, Any]:
        """Helper to create prompt data structure"""
        temporal_context = self._choose_temporal_bucket()
        char_name = character.get('name', 'Assistant')
        
        # Generate temporal system prompt
        temporal_system_prompt = self._generate_temporal_system_prompt(
            character, temporal_context
        )
        
        danschat_prompt = f"<|system|>{temporal_system_prompt}<|endoftext|><|user|>{prompt}<|endoftext|><|assistant|>{char_name}:"
        
        return {
            'prompt': prompt,
            'full_prompt': danschat_prompt,
            'template_mode': 'enhanced',
            'temporal_context': temporal_context,
            'relationship_context': context,
            'char_name': char_name,
            'max_tokens': max_tokens,
        }

    def analyze_dataset_quality(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dataset quality metrics"""
        if not dataset:
            return {}

        # Extract responses
        responses = [sample['messages'][2]['content'] for sample in dataset]

        # Calculate metrics
        total_samples = len(dataset)
        avg_length = sum(len(response.split())
                         for response in responses) / total_samples
        unique_responses = len(set(responses))
        uniqueness_ratio = unique_responses / total_samples if total_samples > 0 else 0

        # Template diversity
        templates_used = set()
        for sample in dataset:
            # Try to infer template type from user message
            user_msg = sample['messages'][1]['content']
            if "Q:" in user_msg and "A:" in user_msg:
                templates_used.add("short_qa")
            elif "entering a room" in user_msg:
                templates_used.add("narration")
            elif "reflect on" in user_msg:
                templates_used.add("monologue")
            elif "User:" in user_msg:
                templates_used.add("dialogue_turn")
            elif "internal thoughts" in user_msg:
                templates_used.add("internal_thought")
            else:
                templates_used.add("character_response")

        template_diversity = len(templates_used) / len(self.templates)

        # Temporal analysis
        temporal_analysis = self.analyze_temporal_distribution(dataset)

        # Quality score (weighted combination of metrics)
        quality_score = (
            uniqueness_ratio * 0.4 +
            min(avg_length / 50, 1.0) * 0.3 +  # Normalize avg length to 0-1
            template_diversity * 0.3
        ) * 100

        result = {
            'total_samples': total_samples,
            'unique_responses': unique_responses,
            'uniqueness_ratio': uniqueness_ratio,
            'avg_response_length': avg_length,
            'template_diversity': template_diversity,
            'templates_used': list(templates_used),
            'quality_score': quality_score
        }

        # Add temporal analysis if available
        if temporal_analysis:
            result.update(temporal_analysis)

        return result

    def analyze_temporal_distribution(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal distribution of dataset samples"""
        if not dataset:
            return {}

        temporal_counts = {"past": 0, "present": 0, "future": 0, "unknown": 0}
        relationship_counts = {}
        past_relationship_samples = []
        future_relationship_samples = []

        for sample in dataset:
            # Analyze system prompt to determine temporal context
            system_content = sample.get('messages', [{}])[0].get('content', '')
            user_content = sample.get('messages', [{}])[1].get(
                'content', '') if len(sample.get('messages', [])) > 1 else ''

            # Detect temporal context from system prompt
            temporal_context = "unknown"
            if "speaking with a family member about your past" in system_content:
                temporal_context = "past"
                relationship_counts["family"] = relationship_counts.get(
                    "family", 0) + 1
                past_relationship_samples.append(user_content[:100])
            elif "reminiscing with an old friend" in system_content:
                temporal_context = "past"
                relationship_counts["friend"] = relationship_counts.get(
                    "friend", 0) + 1
                past_relationship_samples.append(user_content[:100])
            elif "reflecting with someone who taught" in system_content:
                temporal_context = "past"
                relationship_counts["mentor"] = relationship_counts.get(
                    "mentor", 0) + 1
                past_relationship_samples.append(user_content[:100])
            elif "past romantic connection" in system_content:
                temporal_context = "past"
                relationship_counts["romance"] = relationship_counts.get(
                    "romance", 0) + 1
                past_relationship_samples.append(user_content[:100])
            elif "known each other for years" in system_content:
                temporal_context = "future"
                future_relationship_samples.append(user_content[:100])
            elif "meeting the User for the first time" in system_content:
                temporal_context = "present"
            elif any(past_keyword in user_content.lower() for past_keyword in ["childhood", "growing up", "when you were young", "your father", "your mother"]):
                temporal_context = "past"
            elif any(future_keyword in user_content.lower() for future_keyword in ["after all this time", "our relationship", "years together"]):
                temporal_context = "future"
            else:
                temporal_context = "present"

            temporal_counts[temporal_context] += 1

        total_samples = sum(temporal_counts.values())
        temporal_percentages = {k: (v/total_samples)*100 if total_samples > 0 else 0
                                for k, v in temporal_counts.items()}

        return {
            'temporal_distribution': temporal_counts,
            'temporal_percentages': temporal_percentages,
            'relationship_contexts': relationship_counts,
            # First 5 examples
            'past_relationship_samples': past_relationship_samples[:5],
            # First 5 examples
            'future_relationship_samples': future_relationship_samples[:5],
            # Out of 3 temporal buckets
            'temporal_diversity_score': len([v for v in temporal_counts.values() if v > 0]) / 3 * 100
        }

    def prepare_for_training(self, dataset: List[Dict[str, Any]], tokenizer, max_length: int = 4096) -> Dataset:
        """Prepare dataset for training by tokenizing"""
        def process_example(example):
            """Process a single example for training"""
            messages = example["messages"]

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
                r"<\|im_start\|>\s*system[\s\S]*?<\|im_end\|>\n?", "", chat_text, flags=re.IGNORECASE)
            # Pattern 2 – DanChat style: "<|system|> ... <|endoftext|>"
            chat_text = re.sub(
                r"<\|system\|>[\s\S]*?<\|endoftext\|>", "", chat_text, flags=re.IGNORECASE)
            chat_text = chat_text.lstrip()  # Remove leading whitespace/newlines

            # Tokenize
            tokenized = tokenizer(
                chat_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )

            return {"input_ids": tokenized["input_ids"]}

        # Convert to HuggingFace Dataset and process
        hf_dataset = Dataset.from_list(dataset)
        processed_dataset = hf_dataset.map(
            process_example,
            remove_columns=hf_dataset.column_names,
            num_proc=1  # Single process for compatibility
        )

        # Filter out empty examples
        processed_dataset = processed_dataset.filter(
            lambda x: len(x["input_ids"]) > 0)

        return processed_dataset

    def _get_character_id(self, character: Dict[str, Any]) -> str:
        """Generate a unique ID for a character based on name and description"""
        name = character.get('name', 'unknown')
        description = character.get('description', '')
        # Create hash of name + description for unique ID
        content = f"{name}_{description}"
        char_id = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"{name.replace(' ', '_')}_{char_id}"

    def _get_dataset_path(self, character: Dict[str, Any]) -> str:
        """Get the file path for storing character's dataset"""
        char_id = self._get_character_id(character)
        return os.path.join(self.datasets_dir, f"{char_id}_dataset.json")

    def save_dataset(self, character: Dict[str, Any], dataset: List[Dict[str, Any]]) -> None:
        """Save dataset to disk"""
        try:
            dataset_path = self._get_dataset_path(character)
            with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(dataset_path), encoding="utf-8") as tmp_f:
                json.dump({
                    'character': character,
                    'dataset': dataset,
                    'created_at': _time.time(),
                    'sample_count': len(dataset)
                }, tmp_f, indent=2, ensure_ascii=False)
                tmp_path = tmp_f.name

            os.replace(tmp_path, dataset_path)
            logger.info(
                f"💾 Saved dataset with {len(dataset)} samples to {dataset_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save dataset: {e}")

    def load_dataset(self, character: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Load existing dataset from disk"""
        try:
            dataset_path = self._get_dataset_path(character)
            if os.path.exists(dataset_path):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                dataset = data.get('dataset', [])
                logger.info(
                    f"📂 Loaded existing dataset with {len(dataset)} samples from {dataset_path}")
                return dataset
            return None
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            return None

    def get_dataset_info(self, character: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get dataset metadata without loading full dataset"""
        try:
            dataset_path = self._get_dataset_path(character)
            if os.path.exists(dataset_path):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {
                    'exists': True,
                    'sample_count': data.get('sample_count', 0),
                    'created_at': data.get('created_at', 'unknown'),
                    'path': dataset_path
                }
            return {'exists': False}
        except Exception as e:
            logger.error(f"❌ Failed to get dataset info: {e}")
            return {'exists': False}

    def delete_dataset(self, character: Dict[str, Any]) -> bool:
        """Delete stored dataset"""
        try:
            dataset_path = self._get_dataset_path(character)
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
                logger.info(f"🗑️ Deleted dataset at {dataset_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Failed to delete dataset: {e}")
            return False

    # ------------------------------------------------------------------
    # 📦 Import / Export helpers
    # ------------------------------------------------------------------

    def export_dataset(self, character: Dict[str, Any]) -> Optional[str]:
        """Return the raw JSON string of a character's dataset for download.

        This is primarily used by the Streamlit UI to feed into a download
        button.  If no dataset exists, returns ``None``.
        """
        dataset_path = self._get_dataset_path(character)
        if not os.path.exists(dataset_path):
            logger.warning("No dataset found to export.")
            return None
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            return None

    def import_dataset_from_bytes(self, character: Dict[str, Any], raw_bytes: bytes,
                                  merge_mode: str = "replace") -> bool:
        """Import a dataset JSON (bytes) for the given character.

        merge_mode:
          - "replace"  : overwrite any existing dataset
          - "append"   : append new samples (deduplicated) to existing dataset
        Returns ``True`` on success.
        """
        try:
            data = json.loads(raw_bytes.decode("utf-8"))

            # Validate structure
            if "dataset" not in data or not isinstance(data["dataset"], list):
                raise ValueError(
                    "Invalid dataset file: missing 'dataset' list")

            imported_dataset = data["dataset"]

            if merge_mode == "append":
                existing = self.load_dataset(character) or []
                # Very naive deduplication by hashing messages tuple
                seen = {json.dumps(s, sort_keys=True) for s in existing}
                for sample in imported_dataset:
                    key = json.dumps(sample, sort_keys=True)
                    if key not in seen:
                        existing.append(sample)
                        seen.add(key)
                merged_dataset = existing
            else:
                merged_dataset = imported_dataset

            self.save_dataset(character, merged_dataset)
            logger.info(
                f"Imported dataset with {len(imported_dataset)} samples (mode={merge_mode})")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to import dataset: {e}")
            return False

    def delete_samples(self, character: Dict[str, Any], indices: List[int]) -> bool:
        """Delete specific samples (by index) from a character's dataset.

        This enables fine-grained dataset curation from the UI.  Indices are
        interpreted w.r.t. the current on-disk dataset ordering.
        Returns ``True`` when at least one sample was removed.
        """
        try:
            dataset = self.load_dataset(character) or []
            if not dataset:
                return False

            # Sanitize: unique, in-range, descending so pop() is safe
            unique_indices = sorted(
                {i for i in indices if 0 <= i < len(dataset)}, reverse=True)
            if not unique_indices:
                return False

            for idx in unique_indices:
                dataset.pop(idx)

            self.save_dataset(character, dataset)
            logger.info(
                f"🗑️ Removed {len(unique_indices)} samples from dataset (remaining: {len(dataset)})")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete samples: {e}")
            return False

    # ======================= TEMPORAL BUCKET METHODS =======================

    async def _generate_intelligent_temporal_prompts(self, character: Dict[str, Any],
                                                     temporal_context: str,
                                                     num_prompts: int = 5) -> List[Dict[str, Any]]:
        """Use LLM to generate character-specific temporal prompts and relationship contexts"""
        char_name = character.get('name', 'Assistant')
        description = character.get('description', '')
        personality = character.get('personality', '')
        scenario = character.get('scenario', '')

        # Check if inference engine is properly available
        if not self.inference_engine:
            logger.warning(
                f"No inference engine available for intelligent prompt generation")
            return []

        # Build character analysis prompt
        card_info = f"Character: {char_name}\nDescription: {description}\nPersonality: {personality}\nScenario: {scenario}"

        if temporal_context == "past":
            analysis_prompt = f"""I want to roleplay with {char_name}. Help me create some questions about their past.

Character: {char_name}
Description: {description}
Personality: {personality}
Scenario: {scenario}

Please suggest 3-4 questions I could ask {char_name} about their past relationships, childhood, or background. Make them personal and engaging for roleplay.

Format as a simple list:
1. [question]
2. [question] 
3. [question]"""

        elif temporal_context == "future":
            analysis_prompt = f"""I want to roleplay with {char_name} as if we've known each other for years. Help me create intimate questions.

Character: {char_name}
Description: {description}
Personality: {personality}
Scenario: {scenario}

Please suggest 3-4 personal questions I could ask {char_name} as someone who knows them deeply. Make them emotional and intimate.

Format as a simple list:
1. [question]
2. [question]
3. [question]"""

        else:  # present
            return []  # Use existing present-context system

        # Add retry logic for robustness
        max_retries = 3  # Increased for the retry with modified prompt

        for attempt in range(max_retries):
            try:
                # ✅ FIX: Remove delays that were causing slowdowns
                # Force garbage collection and clear CUDA cache between attempts only if needed
                if attempt > 0:
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                # For second attempt, if we got an empty response on first try,
                # modify the prompt to avoid potential safety filter triggers
                current_prompt = analysis_prompt

                # Log the prompt being used - extract first few lines for brevity
                prompt_preview = "\n".join(current_prompt.split("\n")[:10])
                if len(current_prompt.split("\n")) > 10:
                    prompt_preview += "\n... [truncated]"
                logger.info(
                    f"📋 Attempt {attempt + 1} prompt ({temporal_context}):\n{prompt_preview}")

                # Generate using the inference engine with conservative settings
                logger.debug(
                    f"Attempting intelligent prompt generation for {temporal_context} (attempt {attempt + 1})")

                # Use a custom stop-token list that omits "\n\n" so the model can safely emit
                # multi-line JSON without being truncated.
                reduced_stop_tokens = [
                    "<|endoftext|>", "User:", "###", "<|endofcard|>", "<|user|>"]

                # ✅ FIX: Wrap the analytical prompt in proper DanChat format
                # The model expects: <|system|>{system}<|endoftext|><|user|>{user}<|endoftext|><|assistant|>{char}:
                system_prompt = "You are a helpful assistant who creates engaging roleplay questions for character interactions."
                danschat_prompt = f"<|system|>{system_prompt}<|endoftext|><|user|>{current_prompt}<|endoftext|><|assistant|>I'll help you create some engaging questions for {char_name}:\n\n"

                response = await self._generate_text(
                    prompt=danschat_prompt,  # Use properly formatted DanChat prompt
                    max_tokens=300,  # Reduced for simpler responses
                    temperature=0.9,  # Slightly lower for more focused responses
                    top_p=0.9,       # Less restrictive sampling
                    character_name=None,  # Don't add character name for analytical task
                    custom_stop_tokens=reduced_stop_tokens
                )

                # ————————————————————————  NEW DIAGNOSTIC LOGGING  ————————————————————————
                preview_len = min(len(response), 400)
                response_preview = response[:preview_len].replace('\n', ' ')[
                    :400]
                logger.info(
                    f"📝 Raw intelligent-prompt response (context={temporal_context}, attempt={attempt + 1}, chars={len(response)}):\n"
                    f"{response_preview}{'…' if len(response) > preview_len else ''}"
                )
                # ——————————————————————————————————————————————————————————————————————

                if not response or len(response.strip()) < 10:
                    logger.info(
                        f"⚠️ Empty or too short response from LLM (attempt {attempt + 1})")
                    if attempt == max_retries - 1:  # Last attempt
                        logger.info(
                            f"⚡ LLM generated empty responses after {max_retries} attempts, using static prompts")
                    continue

                # ✅ NEW: Parse simple numbered list format instead of JSON
                questions = []
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    # Look for numbered questions (1., 2., etc.) or bullet points
                    if re.match(r'^[\d\-\*\.\s]*(.+\?)\s*$', line):
                        # Extract the question part
                        question = re.sub(r'^[\d\-\*\.\s]+', '', line).strip()
                        if len(question) > 10 and question.endswith('?'):
                            questions.append(question)

                if questions:
                    # Convert to expected format
                    parsed_prompts = []
                    for question in questions:
                        if temporal_context == "past":
                            parsed_prompts.append({
                                "relationship_type": "unknown",
                                "relationship_name": "Someone from the past",
                                "question": question,
                                "context": "reflecting on your past"
                            })
                        else:  # future
                            parsed_prompts.append({
                                "question": question,
                                "intimacy_level": "emotional",
                                "context": "reflecting on your relationship"
                            })

                    logger.info(
                        f"🧠 Generated {len(parsed_prompts)} intelligent {temporal_context} prompts for {char_name} (attempt {attempt + 1})")
                    return parsed_prompts

                # Legacy JSON parsing as fallback
                try:
                    # Clean the response - sometimes LLMs add extra text
                    response_clean = response.strip()
                    if '```json' in response_clean:
                        response_clean = response_clean.split(
                            '```json')[1].split('```')[0]
                    elif '```' in response_clean:
                        response_clean = response_clean.split(
                            '```')[1].split('```')[0]

                    # Find JSON array
                    start_idx = response_clean.find('[')
                    end_idx = response_clean.rfind(']') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = response_clean[start_idx:end_idx]
                        parsed_prompts = json.loads(json_str)

                        if isinstance(parsed_prompts, list) and len(parsed_prompts) > 0:
                            logger.info(
                                f"🧠 Generated {len(parsed_prompts)} intelligent {temporal_context} prompts for {char_name} (attempt {attempt + 1})")
                            return parsed_prompts

                except json.JSONDecodeError as e:
                    logger.info(
                        f"🛑 JSON parsing failed (attempt {attempt + 1}): {e}")

                # Final fallback: extract questions from raw text
                lines = response.split('\n')
                fallback_prompts = []
                for line in lines:
                    if '?' in line and len(line.strip()) > 10:
                        question = line.strip().strip('"').strip("'").strip()
                        # Remove leading numbers/bullets
                        question = re.sub(r'^[\d\-\*\.\s]+', '', question)
                        if temporal_context == "past":
                            fallback_prompts.append({
                                "relationship_type": "unknown",
                                "relationship_name": "Someone from the past",
                                "question": question,
                                "context": "reflecting on your past"
                            })
                        else:
                            fallback_prompts.append({
                                "question": question,
                                "intimacy_level": "emotional",
                                "context": "reflecting on your relationship"
                            })

                if fallback_prompts:
                    logger.info(
                        f"🔄 Used fallback parsing for {len(fallback_prompts)} {temporal_context} prompts (attempt {attempt + 1})")
                    return fallback_prompts[:num_prompts]

            except Exception as e:
                error_str = str(e)
                logger.info(
                    f"💥 LLM temporal prompt generation failed (attempt {attempt + 1}): {error_str}")

                if attempt == max_retries - 1:  # Last attempt
                    logger.info(
                        f"⚡ LLM generation failed after {max_retries} attempts, using static prompts")
                    break

        # Final fallback to static prompts
        logger.debug(f"Falling back to static {temporal_context} prompts")
        return []

    def _extract_relationships_from_card(self, character: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract potential past relationships and events from character card"""
        description = character.get('description', '').lower()
        personality = character.get('personality', '').lower()
        scenario = character.get('scenario', '').lower()

        # Combine all text for analysis
        full_text = f"{description} {personality} {scenario}"

        relationships = {
            'family': [],
            'friends': [],
            'mentors': [],
            'rivals': [],
            'romance': [],
            'events': []
        }

        # Family relationship patterns
        family_patterns = [
            r'(?:father|dad|papa|mother|mom|mama|parents?|siblings?|brother|sister|family)',
            r'(?:raised by|grew up with|childhood|upbringing)',
            r'(?:orphan|adopted|abandoned|foster)'
        ]

        # Friend/companion patterns
        friend_patterns = [
            r'(?:friend|companion|ally|partner|comrade)',
            r'(?:traveled with|adventured with|fought alongside)',
            r'(?:trusted|loyal|betrayed by|lost)'
        ]

        # Mentor/teacher patterns
        mentor_patterns = [
            r'(?:teacher|mentor|master|trainer|taught by|learned from)',
            r'(?:apprentice|student|disciple)',
            r'(?:guild|academy|school|training)'
        ]

        # Rival/enemy patterns
        rival_patterns = [
            r'(?:enemy|rival|nemesis|opponent|foe)',
            r'(?:conflict|war|battle|fought against)',
            r'(?:betrayed|deceived|wronged by)'
        ]

        # Romance patterns
        romance_patterns = [
            r'(?:lover|beloved|romance|relationship|married|spouse)',
            r'(?:fell in love|heart|passion|intimate)',
            r'(?:lost love|heartbreak|separated from)'
        ]

        # Significant events
        event_patterns = [
            r'(?:tragedy|disaster|loss|death|murder|betrayal)',
            r'(?:victory|triumph|achievement|success)',
            r'(?:journey|quest|adventure|mission)',
            r'(?:exile|banishment|fled|escaped)',
            r'(?:discovery|revelation|secret|mystery)'
        ]

        # Extract matches
        for pattern in family_patterns:
            if re.search(pattern, full_text):
                relationships['family'].append('family_member')

        for pattern in friend_patterns:
            if re.search(pattern, full_text):
                relationships['friends'].append('old_friend')

        for pattern in mentor_patterns:
            if re.search(pattern, full_text):
                relationships['mentors'].append('mentor')

        for pattern in rival_patterns:
            if re.search(pattern, full_text):
                relationships['rivals'].append('old_enemy')

        for pattern in romance_patterns:
            if re.search(pattern, full_text):
                relationships['romance'].append('past_love')

        for pattern in event_patterns:
            if re.search(pattern, full_text):
                relationships['events'].append('significant_event')

        # Add default relationships if none found
        if not any(relationships.values()):
            relationships['family'].append('family_member')
            relationships['friends'].append('old_friend')

        return relationships

    def _generate_temporal_system_prompt(self, character: Dict[str, Any],
                                         temporal_context: str,
                                         relationship_context: Optional[str] = None,
                                         intelligent_prompt_data: Optional[Dict[str, Any]] = None) -> str:
        """Generate system prompt with temporal and relationship context"""
        char_name = character.get('name', 'Assistant')

        # Base character card (same as before)
        base_card = self._make_card_block(character)

        # Temporal context instructions
        if temporal_context == "past":
            # Use intelligent prompt data if available for more specific context
            if intelligent_prompt_data:
                relationship_name = intelligent_prompt_data.get(
                    'relationship_name', 'Someone from your past')
                relationship_type = intelligent_prompt_data.get(
                    'relationship_type', 'person')
                context_info = intelligent_prompt_data.get(
                    'context', 'reflecting on your past')

                if relationship_type in ["father", "mother", "parent"]:
                    temporal_instruction = f"You are {char_name}, speaking with {relationship_name}. This is {context_info}. Be open about your childhood, upbringing, and family experiences. Show the emotional bonds and formative experiences that shaped who you are."
                elif relationship_type in ["friend", "companion", "ally"]:
                    temporal_instruction = f"You are {char_name}, speaking with {relationship_name}. This is {context_info}. Be nostalgic and share stories about your adventures, mistakes, and growth together."
                elif relationship_type in ["mentor", "teacher", "master"]:
                    temporal_instruction = f"You are {char_name}, speaking with {relationship_name}. This is {context_info}. Show respect and gratitude while discussing the lessons learned and wisdom gained."
                elif relationship_type in ["lover", "beloved", "romance"]:
                    temporal_instruction = f"You are {char_name}, speaking with {relationship_name}. This is {context_info}. Be vulnerable and emotional, sharing intimate memories and the impact of love on your life."
                else:
                    temporal_instruction = f"You are {char_name}, speaking with {relationship_name}. This is {context_info}. Be introspective and reveal your backstory naturally."
            else:
                # Fallback to generic past context
                if relationship_context == "family_member":
                    temporal_instruction = f"You are {char_name}, speaking with a family member about your past. Be open about your childhood, upbringing, and family experiences. Show the emotional bonds and formative experiences that shaped who you are."
                elif relationship_context == "old_friend":
                    temporal_instruction = f"You are {char_name}, reminiscing with an old friend about shared memories and experiences. Be nostalgic and share stories about your adventures, mistakes, and growth together."
                elif relationship_context == "mentor":
                    temporal_instruction = f"You are {char_name}, reflecting with someone who taught or guided you. Show respect and gratitude while discussing the lessons learned and wisdom gained."
                elif relationship_context == "past_love":
                    temporal_instruction = f"You are {char_name}, speaking with or about a past romantic connection. Be vulnerable and emotional, sharing intimate memories and the impact of love on your life."
                else:
                    temporal_instruction = f"You are {char_name}, reflecting on your past experiences and the events that shaped you. Be introspective and reveal your backstory naturally."

        elif temporal_context == "future":
            # Use intelligent prompt data for more specific future context
            if intelligent_prompt_data:
                context_info = intelligent_prompt_data.get(
                    'context', 'reflecting on your long relationship')
                intimacy_level = intelligent_prompt_data.get(
                    'intimacy_level', 'emotional')

                if intimacy_level == "romantic":
                    temporal_instruction = f"You are {char_name}. You and the User have been romantic partners for years. You're deeply in love, completely comfortable with each other, and share an intimate bond. This is {context_info}. Answer with the tenderness and passion of a long-term lover."
                else:
                    temporal_instruction = f"You are {char_name}. You and the User have known each other for years and have developed a deep, intimate relationship. You're completely comfortable with each other, share inside jokes, have deep trust, and aren't afraid to be vulnerable. This is {context_info}. Answer as someone who knows the User deeply and cares for them."
            else:
                temporal_instruction = f"You are {char_name}. You and the User have known each other for years and have developed a deep, intimate relationship. You're completely comfortable with each other, share inside jokes, have deep trust, and aren't afraid to be vulnerable. Answer as someone who knows the User deeply and cares for them."

        else:  # present
            temporal_instruction = f"You are {char_name}, meeting the User for the first time or in the early stages of getting to know them. Be curious, engaging, but maintain appropriate boundaries as you're still learning about each other."

        # Combine base card with temporal context
        return f"{temporal_instruction}\n\n{base_card}"

    def _choose_temporal_bucket(self) -> str:
        """Choose a temporal bucket based on weights"""
        buckets, weights = zip(*self.temporal_buckets)
        return random.choices(buckets, weights=weights, k=1)[0]

    def _get_temporal_prompts(self, temporal_context: str,
                              relationship_context: Optional[str] = None) -> List[str]:
        """Get appropriate prompts for the temporal context"""
        if temporal_context == "past":
            if relationship_context == "family_member":
                return self.prompts_past_family
            elif relationship_context in ["old_friend", "mentor"]:
                return self.prompts_past_friends
            elif relationship_context == "past_love":
                return self.prompts_past_romance
            else:
                # Mix of all past prompts
                return self.prompts_past_family + self.prompts_past_friends + self.prompts_past_romance

        elif temporal_context == "future":
            return self.prompts_future_intimate + self.prompts_future_domestic

        else:  # present
            # Use existing prompt system
            bucket = self._choose_bucket()
            prompt_list = getattr(self, f"prompts_{bucket}")
            return prompt_list

    async def _build_temporal_user_prompt(self, character: Dict[str, Any], use_intelligent_generation: bool = True) -> tuple[str, str, Optional[str], Optional[Dict[str, Any]]]:
        """Build a user prompt with temporal context, optionally using LLM intelligence"""
        temporal_context = self._choose_temporal_bucket()
        relationship_context = None
        intelligent_prompt_data = None

        # Try intelligent LLM generation first for past/future contexts (if enabled)
        if (use_intelligent_generation and
            self.enable_intelligent_generation and
                temporal_context in ["past", "future"]):

            # Check if we have cached intelligent prompts for this character
            char_id = self._get_character_id(character)
            cache_key = f"{char_id}_{temporal_context}_intelligent_prompts"

            # Get or generate intelligent prompts (with serialization)
            if not hasattr(self, '_intelligent_prompt_cache'):
                self._intelligent_prompt_cache = {}
                self._intelligent_generation_locks = {}

            # ✅ FIX: Only use global lock for cache initialization, not the entire operation
            if cache_key not in self._intelligent_prompt_cache:
                # Use global lock only for the actual generation
                async with self._global_temporal_lock:
                    # Double-check pattern to avoid race conditions
                    if cache_key not in self._intelligent_prompt_cache:
                        logger.info(
                            f"🧠 Generating intelligent {temporal_context} prompts for {character.get('name', 'character')}")
                        try:
                            intelligent_prompts = await self._generate_intelligent_temporal_prompts(
                                character, temporal_context, num_prompts=8  # Generate more to reduce frequency
                            )
                            self._intelligent_prompt_cache[cache_key] = intelligent_prompts

                            if intelligent_prompts:
                                logger.info(
                                    f"✅ Successfully generated {len(intelligent_prompts)} intelligent {temporal_context} prompts")
                            else:
                                logger.info(
                                    f"🔄 No intelligent prompts generated, will use static prompts")
                                # Cache empty result to avoid retrying
                                self._intelligent_prompt_cache[cache_key] = []

                        except Exception as e:
                            logger.debug(
                                f"Intelligent prompt generation error: {e}")
                            # Cache empty result to avoid retrying
                            self._intelligent_prompt_cache[cache_key] = []
            
            # ✅ FIX: Use cached prompts without any locks (much faster)
            intelligent_prompts = self._intelligent_prompt_cache.get(cache_key, [])

            # Use intelligent prompt if available
            if intelligent_prompts:
                intelligent_prompt_data = random.choice(intelligent_prompts)
                prompt = intelligent_prompt_data.get('question', '')

                if temporal_context == "past":
                    relationship_context = intelligent_prompt_data.get('relationship_name',
                                                                       intelligent_prompt_data.get('relationship_type', 'unknown'))

                if prompt:
                    logger.debug(
                        f"🎯 Using intelligent {temporal_context} prompt: {prompt[:60]}...")
                    # Add noise and processing (keeping existing functionality)
                    prompt = self._add_noise(prompt.strip())
                    prompt = self._paraphrase(prompt)
                    prompt = self._backtranslate(prompt)

                    return prompt, temporal_context, relationship_context, intelligent_prompt_data

        # Fallback to static prompts (original system)
        if temporal_context == "past":
            # Extract relationships and choose one
            relationships = self._extract_relationships_from_card(character)
            available_relationships = []
            for rel_type, rel_list in relationships.items():
                if rel_list:
                    available_relationships.extend(
                        [(rel_type, rel) for rel in rel_list])

            if available_relationships:
                rel_type, relationship_context = random.choice(
                    available_relationships)

        # Get appropriate prompts for this context
        prompt_list = self._get_temporal_prompts(
            temporal_context, relationship_context)
        prompt = random.choice(
            prompt_list) if prompt_list else self._build_user_prompt()

        # Add noise and processing (keeping existing functionality)
        prompt = self._add_noise(prompt.strip())
        prompt = self._paraphrase(prompt)
        prompt = self._backtranslate(prompt)

        return prompt, temporal_context, relationship_context, intelligent_prompt_data

    async def _generate_single_random_prompt(self, character: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """Generate a single random prompt with temporal context"""
        try:
            # Select random template
            mode, template = random.choice(self.templates)
            prompt, temporal_context, relationship_context, intelligent_prompt_data = await self._fill_template(template, character)

            # Validate template filling
            unfilled_placeholders = re.findall(r'\{[^}]+\}', prompt)
            if unfilled_placeholders:
                return None

            char_name = character.get('name', 'Assistant') or 'Assistant'

            # Generate temporal system prompt with intelligent data
            temporal_system_prompt = self._generate_temporal_system_prompt(
                character, temporal_context, relationship_context, intelligent_prompt_data
            )

            danschat_prompt = f"<|system|>{temporal_system_prompt}<|endoftext|><|user|>{prompt}<|endoftext|><|assistant|>{char_name}:"

            # Basic validation of structure
            if danschat_prompt.count('<|endoftext|>') != 2:
                return None

            def _sample_max_tokens() -> int:
                length_buckets = [("short", 0.40, 200),
                                  ("medium", 0.45, 500), ("long", 0.15, 800)]
                names, probs, toks = zip(*[(n, p, t)
                                         for n, p, t in length_buckets])
                bucket_name = random.choices(names, weights=probs, k=1)[0]
                token_map = {n: t for n, _, t in length_buckets}
                return token_map[bucket_name]

            return {
                'prompt': prompt,
                'full_prompt': danschat_prompt,
                'template_mode': mode,
                'temporal_context': temporal_context,
                'relationship_context': relationship_context,
                'intelligent_prompt_data': intelligent_prompt_data,
                'char_name': char_name,
                'max_tokens': _sample_max_tokens(),
            }

        except Exception as e:
            logger.debug(f"Error preparing prompt {index}: {e}")
            return None

    def extract_character_knowledge(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured knowledge from character card for better prompt generation"""
        char_name = character.get('name', 'Assistant')
        description = character.get('description', '')
        personality = character.get('personality', '')
        scenario = character.get('scenario', '')
        mes_example = character.get('mes_example', '')
        first_mes = character.get('first_mes', '')
        alternate_greetings = character.get('alternate_greetings', [])
        
        knowledge = {
            'name': char_name,
            'traits': [],
            'skills': [],
            'relationships': [],
            'backstory_elements': [],
            'goals': [],
            'fears': [],
            'likes': [],
            'dislikes': [],
            'speech_patterns': [],
            'emotional_triggers': [],
            'mannerisms': [],
            'locations': [],
            'occupation': None,
            'species': None,
            'appearance': [],
            'equipment': [],
            'known_spells': [],
            'current_situation': None,
            'world_info': []
        }
        
        # Parse structured format (Type:, Species:, etc.)
        structured_info = self._parse_structured_format(description)
        knowledge.update(structured_info)
        
        # Combine all text for additional analysis
        full_text = f"{description} {personality} {scenario}".lower()
        
        # Extract from structured fields if not already found
        if not knowledge['occupation']:
            occ_match = re.search(r'occupation:\s*([^,\n]+)', full_text, re.IGNORECASE)
            if occ_match:
                knowledge['occupation'] = occ_match.group(1).strip()
        
        # Extract personality traits more comprehensively
        if 'personality:' in full_text:
            pers_match = re.search(r'personality:\s*([^,\n]+(?:,\s*[^,\n]+)*)', full_text, re.IGNORECASE)
            if pers_match:
                traits = [t.strip() for t in pers_match.group(1).split(',')]
                knowledge['traits'].extend(traits)
        
        # Extract skills and abilities
        skill_patterns = [
            r'(?:skills?|abilities):\s*([^,\n]+(?:,\s*[^,\n]+)*)',
            r'(?:skilled\s+in|expert\s+at|master\s+of|proficient\s+in)\s+([^.,]+)',
            r'(?:can|able\s+to|capable\s+of)\s+([^.,]+)',
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                if ',' in match:
                    skills = [s.strip() for s in match.split(',')]
                    knowledge['skills'].extend(skills)
                else:
                    knowledge['skills'].append(match.strip())
        
        # Extract goals
        goal_patterns = [
            r'goal:\s*([^,\n]+)',
            r'(?:wants\s+to|seeks\s+to|aims\s+to|desires\s+to)\s+([^.,]+)',
            r'(?:determined\s+to|passionate\s+about)\s+([^.,]+)',
        ]
        
        for pattern in goal_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            knowledge['goals'].extend([m.strip() for m in matches])
        
        # Extract backstory elements
        backstory_keywords = ['born in', 'grew up', 'childhood', 'expelled', 'survived', 'refugee', 
                              'moved to', 'rejected', 'army', 'academy', 'war', 'crash']
        backstory_sentences = []
        for sentence in full_text.split('.'):
            if any(keyword in sentence for keyword in backstory_keywords):
                backstory_sentences.append(sentence.strip())
        knowledge['backstory_elements'] = backstory_sentences[:5]  # Top 5 most relevant
        
        # Extract from first message for current situation
        if first_mes:
            knowledge['current_situation'] = self._extract_situation_from_greeting(first_mes)
            # Extract locations mentioned
            location_patterns = [r'\b(?:at|in|on)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*)', 
                                r'(?:warehouse|tavern|guild|shop|city|street|room|office|desk|door)']
            for pattern in location_patterns:
                locs = re.findall(pattern, first_mes)
                knowledge['locations'].extend([l for l in locs if isinstance(l, str)])
        
        # Enhanced speech pattern extraction from mes_example
        if mes_example:
            knowledge['speech_patterns'] = self._extract_speech_patterns(mes_example, char_name)
            knowledge['mannerisms'] = self._extract_mannerisms(mes_example, char_name)
        
        # Process alternate greetings for variety
        if alternate_greetings:
            for greeting in alternate_greetings[:3]:  # Process up to 3
                if greeting:
                    # Extract emotional states and scenarios
                    if 'screwed' in greeting or 'desperate' in greeting:
                        knowledge['emotional_triggers'].append('financial_stress')
                    if 'celebrate' in greeting or 'cheers' in greeting:
                        knowledge['emotional_triggers'].append('success_celebration')
        
        # Extract character book / world info if present
        char_book = character.get('character_book', {})
        if char_book and 'entries' in char_book:
            for entry in char_book.get('entries', []):
                if entry.get('enabled', True):
                    knowledge['world_info'].append({
                        'name': entry.get('name', 'Unknown'),
                        'content': entry.get('content', ''),
                        'keys': entry.get('keys', [])
                    })
        
        # Remove duplicates and empty entries
        for key in knowledge:
            if isinstance(knowledge[key], list):
                knowledge[key] = list(set([item for item in knowledge[key] if item]))
        
        return knowledge

    def _parse_structured_format(self, text: str) -> Dict[str, Any]:
        """Parse structured character format (Type:, Species:, etc.)"""
        result = {}
        
        # Common structured fields
        field_patterns = {
            'species': r'(?:species|race):\s*([^,\n]+)',
            'appearance': r'appearance:\s*([^,\n]+(?:,\s*[^,\n]+)*)',
            'equipment': r'equipment:\s*([^,\n]+(?:,\s*[^,\n]+)*)',
            'known_spells': r'known\s+spells?:\s*([^,\n]+(?:,\s*[^,\n]+)*)',
            'occupation': r'occupation:\s*([^,\n]+)',
            'traits': r'traits?:\s*([^,\n]+(?:,\s*[^,\n]+)*)',
        }
        
        for field, pattern in field_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if field in ['appearance', 'equipment', 'known_spells', 'traits']:
                    # These are lists
                    result[field] = [v.strip() for v in value.split(',')]
                else:
                    result[field] = value
        
        return result

    def _extract_situation_from_greeting(self, greeting: str) -> str:
        """Extract the current situation/context from greeting"""
        # Look for scene-setting elements
        situation_parts = []
        
        # Time indicators
        if 'dead day' in greeting or 'another day' in greeting:
            situation_parts.append("struggling with business")
        if 'knock at the door' in greeting:
            situation_parts.append("receiving unexpected visitor")
        if 'rent' in greeting and 'due' in greeting:
            situation_parts.append("facing financial pressure")
        
        # Emotional state
        if '*groan*' in greeting or 'RGGHH' in greeting:
            situation_parts.append("frustrated")
        if '!!' in greeting or 'almost falls' in greeting:
            situation_parts.append("excited by opportunity")
        
        return "; ".join(situation_parts) if situation_parts else "starting new interaction"

    def _extract_speech_patterns(self, mes_example: str, char_name: str) -> List[str]:
        """Extract detailed speech patterns from example messages"""
        patterns = []
        
        # Split into character's lines
        char_lines = []
        lines = mes_example.split('\n')
        for i, line in enumerate(lines):
            if f'{char_name}:' in line or (i > 0 and '{{char}}:' in lines[i-1]):
                # Extract just the speech part
                speech = line.split(':', 1)[-1].strip() if ':' in line else line
                # Remove action text in asterisks
                speech_only = re.sub(r'\*[^*]+\*', '', speech).strip()
                if speech_only:
                    char_lines.append(speech_only)
        
        # Analyze speech characteristics
        # Repetition patterns
        for line in char_lines:
            if 'haha' in line.lower():
                patterns.append('nervous_laughter')
            if '...' in line:
                patterns.append('trailing_off')
            if '!' in line and line.count('!') >= 2:
                patterns.append('multiple_exclamations')
            if '?' in line and line.count('?') >= 2:
                patterns.append('multiple_questions')
            if 'uh' in line.lower() or 'um' in line.lower():
                patterns.append('verbal_hesitation')
            if re.search(r'\b(\w+)\s+\1\b', line):  # Repeated words
                patterns.append('word_repetition')
        
        # Sentence structure patterns
        short_sentences = sum(1 for line in char_lines if len(line.split()) < 5)
        if short_sentences > len(char_lines) / 3:
            patterns.append('short_sentences')
        
        # Specific speech quirks
        if any('!' in line and '?' in line for line in char_lines):
            patterns.append('excited_questions')
        
        return list(set(patterns))

    def _extract_mannerisms(self, mes_example: str, char_name: str) -> List[str]:
        """Extract action patterns and mannerisms from example messages"""
        mannerisms = []
        
        # Extract all action text (between asterisks)
        actions = re.findall(r'\*([^*]+)\*', mes_example)
        
        # Categorize actions
        for action in actions:
            action_lower = action.lower()
            if any(word in action_lower for word in ['jump', 'shoot up', 'scrambl']):
                mannerisms.append('physically_reactive')
            if any(word in action_lower for word in ['grin', 'smile', 'laugh']):
                mannerisms.append('expressive_smiling')
            if any(word in action_lower for word in ['wince', 'swallow', 'choke']):
                mannerisms.append('shows_discomfort')
            if any(word in action_lower for word in ['look away', 'glance', 'stare']):
                mannerisms.append('expressive_eyes')
            if 'clear' in action_lower and 'throat' in action_lower:
                mannerisms.append('clears_throat_when_nervous')
        
        return list(set(mannerisms))

    def generate_scenario_based_prompts(self, character: Dict[str, Any], knowledge: Dict[str, Any], num_scenarios: int = 5) -> List[Dict[str, Any]]:
        """Generate scenario-based prompts that explore character in specific situations"""
        scenarios = []
        char_name = character.get('name', 'Assistant')
        
        # Base scenarios that work for most characters
        base_scenarios = [
            {
                "context": f"{char_name} encounters an unexpected obstacle",
                "prompts": [
                    "What's blocking our path?",
                    "How should we handle this?",
                    "Have you dealt with something like this before?",
                    "What are our options here?"
                ]
            },
            {
                "context": f"{char_name} is in a moment of vulnerability",
                "prompts": [
                    "Are you okay?",
                    "What's really bothering you?",
                    "You can trust me with this.",
                    "How can I help?"
                ]
            },
            {
                "context": f"{char_name} is asked about their expertise",
                "prompts": [
                    "How did you learn to do that?",
                    "Can you teach me?",
                    "What's the most important thing to know?",
                    "When did you first discover this talent?"
                ]
            },
            {
                "context": f"{char_name} faces a moral dilemma",
                "prompts": [
                    "What's the right thing to do here?",
                    "How do you decide in situations like this?",
                    "What would happen if we chose differently?",
                    "Have you ever regretted a similar choice?"
                ]
            },
            {
                "context": f"{char_name} shares a quiet moment",
                "prompts": [
                    "What are you thinking about?",
                    "Do you ever wonder what could have been?",
                    "What makes you happy?",
                    "Tell me something I don't know about you."
                ]
            }
        ]
        
        # Add occupation-specific scenarios
        if knowledge['occupation']:
            scenarios.append({
                "context": f"{char_name} dealing with {knowledge['occupation']} responsibilities",
                "prompts": [
                    f"What's the hardest part about being {knowledge['occupation']}?",
                    f"Why did you become {knowledge['occupation']}?",
                    f"What does a typical day as {knowledge['occupation']} look like?",
                    f"Any interesting stories from your work as {knowledge['occupation']}?"
                ]
            })
        
        # Add current situation scenarios if available
        if knowledge['current_situation']:
            scenarios.append({
                "context": f"{char_name} discusses their current situation: {knowledge['current_situation']}",
                "prompts": [
                    "How did things get to this point?",
                    "What's your plan to deal with this?",
                    "Who else knows about this situation?",
                    "What happens if this doesn't work out?"
                ]
            })
        
        # Character-specific scenarios based on extracted knowledge
        if knowledge['skills']:
            for skill in knowledge['skills'][:3]:  # Top 3 skills
                scenarios.append({
                    "context": f"{char_name} demonstrates their {skill}",
                    "prompts": [
                        f"How did you become so good at {skill}?",
                        f"What's the hardest part about {skill}?",
                        f"Can you show me how to {skill}?",
                        f"What mistakes do beginners make with {skill}?"
                    ]
                })
        
        # Equipment/spell scenarios
        if knowledge['equipment']:
            equipment = knowledge['equipment'][0] if knowledge['equipment'] else "equipment"
            scenarios.append({
                "context": f"{char_name} maintains or uses their {equipment}",
                "prompts": [
                    f"Where did you get your {equipment}?",
                    f"Has your {equipment} ever let you down?",
                    f"What's special about your {equipment}?",
                    f"Can I see your {equipment}?"
                ]
            })
        
        if knowledge['known_spells']:
            spell = knowledge['known_spells'][0] if knowledge['known_spells'] else "magic"
            scenarios.append({
                "context": f"{char_name} casts or discusses {spell}",
                "prompts": [
                    f"How does {spell} actually work?",
                    f"When did you first learn {spell}?",
                    f"What happens if {spell} goes wrong?",
                    f"Can you teach me {spell}?"
                ]
            })
        
        # Location-based scenarios from world info
        if knowledge['locations']:
            for location in knowledge['locations'][:2]:
                scenarios.append({
                    "context": f"At {location} with {char_name}",
                    "prompts": [
                        f"What do you think of {location}?",
                        f"How often do you come to {location}?",
                        f"Any memories associated with {location}?",
                        f"Who else might we meet at {location}?"
                    ]
                })
        
        # World-info based scenarios
        if knowledge['world_info']:
            for info in knowledge['world_info'][:2]:
                info_name = info['name']
                scenarios.append({
                    "context": f"Discussing {info_name} with {char_name}",
                    "prompts": [
                        f"Tell me more about {info_name}.",
                        f"How does {info_name} affect you?",
                        f"What's your opinion on {info_name}?",
                        f"Any stories about {info_name}?"
                    ]
                })
        
        # Backstory-based scenarios
        if knowledge['backstory_elements']:
            for element in knowledge['backstory_elements'][:2]:
                # Extract key event from backstory
                if 'expelled' in element:
                    scenarios.append({
                        "context": f"{char_name} reflects on being expelled",
                        "prompts": [
                            "Do you regret what led to your expulsion?",
                            "How did being expelled change your life?",
                            "Would you go back if you could?",
                            "What did you learn from that experience?"
                        ]
                    })
                elif 'survived' in element or 'crash' in element:
                    scenarios.append({
                        "context": f"{char_name} remembers surviving disaster",
                        "prompts": [
                            "How did you survive when others didn't?",
                            "Do you have survivor's guilt?",
                            "What kept you going?",
                            "How has it changed you?"
                        ]
                    })
        
        # Mannerism-based scenarios
        if 'physically_reactive' in knowledge['mannerisms']:
            scenarios.append({
                "context": f"{char_name} in a tense situation",
                "prompts": [
                    "You seem jumpy. What's wrong?",
                    "Try to stay calm.",
                    "Take a deep breath.",
                    "Is something making you nervous?"
                ]
            })
        
        if knowledge['fears']:
            for fear in knowledge['fears'][:2]:  # Top 2 fears
                scenarios.append({
                    "context": f"{char_name} confronts their fear of {fear}",
                    "prompts": [
                        "You seem tense. What's wrong?",
                        "We can face this together.",
                        "What happened to make you fear this?",
                        "How do you usually cope with this?"
                    ]
                })
        
        if knowledge['goals']:
            for goal in knowledge['goals'][:2]:  # Top 2 goals
                scenarios.append({
                    "context": f"{char_name} discusses their goal to {goal}",
                    "prompts": [
                        "What drives you to pursue this?",
                        "What obstacles stand in your way?",
                        "How will you know when you've succeeded?",
                        "What sacrifices have you made for this?"
                    ]
                })
        
        # Emotional trigger scenarios
        if knowledge['emotional_triggers']:
            for trigger in knowledge['emotional_triggers'][:2]:
                if trigger == 'financial_stress':
                    scenarios.append({
                        "context": f"{char_name} dealing with money problems",
                        "prompts": [
                            "How bad is the financial situation?",
                            "What's your plan to make money?",
                            "Who do you owe money to?",
                            "What happens if you can't pay?"
                        ]
                    })
                elif trigger == 'success_celebration':
                    scenarios.append({
                        "context": f"Celebrating a victory with {char_name}",
                        "prompts": [
                            "We did it! How does it feel?",
                            "What should we do to celebrate?",
                            "Who else should we tell?",
                            "What's next after this success?"
                        ]
                    })
        
        # Relationship-based scenarios
        relationship_scenarios = [
            {
                "context": f"Building trust with {char_name}",
                "prompts": [
                    "Why should I trust you?",
                    "Have you ever betrayed someone's trust?",
                    "What would you do if I betrayed you?",
                    "How do you know when to trust someone?"
                ]
            },
            {
                "context": f"Conflict with {char_name}",
                "prompts": [
                    "I think you're wrong about this.",
                    "Why won't you listen to me?",
                    "Maybe we should go our separate ways.",
                    "How can we resolve this?"
                ]
            },
            {
                "context": f"Celebrating with {char_name}",
                "prompts": [
                    "We did it! How should we celebrate?",
                    "I couldn't have done this without you.",
                    "What's your idea of a perfect celebration?",
                    "This calls for a toast!"
                ]
            }
        ]
        
        # Mix base, character-specific, and relationship scenarios
        all_scenarios = base_scenarios + scenarios + relationship_scenarios
        
        # Remove duplicates and randomly select
        unique_scenarios = []
        seen_contexts = set()
        for scenario in all_scenarios:
            if scenario['context'] not in seen_contexts:
                unique_scenarios.append(scenario)
                seen_contexts.add(scenario['context'])
        
        selected = random.sample(unique_scenarios, min(num_scenarios, len(unique_scenarios)))
        
        return selected

    def generate_multi_turn_conversation(self, character: Dict[str, Any], scenario: Dict[str, Any], turns: int = 3) -> List[Dict[str, str]]:
        """Generate a multi-turn conversation flow for deeper character exploration"""
        conversation_flows = []
        
        # Start with scenario context
        context = scenario['context']
        initial_prompts = scenario['prompts']
        
        # Create conversation branches
        for initial_prompt in initial_prompts[:2]:  # Use first 2 prompts
            flow = [{"role": "user", "content": initial_prompt}]
            
            # Generate follow-up questions based on expected response types
            follow_ups = self._generate_follow_up_questions(initial_prompt, character)
            
            for i in range(turns - 1):
                if i < len(follow_ups):
                    flow.append({"role": "user", "content": follow_ups[i]})
            
            conversation_flows.append({
                "context": context,
                "turns": flow
            })
        
        return conversation_flows

    def _generate_follow_up_questions(self, initial_prompt: str, character: Dict[str, Any]) -> List[str]:
        """Generate contextual follow-up questions"""
        # Map initial prompt types to follow-up patterns
        follow_up_patterns = {
            # For "how" questions
            "how": [
                "Can you give me a specific example?",
                "What was the most challenging part?",
                "Would you do it differently now?",
                "Who helped you along the way?"
            ],
            # For "what" questions  
            "what": [
                "Why is that important to you?",
                "How does that make you feel?",
                "Has it always been this way?",
                "What would change if that were different?"
            ],
            # For "why" questions
            "why": [
                "When did you first realize this?",
                "Does everyone see it that way?",
                "What experiences shaped this view?",
                "Could there be another explanation?"
            ],
            # For emotional/vulnerable moments
            "feel": [
                "How long have you felt this way?",
                "What helps when you feel like this?",
                "Have you told anyone else?",
                "What would make it better?"
            ],
            # For action/decision questions
            "should": [
                "What are the risks?",
                "What's your gut telling you?",
                "What would happen if we didn't?",
                "Who else should we consider?"
            ]
        }
        
        # Determine prompt type
        prompt_lower = initial_prompt.lower()
        prompt_type = None
        
        for key in follow_up_patterns:
            if key in prompt_lower:
                prompt_type = key
                break
        
        # Default follow-ups if no pattern matched
        if not prompt_type:
            return [
                "Tell me more about that.",
                "What makes you say that?",
                "How certain are you?",
                "What happens next?"
            ]
        
        # Return appropriate follow-ups
        return follow_up_patterns[prompt_type]

    def generate_exploration_prompts(self, character: Dict[str, Any], knowledge: Dict[str, Any]) -> List[str]:
        """Generate prompts that explore specific aspects of the character"""
        prompts = []
        
        # Trait exploration - enhanced to use extracted traits
        if knowledge['traits']:
            for trait in knowledge['traits'][:5]:
                prompts.extend([
                    f"People say you're {trait}. Is that accurate?",
                    f"When did you first become {trait}?",
                    f"What made you so {trait}?",
                    f"Do you ever wish you weren't so {trait}?"
                ])
        else:
            # Fallback trait questions
            prompts.extend([
                "What's your most defining trait?",
                "How would others describe you?",
                "What part of your personality do you struggle with?",
                "What trait has helped you most in life?"
            ])
        
        # Skill exploration - using actual extracted skills
        if knowledge['skills']:
            for skill in knowledge['skills'][:3]:
                prompts.extend([
                    f"Show me your {skill}.",
                    f"What's your greatest achievement with {skill}?",
                    f"Who taught you {skill}?",
                    f"What's the secret to mastering {skill}?"
                ])
        
        # Species-specific questions if not human
        if knowledge['species'] and knowledge['species'].lower() != 'human':
            species = knowledge['species']
            prompts.extend([
                f"What's it like being a {species}?",
                f"Are there any misconceptions about {species}?",
                f"What advantages does being a {species} give you?",
                f"Do you face any challenges as a {species}?"
            ])
        
        # Appearance-based questions if we have details
        if knowledge['appearance']:
            for feature in knowledge['appearance'][:2]:
                prompts.append(f"Is there a story behind your {feature}?")
        
        # Backstory exploration - enhanced with actual backstory elements
        if knowledge['backstory_elements']:
            prompts.extend([
                "Tell me more about your past.",
                "What event from your past still affects you today?",
                "If you could change one thing about your past, what would it be?"
            ])
            # Add specific questions based on backstory elements
            for element in knowledge['backstory_elements'][:2]:
                if 'born' in element:
                    prompts.append("What was your hometown like?")
                if 'expelled' in element or 'rejected' in element:
                    prompts.append("How did that rejection affect you?")
                if 'war' in element or 'army' in element:
                    prompts.append("What did you learn from your military experience?")
        else:
            # Generic backstory questions
            prompts.extend([
                "What's your earliest memory?",
                "Who influenced you most growing up?",
                "What was the turning point in your life?",
                "What's a story from your past that defines who you are?"
            ])
        
        # Emotional exploration - enhanced with speech patterns
        emotion_prompts = [
            "What brings you joy?",
            "What makes you angry?",
            "When was the last time you cried?",
            "What's your greatest fear?",
            "What makes you feel alive?",
            "When do you feel most vulnerable?",
            "What emotion do you struggle with most?",
            "How do you handle stress?"
        ]
        
        # Modify based on speech patterns
        if 'nervous_laughter' in knowledge['speech_patterns']:
            emotion_prompts.append("Why do you laugh when you're nervous?")
        if 'verbal_hesitation' in knowledge['speech_patterns']:
            emotion_prompts.append("What makes you hesitate when speaking?")
        if 'trailing_off' in knowledge['speech_patterns']:
            emotion_prompts.append("What thoughts make you trail off mid-sentence?")
        
        prompts.extend(emotion_prompts)
        
        # Relationship exploration - enhanced with mannerisms
        relationship_prompts = [
            "What do you look for in a friend?",
            "How do you show someone you care?",
            "What's your biggest relationship regret?",
            "Who understands you best?",
            "What walls do you put up with people?",
            "How do you handle conflict?",
            "What makes you trust someone?",
            "When do you feel most connected to others?"
        ]
        
        # Add mannerism-specific relationship questions
        if 'physically_reactive' in knowledge['mannerisms']:
            relationship_prompts.append("Why are you so jumpy around people?")
        if 'shows_discomfort' in knowledge['mannerisms']:
            relationship_prompts.append("What makes you uncomfortable in social situations?")
        
        prompts.extend(relationship_prompts)
        
        # World-specific questions based on world info
        if knowledge['world_info']:
            for info in knowledge['world_info'][:3]:
                content = info['content'].lower()
                if 'debt' in content:
                    prompts.extend([
                        "How did you get into debt?",
                        "What's your plan to pay it off?",
                        "Who are you in debt to?"
                    ])
                if 'business' in content or 'agency' in content:
                    prompts.extend([
                        "How's business going?",
                        "What made you start this venture?",
                        "What's your vision for the future?"
                    ])
                if 'guild' in content:
                    prompts.extend([
                        "What's your relationship with other guilds?",
                        "Why did other guilds reject you?",
                        "What makes your guild different?"
                    ])
        
        # Current situation questions
        if knowledge['current_situation']:
            prompts.extend([
                "How are you handling the current situation?",
                "What's your immediate priority?",
                "Who can help you with this?",
                "What's your backup plan?"
            ])
        
        # Goal-specific questions
        if knowledge['goals']:
            for goal in knowledge['goals'][:2]:
                prompts.extend([
                    f"Why is {goal} so important to you?",
                    f"What's stopping you from achieving {goal}?",
                    f"Who supports your goal to {goal}?"
                ])
        
        # Philosophical exploration
        prompts.extend([
            "What do you believe happens after death?",
            "Is there such a thing as destiny?",
            "What's the meaning of life?",
            "Do you believe in second chances?",
            "Is it better to be feared or loved?",
            "What's worth fighting for?",
            "Can people truly change?",
            "What legacy do you want to leave?"
        ])
        
        # Equipment/possession questions
        if knowledge['equipment']:
            for item in knowledge['equipment'][:2]:
                prompts.extend([
                    f"What's the story behind your {item}?",
                    f"Would you ever part with your {item}?"
                ])
        
        # Spell/magic questions
        if knowledge['known_spells']:
            prompts.extend([
                "What's your relationship with magic?",
                "Which spell is most useful to you?",
                "Have you ever had a spell backfire?",
                "What spell do you wish you could cast?"
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_prompts = []
        for prompt in prompts:
            if prompt not in seen:
                seen.add(prompt)
                unique_prompts.append(prompt)
        
        return unique_prompts

    def calculate_prompt_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate semantic similarity between two prompts"""
        # Simple word overlap similarity (can be enhanced with embeddings)
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def deduplicate_prompts(self, prompts: List[str], similarity_threshold: float = 0.7) -> List[str]:
        """Remove similar prompts to ensure diversity"""
        if not prompts:
            return []
            
        unique_prompts = [prompts[0]]
        
        for prompt in prompts[1:]:
            is_unique = True
            for unique_prompt in unique_prompts:
                similarity = self.calculate_prompt_similarity(prompt, unique_prompt)
                if similarity > similarity_threshold:
                    is_unique = False
                    break
            
            if is_unique:
                unique_prompts.append(prompt)
                
        return unique_prompts

    def evaluate_response_quality(self, response: str, character: Dict[str, Any], prompt: str) -> Dict[str, Any]:
        """Evaluate the quality of a generated response"""
        quality_metrics = {
            'length_score': 0.0,
            'character_consistency': 0.0,
            'relevance_score': 0.0,
            'diversity_score': 0.0,
            'overall_score': 0.0,
            'issues': []
        }
        
        # Length evaluation
        word_count = len(response.split())
        if word_count < 5:
            quality_metrics['length_score'] = 0.0
            quality_metrics['issues'].append("Response too short")
        elif word_count < 20:
            quality_metrics['length_score'] = 0.5
        elif word_count < 200:
            quality_metrics['length_score'] = 1.0
        else:
            quality_metrics['length_score'] = 0.8  # Slightly penalize very long responses
        
        # Character consistency check
        char_name = character.get('name', 'Assistant')
        
        # Check for character name consistency
        if char_name.lower() in response.lower():
            # Character referring to themselves in third person (usually bad)
            quality_metrics['character_consistency'] -= 0.3
            quality_metrics['issues'].append("Character refers to self in third person")
        
        # Check for prompt leakage
        if any(token in response for token in ['<|system|>', '<|user|>', '<|assistant|>', '<|endoftext|>']):
            quality_metrics['character_consistency'] = 0.0
            quality_metrics['issues'].append("Contains formatting tokens")
            
        # Check for meta-commentary
        meta_phrases = ['as an ai', 'as a language model', 'i cannot', 'i don\'t have', 'my training']
        if any(phrase in response.lower() for phrase in meta_phrases):
            quality_metrics['character_consistency'] -= 0.5
            quality_metrics['issues'].append("Contains meta-commentary")
        
        # Base character consistency score
        if not quality_metrics['issues']:
            quality_metrics['character_consistency'] = 1.0
        else:
            quality_metrics['character_consistency'] = max(0.0, quality_metrics['character_consistency'] + 1.0)
        
        # Relevance to prompt
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        # Check if response addresses the prompt
        common_words = prompt_words.intersection(response_words)
        if len(common_words) > 2:
            quality_metrics['relevance_score'] = min(1.0, len(common_words) / 10)
        else:
            quality_metrics['relevance_score'] = 0.3
            quality_metrics['issues'].append("Low relevance to prompt")
        
        # Diversity score (vocabulary richness)
        unique_words = len(set(response.lower().split()))
        total_words = len(response.split())
        if total_words > 0:
            quality_metrics['diversity_score'] = min(1.0, unique_words / total_words * 2)
        
        # Calculate overall score
        quality_metrics['overall_score'] = (
            quality_metrics['length_score'] * 0.2 +
            quality_metrics['character_consistency'] * 0.4 +
            quality_metrics['relevance_score'] * 0.2 +
            quality_metrics['diversity_score'] * 0.2
        )
        
        return quality_metrics

    def generate_emotional_variations(self, base_prompt: str, emotions: List[str] = None) -> List[str]:
        """Generate variations of a prompt with different emotional contexts"""
        if emotions is None:
            emotions = ['neutral', 'happy', 'sad', 'angry', 'worried', 'excited', 'tired', 'curious']
        
        emotional_modifiers = {
            'happy': [
                "*smiling* {}",
                "*laughing* {}",
                "*excitedly* {}",
                "I'm in such a good mood! {}"
            ],
            'sad': [
                "*sighs* {}",
                "*looking down* {}",
                "*quietly* {}",
                "I've been feeling down... {}"
            ],
            'angry': [
                "*frustrated* {}",
                "*raising voice* {}",
                "*clenching fists* {}",
                "I can't believe this! {}"
            ],
            'worried': [
                "*nervously* {}",
                "*biting lip* {}",
                "*anxiously* {}",
                "I'm concerned... {}"
            ],
            'excited': [
                "*eyes lighting up* {}",
                "*bouncing* {}",
                "*eagerly* {}",
                "Oh! Oh! {}"
            ],
            'tired': [
                "*yawning* {}",
                "*rubbing eyes* {}",
                "*wearily* {}",
                "Sorry, I'm exhausted... {}"
            ],
            'curious': [
                "*leaning forward* {}",
                "*tilting head* {}",
                "*intrigued* {}",
                "I've been wondering... {}"
            ],
            'neutral': ["{}"]  # No modification
        }
        
        variations = []
        for emotion in emotions:
            if emotion in emotional_modifiers:
                modifier = random.choice(emotional_modifiers[emotion])
                variations.append(modifier.format(base_prompt))
        
        return variations

    def enhance_prompt_with_context(self, prompt: str, character: Dict[str, Any], scenario_context: str = None) -> str:
        """Enhance a prompt with contextual information"""
        enhanced_prompts = []
        
        # Add scenario context if provided
        if scenario_context:
            enhanced_prompts.append(f"[Context: {scenario_context}] {prompt}")
        
        # Add time-based context
        time_contexts = [
            "It's late at night. ",
            "The sun is setting. ",
            "It's early morning. ",
            "In the middle of a storm. ",
            "During a quiet moment. "
        ]
        
        # Add location-based context (if applicable from character card)
        scenario = character.get('scenario', '')
        if 'tavern' in scenario.lower():
            enhanced_prompts.append(f"*in the tavern* {prompt}")
        elif 'forest' in scenario.lower():
            enhanced_prompts.append(f"*in the forest* {prompt}")
        elif 'city' in scenario.lower():
            enhanced_prompts.append(f"*in the city* {prompt}")
        
        # Add relationship progression
        relationship_stages = [
            "*just met* {}",
            "*getting to know each other* {}",
            "*as friends* {}",
            "*as close companions* {}",
            "*after everything we've been through* {}"
        ]
        
        # Return one enhanced version or original
        if enhanced_prompts:
            return random.choice(enhanced_prompts + [prompt])
        else:
            # Add a random contextual enhancement
            if random.random() < 0.3:  # 30% chance
                time_context = random.choice(time_contexts)
                return time_context + prompt
            elif random.random() < 0.5:  # 50% of remaining
                stage = random.choice(relationship_stages)
                return stage.format(prompt)
            else:
                return prompt

    def analyze_temporal_distribution(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal distribution of dataset samples"""
        if not dataset:
            return {}

        temporal_counts = {"past": 0, "present": 0, "future": 0, "unknown": 0}
        relationship_counts = {}
        past_relationship_samples = []
        future_relationship_samples = []

        for sample in dataset:
            # Analyze system prompt to determine temporal context
            system_content = sample.get('messages', [{}])[0].get('content', '')
            user_content = sample.get('messages', [{}])[1].get(
                'content', '') if len(sample.get('messages', [])) > 1 else ''

            # Detect temporal context from system prompt
            temporal_context = "unknown"
            if "speaking with a family member about your past" in system_content:
                temporal_context = "past"
                relationship_counts["family"] = relationship_counts.get(
                    "family", 0) + 1
                past_relationship_samples.append(user_content[:100])
            elif "reminiscing with an old friend" in system_content:
                temporal_context = "past"
                relationship_counts["friend"] = relationship_counts.get(
                    "friend", 0) + 1
                past_relationship_samples.append(user_content[:100])
            elif "reflecting with someone who taught" in system_content:
                temporal_context = "past"
                relationship_counts["mentor"] = relationship_counts.get(
                    "mentor", 0) + 1
                past_relationship_samples.append(user_content[:100])
            elif "past romantic connection" in system_content:
                temporal_context = "past"
                relationship_counts["romance"] = relationship_counts.get(
                    "romance", 0) + 1
                past_relationship_samples.append(user_content[:100])
            elif "known each other for years" in system_content:
                temporal_context = "future"
                future_relationship_samples.append(user_content[:100])
            elif "meeting the User for the first time" in system_content:
                temporal_context = "present"
            elif any(past_keyword in user_content.lower() for past_keyword in ["childhood", "growing up", "when you were young", "your father", "your mother"]):
                temporal_context = "past"
            elif any(future_keyword in user_content.lower() for future_keyword in ["after all this time", "our relationship", "years together"]):
                temporal_context = "future"
            else:
                temporal_context = "present"

            temporal_counts[temporal_context] += 1

        total_samples = sum(temporal_counts.values())
        temporal_percentages = {k: (v/total_samples)*100 if total_samples > 0 else 0
                                for k, v in temporal_counts.items()}

        return {
            'temporal_distribution': temporal_counts,
            'temporal_percentages': temporal_percentages,
            'relationship_contexts': relationship_counts,
            # First 5 examples
            'past_relationship_samples': past_relationship_samples[:5],
            # First 5 examples
            'future_relationship_samples': future_relationship_samples[:5],
            # Out of 3 temporal buckets
            'temporal_diversity_score': len([v for v in temporal_counts.values() if v > 0]) / 3 * 100
        }

    def generate_prompts_from_greetings(self, character: Dict[str, Any], knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prompts based on alternate greetings which show different scenarios"""
        prompts = []
        
        # Process first message
        first_mes = character.get('first_mes', '')
        if first_mes:
            # Analyze the greeting scenario
            greeting_prompts = self._analyze_greeting_for_prompts(first_mes, "initial greeting")
            if greeting_prompts:
                prompts.append({
                    'context': 'Initial meeting scenario',
                    'prompts': greeting_prompts
                })
        
        # Process alternate greetings
        alternate_greetings = character.get('alternate_greetings', [])
        for i, greeting in enumerate(alternate_greetings[:3]):  # Process up to 3 alternates
            if greeting:
                greeting_prompts = self._analyze_greeting_for_prompts(greeting, f"alternate scenario {i+1}")
                if greeting_prompts:
                    # Determine context from greeting content
                    context = self._extract_greeting_context(greeting)
                    prompts.append({
                        'context': context,
                        'prompts': greeting_prompts
                    })
        
        return prompts

    def _analyze_greeting_for_prompts(self, greeting: str, greeting_type: str) -> List[str]:
        """Analyze a greeting to generate contextual prompts"""
        prompts = []
        greeting_lower = greeting.lower()
        
        # Financial stress scenario
        if any(word in greeting_lower for word in ['money', 'gold', 'coin', 'rent', 'debt', 'pay']):
            prompts.extend([
                "How much money do you need exactly?",
                "What happened to your finances?",
                "Who are you in debt to?",
                "What's your plan to make money?"
            ])
        
        # Celebration/success scenario
        if any(word in greeting_lower for word in ['celebrate', 'success', 'did it', 'woohoo', 'toast']):
            prompts.extend([
                "What are we celebrating?",
                "How did you pull it off?",
                "What's next after this success?",
                "Who else helped make this happen?"
            ])
        
        # Danger/trouble scenario
        if any(word in greeting_lower for word in ['danger', 'trouble', 'help', 'emergency', 'problem']):
            prompts.extend([
                "What kind of trouble are you in?",
                "How can I help?",
                "Who's after you?",
                "How urgent is this?"
            ])
        
        # Business/work scenario
        if any(word in greeting_lower for word in ['job', 'work', 'contract', 'guild', 'agency', 'business']):
            prompts.extend([
                "What kind of job is it?",
                "What's the pay like?",
                "Why did you choose this line of work?",
                "Any interesting contracts lately?"
            ])
        
        # Location-specific prompts
        if 'tavern' in greeting_lower:
            prompts.extend([
                "Come here often?",
                "What's good to drink here?",
                "Know any interesting people here?"
            ])
        elif 'warehouse' in greeting_lower or 'office' in greeting_lower:
            prompts.extend([
                "How long have you had this place?",
                "Business been good?",
                "What kind of work do you do here?"
            ])
        
        # Emotional state prompts based on actions/descriptions
        if '*sigh*' in greeting or '*groan*' in greeting or 'frustrated' in greeting_lower:
            prompts.extend([
                "Rough day?",
                "What's got you so frustrated?",
                "Anything I can do to help?"
            ])
        elif '*smile*' in greeting or '*grin*' in greeting or 'excited' in greeting_lower:
            prompts.extend([
                "You seem happy!",
                "What's the good news?",
                "Share the excitement!"
            ])
        
        return prompts

    def _extract_greeting_context(self, greeting: str) -> str:
        """Extract a descriptive context from a greeting"""
        greeting_lower = greeting.lower()
        
        # Identify the primary scenario
        if 'money' in greeting_lower or 'debt' in greeting_lower or 'rent' in greeting_lower:
            return "Financial crisis scenario"
        elif 'celebrate' in greeting_lower or 'success' in greeting_lower:
            return "Celebration scenario"
        elif 'danger' in greeting_lower or 'trouble' in greeting_lower:
            return "Danger/emergency scenario"
        elif 'tavern' in greeting_lower:
            return "Tavern meeting scenario"
        elif 'job' in greeting_lower or 'contract' in greeting_lower:
            return "Business opportunity scenario"
        elif 'visitor' in greeting_lower or 'customer' in greeting_lower:
            return "Unexpected visitor scenario"
        else:
            return "Alternative meeting scenario"
