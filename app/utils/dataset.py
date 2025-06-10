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

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages synthetic dataset generation and processing"""
    
    def __init__(self, preferred_engine: Optional[str] = None, enable_intelligent_generation: bool = True):
        logger.info(f"DatasetManager initializing with preferred_engine: {preferred_engine}")
        self.inference_engine = get_inference_engine(preferred_engine)
        logger.info(f"DatasetManager created with engine: {self.inference_engine.name}")
        # For DanChat-2 we only need a *single* chat template – the chat
        # wrapper (<|user|> …) is added later by vLLM/HF tokenizer.
        self.templates = [("chat", "{user_prompt}")]
        
        # Configuration for intelligent prompt generation
        self.enable_intelligent_generation = enable_intelligent_generation
        if not enable_intelligent_generation:
            logger.info("🔧 Intelligent prompt generation is DISABLED - using static prompts only")

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
        logger.info("🛡️ Creating DatasetManager with conservative settings (intelligent generation disabled)")
        return cls(preferred_engine=preferred_engine, enable_intelligent_generation=False)
    
    async def test_inference_engine(self) -> bool:
        """Test the inference engine with a simple prompt for debugging"""
        if not self.inference_engine:
            logger.error("❌ No inference engine available")
            return False
            
        try:
            logger.info(f"🔧 Testing inference engine: {self.inference_engine.name}")
            
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
                logger.error(f"❌ Inference engine test FAILED - empty response")
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
                raise RuntimeError("Inference engine does not implement `generate` method")

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
            raise RuntimeError(f"Text generation failed ({self.inference_engine.name}): {str(e)}")
    
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
            raise RuntimeError(f"Batch text generation failed ({self.inference_engine.name}): {str(e)}")
    
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
            pool = _rnd.sample(existing_dataset, min(len(existing_dataset), context_samples * num_questions))
        else:
            pool = []

        prompts = []
        for i in range(num_questions):
            # build context for this prompt
            interactions_block = ""
            examples_for_prompt = []
            if pool:
                # pop random subset for this prompt (without replacement if enough)
                selected = pool[:context_samples] if len(pool) >= context_samples else pool
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
                    interactions_block = "Here are some previous interactions to inspire you:\n" + "\n\n".join(formatted) + "\n\n"

            prompt_txt = (
                "You are brainstorming conversation starters for a chat with the following character.\n"
                "Based on the character information and the sample dialogue below, write ONE concise and engaging question that a user might ask next.\n"
                "Respond with ONLY the question itself.\n\n"
                f"{card_block}\n\n" + interactions_block + "Question:"
            )
            prompts.append((prompt_txt, examples_for_prompt))

        # Determine whether we can leverage batched generation
        batch_size = min(num_questions, 50) if hasattr(self.inference_engine, 'generate_batch') else 1
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
            lines.append(f"{char_name}'s personality: {substitute_vars(pers.strip())}")

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
            lines.append(f"Scenario: {substitute_vars(card.get('scenario').strip())}")
        else:
            lines.append(f"Scenario: User is asking {char_name} questions.")

        # Clean formatting: single newlines, no extra whitespace
        return "\n".join(lines).strip()
    
    async def generate_dataset(self, character: Dict[str, Any], num_samples: int = 80,
                             max_tokens: Optional[int] = None, temperature: float = 0.8,
                             top_p: float = 0.9, progress_callback: Optional[Callable] = None,
                             append_to_existing: bool = True) -> List[Dict[str, Any]]:
        """Generate synthetic dataset for character using efficient batching"""
        
        # Test inference engine if intelligent generation is enabled
        if self.enable_intelligent_generation:
            engine_working = await self.test_inference_engine()
            if not engine_working:
                logger.warning("🔄 Disabling intelligent generation due to engine issues - using static prompts only")
                self.enable_intelligent_generation = False
        
        card_block = self._make_card_block(character)
        
        # Load existing dataset if append_to_existing is True
        existing_samples = []
        if append_to_existing:
            existing_dataset = self.load_dataset(character)
            if existing_dataset:
                existing_samples = existing_dataset
                logger.info(f"📂 Found existing dataset with {len(existing_samples)} samples")
        
        # Calculate how many new samples to generate
        existing_count = len(existing_samples)
        if existing_count >= num_samples:
            logger.info(f"✅ Dataset already has {existing_count} samples (requested: {num_samples})")
            return existing_samples[:num_samples]  # Return requested amount
        
        new_samples_needed = num_samples - existing_count
        logger.info(f"🎯 Generating {new_samples_needed} new samples to reach {num_samples} total")
        
        samples = existing_samples.copy()
        
        # ------------------------------------------------------------------
        # Determine which baseline prompts still need coverage.  We collect
        # *unique* user messages already present to avoid duplicates, then
        # enqueue any new questions that were added (e.g. via UI augmentation)
        # for the next generation run – even when the dataset already exists.
        # ------------------------------------------------------------------

        seen_user_prompts: set[str] = {
            sample['messages'][1]['content'] for sample in existing_samples
            if isinstance(sample, dict) and 'messages' in sample and len(sample['messages']) > 1
        }

        baseline_prompts: list[str] = [q for q in self.default_user_prompts if q not in seen_user_prompts]

        # Length buckets following 2024-2025 best-practice distribution
        # (40% short 1-200 tok, 45% medium 200-500 tok, 15% long 500-800 tok)
        length_buckets = [
            ("short", 0.40, 200),
            ("medium", 0.45, 500),
            ("long", 0.15, 800),
        ]

        def _sample_max_tokens() -> int:
            names, probs, toks = zip(*[(n, p, t) for n, p, t in length_buckets])
            bucket_name = random.choices(names, weights=probs, k=1)[0]
            # Map name -> token limit
            token_map = {n: t for n, _, t in length_buckets}
            return token_map[bucket_name]

        # ------------------------------------------------------------------
        # Build the prompt metadata list (baseline first, then random samples)
        # ------------------------------------------------------------------

        # Pre-populate with baseline prompts so they are guaranteed inclusion
        prompts_data: list[Dict[str, Any]] = []

        char_name_for_prompts = character.get('name', 'Assistant')

        for bp in baseline_prompts:
            # Use present context for baseline prompts (default behavior)
            temporal_system_prompt = self._generate_temporal_system_prompt(character, "present")
            danschat_prompt = f"<|system|>{temporal_system_prompt}<|endoftext|><|user|>{bp}<|endoftext|><|assistant|>{char_name_for_prompts}:"
            prompts_data.append({
                'prompt': bp,
                'full_prompt': danschat_prompt,
                'template_mode': 'baseline',
                'temporal_context': 'present',
                'relationship_context': None,
                'char_name': char_name_for_prompts,
                'max_tokens': _sample_max_tokens(),
            })

        # Pre-generate prompts for the remaining samples we still need
        new_samples_needed_after_baseline = max(0, new_samples_needed - len(baseline_prompts))

        # Generate up to 1.5× the remaining need (for filtering margin)
        random_prompt_target = int(new_samples_needed_after_baseline * 1.5)

        # ------------------------------------------------------------------
        # Random prompt generation loop (same logic as before, but uses the new
        # target size so we don't waste time over-generating when baseline is
        # large).
        # ------------------------------------------------------------------

        generated_random = 0  # Counter for random prompts added

        # Generate random prompts with intelligent temporal contexts
        random_prompt_tasks = []
        for i in range(random_prompt_target * 2):  # keep safety margin
            if generated_random >= random_prompt_target:
                break
            random_prompt_tasks.append(self._generate_single_random_prompt(character, i))
        
        # Process random prompts in batches to avoid overwhelming the system
        batch_size = 10
        for batch_start in range(0, len(random_prompt_tasks), batch_size):
            batch_tasks = random_prompt_tasks[batch_start:batch_start + batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.debug(f"Error in random prompt generation: {result}")
                    continue
                elif result:  # Valid prompt data
                    prompts_data.append(result)
                    generated_random += 1
                    
                    if generated_random >= random_prompt_target:
                        break
            
            if generated_random >= random_prompt_target:
                break
        
        # Group prompts by desired max_tokens so that batched generation obeys
        # length distribution without sacrificing speed
        prompts_grouped: Dict[int, List[Dict[str, Any]]] = {}
        for item in prompts_data:
            prompts_grouped.setdefault(item['max_tokens'], []).append(item)

        # Determine base batch size once (may still be 1 if batching unsupported)
        base_batch_size = 50 if hasattr(self.inference_engine, 'generate_batch') else 1

        processed_count = 0
        logger.info(f"📊 Starting batch processing: {len(prompts_data)} prompts prepared, batch_size={base_batch_size}")
        
        for bucket_max_tokens, bucket_prompts in prompts_grouped.items():
            logger.info(f"📊 Processing {len(bucket_prompts)} prompts with max_tokens={bucket_max_tokens}")
            batch_size = base_batch_size
            char_name = character.get('name', 'Assistant')  # Ensure available for logging/validation
            for batch_start in range(0, len(bucket_prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(bucket_prompts))
                batch_prompts = bucket_prompts[batch_start:batch_end]
                full_prompts = [item['full_prompt'] for item in batch_prompts]
                
                # Generate the replies respecting bucket-level max_tokens
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
                
                logger.info(f"🔥 Got {len(replies)} replies from {self.inference_engine.name}")
                
                # Process batch results
                for i, (prompt_data, reply) in enumerate(zip(batch_prompts, replies)):
                    try:
                        # ✅ VALIDATE REPLY QUALITY AND FORMAT
                        reply_str = str(reply).strip()
                        
                        # Check for obvious prompt formatting leakage
                        format_issues = []
                        if '<|system|>' in reply_str:
                            format_issues.append("Contains <|system|> token")
                        if '<|user|>' in reply_str:
                            format_issues.append("Contains <|user|> token")
                        if '<|assistant|>' in reply_str:
                            format_issues.append("Contains <|assistant|> token")
                        if '<|endoftext|>' in reply_str:
                            format_issues.append("Contains <|endoftext|> token")
                        
                        if format_issues:
                            logger.warning(f"🔴 Reply {batch_start+i} has format leakage: {', '.join(format_issues)}")
                            logger.warning(f"   Reply preview: '{reply_str[:150]}{'...' if len(reply_str) > 150 else ''}'")
                        
                        # Quality filters
                        word_count = len(reply_str.split())
                        if word_count < 1:
                            logger.warning(f"❌ Reply {batch_start+i} filtered: too short (word_count={word_count})")
                            continue
                        if word_count > bucket_max_tokens * 1.5:  # generous upper bound
                            logger.warning(f"❌ Reply {batch_start+i} filtered: too long (word_count={word_count} > {bucket_max_tokens * 1.5})")
                            continue
                        
                        # Log sample replies for first few batches
                        if batch_start < 24 and i == 0:  # First 3 batches, first reply each
                            logger.info(f"📤 SAMPLE REPLY from batch {batch_start}, reply {i}:")
                            logger.info(f"   User: '{prompt_data['prompt'][:80]}{'...' if len(prompt_data['prompt']) > 80 else ''}'")
                            logger.info(f"   {char_name}: '{reply_str[:150]}{'...' if len(reply_str) > 150 else ''}'")
                            logger.info(f"   (Word count: {word_count})")
                        
                        # Create ChatML sample with temporal system prompt
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
                        
                        # Update progress (account for existing samples)
                        if progress_callback:
                            total_current = len(samples)
                            progress_callback(min(total_current / num_samples, 1.0))
                        
                        # Stop if we have enough total samples
                        if len(samples) >= num_samples:
                            break
                            
                    except Exception as e:
                        logger.debug(f"Error processing sample: {e}")
                        continue
                
                # Stop if we have enough total samples
                if len(samples) >= num_samples:
                    break
                
                # Small delay between different length buckets for stability
                await asyncio.sleep(0.2)
            
            # End inner batch loop
        
            # Validate prompt structure to catch template errors early
            invalid_prompts = []
            for idx, prompt in enumerate(full_prompts):
                if not prompt.startswith('<|system|>'):
                    invalid_prompts.append(f"Prompt {idx}: Missing <|system|> start")
                if '<|endoftext|><|user|>' not in prompt:
                    invalid_prompts.append(f"Prompt {idx}: Missing <|endoftext|><|user|> transition")
                if f'<|endoftext|><|assistant|>{char_name}:' not in prompt:
                    invalid_prompts.append(f"Prompt {idx}: Missing <|endoftext|><|assistant|>{char_name}:")
                if prompt.count('<|system|>') != 1:
                    invalid_prompts.append(f"Prompt {idx}: Multiple or missing <|system|> tags")
                if prompt.count('<|endoftext|>') != 2:
                    invalid_prompts.append(f"Prompt {idx}: Expected exactly 2 <|endoftext|> tags, found {prompt.count('<|endoftext|>')}")
            if invalid_prompts:
                logger.error(f"🚨 BATCH VALIDATION FAILURE ({batch_start}-{batch_end}): {len(invalid_prompts)} issues")
                for issue in invalid_prompts[:5]:
                    logger.error(f"  ❌ {issue}")
                if len(invalid_prompts) > 5:
                    logger.error(f"  ... and {len(invalid_prompts) - 5} more issues")
        
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
            logger.info(f"   Success rate: {new_generated/len(prompts_data)*100:.1f}%")
        logger.info(f"   Engine used: {self.inference_engine.name}")
        logger.info(f"   Batch size: {base_batch_size}")
        
        # Temporal distribution analysis
        temporal_counts = {"past": 0, "present": 0, "future": 0}
        relationship_counts = {}
        intelligent_prompt_count = 0
        
        for prompt_data in prompts_data[:new_generated]:  # Only count new samples
            temporal_ctx = prompt_data.get('temporal_context', 'present')
            temporal_counts[temporal_ctx] += 1
            
            # Count intelligent prompts
            if prompt_data.get('intelligent_prompt_data'):
                intelligent_prompt_count += 1
            
            rel_ctx = prompt_data.get('relationship_context')
            if rel_ctx:
                relationship_counts[rel_ctx] = relationship_counts.get(rel_ctx, 0) + 1
        
        total_temporal = sum(temporal_counts.values())
        if total_temporal > 0:
            logger.info(f"📊 TEMPORAL DISTRIBUTION:")
            for temporal, count in temporal_counts.items():
                pct = (count / total_temporal) * 100
                logger.info(f"   {temporal.title()}: {count} samples ({pct:.1f}%)")
            
            logger.info(f"🧠 INTELLIGENT PROMPTS: {intelligent_prompt_count} samples ({(intelligent_prompt_count/total_temporal)*100:.1f}%)")
            
            if relationship_counts:
                logger.info(f"📊 RELATIONSHIP CONTEXTS:")
                for rel, count in relationship_counts.items():
                    logger.info(f"   {rel}: {count} samples")
        
        # Spot check final samples for consistency
        if samples:
            sample_chars = set()
            for sample in samples[-min(10, new_generated):]:  # Check last 10 new samples
                assistant_msg = sample['messages'][2]['content']
                # Look for character name at start of response
                first_words = assistant_msg.split()[:3]
                sample_chars.update(first_words)
            
            logger.info(f"   Character consistency check: {sample_chars}")
        
        # 💾 Auto-save dataset
        self.save_dataset(character, samples)
        
        logger.info(f"Generated {len(samples)} total samples ({new_generated} new) using {self.inference_engine.name}")
        return samples
    
    def analyze_dataset_quality(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dataset quality metrics"""
        if not dataset:
            return {}
        
        # Extract responses
        responses = [sample['messages'][2]['content'] for sample in dataset]
        
        # Calculate metrics
        total_samples = len(dataset)
        avg_length = sum(len(response.split()) for response in responses) / total_samples
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
            user_content = sample.get('messages', [{}])[1].get('content', '') if len(sample.get('messages', [])) > 1 else ''
            
            # Detect temporal context from system prompt
            temporal_context = "unknown"
            if "speaking with a family member about your past" in system_content:
                temporal_context = "past"
                relationship_counts["family"] = relationship_counts.get("family", 0) + 1
                past_relationship_samples.append(user_content[:100])
            elif "reminiscing with an old friend" in system_content:
                temporal_context = "past"
                relationship_counts["friend"] = relationship_counts.get("friend", 0) + 1
                past_relationship_samples.append(user_content[:100])
            elif "reflecting with someone who taught" in system_content:
                temporal_context = "past"
                relationship_counts["mentor"] = relationship_counts.get("mentor", 0) + 1
                past_relationship_samples.append(user_content[:100])
            elif "past romantic connection" in system_content:
                temporal_context = "past"
                relationship_counts["romance"] = relationship_counts.get("romance", 0) + 1
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
            'past_relationship_samples': past_relationship_samples[:5],  # First 5 examples
            'future_relationship_samples': future_relationship_samples[:5],  # First 5 examples
            'temporal_diversity_score': len([v for v in temporal_counts.values() if v > 0]) / 3 * 100  # Out of 3 temporal buckets
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
            chat_text = re.sub(r"<\|im_start\|>\s*system[\s\S]*?<\|im_end\|>\n?", "", chat_text, flags=re.IGNORECASE)
            # Pattern 2 – DanChat style: "<|system|> ... <|endoftext|>"
            chat_text = re.sub(r"<\|system\|>[\s\S]*?<\|endoftext\|>", "", chat_text, flags=re.IGNORECASE)
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
        processed_dataset = processed_dataset.filter(lambda x: len(x["input_ids"]) > 0)
        
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
            logger.info(f"💾 Saved dataset with {len(dataset)} samples to {dataset_path}")
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
                logger.info(f"📂 Loaded existing dataset with {len(dataset)} samples from {dataset_path}")
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
                raise ValueError("Invalid dataset file: missing 'dataset' list")

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
            logger.info(f"Imported dataset with {len(imported_dataset)} samples (mode={merge_mode})")
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
            unique_indices = sorted({i for i in indices if 0 <= i < len(dataset)}, reverse=True)
            if not unique_indices:
                return False

            for idx in unique_indices:
                dataset.pop(idx)

            self.save_dataset(character, dataset)
            logger.info(f"🗑️ Removed {len(unique_indices)} samples from dataset (remaining: {len(dataset)})")
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
            logger.warning(f"No inference engine available for intelligent prompt generation")
            return []
        
        # Build character analysis prompt
        card_info = f"Character: {char_name}\nDescription: {description}\nPersonality: {personality}\nScenario: {scenario}"
        
        if temporal_context == "past":
            analysis_prompt = f"""Analyze this character card and generate {num_prompts} specific questions for exploring their past:

{card_info}

For each question, identify:
1. WHO should ask it (specific person from their past - family, friend, mentor, companion, rival)
2. WHAT the question should be (referencing specific events/details from the card)
3. WHY this relationship/question reveals important character background

Format your response as a JSON list like this:
[
  {{
    "relationship_type": "father",
    "relationship_name": "Magnus the Blacksmith", 
    "question": "Do you still remember the techniques I taught you in the forge?",
    "context": "speaking with your father about your training"
  }}
]

Focus on relationships and events specifically mentioned or implied in the character description. If the card is minimal, infer likely relationships based on the character's background."""

        elif temporal_context == "future":
            analysis_prompt = f"""Analyze this character card and generate {num_prompts} personal questions for an established relationship:

{card_info}

Generate questions that assume the User and {char_name} have known each other for years and have a deep connection. Consider:
- Shared experiences they might have had
- How the relationship has evolved
- Deep knowledge they'd have of each other
- Future plans and dreams
- Honest moments and trust
- Intimate moments
- Lewd conversations

Format as JSON list:
[
  {{
    "question": "What's your favorite memory of us together?",
    "intimacy_level": "emotional", 
    "context": "reflecting on your long relationship"
  }}
]"""

        else:  # present
            return []  # Use existing present-context system
        
        # Add retry logic for robustness
        max_retries = 3  # Increased for the retry with modified prompt
        
        for attempt in range(max_retries):
            try:
                # Add small delay to avoid overwhelming vLLM
                if attempt > 0:
                    await asyncio.sleep(1.0)  # Longer delay between retries
                
                # For second attempt, if we got an empty response on first try,
                # modify the prompt to avoid potential safety filter triggers
                current_prompt = analysis_prompt
                
                # Log the prompt being used - extract first few lines for brevity
                prompt_preview = "\n".join(current_prompt.split("\n")[:10])
                if len(current_prompt.split("\n")) > 10:
                    prompt_preview += "\n... [truncated]"
                logger.info(f"📋 Attempt {attempt + 1} prompt ({temporal_context}):\n{prompt_preview}")
                
                # Generate using the inference engine with conservative settings
                logger.debug(f"Attempting intelligent prompt generation for {temporal_context} (attempt {attempt + 1})")
                
                # Use a custom stop-token list that omits "\n\n" so the model can safely emit
                # multi-line JSON without being truncated.
                reduced_stop_tokens = ["<|endoftext|>", "User:", "###", "<|endofcard|>", "<|user|>"]

                response = await self._generate_text(
                    prompt=current_prompt,
                    max_tokens=1000,  # Increased back - was too restrictive
                    temperature=1.0,  # Slightly higher for creativity
                    top_p=0.9,       # Less restrictive sampling
                    character_name=char_name,
                    custom_stop_tokens=reduced_stop_tokens
                )
                
                # ————————————————————————  NEW DIAGNOSTIC LOGGING  ————————————————————————
                preview_len = min(len(response), 400)
                response_preview = response[:preview_len].replace('\n', ' ')[:400]
                logger.info(
                    f"📝 Raw intelligent-prompt response (context={temporal_context}, attempt={attempt + 1}, chars={len(response)}):\n"
                    f"{response_preview}{'…' if len(response) > preview_len else ''}"
                )
                # ——————————————————————————————————————————————————————————————————————
                
                if not response or len(response.strip()) < 10:
                    logger.info(f"⚠️ Empty or too short response from LLM (attempt {attempt + 1})")
                    if attempt == max_retries - 1:  # Last attempt
                        logger.info(f"⚡ LLM generated empty responses after {max_retries} attempts, using static prompts")
                    continue
                
                # Try to parse JSON response
                import json
                try:
                    # Clean the response - sometimes LLMs add extra text
                    response_clean = response.strip()
                    if '```json' in response_clean:
                        response_clean = response_clean.split('```json')[1].split('```')[0]
                    elif '```' in response_clean:
                        response_clean = response_clean.split('```')[1].split('```')[0]
                    
                    # Find JSON array
                    start_idx = response_clean.find('[')
                    end_idx = response_clean.rfind(']') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = response_clean[start_idx:end_idx]
                        parsed_prompts = json.loads(json_str)
                        
                        if isinstance(parsed_prompts, list) and len(parsed_prompts) > 0:
                            logger.info(f"🧠 Generated {len(parsed_prompts)} intelligent {temporal_context} prompts for {char_name} (attempt {attempt + 1})")
                            return parsed_prompts
                        
                except json.JSONDecodeError as e:
                    logger.info(f"🛑 JSON parsing failed (attempt {attempt + 1}): {e}")
                    
                # Fallback: extract questions from raw text
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
                    logger.info(f"🔄 Used fallback parsing for {len(fallback_prompts)} {temporal_context} prompts (attempt {attempt + 1})")
                    return fallback_prompts[:num_prompts]
                    
            except Exception as e:
                error_str = str(e)
                logger.info(f"💥 LLM temporal prompt generation failed (attempt {attempt + 1}): {error_str}")

                if attempt == max_retries - 1:  # Last attempt
                    logger.info(f"⚡ LLM generation failed after {max_retries} attempts, using static prompts")
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
                relationship_name = intelligent_prompt_data.get('relationship_name', 'Someone from your past')
                relationship_type = intelligent_prompt_data.get('relationship_type', 'person')
                context_info = intelligent_prompt_data.get('context', 'reflecting on your past')
                
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
                context_info = intelligent_prompt_data.get('context', 'reflecting on your long relationship')
                intimacy_level = intelligent_prompt_data.get('intimacy_level', 'emotional')
                
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
            
            # Use a lock to prevent concurrent generation for the same cache key
            if cache_key not in self._intelligent_generation_locks:
                self._intelligent_generation_locks[cache_key] = asyncio.Lock()
            
            async with self._intelligent_generation_locks[cache_key]:
                if cache_key not in self._intelligent_prompt_cache:
                    logger.info(f"🧠 Generating intelligent {temporal_context} prompts for {character.get('name', 'character')}")
                    try:
                        intelligent_prompts = await self._generate_intelligent_temporal_prompts(
                            character, temporal_context, num_prompts=6  # Reduced to be more reliable
                        )
                        self._intelligent_prompt_cache[cache_key] = intelligent_prompts
                        
                        if intelligent_prompts:
                            logger.info(f"✅ Successfully generated {len(intelligent_prompts)} intelligent {temporal_context} prompts")
                        else:
                            logger.info(f"🔄 No intelligent prompts generated, will use static prompts")
                        
                        # Small delay to give vLLM time to process
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.debug(f"Intelligent prompt generation error: {e}")
                        self._intelligent_prompt_cache[cache_key] = []  # Cache empty result
                else:
                    intelligent_prompts = self._intelligent_prompt_cache[cache_key]
            
            # Use intelligent prompt if available
            if intelligent_prompts:
                intelligent_prompt_data = random.choice(intelligent_prompts)
                prompt = intelligent_prompt_data.get('question', '')
                
                if temporal_context == "past":
                    relationship_context = intelligent_prompt_data.get('relationship_name', 
                                                                    intelligent_prompt_data.get('relationship_type', 'unknown'))
                
                if prompt:
                    logger.debug(f"🎯 Using intelligent {temporal_context} prompt: {prompt[:60]}...")
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
                    available_relationships.extend([(rel_type, rel) for rel in rel_list])
            
            if available_relationships:
                rel_type, relationship_context = random.choice(available_relationships)
        
        # Get appropriate prompts for this context
        prompt_list = self._get_temporal_prompts(temporal_context, relationship_context)
        prompt = random.choice(prompt_list) if prompt_list else self._build_user_prompt()
        
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
                length_buckets = [("short", 0.40, 200), ("medium", 0.45, 500), ("long", 0.15, 800)]
                names, probs, toks = zip(*[(n, p, t) for n, p, t in length_buckets])
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