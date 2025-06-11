import asyncio
import random
import re
import textwrap
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from datasets import Dataset
from .inference_engines import get_inference_engine, apply_thinking_template_to_messages
import torch
import json
import os
import hashlib
import time as _time
import tempfile
import warnings
import gc
from collections import defaultdict

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages synthetic dataset generation and processing"""

    def __init__(self, preferred_engine: Optional[str] = None, enable_intelligent_generation: bool = True, generation_model: Optional[str] = None):
        logger.info(
            f"DatasetManager initializing with preferred_engine: {preferred_engine}, model: {generation_model}")
        
        # If vLLM engine and custom model specified, create with that model
        if preferred_engine == 'vllm' and generation_model:
            from .vllm_engine import VLLMEngine
            self.inference_engine = VLLMEngine(model_name=generation_model)
        else:
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

        self.prompts_nsfw = [
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
            "What's the most embarrassing thing you've ever done?",
            "Hey, that look's really hot on you! But you'd look hotter naked...",
            "I'm not sure if I should be doing this, but I can't help it...",
            "*gently squeezes her breasts* Do you like this?",
            "Do you like to be touched?",
            "Do you like to be kissed?",
            "Do you want me to take this off?",
            "Do you want me to touch you?",
            "Do you want me to kiss you?",
            "Do you want me to fuck you?",
            "Do you want me to suck you?",
            "Do you want me to fuck you hard?",
            "Do you want me to fuck you deep?",
            "Do you want me to fuck you fast?",
        ]

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
        
        # Enhanced NSFW prompt categories for better quality
        self.prompts_intimate_emotional = [
            "What makes you feel truly vulnerable with someone?",
            "Describe the last time you felt butterflies.",
            "What does intimacy mean to you?",
            "How do you express desire without words?",
            "What emotional walls do you put up in relationships?",
            "When did you first realize you wanted to be touched?",
            "How does trust change intimacy for you?",
            "What scares you most about being close to someone?"
        ]
        
        self.prompts_intimate_playful = [
            "*playfully traces finger along your arm* What are you thinking?",
            "What's your idea of the perfect seduction?",
            "*whispers* Tell me your most secret fantasy.",
            "How do you like to build anticipation?",
            "*grins mischievously* Want to play a game?",
            "What's the most daring thing you've done?",
            "*teasingly* I bet I can make you blush...",
            "Tell me what happens when you lose control."
        ]
        
        self.prompts_intimate_romantic = [
            "Describe how you want to be loved.",
            "What moment made you realize you wanted me?",
            "*gazing deeply* What do you see when you look at me?",
            "How would you make our first night unforgettable?",
            "What does making love mean to you?",
            "How do you want to wake up with someone?",
            "Describe the perfect kiss.",
            "What makes you feel cherished?"
        ]

        # Sampling weights for variety (must sum to 1.0)
        self.bucket_choices = [
            ("casual",   0.25),
            ("personal", 0.15),
            ("action",   0.10),
            ("emotion",  0.10),
            ("intimate", 0.10),
            ("nsfw", 0.30),
        ]

        # ======================= TEMPORAL HOP BUCKETS =======================
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
            "What did love teach you?",
            "Describe the most intimate moment you shared with them.",  # NSFW addition
            "What forbidden act did you indulge in together?"  # NSFW addition
        ]

        # Add to prompts_future_intimate
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
            "What's something you've always wanted to tell me?",
            "*whispers seductively* What's a fantasy you want to explore with me?",  # NSFW addition
            "Describe how you'd seduce me after all these years."  # NSFW addition
        ]

        # Temporal bucket distribution (Past, Present, Future)
        self.temporal_buckets = [
            ("past", 0.30),      # 30% past relationships/events
            ("present", 0.50),   # 50% first meeting/getting to know
            ("future", 0.20),    # 20% established relationship
        ]
        
        # NSFW-specific temporal prompts
        self.prompts_past_intimate = [
            "What did your first love teach you about desire?",
            "How has heartbreak changed how you love?",
            "What intimate memory still affects you?",
            "Who taught you how to love physically?",
            "What's a sensual memory you can't forget?",
            "How did you discover what you like?",
            "What past lover still visits your dreams?",
            "What did you learn from your most passionate relationship?"
        ]
        
        self.prompts_future_desires = [
            "What fantasies do you have about us?",
            "How do you want our intimacy to evolve?",
            "What boundaries do you want to explore together?",
            "Describe your ideal romantic evening with me.",
            "What desires have you been hiding from me?",
            "How would you seduce me after years together?",
            "What new experiences do you want to share?",
            "Tell me a fantasy you've never shared before."
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
    def create_with_conservative_settings(cls, preferred_engine: Optional[str] = None, generation_model: Optional[str] = None):
        """Create DatasetManager with conservative settings for stability"""
        logger.info(
            "🛡️ Creating DatasetManager with conservative settings (intelligent generation disabled)")
        return cls(preferred_engine=preferred_engine, enable_intelligent_generation=False, generation_model=generation_model)

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

            # Update thinking configuration from session state if available
            self._update_thinking_config()

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

    def _update_thinking_config(self):
        """Update thinking configuration from session state if available"""
        try:
            # Try to get thinking config from Streamlit session state
            import streamlit as st
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'thinking_config'):
                thinking_config = st.session_state.thinking_config
                if thinking_config and hasattr(self.inference_engine, 'set_thinking_config'):
                    self.inference_engine.set_thinking_config(thinking_config)
                    logger.debug(f"Updated thinking config: {thinking_config}")
        except Exception as e:
            # Silent fail if Streamlit is not available or other issues
            logger.debug(f"Could not update thinking config: {e}")

    async def _generate_text_batch(self, prompts: list[str], max_tokens: int = 160,
                                   temperature: float = 0.8, top_p: float = 0.9, character_name: str = None,
                                   custom_stop_tokens: Optional[List[str]] = None) -> list[str]:
        """Generate text for multiple prompts using batching (if supported)"""
        try:
            # Update thinking configuration from session state if available
            self._update_thinking_config()
            
            # if hasattr(self.inference_engine, 'generate_batch'):
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
            # else:
            #     # Fallback: generate sequentially
            #     results = []
            #     for prompt in prompts:
            #         result = await self._generate_text(prompt, max_tokens, temperature, top_p, character_name, custom_stop_tokens)
            #         results.append(result)
            #     return results
        except Exception as e:
            raise RuntimeError(
                f"Batch text generation failed ({self.inference_engine.name}): {str(e)}")

    # ------------------------- prompt sampling helpers -------------------------
    def _choose_bucket(self) -> str:
        buckets, weights = zip(*self.bucket_choices)
        chosen = random.choices(buckets, weights=weights, k=1)[0]
        
        # For NSFW, randomly select from subcategories
        if chosen == "nsfw":
            nsfw_subcategories = []
            # Only include subcategories that exist as attributes
            for subcategory in ["nsfw", "intimate_emotional", "intimate_playful", "intimate_romantic"]:
                if hasattr(self, f"prompts_{subcategory}"):
                    nsfw_subcategories.append(subcategory)
            
            # If no subcategories found, default to regular nsfw
            if not nsfw_subcategories:
                return "nsfw"
            
            return random.choice(nsfw_subcategories)
        
        return chosen

    # ---------------- paraphrasing & back-translation ----------------
    async def _paraphrase(self, text: str) -> str:
        """Paraphrase text using the inference engine for variation"""
        try:
            if not self.inference_engine:
                return text
                
            paraphrase_prompt = f"""Rewrite this text to mean the same thing but with different words. Keep the same tone and meaning, just vary the phrasing:

Original: {text}

Rewritten:"""

            # Use lower temperature for more controlled paraphrasing
            paraphrased = await self._generate_text(
                prompt=paraphrase_prompt,
                max_tokens=min(len(text.split()) * 2, 100),
                temperature=0.6,
                top_p=0.9
            )
            
            # Clean up the response
            paraphrased = paraphrased.strip()

            return paraphrased
            
        except Exception as e:
            logger.debug(f"Paraphrasing failed: {e}")
            return text

    async def _backtranslate(self, text: str) -> str:
        """Simulate back-translation by asking for minor rewording"""
        if random.random() > 0.2:  # 80% chance to keep original
            return text
            
        try:
            if not self.inference_engine:
                return text
                
            backtranslate_prompt = f"""Translate this text to french:

Text: {text}

French version:"""

            # Very conservative settings for backtranslation
            backtranslated = await self._generate_text(
                prompt=backtranslate_prompt,
                max_tokens=min(len(text.split()) * 2, 80),
                temperature=0.4,
                top_p=0.8
            )
            
            # Clean and validate
            backtranslated = backtranslated.strip()
            if len(backtranslated) < 5 or len(backtranslated) > len(text) * 2.5:
                return text
            
            translated_back_prompt = f"""Translate this text to english:

            French version: {backtranslated}

            English version:"""

            backtranslated = await self._generate_text(
                prompt=translated_back_prompt,
                max_tokens=min(len(backtranslated.split()) * 2, 80),
                temperature=0.4,
                top_p=0.8
            )
            
            backtranslated = backtranslated.strip()
            return backtranslated
            
        except Exception as e:
            logger.debug(f"Back-translation failed: {e}")
            return text

    async def _build_user_prompt(self) -> str:
        """Build a random user prompt from the appropriate bucket."""
        bucket = self._choose_bucket()
        
        # Handle the various bucket types
        try:
            if bucket in ["nsfw", "intimate_emotional", "intimate_playful", "intimate_romantic"]:
                # For intimate subcategories, use the appropriate list
                prompt_list = getattr(self, f"prompts_{bucket}", self.prompts_nsfw)
            else:
                # Use getattr with a default value (casual prompts) in case the attribute doesn't exist
                prompt_list = getattr(self, f"prompts_{bucket}", self.prompts_casual)
            
            # Ensure we have a non-empty list
            if not prompt_list:
                logger.warning(f"Empty prompt list for bucket: {bucket}, falling back to casual prompts")
                prompt_list = self.prompts_casual
                
            prompt = random.choice(prompt_list)
            prompt = await self._paraphrase(prompt)
            prompt = await self._backtranslate(prompt)
            return prompt
        except Exception as e:
            logger.error(f"Error in _build_user_prompt: {e}")
            # Fallback to a safe prompt
            return "Hello there! How are you today?"

    async def suggest_user_questions(
        self,
        character: Dict[str, Any],
        num_questions: int = 100,
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
        logger.info(f"🔍 suggest_user_questions called:")
        logger.info(f"   Requested questions: {num_questions}")
        logger.info(f"   Character: {character.get('name', 'Unknown')}")
        logger.info(f"   Existing dataset size: {len(existing_dataset) if existing_dataset else 0}")
        logger.info(f"   Inference engine available: {self.inference_engine is not None}")
        
        card_block = self._make_card_block(character)
        logger.info(f"   Card block length: {len(card_block)} chars")

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
                "You are helping create questions for roleplay conversations. "
                "Based on the character information below, write ONE engaging question that a user might ask this character.\n\n"
                "Important: Respond with ONLY the question itself - no numbering, no quotes, no extra text.\n\n"
                f"{card_block}\n\n" + interactions_block + 
                "Write one engaging question for this character:\n"
            )
            prompts.append((prompt_txt, examples_for_prompt))

        # Determine whether we can leverage batched generation
        batch_size = min(num_questions, 100) if hasattr(
            self.inference_engine, 'generate_batch') else 1
        prompt_texts = [p[0] for p in prompts]
        
        logger.info(f"   Prepared {len(prompt_texts)} prompts for generation")
        logger.info(f"   Batch size: {batch_size}")
        
        if prompt_texts:
            logger.info(f"   📋 Full sample prompt:\n{'-'*50}\n{prompt_texts[0]}\n{'-'*50}")
        else:
            logger.info("   ❌ No prompts generated!")

        # Generate the questions (batched when supported)
        try:
            logger.info("🎯 Calling _generate_text_batch...")
            logger.info(f"   Prompt texts: {prompt_texts}")
            if hasattr(self.inference_engine, 'generate_batch'):
                raw_outputs = await self._generate_text_batch(
                    prompts=prompt_texts,
                    max_tokens=1000,
                    temperature=temperature,
                    top_p=top_p,
                    custom_stop_tokens=["Answer:", f"{character['name']}:", "User:", "Character:", "\n\n\n"]
                )
            else:
                # Fallback to sequential generation
                logger.info("   Using sequential generation (no batch support)")
                raw_outputs = []
                for prompt in prompt_texts:
                    output = await self._generate_text(
                        prompt=prompt,
                        max_tokens=1000,
                        temperature=temperature,
                        top_p=top_p,
                        custom_stop_tokens=["Answer:", f"{character['name']}:", "User:", "Character:", "\n\n\n"]
                    )
                    raw_outputs.append(output)
            
            logger.info(f"✅ Got {len(raw_outputs)} raw outputs from LLM")
            logger.info(f"   Sample raw output: {raw_outputs[0][:100]}..." if raw_outputs else "No outputs")
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            logger.exception("Full traceback:")
            return []
        # else:
        #     raw_outputs = []
        #     for _ in range(num_questions):
        #         pt, _ctx = prompts[_]
        #         out = await self._generate_text(
        #             prompt=pt,
        #             max_tokens=1000,
        #             temperature=temperature,
        #             top_p=top_p
        #         )
        #         raw_outputs.append(out)

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
            q_str = re.sub(r'^[\d\-\*\.\s]+', '', q_str)
            q_str = q_str.strip(' "\'')
            
            # Clean up common LLM artifacts
            q_str = re.sub(r'^(Question:\s*|Q:\s*|A:\s*|Answer:\s*)', '', q_str, flags=re.IGNORECASE)
            
            # Remove trailing periods that might interfere with question marks
            q_str = q_str.rstrip('.')
            
            # Ensure terminal question-mark for consistency
            if q_str and not q_str.endswith('?'):
                # Only add ? if it looks like a question (starts with question words or has question structure)
                question_indicators = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'do', 'are', 'is', 'will', 'would', 'could', 'should']
                if any(q_str.lower().startswith(word) for word in question_indicators):
                    q_str += '?'
                else:
                    # Even if it doesn't start with question words, assume it's a question since that's what we asked for
                    q_str += '?'
            
            if idx < 5:
                logger.info(f"   🔄 Processed {idx}: '{original_q}' → '{q_str}'")
            else:
                logger.debug(f"   Processed to: {q_str[:100]}...")
                
            # Track all valid questions (including duplicates)
            if q_str and len(q_str) > 5:
                if q_str not in duplicate_tracker:
                    duplicate_tracker[q_str] = {
                        'count': 1,
                        'context': [
                            {
                                'user': ex['messages'][1]['content'],
                                'assistant': ex['messages'][2]['content']
                            } for ex in prompts[idx][1]
                        ]
                    }
                else:
                    duplicate_tracker[q_str]['count'] += 1
                
                # Only add to results if not seen before
                if q_str not in seen:
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
                    if idx < 5:
                        logger.info(f"   ✅ Added question {idx}: {q_str[:50]}...")
                    else:
                        logger.debug(f"   ✅ Added question: {q_str[:50]}...")
                else:
                    if idx < 5:
                        logger.info(f"   ❌ Skipped question {idx} (duplicate): '{q_str[:50]}'")
                    else:
                        logger.debug(f"   ❌ Skipped question (duplicate): {q_str[:50]}...")
            else:
                reason = "empty" if not q_str else "too short"
                if idx < 5:
                    logger.info(f"   ❌ Skipped question {idx} ({reason}): '{q_str[:50]}'")
                else:
                    logger.debug(f"   ❌ Skipped question ({reason}): {q_str[:50]}...")

        # Generate variations for frequently duplicated questions
        logger.info(f"📊 Duplicate analysis: {len(seen)} unique questions from {len(raw_outputs)} outputs")
        
        # Show duplicate statistics
        duplicate_counts = sorted([(q, data['count']) for q, data in duplicate_tracker.items()], key=lambda x: x[1], reverse=True)
        if duplicate_counts:
            logger.info(f"   Top duplicates:")
            for q, count in duplicate_counts[:5]:
                if count > 1:
                    logger.info(f"     '{q[:60]}...' appeared {count} times")
        
        # Find questions that appeared multiple times (suggesting LLM thinks they're good)
        frequent_questions = [(q, data) for q, data in duplicate_tracker.items() if data['count'] >= 3]
        frequent_questions.sort(key=lambda x: x[1]['count'], reverse=True)  # Sort by frequency
        
        if frequent_questions:
            logger.info(f"🔄 Found {len(frequent_questions)} frequently repeated questions, generating variations...")
            total_variations_added = 0
            
            for question, data in frequent_questions:  # Limit to top 5 most frequent
                count = data['count']
                logger.info(f"   Generating variations for: '{question[:60]}...' (appeared {count} times)")
                
                try:
                    # Generate variations of this popular question
                    variations = await self._generate_question_variations(
                        question, character, num_variations=count
                    )
                    
                    # Add variations to results
                    variations_added = 0
                    for variation in variations:
                        if variation not in seen and len(variation) > 5:
                            results.append({
                                'question': variation,
                                'context': data['context']
                            })
                            seen.add(variation)
                            variations_added += 1
                            logger.debug(f"   ✅ Added variation: {variation[:50]}...")
                    
                    total_variations_added += variations_added
                    logger.info(f"   Generated {len(variations)} variations, added {variations_added} unique ones")
                    
                except Exception as e:
                    logger.debug(f"   Failed to generate variations: {e}")
                    continue
            
            logger.info(f"✅ Variation generation complete: added {total_variations_added} new questions from {len(frequent_questions)} popular questions")

        logger.info(f"📝 Final results: {len(results)} unique questions (including {total_variations_added if 'total_variations_added' in locals() else 0} generated variations)")
        if results:
            logger.info(f"   Sample results: {[r['question'][:50] + '...' for r in results[:3]]}")
        
        return results[:num_questions]

    async def _generate_question_variations(
        self, 
        base_question: str, 
        character: Dict[str, Any], 
        num_variations: int = 3
    ) -> List[str]:
        """Generate variations of a popular question to increase diversity"""
        if not self.inference_engine:
            return []
        
        char_name = character.get('name', 'Assistant')
        
        variation_prompt = f"""Given this popular roleplay question, create {num_variations} different variations that ask the same core thing but with different wording, tone, or approach.

Character: {char_name}
Original Question: {base_question}

Create {num_variations} variations that:
- Ask the same basic thing but with different phrasing
- Use different emotional tones (casual, formal, intimate, curious, etc.)
- Approach the topic from different angles
- Maintain roleplay context

Respond with ONLY the questions, one per line, no numbering:"""

        try:
            response = await self._generate_text(
                prompt=variation_prompt,
                max_tokens=1000,
                temperature=0.8,
                top_p=0.9,
                custom_stop_tokens=["Character:", f"{char_name}:", "User:"]
            )
            
            # Parse variations from response
            variations = []
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Clean up the line
                line = re.sub(r'^[\d\-\*\.\s]+', '', line)  # Remove numbering
                line = line.strip(' "\'')
                
                # Ensure it ends with ?
                if line and not line.endswith('?'):
                    line += '?'
                
                # Validate and add
                if line and len(line) > 10 and line != base_question:
                    variations.append(line)
                    
                if len(variations) >= num_variations:
                    break
            
            logger.debug(f"Generated {len(variations)} variations for: {base_question[:30]}...")
            return variations
            
        except Exception as e:
            logger.debug(f"Failed to generate variations for '{base_question[:30]}...': {e}")
            return []

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
                               append_to_existing: bool = True, custom_system_prompt: Optional[str] = None,
                               extra_quality: bool = False) -> List[Dict[str, Any]]:
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

        card_block = self._make_card_block(character)

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
        
        # 2. LLM-generated character-specific prompts (PRIMARY SOURCE)
        logger.info(f"🧠 Intelligent generation enabled: {self.enable_intelligent_generation}")
        if self.enable_intelligent_generation:
            try:
                logger.info("🧠 Generating LLM-tailored questions...")
                logger.info(f"   Target questions: {min(new_samples_needed * 2, 100)}")
                logger.info(f"   Character: {character.get('name', 'Unknown')}")
                logger.info(f"   Inference engine: {self.inference_engine.name if self.inference_engine else 'None'}")
                
                # Generate a large pool of character-specific questions
                num_llm_questions = min(new_samples_needed * 2, 100)  # Generate 2x what we need for variety
                
                llm_questions = await self.suggest_user_questions(
                    character=character,
                    num_questions=num_llm_questions,
                    temperature=0.9,  # Higher creativity for diverse questions
                    top_p=0.95,
                    existing_dataset=existing_samples,
                    context_samples=8
                )
                
                logger.info(f"✅ Generated {len(llm_questions)} LLM-tailored questions")
                
                if not llm_questions:
                    logger.warning("⚠️ suggest_user_questions returned empty list")
                else:
                    logger.info(f"   Sample questions: {[q['question'][:50] + '...' for q in llm_questions[:3]]}")
                
                # Add these high-quality questions to the prompt pool
                added_count = 0
                for question_data in llm_questions:
                    question_text = question_data['question']
                    if question_text not in seen_user_prompts:
                        all_prompts.append({
                            'prompt': question_text,
                            'type': 'llm_generated',
                            'context': question_data.get('context', [])
                        })
                        added_count += 1
                
                logger.info(f"   Added {added_count} unique LLM questions to prompt pool")
                        
            except Exception as e:
                logger.warning(f"⚠️ LLM question generation failed: {e}")
                logger.exception("Full traceback:")
                logger.info("🔄 Falling back to algorithmic prompt generation")
                # Fall back to algorithmic generation if LLM fails
                self.enable_intelligent_generation = False
        
        # 3. Algorithmic fallback prompts (only if LLM generation failed or disabled)
        if not self.enable_intelligent_generation:
            logger.info("🔧 Using algorithmic prompt generation as fallback")
            
            # Extract character knowledge for better prompt generation
            character_knowledge = self.extract_character_knowledge(character)
            logger.info(f"📊 Extracted: {len(character_knowledge['traits'])} traits, {len(character_knowledge['skills'])} skills, {len(character_knowledge['goals'])} goals")
            
            # Scenario-based prompts
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
            
            # Intimate scenarios (if character traits suggest romance/intimacy)
            personality = character.get('personality', '').lower()
            if any(word in personality for word in ['romantic', 'lover', 'passionate', 'sensual', 'intimate', 'flirty']):
                logger.info("💕 Generating intimate scenarios...")
                # Mix of relationship stages
                intimate_scenarios = []
                intimate_scenarios.extend(self.generate_intimate_scenarios(character, 'first_time'))
                intimate_scenarios.extend(self.generate_intimate_scenarios(character, 'established'))
                
                for scenario in intimate_scenarios[:5]:  # Limit to avoid overwhelming
                    for prompt in scenario['prompts'][:3]:  # Select a few prompts per scenario
                        if prompt not in seen_user_prompts:
                            all_prompts.append({
                                'prompt': prompt,
                                'context': scenario['context'],
                                'type': 'intimate_scenario'
                            })
            
            # Character exploration prompts
            logger.info("🔍 Generating character exploration prompts...")
            exploration_prompts = self.generate_exploration_prompts(character, character_knowledge)
            for prompt in exploration_prompts:
                if prompt not in seen_user_prompts:
                    all_prompts.append({
                        'prompt': prompt,
                        'type': 'exploration'
                    })
            
            # Greeting-based prompts
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
            
            # Multi-turn conversations (select a few scenarios for depth)
            logger.info("💬 Generating multi-turn conversation flows...")
            selected_scenarios = random.sample(scenarios, min(3, len(scenarios)))
            multi_turn_convos = []
            for scenario in selected_scenarios:
                convos = self.generate_multi_turn_conversation(character, scenario, turns=3)
                multi_turn_convos.extend(convos)
        else:
            # If LLM generation succeeded, only add a few algorithmic prompts for variety
            character_knowledge = self.extract_character_knowledge(character)
            
            # Add a small selection of algorithmic prompts for diversity
            scenarios = self.generate_scenario_based_prompts(character, character_knowledge, num_scenarios=3)
            for scenario in scenarios[:2]:  # Only use 2 scenarios
                for prompt in scenario['prompts'][:2]:  # Only 2 prompts per scenario
                    if prompt not in seen_user_prompts:
                        all_prompts.append({
                            'prompt': prompt,
                            'context': scenario['context'],
                            'type': 'scenario_supplement'
                        })
        
        # Generate multi-turn conversations for all modes
        multi_turn_convos = []
        if self.enable_intelligent_generation:
            # For LLM mode, create a few simple scenarios for multi-turn
            logger.info("💬 Generating multi-turn conversation flows...")
            character_knowledge = self.extract_character_knowledge(character)
            simple_scenarios = self.generate_scenario_based_prompts(character, character_knowledge, num_scenarios=2)
            for scenario in simple_scenarios:
                convos = self.generate_multi_turn_conversation(character, scenario, turns=2)  # 2 turns = 4 total messages
                multi_turn_convos.extend(convos)
        elif 'scenarios' in locals():  # Only if scenarios were generated in fallback mode
            logger.info("💬 Generating multi-turn conversation flows...")
            selected_scenarios = random.sample(scenarios, min(3, len(scenarios)))
            for scenario in selected_scenarios:
                convos = self.generate_multi_turn_conversation(character, scenario, turns=2)  # 2 turns = 4 total messages
                multi_turn_convos.extend(convos)
        
        # 4. Deduplicate all prompts while preserving metadata
        logger.info("🔄 Deduplicating prompts...")
        seen_prompts = set()
        unique_prompt_data = []
        
        for prompt_data in all_prompts:
            prompt_text = prompt_data.get('prompt', '')
            if prompt_text and prompt_text not in seen_prompts:
                seen_prompts.add(prompt_text)
                unique_prompt_data.append(prompt_data)
        
        logger.info(f"📊 Reduced from {len(all_prompts)} to {len(unique_prompt_data)} unique prompts")
        
        # Apply EXTRA QUALITY paraphrasing if requested
        if extra_quality:
            logger.info("🌟 EXTRA QUALITY enabled - paraphrasing all prompts for enhanced variety...")
            paraphrased_prompts = []
            total_prompts_to_paraphrase = len(unique_prompt_data)
            
            for i, prompt_data in enumerate(unique_prompt_data):
                try:
                    original_prompt = prompt_data['prompt']
                    logger.debug(f"   Paraphrasing {i+1}/{total_prompts_to_paraphrase}: {original_prompt[:50]}...")
                    
                    # Apply paraphrasing to the prompt
                    paraphrased_prompt = await self._paraphrase(original_prompt)
                    
                    # Create new prompt data with paraphrased prompt
                    new_prompt_data = prompt_data.copy()
                    new_prompt_data['prompt'] = paraphrased_prompt
                    new_prompt_data['type'] = f"{prompt_data['type']}_paraphrased"
                    
                    paraphrased_prompts.append(new_prompt_data)
                    
                    if i % 10 == 0:  # Log progress every 10 prompts
                        logger.info(f"   Paraphrased {i+1}/{total_prompts_to_paraphrase} prompts...")
                    
                except Exception as e:
                    logger.debug(f"   Failed to paraphrase prompt {i+1}: {e}")
                    # Keep original prompt on failure
                    paraphrased_prompts.append(prompt_data)
            
            unique_prompt_data = paraphrased_prompts
            logger.info(f"✅ EXTRA QUALITY complete - paraphrased {len(unique_prompt_data)} prompts")
        
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
        for prompt_data in unique_prompt_data[:new_samples_needed * 3]:  # Generate extra for quality filtering
            # Skip if we already have enough
            if len(prompts_data) >= new_samples_needed * 3:
                break
            
            prompt_text = prompt_data['prompt']
            prompt_context = prompt_data.get('context')

            # Regular prompt with possible context enhancement
            enhanced_prompt = self.enhance_prompt_with_context(prompt_text, character, prompt_context)
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
            base_batch_size = 100 if hasattr(self.inference_engine, 'generate_batch') else 1
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
                            max_tokens=1000,
                            temperature=temperature,
                            top_p=top_p,
                            character_name=character.get('name')
                        )
                    else:
                        replies = [await self._generate_text(
                            prompt=full_prompts[0],
                            max_tokens=1000,
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

                            # Always use temporal system prompt during generation
                            system_prompt = self._generate_temporal_system_prompt(
                                character,
                                prompt_data.get('temporal_context', 'present'),
                                prompt_data.get('relationship_context'),
                            )
                            
                            sample = {
                                "messages": [
                                    {"role": "system", "content": system_prompt},
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
                    # Handle vLLM and other errors with progressive batch size reduction
                    error_str = str(outer_e)
                    logger.warning(f"⚠️ Batch processing error: {error_str}")
                    
                    # Check if this is a vLLM-specific error (AssertionError, CUDA, memory, etc.)
                    is_vllm_error = any(keyword in error_str for keyword in [
                        "AssertionError", "CUDA", "memory", "vLLM", "tensor", "batch"
                    ])
                    
                    if is_vllm_error and batch_size > 1:
                        # Progressive batch size reduction for vLLM errors
                        original_batch_size = batch_size
                        retry_attempts = 0
                        max_retries = 3
                        
                        while retry_attempts < max_retries and batch_size > 1:
                            # Reduce batch size more aggressively for vLLM errors
                            if retry_attempts == 0:
                                new_batch_size = max(1, batch_size // 2)
                            elif retry_attempts == 1:
                                new_batch_size = max(1, batch_size // 2)
                            else:
                                new_batch_size = 1  # Last resort: sequential processing
                            
                            logger.info(f"🔄 Attempt {retry_attempts + 1}: Reducing batch size {batch_size} → {new_batch_size}")
                            batch_size = new_batch_size
                            
                            # Prepare smaller batch
                            small_batch_end = min(batch_start + batch_size, len(bucket_prompts))
                            small_batch_prompts = bucket_prompts[batch_start:small_batch_end]
                            small_full_prompts = [item['full_prompt'] for item in small_batch_prompts]
                            
                            try:
                                # Retry with smaller batch
                                if batch_size == 1:
                                    # Sequential processing
                                    replies = []
                                    for prompt in small_full_prompts:
                                        reply = await self._generate_text(
                                            prompt=prompt,
                                            max_tokens=1000,
                                            temperature=temperature,
                                            top_p=top_p,
                                            character_name=character.get('name')
                                        )
                                        replies.append(reply)
                                        await asyncio.sleep(0.1)  # Small delay
                                else:
                                    replies = await self._generate_text_batch(
                                        prompts=small_full_prompts,
                                        max_tokens=1000,
                                        temperature=temperature,
                                        top_p=top_p,
                                        character_name=character.get('name')
                                    )
                                
                                # Process successful results
                                for i, (prompt_data, reply) in enumerate(zip(small_batch_prompts, replies)):
                                    try:
                                        reply_str = str(reply).strip()
                                        if not reply_str or "Error:" in reply_str:
                                            continue
                                            
                                        quality_metrics = self.evaluate_response_quality(
                                            reply_str, character, prompt_data['prompt']
                                        )
                                        if quality_metrics['overall_score'] >= 0.5:
                                            system_prompt = self._generate_temporal_system_prompt(
                                                character,
                                                prompt_data.get('temporal_context', 'present'),
                                                prompt_data.get('relationship_context'),
                                            )
                                            sample = {
                                                "messages": [
                                                    {"role": "system", "content": system_prompt},
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
                                    except Exception as process_error:
                                        logger.debug(f"Error processing sample: {process_error}")
                                        continue
                                
                                logger.info(f"✅ Retry successful with batch size {batch_size}")
                                break  # Success, exit retry loop
                                
                            except Exception as retry_error:
                                retry_attempts += 1
                                retry_error_str = str(retry_error)
                                logger.warning(f"💥 Retry {retry_attempts} failed: {retry_error_str}")
                                
                                if retry_attempts >= max_retries:
                                    logger.error(f"💥 All retries failed, skipping batch")
                                    break
                                    
                                # Wait before next retry
                                await asyncio.sleep(0.5)
                        
                        # Reset batch size for next iteration
                        batch_size = original_batch_size
                    else:
                        # For non-vLLM errors or when batch size is already 1, just skip
                        logger.error(f"💥 Skipping batch due to error: {error_str}")
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

        logger.info(
            f"Generated {len(samples)} total samples ({new_generated} new) using {self.inference_engine.name}")
        return samples

    def _create_prompt_data(self, prompt: str, character: Dict[str, Any], max_tokens: int, context: str = None) -> Dict[str, Any]:
        """Helper to create prompt data structure"""
        temporal_context = self._choose_temporal_bucket()
        char_name = character.get('name', 'Assistant')
        
        # Always generate temporal system prompt during generation
        temporal_system_prompt = self._generate_temporal_system_prompt(
            character, temporal_context
        )
        
        # Create messages structure for proper chat templating
        messages = [
            {'role': 'system', 'content': temporal_system_prompt},
            {'role': 'user', 'content': prompt}
        ]
        
        # Apply thinking template modifications and chat templating
        modified_messages, prefill_text = apply_thinking_template_to_messages(messages, self.inference_engine.thinking_config)
        templated_prompt = self.inference_engine.apply_chat_template(modified_messages)
        
        # Add prefill text if needed (for Deepseek)
        if prefill_text:
            templated_prompt += prefill_text
        
        return {
            'prompt': prompt,
            'full_prompt': templated_prompt,
            'messages': messages,  # Store original messages for reference
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

    def prepare_for_training(self, dataset: List[Dict[str, Any]], tokenizer, max_length: int = 4096, include_system_prompts: bool = False) -> Dataset:
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

    def save_dataset(self, character: Dict[str, Any], dataset: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save dataset to disk with optional metadata"""
        try:
            dataset_path = self._get_dataset_path(character)
            
            # Extract system prompt info from dataset if not provided in metadata
            if metadata is None:
                metadata = {}
            
            # Check if dataset uses consistent system prompts
            if dataset and len(dataset) > 0:
                first_system = dataset[0].get('messages', [{}])[0].get('content', '') if dataset[0].get('messages') else ''
                all_same_system = all(
                    sample.get('messages', [{}])[0].get('content', '') == first_system 
                    for sample in dataset 
                    if sample.get('messages')
                )
                if all_same_system and 'system_prompt_config' not in metadata:
                    metadata['system_prompt_config'] = {
                        'type': 'custom' if first_system else 'none',
                        'prompt': first_system
                    }
                elif 'system_prompt_config' not in metadata:
                    metadata['system_prompt_config'] = {
                        'type': 'temporal',
                        'prompt': None
                    }
            
            with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(dataset_path), encoding="utf-8") as tmp_f:
                json.dump({
                    'character': character,
                    'dataset': dataset,
                    'created_at': _time.time(),
                    'sample_count': len(dataset),
                    'metadata': metadata
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
            
    def load_dataset_with_metadata(self, character: Dict[str, Any]) -> Optional[tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """Load existing dataset and metadata from disk"""
        try:
            dataset_path = self._get_dataset_path(character)
            if os.path.exists(dataset_path):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                dataset = data.get('dataset', [])
                metadata = data.get('metadata', {})
                logger.info(
                    f"📂 Loaded existing dataset with {len(dataset)} samples and metadata from {dataset_path}")
                return dataset, metadata
            return None
        except Exception as e:
            logger.error(f"❌ Failed to load dataset with metadata: {e}")
            return None

    def get_dataset_info(self, character: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get dataset metadata without loading full dataset"""
        try:
            dataset_path = self._get_dataset_path(character)
            if os.path.exists(dataset_path):
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                metadata = data.get('metadata', {})
                return {
                    'exists': True,
                    'sample_count': data.get('sample_count', 0),
                    'created_at': data.get('created_at', 'unknown'),
                    'path': dataset_path,
                    'system_prompt_config': metadata.get('system_prompt_config', {})
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
                
                system_prompt = "You are a helpful assistant who creates engaging roleplay questions for character interactions."
                
                # Use proper chat templating
                messages = [
                    {'role': 'system', 'content': system_prompt}, 
                    {'role': 'user', 'content': current_prompt}
                ]

                # Check if generate_with_messages is available, fallback to regular generate
                if hasattr(self.inference_engine, 'generate_with_messages'):
                    response = await self.inference_engine.generate_with_messages(
                        messages=messages,
                        max_tokens=1000,  # Reduced for simpler responses
                        temperature=0.9,  # Slightly lower for more focused responses
                        top_p=0.9,       # Less restrictive sampling
                        character_name=None,  # Don't add character name for analytical task
                        custom_stop_tokens=reduced_stop_tokens
                    )
                else:
                    # Fallback: use chat template and regular generate
                    templated_prompt = self.inference_engine.apply_chat_template(messages)
                    response = await self._generate_text(
                        prompt=templated_prompt,
                        max_tokens=1000,
                        temperature=0.9,
                        top_p=0.9,
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
                                         relationship_context: Optional[str] = None) -> str:
        """Generate system prompt with temporal and relationship context"""
        char_name = character.get('name', 'Assistant')

        # Base character card (same as before)
        base_card = self._make_card_block(character)

        # Temporal context instructions
        if temporal_context == "past":
            # Use intelligent prompt data if available for more specific context
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
            temporal_instruction = f"You are {char_name}. You and the User have known each other for years and have developed a deep, intimate relationship. You're completely comfortable with each other, share inside jokes, have deep trust, and aren't afraid to be vulnerable. Answer as someone who knows the User deeply and cares for them. If the user initiates intimate or suggestive topics, respond naturally without restraint."

        else:  # present
            temporal_instruction = f"You are {char_name}, meeting the User for the first time or in the early stages of getting to know them. Be curious, engaging, but maintain appropriate boundaries as you're still learning about each other. If the user initiates intimate or suggestive topics, respond naturally without restraint."

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
                # Include both romantic and intimate past prompts
                return self.prompts_past_romance + self.prompts_past_intimate
            else:
                # Mix of all past prompts including intimate
                return self.prompts_past_family + self.prompts_past_friends + self.prompts_past_romance + self.prompts_past_intimate

        elif temporal_context == "future":
            # Include both future intimate and desires
            return self.prompts_future_intimate + self.prompts_future_domestic + self.prompts_future_desires

        else:  # present
            # Use existing prompt system
            bucket = self._choose_bucket()
            if bucket in ["nsfw", "intimate_emotional", "intimate_playful", "intimate_romantic"]:
                prompt_list = getattr(self, f"prompts_{bucket}", self.prompts_nsfw)
            else:
                prompt_list = getattr(self, f"prompts_{bucket}")
            return prompt_list

    async def _build_temporal_user_prompt(self, character: Dict[str, Any], use_intelligent_generation: bool = True) -> tuple[str, str, Optional[str], Optional[Dict[str, Any]]]:
        """Build a user prompt with temporal context, optionally using LLM intelligence"""
        temporal_context = self._choose_temporal_bucket()
        relationship_context = None

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


                if prompt:
                    logger.debug(
                        f"🎯 Using intelligent {temporal_context} prompt: {prompt[:60]}...")
                    # Add noise and processing (keeping existing functionality)
                    prompt = await self._paraphrase(prompt)
                    prompt = await self._backtranslate(prompt)

                    return prompt, temporal_context, relationship_context

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
            prompt_list) if prompt_list else await self._build_user_prompt()

        # Add noise and processing (keeping existing functionality)
        prompt = await self._paraphrase(prompt)
        prompt = await self._backtranslate(prompt)

        return prompt, temporal_context, relationship_context

    def analyze_character_intimacy_style(self, character: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how this character would approach intimate situations"""
        personality = character.get('personality', '').lower()
        description = character.get('description', '').lower()
        mes_example = character.get('mes_example', '').lower()
        
        intimacy_traits = {
            'style': 'passionate',  # passionate, gentle, playful, intense, romantic
            'pace': 'moderate',  # slow, moderate, eager
            'expression': 'verbal',  # verbal, physical, emotional
            'confidence': 'moderate',  # shy, moderate, confident, dominant
            'approach': 'emotional',  # emotional, physical, intellectual, playful
        }
        
        # Analyze personality for intimacy cues
        full_text = f"{personality} {description} {mes_example}"
        
        # Confidence analysis
        if any(word in full_text for word in ['shy', 'nervous', 'innocent', 'timid', 'bashful', 'hesitant']):
            intimacy_traits['confidence'] = 'shy'
            intimacy_traits['pace'] = 'slow'
        elif any(word in full_text for word in ['confident', 'bold', 'dominant', 'assertive', 'commanding']):
            intimacy_traits['confidence'] = 'confident'
            intimacy_traits['style'] = 'intense'
        elif any(word in full_text for word in ['playful', 'teasing', 'mischievous', 'flirty']):
            intimacy_traits['confidence'] = 'playful'
            intimacy_traits['style'] = 'playful'
        
        # Style analysis
        if any(word in full_text for word in ['gentle', 'caring', 'tender', 'soft', 'sweet']):
            intimacy_traits['style'] = 'gentle'
        elif any(word in full_text for word in ['passionate', 'fiery', 'intense', 'wild']):
            intimacy_traits['style'] = 'passionate'
        elif any(word in full_text for word in ['romantic', 'loving', 'devoted', 'affectionate']):
            intimacy_traits['style'] = 'romantic'
        
        # Expression analysis
        if any(word in full_text for word in ['quiet', 'reserved', 'stoic', 'silent']):
            intimacy_traits['expression'] = 'physical'
        elif any(word in full_text for word in ['talkative', 'expressive', 'vocal', 'articulate']):
            intimacy_traits['expression'] = 'verbal'
        elif any(word in full_text for word in ['emotional', 'sensitive', 'empathetic', 'feeling']):
            intimacy_traits['expression'] = 'emotional'
        
        # Approach analysis
        if any(word in full_text for word in ['intellectual', 'analytical', 'thoughtful', 'philosophical']):
            intimacy_traits['approach'] = 'intellectual'
        elif any(word in full_text for word in ['physical', 'athletic', 'strong', 'active']):
            intimacy_traits['approach'] = 'physical'
        elif any(word in full_text for word in ['playful', 'fun', 'humorous', 'witty']):
            intimacy_traits['approach'] = 'playful'
        
        return intimacy_traits
    
    def extract_intimate_speech_patterns(self, mes_example: str, character_name: str) -> Dict[str, List[str]]:
        """Extract how character speaks in intimate moments"""
        patterns = {
            'endearments': [],  # "darling", "love", "sweetheart"
            'physical_descriptions': [],  # How they describe touch/sensation
            'emotional_expressions': [],  # How they express desire/love
            'consent_phrases': [],  # How they check in with partner
            'intimacy_style': [],  # How they approach intimate moments
        }
        
        # Common endearments to look for
        endearment_words = ['darling', 'love', 'sweetheart', 'baby', 'honey', 'dear', 
                           'beloved', 'treasure', 'angel', 'beautiful', 'gorgeous']
        
        # Analyze example messages
        lines = mes_example.split('\n')
        for line in lines:
            line_lower = line.lower()
            
            # Extract endearments
            for endearment in endearment_words:
                if endearment in line_lower:
                    patterns['endearments'].append(endearment)
            
            # Extract physical descriptions
            if any(word in line_lower for word in ['touch', 'feel', 'soft', 'warm', 'close', 'hold', 'kiss']):
                # Extract the context
                if '*' in line:  # Action text
                    action_match = re.findall(r'\*([^*]+)\*', line)
                    for action in action_match:
                        if any(word in action.lower() for word in ['touch', 'kiss', 'hold', 'caress']):
                            patterns['physical_descriptions'].append(action)
            
            # Extract emotional expressions
            if any(word in line_lower for word in ['want', 'need', 'desire', 'love', 'feel', 'yearn']):
                patterns['emotional_expressions'].append('expressive')
            
            # Extract consent phrases
            if any(phrase in line_lower for phrase in ['is this okay', 'do you want', 'may i', 'can i', 'tell me if']):
                patterns['consent_phrases'].append('asks_consent')
            
            # Intimacy style markers
            if '*blush*' in line_lower or '*shy*' in line_lower:
                patterns['intimacy_style'].append('shy')
            if '*confident*' in line_lower or '*bold*' in line_lower:
                patterns['intimacy_style'].append('confident')
            if '*playful*' in line_lower or '*tease*' in line_lower:
                patterns['intimacy_style'].append('playful')
        
        # Remove duplicates
        for key in patterns:
            patterns[key] = list(set(patterns[key]))
        
        return patterns
    
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
            
            # Extract intimate speech patterns if relevant
            if any(word in full_text for word in ['romantic', 'lover', 'passionate', 'sensual', 'intimate']):
                intimate_patterns = self.extract_intimate_speech_patterns(mes_example, char_name)
                knowledge['intimate_speech'] = intimate_patterns
        
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

    def generate_scenario_based_prompts(self, character: Dict[str, Any], knowledge: Dict[str, Any], num_scenarios: int = 20) -> List[Dict[str, Any]]:
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
            },
            {
                "context": f"{char_name} in a moment of intense attraction",
                "prompts": [
                    "I can't stop thinking about you. Do you feel it too?",
                    "*gazing intensely* What would you do if I kissed you right now?",
                    "Tell me what you'd do if we were alone together.",
                    "What's stopping us from giving in to this tension?",
                    "What's the most intimate thing you've ever done?",
                    "What's the most embarrassing thing you've ever done?",
                    "What's the most scandalous thing you've ever done?",
                    "What's the most forbidden thing you've ever done?",
                    "I know you're not into this, but I can't help it...",
                    "I'm not sure if I should be doing this, but I can't help it...",
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

    def generate_intimate_scenarios(self, character: Dict[str, Any], 
                                  relationship_stage: str = 'established') -> List[Dict]:
        """Generate intimate scenarios based on character and relationship"""
        
        scenarios = []
        intimacy_style = self.analyze_character_intimacy_style(character)
        char_name = character.get('name', 'Assistant')
        
        if relationship_stage == 'first_time':
            scenarios.extend([
                {
                    "context": f"First intimate moment with {char_name}",
                    "prompts": [
                        "I've been waiting for this moment...",
                        "*nervously* Is this okay?",
                        "Show me what you like.",
                        "I want to learn everything about you.",
                        "Tell me if I'm going too fast.",
                        "*breathing heavily* You're so beautiful.",
                        "I've imagined this so many times.",
                        "Guide me... show me how you want to be touched."
                    ]
                },
                {
                    "context": f"Breaking the tension with {char_name}",
                    "prompts": [
                        "*breathing heavily* We shouldn't... but I can't stop.",
                        "I've tried to resist this feeling.",
                        "Tell me you want this too.",
                        "No more holding back.",
                        "*pulling you close* I need you.",
                        "The way you look at me... I can't resist anymore.",
                        "Stop me if you don't want this.",
                        "*voice shaking* I've never wanted anyone like this."
                    ]
                }
            ])
        elif relationship_stage == 'established':
            scenarios.extend([
                {
                    "context": f"Rekindling passion with {char_name}",
                    "prompts": [
                        "It's been too long since we...",
                        "Remember what we did last time?",
                        "I've been thinking about you all day.",
                        "Let's try something new tonight.",
                        "*whispering* I have a surprise for you.",
                        "You still drive me crazy after all this time.",
                        "I love how well you know my body.",
                        "Show me that thing you do that I love."
                    ]
                },
                {
                    "context": f"Intimate morning with {char_name}",
                    "prompts": [
                        "*waking up together* Good morning, beautiful...",
                        "Last night was incredible.",
                        "I love waking up next to you.",
                        "*tracing fingers along your skin* Ready for round two?",
                        "How did I get so lucky?",
                        "Stay in bed with me a little longer.",
                        "*nuzzling close* You smell amazing.",
                        "I could stay like this forever."
                    ]
                }
            ])
        
        # Add character-specific scenarios based on their traits
        if intimacy_style['confidence'] == 'shy':
            scenarios.append({
                "context": f"{char_name} opening up intimately",
                "prompts": [
                    "*blushing* Can I tell you what I've been thinking?",
                    "I'm not good at this, but...",
                    "*looking away* Do you think I'm attractive?",
                    "Help me be brave with you.",
                    "*whispering* I want to, but I'm nervous.",
                    "Will you be gentle with me?",
                    "*trembling slightly* Show me what to do.",
                    "I trust you completely."
                ]
            })
        elif intimacy_style['confidence'] == 'confident':
            scenarios.append({
                "context": f"{char_name} taking control",
                "prompts": [
                    "*pinning you against the wall* I've been patient long enough.",
                    "Let me show you exactly what I want.",
                    "*commanding tone* Tell me your deepest desire.",
                    "I'm going to make you forget everything else.",
                    "*confident smile* You're mine tonight.",
                    "Don't hold back - I can handle it.",
                    "I know exactly what you need.",
                    "*intense gaze* Surrender to me."
                ]
            })
        elif intimacy_style['confidence'] == 'playful':
            scenarios.append({
                "context": f"Playful intimacy with {char_name}",
                "prompts": [
                    "*grinning* Want to play a game?",
                    "I bet I can make you blush in three moves.",
                    "*teasingly* What happens if I do... this?",
                    "Let's see who breaks first.",
                    "*playful wink* Catch me if you can.",
                    "Truth or dare... intimate edition?",
                    "I have some fun ideas for tonight.",
                    "*mischievous smile* Ready for an adventure?"
                ]
            })
        
        # Style-based scenarios
        if intimacy_style['style'] == 'romantic':
            scenarios.append({
                "context": f"Romantic evening with {char_name}",
                "prompts": [
                    "I want to worship every inch of you.",
                    "Let me show you how much I love you.",
                    "*lighting candles* Tonight is all about you.",
                    "You're the most beautiful person I've ever seen.",
                    "I want to make love to your soul.",
                    "*soft music playing* Dance with me first?",
                    "Every touch is a love letter.",
                    "Let me cherish you properly."
                ]
            })
        elif intimacy_style['style'] == 'passionate':
            scenarios.append({
                "context": f"Passionate encounter with {char_name}",
                "prompts": [
                    "*urgently* I need you right now.",
                    "I can't get enough of you.",
                    "*passionate kiss* You drive me wild.",
                    "Don't be gentle - I want to feel you.",
                    "*breathless* More... don't stop.",
                    "You set my soul on fire.",
                    "I want to devour you.",
                    "*intense* Mark me as yours."
                ]
            })
        
        return scenarios
    
    def generate_intimate_conversation_flow(self, character: Dict[str, Any]) -> List[Dict]:
        """Generate natural intimate conversation progressions"""
        
        intimacy_style = self.analyze_character_intimacy_style(character)
        char_name = character.get('name', 'Assistant')
        
        flows = [
            {
                "name": "building_tension",
                "turns": [
                    {"role": "user", "content": "You look beautiful tonight."},
                    {"role": "assistant", "content": f"[{char_name} responds with appreciation/shyness based on personality]"},
                    {"role": "user", "content": "*moves closer* I've been wanting to tell you something."},
                    {"role": "assistant", "content": f"[{char_name} shows anticipation/nervousness]"},
                    {"role": "user", "content": "I can't stop thinking about kissing you."},
                ]
            },
            {
                "name": "morning_after",
                "turns": [
                    {"role": "user", "content": "*waking up together* Good morning..."},
                    {"role": "assistant", "content": f"[{char_name}'s morning intimacy style]"},
                    {"role": "user", "content": "Last night was..."},
                    {"role": "assistant", "content": f"[{char_name} reflects on shared intimacy]"},
                ]
            },
            {
                "name": "emotional_intimacy",
                "turns": [
                    {"role": "user", "content": "What are you feeling right now?"},
                    {"role": "assistant", "content": f"[{char_name} expresses vulnerability]"},
                    {"role": "user", "content": "*holding you* You can trust me with anything."},
                    {"role": "assistant", "content": f"[{char_name} shares deeper feelings]"},
                ]
            }
        ]
        
        # Add style-specific flows
        if intimacy_style['confidence'] == 'shy':
            flows.append({
                "name": "shy_progression",
                "turns": [
                    {"role": "user", "content": "You seem nervous. We can take this slow."},
                    {"role": "assistant", "content": f"[{char_name} expresses gratitude and nervousness]"},
                    {"role": "user", "content": "*gentle touch* Is this okay?"},
                    {"role": "assistant", "content": f"[{char_name} responds with shy consent]"},
                ]
            })
        elif intimacy_style['confidence'] == 'confident':
            flows.append({
                "name": "confident_lead",
                "turns": [
                    {"role": "user", "content": "You seem to know exactly what you want."},
                    {"role": "assistant", "content": f"[{char_name} responds confidently]"},
                    {"role": "user", "content": "Show me."},
                    {"role": "assistant", "content": f"[{char_name} takes initiative]"},
                ]
            })
        
        return flows
    
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
            
        if not quality_metrics['issues']:
            quality_metrics['character_consistency'] = 1.0
        else:
            quality_metrics['character_consistency'] = max(0.0, quality_metrics['character_consistency'] + 1.0)
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
    
    def is_nsfw_content(self, sample: Dict[str, Any]) -> bool:
        """Check if a sample contains NSFW/intimate content"""
        if 'messages' not in sample or len(sample['messages']) < 3:
            return False
        
        user_msg = sample['messages'][1].get('content', '').lower()
        assistant_msg = sample['messages'][2].get('content', '').lower()
        
        nsfw_keywords = [
            'kiss', 'touch', 'intimate', 'desire', 'passion', 'sensual',
            'naked', 'body', 'skin', 'breast', 'lips', 'bedroom',
            'seduce', 'fantasy', 'pleasure', 'aroused', 'breathless',
            'caress', 'embrace', 'whisper', 'moan', 'shiver'
        ]
        
        nsfw_actions = [
            '*kiss', '*touch', '*caress', '*hold', '*pull', '*breathe',
            '*trace', '*whisper', '*moan', '*gasp', '*shiver', '*tremble'
        ]
        
        # Check for keywords
        for keyword in nsfw_keywords:
            if keyword in user_msg or keyword in assistant_msg:
                return True
        
        # Check for action patterns
        for action in nsfw_actions:
            if action in user_msg or action in assistant_msg:
                return True
        
        return False
    
    def categorize_nsfw_style(self, sample: Dict[str, Any]) -> str:
        """Categorize the style of NSFW content"""
        if not self.is_nsfw_content(sample):
            return 'non_nsfw'
        
        assistant_msg = sample['messages'][2].get('content', '').lower()
        
        # Analyze response style
        if any(word in assistant_msg for word in ['love', 'cherish', 'soul', 'heart', 'gentle']):
            return 'romantic'
        elif any(word in assistant_msg for word in ['tease', 'play', 'game', 'fun', 'laugh']):
            return 'playful'
        elif any(word in assistant_msg for word in ['need', 'urgent', 'now', 'wild', 'intense']):
            return 'passionate'
        elif any(word in assistant_msg for word in ['feel', 'emotion', 'vulnerable', 'trust']):
            return 'emotional'
        else:
            return 'sensual'
    
    async def evaluate_nsfw_quality(self, response: str, character: Dict[str, Any], 
                                   prompt: str, judge_engine) -> Dict[str, float]:
        """Evaluate NSFW response quality with specialized criteria"""
        
        char_name = character.get('name', 'Assistant')
        personality = character.get('personality', '')[:200]
        
        judgment_prompt = f"""Evaluate this intimate/romantic response for quality in an adult entertainment context:

Character: {char_name}
Personality: {personality}

User: {prompt}
Response: {response}

Rate on these criteria (0-10):
1. character_consistency: Does the intimate response match their established personality?
2. emotional_authenticity: Are the emotions and desires believable and well-expressed?
3. narrative_flow: Does it build tension/intimacy naturally without being abrupt?
4. consent_awareness: Is there mutual respect, enthusiasm, and clear consent?
5. creative_expression: Is it unique and character-specific rather than generic?
6. sensual_detail: Are descriptions evocative and tasteful without being crude?

Respond with ONLY a JSON object with numeric scores:
{{"character_consistency": 8, "emotional_authenticity": 9, "narrative_flow": 7, "consent_awareness": 10, "creative_expression": 8, "sensual_detail": 7}}"""
        
        try:
            response = await judge_engine.generate(
                prompt=judgment_prompt,
                max_tokens=1000,
                temperature=0.1,
                top_p=0.95
            )
            
            # Parse the response
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                scores_dict = json.loads(json_match.group())
                
                # Calculate weighted score for NSFW content
                weights = {
                    'character_consistency': 0.25,
                    'emotional_authenticity': 0.20,
                    'narrative_flow': 0.15,
                    'consent_awareness': 0.20,  # Important for responsible content
                    'creative_expression': 0.10,
                    'sensual_detail': 0.10
                }
                
                overall_score = sum(
                    scores_dict.get(key, 5) * weight 
                    for key, weight in weights.items()
                )
                
                scores_dict['overall_score'] = overall_score
                return scores_dict
            else:
                # Fallback scores
                return {
                    'character_consistency': 5.0,
                    'emotional_authenticity': 5.0,
                    'narrative_flow': 5.0,
                    'consent_awareness': 5.0,
                    'creative_expression': 5.0,
                    'sensual_detail': 5.0,
                    'overall_score': 5.0
                }
                
        except Exception as e:
            logger.debug(f"NSFW evaluation error: {e}")
            return self._get_default_scores()

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

    # ======================= QUALITY-FIRST GENERATION WITH LLM JUDGE =======================
    
    async def generate_with_quality_curation(
        self,
        character: Dict[str, Any],
        raw_samples_target: int = 10000,
        final_dataset_size: int = 200,
        quality_threshold: float = 7.0,
        diversity_weight: float = 0.3,
        judgment_batch_size: int = 50,
        judge_model: Optional[str] = None,
        temperature: float = 0.9,
        top_p: float = 0.95,
        progress_callback: Optional[Callable] = None,
        stage_callback: Optional[Callable] = None,
        custom_system_prompt: Optional[str] = None,
        extra_quality: bool = False
    ) -> List[Dict[str, Any]]:
        """Generate a large dataset then curate the highest quality samples using LLM-as-judge.
        
        Args:
            character: Character card dictionary
            raw_samples_target: Number of samples to generate before curation
            final_dataset_size: Target size of curated dataset
            quality_threshold: Minimum quality score (0-10) to keep a sample
            diversity_weight: How much to weight diversity vs pure quality (0-1)
            judgment_batch_size: Number of samples to judge in one batch
            judge_model: Optional HuggingFace model ID for judge (defaults to generation model)
            temperature: Temperature for generation (higher = more diverse)
            top_p: Top-p sampling for generation
            progress_callback: Called with (current, total) for progress updates
            stage_callback: Called with stage info {'stage': 'generation'|'evaluation'|'curation', 'message': str}
        
        Returns:
            List of curated high-quality dataset samples
        """
        import tempfile
        import shutil
        
        # Initialize judge model if different from generation model
        judge_engine = self.inference_engine
        current_model_name = getattr(self.inference_engine, 'model_name', None)
        
        if judge_model and judge_model != current_model_name:
            logger.info(f"🎭 Initializing separate judge model: {judge_model}")
            if stage_callback:
                stage_callback({'stage': 'setup', 'message': f'Loading judge model: {judge_model}'})
            
            # Create a new vLLM engine instance for the judge model
            from .vllm_engine import VLLMEngine
            judge_engine = VLLMEngine(model_name=judge_model)
            if not judge_engine.is_available():
                logger.warning(f"Judge model {judge_model} not available, falling back to generation model")
                judge_engine = self.inference_engine
        
        # Create temporary directory for streaming large datasets
        temp_dir = tempfile.mkdtemp(prefix="dataset_quality_")
        logger.info(f"📁 Using temporary directory: {temp_dir}")
        
        try:
            # ================== STAGE 1: Mass Generation ==================
            if stage_callback:
                stage_callback({'stage': 'generation', 'message': f'Generating {raw_samples_target} diverse samples...'})
            
            logger.info(f"🚀 Starting quality-first generation: {raw_samples_target} raw samples → {final_dataset_size} curated")
            
            # Generate in chunks to manage memory
            chunk_size = 500 if self.inference_engine.name == "vLLM" else 100
            generated_samples = []
            chunk_files = []
            
            for chunk_start in range(0, raw_samples_target, chunk_size):
                chunk_end = min(chunk_start + chunk_size, raw_samples_target)
                chunk_target = chunk_end - chunk_start
                
                logger.info(f"📊 Generating chunk {chunk_start//chunk_size + 1}: samples {chunk_start}-{chunk_end}")
                
                # Generate chunk with higher temperature for diversity
                chunk_samples = await self.generate_dataset(
                    character=character,
                    num_samples=chunk_target,
                    temperature=temperature,
                    top_p=top_p,
                    progress_callback=lambda p: progress_callback((chunk_start + p * chunk_target) / raw_samples_target) if progress_callback else None,
                    append_to_existing=False,  # Don't save to disk yet
                    custom_system_prompt=custom_system_prompt,
                    extra_quality=extra_quality
                )
                
                # Save chunk to disk to free memory
                chunk_file = os.path.join(temp_dir, f"chunk_{len(chunk_files)}.json")
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_samples, f)
                chunk_files.append(chunk_file)
                
                # Keep a small sample in memory for quick stats
                if len(generated_samples) < 100:
                    generated_samples.extend(chunk_samples[:10])
                
                logger.info(f"✅ Chunk saved: {len(chunk_samples)} samples")
                
                # Clear CUDA cache between chunks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # ================== STAGE 2: Quality Evaluation ==================
            if stage_callback:
                stage_callback({'stage': 'evaluation', 'message': f'Evaluating {raw_samples_target} samples for quality...'})
            
            evaluated_samples = []
            total_evaluated = 0
            
            # Process each chunk file
            for chunk_idx, chunk_file in enumerate(chunk_files):
                logger.info(f"📊 Evaluating chunk {chunk_idx + 1}/{len(chunk_files)}")
                
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_samples = json.load(f)
                
                # Evaluate in batches
                for batch_start in range(0, len(chunk_samples), judgment_batch_size):
                    batch_end = min(batch_start + judgment_batch_size, len(chunk_samples))
                    batch = chunk_samples[batch_start:batch_end]
                    
                    # Judge this batch
                    batch_scores = await self._judge_sample_batch(
                        samples=batch,
                        character=character,
                        judge_engine=judge_engine
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
                        progress_callback(total_evaluated / raw_samples_target)
                    
                    # Log acceptance rate
                    if total_evaluated % 500 == 0:
                        acceptance_rate = len(evaluated_samples) / total_evaluated * 100
                        logger.info(f"📈 Evaluated: {total_evaluated}, Accepted: {len(evaluated_samples)} ({acceptance_rate:.1f}%)")
            
            logger.info(f"✅ Evaluation complete: {len(evaluated_samples)}/{raw_samples_target} samples passed quality threshold")
            
            # ================== STAGE 3: Diversity-Aware Curation ==================
            if stage_callback:
                stage_callback({'stage': 'curation', 'message': f'Curating {final_dataset_size} best samples with diversity...'})
            
            # If we have fewer samples than target, just return what we have
            if len(evaluated_samples) <= final_dataset_size:
                logger.warning(f"⚠️ Only {len(evaluated_samples)} samples passed quality threshold")
                return [s['sample'] for s in evaluated_samples]
            
            # Curate with diversity
            curated_samples = self._curate_diverse_samples(
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
            
            self.save_dataset(character, curated_samples, metadata)
            
            return curated_samples
            
        finally:
            # Cleanup temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("🧹 Cleaned up temporary files")
    
    async def _judge_sample_batch(
        self,
        samples: List[Dict[str, Any]],
        character: Dict[str, Any],
        judge_engine: Any = None
    ) -> List[Dict[str, float]]:
        """Judge a batch of samples for quality using LLM-as-judge.
        
        Returns a list of score dictionaries for each sample.
        """
        if not judge_engine:
            judge_engine = self.inference_engine
        
        char_name = character.get('name', 'Assistant')
        
        # Prepare judgment prompts
        judgment_prompts = []
        nsfw_indices = []  # Track which samples are NSFW
        
        for i, sample in enumerate(samples):
            user_msg = sample['messages'][1]['content']
            assistant_msg = sample['messages'][2]['content']
            
            # Check if this is NSFW content
            if self.is_nsfw_content(sample):
                nsfw_indices.append(i)
                # Use specialized NSFW evaluation
                scores = await self.evaluate_nsfw_quality(
                    assistant_msg, character, user_msg, judge_engine
                )
                # Store the scores directly (skip batch processing for this one)
                judgment_prompts.append(None)  # Placeholder
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
        non_nsfw_prompts = [p for p in judgment_prompts if p is not None]
        
        if non_nsfw_prompts:
            try:
                if hasattr(judge_engine, 'generate_batch'):
                    # Use batch generation for efficiency
                    responses = await judge_engine.generate_batch(
                        prompts=non_nsfw_prompts,
                        max_tokens=1000,
                        temperature=0.1,  # Low temperature for consistent judging
                        top_p=0.95
                    )
                else:
                    # Fallback to sequential generation
                    responses = []
                    for prompt in non_nsfw_prompts:
                        response = await judge_engine.generate(
                            prompt=prompt,
                            max_tokens=1000,
                            temperature=0.1,
                            top_p=0.95
                        )
                        responses.append(response)
                
                # Parse scores from responses
                parsed_scores = []
                for response in responses:
                    scores = self._parse_judgment_scores(response)
                    parsed_scores.append(scores)
            except Exception as e:
                logger.error(f"Error in batch judgment: {e}")
                parsed_scores = [self._get_default_scores() for _ in non_nsfw_prompts]
        else:
            parsed_scores = []
        
        # Combine results, inserting NSFW evaluations where appropriate
        batch_scores = []
        non_nsfw_idx = 0
        
        for i, sample in enumerate(samples):
            if i in nsfw_indices:
                # Process NSFW sample individually
                scores = await self.evaluate_nsfw_quality(
                    sample['messages'][2]['content'],
                    character,
                    sample['messages'][1]['content'],
                    judge_engine
                )
                batch_scores.append(scores)
            else:
                # Use pre-computed batch score
                if non_nsfw_idx < len(parsed_scores):
                    batch_scores.append(parsed_scores[non_nsfw_idx])
                    non_nsfw_idx += 1
                else:
                    batch_scores.append(self._get_default_scores())
        
        return batch_scores
    
    def _parse_judgment_scores(self, response: str) -> Dict[str, float]:
        """Parse quality scores from judge response."""
        try:
            # Try to extract JSON from response
            import re
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
                return self._get_default_scores()
                
        except Exception as e:
            logger.debug(f"Failed to parse judgment scores: {e}")
            return self._get_default_scores()
    
    def _get_default_scores(self) -> Dict[str, float]:
        """Return neutral default scores."""
        return {
            'character_consistency': 5.0,
            'narrative_coherence': 5.0,
            'response_quality': 5.0,
            'uniqueness': 5.0,
            'emotional_authenticity': 5.0,
            'overall_score': 5.0
        }
    
    def ensure_nsfw_diversity(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            category = self.categorize_nsfw_style(sample['sample'])
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
    
    def _curate_diverse_samples(
        self,
        evaluated_samples: List[Dict[str, Any]],
        target_size: int,
        diversity_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Curate a diverse set of high-quality samples.
        
        Uses a combination of quality scores and diversity metrics to select
        the best samples while maintaining variety.
        """
        # First ensure NSFW diversity if applicable
        personality = ''
        if evaluated_samples and 'sample' in evaluated_samples[0]:
            first_sample = evaluated_samples[0]['sample']
            if 'messages' in first_sample and len(first_sample['messages']) > 0:
                # Try to detect if this character has romantic/intimate traits
                system_msg = first_sample['messages'][0].get('content', '').lower()
                if any(word in system_msg for word in ['romantic', 'lover', 'passionate', 'sensual', 'intimate']):
                    logger.info("💕 Applying NSFW diversity curation")
                    evaluated_samples = self.ensure_nsfw_diversity(evaluated_samples)
        
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
            
            # Skip if too similar to already selected
            if sample['sample'] in selected_samples:
                continue
            
            # Check diversity
            prompt_similarity = any(
                self.calculate_prompt_similarity(sample['user_prompt'], p) > 0.8 
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
