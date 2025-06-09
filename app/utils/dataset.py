import asyncio
import random
import re
import textwrap
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from datasets import Dataset
from .inference_engines import get_inference_engine
import torch

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages synthetic dataset generation and processing"""
    
    def __init__(self, preferred_engine: Optional[str] = None):
        self.inference_engine = get_inference_engine(preferred_engine)
        # For DanChat-2 we only need a *single* chat template – the chat
        # wrapper (<|user|> …) is added later by vLLM/HF tokenizer.
        self.templates = [("chat", "{user_prompt}")]

        # High-fidelity prompt buckets -------------------------------------------------
        self.prompts_casual = [
            "hey there!",
            "hi 👋", "yo, got a sec?", "morning ☀️", "sup?",
            "hellooo~", "🖐️ hi!"
        ]

        self.prompts_personal = [
            "do you ever feel lonely?", "what keeps you awake at night?",
            "what's your biggest fear?", "have you ever been in love?",
            "what are you most proud of?", "any secret dreams?"
        ]

        self.prompts_action = [
            "*pushes the door open* you coming?",
            "quick, hide with me!", "help me pick this lock…",
            "look out! what's that?", "hold my hand and run!"
        ]

        self.prompts_emotion = [
            "i'm feeling kinda down today…", "haha that was hilarious 😂",
            "ugh, this place gives me the creeps…", "i'm so excited!!",
            "why am i crying?", "that makes me angry!"
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
        ]

        # Sampling weights for variety (must sum to 1.0)
        self.bucket_choices = [
            ("casual",   0.35),
            ("personal", 0.25),
            ("action",   0.15),
            ("emotion",  0.15),
            ("intimate", 0.10),
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
    
    async def _generate_text(self, prompt: str, max_tokens: int = 160, 
                           temperature: float = 0.8, top_p: float = 0.9) -> str:
        """Generate text using the configured inference engine"""
        try:
            return await self.inference_engine.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
        except Exception as e:
            raise RuntimeError(f"Text generation failed ({self.inference_engine.name}): {str(e)}")
    
    async def _generate_text_batch(self, prompts: list[str], max_tokens: int = 160,
                                 temperature: float = 0.8, top_p: float = 0.9) -> list[str]:
        """Generate text for multiple prompts using batching (if supported)"""
        try:
            # Check if engine supports batching
            if hasattr(self.inference_engine, 'generate_batch'):
                return await self.inference_engine.generate_batch(
                    prompts=prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
            else:
                # Fallback to individual generation
                results = []
                for prompt in prompts:
                    result = await self._generate_text(prompt, max_tokens, temperature, top_p)
                    results.append(result)
                return results
        except Exception as e:
            raise RuntimeError(f"Batch text generation failed ({self.inference_engine.name}): {str(e)}")
    
    # ------------------------- prompt sampling helpers -------------------------
    def _choose_bucket(self) -> str:
        buckets, weights = zip(*self.bucket_choices)
        return random.choices(buckets, weights=weights, k=1)[0]

    def _add_noise(self, text: str) -> str:
        """Apply light realism noise: random lowercase, ellipsis, emoji, etc."""
        if random.random() < 0.2:
            text = text.capitalize() if random.random() < 0.5 else text.lower()
        if random.random() < 0.15:
            text += random.choice(["…", " :)", " 😅", " ;)", " 🤔"])
        # occasional stutter
        if random.random() < 0.05 and len(text.split()) > 2:
            first = text.split()[0]
            text = f"{first[0]}-{first} " + " ".join(text.split()[1:])
        return text

    # ---------------- paraphrasing & back-translation ----------------
    def _paraphrase(self, text: str) -> str:
        if not self.paraphraser or random.random() > 0.5:  # 50% keep original
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
        if random.random() > 0.1:  # only 10% of prompts
            return text

        try:
            def _get(model_name):
                if model_name not in self._bt_models:
                    tok = self._AutoTokenizer.from_pretrained(model_name)
                    mod = self._AutoModel.from_pretrained(model_name)
                    self._bt_tokenizers[model_name] = tok
                    self._bt_models[model_name] = mod
                return self._bt_tokenizers[model_name], self._bt_models[model_name]

            tok_en_fr, mod_en_fr = _get("Helsinki-NLP/opus-mt-en-fr")
            tok_fr_en, mod_fr_en = _get("Helsinki-NLP/opus-mt-fr-en")

            tgt = mod_en_fr.generate(**tok_en_fr(text, return_tensors="pt", truncation=True))
            fr = tok_en_fr.decode(tgt[0], skip_special_tokens=True)
            src = mod_fr_en.generate(**tok_fr_en(fr, return_tensors="pt", truncation=True))
            return tok_fr_en.decode(src[0], skip_special_tokens=True)
        except Exception:
            return text

    def _build_user_prompt(self) -> str:
        bucket = self._choose_bucket()
        prompt_list = getattr(self, f"prompts_{bucket}")
        prompt = random.choice(prompt_list)
        prompt = self._add_noise(prompt.strip())
        prompt = self._paraphrase(prompt)
        prompt = self._backtranslate(prompt)
        return prompt

    # ---------------------------------------------------------------------------

    def _fill_template(self, template: str, card: Dict[str, str]) -> str:
        """Return a single realistic user prompt (no formatting placeholders)."""
        return self._build_user_prompt()
    
    def _make_card_block(self, card: Dict[str, str]) -> str:
        """Build a DanChat-2 compatible *system* prompt block.

        The real chat template (added later by the tokenizer or vLLM) will wrap
        this string in ``<|system|> ... <|endoftext|>``, so **do not** add
        those markers here – just construct the inner text exactly like
        SillyTavern does when using the DanChat-2 preset (see `story_string` in
        the question):

            <|system|>{system/wiBefore/description/personality/scenario/…}<|endoftext|>

        We include only the fields that are present in the character card and
        keep the order identical to SillyTavern:

            system
            wiBefore
            description
            {name}'s personality: {personality}
            Scenario: {scenario}
            wiAfter
            persona
        """

        lines: list[str] = [f"You are {card.get('name', 'a character in a story')}. Bellow are your details. Behave accordingly."]

        # 1. explicit system string if provided
        if sys_msg := card.get("system"):
            lines.append(sys_msg.strip())

        # 2. wiBefore – extra world-info before the description
        if wi_before := card.get("wiBefore"):
            lines.append(wi_before.strip())

        # 3. description
        if desc := card.get("description"):
            lines.append(desc.strip())

        # 4. personality
        if pers := card.get("personality"):
            char_name = card.get("name", "The character")
            lines.append(f"{char_name}'s personality: {pers.strip()}")

        # 5. scenario
        if scen := card.get("scenario"):
            lines.append(f"Scenario: {scen.strip()}")

        # 6. wiAfter – world-info appended after the scenario
        if wi_after := card.get("wiAfter"):
            lines.append(wi_after.strip())

        # 7. persona (rarely used but supported by ST)
        if persona := card.get("persona"):
            lines.append(persona.strip())

        # If everything is missing fall back to a generic instruction so the
        # prompt is never empty.
        if not lines:
            lines.append("You are a fictional character. Respond in character at all times.")

        # The story_string uses single new-line separators and gets trimmed.
        return "\n".join(lines).strip()
    
    async def generate_dataset(self, character: Dict[str, Any], num_samples: int = 200,
                             max_tokens: int = 300, temperature: float = 0.8,
                             top_p: float = 0.9, progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Generate synthetic dataset for character using efficient batching"""
        card_block = self._make_card_block(character)
        samples = []
        
        # Determine batch size based on inference engine
        batch_size = 8 if hasattr(self.inference_engine, 'generate_batch') else 1
        
        # Pre-generate all prompts
        prompts_data = []
        for i in range(num_samples * 2):  # Generate extra to account for filtering
            try:
                # Select random template
                mode, template = random.choice(self.templates)
                prompt = self._fill_template(template, character)
                
                # Validate template filling
                unfilled_placeholders = re.findall(r'\{[^}]+\}', prompt)
                if unfilled_placeholders:
                    continue
                
                prompts_data.append({
                    'prompt': prompt,
                    'full_prompt': f"{card_block}\n\n{prompt}",
                    'template_mode': mode
                })
                
                if len(prompts_data) >= num_samples * 1.5:  # Buffer for filtering
                    break
                    
            except Exception as e:
                logger.debug(f"Error preparing prompt {i}: {e}")
                continue
        
        # Process in batches
        processed_count = 0
        for batch_start in range(0, len(prompts_data), batch_size):
            try:
                batch_end = min(batch_start + batch_size, len(prompts_data))
                batch_prompts = prompts_data[batch_start:batch_end]
                
                # Extract full prompts for generation
                full_prompts = [item['full_prompt'] for item in batch_prompts]
                
                # Generate batch
                if batch_size > 1:
                    replies = await self._generate_text_batch(
                        prompts=full_prompts,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                else:
                    # Single generation fallback
                    replies = [await self._generate_text(
                        prompt=full_prompts[0],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )]
                
                # Process batch results
                for i, (prompt_data, reply) in enumerate(zip(batch_prompts, replies)):
                    try:
                        # Quality filters
                        word_count = len(reply.split())
                        if word_count < 3 or word_count > 420:
                            continue
                        
                        # Create ChatML sample
                        sample = {
                            "messages": [
                                {"role": "system", "content": card_block},
                                {"role": "user", "content": prompt_data['prompt']},
                                {"role": "assistant", "content": reply},
                            ]
                        }
                        samples.append(sample)
                        processed_count += 1
                        
                        # Update progress
                        if progress_callback:
                            progress_callback(min(processed_count / num_samples, 1.0))
                        
                        # Stop if we have enough samples
                        if processed_count >= num_samples:
                            break
                            
                    except Exception as e:
                        logger.debug(f"Error processing sample: {e}")
                        continue
                
                # Stop if we have enough samples
                if processed_count >= num_samples:
                    break
                
                # Small delay between batches (much less than before)
                if batch_size > 1:
                    await asyncio.sleep(0.5)
                else:
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Generation error for batch {batch_start}-{batch_end}: {e}")
                continue
        
        if progress_callback:
            progress_callback(1.0)
        
        logger.info(f"Generated {len(samples)} samples using {self.inference_engine.name} (batch_size: {batch_size})")
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
        
        # Quality score (weighted combination of metrics)
        quality_score = (
            uniqueness_ratio * 0.4 +
            min(avg_length / 50, 1.0) * 0.3 +  # Normalize avg length to 0-1
            template_diversity * 0.3
        ) * 100
        
        return {
            'total_samples': total_samples,
            'unique_responses': unique_responses,
            'uniqueness_ratio': uniqueness_ratio,
            'avg_response_length': avg_length,
            'template_diversity': template_diversity,
            'templates_used': list(templates_used),
            'quality_score': quality_score
        }
    
    def prepare_for_training(self, dataset: List[Dict[str, Any]], tokenizer, max_length: int = 4096) -> Dataset:
        """Prepare dataset for training by tokenizing"""
        def process_example(example):
            """Process a single example for training"""
            messages = example["messages"]
            
            # Apply chat template
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
            
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