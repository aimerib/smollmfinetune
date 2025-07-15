#!/usr/bin/env python3
"""
Synthetic Conversation Generation System (R4-2.5)

This module provides comprehensive tools for generating high-quality synthetic
training data for the Narrative Engine, including conversation generation,
quality assessment, and dataset curation.
"""

import asyncio
import json
import uuid
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from backend.app.narrative_engine.data_schema import DatasetSample, Turn

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for synthetic data generation"""
    conversation_length: int = 6
    action_frequency: float = 0.3
    scenario_diversity: str = "medium"  # low, medium, high
    api_provider: str = "openai"
    temperature: float = 0.8
    max_tokens: int = 1000
    quality_threshold: float = 0.6


class ConversationTemplates:
    """Manages conversation templates for different scenarios"""
    
    def __init__(self):
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self) -> List[Dict[str, Any]]:
        """Load default conversation templates"""
        return [
            {
                "name": "research_session",
                "description": "Character engaged in research or investigation",
                "prompts": [
                    "What are you currently researching?",
                    "Can you search for more information about this?",
                    "What have you discovered so far?",
                    "How does this relate to your previous work?"
                ],
                "expected_actions": ["search", "memory_recall", "analysis"],
                "action_probability": 0.6
            },
            {
                "name": "teaching_moment", 
                "description": "Character explaining or teaching concepts",
                "prompts": [
                    "Can you explain this concept to me?",
                    "What's the most important thing to understand?",
                    "Could you give me an example?",
                    "How would you teach this to a beginner?"
                ],
                "expected_actions": ["knowledge_lookup", "example_generation"],
                "action_probability": 0.3
            },
            {
                "name": "discovery_event",
                "description": "Character making new discoveries or insights",
                "prompts": [
                    "What's this new discovery you've made?",
                    "How did you figure this out?",
                    "What does this mean for your work?",
                    "Can you verify this finding?"
                ],
                "expected_actions": ["verification", "cross_reference", "documentation"],
                "action_probability": 0.7
            },
            {
                "name": "collaboration",
                "description": "Character working with others or seeking help",
                "prompts": [
                    "I'd like to collaborate on this project",
                    "What do you think about my approach?",
                    "Can you help me with this problem?",
                    "How should we tackle this together?"
                ],
                "expected_actions": ["coordination", "resource_sharing", "planning"],
                "action_probability": 0.4
            },
            {
                "name": "casual_conversation",
                "description": "Informal dialogue showing personality",
                "prompts": [
                    "How are you doing today?",
                    "What's on your mind?",
                    "Tell me about yourself",
                    "What do you enjoy most about your work?"
                ],
                "expected_actions": ["memory_recall", "reflection"],
                "action_probability": 0.2
            }
        ]
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """Get all available templates"""
        return self.templates
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific template by name"""
        return next((t for t in self.templates if t["name"] == name), None)


class SyntheticDataGenerator:
    """Main class for generating synthetic conversation data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the generator with configuration"""
        self.config = GenerationConfig(**config) if config else GenerationConfig()
        self.templates = ConversationTemplates()
        
        # Import OpenAI client lazily
        try:
            from backend.app.core.openai_client import get_client
            self.client = get_client()
        except ImportError:
            logger.warning("OpenAI client not available, using mock responses")
            self.client = None
    
    async def generate_conversation(
        self, 
        character: Dict[str, Any], 
        scenario_type: Optional[str] = None
    ) -> DatasetSample:
        """
        Generate a single synthetic conversation for a character
        
        Args:
            character: Character definition with personality, goals, etc.
            scenario_type: Optional specific scenario template to use
            
        Returns:
            DatasetSample object ready for training
        """
        # Select scenario template
        if scenario_type:
            template = self.templates.get_template(scenario_type)
            if not template:
                logger.warning(f"Unknown scenario type: {scenario_type}, using random")
                template = self._select_random_template()
        else:
            template = self._select_random_template()
        
        # Generate session ID
        scenario_name = template["name"] if template else "general"
        session_id = f"synthetic_{scenario_name}_{character['name'].lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        
        # Create persona mix from character
        persona_mix = self._create_persona_mix(character)
        
        # Generate memory slots
        memory_slots = self._create_memory_slots(character, template)
        
        # Generate conversation turns
        turns = await self._generate_turns(character, template)
        
        return DatasetSample(
            session_id=session_id,
            persona_mix=persona_mix,
            memory_slots=memory_slots,
            turns=turns
        )
    
    async def generate_batch(
        self, 
        characters: List[Dict[str, Any]], 
        conversations_per_character: int = 3
    ) -> List[DatasetSample]:
        """
        Generate multiple conversations in batch
        
        Args:
            characters: List of character definitions
            conversations_per_character: Number of conversations per character
            
        Returns:
            List of generated DatasetSample objects
        """
        tasks = []
        for character in characters:
            for _ in range(conversations_per_character):
                task = self.generate_conversation(character)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Failed to generate conversation: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def _select_random_template(self) -> Dict[str, Any]:
        """Select a random conversation template"""
        import random
        templates = self.templates.get_templates()
        return random.choice(templates)
    
    def _create_persona_mix(self, character: Dict[str, Any]) -> Dict[str, float]:
        """Create persona mix from character tags and traits"""
        persona_mix = {}
        
        # Tag-based persona mapping
        tag_mapping = {
            'scholar': 'Scholar',
            'academic': 'Scholar', 
            'research': 'Scholar',
            'detective': 'Detective',
            'investigator': 'Detective',
            'teacher': 'Mentor',
            'mentor': 'Mentor',
            'guide': 'Mentor',
            'warrior': 'Warrior',
            'fighter': 'Warrior',
            'healer': 'Healer',
            'doctor': 'Healer',
            'artist': 'Creator',
            'creator': 'Creator',
            'explorer': 'Explorer',
            'adventurer': 'Explorer'
        }
        
        # Build persona from tags
        tags = character.get('tags', [])
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in tag_mapping:
                persona_name = tag_mapping[tag_lower]
                persona_mix[persona_name] = persona_mix.get(persona_name, 0) + 0.5
        
        # If no tags matched, use personality traits
        if not persona_mix:
            personality = character.get('personality_traits', {})
            
            # Analyze Big Five traits for persona
            if personality.get('openness', 0.5) > 0.7:
                persona_mix['Explorer'] = 0.6
            if personality.get('conscientiousness', 0.5) > 0.7:
                persona_mix['Scholar'] = 0.6
            if personality.get('extraversion', 0.5) > 0.7:
                persona_mix['Leader'] = 0.6
            if personality.get('agreeableness', 0.5) > 0.7:
                persona_mix['Helper'] = 0.6
        
        # Default fallback
        if not persona_mix:
            persona_mix = {'Balanced': 1.0}
        
        # Normalize to sum to 1.0
        total = sum(persona_mix.values())
        if total > 0:
            persona_mix = {k: v / total for k, v in persona_mix.items()}
        
        return persona_mix
    
    def _create_memory_slots(self, character: Dict[str, Any], template: Dict[str, Any]) -> List[str]:
        """Create memory slots from character and scenario context"""
        memory_slots = []
        
        # Character identity
        memory_slots.append(f"I am {character['name']}")
        
        # Key traits and expertise
        if character.get('description'):
            memory_slots.append(f"My expertise: {character['description'][:80]}...")
        
        # Current goals
        goals = character.get('goals', [])
        if goals:
            memory_slots.append(f"Current focus: {goals[0]}")
        
        # Scenario context
        if template:
            memory_slots.append(f"Current activity: {template['description']}")
        
        # Current situation
        scenario = character.get('scenario', '')
        if scenario:
            memory_slots.append(f"Context: {scenario[:60]}...")
        
        return memory_slots[:5]  # Limit to 5 slots
    
    async def _generate_turns(self, character: Dict[str, Any], template: Dict[str, Any]) -> List[Turn]:
        """Generate conversation turns using LLM API"""
        if not self.client:
            # Return mock data for testing
            return self._generate_mock_turns(character, template)
        
        # Build prompt for LLM
        prompt = self._build_generation_prompt(character, template)
        
        # Call LLM API
        response = await self._call_llm_api(prompt)
        
        # Parse response into turns
        turns = self._parse_llm_response(response, character)
        
        return turns
    
    def _generate_mock_turns(self, character: Dict[str, Any], template: Dict[str, Any]) -> List[Turn]:
        """Generate mock turns for testing"""
        char_name = character['name']
        
        turns = [
            Turn(
                sender="user",
                text="Hello! Can you tell me what you're working on?",
                channel="text"
            ),
            Turn(
                sender="assistant",
                text=f"Hello! I'm {char_name}. I'm currently {character.get('scenario', 'working on various projects')}.",
                channel="text"
            )
        ]
        
        # Add action turn if template suggests it
        if template and template.get('action_probability', 0) > 0.3:
            turns.extend([
                Turn(
                    sender="user",
                    text="Could you search for more information about that?",
                    channel="text"
                ),
                Turn(
                    sender="assistant",
                    text="I'll look that up for you right now.",
                    channel="action",
                    action={"tool": "search", "query": "research information"}
                )
            ])
        
        return turns
    
    def _build_generation_prompt(self, character: Dict[str, Any], template: Dict[str, Any]) -> str:
        """Build prompt for LLM conversation generation"""
        prompt = f"""Generate a natural conversation between a user and an AI character.

Character Profile:
- Name: {character['name']}
- Description: {character.get('description', '')}
- Personality: {self._format_personality(character.get('personality_traits', {}))}
- Goals: {', '.join(character.get('goals', []))}
- Current Situation: {character.get('scenario', '')}

Conversation Template: {template['name']} - {template['description']}

Requirements:
1. Generate {self.config.conversation_length} total turns (alternating user/assistant)
2. Include {int(self.config.action_frequency * 100)}% action-channel responses where the assistant performs structured actions
3. Use these potential user prompts: {', '.join(template['prompts'][:2])}
4. Assistant should demonstrate personality and expertise naturally
5. Action turns should use tools like: {', '.join(template['expected_actions'])}

Format your response as JSON:
{{
  "conversation": [
    {{"sender": "user", "text": "...", "channel": "text"}},
    {{"sender": "assistant", "text": "...", "channel": "text"}},
    {{"sender": "assistant", "text": "...", "channel": "action", "action": {{"tool": "search", "query": "..."}}}}
  ]
}}

The conversation should feel natural and showcase the character's personality while including appropriate actions."""
        
        return prompt
    
    def _format_personality(self, traits: Dict[str, float]) -> str:
        """Format personality traits for prompt"""
        if not traits:
            return "balanced personality"
        
        descriptions = []
        for trait, value in traits.items():
            if value > 0.7:
                descriptions.append(f"highly {trait}")
            elif value < 0.3:
                descriptions.append(f"low {trait}")
        
        return ", ".join(descriptions) if descriptions else "balanced personality"
    
    async def _call_llm_api(self, prompt: str) -> Dict[str, Any]:
        """Call LLM API to generate conversation"""
        try:
            response = await self.client.generate(
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response)
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            # Return fallback mock response
            return {
                "conversation": [
                    {"sender": "user", "text": "Hello, what can you help me with?"},
                    {"sender": "assistant", "text": "I'm here to assist you with any questions or tasks you might have."}
                ]
            }
    
    def _parse_llm_response(self, response: Dict[str, Any], character: Dict[str, Any]) -> List[Turn]:
        """Parse LLM response into Turn objects"""
        turns = []
        conversation = response.get('conversation', [])
        
        for turn_data in conversation:
            try:
                # Create Turn object
                turn = Turn(
                    sender=turn_data['sender'],
                    text=turn_data['text'],
                    channel=turn_data.get('channel', 'text'),
                    action=turn_data.get('action')
                )
                turns.append(turn)
                
            except Exception as e:
                logger.warning(f"Failed to parse turn: {turn_data}, error: {e}")
                continue
        
        return turns


class ConversationQualityScorer:
    """Assesses quality of generated conversations"""
    
    def __init__(self):
        self.weights = {
            'dialogue_coherence': 0.3,
            'action_integration': 0.2,
            'character_consistency': 0.3,
            'conversation_flow': 0.2
        }
    
    def score_conversation(self, sample: DatasetSample) -> Dict[str, float]:
        """Score a conversation on multiple quality dimensions"""
        scores = {
            'dialogue_coherence': self._score_coherence(sample),
            'action_integration': self._score_action_integration(sample),
            'character_consistency': self._score_character_consistency(sample),
            'conversation_flow': self._score_conversation_flow(sample)
        }
        
        # Calculate overall quality
        overall = sum(scores[key] * self.weights[key] for key in scores)
        scores['overall_quality'] = overall
        
        return scores
    
    def _score_coherence(self, sample: DatasetSample) -> float:
        """Score dialogue coherence and relevance"""
        # Simple heuristic: longer turns and varied vocabulary indicate better coherence
        turns = sample.turns
        if len(turns) < 2:
            return 0.0
        
        # Average turn length
        avg_length = sum(len(turn.text.split()) for turn in turns) / len(turns)
        length_score = min(1.0, avg_length / 15)  # Normalize around 15 words
        
        # Vocabulary diversity
        all_words = ' '.join(turn.text for turn in turns).lower().split()
        unique_words = len(set(all_words))
        diversity_score = min(1.0, unique_words / max(1, len(all_words) * 0.7))
        
        return (length_score + diversity_score) / 2
    
    def _score_action_integration(self, sample: DatasetSample) -> float:
        """Score how well actions are integrated into conversation"""
        action_turns = [turn for turn in sample.turns if turn.channel == "action"]
        total_turns = len(sample.turns)
        
        if total_turns == 0:
            return 0.0
        
        # Check if actions exist and are well-formed
        if not action_turns:
            return 0.5  # Neutral score for text-only conversations
        
        action_ratio = len(action_turns) / total_turns
        valid_actions = sum(1 for turn in action_turns if turn.action and isinstance(turn.action, dict))
        
        integration_score = valid_actions / len(action_turns) if action_turns else 0
        balance_score = min(1.0, action_ratio * 3)  # Prefer some but not excessive actions
        
        return (integration_score + balance_score) / 2
    
    def _score_character_consistency(self, sample: DatasetSample) -> float:
        """Score character consistency across turns"""
        assistant_turns = [turn for turn in sample.turns if turn.sender == "assistant"]
        
        if len(assistant_turns) < 2:
            return 0.8  # Hard to judge with few turns
        
        # Simple consistency check: similar response patterns
        response_lengths = [len(turn.text.split()) for turn in assistant_turns]
        avg_length = sum(response_lengths) / len(response_lengths)
        
        # Consistent response length suggests consistent character voice
        length_variance = sum((length - avg_length) ** 2 for length in response_lengths) / len(response_lengths)
        consistency_score = max(0.0, 1.0 - length_variance / 100)  # Lower variance = higher consistency
        
        return min(1.0, consistency_score)
    
    def _score_conversation_flow(self, sample: DatasetSample) -> float:
        """Score natural conversation flow"""
        turns = sample.turns
        if len(turns) < 3:
            return 0.7
        
        # Check for natural alternation
        alternation_score = 1.0
        for i in range(1, len(turns)):
            if turns[i].sender == turns[i-1].sender:
                alternation_score *= 0.8  # Penalize consecutive same-sender turns
        
        # Check for question-answer patterns
        qa_pattern_score = 0.0
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would']
        
        for i, turn in enumerate(turns[:-1]):
            if turn.sender == "user" and any(qw in turn.text.lower() for qw in question_words):
                next_turn = turns[i+1]
                if next_turn.sender == "assistant":
                    qa_pattern_score += 1
        
        qa_score = qa_pattern_score / max(1, len(turns) // 2)
        
        return (alternation_score + min(1.0, qa_score)) / 2


class SyntheticDataExporter:
    """Exports synthetic data to various formats"""
    
    def export_to_jsonl(self, samples: List[DatasetSample], output_file: Path) -> bool:
        """Export DatasetSample objects to JSONL format"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    # Convert to dict and write as JSON line
                    sample_dict = sample.model_dump()
                    f.write(json.dumps(sample_dict, ensure_ascii=False) + '\n')
            
            logger.info(f"Exported {len(samples)} samples to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export samples: {e}")
            return False
    
    def export_to_training_format(self, samples: List[DatasetSample], output_dir: Path) -> bool:
        """Export in format ready for training pipeline"""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export main dataset
            dataset_file = output_dir / "synthetic_dataset.jsonl"
            self.export_to_jsonl(samples, dataset_file)
            
            # Export metadata
            metadata = {
                "total_samples": len(samples),
                "generation_date": str(uuid.uuid4()),
                "format_version": "1.0"
            }
            
            metadata_file = output_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export training format: {e}")
            return False


# Additional classes for annotation and curation
class ConversationAnnotator:
    """Tools for annotating existing conversations"""
    
    def create_annotation_interface(self, raw_conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Create annotation interface for raw conversation data"""
        conversation_id = uuid.uuid4().hex[:12]
        
        turns_to_annotate = []
        for i, turn in enumerate(raw_conversation):
            turns_to_annotate.append({
                "turn_id": i,
                "role": turn.get("role", "unknown"),
                "content": turn.get("content", ""),
                "suggested_channel": "text",  # Default suggestion
                "suggested_actions": []
            })
        
        return {
            "conversation_id": conversation_id,
            "turns_to_annotate": turns_to_annotate,
            "annotation_status": "pending"
        }


class ActionAnnotationSuggester:
    """Suggests action annotations for conversation turns"""
    
    def __init__(self):
        self.action_patterns = {
            "search": [r"search", r"look up", r"find", r"query"],
            "memory_recall": [r"remember", r"recall", r"think back"],
            "analysis": [r"analyze", r"examine", r"investigate"],
            "verification": [r"verify", r"check", r"confirm"],
            "documentation": [r"record", r"document", r"note"]
        }
    
    def suggest_actions(self, text: str) -> List[Dict[str, Any]]:
        """Suggest possible actions for a text turn"""
        suggestions = []
        text_lower = text.lower()
        
        for action_type, patterns in self.action_patterns.items():
            confidence = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    confidence += 0.3
            
            if confidence > 0:
                suggestions.append({
                    "action_type": action_type,
                    "confidence": min(1.0, confidence),
                    "suggested_parameters": self._extract_parameters(text, action_type)
                })
        
        return sorted(suggestions, key=lambda x: x["confidence"], reverse=True)
    
    def _extract_parameters(self, text: str, action_type: str) -> Dict[str, str]:
        """Extract parameters for suggested action"""
        if action_type == "search":
            # Try to extract search query
            search_patterns = [r"search for (.+)", r"look up (.+)", r"find (.+)"]
            for pattern in search_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    return {"query": match.group(1).strip()}
        
        return {"description": text[:50] + "..." if len(text) > 50 else text}


class DuplicateDetector:
    """Detects and removes duplicate conversations"""
    
    def remove_duplicates(self, samples: List[DatasetSample]) -> List[DatasetSample]:
        """Remove duplicate conversations based on content similarity"""
        seen_hashes = set()
        unique_samples = []
        
        for sample in samples:
            content_hash = self._compute_content_hash(sample)
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_samples.append(sample)
        
        removed_count = len(samples) - len(unique_samples)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate conversations")
        
        return unique_samples
    
    def _compute_content_hash(self, sample: DatasetSample) -> str:
        """Compute hash based on conversation content"""
        # Create content signature from turn texts
        content_parts = []
        for turn in sample.turns:
            content_parts.append(f"{turn.sender}:{turn.text}")
        
        content_str = "|".join(content_parts)
        return hashlib.md5(content_str.encode()).hexdigest()


class DatasetBalanceAnalyzer:
    """Analyzes dataset balance and composition"""
    
    def analyze_balance(self, samples: List[DatasetSample]) -> Dict[str, float]:
        """Analyze balance between text and action channels"""
        if not samples:
            return {"text_channel_ratio": 0, "action_channel_ratio": 0, "balance_score": 0}
        
        total_turns = 0
        text_turns = 0
        action_turns = 0
        
        for sample in samples:
            for turn in sample.turns:
                if turn.sender == "assistant":  # Only count assistant turns
                    total_turns += 1
                    if turn.channel == "text":
                        text_turns += 1
                    elif turn.channel == "action":
                        action_turns += 1
        
        if total_turns == 0:
            return {"text_channel_ratio": 0, "action_channel_ratio": 0, "balance_score": 0}
        
        text_ratio = text_turns / total_turns
        action_ratio = action_turns / total_turns
        
        # Balance score: how close to ideal ratio (0.7 text, 0.3 action)
        ideal_text_ratio = 0.7
        ideal_action_ratio = 0.3
        
        text_deviation = abs(text_ratio - ideal_text_ratio)
        action_deviation = abs(action_ratio - ideal_action_ratio)
        balance_score = 1.0 - (text_deviation + action_deviation) / 2
        
        return {
            "text_channel_ratio": text_ratio,
            "action_channel_ratio": action_ratio,
            "balance_score": max(0.0, balance_score),
            "total_assistant_turns": total_turns,
            "text_turns": text_turns,
            "action_turns": action_turns
        }


# CLI interface
async def main():
    """Main CLI interface for synthetic data generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic conversation data")
    parser.add_argument("--character-file", type=str, help="JSON file with character definition")
    parser.add_argument("--output-dir", type=str, default="synthetic_output", help="Output directory")
    parser.add_argument("--count", type=int, default=10, help="Number of conversations to generate")
    parser.add_argument("--config", type=str, help="JSON config file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize generator
    generator = SyntheticDataGenerator(config=config)
    
    # Load character
    if args.character_file:
        with open(args.character_file, 'r') as f:
            character = json.load(f)
    else:
        # Use default test character
        character = {
            "name": "Research Assistant",
            "description": "A helpful AI assistant specialized in research and analysis",
            "personality_traits": {"openness": 0.8, "conscientiousness": 0.7},
            "goals": ["Help users find information", "Provide accurate analysis"],
            "tags": ["research", "helpful"],
            "scenario": "Working in a digital library"
        }
    
    # Generate conversations
    print(f"Generating {args.count} conversations...")
    samples = await generator.generate_batch([character], args.count)
    
    # Export results
    output_dir = Path(args.output_dir)
    exporter = SyntheticDataExporter()
    success = exporter.export_to_training_format(samples, output_dir)
    
    if success:
        print(f"✅ Generated {len(samples)} conversations in {output_dir}")
        
        # Show quality stats
        scorer = ConversationQualityScorer()
        scores = [scorer.score_conversation(sample) for sample in samples]
        avg_quality = sum(score['overall_quality'] for score in scores) / len(scores)
        print(f"📊 Average quality score: {avg_quality:.2f}")
        
        # Show balance analysis
        analyzer = DatasetBalanceAnalyzer()
        balance = analyzer.analyze_balance(samples)
        print(f"⚖️  Text/Action balance: {balance['text_channel_ratio']:.1%} / {balance['action_channel_ratio']:.1%}")
    else:
        print("❌ Failed to export conversations")


if __name__ == "__main__":
    asyncio.run(main()) 