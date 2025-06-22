"""
Character Intelligence Service

This service bridges character management with the sophisticated dataset generation pipeline,
providing AI-powered character creation, synthesis, and validation capabilities.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from .models import CharacterCore, Personality
from .world_character_integration import WorldCharacterIntegrator, WorldIntegrationSuggestions
from ..dataset import DatasetManager, character_analysis, prompt_generators, quality_curation
from ..dataset.content_evaluation import is_nsfw_content, categorize_nsfw_style
from ..world import WorldManager
from ..openai_client import get_client
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CharacterSynthesis:
    """Real-time character synthesis results"""
    personality_summary: str
    sample_dialogue: List[str]
    voice_consistency_score: float
    world_connections: List[str]
    character_archetype: str
    development_suggestions: List[str]
    nsfw_assessment: Dict[str, Any]
    training_readiness: float


@dataclass
class ConversationSuggestion:
    """AI-powered conversation suggestions for character building"""
    question: str
    reasoning: str
    focus_area: str  # "personality", "backstory", "relationships", "goals"
    priority: int  # 1-5, higher is more important


@dataclass
class CharacterValidation:
    """Character validation results before training"""
    is_ready_for_training: bool
    consistency_score: float
    completeness_score: float
    issues: List[str]
    recommendations: List[str]
    estimated_dataset_quality: float


@dataclass
class UserPreferenceProfile:
    """Learned user preferences for character creation"""
    personality_preferences: Dict[str, float]  # Big Five preferences
    narrative_style: str  # "dramatic", "subtle", "complex", "simple"
    character_archetypes: List[str]  # Preferred archetypes
    relationship_patterns: List[str]  # Preferred relationship types
    content_preferences: Dict[str, float]  # NSFW, violence, etc.
    total_interactions: int
    confidence_score: float


@dataclass
class EcosystemAnalysis:
    """Analysis of character ecosystem within a world"""
    character_roles: Dict[str, List[str]]  # archetype -> character names
    relationship_dynamics: Dict[str, Dict[str, int]]  # character -> {other: affinity}
    narrative_opportunities: List[str]  # Story potential
    ecosystem_gaps: List[str]  # Missing character types
    character_distribution: Dict[str, int]  # personality distribution
    world_integration_score: float


class CharacterIntelligenceService:
    """
    Advanced AI service for character creation, analysis, and dataset preparation.
    
    This service leverages the existing sophisticated dataset pipeline while providing
    enhanced UX for character creators and editors.
    """
    
    def __init__(self, world_manager: Optional[WorldManager] = None):
        self.client = get_client()
        self.dataset_manager = DatasetManager()
        self.world_manager = world_manager or WorldManager()
        self.world_integrator = WorldCharacterIntegrator(self.world_manager)
        
        # Character creation state
        self.conversation_history: List[Dict[str, str]] = []
        self.character_evolution: List[CharacterCore] = []
        
        # R1-4e: Advanced preference learning
        self.user_preferences: Dict[str, Any] = {}
        self.preference_history: List[Dict[str, Any]] = []
        self.learned_profile: Optional[UserPreferenceProfile] = None
        
        # Performance caches
        self._synthesis_cache: Dict[str, CharacterSynthesis] = {}
        self._suggestion_cache: Dict[str, List[ConversationSuggestion]] = {}
        self._ecosystem_cache: Dict[str, EcosystemAnalysis] = {}
    
    async def start_conversational_creation(self, initial_name: str = "", 
                                          world_context: Optional[str] = None) -> ConversationSuggestion:
        """
        Start the conversational character creation process.
        
        Returns the first AI question to begin the conversation.
        """
        self.conversation_history = []
        self.character_evolution = []
        
        # Create initial character core
        initial_core = CharacterCore(name=initial_name or "")
        self.character_evolution.append(initial_core)
        
        # Generate opening question based on context
        if world_context:
            question = f"I see we're creating a character for the {world_context} world. What's their name, and what's the first thing that comes to mind about them?"
        else:
            question = "Let's bring your character to life! What's their name, and tell me the first thing that comes to mind about them."
        
        suggestion = ConversationSuggestion(
            question=question,
            reasoning="Starting with name and initial impression helps establish the character's core identity",
            focus_area="personality",
            priority=5
        )
        
        self.conversation_history.append({
            "role": "assistant",
            "content": question,
            "focus_area": "introduction"
        })
        
        return suggestion
    
    async def process_conversation_response(self, user_response: str, 
                                         current_character: CharacterCore) -> Tuple[CharacterCore, ConversationSuggestion, CharacterSynthesis]:
        """
        Process user's conversational response and update character.
        
        Returns:
            - Updated character
            - Next conversation suggestion
            - Real-time character synthesis
        """
        # Record the conversation
        self.conversation_history.append({
            "role": "user", 
            "content": user_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Extract insights from response and update character
        updated_character = await self._extract_character_insights(user_response, current_character)
        self.character_evolution.append(updated_character)
        
        # Generate next conversation suggestion
        next_suggestion = await self._generate_next_conversation_question(updated_character)
        
        # Create real-time synthesis
        synthesis = await self.synthesize_character(updated_character)
        
        return updated_character, next_suggestion, synthesis
    
    async def synthesize_character(self, character: CharacterCore) -> CharacterSynthesis:
        """
        Provide real-time character synthesis showing the character "coming alive".
        """
        # Check cache first
        cache_key = self._get_character_cache_key(character)
        if cache_key in self._synthesis_cache:
            return self._synthesis_cache[cache_key]
        
        # Convert to dict for existing analysis tools
        char_dict = self._character_core_to_dict(character)
        
        # Use existing character analysis
        knowledge = character_analysis.extract_character_knowledge(char_dict)
        intimacy_style = character_analysis.analyze_character_intimacy_style(char_dict)
        
        # Generate sample dialogue using existing prompt generators
        sample_prompts = await prompt_generators.generate_exploration_prompts(
            self.client, char_dict, num_prompts=3
        )
        
        # Generate responses to show character voice
        sample_dialogue = []
        for prompt in sample_prompts[:2]:  # Just 2 samples for synthesis
            try:
                system_prompt = self.dataset_manager._create_system_prompt(char_dict)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                response = await self.client.chat_complete(
                    messages=messages, 
                    max_tokens=150, 
                    temperature=0.8
                )
                if response:
                    sample_dialogue.append(f'User: "{prompt}"\n{character.name}: {response}')
            except Exception as e:
                logger.debug(f"Error generating sample dialogue: {e}")
        
        # Analyze personality summary
        personality_summary = self._create_personality_summary(character.personality_traits)
        
        # Determine character archetype
        archetype = self._determine_character_archetype(character, knowledge)
        
        # Check world connections
        world_connections = await self._find_world_connections(character)
        
        # Voice consistency analysis
        voice_score = self._calculate_voice_consistency(character, sample_dialogue)
        
        # NSFW assessment if relevant
        nsfw_assessment = self._assess_nsfw_content(char_dict, knowledge)
        
        # Development suggestions
        suggestions = self._generate_development_suggestions(character, knowledge)
        
        # Training readiness score
        training_readiness = self._calculate_training_readiness(character, knowledge)
        
        synthesis = CharacterSynthesis(
            personality_summary=personality_summary,
            sample_dialogue=sample_dialogue,
            voice_consistency_score=voice_score,
            world_connections=world_connections,
            character_archetype=archetype,
            development_suggestions=suggestions,
            nsfw_assessment=nsfw_assessment,
            training_readiness=training_readiness
        )
        
        # Cache result
        self._synthesis_cache[cache_key] = synthesis
        
        return synthesis
    
    async def validate_character_for_training(self, character: CharacterCore) -> CharacterValidation:
        """
        Comprehensive validation of character readiness for dataset generation and training.
        """
        char_dict = self._character_core_to_dict(character)
        knowledge = character_analysis.extract_character_knowledge(char_dict)
        
        issues = []
        recommendations = []
        
        # Completeness checks
        completeness_factors = []
        
        if not character.name or len(character.name.strip()) < 2:
            issues.append("Character name is missing or too short")
            recommendations.append("Provide a clear character name")
        else:
            completeness_factors.append(1.0)
        
        if not character.description or len(character.description.strip()) < 50:
            issues.append("Character description is too brief")
            recommendations.append("Expand character description to at least 50 characters")
            completeness_factors.append(0.3)
        else:
            completeness_factors.append(1.0)
        
        if not character.goals or len(character.goals) == 0:
            issues.append("No character goals defined")
            recommendations.append("Add at least 2-3 character goals or motivations")
            completeness_factors.append(0.0)
        else:
            completeness_factors.append(min(1.0, len(character.goals) / 3))
        
        # Personality trait validation
        traits = character.personality_traits
        trait_variance = self._calculate_trait_variance(traits)
        if trait_variance < 0.1:
            issues.append("Personality traits are too uniform")
            recommendations.append("Vary personality traits to create a more interesting character")
            completeness_factors.append(0.5)
        else:
            completeness_factors.append(1.0)
        
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        # Consistency checks using existing analysis
        consistency_issues = self._check_character_consistency(character, knowledge)
        issues.extend(consistency_issues)
        
        consistency_score = max(0.0, 1.0 - (len(consistency_issues) * 0.2))
        
        # Estimate dataset quality
        estimated_quality = await self._estimate_dataset_quality(character, knowledge)
        
        # Overall readiness
        is_ready = (
            completeness_score >= 0.7 and 
            consistency_score >= 0.6 and 
            len(issues) <= 3
        )
        
        return CharacterValidation(
            is_ready_for_training=is_ready,
            consistency_score=consistency_score,
            completeness_score=completeness_score,
            issues=issues,
            recommendations=recommendations,
            estimated_dataset_quality=estimated_quality
        )
    
    async def generate_character_dataset_preview(self, character: CharacterCore, 
                                               num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Generate a preview of what the training dataset would look like.
        """
        char_dict = self._character_core_to_dict(character)
        
        # Use existing dataset generation with minimal samples
        try:
            preview_dataset = await self.dataset_manager.generate_fast_templated_dataset(
                character=char_dict,
                num_samples=num_samples,
                temperature=0.8,
                max_tokens=200
            )
            return preview_dataset
        except Exception as e:
            logger.error(f"Error generating dataset preview: {e}")
            return []
    
    async def suggest_character_enhancements(self, character: CharacterCore) -> List[ConversationSuggestion]:
        """
        Suggest specific improvements to enhance the character.
        """
        suggestions = []
        char_dict = self._character_core_to_dict(character)
        knowledge = character_analysis.extract_character_knowledge(char_dict)
        
        # Analyze gaps and suggest improvements
        if not character.backstory or len(character.backstory) < 100:
            suggestions.append(ConversationSuggestion(
                question="What shaped your character into who they are today? Tell me about a pivotal moment in their past.",
                reasoning="A rich backstory provides context for personality traits and motivations",
                focus_area="backstory",
                priority=4
            ))
        
        if len(character.relationships) < 2:
            suggestions.append(ConversationSuggestion(
                question="Who are the important people in your character's life? Tell me about their relationships.",
                reasoning="Relationships add depth and provide context for social interactions",
                focus_area="relationships", 
                priority=3
            ))
        
        if not character.appearance:
            suggestions.append(ConversationSuggestion(
                question="How would someone recognize your character in a crowd? Describe their appearance.",
                reasoning="Physical descriptions help readers/players visualize the character",
                focus_area="appearance",
                priority=2
            ))
        
        # Check for personality depth
        trait_variance = self._calculate_trait_variance(character.personality_traits)
        if trait_variance < 0.15:
            suggestions.append(ConversationSuggestion(
                question="What's something surprising or contradictory about your character that people might not expect?",
                reasoning="Complex personality traits create more interesting and realistic characters",
                focus_area="personality",
                priority=4
            ))
        
        return suggestions
    
    async def export_for_sillytavern(self, character: CharacterCore) -> Dict[str, Any]:
        """
        Export character back to SillyTavern format for testing.
        """
        # Convert CharacterCore back to SillyTavern format
        sillytavern_card = {
            "name": character.name,
            "description": character.description,
            "personality": self._personality_to_text(character.personality_traits),
            "scenario": character.scenario or "",
            "first_mes": f"*{character.name} greets you warmly*",
            "mes_example": "",  # Could be populated from examples
            "creator_notes": f"Created with Character Intelligence Service on {datetime.now().strftime('%Y-%m-%d')}",
            "system_prompt": "",
            "post_history_instructions": "",
            "alternate_greetings": [],
            "character_book": None,
            "tags": character.tags,
            "creator": "SmollM Character Studio",
            "character_version": "1.0.0"
        }
        
        # Add goals to description if present
        if character.goals:
            goals_text = " Goals: " + ", ".join(character.goals)
            sillytavern_card["description"] += goals_text
        
        return sillytavern_card
    
    def clear_conversation_state(self):
        """Clear conversation history and caches."""
        self.conversation_history = []
        self.character_evolution = []
        self._synthesis_cache = {}
        self._suggestion_cache = {}
    
    # === Private Helper Methods ===
    
    async def _extract_character_insights(self, user_response: str, current_character: CharacterCore) -> CharacterCore:
        """Extract character insights from user's conversational response."""
        # Use LLM to extract structured information from user response
        extraction_prompt = f"""
        Analyze this user response about their character and extract specific character information:
        
        User Response: "{user_response}"
        Current Character: {current_character.name}
        
        Extract and return JSON with any of these fields that can be inferred:
        - name_updates: any name changes or clarifications
        - description_additions: new description elements
        - personality_insights: personality traits observed
        - backstory_elements: backstory information
        - goals: character goals or motivations
        - relationships: relationships mentioned
        - appearance: physical descriptions
        
        Only include fields where you found actual information. Return valid JSON.
        """
        
        try:
            response = await self.client.generate(
                prompt=extraction_prompt,
                max_tokens=400,
                temperature=0.3
            )
            
            # Parse LLM response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                insights = json.loads(json_match.group())
                
                # Update character based on insights
                updated_character = CharacterCore(
                    name=insights.get('name_updates', current_character.name) or current_character.name,
                    description=self._merge_text(current_character.description, insights.get('description_additions', '')),
                    scenario=current_character.scenario,
                    backstory=self._merge_text(current_character.backstory, insights.get('backstory_elements', '')),
                    appearance=self._merge_text(current_character.appearance, insights.get('appearance', '')),
                    personality_traits=current_character.personality_traits,  # TODO: Update based on insights
                    goals=current_character.goals + insights.get('goals', []),
                    relationships=current_character.relationships,  # TODO: Parse relationships
                    tags=current_character.tags
                )
                
                return updated_character
                
        except Exception as e:
            logger.debug(f"Error extracting insights: {e}")
        
        return current_character
    
    async def _generate_next_conversation_question(self, character: CharacterCore) -> ConversationSuggestion:
        """Generate the next conversation question based on character state."""
        # Analyze what's missing or could be improved
        gaps = self._analyze_character_gaps(character)
        
        if gaps['major_gaps']:
            # Focus on major missing elements
            focus_area = gaps['major_gaps'][0]
            question_templates = {
                'description': "Tell me more about what makes your character unique. What would someone notice about them first?",
                'backstory': "What's a story from their past that really shaped who they became?",
                'goals': "What does your character want most in life? What drives them?",
                'relationships': "Who are the important people in their life? Tell me about someone they care about.",
                'personality': "What's something surprising about their personality that people might not expect?"
            }
            question = question_templates.get(focus_area, "Tell me more about your character.")
            priority = 5
        else:
            # Dive deeper into existing elements
            question = "That's interesting! What else can you tell me about them?"
            focus_area = 'depth'
            priority = 3
        
        return ConversationSuggestion(
            question=question,
            reasoning=f"Exploring {focus_area} to develop the character further",
            focus_area=focus_area,
            priority=priority
        )
    
    def _analyze_character_gaps(self, character: CharacterCore) -> Dict[str, List[str]]:
        """Analyze what's missing from the character."""
        major_gaps = []
        minor_gaps = []
        
        if not character.description or len(character.description) < 50:
            major_gaps.append('description')
        
        if not character.backstory:
            major_gaps.append('backstory')
        
        if not character.goals:
            major_gaps.append('goals')
        
        if len(character.relationships) == 0:
            minor_gaps.append('relationships')
        
        if not character.appearance:
            minor_gaps.append('appearance')
        
        return {'major_gaps': major_gaps, 'minor_gaps': minor_gaps}
    
    def _character_core_to_dict(self, character: CharacterCore) -> Dict[str, Any]:
        """Convert CharacterCore to dictionary for existing analysis tools."""
        return {
            'name': character.name,
            'description': character.description,
            'personality': self._personality_to_text(character.personality_traits),
            'scenario': character.scenario or '',
            'mes_example': '',  # Could be populated from character examples
            'backstory': character.backstory or '',
            'appearance': character.appearance or '',
            'goals': character.goals,
            'relationships': [{'name': r.name, 'affinity': r.affinity} for r in character.relationships],
            'tags': character.tags
        }
    
    def _personality_to_text(self, personality: Personality) -> str:
        """Convert Big Five traits to descriptive text."""
        traits = []
        
        if personality.openness > 0.7:
            traits.append("creative and open-minded")
        elif personality.openness < 0.3:
            traits.append("traditional and practical")
        
        if personality.conscientiousness > 0.7:
            traits.append("organized and disciplined")
        elif personality.conscientiousness < 0.3:
            traits.append("spontaneous and flexible")
        
        if personality.extraversion > 0.7:
            traits.append("outgoing and energetic")
        elif personality.extraversion < 0.3:
            traits.append("reserved and introspective")
        
        if personality.agreeableness > 0.7:
            traits.append("cooperative and trusting")
        elif personality.agreeableness < 0.3:
            traits.append("competitive and skeptical")
        
        if personality.neuroticism > 0.7:
            traits.append("emotionally sensitive")
        elif personality.neuroticism < 0.3:
            traits.append("emotionally stable")
        
        return ", ".join(traits) if traits else "balanced personality"
    
    def _create_personality_summary(self, personality: Personality) -> str:
        """Create a human-readable personality summary."""
        high_traits = []
        low_traits = []
        
        trait_names = {
            'openness': 'Openness to Experience',
            'conscientiousness': 'Conscientiousness', 
            'extraversion': 'Extraversion',
            'agreeableness': 'Agreeableness',
            'neuroticism': 'Neuroticism'
        }
        
        for trait_key, trait_name in trait_names.items():
            value = getattr(personality, trait_key)
            if value >= 0.7:
                high_traits.append(trait_name)
            elif value <= 0.3:
                low_traits.append(f"Low {trait_name}")
        
        summary_parts = []
        if high_traits:
            summary_parts.append(f"High in: {', '.join(high_traits)}")
        if low_traits:
            summary_parts.append(f"{', '.join(low_traits)}")
        
        return ". ".join(summary_parts) if summary_parts else "Balanced personality across all traits"
    
    def _determine_character_archetype(self, character: CharacterCore, knowledge: Dict[str, Any]) -> str:
        """Determine character archetype based on traits and knowledge."""
        # Simple archetype determination logic
        traits = character.personality_traits
        
        if traits.extraversion > 0.7 and traits.agreeableness > 0.7:
            return "The Helper"
        elif traits.openness > 0.7 and traits.conscientiousness < 0.4:
            return "The Creator"
        elif traits.conscientiousness > 0.7 and traits.neuroticism < 0.4:
            return "The Leader"
        elif traits.openness > 0.7 and traits.extraversion < 0.4:
            return "The Sage"
        elif traits.neuroticism > 0.6 and traits.agreeableness < 0.4:
            return "The Rebel"
        else:
            return "The Everyperson"
    
    async def _find_world_connections(self, character: CharacterCore) -> List[str]:
        """Find connections between character and world lore."""
        connections = []
        
        # Basic keyword matching for now
        # In a full implementation, this would use more sophisticated analysis
        if 'scientist' in character.description.lower():
            connections.append("Fits into the world's scientific community")
        
        if 'magic' in character.description.lower():
            connections.append("Has magical abilities relevant to world lore")
        
        return connections
    
    def _calculate_voice_consistency(self, character: CharacterCore, sample_dialogue: List[str]) -> float:
        """Calculate how consistent the character's voice is."""
        # Placeholder implementation
        # In full version, this would analyze speech patterns, vocabulary, etc.
        if len(sample_dialogue) < 2:
            return 0.5
        
        # Simple consistency check based on length and complexity
        lengths = [len(dialogue) for dialogue in sample_dialogue]
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Lower variance in response length suggests better consistency
        consistency = max(0.0, 1.0 - (length_variance / (avg_length * avg_length)))
        return min(1.0, consistency)
    
    def _assess_nsfw_content(self, char_dict: Dict[str, Any], knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Assess NSFW content using existing tools."""
        description = char_dict.get('description', '')
        personality = char_dict.get('personality', '')
        
        # Use existing NSFW analysis
        is_nsfw = is_nsfw_content(description + " " + personality)
        
        if is_nsfw:
            style = categorize_nsfw_style(description + " " + personality)
            intimacy_style = character_analysis.analyze_character_intimacy_style(char_dict)
            
            return {
                'has_nsfw_content': True,
                'nsfw_style': style,
                'intimacy_style': intimacy_style,
                'requires_careful_dataset_generation': True
            }
        else:
            return {
                'has_nsfw_content': False,
                'requires_careful_dataset_generation': False
            }
    
    def _generate_development_suggestions(self, character: CharacterCore, knowledge: Dict[str, Any]) -> List[str]:
        """Generate suggestions for character development."""
        suggestions = []
        
        if not character.backstory:
            suggestions.append("Add backstory to explain character motivations")
        
        if len(character.goals) < 2:
            suggestions.append("Define more character goals for depth")
        
        if not character.relationships:
            suggestions.append("Add relationships to create social context")
        
        trait_variance = self._calculate_trait_variance(character.personality_traits)
        if trait_variance < 0.1:
            suggestions.append("Vary personality traits for more complexity")
        
        return suggestions
    
    def _calculate_training_readiness(self, character: CharacterCore, knowledge: Dict[str, Any]) -> float:
        """Calculate how ready the character is for training."""
        readiness_factors = []
        
        # Name completeness
        readiness_factors.append(1.0 if character.name and len(character.name) > 2 else 0.0)
        
        # Description completeness
        desc_score = min(1.0, len(character.description) / 100) if character.description else 0.0
        readiness_factors.append(desc_score)
        
        # Goal completeness
        goal_score = min(1.0, len(character.goals) / 3) if character.goals else 0.0
        readiness_factors.append(goal_score)
        
        # Personality complexity
        trait_variance = self._calculate_trait_variance(character.personality_traits)
        personality_score = min(1.0, trait_variance * 10)  # Scale variance to 0-1
        readiness_factors.append(personality_score)
        
        return sum(readiness_factors) / len(readiness_factors)
    
    def _calculate_trait_variance(self, personality: Personality) -> float:
        """Calculate variance in personality traits."""
        traits = [
            personality.openness,
            personality.conscientiousness,
            personality.extraversion,
            personality.agreeableness,
            personality.neuroticism
        ]
        
        mean_trait = sum(traits) / len(traits)
        variance = sum((t - mean_trait) ** 2 for t in traits) / len(traits)
        return variance
    
    def _check_character_consistency(self, character: CharacterCore, knowledge: Dict[str, Any]) -> List[str]:
        """Check for character consistency issues."""
        issues = []
        
        # Check for contradictions between description and goals
        desc_lower = character.description.lower() if character.description else ""
        goals_lower = " ".join(character.goals).lower() if character.goals else ""
        
        # Simple contradiction detection
        if "shy" in desc_lower and "leader" in goals_lower:
            issues.append("Potential contradiction: shy character with leadership goals")
        
        if "peaceful" in desc_lower and "revenge" in goals_lower:
            issues.append("Potential contradiction: peaceful character seeking revenge")
        
        return issues
    
    async def _estimate_dataset_quality(self, character: CharacterCore, knowledge: Dict[str, Any]) -> float:
        """Estimate the quality of dataset that would be generated."""
        # Factors that affect dataset quality
        quality_factors = []
        
        # Character complexity
        complexity = self._calculate_training_readiness(character, knowledge)
        quality_factors.append(complexity)
        
        # Description richness
        desc_richness = min(1.0, len(character.description) / 200) if character.description else 0.0
        quality_factors.append(desc_richness)
        
        # Goal clarity
        goal_clarity = 1.0 if character.goals and len(character.goals) >= 2 else 0.5
        quality_factors.append(goal_clarity)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _merge_text(self, existing: str, addition: str) -> str:
        """Merge text additions intelligently."""
        if not existing:
            return addition
        if not addition:
            return existing
        
        # Simple merge - in full implementation, this would be more sophisticated
        if addition.lower() not in existing.lower():
            return f"{existing} {addition}".strip()
        return existing
    
    def _get_character_cache_key(self, character: CharacterCore) -> str:
        """Generate cache key for character."""
        # Simple hash of key character attributes
        key_data = f"{character.name}_{character.description}_{len(character.goals)}_{character.personality_traits.openness}"
        return str(hash(key_data))
    
    # R1-4e: Advanced Character Intelligence Features
    
    def track_user_preference(self, context: str, options: List[str], chosen: str, 
                            character_context: Optional[CharacterCore] = None):
        """
        Track user's AI suggestion choices to learn preferences.
        
        Args:
            context: The context of the choice (e.g., "personality_suggestion", "goal_suggestion")
            options: List of options presented to user
            chosen: The option the user selected
            character_context: Character being edited when choice was made
        """
        preference_event = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'options': options,
            'chosen': chosen,
            'character_traits': None
        }
        
        # Add character context if available
        if character_context:
            preference_event['character_traits'] = {
                'openness': character_context.personality_traits.openness,
                'conscientiousness': character_context.personality_traits.conscientiousness,
                'extraversion': character_context.personality_traits.extraversion,
                'agreeableness': character_context.personality_traits.agreeableness,
                'neuroticism': character_context.personality_traits.neuroticism,
                'archetype': self._determine_character_archetype(character_context, {})
            }
        
        self.preference_history.append(preference_event)
        
        # Update learned profile if we have enough data
        if len(self.preference_history) >= 5:  # Minimum 5 interactions
            self.learned_profile = self._analyze_user_preferences()
    
    def _analyze_user_preferences(self) -> UserPreferenceProfile:
        """Analyze user preference history to create learned profile"""
        if not self.preference_history:
            return self._create_default_profile()
        
        # Analyze personality trait preferences
        personality_prefs = defaultdict(list)
        archetype_counter = Counter()
        narrative_styles = []
        relationship_patterns = []
        
        for event in self.preference_history:
            context = event['context']
            chosen = event['chosen']
            char_traits = event.get('character_traits')
            
            # Track personality preferences
            if char_traits and 'personality' in context.lower():
                for trait, value in char_traits.items():
                    if trait != 'archetype':
                        personality_prefs[trait].append(value)
            
            # Track archetype preferences
            if char_traits and char_traits.get('archetype'):
                archetype_counter[char_traits['archetype']] += 1
            
            # Analyze narrative style from choices
            if 'complex' in chosen.lower() or 'nuanced' in chosen.lower():
                narrative_styles.append('complex')
            elif 'simple' in chosen.lower() or 'straightforward' in chosen.lower():
                narrative_styles.append('simple')
            elif 'dramatic' in chosen.lower() or 'intense' in chosen.lower():
                narrative_styles.append('dramatic')
            else:
                narrative_styles.append('subtle')
            
            # Track relationship patterns
            if 'relationship' in context.lower() or 'mentor' in chosen.lower():
                relationship_patterns.append('mentor')
            elif 'rival' in chosen.lower():
                relationship_patterns.append('rival')
            elif 'friend' in chosen.lower():
                relationship_patterns.append('friend')
        
        # Calculate personality preferences (average trait values)
        personality_preferences = {}
        for trait, values in personality_prefs.items():
            if values:
                personality_preferences[trait] = np.mean(values)
        
        # Determine dominant narrative style
        style_counter = Counter(narrative_styles)
        dominant_style = style_counter.most_common(1)[0][0] if style_counter else 'balanced'
        
        # Get preferred archetypes
        preferred_archetypes = [arch for arch, count in archetype_counter.most_common(3)]
        
        # Calculate confidence based on number of interactions
        confidence = min(1.0, len(self.preference_history) / 20.0)  # Full confidence at 20+ interactions
        
        return UserPreferenceProfile(
            personality_preferences=personality_preferences,
            narrative_style=dominant_style,
            character_archetypes=preferred_archetypes,
            relationship_patterns=list(set(relationship_patterns)),
            content_preferences={},  # Could be expanded later
            total_interactions=len(self.preference_history),
            confidence_score=confidence
        )
    
    def _create_default_profile(self) -> UserPreferenceProfile:
        """Create default preference profile for new users"""
        return UserPreferenceProfile(
            personality_preferences={},
            narrative_style='balanced',
            character_archetypes=[],
            relationship_patterns=[],
            content_preferences={},
            total_interactions=0,
            confidence_score=0.0
        )
    
    async def analyze_character_ecosystem(self, world_name: str) -> EcosystemAnalysis:
        """
        Analyze the complete character ecosystem within a world.
        
        This provides insights into character distribution, relationship dynamics,
        and narrative opportunities across all characters in the world.
        """
        # Check cache first
        if world_name in self._ecosystem_cache:
            return self._ecosystem_cache[world_name]
        
        # Get all characters in the world
        from .character import CharacterManager
        char_manager = CharacterManager(self.world_manager)
        char_manager.set_current_world(world_name)
        character_names = char_manager.list_characters_in_world(world_name)
        
        if not character_names:
            return self._create_empty_ecosystem()
        
        # Load all characters
        characters = []
        world_path = self.world_manager.get_world_path(world_name)
        for char_name in character_names:
            char_path = world_path / "characters" / char_name
            character = char_manager.load_character_core(char_path)
            if character:
                characters.append(character)
        
        # Analyze character roles and archetypes
        character_roles = defaultdict(list)
        personality_distribution = defaultdict(int)
        
        for character in characters:
            # Determine archetype
            char_dict = self._character_core_to_dict(character)
            knowledge = character_analysis.extract_character_knowledge(char_dict)
            archetype = self._determine_character_archetype(character, knowledge)
            character_roles[archetype].append(character.name)
            
            # Track personality distribution
            traits = character.personality_traits
            dominant_trait = self._get_dominant_personality_trait(traits)
            personality_distribution[dominant_trait] += 1
        
        # Analyze relationship dynamics
        relationship_dynamics = {}
        for character in characters:
            char_relationships = {}
            for rel in character.relationships:
                char_relationships[rel.name] = rel.affinity
            relationship_dynamics[character.name] = char_relationships
        
        # Identify narrative opportunities
        narrative_opportunities = await self._identify_narrative_opportunities(characters)
        
        # Identify ecosystem gaps
        ecosystem_gaps = self._identify_ecosystem_gaps(character_roles, characters)
        
        # Calculate world integration score
        world_integration_score = await self._calculate_world_integration_score(characters, world_name)
        
        analysis = EcosystemAnalysis(
            character_roles=dict(character_roles),
            relationship_dynamics=relationship_dynamics,
            narrative_opportunities=narrative_opportunities,
            ecosystem_gaps=ecosystem_gaps,
            character_distribution=dict(personality_distribution),
            world_integration_score=world_integration_score
        )
        
        # Cache the result
        self._ecosystem_cache[world_name] = analysis
        
        return analysis
    
    def _get_dominant_personality_trait(self, personality: Personality) -> str:
        """Get the most dominant personality trait"""
        traits = {
            'Openness': personality.openness,
            'Conscientiousness': personality.conscientiousness,
            'Extraversion': personality.extraversion,
            'Agreeableness': personality.agreeableness,
            'Neuroticism': personality.neuroticism
        }
        return max(traits, key=traits.get)
    
    async def _identify_narrative_opportunities(self, characters: List[CharacterCore]) -> List[str]:
        """Identify narrative opportunities based on character combinations"""
        opportunities = []
        
        # Look for complementary personality types
        high_openness = [c for c in characters if c.personality_traits.openness >= 0.7]
        low_openness = [c for c in characters if c.personality_traits.openness <= 0.3]
        
        if high_openness and low_openness:
            opportunities.append(f"Creative tension between innovative characters ({', '.join([c.name for c in high_openness[:2]])}) and traditional characters ({', '.join([c.name for c in low_openness[:2]])})")
        
        # Look for conflicting goals
        all_goals = []
        for char in characters:
            for goal in char.goals:
                all_goals.append((char.name, goal))
        
        # Simple conflict detection (could be more sophisticated)
        conflict_keywords = [('power', 'peace'), ('revenge', 'forgiveness'), ('change', 'tradition')]
        for keyword1, keyword2 in conflict_keywords:
            chars_goal1 = [name for name, goal in all_goals if keyword1 in goal.lower()]
            chars_goal2 = [name for name, goal in all_goals if keyword2 in goal.lower()]
            
            if chars_goal1 and chars_goal2:
                opportunities.append(f"Goal conflict: {keyword1} vs {keyword2} between {chars_goal1[0]} and {chars_goal2[0]}")
        
        # Look for relationship potential
        unconnected_pairs = []
        for i, char1 in enumerate(characters):
            for char2 in characters[i+1:]:
                # Check if they have any existing relationship
                has_relationship = any(rel.name == char2.name for rel in char1.relationships)
                if not has_relationship:
                    unconnected_pairs.append((char1, char2))
        
        if unconnected_pairs:
            opportunities.append(f"Unexplored relationships: {len(unconnected_pairs)} character pairs with no defined connections")
        
        return opportunities
    
    def _identify_ecosystem_gaps(self, character_roles: Dict[str, List[str]], 
                               characters: List[CharacterCore]) -> List[str]:
        """Identify missing character types in the ecosystem"""
        gaps = []
        
        # Essential narrative roles
        essential_roles = ['mentor', 'antagonist', 'comic relief', 'wise elder', 'innocent', 'rebel']
        
        existing_roles = set(character_roles.keys())
        missing_roles = [role for role in essential_roles if role not in existing_roles]
        
        for role in missing_roles:
            gaps.append(f"Missing {role} archetype - could add narrative depth")
        
        # Check personality distribution balance
        if len(characters) >= 3:
            personality_counts = defaultdict(int)
            for char in characters:
                dominant = self._get_dominant_personality_trait(char.personality_traits)
                personality_counts[dominant] += 1
            
            total_chars = len(characters)
            for trait, count in personality_counts.items():
                if count / total_chars > 0.6:  # More than 60% have same dominant trait
                    gaps.append(f"Personality imbalance: {count}/{total_chars} characters are primarily {trait}")
        
        return gaps
    
    async def _calculate_world_integration_score(self, characters: List[CharacterCore], 
                                               world_name: str) -> float:
        """Calculate how well characters are integrated with world lore"""
        if not characters:
            return 0.0
        
        integration_scores = []
        
        for character in characters:
            try:
                integration = await self.world_integrator.analyze_character_world_fit(character, world_name)
                integration_scores.append(integration.integration_score)
            except Exception as e:
                logger.debug(f"Failed to analyze world integration for {character.name}: {e}")
                integration_scores.append(0.5)  # Default score
        
        return np.mean(integration_scores) if integration_scores else 0.0
    
    def _create_empty_ecosystem(self) -> EcosystemAnalysis:
        """Create empty ecosystem analysis for worlds with no characters"""
        return EcosystemAnalysis(
            character_roles={},
            relationship_dynamics={},
            narrative_opportunities=["Create your first character to start building the world's story"],
            ecosystem_gaps=["No characters exist yet - unlimited potential"],
            character_distribution={},
            world_integration_score=0.0
        )
    
    async def suggest_character_based_on_ecosystem(self, world_name: str) -> ConversationSuggestion:
        """
        Suggest a new character based on ecosystem analysis and user preferences.
        
        This combines ecosystem gaps with learned user preferences to suggest
        the most valuable character addition.
        """
        ecosystem = await self.analyze_character_ecosystem(world_name)
        
        # Prioritize ecosystem gaps
        if ecosystem.ecosystem_gaps:
            gap = ecosystem.ecosystem_gaps[0]  # Most important gap
            
            # Adapt suggestion based on user preferences if available
            if self.learned_profile and self.learned_profile.confidence_score > 0.3:
                preferred_style = self.learned_profile.narrative_style
                
                if 'missing' in gap.lower() and 'archetype' in gap.lower():
                    archetype = gap.split('Missing ')[1].split(' archetype')[0]
                    
                    question = f"Based on your world's character ecosystem, I notice you're missing a {archetype} character. Given your preference for {preferred_style} narratives, what kind of {archetype} would you like to create?"
                    
                    return ConversationSuggestion(
                        question=question,
                        reasoning=f"Ecosystem analysis shows gap in {archetype} role, adapted for user's {preferred_style} style preference",
                        focus_area="archetype",
                        priority=5
                    )
        
        # Default suggestion if no specific gaps
        return ConversationSuggestion(
            question="Looking at your world's characters, what kind of character would create the most interesting story possibilities?",
            reasoning="General ecosystem-aware character creation prompt",
            focus_area="ecosystem",
            priority=3
        )
    
    def get_user_preference_insights(self) -> Dict[str, Any]:
        """Get insights about learned user preferences"""
        if not self.learned_profile:
            return {"status": "insufficient_data", "interactions": len(self.preference_history)}
        
        insights = {
            "status": "learned",
            "confidence": self.learned_profile.confidence_score,
            "total_interactions": self.learned_profile.total_interactions,
            "narrative_style": self.learned_profile.narrative_style,
            "preferred_archetypes": self.learned_profile.character_archetypes,
            "personality_tendencies": {}
        }
        
        # Add personality insights
        for trait, value in self.learned_profile.personality_preferences.items():
            if value > 0.6:
                insights["personality_tendencies"][trait] = f"tends_high ({value:.2f})"
            elif value < 0.4:
                insights["personality_tendencies"][trait] = f"tends_low ({value:.2f})"
            else:
                insights["personality_tendencies"][trait] = f"balanced ({value:.2f})"
        
        return insights 