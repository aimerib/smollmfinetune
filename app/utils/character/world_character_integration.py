"""
World-Character Integration Service

Advanced AI-powered service for integrating characters with world lore,
including timeline placement, faction analysis, and NPC relationship suggestions.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .models import CharacterCore, Personality, Relationship
from ..world import WorldManager, WorldLore
from ..openai_client import get_client

logger = logging.getLogger(__name__)


@dataclass
class TimelineConnection:
    """Suggested connection between character and world timeline"""
    event_year: int
    event_description: str
    character_involvement: str
    relevance_score: float
    suggested_age: Optional[int] = None


@dataclass
class FactionRecommendation:
    """Faction membership recommendation"""
    faction_name: str
    compatibility_score: float
    reasoning: str
    suggested_role: str
    potential_conflicts: List[str]


@dataclass
class NPCRelationshipSuggestion:
    """Suggested relationship with existing NPC"""
    npc_name: str
    relationship_type: str  # "mentor", "rival", "friend", "romantic", "family"
    affinity_score: int  # -10 to +10
    backstory: str
    interaction_history: str


@dataclass
class WorldLoreProposal:
    """Proposed addition to world lore based on character"""
    category: str  # "timeline", "faction", "place", "fact"
    title: str
    description: str
    justification: str
    impact_assessment: str


@dataclass
class WorldIntegrationSuggestions:
    """Complete world integration analysis"""
    timeline_connections: List[TimelineConnection]
    faction_recommendations: List[FactionRecommendation]
    npc_relationships: List[NPCRelationshipSuggestion]
    proposed_world_updates: List[WorldLoreProposal]
    consistency_warnings: List[str]
    integration_score: float


class WorldCharacterIntegrator:
    """Advanced service for integrating characters with world lore"""
    
    def __init__(self, world_manager: WorldManager):
        self.world_manager = world_manager
        self.client = get_client()
    
    async def analyze_character_world_fit(self, character: CharacterCore, 
                                        world_name: str) -> WorldIntegrationSuggestions:
        """
        Comprehensive analysis of how character fits into world lore
        """
        # Load world lore
        world_lore = self.world_manager.load_world(world_name)
        if not world_lore:
            return self._create_empty_suggestions()
        
        # Analyze different aspects in parallel
        timeline_task = self._suggest_timeline_placement(character, world_lore)
        faction_task = self._analyze_faction_compatibility(character, world_lore)
        npc_task = self._suggest_npc_connections(character, world_lore)
        lore_task = self._generate_lore_expansions(character, world_lore)
        consistency_task = self._check_world_conflicts(character, world_lore)
        
        # Execute all analyses
        timeline_connections, faction_recommendations, npc_relationships, \
        proposed_updates, consistency_warnings = await asyncio.gather(
            timeline_task, faction_task, npc_task, lore_task, consistency_task
        )
        
        # Calculate overall integration score
        integration_score = self._calculate_integration_score(
            timeline_connections, faction_recommendations, 
            npc_relationships, consistency_warnings
        )
        
        return WorldIntegrationSuggestions(
            timeline_connections=timeline_connections,
            faction_recommendations=faction_recommendations,
            npc_relationships=npc_relationships,
            proposed_world_updates=proposed_updates,
            consistency_warnings=consistency_warnings,
            integration_score=integration_score
        )
    
    async def _suggest_timeline_placement(self, character: CharacterCore, 
                                        world_lore: WorldLore) -> List[TimelineConnection]:
        """Suggest where character fits in world timeline"""
        if not world_lore.timeline:
            return []
        
        # Build context for AI analysis
        character_context = f"""
        Character: {character.name}
        Description: {character.description}
        Goals: {', '.join(character.goals)}
        Personality: {self._personality_summary(character.personality_traits)}
        Backstory: {character.backstory or 'Not specified'}
        """
        
        timeline_context = "\n".join([
            f"Year {event.year}: {event.event}"
            for event in world_lore.timeline[-10:]  # Last 10 events
        ])
        
        prompt = f"""Analyze how this character could be connected to the world's timeline:

{character_context}

World Timeline:
{timeline_context}

For each relevant timeline event, suggest:
1. How the character might have been involved
2. What age they would have been
3. How it shaped their current personality/goals
4. Relevance score (0.0-1.0)

Respond in JSON format:
{{
    "connections": [
        {{
            "event_year": 2150,
            "event_description": "The Great War begins",
            "character_involvement": "Served as a field medic, saw the horrors of war",
            "relevance_score": 0.8,
            "suggested_age": 25
        }}
    ]
}}"""

        try:
            response = await self.client.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.7
            )
            
            # Parse JSON response
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                connections = []
                
                for conn in data.get('connections', []):
                    connections.append(TimelineConnection(
                        event_year=conn.get('event_year', 0),
                        event_description=conn.get('event_description', ''),
                        character_involvement=conn.get('character_involvement', ''),
                        relevance_score=conn.get('relevance_score', 0.0),
                        suggested_age=conn.get('suggested_age')
                    ))
                
                return connections
                
        except Exception as e:
            logger.error(f"Timeline analysis failed: {e}")
        
        return []
    
    async def _analyze_faction_compatibility(self, character: CharacterCore, 
                                           world_lore: WorldLore) -> List[FactionRecommendation]:
        """Analyze character compatibility with world factions"""
        if not world_lore.factions:
            return []
        
        character_context = f"""
        Character: {character.name}
        Description: {character.description}
        Goals: {', '.join(character.goals)}
        Personality: {self._personality_summary(character.personality_traits)}
        """
        
        factions_context = "\n".join([
            f"Faction: {faction.name}\nHistory: {', '.join([f'{e.year}: {e.event}' for e in faction.timeline])}"
            for faction in world_lore.factions
        ])
        
        prompt = f"""Analyze which factions this character would be compatible with:

{character_context}

Available Factions:
{factions_context}

For each faction, assess:
1. Compatibility score (0.0-1.0)
2. Reasoning for compatibility/incompatibility
3. Suggested role within faction
4. Potential conflicts or challenges

Respond in JSON format:
{{
    "recommendations": [
        {{
            "faction_name": "The Scholars Guild",
            "compatibility_score": 0.9,
            "reasoning": "High openness and conscientiousness align with scholarly pursuits",
            "suggested_role": "Research Coordinator",
            "potential_conflicts": ["May clash with rigid hierarchy"]
        }}
    ]
}}"""

        try:
            response = await self.client.generate(
                prompt=prompt,
                max_tokens=600,
                temperature=0.7
            )
            
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                recommendations = []
                
                for rec in data.get('recommendations', []):
                    recommendations.append(FactionRecommendation(
                        faction_name=rec.get('faction_name', ''),
                        compatibility_score=rec.get('compatibility_score', 0.0),
                        reasoning=rec.get('reasoning', ''),
                        suggested_role=rec.get('suggested_role', ''),
                        potential_conflicts=rec.get('potential_conflicts', [])
                    ))
                
                return recommendations
                
        except Exception as e:
            logger.error(f"Faction analysis failed: {e}")
        
        return []
    
    async def _suggest_npc_connections(self, character: CharacterCore, 
                                     world_lore: WorldLore) -> List[NPCRelationshipSuggestion]:
        """Suggest relationships with existing NPCs"""
        npcs = []
        for place in world_lore.places:
            npcs.extend(place.npcs)
        
        if not npcs:
            return []
        
        character_context = f"""
        Character: {character.name}
        Description: {character.description}
        Goals: {', '.join(character.goals)}
        Personality: {self._personality_summary(character.personality_traits)}
        """
        
        npcs_context = "\n".join([
            f"NPC: {npc.name} - {npc.description}"
            for npc in npcs[:10]  # Limit to 10 NPCs
        ])
        
        prompt = f"""Suggest relationships between this character and existing NPCs:

{character_context}

Available NPCs:
{npcs_context}

For each relevant NPC, suggest:
1. Relationship type (mentor, rival, friend, romantic, family)
2. Affinity score (-10 to +10)
3. Backstory of their connection
4. Brief interaction history

Respond in JSON format:
{{
    "relationships": [
        {{
            "npc_name": "Master Elara",
            "relationship_type": "mentor",
            "affinity_score": 8,
            "backstory": "Taught character advanced techniques",
            "interaction_history": "Weekly training sessions for 2 years"
        }}
    ]
}}"""

        try:
            response = await self.client.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.7
            )
            
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                relationships = []
                
                for rel in data.get('relationships', []):
                    relationships.append(NPCRelationshipSuggestion(
                        npc_name=rel.get('npc_name', ''),
                        relationship_type=rel.get('relationship_type', ''),
                        affinity_score=rel.get('affinity_score', 0),
                        backstory=rel.get('backstory', ''),
                        interaction_history=rel.get('interaction_history', '')
                    ))
                
                return relationships
                
        except Exception as e:
            logger.error(f"NPC relationship analysis failed: {e}")
        
        return []
    
    async def _generate_lore_expansions(self, character: CharacterCore, 
                                      world_lore: WorldLore) -> List[WorldLoreProposal]:
        """Generate proposals for expanding world lore based on character"""
        character_context = f"""
        Character: {character.name}
        Description: {character.description}
        Goals: {', '.join(character.goals)}
        Backstory: {character.backstory or 'Not specified'}
        """
        
        prompt = f"""Based on this character, suggest new additions to world lore:

{character_context}

Suggest new world lore elements that this character's existence implies:
1. New timeline events
2. New factions or organizations
3. New locations
4. New world facts

For each suggestion, provide:
- Category (timeline/faction/place/fact)
- Title
- Description
- Justification for why this character implies this lore
- Impact assessment

Respond in JSON format:
{{
    "proposals": [
        {{
            "category": "timeline",
            "title": "The Academy Reforms (2155)",
            "description": "Educational system restructured to focus on practical skills",
            "justification": "Character's background suggests formal education system",
            "impact_assessment": "Affects all scholarly characters and institutions"
        }}
    ]
}}"""

        try:
            response = await self.client.generate(
                prompt=prompt,
                max_tokens=600,
                temperature=0.8
            )
            
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                proposals = []
                
                for prop in data.get('proposals', []):
                    proposals.append(WorldLoreProposal(
                        category=prop.get('category', ''),
                        title=prop.get('title', ''),
                        description=prop.get('description', ''),
                        justification=prop.get('justification', ''),
                        impact_assessment=prop.get('impact_assessment', '')
                    ))
                
                return proposals
                
        except Exception as e:
            logger.error(f"Lore expansion analysis failed: {e}")
        
        return []
    
    async def _check_world_conflicts(self, character: CharacterCore, 
                                   world_lore: WorldLore) -> List[str]:
        """Check for conflicts between character and world lore"""
        conflicts = []
        
        # Simple conflict detection
        char_desc_lower = character.description.lower() if character.description else ""
        
        # Check against world facts
        for key, value in world_lore.facts.items():
            if key.lower() in char_desc_lower:
                # This is a very basic check - in practice, would be more sophisticated
                if "no magic" in value.lower() and "magic" in char_desc_lower:
                    conflicts.append(f"Character mentions magic, but world fact '{key}' suggests: {value}")
        
        return conflicts
    
    def _calculate_integration_score(self, timeline_connections: List[TimelineConnection],
                                   faction_recommendations: List[FactionRecommendation],
                                   npc_relationships: List[NPCRelationshipSuggestion],
                                   consistency_warnings: List[str]) -> float:
        """Calculate overall world integration score"""
        score = 0.0
        factors = 0
        
        # Timeline integration
        if timeline_connections:
            avg_relevance = sum(conn.relevance_score for conn in timeline_connections) / len(timeline_connections)
            score += avg_relevance * 0.3
            factors += 0.3
        
        # Faction compatibility
        if faction_recommendations:
            avg_compatibility = sum(rec.compatibility_score for rec in faction_recommendations) / len(faction_recommendations)
            score += avg_compatibility * 0.3
            factors += 0.3
        
        # NPC relationships
        if npc_relationships:
            score += 0.2  # Having relationships is good
            factors += 0.2
        
        # Consistency penalty
        consistency_score = max(0.0, 1.0 - (len(consistency_warnings) * 0.2))
        score += consistency_score * 0.2
        factors += 0.2
        
        return score / factors if factors > 0 else 0.5
    
    def _personality_summary(self, personality: Personality) -> str:
        """Create readable personality summary"""
        traits = []
        if personality.openness >= 0.7:
            traits.append("creative and open-minded")
        if personality.conscientiousness >= 0.7:
            traits.append("organized and disciplined")
        if personality.extraversion >= 0.7:
            traits.append("outgoing and energetic")
        if personality.agreeableness >= 0.7:
            traits.append("cooperative and trusting")
        if personality.neuroticism >= 0.7:
            traits.append("emotionally sensitive")
        
        return ", ".join(traits) if traits else "balanced personality"
    
    def _create_empty_suggestions(self) -> WorldIntegrationSuggestions:
        """Create empty suggestions when world lore is not available"""
        return WorldIntegrationSuggestions(
            timeline_connections=[],
            faction_recommendations=[],
            npc_relationships=[],
            proposed_world_updates=[],
            consistency_warnings=["No world lore available for analysis"],
            integration_score=0.0
        ) 