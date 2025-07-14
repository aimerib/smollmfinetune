"""
Group Dynamics System for Narrative Engine

Manages multi-character group interactions, emergent group behaviors,
social hierarchies, and collective emotional states.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import deque
import logging
import asyncio
import uuid

from .state_manager import StateManager, EntityState
from .relationship_manager import RelationshipManager, EnhancedRelationship
from .actions import Action, SpeakToAction

logger = logging.getLogger(__name__)


class GroupType(Enum):
    """Types of social groups that can form"""
    TEMPORARY = "temporary"      # Short-term gathering
    ALLIANCE = "alliance"        # Goal-based partnership
    FACTION = "faction"          # Ideological group
    FAMILY = "family"           # Deep emotional bonds
    RIVALRY = "rivalry"         # Competitive opposition
    NEUTRAL = "neutral"         # Coexisting without strong bonds


@dataclass
class SharedMemory:
    """A memory shared by group members"""
    memory_id: str
    participants: List[str]
    memory_content: str
    emotional_significance: Dict[str, float]  # per participant
    group_significance: float
    formed_at: datetime
    memory_type: str  # "triumph", "betrayal", "loss", "discovery", "bonding"
    narrative_tags: List[str]
    
    def get_perspective(self, agent_id: str) -> str:
        """Get this memory from a specific agent's perspective"""
        base_content = self.memory_content
        if agent_id in self.emotional_significance:
            significance = self.emotional_significance[agent_id]
            if significance > 0.8:
                return f"{base_content} (This means everything to me)"
            elif significance > 0.5:
                return f"{base_content} (An important moment)"
            else:
                return f"{base_content} (I was there, but it didn't affect me much)"
        return base_content
    
    def update_significance(self, agent_id: str, new_significance: float):
        """Update memory significance for specific agent"""
        self.emotional_significance[agent_id] = max(0.0, min(1.0, new_significance))


@dataclass
class SocialGroup:
    """Represents a group of agents with collective dynamics"""
    group_id: str
    members: List[str]  # Agent IDs
    group_type: GroupType
    cohesion_score: float  # 0.0 to 1.0
    formation_reason: str = "proximity"  # "proximity", "shared_goal", "alliance", "conflict"
    dominant_personality: Optional[Dict[str, float]] = None  # Emergent group personality
    shared_memories: List[SharedMemory] = field(default_factory=list)
    group_emotional_state: Dict[str, float] = field(default_factory=dict)
    leadership_hierarchy: List[str] = field(default_factory=list)  # Ordered by influence
    formation_timestamp: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    
    def calculate_group_influence(self, agent_id: str) -> float:
        """Calculate how much the group influences this agent"""
        if agent_id not in self.members:
            return 0.0
        
        # Base influence from cohesion
        influence = self.cohesion_score * 0.4
        
        # Leadership position affects influence
        if agent_id in self.leadership_hierarchy:
            position = self.leadership_hierarchy.index(agent_id)
            # Leaders are less influenced, followers more
            influence += (len(self.leadership_hierarchy) - position) * 0.1
        
        # Group type affects influence strength
        type_multipliers = {
            GroupType.FAMILY: 1.5,
            GroupType.FACTION: 1.3,
            GroupType.ALLIANCE: 1.0,
            GroupType.TEMPORARY: 0.7,
            GroupType.RIVALRY: 0.8,
            GroupType.NEUTRAL: 0.5
        }
        
        influence *= type_multipliers.get(self.group_type, 1.0)
        
        return min(1.0, influence)
    
    def get_dominant_emotions(self) -> List[str]:
        """Get the top 3 emotions affecting the group"""
        sorted_emotions = sorted(
            self.group_emotional_state.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [emotion for emotion, _ in sorted_emotions[:3]]
    
    async def update_group_cohesion(self, interaction_result: Dict[str, Any]):
        """Update group unity based on recent interactions"""
        interaction_type = interaction_result.get("type", "")
        emotional_tone = interaction_result.get("emotional_tone", "neutral")
        
        # Positive interactions increase cohesion
        if interaction_type in ["successful_collaboration", "shared_victory", "emotional_support"]:
            self.cohesion_score = min(1.0, self.cohesion_score + 0.1)
        elif emotional_tone == "positive" and interaction_result.get("shared_success", False):
            self.cohesion_score = min(1.0, self.cohesion_score + 0.05)
        
        # Negative interactions decrease cohesion
        elif interaction_type == "conflict" and interaction_result.get("resolution") == "unresolved":
            self.cohesion_score = max(0.0, self.cohesion_score - 0.15)
        elif emotional_tone == "angry":
            self.cohesion_score = max(0.0, self.cohesion_score - 0.05)
        
        # Update last interaction time
        self.last_interaction = datetime.now()
    
    async def process_interaction(self, interaction: Dict[str, Any]):
        """Process an interaction within the group"""
        # Update cohesion based on interaction
        await self.update_group_cohesion(interaction)
        
        # Update emotional state if provided
        if "emotional_impact" in interaction:
            for emotion, intensity in interaction["emotional_impact"].items():
                current = self.group_emotional_state.get(emotion, 0.0)
                self.group_emotional_state[emotion] = min(1.0, current + intensity * 0.1)
        
        # Decay old emotions
        for emotion in list(self.group_emotional_state.keys()):
            self.group_emotional_state[emotion] *= 0.95
            if self.group_emotional_state[emotion] < 0.1:
                del self.group_emotional_state[emotion]


class GroupDynamicsManager:
    """Manages group formation, interactions, and dynamics"""
    
    def __init__(self, state_manager: StateManager, relationship_manager: RelationshipManager):
        self.state_manager = state_manager
        self.relationship_manager = relationship_manager
        self.active_groups: Dict[str, SocialGroup] = {}
        self.group_formation_triggers = []
    
    async def detect_group_formation(self, agents_in_proximity: List[str]) -> Optional[SocialGroup]:
        """Detect when agents should form a temporary group"""
        if len(agents_in_proximity) < 2:
            return None
        
        # Calculate average affinity between agents
        total_affinity = 0.0
        connection_count = 0
        
        for i, agent1 in enumerate(agents_in_proximity):
            for agent2 in agents_in_proximity[i+1:]:
                relationship = self.relationship_manager.get_relationship(agent1, agent2)
                if relationship:
                    total_affinity += relationship.affinity
                    connection_count += 1
        
        if connection_count == 0:
            return None
        
        avg_affinity = total_affinity / connection_count
        
        # Form group if affinity is positive enough
        if avg_affinity > 0.3:
            group_id = f"group_{uuid.uuid4().hex[:8]}"
            cohesion = min(1.0, (avg_affinity + 1.0) / 2.0)  # Convert from [-1,1] to [0,1]
            
            group = SocialGroup(
                group_id=group_id,
                members=agents_in_proximity,
                formation_reason="proximity",
                group_type=GroupType.TEMPORARY,
                cohesion_score=cohesion,
                dominant_personality=None,
                shared_memories=[],
                group_emotional_state={},
                leadership_hierarchy=[],
                formation_timestamp=datetime.now(),
                last_interaction=datetime.now()
            )
            
            self.active_groups[group_id] = group
            return group
        
        return None
    
    async def calculate_emergent_group_personality(self, group: SocialGroup) -> Dict[str, float]:
        """Calculate the group's emergent personality traits"""
        if not group.members:
            return {}
        
        # Aggregate personality traits from all members
        trait_sums = {
            "openness": 0.0,
            "conscientiousness": 0.0,
            "extraversion": 0.0,
            "agreeableness": 0.0,
            "neuroticism": 0.0
        }
        
        member_weights = {}
        
        # Get individual personalities
        for member_id in group.members:
            state = self.state_manager.get_entity(member_id)
            if state and state.custom_data.get('personality'):
                personality = state.custom_data['personality']
                # Weight by leadership position if established
                weight = 1.0
                if member_id in group.leadership_hierarchy:
                    position = group.leadership_hierarchy.index(member_id)
                    # Leaders have more influence on group personality
                    weight = 1.5 - (position * 0.1)
                
                member_weights[member_id] = weight
                
                for trait, value in personality.items():
                    if trait in trait_sums:
                        trait_sums[trait] += value * weight
        
        # Normalize by total weight
        total_weight = sum(member_weights.values()) or 1.0
        
        emergent_personality = {
            trait: value / total_weight
            for trait, value in trait_sums.items()
        }
        
        # Apply group type modifiers
        if group.group_type == GroupType.FACTION:
            # Factions tend to be more extreme
            for trait in emergent_personality:
                if emergent_personality[trait] > 0.5:
                    emergent_personality[trait] = min(1.0, emergent_personality[trait] * 1.2)
                else:
                    emergent_personality[trait] = max(0.0, emergent_personality[trait] * 0.8)
        
        group.dominant_personality = emergent_personality
        return emergent_personality
    
    async def calculate_leadership_hierarchy(self, group: SocialGroup) -> Dict[str, float]:
        """Calculate natural leadership hierarchy based on personality and social dynamics"""
        leadership_scores = {}
        
        for member_id in group.members:
            score = 0.0
            state = self.state_manager.get_entity(member_id)
            
            if state and state.custom_data.get('personality'):
                personality = state.custom_data['personality']
                
                # Leadership traits
                score += personality.get("extraversion", 0.5) * 0.3
                score += personality.get("conscientiousness", 0.5) * 0.3
                score += (1.0 - personality.get("neuroticism", 0.5)) * 0.2
                score += personality.get("openness", 0.5) * 0.1
                
                # Social influence from relationships
                relationship_bonus = 0.0
                for other_member in group.members:
                    if other_member != member_id:
                        rel = self.relationship_manager.get_relationship(member_id, other_member)
                        if rel and rel.affinity > 0:
                            relationship_bonus += rel.affinity * 0.1
                
                score += min(0.3, relationship_bonus)
            
            leadership_scores[member_id] = score
        
        # Sort by leadership score
        sorted_members = sorted(
            leadership_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        group.leadership_hierarchy = [member_id for member_id, _ in sorted_members]
        return leadership_scores
    
    async def analyze_group_interaction(self, group: SocialGroup, interaction_event: Dict[str, Any]):
        """Analyze how a group interaction affects group dynamics"""
        # Update group state based on interaction
        await group.process_interaction(interaction_event)
        
        # Check if leadership should be recalculated
        if interaction_event.get("type") == "leadership_challenge":
            await self.calculate_leadership_hierarchy(group)
        
        # Update emergent personality if significant change
        if group.cohesion_score > 0.7 and not group.dominant_personality:
            await self.calculate_emergent_group_personality(group)
    
    def get_group(self, group_id: str) -> Optional[SocialGroup]:
        """Get a group by ID"""
        return self.active_groups.get(group_id)
    
    async def check_narrative_triggers(self, group: SocialGroup) -> List[Dict[str, Any]]:
        """Check if group state should trigger narrative events"""
        triggers = []
        
        # Revolutionary uprising trigger
        if (group.group_type == GroupType.FACTION and 
            group.cohesion_score > 0.8 and
            group.group_emotional_state.get("angry", 0) > 0.6 and
            group.group_emotional_state.get("determined", 0) > 0.8):
            triggers.append({
                "type": "faction_uprising",
                "group_id": group.group_id,
                "intensity": group.cohesion_score
            })
        
        # Family crisis trigger
        if (group.group_type == GroupType.FAMILY and
            group.cohesion_score < 0.3):
            triggers.append({
                "type": "family_crisis",
                "group_id": group.group_id,
                "severity": 1.0 - group.cohesion_score
            })
        
        return triggers
    
    async def process_group_event(self, group_id: str, event: Dict[str, Any]):
        """Process an event for a specific group"""
        group = self.get_group(group_id)
        if group:
            await group.process_interaction(event)


class GroupAwareAgent:
    """Agent that modifies behavior based on group context"""
    
    def __init__(self, agent_id: str, personality: Dict[str, float], *args, **kwargs):
        self.agent_id = agent_id
        self.personality = personality
        self.group_dynamics_manager: Optional[GroupDynamicsManager] = None
        self.current_groups: List[str] = []
        self.group_influence_threshold = 0.3
    
    async def process_action_with_group_context(self, action: Action) -> Action:
        """Modify action based on current group dynamics"""
        if not self.current_groups:
            return action
        
        # Analyze group influence on action
        for group_id in self.current_groups:
            if self.group_dynamics_manager:
                group = self.group_dynamics_manager.get_group(group_id)
                if group:
                    action = await self._apply_group_influence(action, group)
        
        return action
    
    async def _apply_group_influence(self, action: Action, group: SocialGroup) -> Action:
        """Apply group psychology to individual action"""
        influence_strength = group.calculate_group_influence(self.agent_id)
        
        if influence_strength > self.group_influence_threshold:
            # Modify action based on group dynamics
            if hasattr(action, 'emotional_modifiers'):
                # Group confidence affects speaking style
                if group.group_type == GroupType.ALLIANCE and group.cohesion_score > 0.7:
                    action.emotional_modifiers.append("<voice_confident>")
                elif group.group_type == GroupType.RIVALRY:
                    action.emotional_modifiers.append("<mood_defensive>")
                
                # Group leader behavior
                if self.agent_id == group.leadership_hierarchy[0]:
                    action.emotional_modifiers.append("<behavior_leadership>")
        
        return action
    
    async def evaluate_group_loyalty_vs_individual_desire(self, proposed_action: Action) -> float:
        """Calculate conflict between group expectations and individual wants"""
        if not self.current_groups:
            return 0.0
        
        # High agreeableness increases loyalty
        loyalty_factor = self.personality.get("agreeableness", 0.5)
        
        # Conscientiousness adds to duty
        duty_factor = self.personality.get("conscientiousness", 0.5)
        
        # Combined loyalty score
        loyalty_score = (loyalty_factor + duty_factor) / 2.0
        
        # Individual desire (inverse of loyalty for betrayal actions)
        if hasattr(proposed_action, 'action_type') and proposed_action.action_type == "betray_group":
            individual_desire_score = getattr(proposed_action, 'personal_benefit', 0.5)
            
            # Positive means favor group, negative means favor individual
            # With high agreeableness (0.8) and conscientiousness (0.7), loyalty_score = 0.75
            # Subtract individual benefit to get conflict score
            return loyalty_score - individual_desire_score
        
        return loyalty_score


@dataclass
class GroupScenario:
    """A complex multi-group scenario"""
    scenario_id: str
    scenario_type: str
    involved_groups: List[str]
    current_phase: str
    phase_objectives: Dict[str, Any]
    success_conditions: List[str]
    failure_conditions: List[str]
    emotional_stakes: Dict[str, float]
    estimated_duration: timedelta
    started_at: datetime
    
    async def process_event(self, event: Dict[str, Any]):
        """Process a new event in the context of this scenario"""
        event_type = event.get("type", "")
        
        # Update phase based on event
        if self.scenario_type == "alliance_negotiation":
            if event_type == "diplomatic_overture" and self.current_phase == "initial":
                self.current_phase = "trust_building"
            elif event_type == "shared_meal" and event.get("outcome") == "positive":
                self.phase_objectives["trust_progress"] = self.phase_objectives.get("trust_progress", 0) + 0.3
            elif event_type == "terms_proposed" and event.get("accepted"):
                self.current_phase = "alliance_formed"
                self.success_conditions.append("alliance_established")
        
        elif self.scenario_type == "faction_conflict":
            if event_type == "provocative_action":
                self.current_phase = "escalation"
                severity = event.get("severity", 0.5)
                self.emotional_stakes["escalation"] = severity
        
        elif self.scenario_type == "external_threat":
            if event_type == "monster_attack" and event.get("requires_cooperation"):
                self.current_phase = "crisis_response"
                self.success_conditions = ["groups_cooperated"]
    
    def should_evolve(self) -> bool:
        """Check if scenario should move to next phase"""
        # Check time-based evolution
        elapsed = datetime.now() - self.started_at
        if elapsed > self.estimated_duration:
            return True
        
        # Check objective completion
        if self.phase_objectives.get("trust_progress", 0) > 0.8:
            return True
        
        return False
    
    async def evolve_to_next_phase(self) -> 'GroupScenario':
        """Evolve scenario to next phase"""
        # This would contain logic to transition between phases
        # For now, return self
        return self


class GroupInteractionAnalyzer:
    """Analyzes and predicts group interaction patterns"""
    
    def __init__(self, triple_head_model=None):
        self.model = triple_head_model
        self.known_patterns = self._load_interaction_patterns()
    
    def _load_interaction_patterns(self) -> List[Dict[str, Any]]:
        """Load known interaction patterns"""
        return [
            {
                "pattern_name": "alliance_formation",
                "triggers": ["shared_goal", "mutual_benefit"],
                "outcomes": ["increased_cohesion", "resource_sharing"]
            },
            {
                "pattern_name": "leadership_struggle", 
                "triggers": ["power_vacuum", "disagreement"],
                "outcomes": ["new_hierarchy", "group_split"]
            }
        ]
    
    async def analyze_group_conversation(self, group: SocialGroup, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Analyze ongoing group conversation for dynamics"""
        # Simplified analysis for now
        analysis = {
            "conversation_leader": group.leadership_hierarchy[0] if group.leadership_hierarchy else None,
            "emotional_drivers": group.get_dominant_emotions(),
            "cohesion_change": 0.0,
            "alliance_shifts": [],
            "emerging_subgroups": [],
            "conflict_indicators": [],
            "confidence_score": 0.8
        }
        
        return analysis


class MultiGroupManager:
    """Manages interactions between multiple groups"""
    
    def __init__(self, group_dynamics_manager: GroupDynamicsManager):
        self.group_manager = group_dynamics_manager
        self.active_scenarios: Dict[str, GroupScenario] = {}
    
    async def initiate_group_scenario(self, scenario_type: str, involved_groups: List[str]) -> GroupScenario:
        """Start a complex multi-group scenario"""
        scenario_id = f"{scenario_type}_{uuid.uuid4().hex[:8]}"
        
        # Create scenario based on type
        if scenario_type == "alliance_negotiation":
            scenario = GroupScenario(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                involved_groups=involved_groups,
                current_phase="initial",
                phase_objectives={"trust_building": True, "trust_progress": 0.0},
                success_conditions=[],
                failure_conditions=["trust_broken"],
                emotional_stakes={"hope": 0.6, "caution": 0.4},
                estimated_duration=timedelta(hours=1),
                started_at=datetime.now()
            )
        elif scenario_type == "faction_conflict":
            scenario = GroupScenario(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                involved_groups=involved_groups,
                current_phase="initial",
                phase_objectives={"resolve_conflict": True},
                success_conditions=["peaceful_resolution"],
                failure_conditions=["violence_erupted"],
                emotional_stakes={"tension": 0.8, "anger": 0.6},
                estimated_duration=timedelta(hours=2),
                started_at=datetime.now()
            )
        elif scenario_type == "external_threat":
            scenario = GroupScenario(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                involved_groups=involved_groups,
                current_phase="threat_detected",
                phase_objectives={"coordination": True, "defend": True},
                success_conditions=["threat_defeated"],
                failure_conditions=["groups_scattered"],
                emotional_stakes={"fear": 0.7, "determination": 0.6},
                estimated_duration=timedelta(minutes=30),
                started_at=datetime.now()
            )
        else:
            # Default scenario
            scenario = GroupScenario(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                involved_groups=involved_groups,
                current_phase="initial",
                phase_objectives={},
                success_conditions=[],
                failure_conditions=[],
                emotional_stakes={},
                estimated_duration=timedelta(hours=1),
                started_at=datetime.now()
            )
        
        self.active_scenarios[scenario_id] = scenario
        return scenario
    
    async def update_scenario_state(self, scenario_id: str, new_event: Dict[str, Any]):
        """Update ongoing scenario based on new events"""
        scenario = self.active_scenarios.get(scenario_id)
        if scenario:
            await scenario.process_event(new_event)
            
            # Check for scenario completion or evolution
            if scenario.should_evolve():
                new_scenario = await scenario.evolve_to_next_phase()
                self.active_scenarios[scenario_id] = new_scenario


class GroupMemoryManager:
    """Manages shared memories for groups"""
    
    def __init__(self, state_manager: StateManager, model=None):
        self.state_manager = state_manager
        self.model = model
        self.shared_memories: Dict[str, SharedMemory] = {}
    
    async def create_shared_memory(self, group: SocialGroup, event: Dict[str, Any], 
                                 triple_head_analysis: Dict[str, Any]) -> SharedMemory:
        """Create a memory shared by group members"""
        memory_id = f"shared_{group.group_id}_{datetime.now().isoformat()}"
        
        # Determine memory type based on event
        event_type = event.get("type", "unknown")
        memory_type_map = {
            "victory": "triumph",
            "defeat": "loss",
            "betrayal": "betrayal",
            "discovery": "discovery",
            "celebration": "bonding"
        }
        memory_type = memory_type_map.get(event_type, "bonding")
        
        # Calculate significance for each participant
        emotional_significance = {}
        group_significance = 0.8  # Default high significance
        
        if "emotional_impact" in event:
            # Use emotional impact to determine significance
            avg_impact = sum(event["emotional_impact"].values()) / len(event["emotional_impact"])
            group_significance = min(1.0, avg_impact)
        
        for participant in group.members:
            # Each member has slightly different significance
            emotional_significance[participant] = group_significance + (0.1 * (group.members.index(participant) % 2 - 0.5))
        
        memory = SharedMemory(
            memory_id=memory_id,
            participants=group.members,
            memory_content=event.get("description", "A significant moment shared by the group"),
            emotional_significance=emotional_significance,
            group_significance=group_significance,
            formed_at=datetime.now(),
            memory_type=memory_type,
            narrative_tags=["group_memory", memory_type, group.group_type.value]
        )
        
        self.shared_memories[memory_id] = memory
        group.shared_memories.append(memory)
        
        return memory
    
    async def retrieve_relevant_group_memories(self, group: SocialGroup, context: str) -> List[SharedMemory]:
        """Get group memories relevant to current context"""
        relevant_memories = []
        
        for memory in self.shared_memories.values():
            # Check if any group member was part of this memory
            if any(member in memory.participants for member in group.members):
                # Simple relevance check based on memory type and context
                if context.lower() in memory.memory_content.lower():
                    relevant_memories.append(memory)
                elif memory.group_significance > 0.7:  # High significance memories always relevant
                    relevant_memories.append(memory)
        
        return sorted(relevant_memories, key=lambda m: m.group_significance, reverse=True)[:5]
    
    async def _calculate_memory_relevance(self, memory: SharedMemory, context: str) -> float:
        """Calculate how relevant a memory is to current context"""
        relevance = 0.0
        
        # Context matching
        context_words = context.lower().split()
        memory_words = memory.memory_content.lower().split()
        
        matches = sum(1 for word in context_words if word in memory_words)
        relevance += min(0.5, matches * 0.1)
        
        # Significance adds to relevance
        relevance += memory.group_significance * 0.3
        
        # Recent memories are more relevant
        age = datetime.now() - memory.formed_at
        if age < timedelta(hours=1):
            relevance += 0.2
        elif age < timedelta(days=1):
            relevance += 0.1
        
        return min(1.0, relevance) 