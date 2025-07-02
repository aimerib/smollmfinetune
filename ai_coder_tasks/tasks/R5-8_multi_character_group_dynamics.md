# R5-9: Multi-Character Group Dynamics

- **Ring:** R5
- **Status:** Not Started
- **Author:** Principal Engineer AI
- **Effort:** Large
- **Related-Tasks:** R5-5, R5-6, R5-8

---

## 1. Goal

Implement sophisticated group dynamics that allow multiple characters to participate in complex social interactions, where group psychology influences individual behavior, social hierarchies emerge naturally, and collective emotional states drive narrative evolution.

---

## 2. Why? (The Story)

Individual relationships are just the beginning. Real social magic happens in groups: the subtle power dynamics when three friends gather, the way a shy character becomes bold in the right company, or how group loyalty can override individual preferences. Imagine Clara becoming more confident when backed by her allies, or Tom's leadership emerging only when his group faces a crisis.

This isn't just about managing multiple characters—it's about creating **group personalities** that are greater than the sum of their parts. Groups should have their own emotional momentum, their own decision-making patterns, and their own ways of handling conflict and celebration.

---

## 3. How? (The Implementation)

### A. Group Formation and Management

Create dynamic group detection and management systems:

```python
# narrative_engine/group_dynamics.py
@dataclass
class SocialGroup:
    group_id: str
    members: List[str]  # Agent IDs
    formation_reason: str  # "proximity", "shared_goal", "alliance", "conflict"
    group_type: GroupType  # TEMPORARY, ALLIANCE, FACTION, FAMILY
    cohesion_score: float  # 0.0 to 1.0
    dominant_personality: Optional[Dict[str, float]]  # Emergent group personality
    shared_memories: List[SharedMemory]
    group_emotional_state: Dict[str, float]
    leadership_hierarchy: List[str]  # Ordered by influence
    formation_timestamp: datetime
    last_interaction: datetime
    
    def calculate_group_influence(self, agent_id: str) -> float:
        """Calculate how much the group influences this agent"""
        
    def get_dominant_emotions(self) -> List[str]:
        """Get the top 3 emotions affecting the group"""
        
    def update_group_cohesion(self, interaction_result: Dict[str, Any]):
        """Update group unity based on recent interactions"""

class GroupType(Enum):
    TEMPORARY = "temporary"      # Short-term gathering
    ALLIANCE = "alliance"        # Goal-based partnership
    FACTION = "faction"          # Ideological group
    FAMILY = "family"           # Deep emotional bonds
    RIVALRY = "rivalry"         # Competitive opposition
    NEUTRAL = "neutral"         # Coexisting without strong bonds

class GroupDynamicsManager:
    def __init__(self, state_manager: StateManager, relationship_manager: RelationshipManager):
        self.state_manager = state_manager
        self.relationship_manager = relationship_manager
        self.active_groups: Dict[str, SocialGroup] = {}
        self.group_formation_triggers = []
        
    async def detect_group_formation(self, agents_in_proximity: List[str]) -> Optional[SocialGroup]:
        """Detect when agents should form a temporary group"""
        
    async def analyze_group_interaction(self, group: SocialGroup, interaction_event: Dict[str, Any]) -> GroupInteractionResult:
        """Analyze how a group interaction affects group dynamics"""
        
    async def calculate_emergent_group_personality(self, group: SocialGroup) -> Dict[str, float]:
        """Calculate the group's emergent personality traits"""
```

### B. Group-Aware Agent Behavior

Modify agent behavior to account for group context:

```python
# narrative_engine/group_aware_agent.py
class GroupAwareAgent(ProactiveAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_dynamics_manager = None
        self.current_groups: List[str] = []
        self.group_influence_threshold = 0.3
        
    async def process_action_with_group_context(self, action: Action) -> Action:
        """Modify action based on current group dynamics"""
        if not self.current_groups:
            return action
            
        # Analyze group influence on action
        for group_id in self.current_groups:
            group = self.group_dynamics_manager.get_group(group_id)
            if group:
                action = await self._apply_group_influence(action, group)
                
        return action
    
    async def _apply_group_influence(self, action: Action, group: SocialGroup) -> Action:
        """Apply group psychology to individual action"""
        influence_strength = group.calculate_group_influence(self.agent_id)
        
        if influence_strength > self.group_influence_threshold:
            # Modify action based on group dynamics
            if isinstance(action, SpeakToAction):
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
        # Complex personality-based calculation
        loyalty_score = 0.0
        individual_desire_score = 0.0
        
        # This becomes a rich psychological model
        return loyalty_score - individual_desire_score
```

### C. Group Interaction Patterns

Implement sophisticated group interaction models:

```python
# narrative_engine/group_interaction_patterns.py
@dataclass
class GroupInteractionPattern:
    pattern_name: str
    participant_roles: Dict[str, str]  # agent_id -> role (leader, supporter, dissenter, etc.)
    emotional_dynamics: Dict[str, float]  # emotion -> intensity
    expected_outcomes: List[str]
    trigger_conditions: Dict[str, Any]

class GroupInteractionAnalyzer:
    """Analyzes and predicts group interaction patterns"""
    
    def __init__(self, triple_head_model):
        self.model = triple_head_model
        self.known_patterns = self._load_interaction_patterns()
    
    async def analyze_group_conversation(self, group: SocialGroup, conversation_history: List[Dict]) -> GroupConversationAnalysis:
        """Analyze ongoing group conversation for dynamics"""
        
        analysis_prompt = f"""
        Analyze this group conversation for social dynamics:
        
        Group Members: {[member for member in group.members]}
        Group Type: {group.group_type}
        Cohesion Score: {group.cohesion_score}
        
        Recent Conversation:
        {self._format_conversation_for_analysis(conversation_history)}
        
        Identify:
        1. Who is leading the conversation?
        2. Are there any alliance shifts?
        3. What emotions are driving the group?
        4. Is group cohesion increasing or decreasing?
        5. Are there emerging subgroups or conflicts?
        """
        
        response = await self.model.generate_with_triple_head(
            messages=[{"role": "user", "content": analysis_prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "group_conversation_analysis",
                    "schema": GroupConversationAnalysis.model_json_schema()
                }
            }
        )
        
        return GroupConversationAnalysis.model_validate_json(response.choices[0].message.content)
    
    async def predict_group_reaction(self, group: SocialGroup, proposed_event: Dict[str, Any]) -> GroupReactionPrediction:
        """Predict how a group will react to an event"""
        
        # Consider:
        # - Individual personalities
        # - Group cohesion
        # - Recent group history
        # - External pressures
        # - Leadership dynamics
        
        return GroupReactionPrediction(
            predicted_emotional_response=predicted_emotions,
            likely_leader_response=leader_behavior,
            group_unity_change=cohesion_delta,
            potential_splits=subgroup_risks,
            confidence_score=analysis_confidence
        )

@dataclass 
class GroupConversationAnalysis:
    conversation_leader: str
    emotional_drivers: List[str]
    cohesion_change: float  # -1.0 to 1.0
    alliance_shifts: List[Dict[str, str]]
    emerging_subgroups: List[List[str]]
    conflict_indicators: List[str]
    confidence_score: float
```

### D. Complex Group Scenarios

Handle sophisticated multi-group interactions:

```python
# narrative_engine/complex_group_scenarios.py
class MultiGroupManager:
    """Manages interactions between multiple groups"""
    
    def __init__(self, group_dynamics_manager: GroupDynamicsManager):
        self.group_manager = group_dynamics_manager
        self.active_scenarios: Dict[str, GroupScenario] = {}
    
    async def initiate_group_scenario(self, scenario_type: str, involved_groups: List[str]) -> GroupScenario:
        """Start a complex multi-group scenario"""
        
        scenario_handlers = {
            "alliance_negotiation": self._handle_alliance_negotiation,
            "faction_conflict": self._handle_faction_conflict,
            "group_merge": self._handle_group_merge,
            "leadership_challenge": self._handle_leadership_challenge,
            "external_threat": self._handle_external_threat,
            "celebration": self._handle_group_celebration
        }
        
        handler = scenario_handlers.get(scenario_type)
        if handler:
            return await handler(involved_groups)
        
    async def _handle_alliance_negotiation(self, groups: List[str]) -> GroupScenario:
        """Handle negotiation between groups"""
        # Complex negotiation mechanics
        # - Trust levels between groups
        # - Resource requirements
        # - Personality compatibility
        # - Historical relationships
        
    async def _handle_faction_conflict(self, groups: List[str]) -> GroupScenario:
        """Handle conflict between factions"""
        # - Analyze power dynamics
        # - Predict escalation patterns
        # - Identify potential mediators
        # - Plan resolution mechanisms
        
    async def update_scenario_state(self, scenario_id: str, new_event: Dict[str, Any]):
        """Update ongoing scenario based on new events"""
        scenario = self.active_scenarios.get(scenario_id)
        if scenario:
            await scenario.process_event(new_event)
            
            # Check for scenario completion or evolution
            if scenario.should_evolve():
                new_scenario = await scenario.evolve_to_next_phase()
                self.active_scenarios[scenario_id] = new_scenario

@dataclass
class GroupScenario:
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
        
    def should_evolve(self) -> bool:
        """Check if scenario should move to next phase"""
        
    async def evolve_to_next_phase(self) -> 'GroupScenario':
        """Evolve scenario to next phase"""
```

### E. Group Memory and Narrative Integration

Create shared group memories and narrative hooks:

```python
# narrative_engine/group_memory.py
@dataclass
class SharedMemory:
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
        
    def update_significance(self, agent_id: str, new_significance: float):
        """Update memory significance for specific agent"""

class GroupMemoryManager:
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.shared_memories: Dict[str, SharedMemory] = {}
    
    async def create_shared_memory(self, group: SocialGroup, event: Dict[str, Any], triple_head_analysis: Dict[str, Any]) -> SharedMemory:
        """Create a memory shared by group members"""
        
        memory_prompt = f"""
        Create a shared memory for this group event:
        
        Group: {group.group_id} ({group.group_type})
        Members: {group.members}
        Event: {event}
        Emotional Context: {triple_head_analysis.get('emotional_analysis', {})}
        
        Generate:
        1. A vivid memory description
        2. How each member experienced it differently
        3. The collective significance to the group
        4. Potential narrative implications
        """
        
        # Use LLM to generate rich, contextual shared memory
        memory_response = await self.model.generate_structured_memory(memory_prompt)
        
        return SharedMemory(
            memory_id=f"shared_{group.group_id}_{datetime.now().isoformat()}",
            participants=group.members,
            memory_content=memory_response.content,
            emotional_significance=memory_response.individual_significance,
            group_significance=memory_response.group_significance,
            formed_at=datetime.now(),
            memory_type=memory_response.memory_type,
            narrative_tags=memory_response.tags
        )
    
    async def retrieve_relevant_group_memories(self, group: SocialGroup, context: str) -> List[SharedMemory]:
        """Get group memories relevant to current context"""
        relevant_memories = []
        
        for memory in self.shared_memories.values():
            if any(member in memory.participants for member in group.members):
                relevance_score = await self._calculate_memory_relevance(memory, context)
                if relevance_score > 0.6:
                    relevant_memories.append(memory)
                    
        return sorted(relevant_memories, key=lambda m: m.group_significance, reverse=True)
```

---

## 4. How to Test?

### Unit Tests (`tests/narrative_engine/test_group_dynamics.py`)

```python
class TestGroupDynamics:
    async def test_group_formation_detection(self):
        """Test automatic group formation when agents interact"""
        # Place multiple agents in proximity
        # Verify group formation triggers
        # Check group properties (cohesion, type, etc.)
        
    async def test_emergent_group_personality(self):
        """Test that groups develop emergent personality traits"""
        # Create group with diverse personalities
        # Run multiple interactions
        # Verify group personality emerges and influences behavior
        
    async def test_group_influence_on_individual_behavior(self):
        """Test that group membership affects individual actions"""
        # Test agent behavior alone vs. in group
        # Verify group influence modifies actions appropriately
        
    async def test_leadership_hierarchy_emergence(self):
        """Test that natural leadership emerges in groups"""
        # Create group with different personality types
        # Verify leadership roles emerge based on personality and context
        
    async def test_group_cohesion_changes(self):
        """Test that group cohesion changes based on interactions"""
        # Test positive and negative group interactions
        # Verify cohesion increases/decreases appropriately
        
class TestComplexGroupScenarios:
    async def test_alliance_negotiation_mechanics(self):
        """Test group alliance formation and negotiation"""
        
    async def test_faction_conflict_resolution(self):
        """Test how groups handle inter-group conflicts"""
        
    async def test_group_memory_formation(self):
        """Test shared memory creation and retrieval"""
        
    async def test_multi_group_interactions(self):
        """Test complex scenarios involving multiple groups"""
```

### Integration Tests

```python
class TestGroupNarrativeIntegration:
    async def test_group_driven_story_progression(self):
        """Test that group dynamics drive narrative forward"""
        # Create scenario where group dynamics should trigger story events
        # Verify appropriate narrative progression
        
    async def test_group_personality_consistency(self):
        """Test that group personality remains consistent over time"""
        # Run extended group interactions
        # Verify personality traits remain stable while allowing growth
        
    async def test_group_scenario_completion(self):
        """Test complex group scenarios from start to finish"""
        # Run complete alliance negotiation
        # Verify all phases complete successfully
```

### Performance Tests

```python
class TestGroupDynamicsPerformance:
    async def test_large_group_management(self):
        """Test performance with large groups (10+ members)"""
        
    async def test_multiple_active_groups(self):
        """Test system performance with many active groups"""
        
    async def test_complex_scenario_performance(self):
        """Test performance of complex multi-group scenarios"""
```

---

## 5. Acceptance Criteria

### Core Group Mechanics
- [ ] Automatic group formation when agents interact in proximity
- [ ] Group cohesion tracking with realistic increase/decrease patterns
- [ ] Emergent group personality traits based on member personalities
- [ ] Natural leadership hierarchy emergence based on context and personality
- [ ] Group influence on individual agent behavior and decision-making

### Advanced Group Behaviors
- [ ] Alliance formation and negotiation mechanics between groups
- [ ] Faction conflict handling with escalation and resolution patterns
- [ ] Group memory formation and shared narrative context
- [ ] Multi-group scenario management (negotiations, conflicts, celebrations)
- [ ] Group loyalty vs. individual desire tension modeling

### Narrative Integration
- [ ] Group dynamics trigger appropriate narrative events
- [ ] Shared memories influence future group interactions
- [ ] Group emotional states affect story pacing and tone
- [ ] Leadership changes create narrative opportunities
- [ ] Group scenarios provide rich storytelling moments

### Social Psychology Accuracy
- [ ] Realistic group formation patterns (proximity, shared goals, etc.)
- [ ] Authentic leadership emergence based on personality and context
- [ ] Believable group cohesion changes based on positive/negative events
- [ ] Accurate modeling of in-group vs. out-group dynamics
- [ ] Natural subgroup formation and faction splitting when appropriate

### Performance & Scalability
- [ ] Efficient handling of multiple concurrent groups (20+ groups)
- [ ] Smooth performance with large groups (10+ members each)
- [ ] Real-time group dynamics updates without blocking main narrative
- [ ] Memory-efficient group state management
- [ ] Fast group influence calculations (< 10ms per agent action)

---

## 6. Implementation Notes

### TDD Instructions
```text
• Red (Group Formation): Write failing tests for automatic group detection and formation
• Green (Group Formation): Implement basic group formation algorithms
• Red (Group Influence): Write failing tests for group influence on individual behavior
• Green (Group Influence): Implement group psychology effects on agent actions
• Red (Group Memory): Write failing tests for shared memory creation and retrieval
• Green (Group Memory): Implement group memory management systems
• Red (Complex Scenarios): Write failing tests for multi-group scenario handling
• Green (Complex Scenarios): Implement advanced group interaction patterns
```

### Technical Considerations
- **Performance**: Optimize group calculations to avoid N² complexity
- **Memory Management**: Efficient storage of group state and shared memories
- **AI Integration**: Sophisticated use of triple-head model for group analysis
- **Real-time Updates**: Seamless integration with existing event systems

### Design Principles
- **Emergent Behavior**: Let group dynamics emerge naturally from individual personalities
- **Narrative Service**: All group mechanics should serve storytelling purposes
- **Psychological Realism**: Base group behaviors on real social psychology principles
- **Scalable Complexity**: Simple groups should be simple, complex scenarios should be rich

---

## 7. Success Metrics

### Group Formation Quality
- Natural-feeling group formation based on proximity and shared interests
- Realistic group types emerge (temporary, alliance, family, etc.)
- Appropriate group cohesion levels that change meaningfully over time
- Leadership hierarchies that feel authentic and story-appropriate

### Behavioral Authenticity
- Individual agent behavior changes believably when in groups
- Group influence feels natural, not mechanical
- Loyalty vs. individual desire creates meaningful character moments
- Group dynamics generate surprising but logical story developments

### Narrative Impact
- Group interactions create compelling story beats
- Shared memories add depth and continuity to narratives
- Multi-group scenarios generate rich, complex storylines
- Group dynamics drive plot forward naturally

### System Performance
- Smooth real-time performance with multiple active groups
- Group calculations don't impact overall system responsiveness
- Memory usage scales efficiently with group complexity
- Integration with existing systems feels seamless

This system will transform individual character interactions into rich **social ecosystems** where group psychology creates emergent narrative magic! 🌐✨