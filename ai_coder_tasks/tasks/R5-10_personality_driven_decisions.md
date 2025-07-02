# R5-10: Personality-Driven Decision Making

- **Ring:** R5
- **Status:** Not Started
- **Related-Tasks:** R5-5, R5-6, R5-7, R5-9, R4-6

---

## 1. Goal

Implement a sophisticated personality-driven decision making system that uses Big Five personality traits, emotional states, relationship contexts, and group dynamics to generate authentic character choices that feel psychologically consistent and narratively compelling.

---

## 2. Why? (The Story)

This is where psychology meets storytelling magic. Characters should make decisions that feel **inevitable** given who they are, yet still surprise us with their complexity. Imagine Clara's high Openness leading her to trust a mysterious stranger, but her low Agreeableness making her suspicious of their motives. Or Tom's Conscientiousness driving him to keep a promise even when it conflicts with his group's wishes.

We're building a **digital psychology engine** where every choice emerges from the deep structure of personality, filtered through emotional states, shaped by relationships, and influenced by social context. No more random responses—every decision should feel like it came from a living mind.

---

## 3. How? (The Implementation)

### A. Core Decision Architecture

Build a multi-layered decision making system:

```python
# narrative_engine/personality_decision_engine.py
@dataclass
class DecisionContext:
    agent_id: str
    available_choices: List[ChoiceOption]
    situation_description: str
    emotional_state: Dict[str, float]
    relationship_context: Dict[str, Any]  # Relevant relationships
    group_context: Optional[SocialGroup]
    time_pressure: float  # 0.0 to 1.0
    stakes_level: float   # 0.0 to 1.0 
    moral_dimensions: List[str]  # ["honesty", "loyalty", "fairness", etc.]
    
@dataclass
class ChoiceOption:
    choice_id: str
    description: str
    predicted_outcomes: List[str]
    personality_alignment: Dict[str, float]  # Big Five alignment scores
    emotional_cost: Dict[str, float]
    relationship_impact: Dict[str, float]  # Impact on specific relationships
    moral_weight: Dict[str, float]  # Alignment with moral dimensions
    group_acceptance: float  # How group would view this choice
    
@dataclass
class PersonalityWeights:
    """How each Big Five trait influences different decision types"""
    openness_weights: Dict[str, float]      # exploration, creativity, change
    conscientiousness_weights: Dict[str, float]  # duty, planning, persistence
    extraversion_weights: Dict[str, float]  # social, assertive, excitement
    agreeableness_weights: Dict[str, float] # cooperation, trust, altruism
    neuroticism_weights: Dict[str, float]   # anxiety, stress, emotion

class PersonalityDecisionEngine:
    def __init__(self, triple_head_model, relationship_manager: RelationshipManager):
        self.model = triple_head_model
        self.relationship_manager = relationship_manager
        self.personality_weights = self._load_personality_weights()
        self.decision_history = {}
        
    async def make_personality_driven_decision(self, context: DecisionContext) -> DecisionResult:
        """Generate a decision based on personality psychology"""
        
        # 1. Analyze agent's personality profile
        agent_personality = await self._get_agent_personality(context.agent_id)
        
        # 2. Score each choice against personality
        choice_scores = await self._score_choices_by_personality(
            context.available_choices, 
            agent_personality,
            context
        )
        
        # 3. Apply emotional modifiers
        choice_scores = await self._apply_emotional_influence(
            choice_scores, 
            context.emotional_state,
            agent_personality
        )
        
        # 4. Factor in relationship considerations
        choice_scores = await self._apply_relationship_influence(
            choice_scores,
            context.relationship_context,
            agent_personality
        )
        
        # 5. Account for group pressure/influence
        if context.group_context:
            choice_scores = await self._apply_group_influence(
                choice_scores,
                context.group_context,
                agent_personality
            )
        
        # 6. Generate final decision with reasoning
        decision = await self._generate_final_decision(
            choice_scores, 
            context,
            agent_personality
        )
        
        # 7. Record decision for personality consistency tracking
        await self._record_decision(context.agent_id, decision, context)
        
        return decision

    async def _score_choices_by_personality(self, choices: List[ChoiceOption], personality: Dict[str, float], context: DecisionContext) -> Dict[str, float]:
        """Score each choice based on personality alignment"""
        
        scores = {}
        
        for choice in choices:
            personality_score = 0.0
            
            # Openness to Experience
            if personality.get('openness', 0.5) > 0.6:
                # High openness: prefers novel, creative choices
                personality_score += choice.personality_alignment.get('openness_appeal', 0) * 0.2
            else:
                # Low openness: prefers familiar, conventional choices
                personality_score += choice.personality_alignment.get('conventional_appeal', 0) * 0.2
                
            # Conscientiousness
            if personality.get('conscientiousness', 0.5) > 0.6:
                # High conscientiousness: values duty, planning, responsibility
                personality_score += choice.personality_alignment.get('duty_alignment', 0) * 0.2
                personality_score += choice.personality_alignment.get('long_term_benefit', 0) * 0.15
            else:
                # Low conscientiousness: more impulsive, present-focused
                personality_score += choice.personality_alignment.get('immediate_gratification', 0) * 0.2
                
            # Extraversion
            if personality.get('extraversion', 0.5) > 0.6:
                # High extraversion: social, assertive choices
                personality_score += choice.personality_alignment.get('social_engagement', 0) * 0.2
                personality_score += choice.personality_alignment.get('assertiveness', 0) * 0.15
            else:
                # Low extraversion (introversion): private, reflective choices
                personality_score += choice.personality_alignment.get('privacy_preference', 0) * 0.2
                
            # Agreeableness  
            if personality.get('agreeableness', 0.5) > 0.6:
                # High agreeableness: cooperative, trusting, altruistic
                personality_score += choice.personality_alignment.get('cooperation', 0) * 0.2
                personality_score += choice.personality_alignment.get('altruism', 0) * 0.15
            else:
                # Low agreeableness: competitive, skeptical, self-interested
                personality_score += choice.personality_alignment.get('self_interest', 0) * 0.2
                personality_score += choice.personality_alignment.get('competitive_advantage', 0) * 0.15
                
            # Neuroticism
            if personality.get('neuroticism', 0.5) > 0.6:
                # High neuroticism: anxiety-driven, risk-averse choices
                personality_score -= choice.emotional_cost.get('anxiety', 0) * 0.3
                personality_score -= choice.personality_alignment.get('risk_level', 0) * 0.2
            else:
                # Low neuroticism (emotional stability): calm, risk-tolerant
                personality_score += choice.personality_alignment.get('risk_tolerance', 0) * 0.1
                
            scores[choice.choice_id] = personality_score
            
        return scores
```

### B. Emotional Decision Modifiers

Account for how emotions influence personality expression:

```python
# narrative_engine/emotional_decision_modifiers.py
class EmotionalDecisionModifier:
    """Modifies personality-based decisions using current emotional state"""
    
    async def apply_emotional_influence(self, choice_scores: Dict[str, float], emotional_state: Dict[str, float], personality: Dict[str, float]) -> Dict[str, float]:
        """Modify choice scores based on current emotions"""
        
        modified_scores = choice_scores.copy()
        
        # Anger increases impulsivity, decreases agreeableness temporarily
        if emotional_state.get('angry', 0) > 0.5:
            anger_intensity = emotional_state['angry']
            for choice_id, score in modified_scores.items():
                # Angry people make more aggressive, less cooperative choices
                if self._choice_involves_confrontation(choice_id):
                    modified_scores[choice_id] += anger_intensity * 0.3
                if self._choice_involves_cooperation(choice_id):
                    modified_scores[choice_id] -= anger_intensity * 0.2
                    
        # Fear increases risk aversion across all personality types
        if emotional_state.get('fearful', 0) > 0.5:
            fear_intensity = emotional_state['fearful']
            for choice_id, score in modified_scores.items():
                risk_level = self._assess_choice_risk(choice_id)
                modified_scores[choice_id] -= fear_intensity * risk_level * 0.4
                
        # Joy increases openness and extraversion temporarily
        if emotional_state.get('happy', 0) > 0.6:
            joy_intensity = emotional_state['happy']
            for choice_id, score in modified_scores.items():
                if self._choice_involves_social_interaction(choice_id):
                    modified_scores[choice_id] += joy_intensity * 0.2
                if self._choice_involves_novelty(choice_id):
                    modified_scores[choice_id] += joy_intensity * 0.15
                    
        # Sadness decreases energy, increases need for comfort/support
        if emotional_state.get('sad', 0) > 0.5:
            sadness_intensity = emotional_state['sad']
            for choice_id, score in modified_scores.items():
                if self._choice_provides_comfort(choice_id):
                    modified_scores[choice_id] += sadness_intensity * 0.25
                if self._choice_requires_high_energy(choice_id):
                    modified_scores[choice_id] -= sadness_intensity * 0.3
                    
        return modified_scores
    
    def _choice_involves_confrontation(self, choice_id: str) -> bool:
        """Determine if choice involves confrontational behavior"""
        # Use LLM analysis or predefined tags
        
    def _assess_choice_risk(self, choice_id: str) -> float:
        """Assess risk level of choice (0.0 to 1.0)"""
        # Use LLM analysis to evaluate potential negative outcomes
```

### C. Relationship-Influenced Decisions

Factor in how relationships affect choices:

```python
# narrative_engine/relationship_decision_influence.py
class RelationshipDecisionInfluence:
    """Modifies decisions based on relationship considerations"""
    
    def __init__(self, relationship_manager: RelationshipManager):
        self.relationship_manager = relationship_manager
        
    async def apply_relationship_influence(self, choice_scores: Dict[str, float], relationship_context: Dict[str, Any], personality: Dict[str, float]) -> Dict[str, float]:
        """Modify choice scores based on relationship implications"""
        
        modified_scores = choice_scores.copy()
        
        for choice_id, base_score in choice_scores.items():
            relationship_modifier = 0.0
            
            # Analyze impact on each relevant relationship
            for other_agent_id, relationship_data in relationship_context.items():
                if isinstance(relationship_data, dict):
                    relationship = EnhancedRelationship.from_dict(relationship_data)
                    
                    # Choice impact on this relationship
                    choice_impact = await self._analyze_choice_relationship_impact(
                        choice_id, other_agent_id, relationship, personality
                    )
                    
                    # Weight by relationship importance
                    relationship_importance = self._calculate_relationship_importance(
                        relationship, personality
                    )
                    
                    relationship_modifier += choice_impact * relationship_importance
                    
            modified_scores[choice_id] = base_score + relationship_modifier
            
        return modified_scores
    
    async def _analyze_choice_relationship_impact(self, choice_id: str, other_agent_id: str, relationship: EnhancedRelationship, personality: Dict[str, float]) -> float:
        """Analyze how a choice would affect a specific relationship"""
        
        impact_analysis_prompt = f"""
        Analyze how this choice would affect the relationship:
        
        Choice: {choice_id}
        Relationship Status: {relationship.status}
        Current Affinity: {relationship.affinity}
        Recent Emotions: {relationship.emotional_history[-3:]}
        
        Consider:
        1. Would this choice strengthen or weaken the relationship?
        2. How would the other person likely react?
        3. Does this align with the relationship's history?
        
        Rate the impact from -1.0 (very negative) to 1.0 (very positive).
        """
        
        # Use LLM for nuanced relationship impact analysis
        response = await self.model.analyze_relationship_impact(impact_analysis_prompt)
        return response.impact_score
    
    def _calculate_relationship_importance(self, relationship: EnhancedRelationship, personality: Dict[str, float]) -> float:
        """Calculate how much this relationship should influence decisions"""
        
        importance = 0.0
        
        # Base importance on relationship strength and type
        importance += abs(relationship.affinity) * 0.4
        importance += relationship.memory_significance * 0.3
        
        # Personality modifiers
        agreeableness = personality.get('agreeableness', 0.5)
        importance += agreeableness * 0.3  # Agreeable people weight relationships more
        
        # Relationship status modifiers
        status_weights = {
            "Family": 0.9,
            "Best Friend": 0.8, 
            "Romantic": 0.85,
            "Close Friend": 0.7,
            "Friend": 0.5,
            "Acquaintance": 0.2,
            "Rival": 0.6,  # Rivals still influence decisions significantly
            "Enemy": 0.4,
            "Stranger": 0.1
        }
        
        importance *= status_weights.get(relationship.status, 0.3)
        
        return min(importance, 1.0)
```

### D. Decision Consistency and Growth

Track decisions over time for personality consistency:

```python
# narrative_engine/decision_consistency_tracker.py
@dataclass
class DecisionRecord:
    decision_id: str
    agent_id: str
    context: DecisionContext
    chosen_option: ChoiceOption
    personality_scores: Dict[str, float]
    final_score: float
    reasoning: str
    timestamp: datetime
    outcomes: List[str]  # Actual outcomes (filled in later)
    regret_level: float  # How much the agent regrets this decision (0.0 to 1.0)

class DecisionConsistencyTracker:
    """Tracks decision patterns to ensure personality consistency"""
    
    def __init__(self):
        self.decision_history: Dict[str, List[DecisionRecord]] = {}
        self.personality_drift_threshold = 0.3
        
    async def analyze_decision_consistency(self, agent_id: str, recent_decisions: int = 10) -> PersonalityConsistencyReport:
        """Analyze if recent decisions are consistent with agent's personality"""
        
        if agent_id not in self.decision_history:
            return PersonalityConsistencyReport(consistency_score=1.0, analysis="No decision history")
            
        recent_decisions_list = self.decision_history[agent_id][-recent_decisions:]
        
        # Analyze patterns in decision making
        consistency_scores = []
        
        for decision in recent_decisions_list:
            # Compare decision against expected personality behavior
            expected_score = await self._calculate_expected_personality_score(
                decision.context, decision.chosen_option, agent_id
            )
            actual_score = decision.final_score
            
            # Calculate consistency (how close actual choice was to expected)
            consistency = 1.0 - abs(expected_score - actual_score)
            consistency_scores.append(consistency)
            
        overall_consistency = sum(consistency_scores) / len(consistency_scores)
        
        # Identify any concerning patterns
        concerning_patterns = await self._identify_concerning_patterns(recent_decisions_list)
        
        return PersonalityConsistencyReport(
            consistency_score=overall_consistency,
            concerning_patterns=concerning_patterns,
            recommendations=await self._generate_consistency_recommendations(agent_id, overall_consistency)
        )
    
    async def suggest_personality_growth(self, agent_id: str) -> PersonalityGrowthSuggestion:
        """Suggest how personality might evolve based on experiences"""
        
        decision_history = self.decision_history.get(agent_id, [])
        if len(decision_history) < 5:
            return PersonalityGrowthSuggestion(growth_areas=[], reasoning="Insufficient decision history")
            
        # Analyze patterns that might indicate personality change
        growth_patterns = await self._analyze_growth_patterns(decision_history)
        
        # Suggest realistic personality evolution
        # (People don't change drastically, but experiences can shift traits slightly)
        
        return PersonalityGrowthSuggestion(
            growth_areas=growth_patterns.growth_opportunities,
            reasoning=growth_patterns.evidence,
            suggested_trait_adjustments=growth_patterns.trait_deltas  # Small adjustments like +0.1 to openness
        )

@dataclass
class PersonalityConsistencyReport:
    consistency_score: float  # 0.0 to 1.0
    concerning_patterns: List[str]
    recommendations: List[str]
    analysis: str

@dataclass 
class PersonalityGrowthSuggestion:
    growth_areas: List[str]
    reasoning: str
    suggested_trait_adjustments: Dict[str, float]
```

---

## 4. How to Test?

### Unit Tests (`tests/narrative_engine/test_personality_decisions.py`)

```python
class TestPersonalityDecisionEngine:
    async def test_high_openness_prefers_novel_choices(self):
        """Test that high openness agents choose novel/creative options"""
        # Create agent with high openness (0.8)
        # Present choice between familiar and novel options
        # Verify novel option is preferred
        
    async def test_conscientiousness_drives_duty_based_decisions(self):
        """Test that conscientious agents prioritize duty and responsibility"""
        # Create agent with high conscientiousness
        # Present conflict between fun and duty
        # Verify duty is chosen despite personal cost
        
    async def test_emotional_state_modifies_personality_expression(self):
        """Test that emotions temporarily modify personality-based choices"""
        # Test same agent making decisions in different emotional states
        # Verify choices change predictably based on emotions
        
    async def test_relationship_influence_on_decisions(self):
        """Test that relationship considerations affect choices"""
        # Create decision that impacts important relationships
        # Verify relationship-aware choice modification
        
    async def test_decision_consistency_over_time(self):
        """Test that agents make consistent decisions over multiple interactions"""
        # Run series of similar decisions
        # Verify personality consistency maintained
        
class TestEmotionalDecisionModification:
    async def test_anger_increases_confrontational_choices(self):
        """Test that anger makes agents more likely to choose confrontation"""
        
    async def test_fear_increases_risk_aversion(self):
        """Test that fear makes agents avoid risky choices"""
        
    async def test_joy_increases_social_openness(self):
        """Test that happiness makes agents more socially open"""

class TestRelationshipInfluence:
    async def test_high_affinity_relationships_strongly_influence_decisions(self):
        """Test that close relationships have strong decision influence"""
        
    async def test_agreeable_agents_weight_relationships_more_heavily(self):
        """Test that agreeable personalities prioritize relationship harmony"""
        
    async def test_relationship_conflict_resolution_patterns(self):
        """Test how agents handle decisions that conflict with relationship needs"""
```

### Integration Tests

```python
class TestPersonalityDecisionIntegration:
    async def test_complete_decision_workflow(self):
        """Test full decision making process from context to choice"""
        # Create complex decision context
        # Verify full pipeline produces reasonable choice
        # Check all influence factors are properly weighted
        
    async def test_group_and_relationship_interaction(self):
        """Test how group pressure and relationship influence interact"""
        # Create scenario with conflicting group and relationship pressures
        # Verify realistic resolution based on personality
        
    async def test_personality_growth_over_narrative_arc(self):
        """Test gradual personality evolution through experiences"""
        # Run extended narrative with challenging decisions
        # Verify subtle personality growth occurs naturally
```

---

## 5. Acceptance Criteria

### Core Decision Psychology
- [ ] Big Five personality traits drive decision preferences predictably
- [ ] Emotional states modify personality expression realistically
- [ ] Relationship considerations influence choices based on relationship strength
- [ ] Group pressure affects decisions according to personality and social context
- [ ] Decision patterns remain consistent with established personality profiles

### Advanced Psychological Modeling
- [ ] Personality trait interactions create nuanced decision patterns
- [ ] Moral reasoning reflects personality-based value systems
- [ ] Time pressure and stakes level appropriately influence decision quality
- [ ] Individual decision history creates recognizable character patterns
- [ ] Personality growth occurs gradually through meaningful experiences

### Narrative Integration
- [ ] Decisions feel authentic and surprising simultaneously
- [ ] Character choices drive story forward naturally
- [ ] Personality-driven conflicts create compelling narrative tension
- [ ] Decision consequences feed back into personality development
- [ ] Character arcs emerge organically from decision patterns

### Consistency and Growth
- [ ] Decision consistency tracker identifies personality inconsistencies
- [ ] Personality growth suggestions feel psychologically realistic
- [ ] Character development maintains core identity while allowing evolution
- [ ] Decision patterns create recognizable "personality signatures"
- [ ] Long-term character consistency maintained across narrative arcs

### Performance & Usability
- [ ] Decision analysis completes in < 500ms for complex scenarios
- [ ] Memory usage scales efficiently with decision history
- [ ] Decision reasoning is transparent and debuggable
- [ ] Personality weights are tunable for different narrative needs
- [ ] System gracefully handles edge cases and conflicting influences

---

## 6. Implementation Notes

### TDD Instructions
```text
• Red (Personality Scoring): Write failing tests for Big Five trait influence on choice preferences
• Green (Personality Scoring): Implement basic personality-based choice scoring
• Red (Emotional Modifiers): Write failing tests for emotional state influence on decisions
• Green (Emotional Modifiers): Implement emotional decision modification systems
• Red (Relationship Influence): Write failing tests for relationship-based choice modification
• Green (Relationship Influence): Implement relationship consideration algorithms
• Red (Consistency Tracking): Write failing tests for decision pattern analysis
• Green (Consistency Tracking): Implement personality consistency monitoring
```

### Technical Considerations
- **Performance**: Optimize decision calculations for real-time interaction
- **Psychological Accuracy**: Base algorithms on validated personality psychology research
- **Narrative Balance**: Ensure predictability doesn't eliminate surprise
- **Memory Management**: Efficiently store and query decision history

### Design Principles
- **Psychology-First**: Ground all decision logic in established personality theory
- **Emergent Complexity**: Let complex behaviors emerge from simple personality rules
- **Narrative Service**: Every decision should serve character development and story
- **Authentic Surprise**: Maintain character authenticity while allowing unexpected choices

---

## 7. Success Metrics

### Psychological Authenticity
- Character decisions feel genuinely motivated by personality
- Emotional influences on choices feel realistic and nuanced
- Relationship considerations create believable social dynamics
- Personality consistency maintained while allowing for growth

### Narrative Quality
- Decisions create compelling character moments and story beats
- Character arcs emerge naturally from personality-driven choices
- Decision conflicts generate meaningful narrative tension
- Character growth feels earned through experience rather than arbitrary

### System Performance
- Decision analysis processes in real-time during interactions
- Personality tracking scales effectively with long narrative arcs
- Decision history provides useful insights for character development
- Integration with other systems (relationships, groups, emotions) feels seamless

This system will transform our characters from reactive chatbots into **psychologically authentic digital beings** whose every choice reveals the depth of their inner lives! 🧠✨ 