---
# R1-4c  Live Character Synthesis  
Status: **Todo**
Ring: R1
Created: 2025-01-14
---

## Goal
Create a real-time character preview system that shows the character "coming alive" as users build them, with dynamic personality visualization and auto-generated sample dialogue.

## Context
Current character creation feels static - users fill forms without seeing how changes affect the character's personality or voice. This card adds a live preview panel that synthesizes all character data into a coherent, evolving representation. Users will see personality traits emerge, sample dialogue generate, and character consistency maintained in real-time.

## Page Location
Component for use in both `character_management.py` (R1-4a) and `character_builder.py` (R1-4b)
New component: `components/character_synthesis_preview.py`

## Acceptance Criteria
- [ ] Real-time character preview panel updates as any field changes
- [ ] Auto-generated sample dialogue reflects current personality traits
- [ ] Dynamic personality insights show trait interpretations
- [ ] Character consistency warnings when fields conflict
- [ ] Visual personality "shape" evolves with trait changes
- [ ] Preview includes world context integration
- [ ] Performance: updates complete within 200ms of field changes
- [ ] Unit test: character preview accuracy across different personality profiles
- [ ] Integration test: preview correctly reflects changes from both conversation and form interfaces

## Implementation Notes
```python
# Core synthesis engine
class CharacterSynthesizer:
    def __init__(self, core: CharacterCore, world_context: Dict):
        self.core = core
        self.world_context = world_context
        
    async def synthesize(self) -> CharacterSynthesis:
        # Generate real-time insights
        personality_summary = self._analyze_personality()
        sample_dialogue = await self._generate_sample_dialogue()
        consistency_check = self._check_consistency()
        world_connections = self._find_world_connections()
        
        return CharacterSynthesis(
            personality_summary=personality_summary,
            sample_dialogue=sample_dialogue,
            consistency_warnings=consistency_check,
            world_connections=world_connections,
            character_archetype=self._determine_archetype()
        )

# UI Component
def render_character_synthesis_preview(core: CharacterCore, key_prefix: str = ""):
    with st.container():
        st.markdown(f"### 🎭 {core.name or 'Your Character'} (Evolving...)")
        
        # Sample dialogue section
        synthesis = st.cache_data(synthesize_character)(core)
        if synthesis.sample_dialogue:
            st.markdown("**Character Voice Preview:**")
            st.markdown(f'> "{synthesis.sample_dialogue}"  \n> — {core.name}')
        
        # Personality insights
        st.markdown("**Personality Insights:**")
        render_personality_synthesis(core.personality_traits)
        
        # Emerging goals and connections
        render_emerging_patterns(synthesis)
```

## Feature Specifications

### Real-Time Dialogue Generation
- Generate contextual sample dialogue based on current traits
- Update dialogue when personality sliders change
- Show different dialogue styles (greeting, conflict, curiosity)
- Include character-specific speech patterns and vocabulary

### Dynamic Personality Insights
- Translate Big Five scores into human-readable insights
- Show personality "shape" using visual metaphors
- Highlight dominant traits and their implications
- Suggest character archetypes based on trait combinations

### Consistency Monitoring
- Detect conflicts between description and personality traits
- Flag goals that don't align with personality
- Suggest resolutions for inconsistencies
- Visual indicators for character coherence level

### World Integration Preview
- Show relevant world lore connections
- Suggest character placement in world timeline
- Highlight potential relationships with existing NPCs
- Preview character's role in world factions

## Checklist / Steps
1. Create `CharacterSynthesizer` class with real-time analysis
2. Implement sample dialogue generation based on personality
3. Build personality insights translator (Big Five → human language)
4. Add consistency checking across all character fields
5. Create world context integration for character placement
6. Design visual personality "shape" representation
7. Implement caching strategy for performance optimization
8. Add character archetype determination algorithm
9. Create responsive UI component with smooth animations
10. Integrate with both conversation builder and form editor
11. Add comprehensive testing for synthesis accuracy
12. Polish visual design and micro-interactions

## Dependencies
- R1-4a Character Management Studio (for form integration)
- R1-4b Conversational Character Builder (for conversation integration)
- CharacterCore data model from R1-2
- World management system for context integration
- OpenAI client for dialogue generation

## References
- Character psychology principles for trait interpretation
- World lore integration patterns
- Real-time UI update patterns in Streamlit
- Character consistency methodologies from game design 