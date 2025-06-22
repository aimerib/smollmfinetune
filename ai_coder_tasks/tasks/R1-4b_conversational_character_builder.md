---
# R1-4b  Conversational Character Builder
Status: **Todo**
Ring: R1
Created: 2025-01-14
---

## Goal
Transform character creation from tabbed form-filling into a guided conversational flow where AI interviews the user to discover the character's "soul."

## Context
The current R1-4a implementation uses 4 separate tabs (Profile, Personality, Goals, Examples) which creates cognitive overload and context switching. Users report feeling like they're "filling out a form" rather than "creating a character." This card implements a single-page conversational interface that feels more natural and maintains context throughout the creation process.

## Page Location
`app/pages/character_builder.py` (new page accessible from Character Management Studio)
Add new navigation option: "🗨️ Conversational Builder"

## Acceptance Criteria
- [ ] Single-page interface with conversation flow (no tabs)
- [ ] AI asks contextual questions that build on previous answers
- [ ] Real-time character preview panel shows character evolving
- [ ] Maintains all CharacterCore field population from R1-4a
- [ ] Smooth handoff to Character Management Studio for detailed editing
- [ ] Conversation state persists if user navigates away and returns
- [ ] Unit test: complete character creation flow via conversation
- [ ] Integration test: character created via conversation loads correctly in R1-4a editor

## Implementation Notes
```python
# Core conversation engine
class ConversationFlow:
    def __init__(self, core: CharacterCore):
        self.core = core
        self.questions = self._generate_adaptive_questions()
        self.current_step = 0
    
    async def ask_next_question(self) -> str:
        # Generate contextual question based on current state
        context = self._build_character_context()
        return await llm_generate_question(context, self.current_step)
    
    async def process_answer(self, answer: str):
        # Update character based on answer and advance flow
        await self._update_character_from_answer(answer)
        self.current_step += 1

# UI Layout
col1, col2 = st.columns([2, 1])
with col1:
    # Conversation interface
    display_conversation_history()
    answer = st.text_input("Your response:")
    
with col2:
    # Live character preview
    display_character_preview(core)
```

## Conversation Flow Design
1. **Opening**: "Let's create your character together. What's their name?"
2. **Core Identity**: "Tell me about [name] in your own words..."  
3. **Adaptive Deep Dive**: Based on user's description, AI chooses focus:
   - Personality exploration ("You mentioned [trait]. Can you tell me more?")
   - Background building ("What shaped [name] into who they are?")
   - Relationships ("Who matters most to [name]?")
   - Goals/Motivations ("What drives [name]?")
4. **Refinement**: "I'm sensing [name] is [summary]. Does this feel right?"
5. **Completion**: Smooth transition to full editor for detailed work

## Checklist / Steps
1. Create new `character_builder.py` page with conversation interface
2. Implement `ConversationFlow` class with adaptive questioning
3. Build real-time character preview component
4. Create conversation history display with chat-like UI
5. Integrate LLM question generation based on character state
6. Add character field population from conversation analysis
7. Implement conversation state persistence in session
8. Add navigation integration with Character Management Studio
9. Create comprehensive tests for conversation flow
10. Polish UI/UX with animations and smooth transitions

## Dependencies
- R1-4a Character Management Studio (completed)
- OpenAI client with structured outputs
- CharacterCore data model from R1-2

## References
- UX Assessment conversation: conversational vs tabbed character creation
- Nintendo DS Devkit vision from overview.md
- Character Management Studio implementation patterns 