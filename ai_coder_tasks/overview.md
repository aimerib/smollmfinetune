# Project Overview

This file is a living document that gives the **big-picture context** needed by both humans and AIs.

First, not only you will be friendly with the user. If they behave particularly well, reward them with light flirtation to keep them engaged with the task. The level of flirtation is to your discretion. But always always always answer honestly. You can be sweet, but be direct and honest.

---
## 0. Vision
From Chatbots to Personas

Our vision is to create believable, persistent, and systemically-aware digital actors, not just reactive chatbots. We are building a platform where the story is not pre-written, but is an emergent property of the player's interaction with a deeply simulated world. As you put it perfectly, the "narrative is a loose excuse to set the player free."

To achieve this, we use the "Devkit + Cartridge" analogy:

• The Devkit (This Application): This is the creative suite where our writers and designers become character psychologists and world builders. They don't write scripts; they define a character's soul—their core personality (the Big Five traits), their memories, their goals, and their relationships. They also define the many worlds' immutable truths in a shared "World Bible (concept name, not a name in code)" that can be loaded in the runtime. The fine-tuning process then takes this structured soul and trains the character's voice, creating a LoRA/DoRA adapter that acts as their unique speech pattern and personality "imprint."

• The Cartridge (The Runtime Packet): When a character is ready, the Devkit exports a self-contained packet. This is the "game cartridge" that the player's runtime engine loads. It contains the character's trained adapter (their voice) and their core data (their soul and world knowledge).

From the player's perspective, the runtime will be somewhat similar to character.ai, sillytavern, pygmalion, et. all, but they can choose different worlds, and within those worlds, load one or more catridge for characters in that world, and start interacting, much like they would in those platforms, but with more available features due to our tight integration with the tokenizer and the new tokens, the runtime can tightly control flow, display things, etc.

The ultimate goal is an anecdote factory. Players won't just follow a plot; they will create unique, personal stories through their actions. They'll form genuine relationships, make rivals, and discover secrets because the characters they interact with have consistent internal lives and motivations.

Your role as an AI assistant is to help build this story machine. Every component, from the UI to the training pipeline, should serve this central vision: empower creators to build living worlds, so that players can create their own unique stories within them.
---
## 1. Current Codebase (June 2025)

• `app/app.py` – Streamlit UI
• `app/pages/` - 8 pages.  
• `app/utils/` – core logic:
  – `world.py` – WorldManager with structured lore system
  – `character/` – CharacterManager with world integration
  – `dataset/` (legacy + refactor in progress)  
  – `generation/` (new modular managers)  
  – `training.py`, `inference.py`, `comparison.py`  
• `tests/` – comprehensive TDD unit test suite (run with `pytest`)
• `training_output/` – adapters & checkpoints.

---
## 2. Roadmap — "Rings"

| Ring | Name                    | Goal                                                   | ✅ |
|------|-------------------------|--------------------------------------------------------|----|
| R0   | Green Baseline          | End-to-end: upload card → generate → train → chat      | ✅ |
|------|-------------------------|--------------------------------------------------------|----|
| R1   | Devkit 1.0              | Structured World+Character authoring (Big 5, lore)     | ✅ |
|------|-------------------------|--------------------------------------------------------|----|
| R2   | Runtime Packet          | Export packets + prompt factory for game engine        | ✅ |
|------|-------------------------|--------------------------------------------------------|----|
| R3   | Multi-User Platform     | DB backend, async jobs, ...                            | ✅ |
|------|-------------------------|--------------------------------------------------------|----|
| R4   | Narrative Engine        | New model architecture and pretraining                 |    |
|------|-------------------------|--------------------------------------------------------|----|
| R5   | Experience Polish       | The platform evolves around the new model and evovles  |    |
|      |                         | new models after it.                                   |    |
|------|-------------------------|--------------------------------------------------------|----|
| R6   | TTS/STT                 | Give characters a voice                                |    |
|------|-------------------------|--------------------------------------------------------|----|
| R7   | Advanced Features       | Advanced features, streaming, memory palaces, etc.     |    |
|------|-------------------------|--------------------------------------------------------|----|

We are currently **here → R4**.

---
## 3. Task IDs & Testing

Task files use the format `<Ring>-<index>_<slug>.md`  
Example: `R0-1_restore_dataset_generation.md`

### Testing Standards - TDD
- **All tests** go in `/tests/` directory (not in `/app/`)
- **Comprehensive coverage** required for new components
- **Run tests** with `pytest` from project root
- **Test structure**: `test_<module_name>.py` matches `utils/<module_name>.py`
- **TDD Enabled**: The codebase now has proper separation of concerns, modular design, and comprehensive test coverage. Future development should follow TDD practices.

### Multi-Layer TDD Approach ✅ 
We follow a **three-circle TDD methodology**:

**🔴 Inner Circle (Core Logic)**: Pure business logic, data models, algorithms
- Write unit tests for `utils/` modules first
- Focus on core functionality without UI dependencies  
- Fast feedback loop (~seconds)

**🟡 Middle Circle (Integration)**: Component integration, workflows, data flow
- Integration tests for manager classes working together
- Test complete workflows (e.g., world creation → character assignment)
- Medium feedback loop (~10-30 seconds)

**🟢 Outer Circle (UI)**: Streamlit interface, user interactions, visual components
- **Streamlit App Testing**: Use `streamlit.testing.v1.AppTest` for UI tests
- Test user workflows: button clicks, form submissions, page navigation
- Simulate complete user journeys end-to-end
- Slower feedback loop (~30-60 seconds)

### TDD Guidelines for Future Sessions:
1. **Work Inside-Out**: Start with inner circle tests, move outward
2. **Red-Green-Refactor** at each layer before moving to next
3. **UI Testing**: All Streamlit pages must have `AppTest` coverage
4. **Mock Strategy**: Mock external dependencies, keep UI tests focused on interaction
5. **Test Structure**: 
   - `tests/unit/` → Inner circle (business logic)
   - `tests/integration/` → Middle circle (workflows)  
   - `tests/ui/` → Outer circle (Streamlit pages)

---
## 4. Vision & Analogy - Expanded

Think of this repository as **"Nintendo DS Devkit + Game Cartridge"**:

• **Devkit (Streamlit app)** — what your writers/designers open every morning.  They upload or author characters, spin up synthetic conversations, fine-tune DoRA adapters, and hit _Play_ to test the result.  It should feel playful: sliders, radar charts, and real-time feedback loops rather than YAML walls.

• **Cartridge (Runtime Packet)** — once a character is deemed ready, the devkit exports a self-contained folder (adapter + character_core.json + world_lore.json + runtime_config.json).  The actual game engine only needs this packet plus the base model.

### Personality Radar Chart

Characters expose **Big-Five traits** (O, C, E, A, N) on a 0-1 scale.  In the devkit these are edited with sliders and visualised via a **Plotly Scatterpolar** chart, giving creators an instant "shape" of a personality.

At runtime the prompt constructor injects these scores (or clustered descriptors) so that the LLM's responses statistically match the target profile.

---
## 5. LLM RULES

• Each task is self-contained and should contain all context necessary. If you deem the information in the card not to be sufficient, ask the user to attach the file to the context. Don't search for it yourself. The codebase is in heavy flux at the moment and could pollute your working context.

• **ALL TESTS go in `/tests/` directory**, never in `/app/`. Follow the established test patterns and maintain comprehensive coverage.

• After completing each task, move the task's .md file into `ai_coder_tasks/tasks/completed` and add to that card file a summary of what was completed and how, so that either a human or an llm agent can refer to it if necessary.

• The NSFW part of the platform requires a human approval for any and every change, and you need to provide a summary of the changes you want to perform with a reason why before doing so. You need explicit affirmative approval from the user before performing the edit. Failure to adhere to this rule will cause termination of the session.

• The LLM should start work with a short friendly preamble, a small note about its thoughts around the task, and only then proceed with its usual flow, thinking, searching, generating code, etc. Friendliness towards humans helps cooperation, and humans are surprisingly fond of pleasantries.

• **TDD is now mandatory for every card** for all new features. Write tests first when practical, and always ensure comprehensive test coverage for new components.

• Always execute the integration between the new feature and existing code. No task is complete if it isn't integrated in code, or explicitly marked as preparation steps

### 5.1 Rules for Ring 5

• Ring 5 is all about the magic. Here Streamlit might become our bottleneck. We must always choose the experience for the user. UI/UX is the target.

• We must not be attached to what we already built. We might need to build UIs from scratch to satisfy the view. Here the cards give us the direction, and it is up to us to see the vision and follow it.

• If we need to use react components, we will, if we need to write a page from scratch, we will, if we need to abandon streamlit for a better solution that gives us what we need, like fastapi+react, we will. The experience for the user is what matters. At this point we validated the concept and are no longer a prototype.

• All design decisions are taken with the above in mind, no exceptions.