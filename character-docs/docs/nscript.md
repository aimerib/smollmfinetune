# N-Script: The Narrative Scripting Engine

**From ELI5 to Expert: A Complete Guide to Triple-Head Narrative Scripting**

## Table of Contents

1. [What is N-Script? (ELI5)](#what-is-n-script-eli5)
2. [The Triple-Head Architecture](#the-triple-head-architecture)
3. [Getting Started](#getting-started)
4. [Basic Concepts](#basic-concepts)
5. [Trigger Types Reference](#trigger-types-reference)
6. [Action Types Reference](#action-types-reference)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Technical Reference](#technical-reference)
10. [Expert Tips & Tricks](#expert-tips--tricks)

---

## What is N-Script? (ELI5)

Imagine you're writing a story, but instead of just writing what happens, you can also control:
- **What the characters think and feel** (emotions, mood)
- **What they remember** (important moments, relationships)
- **How they tell the story** (dramatic, funny, mysterious)

N-Script is like having three different "storytellers" working together:

1. **🎭 The Actor** (Generation Head) - Decides what to say and how to say it
2. **💭 The Emotions** (Control Head) - Manages feelings and body language
3. **🧠 The Memory** (Memory Head) - Remembers important moments and relationships

When something happens in your story (like a player entering a room), N-Script can automatically trigger all three storytellers to work together, creating rich, believable character interactions.

---

## The Triple-Head Architecture

The power of N-Script comes from its **Triple-Head Architecture** - three specialized AI "heads" that work together:

### 🎭 Generation Head
- **What it does**: Creates the actual dialogue and narrative text
- **Controls**: Speaking style, tone, narrative focus
- **Example**: "Make the dragon sound ancient and menacing"

### 💭 Control Head
- **What it does**: Manages emotions, mood, and physical expressions
- **Controls**: Emotional states, control tokens, mood transitions
- **Example**: "Make the character feel intimidated and show it with body language"

### 🧠 Memory Head
- **What it does**: Decides what gets remembered and how memories influence behavior
- **Controls**: Memory formation, importance, emotional impact
- **Example**: "This moment is so important it will be remembered forever"

### 🎼 Coordination
When all three heads work together, you get:
- **Emotionally consistent** characters
- **Memorable** story moments
- **Natural feeling** interactions
- **Rich narrative** depth

---

## Getting Started

### Your First N-Script

Let's create a simple script that makes a guard greet players when they enter a village:

```yaml
script_id: my_first_script
trigger:
  type: ON_ENTER_LOCATION
  location_id: "village_square"
  actor_filter: "player"

actions:
  - type: TRIPLE_HEAD_ACTION
    target_agent_id: "village_guard"
    generation_params:
      style: "friendly_greeting"
      tone: "welcoming"
    control_params:
      emotions: ["welcoming", "friendly"]
      control_tokens: ["<expression_smile>"]
    memory_params:
      importance: 0.3
      tags: ["greeting", "village"]
    action:
      type: SpeakToAction
      message: "Welcome to our village, traveler!"
```

**What this does:**
1. **Trigger**: When player enters "village_square"
2. **Generation**: Guard speaks in a friendly, welcoming way
3. **Control**: Guard feels welcoming and smiles
4. **Memory**: Guard remembers greeting the visitor (low importance)
5. **Action**: Guard says the welcome message

---

## Basic Concepts

### Script Structure

Every N-Script has three main parts:

```yaml
script_id: unique_name_for_this_script

trigger:
  # When should this script activate?
  
actions:
  # What should happen when it activates?
```

### Triggers: When Things Happen

Triggers tell N-Script **when** to activate. Think of them as "event listeners":

- **Location Triggers**: When someone enters/leaves a place
- **Emotional Triggers**: When someone's mood changes
- **Memory Triggers**: When important memories are formed
- **Coordination Triggers**: When the AI heads need to sync up

### Actions: What Should Happen

Actions tell N-Script **what to do** when triggered:

- **TRIPLE_HEAD_ACTION**: Coordinate all three heads
- **CONTROL_INJECTION**: Just change emotions/mood
- **MEMORY_FORMATION**: Force a specific memory
- **GENERATION_OVERRIDE**: Change speaking style temporarily
- **HEAD_SYNCHRONIZATION**: Make all heads work together

---

## Trigger Types Reference

### 1. ON_ENTER_LOCATION
**When**: A character enters a specific location
**Use for**: Greetings, location-specific events, ambushes

```yaml
trigger:
  type: ON_ENTER_LOCATION
  location_id: "dragon_lair"
  actor_filter: "player"  # Optional: only for specific characters
  once: true              # Optional: only trigger once
```

### 2. ON_EMOTIONAL_STATE
**When**: A character's emotional state changes
**Use for**: Emotional support, mood-based reactions, character development

```yaml
trigger:
  type: ON_EMOTIONAL_STATE
  target_agent_id: "player"
  conditions:
    emotional_state: "angry|frustrated"  # Multiple states with |
    mood_intensity: "> 0.6"              # Intensity threshold
```

### 3. ON_MEMORY_FORMATION
**When**: A significant memory is being formed
**Use for**: Narrative emphasis, pivotal moments, story beats

```yaml
trigger:
  type: ON_MEMORY_FORMATION
  conditions:
    memory_significance: "> 0.8"
    emotional_impact: "> 0.7"
```

### 4. ON_GENERATION_QUALITY
**When**: The generation head produces high/low quality output
**Use for**: Quality control, adaptive storytelling

```yaml
trigger:
  type: ON_GENERATION_QUALITY
  conditions:
    quality_score: "> 0.85"
```

### 5. ON_HEAD_COORDINATION
**When**: The three heads achieve specific coordination patterns
**Use for**: Synchronized narrative moments, harmony checks

```yaml
trigger:
  type: ON_HEAD_COORDINATION
  conditions:
    generation_quality: "> 0.8"
    control_consistency: "> 0.7"
    memory_relevance: "> 0.6"
```

---

## Action Types Reference

### 1. TRIPLE_HEAD_ACTION
**The Swiss Army Knife** - Coordinates all three heads for complex interactions

```yaml
- type: TRIPLE_HEAD_ACTION
  target_agent_id: "character_name"
  generation_params:
    style: "dramatic_monologue"
    tone: "mysterious"
    narrative_focus: "character_revelation"
  control_params:
    emotions: ["mysterious", "confident"]
    mood_shift: "contemplative"
    control_tokens: ["<voice_low>", "<eyes_distant>"]
  memory_params:
    importance: 0.9
    emotional_impact: 0.8
    tags: ["revelation", "important"]
    content: "Revealed a crucial secret"
  action:
    type: SpeakToAction
    message: "There's something you need to know..."
```

### 2. CONTROL_INJECTION
**Quick Emotion Changes** - Just modify emotions/mood without full coordination

```yaml
- type: CONTROL_INJECTION
  target_agent_id: "companion"
  control_params:
    emotions: ["concerned", "protective"]
    mood_shift: "alert"
    control_tokens: ["<stance_defensive>", "<expression_worried>"]
```

### 3. MEMORY_FORMATION
**Force Specific Memories** - Make sure something gets remembered

```yaml
- type: MEMORY_FORMATION
  target_agent_id: "player"
  memory_params:
    importance: 0.95
    emotional_impact: 0.9
    tags: ["turning_point", "character_growth"]
    content: "The moment everything changed"
```

### 4. GENERATION_OVERRIDE
**Change Speaking Style** - Temporarily modify how characters express themselves

```yaml
- type: GENERATION_OVERRIDE
  target_agent_id: "narrator"
  generation_params:
    style: "poetic"
    tone: "melancholic"
    narrative_focus: "inner_thoughts"
    override_duration: 5  # Next 5 responses
```

### 5. HEAD_SYNCHRONIZATION
**Perfect Harmony** - Make all heads work in perfect coordination

```yaml
- type: HEAD_SYNCHRONIZATION
  target_agent_id: "main_character"
  coordination_params:
    sync_emotional_state: true
    align_memory_focus: true
    harmonize_narrative_tone: true
```

---

## Advanced Features

### Conditional Logic

You can use complex conditions in triggers:

```yaml
trigger:
  type: ON_EMOTIONAL_STATE
  conditions:
    emotional_state: "angry|frustrated|betrayed"
    mood_intensity: "> 0.8"
    # Custom conditions
    player_relationship: "< 0.3"  # Low relationship
    story_chapter: "2|3|4"        # Specific chapters
```

### Multi-Action Scripts

Chain multiple actions for complex behaviors:

```yaml
actions:
  # First, change the character's mood
  - type: CONTROL_INJECTION
    target_agent_id: "companion"
    control_params:
      emotions: ["determined", "protective"]
  
  # Then, have them speak with that emotion
  - type: TRIPLE_HEAD_ACTION
    target_agent_id: "companion"
    generation_params:
      style: "determined_declaration"
    control_params:
      emotions: ["brave", "loyal"]
    memory_params:
      importance: 0.7
      tags: ["loyalty", "protection"]
    action:
      type: SpeakToAction
      message: "I won't let anyone hurt you!"
  
  # Finally, form a memory for the player
  - type: MEMORY_FORMATION
    target_agent_id: "player"
    memory_params:
      importance: 0.6
      tags: ["companion_loyalty", "protection"]
      content: "My companion stood up for me"
```

### Dynamic Parameters

Use variables and conditions for dynamic scripting:

```yaml
# Example: Different reactions based on relationship level
actions:
  - type: TRIPLE_HEAD_ACTION
    target_agent_id: "love_interest"
    generation_params:
      style: "{{relationship > 0.8 ? 'intimate_conversation' : 'friendly_chat'}}"
      tone: "{{emotional_state == 'happy' ? 'playful' : 'sincere'}}"
    control_params:
      emotions: ["{{relationship > 0.5 ? 'affectionate' : 'friendly'}}"]
```

---

## Best Practices

### 1. Start Simple, Build Complex
```yaml
# ✅ Good: Start with basic triple-head actions
- type: TRIPLE_HEAD_ACTION
  target_agent_id: "character"
  generation_params:
    style: "casual"
  control_params:
    emotions: ["friendly"]
  memory_params:
    importance: 0.5

# ❌ Avoid: Don't start with complex coordination
```

### 2. Use Meaningful IDs and Tags
```yaml
# ✅ Good: Descriptive IDs and tags
script_id: dragon_first_encounter_intimidation
memory_params:
  tags: ["first_encounter", "dragon", "intimidation", "pivotal"]

# ❌ Avoid: Generic names
script_id: script1
tags: ["stuff", "thing"]
```

### 3. Balance Memory Importance
```yaml
# ✅ Good: Reserve high importance for truly pivotal moments
memory_params:
  importance: 0.95  # Only for life-changing events
  importance: 0.7   # Important character moments
  importance: 0.3   # Casual interactions

# ❌ Avoid: Everything can't be max importance
memory_params:
  importance: 1.0   # Don't overuse this!
```

### 4. Layer Emotions Naturally
```yaml
# ✅ Good: Natural emotion combinations
control_params:
  emotions: ["curious", "cautious"]        # Makes sense together
  emotions: ["angry", "hurt", "betrayed"]  # Natural progression

# ❌ Avoid: Contradictory emotions
control_params:
  emotions: ["happy", "sad", "angry"]  # Confusing mix
```

### 5. Use Control Tokens Appropriately
```yaml
# ✅ Good: Enhance the narrative
control_tokens: ["<voice_whisper>", "<lean_closer>"]  # Supports intimacy
control_tokens: ["<eyes_widen>", "<step_back>"]       # Shows surprise

# ❌ Avoid: Overusing or inappropriate tokens
control_tokens: ["<eyes_glow>", "<voice_echo>", "<hair_float>"]  # Too much
```

---

## Technical Reference

### Performance Considerations

**Memory Usage**: N-Scripts are loaded into memory. Keep scripts focused and avoid massive files.

**Trigger Efficiency**: Location triggers are fastest, memory triggers are most expensive.

**Action Complexity**: TRIPLE_HEAD_ACTION is most expensive, CONTROL_INJECTION is fastest.

### Integration Points

```python
# Loading scripts in your code
from narrative_engine.nscript import ScriptManager

script_manager = ScriptManager()
script = script_manager.load_script_from_file("path/to/script.nscript")
script_manager.add_script(script)

# Monitoring triggers
trigger_monitor = TriggerMonitor(state_manager, script_manager)
trigger_monitor.start_monitoring()

# Executing actions
action_executor = ActionExecutor(state_manager, narrative_model)
result = await action_executor.execute_action(action)
```

### Hot Reloading

Scripts support hot reloading for rapid iteration:

```python
# Scripts are automatically reloaded when files change
script_manager.add_script(updated_script)  # Replaces existing script with same ID
```

### Error Handling

N-Script includes comprehensive error handling:

- **Parse Errors**: Invalid YAML or missing required fields
- **Validation Errors**: Parameters that don't meet schema requirements  
- **Runtime Errors**: Missing target agents or failed model calls
- **Fallback Behavior**: Graceful degradation when AI models unavailable

---

## Expert Tips & Tricks

### 1. Narrative Flow Control

Use generation overrides to create narrative "modes":

```yaml
# Create a "flashback mode"
- type: GENERATION_OVERRIDE
  target_agent_id: "narrator"
  generation_params:
    style: "flashback_narration"
    tone: "nostalgic"
    narrative_focus: "past_events"
    override_duration: 10
```

### 2. Emotional Arc Management

Chain scripts to create complex emotional journeys:

```yaml
# Script 1: Establish tension
script_id: tension_buildup
# ...creates suspense

# Script 2: Triggered by high tension
script_id: tension_release  
trigger:
  type: ON_EMOTIONAL_STATE
  conditions:
    emotional_state: "tense|anxious"
    mood_intensity: "> 0.8"
```

### 3. Memory Archaeology

Use memory triggers to callback to earlier events:

```yaml
trigger:
  type: ON_MEMORY_FORMATION
  conditions:
    tags: "contains:first_meeting"  # References earlier memories
actions:
  - type: TRIPLE_HEAD_ACTION
    # Character reflects on how far they've come
```

### 4. Dynamic Difficulty

Adjust narrative complexity based on player engagement:

```yaml
trigger:
  type: ON_HEAD_COORDINATION
  conditions:
    player_engagement: "> 0.8"
actions:
  - type: GENERATION_OVERRIDE
    generation_params:
      style: "complex_literary"  # Increase complexity for engaged players
```

### 5. Layered Storytelling

Create scripts that build on each other:

```yaml
# Layer 1: Surface level
script_id: casual_conversation

# Layer 2: Emotional subtext (triggered by control head)
script_id: hidden_feelings
trigger:
  type: ON_EMOTIONAL_STATE
  conditions:
    emotional_state: "conflicted"

# Layer 3: Deep revelation (triggered by memory significance)
script_id: truth_revealed
trigger:
  type: ON_MEMORY_FORMATION
  conditions:
    memory_significance: "> 0.9"
```

### 6. Character Development Tracking

Use memory tags to track character growth:

```yaml
memory_params:
  tags: ["character_growth", "confidence_+1", "relationship_deepened"]
  # Later scripts can reference these tags for character arc progression
```

### 7. Contextual Adaptation

Make scripts adapt to story context:

```yaml
control_params:
  emotions: ["{{time_of_day == 'night' ? 'mysterious' : 'friendly'}}"]
  control_tokens: ["{{weather == 'storm' ? '<voice_raised>' : '<voice_normal>'>"]
```

---

## Script Examples Library

### Beginner Examples
- `simple_greeting.nscript` - Basic location trigger
- `village_welcome.nscript` - Friendly NPC interaction

### Intermediate Examples  
- `emotional_support.nscript` - Emotional state triggers
- `dragon_encounter.nscript` - Complex triple-head coordination

### Advanced Examples
- `memory_amplification.nscript` - Memory-triggered narrative emphasis
- `character_development.nscript` - Multi-layered character growth
- `dynamic_world_state.nscript` - Context-adaptive behaviors

---

## Troubleshooting

### Common Issues

**Script Not Triggering**
- Check trigger conditions are met
- Verify target agents exist
- Ensure monitoring is started

**Actions Failing**
- Validate required parameters are provided
- Check target agent IDs are correct
- Review console logs for detailed errors

**Performance Issues**
- Reduce memory trigger complexity
- Batch related actions in single scripts
- Use simpler conditions when possible

**Memory Overload**
- Don't set every memory to high importance
- Use appropriate memory cleanup
- Tag memories for easy categorization

---

## Conclusion

N-Script is a powerful tool that transforms static NPCs into dynamic, emotionally intelligent characters. By understanding the triple-head architecture and following these best practices, you can create rich, believable narrative experiences that adapt and evolve with your players.

**Remember**: Start simple, iterate quickly, and let the three heads work together to create something greater than the sum of their parts.

---

**Happy Scripting!** 🎭💭🧠 