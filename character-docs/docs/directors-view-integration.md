# Director's View Integration Guide

## Current State vs. Target State

The Director's View is a beautifully designed real-time monitoring interface that currently operates as a demonstration. This document outlines what needs to be integrated for full functionality.

## 🚧 What's Currently Missing

### Director's View Backend Issues

#### 1. **Mock Data Generation**
- **Current**: WebSocket sends fake character positions, emotions, and memories
- **Needed**: Real-time data from active character inference sessions

#### 2. **Service Integration**
Missing connections to real services:
- `state_service` - Should track actual character states from inference
- `memory_service` - Should receive real memory formations from narrative engine
- `emotion_service` - Should track emotional states from control head outputs
- `event_bus` - Should connect to actual model inference events

#### 3. **Triple-Head Metrics**
- **Current**: Random metrics generation
- **Needed**: Real metrics from NarrativeLLM's triple-head architecture:
  - Generation head coherence scores
  - Control head token usage
  - Memory head formation rates

### Chat Interface Issues

#### 1. **Inference Integration**
- **Current**: Hardcoded responses based on character personality
- **Needed**: Actual model inference using:
  - Trained LoRA/DoRA adapters
  - SmolLM2 base model
  - NarrativeLLM with triple-head architecture

#### 2. **Character Loading**
- **Current**: Mock character list (Alice, Max, Luna)
- **Needed**: Dynamic loading of:
  - Trained adapters from `training_output/`
  - Runtime packets from `runtime_packets/`
  - Character core data and world integration

#### 3. **Runtime Integration**
Missing connections:
- `RuntimePromptConstructor` for dynamic prompt generation
- `InferenceManager` for model loading and generation
- `PlatformRuntimeEngine` for multi-character orchestration

## 🔧 Integration Plan

### Phase 1: Connect Inference Pipeline
1. Update `/api/inference` endpoint to use real InferenceManager
2. Load actual trained adapters instead of mock characters
3. Implement proper session management with conversation history

### Phase 2: Event Bus Integration
1. Connect inference events to WebSocket broadcast
2. Emit real events:
   - Memory formation from memory head outputs
   - Emotion changes from control tokens
   - Subtext from character's internal monologue

### Phase 3: State Synchronization
1. Track active inference sessions
2. Update character positions based on narrative context
3. Maintain persistent state across sessions

### Phase 4: Metrics Collection
1. Capture real triple-head metrics during inference
2. Track performance indicators:
   - Response time
   - Token generation rate
   - Memory formation frequency
   - Emotional coherence

## 📝 Required Components

### For Director's View:
```python
# Event emission during inference
async def emit_inference_events(character_id: str, output: TripleHeadOutput):
    # Emit memory formation
    if output.memory_metadata.get('importance', 0) > 0.7:
        await event_bus.emit(EventType.MEMORY_FORMED, {
            'character_id': character_id,
            'content': output.memory_metadata.get('content'),
            'importance': output.memory_metadata.get('importance')
        })
    
    # Emit emotion change
    if output.control_tokens:
        emotions = extract_emotions(output.control_tokens)
        await event_bus.emit(EventType.EMOTION_CHANGED, {
            'character_id': character_id,
            'emotions': emotions
        })
```

### For Chat Interface:
```python
# Real inference endpoint
@router.post("/inference")
async def generate_response(request: InferenceRequest):
    # Load character model
    model = await inference_engine.load_character(request.character_id)
    
    # Construct prompt with runtime
    prompt = runtime_engine.construct_prompt(
        character_id=request.character_id,
        conversation_history=request.context_window,
        user_input=request.prompt
    )
    
    # Generate with triple-head model
    output = await model.generate_with_control(
        prompt=prompt,
        max_tokens=request.max_tokens
    )
    
    # Emit events for Director's View
    await emit_inference_events(request.character_id, output)
    
    return InferenceResponse.from_triple_head(request, output)
```

## 🎯 Success Criteria

The integration is complete when:
1. Chat messages trigger real model inference
2. Director's View shows actual memory formations
3. Emotions update based on conversation content
4. Triple-head metrics reflect real model performance
5. Multiple characters can interact with proper state management

## 🚀 Next Steps

1. Start with connecting the inference pipeline (most critical)
2. Add event emission during inference
3. Update WebSocket handlers to use real data
4. Test with trained character models
5. Add performance monitoring and optimization 