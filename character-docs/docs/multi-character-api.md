# Multi-Character Conversation API

Complete API reference for the multi-character conversation system, including REST endpoints and WebSocket streaming.

## Overview

The Multi-Character Conversation API provides:
- **REST Endpoints**: CRUD operations for conversation management
- **WebSocket Streaming**: Real-time audio delivery and session management
- **Spatial Audio**: 3D positioning and environmental effects
- **Character Management**: Voice switching and relationship dynamics

**Base URL**: `http://localhost:8000/api/multi-character`
**WebSocket URL**: `ws://localhost:8000/ws/multi-character`

## Authentication

All endpoints require platform authentication. Include the session token in the Authorization header:

```http
Authorization: Bearer <your-session-token>
```

## REST Endpoints

### Create Conversation Session

Create a new multi-character conversation session.

```http
POST /api/multi-character/conversations
Content-Type: application/json

{
  "session_id": "unique-session-id",
  "character_ids": ["char1", "char2", "char3"],
  "environment": "room",
  "spatial_enabled": true
}
```

**Response:**
```json
{
  "conversation_id": "conv-uuid-123",
  "session_id": "unique-session-id",
  "character_count": 3,
  "status": "active",
  "created_at": "2025-01-20T10:00:00Z",
  "settings": {
    "environment": "room",
    "spatial_enabled": true,
    "master_volume": 0.8,
    "conversation_pacing": 1.0
  }
}
```

### Get Conversation

Retrieve conversation details and current state.

```http
GET /api/multi-character/conversations/{conversation_id}
```

**Response:**
```json
{
  "conversation_id": "conv-uuid-123",
  "session_id": "unique-session-id",
  "characters": [
    {
      "character_id": "char1",
      "name": "Alice",
      "voice_provider": "kokoro",
      "position": {"x": -1.0, "y": 0.0, "z": 0.0},
      "is_active": true
    }
  ],
  "dialogue_history": [
    {
      "turn_id": "turn-1",
      "character_id": "char1",
      "text": "Hello everyone!",
      "timestamp": "2025-01-20T10:01:00Z",
      "audio_metadata": {
        "duration": 2.5,
        "spatial_position": {"x": -1.0, "y": 0.0, "z": 0.0}
      }
    }
  ],
  "conversation_state": {
    "active_speakers": ["char1"],
    "emotional_context": {
      "tension": 0.3,
      "overall_mood": "friendly"
    }
  }
}
```

### Add Dialogue Turn

Add a new dialogue turn to the conversation.

```http
POST /api/multi-character/conversations/{conversation_id}/dialogue
Content-Type: application/json

{
  "character_id": "char1",
  "text": "This is what Alice would say next",
  "audio_settings": {
    "emotion_override": "excited",
    "pace_modifier": 1.2
  }
}
```

**Response:**
```json
{
  "turn_id": "turn-uuid-456",
  "character_id": "char1",
  "text": "This is what Alice would say next",
  "audio_url": "/audio/chunks/turn-uuid-456",
  "spatial_audio_data": {
    "position": {"x": -1.0, "y": 0.0, "z": 0.0},
    "distance": 1.0,
    "azimuth": -90.0,
    "elevation": 0.0
  },
  "estimated_duration": 3.2,
  "status": "processing"
}
```

### Update Character Position

Update a character's position in 3D space.

```http
PATCH /api/multi-character/conversations/{conversation_id}/characters/{character_id}/position
Content-Type: application/json

{
  "position": {"x": 1.5, "y": 0.0, "z": -0.5}
}
```

### Update Environment Settings

Change acoustic environment and effects.

```http
PATCH /api/multi-character/conversations/{conversation_id}/environment
Content-Type: application/json

{
  "environment": "hall",
  "reverb_level": 0.6,
  "ambient_noise": 0.2,
  "distance_attenuation": 0.8
}
```

### List Active Conversations

Get all active conversation sessions.

```http
GET /api/multi-character/conversations?status=active&limit=10
```

### Delete Conversation

End and delete a conversation session.

```http
DELETE /api/multi-character/conversations/{conversation_id}
```

## WebSocket Streaming

### Connection

Connect to the WebSocket endpoint for real-time streaming:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/multi-character');

ws.onopen = () => {
  // Send authentication and session info
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-session-token',
    session_id: 'unique-session-id'
  }));
};
```

### Message Types

#### Client → Server Messages

**Join Session:**
```json
{
  "type": "join_session",
  "session_id": "unique-session-id",
  "conversation_id": "conv-uuid-123"
}
```

**Update Character Position:**
```json
{
  "type": "update_character_position",
  "character_id": "char1",
  "position": {"x": 1.0, "y": 0.0, "z": 0.0}
}
```

**Update Audio Settings:**
```json
{
  "type": "update_audio_settings",
  "settings": {
    "master_volume": 0.9,
    "conversation_pacing": 1.2,
    "environment": "outdoor"
  }
}
```

#### Server → Client Messages

**Audio Chunk:**
```json
{
  "type": "audio_chunk",
  "turn_id": "turn-uuid-456",
  "character_id": "char1",
  "audio_data": "base64-encoded-audio-chunk",
  "spatial_info": {
    "position": {"x": -1.0, "y": 0.0, "z": 0.0},
    "distance": 1.0,
    "azimuth": -90.0,
    "hrtf_applied": true
  },
  "chunk_index": 0,
  "total_chunks": 5
}
```

**Conversation State Update:**
```json
{
  "type": "conversation_state_update",
  "conversation_id": "conv-uuid-123",
  "active_speakers": ["char1"],
  "character_positions": {
    "char1": {"x": -1.0, "y": 0.0, "z": 0.0},
    "char2": {"x": 1.0, "y": 0.0, "z": 0.0}
  },
  "emotional_context": {
    "tension": 0.4,
    "primary_emotion": "curiosity"
  }
}
```

**Character Activity Update:**
```json
{
  "type": "character_activity",
  "character_id": "char1",
  "activity": "speaking",
  "interruption_probability": 0.3
}
```

**Error Message:**
```json
{
  "type": "error",
  "error_code": "AUDIO_GENERATION_FAILED",
  "message": "Failed to generate audio for character char1",
  "turn_id": "turn-uuid-456"
}
```

## Data Models

### Vector3D
```typescript
interface Vector3D {
  x: number;  // Left(-) to Right(+)
  y: number;  // Down(-) to Up(+)  
  z: number;  // Forward(-) to Back(+)
}
```

### DialogueTurn
```typescript
interface DialogueTurn {
  turn_id: string;
  character_id: string;
  text: string;
  timestamp: string;
  audio_metadata?: {
    duration: number;
    file_size: number;
    spatial_position: Vector3D;
  };
  emotional_context?: {
    primary_emotion: string;
    intensity: number;
  };
}
```

### ConversationState
```typescript
interface ConversationState {
  conversation_id: string;
  active_speakers: string[];
  character_positions: Record<string, Vector3D>;
  environment_settings: {
    environment: 'studio' | 'room' | 'hall' | 'outdoor';
    reverb_level: number;
    ambient_noise: number;
    distance_attenuation: number;
  };
  audio_settings: {
    master_volume: number;
    conversation_pacing: number;
    spatial_enabled: boolean;
  };
  emotional_context: {
    tension: number;
    overall_mood: string;
    character_emotions: Record<string, string>;
  };
}
```

## Error Handling

### HTTP Status Codes

- **200 OK**: Request successful
- **201 Created**: Resource created successfully
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Authentication required
- **404 Not Found**: Resource not found
- **409 Conflict**: Resource already exists
- **422 Unprocessable Entity**: Validation errors
- **500 Internal Server Error**: Server error

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Character not found in conversation",
    "details": {
      "character_id": "char1",
      "conversation_id": "conv-uuid-123"
    }
  }
}
```

### Common Error Codes

- `CONVERSATION_NOT_FOUND`: Conversation ID doesn't exist
- `CHARACTER_NOT_FOUND`: Character not in conversation
- `AUDIO_GENERATION_FAILED`: TTS generation error
- `WEBSOCKET_CONNECTION_LOST`: Real-time connection dropped
- `SPATIAL_AUDIO_ERROR`: 3D audio processing failed
- `VOICE_MODEL_UNAVAILABLE`: Character voice provider offline

## Rate Limits

- **REST API**: 100 requests per minute per session
- **WebSocket**: 50 messages per second per connection
- **Audio Generation**: 10 concurrent generations per session

## SDK Examples

### JavaScript/TypeScript

```typescript
import { MultiCharacterAPI } from '@character-devkit/client';

const api = new MultiCharacterAPI({
  baseUrl: 'http://localhost:8000',
  token: 'your-session-token'
});

// Create conversation
const conversation = await api.createConversation({
  session_id: 'my-session',
  character_ids: ['alice', 'bob'],
  environment: 'room',
  spatial_enabled: true
});

// Add dialogue
const turn = await api.addDialogue(conversation.conversation_id, {
  character_id: 'alice',
  text: 'Hello Bob, how are you today?'
});

// Stream audio
api.connectWebSocket({
  onAudioChunk: (chunk) => {
    // Play spatial audio chunk
    audioPlayer.playSpatialChunk(chunk);
  },
  onStateUpdate: (state) => {
    // Update UI with conversation state
    updateConversationUI(state);
  }
});
```

### Python

```python
from character_devkit import MultiCharacterClient

client = MultiCharacterClient(
    base_url="http://localhost:8000",
    token="your-session-token"
)

# Create conversation
conversation = client.create_conversation(
    session_id="my-session",
    character_ids=["alice", "bob"],
    environment="room",
    spatial_enabled=True
)

# Add dialogue
turn = client.add_dialogue(
    conversation_id=conversation.conversation_id,
    character_id="alice",
    text="Hello Bob, how are you today?"
)

# Stream with WebSocket
async for message in client.stream_conversation(conversation.conversation_id):
    if message.type == "audio_chunk":
        await audio_player.play_spatial_chunk(message.audio_data)
```

## Performance Considerations

### Optimization Tips

1. **Connection Pooling**: Reuse WebSocket connections for multiple conversations
2. **Audio Buffering**: Buffer audio chunks to prevent stuttering
3. **Spatial Processing**: Use Web Audio API for client-side spatial processing when possible
4. **Character Limits**: Optimal performance with 2-4 active characters
5. **Environment Settings**: "Studio" environment has lowest processing overhead

### Monitoring

Monitor these metrics for optimal performance:

- **WebSocket Latency**: < 100ms for real-time experience
- **Audio Generation Time**: < 2 seconds per turn
- **Memory Usage**: Monitor client-side audio buffer usage
- **Network Bandwidth**: ~100kbps per active character for audio streaming

This API enables rich, immersive multi-character conversations with console-quality audio experiences. The combination of REST endpoints for session management and WebSocket streaming for real-time audio creates a powerful foundation for interactive narrative experiences. 