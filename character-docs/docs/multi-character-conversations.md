# Multi-Character Conversations

Experience console-quality multi-character conversations with spatial audio, real-time mixing, and seamless character interactions. This advanced feature transforms the platform into an immersive audio experience that rivals AAA game audio systems.

## Overview

The Multi-Character Conversation system enables:

- **Spatial Audio Positioning**: Characters can be positioned in 3D space with HRTF processing
- **Real-time Voice Switching**: Seamless transitions between characters with natural interruptions
- **Environmental Effects**: Professional acoustic modeling with configurable environments
- **Live Audio Mixing**: Real-time controls for volume, pacing, and effects
- **Conversation Dynamics**: Relationship-based interruptions and emotional contagion

## Getting Started

### Prerequisites

1. **Platform Setup**: Ensure the React+FastAPI platform is running
2. **Character Creation**: Have at least 2 characters created in your world
3. **Voice Configuration**: Characters should have voice profiles configured

### Accessing the Audio Mixer

1. Start the platform with `./launch-client.sh`
2. Navigate to the React client at `http://localhost:3000`
3. Access the **Multi-Character Audio Mixer** from the main navigation

## Features in Detail

### 🎧 Spatial Audio Engine

The spatial audio system provides immersive 3D positioning:

- **HRTF Processing**: Head-Related Transfer Function for realistic directional audio
- **Distance Modeling**: Automatic volume and frequency adjustments based on distance
- **3D Positioning**: Drag-and-drop character placement in virtual space
- **Listener Position**: Central listener position with characters positioned around

**Controls:**
- Toggle 3D audio on/off
- Drag character icons to reposition them
- Visual indicators show character positions relative to listener

### 🎭 Character Voice Management

Seamless switching between character voices:

- **Active Speaker Indicators**: Visual feedback showing which character is speaking
- **Voice Profile Display**: Shows the TTS provider (Kokoro, Orpheus, etc.) for each character
- **Real-time Switching**: No delay when switching between character voices
- **Interrupt Handling**: Natural conversation interruptions based on character relationships

### 🌍 Environmental Effects

Professional acoustic modeling:

**Acoustic Environments:**
- **Studio**: Clean, dry sound with minimal reverb
- **Room**: Indoor room acoustics with moderate reverb
- **Hall**: Large space with long reverb tails
- **Outdoor**: Open space with natural ambience

**Effect Controls:**
- **Reverb Level**: Adjustable room reflection intensity (0-100%)
- **Ambient Noise**: Background environmental sounds (0-100%)
- **Distance Attenuation**: How much distance affects volume (0-100%)

### 🎚️ Master Controls

Real-time mixing capabilities:

- **Master Volume**: Overall conversation volume control
- **Conversation Pacing**: Speed up or slow down conversation delivery (0.5x - 2.0x)
- **Recording Controls**: Start/stop conversation recording
- **Session Management**: Track active sessions and character count

### 📊 Conversation Timeline

Visual conversation management:

- **Live Timeline**: Real-time display of conversation history
- **Speaker Attribution**: Clear indication of which character said what
- **Dialogue Input**: Add new dialogue with character selection
- **Conversation History**: Scrollable history of the entire conversation

## Advanced Usage

### Conversation Dynamics

The system includes sophisticated conversation dynamics:

**Interruption System:**
- Characters can interrupt each other based on relationship familiarity
- Personality traits (assertiveness) influence interruption probability
- Narrative tension affects conversation pacing

**Emotional Contagion:**
- Characters react to each other's emotional states
- Emotional influence based on character relationships
- Gradual emotional state changes during conversations

### WebSocket Integration

Real-time streaming architecture:

- **Low Latency**: WebSocket streaming for immediate audio delivery
- **Session Management**: Multiple conversation sessions supported
- **Live Updates**: Real-time conversation state synchronization
- **Error Handling**: Robust connection management and reconnection

### Performance Optimization

Efficient processing for smooth performance:

- **Chunk-based Processing**: Audio processed in real-time chunks
- **Memory Management**: Efficient handling of multiple character states
- **Concurrent Processing**: Multiple characters can be processed simultaneously
- **Caching**: Intelligent caching of voice models and audio data

## Troubleshooting

### Common Issues

**Audio Not Playing:**
1. Check browser audio permissions
2. Verify WebSocket connection is active
3. Ensure characters have voice profiles configured

**Spatial Audio Not Working:**
1. Use headphones for best spatial audio experience
2. Enable 3D audio in the mixer controls
3. Check character positioning in the spatial visualizer

**Character Switching Delays:**
1. Verify stable internet connection
2. Check server performance and load
3. Ensure voice models are properly loaded

### Performance Tips

1. **Limit Active Characters**: Best performance with 2-4 active characters
2. **Optimize Environment**: Use "Studio" environment for best performance
3. **Adjust Quality**: Lower reverb and effects for better performance on slower devices
4. **Use Headphones**: Best spatial audio experience requires headphones

## Technical Architecture

### Backend Components

- **MultiCharacterConversationManager**: Core orchestration system
- **SpatialAudioEngine**: 3D audio processing with HRTF
- **VoiceScheduler**: Character voice timing and scheduling
- **WebSocket Manager**: Real-time streaming infrastructure

### Frontend Components

- **MultiCharacterAudioMixer**: Main React interface
- **SpatialAudioControls**: 3D positioning controls
- **ConversationVisualizer**: Timeline and dialogue management
- **EnvironmentalEffectsPanel**: Acoustic environment controls

### Data Flow

1. **User Input**: Character selection and dialogue entry
2. **Backend Processing**: Voice generation with spatial positioning
3. **Real-time Streaming**: WebSocket delivery to client
4. **Audio Rendering**: Browser audio with spatial effects
5. **UI Updates**: Real-time conversation timeline updates

## Integration with Platform

The multi-character system integrates seamlessly with:

- **Character Management**: Uses existing character profiles and personalities
- **Narrative Engine**: Incorporates character relationships and emotional states
- **Voice System**: Leverages existing TTS orchestration and voice profiles
- **WebSocket Infrastructure**: Uses platform's real-time communication system

This creates a cohesive experience where multi-character conversations feel natural and integrated with the broader platform ecosystem.

## Next Steps

- Explore **Director's Chair** for advanced conversation monitoring
- Try **Character Builder** to create characters optimized for multi-character conversations
- Experiment with **World Builder** to create environments that enhance spatial audio
- Use **Dataset Studio** to generate conversation datasets for training

The Multi-Character Conversation system represents the future of AI character interaction, providing console-quality experiences that make characters feel truly alive and present in shared virtual spaces. 