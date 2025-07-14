/**
 * Multi-Character Audio Mixer Component
 * 
 * Provides a comprehensive interface for managing multi-character conversations
 * with real-time audio mixing, spatial positioning, and conversation visualization.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import './MultiCharacterAudioMixer.css';

// Types for multi-character conversation
interface CharacterVoice {
  id: string;
  name: string;
  voiceProfile: VoiceProfile;
  isActive: boolean;
  volume: number;
  spatialPosition: Vector3D;
}

interface VoiceProfile {
  provider: 'kokoro' | 'orpheus' | 'xtts';
  voiceId: string;
  emotionalRange: number;
  baseFrequency: number;
  expressiveness: number;
}

interface Vector3D {
  x: number;
  y: number;
  z: number;
}

interface ConversationState {
  sessionId: string;
  activeCharacters: string[];
  context: Record<string, any>;
  spatialPositions: Record<string, Vector3D>;
}

interface DialogueTurn {
  characterId: string;
  text: string;
  timestamp: number;
  emotionContext?: Record<string, number>;
  interruptsPrevious?: boolean;
}

interface MixerSettings {
  masterVolume: number;
  spatialAudioEnabled: boolean;
  environmentalEffects: EnvironmentalEffects;
  conversationPacing: number;
}

interface EnvironmentalEffects {
  reverbLevel: number;
  ambientNoise: number;
  acousticEnvironment: 'studio' | 'room' | 'hall' | 'outdoor';
  distanceAttenuation: number;
}

interface MultiCharacterAudioMixerProps {
  sessionId: string;
  characters: CharacterVoice[];
  onCharacterUpdate: (character: CharacterVoice) => void;
  onDialogueTurn: (turn: DialogueTurn) => void;
}

const MultiCharacterAudioMixer: React.FC<MultiCharacterAudioMixerProps> = ({
  sessionId,
  characters,
  onCharacterUpdate,
  onDialogueTurn
}) => {
  const [conversationState, setConversationState] = useState<ConversationState | null>(null);
  const [mixerSettings, setMixerSettings] = useState<MixerSettings>({
    masterVolume: 0.8,
    spatialAudioEnabled: true,
    environmentalEffects: {
      reverbLevel: 0.3,
      ambientNoise: 0.1,
      acousticEnvironment: 'room',
      distanceAttenuation: 0.7
    },
    conversationPacing: 1.0
  });
  const [selectedCharacter, setSelectedCharacter] = useState<string | null>(null);
  const [conversationHistory, setConversationHistory] = useState<DialogueTurn[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // WebSocket connection for real-time communication
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/api/v1/multi-character/conversations/${sessionId}/stream`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('Multi-character WebSocket connected');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      console.log('Multi-character WebSocket disconnected');
    };

    ws.onerror = (error) => {
      console.error('Multi-character WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, [sessionId]);

  const handleWebSocketMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'conversation_state':
        setConversationState(data.data);
        break;
      
      case 'audio_chunk':
        // Handle incoming audio chunk
        handleAudioChunk(data.data);
        break;
      
      case 'spatial_position_updated':
        // Update character position in UI
        updateCharacterPosition(data.data);
        break;
      
      case 'conversation_paused':
      case 'conversation_resumed':
        // Handle conversation control updates
        break;
      
      default:
        console.log('Unknown WebSocket message:', data);
    }
  }, []);

  const handleAudioChunk = (audioData: any) => {
    // Process incoming audio chunk for real-time playback
    const { character_id, audio_data, spatial_info } = audioData;
    
    // Update conversation history
    if (audioData.text) {
      const newTurn: DialogueTurn = {
        characterId: character_id,
        text: audioData.text,
        timestamp: Date.now(),
        emotionContext: audioData.emotion_context
      };
      setConversationHistory(prev => [...prev, newTurn]);
    }
  };

  const updateCharacterPosition = (positionData: any) => {
    const { character_id, position, distance } = positionData;
    
    // Update character in local state
    const updatedCharacter = characters.find(c => c.id === character_id);
    if (updatedCharacter) {
      updatedCharacter.spatialPosition = position;
      onCharacterUpdate(updatedCharacter);
    }
  };

  const handleSpatialPositionChange = (characterId: string, position: Vector3D) => {
    // Send position update to backend
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'spatial_position',
        data: {
          character_id: characterId,
          position: position
        }
      }));
    }
  };

  const handleVolumeChange = (characterId: string, volume: number) => {
    const character = characters.find(c => c.id === characterId);
    if (character) {
      character.volume = volume;
      onCharacterUpdate(character);
    }
  };

  const handleDialogueSubmit = (characterId: string, text: string) => {
    const turn: DialogueTurn = {
      characterId,
      text,
      timestamp: Date.now()
    };

    // Send to backend via WebSocket
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'dialogue_turn',
        data: {
          character_id: characterId,
          text: text,
          emotion_context: {},
          interrupts_previous: false,
          urgency_level: 0.5
        }
      }));
    }

    onDialogueTurn(turn);
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
  };

  return (
    <div className="multi-character-audio-mixer">
      {/* Header */}
      <div className="mixer-header">
        <h2>Multi-Character Audio Mixer</h2>
        <div className="session-info">
          <span>Session: {sessionId}</span>
          <span>Characters: {characters.length}</span>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="mixer-content">
        {/* Left Panel - Character Controls */}
        <div className="character-panel">
          <h3>Character Controls</h3>
          
          {characters.map(character => (
            <CharacterController
              key={character.id}
              character={character}
              isSelected={selectedCharacter === character.id}
              onSelect={() => setSelectedCharacter(character.id)}
              onVolumeChange={(volume) => handleVolumeChange(character.id, volume)}
              onPositionChange={(position) => handleSpatialPositionChange(character.id, position)}
            />
          ))}
        </div>

        {/* Center Panel - Conversation Visualizer */}
        <div className="conversation-panel">
          <ConversationVisualizer
            characters={characters}
            conversationHistory={conversationHistory}
            onDialogueSubmit={handleDialogueSubmit}
          />
        </div>

        {/* Right Panel - Spatial & Environmental Controls */}
        <div className="effects-panel">
          <SpatialAudioControls
            characters={characters}
            onPositionChange={handleSpatialPositionChange}
            enabled={mixerSettings.spatialAudioEnabled}
            onToggle={(enabled) => setMixerSettings(prev => ({
              ...prev,
              spatialAudioEnabled: enabled
            }))}
          />
          
          <EnvironmentalEffectsPanel
            effects={mixerSettings.environmentalEffects}
            onEffectChange={(effects) => setMixerSettings(prev => ({
              ...prev,
              environmentalEffects: effects
            }))}
          />
        </div>
      </div>

      {/* Bottom Panel - Master Controls */}
      <div className="master-controls">
        <MasterMixer
          settings={mixerSettings}
          onSettingsChange={setMixerSettings}
          onRecordToggle={toggleRecording}
          isRecording={isRecording}
        />
      </div>
    </div>
  );
};

// Character Controller Component
interface CharacterControllerProps {
  character: CharacterVoice;
  isSelected: boolean;
  onSelect: () => void;
  onVolumeChange: (volume: number) => void;
  onPositionChange: (position: Vector3D) => void;
}

const CharacterController: React.FC<CharacterControllerProps> = ({
  character,
  isSelected,
  onSelect,
  onVolumeChange,
  onPositionChange
}) => {
  return (
    <div className={`character-controller ${isSelected ? 'selected' : ''}`}>
      <div className="character-header" onClick={onSelect}>
        <div className="character-info">
          <span className="character-name">{character.name}</span>
          <span className="voice-provider">{character.voiceProfile.provider}</span>
        </div>
        <div className={`activity-indicator ${character.isActive ? 'active' : ''}`} />
      </div>
      
      {isSelected && (
        <div className="character-controls">
          {/* Volume Control */}
          <div className="control-group">
            <label>Volume</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={character.volume}
              onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
            />
            <span>{Math.round(character.volume * 100)}%</span>
          </div>

          {/* Position Controls */}
          <div className="control-group">
            <label>Position</label>
            <div className="position-controls">
              <input
                type="range"
                min="-5"
                max="5"
                step="0.1"
                value={character.spatialPosition.x}
                onChange={(e) => onPositionChange({
                  ...character.spatialPosition,
                  x: parseFloat(e.target.value)
                })}
                title="X Position"
              />
              <input
                type="range"
                min="-5"
                max="5"
                step="0.1"
                value={character.spatialPosition.z}
                onChange={(e) => onPositionChange({
                  ...character.spatialPosition,
                  z: parseFloat(e.target.value)
                })}
                title="Z Position"
              />
            </div>
          </div>

          {/* Voice Profile */}
          <div className="control-group">
            <label>Voice Profile</label>
            <div className="voice-info">
              <span>Expressiveness: {Math.round(character.voiceProfile.expressiveness * 100)}%</span>
              <span>Emotional Range: {Math.round(character.voiceProfile.emotionalRange * 100)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Conversation Visualizer Component
interface ConversationVisualizerProps {
  characters: CharacterVoice[];
  conversationHistory: DialogueTurn[];
  onDialogueSubmit: (characterId: string, text: string) => void;
}

const ConversationVisualizer: React.FC<ConversationVisualizerProps> = ({
  characters,
  conversationHistory,
  onDialogueSubmit
}) => {
  const [newMessage, setNewMessage] = useState('');
  const [selectedSpeaker, setSelectedSpeaker] = useState<string>('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (newMessage.trim() && selectedSpeaker) {
      onDialogueSubmit(selectedSpeaker, newMessage.trim());
      setNewMessage('');
    }
  };

  return (
    <div className="conversation-visualizer">
      <h3>Conversation Timeline</h3>
      
      {/* Conversation History */}
      <div className="conversation-history">
        {conversationHistory.map((turn, index) => {
          const character = characters.find(c => c.id === turn.characterId);
          return (
            <div key={index} className="dialogue-turn">
              <div className="speaker-info">
                <span className="speaker-name">{character?.name || turn.characterId}</span>
                <span className="timestamp">
                  {new Date(turn.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <div className="dialogue-text">{turn.text}</div>
              {turn.emotionContext && (
                <div className="emotion-indicators">
                  {Object.entries(turn.emotionContext).map(([emotion, intensity]) => (
                    <span key={emotion} className="emotion-tag">
                      {emotion}: {Math.round(intensity * 100)}%
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="dialogue-input">
        <select
          value={selectedSpeaker}
          onChange={(e) => setSelectedSpeaker(e.target.value)}
          required
        >
          <option value="">Select Speaker</option>
          {characters.map(character => (
            <option key={character.id} value={character.id}>
              {character.name}
            </option>
          ))}
        </select>
        
        <textarea
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          placeholder="Enter dialogue..."
          aria-label="Enter dialogue"
          rows={2}
          required
        />
        
        <button type="submit" disabled={!selectedSpeaker || !newMessage.trim()}>
          Send
        </button>
      </form>
    </div>
  );
};

// Spatial Audio Controls Component
interface SpatialAudioControlsProps {
  characters: CharacterVoice[];
  onPositionChange: (characterId: string, position: Vector3D) => void;
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
}

const SpatialAudioControls: React.FC<SpatialAudioControlsProps> = ({
  characters,
  onPositionChange,
  enabled,
  onToggle
}) => {
  return (
    <div className="spatial-audio-controls">
      <h4>Spatial Audio</h4>
      
      <div className="spatial-toggle">
        <label>
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onToggle(e.target.checked)}
          />
          Enable 3D Audio
        </label>
      </div>

      {enabled && (
        <div className="spatial-visualizer">
          <div className="spatial-grid">
            {/* Listener position (center) */}
            <div className="listener-position" style={{ left: '50%', top: '50%' }}>
              🎧
            </div>
            
            {/* Character positions */}
            {characters.map(character => {
              const x = (character.spatialPosition.x + 5) * 10; // Scale to percentage
              const z = (character.spatialPosition.z + 5) * 10;
              
              return (
                <div
                  key={character.id}
                  className="character-position"
                  style={{ left: `${x}%`, top: `${z}%` }}
                  title={character.name}
                >
                  🗣️
                </div>
              );
            })}
          </div>
          
          <div className="spatial-legend">
            <span>Drag characters to reposition</span>
          </div>
        </div>
      )}
    </div>
  );
};

// Environmental Effects Panel Component
interface EnvironmentalEffectsPanelProps {
  effects: EnvironmentalEffects;
  onEffectChange: (effects: EnvironmentalEffects) => void;
}

const EnvironmentalEffectsPanel: React.FC<EnvironmentalEffectsPanelProps> = ({
  effects,
  onEffectChange
}) => {
  const handleChange = (key: keyof EnvironmentalEffects, value: any) => {
    onEffectChange({ ...effects, [key]: value });
  };

  return (
    <div className="environmental-effects-panel">
      <h4>Environmental Effects</h4>
      
      <div className="effect-control">
        <label>Acoustic Environment</label>
        <select
          value={effects.acousticEnvironment}
          onChange={(e) => handleChange('acousticEnvironment', e.target.value)}
        >
          <option value="studio">Studio</option>
          <option value="room">Room</option>
          <option value="hall">Hall</option>
          <option value="outdoor">Outdoor</option>
        </select>
      </div>

      <div className="effect-control">
        <label>Reverb Level</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={effects.reverbLevel}
          onChange={(e) => handleChange('reverbLevel', parseFloat(e.target.value))}
        />
        <span>{Math.round(effects.reverbLevel * 100)}%</span>
      </div>

      <div className="effect-control">
        <label>Ambient Noise</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={effects.ambientNoise}
          onChange={(e) => handleChange('ambientNoise', parseFloat(e.target.value))}
        />
        <span>{Math.round(effects.ambientNoise * 100)}%</span>
      </div>

      <div className="effect-control">
        <label>Distance Attenuation</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={effects.distanceAttenuation}
          onChange={(e) => handleChange('distanceAttenuation', parseFloat(e.target.value))}
        />
        <span>{Math.round(effects.distanceAttenuation * 100)}%</span>
      </div>
    </div>
  );
};

// Master Mixer Component
interface MasterMixerProps {
  settings: MixerSettings;
  onSettingsChange: (settings: MixerSettings) => void;
  onRecordToggle: () => void;
  isRecording: boolean;
}

const MasterMixer: React.FC<MasterMixerProps> = ({
  settings,
  onSettingsChange,
  onRecordToggle,
  isRecording
}) => {
  const handleChange = (key: keyof MixerSettings, value: any) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  return (
    <div className="master-mixer">
      <div className="master-controls-group">
        <div className="control">
          <label>Master Volume</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={settings.masterVolume}
            onChange={(e) => handleChange('masterVolume', parseFloat(e.target.value))}
          />
          <span>{Math.round(settings.masterVolume * 100)}%</span>
        </div>

        <div className="control">
          <label>Conversation Pacing</label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            value={settings.conversationPacing}
            onChange={(e) => handleChange('conversationPacing', parseFloat(e.target.value))}
          />
          <span>{settings.conversationPacing}x</span>
        </div>

        <div className="control">
          <button
            className={`record-button ${isRecording ? 'recording' : ''}`}
            onClick={onRecordToggle}
          >
            {isRecording ? '⏹️ Stop Recording' : '🔴 Start Recording'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default MultiCharacterAudioMixer; 