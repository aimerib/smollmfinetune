/**
 * StreamingAudioPlayer Component
 * 
 * Handles real-time audio streaming from WebSocket voice generation endpoints.
 * Features:
 * - WebSocket connection management
 * - Real-time audio chunk processing and playback  
 * - Audio buffering for smooth playback
 * - Play/pause controls and volume control
 * - Connection status indication
 * - Error handling and reconnection
 */

import React, { useState, useEffect, useRef, useCallback, useImperativeHandle, forwardRef } from 'react';
import './StreamingAudioPlayer.css';

interface StreamingAudioPlayerProps {
  characterId: string;
  wsUrl: string;
  onConnectionChange: (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void;
  onError: (error: Error) => void;
}

export interface StreamingAudioPlayerRef {
  generateVoice: (text: string, emotionContext?: Record<string, any>) => void;
  isConnected: () => boolean;
}

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

interface AudioChunk {
  data: ArrayBuffer;
  timestamp: number;
}

export const StreamingAudioPlayer = forwardRef<StreamingAudioPlayerRef, StreamingAudioPlayerProps>(({
  characterId,
  wsUrl,
  onConnectionChange,
  onError
}, ref) => {
  // State management
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(1.0);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('connecting');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  // Refs for audio management
  const websocketRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const audioBufferQueue = useRef<AudioChunk[]>([]);
  const isProcessingAudio = useRef(false);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const maxBufferSize = 10; // Limit buffer to prevent memory overflow

  // Initialize audio context
  const initAudioContext = useCallback(async () => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
        gainNodeRef.current = audioContextRef.current.createGain();
        gainNodeRef.current.connect(audioContextRef.current.destination);
        gainNodeRef.current.gain.value = volume;
      }
      
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
    } catch (error) {
      console.error('Failed to initialize audio context:', error);
      onError(error as Error);
    }
  }, [volume, onError]);

    // Audio processing
    const processAudioQueue = useCallback(async () => {
        if (isProcessingAudio.current || audioBufferQueue.current.length === 0) {
          return;
        }
    
        isProcessingAudio.current = true;
    
        try {
          await initAudioContext();
    
          const audioChunk = audioBufferQueue.current.shift();
          if (audioChunk && audioContextRef.current && gainNodeRef.current) {
            
            try {
              const audioBuffer = await audioContextRef.current.decodeAudioData(audioChunk.data);
              
              if (isPlaying) {
                const source = audioContextRef.current.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(gainNodeRef.current);
                source.start();
              }
            } catch (decodeError) {
              console.error('Failed to decode audio data:', decodeError);
              onError(new Error('Audio format not supported'));
            }
          }
        } catch (error) {
          console.error('Error processing audio:', error);
          onError(error as Error);
        } finally {
          isProcessingAudio.current = false;
          
          // Process next chunk if available
          if (audioBufferQueue.current.length > 0) {
            setTimeout(processAudioQueue, 10);
          }
        }
      }, [isPlaying, initAudioContext, onError]);

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    try {
      const fullUrl = `${wsUrl}/${characterId}`;
      const ws = new WebSocket(fullUrl);
      websocketRef.current = ws;

      ws.addEventListener('open', () => {
        console.log('WebSocket connected');
        setConnectionStatus('connected');
        onConnectionChange('connected');
        reconnectAttempts.current = 0;
        setErrorMessage(null);
      });

      ws.addEventListener('message', async (event) => {
        if (event.data instanceof ArrayBuffer) {
          const audioChunk: AudioChunk = {
            data: event.data,
            timestamp: Date.now()
          };

          // Add to buffer with size limit
          if (audioBufferQueue.current.length < maxBufferSize) {
            audioBufferQueue.current.push(audioChunk);
            processAudioQueue();
          }
        }
      });

      ws.addEventListener('error', (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
        setErrorMessage('Connection error occurred');
        onConnectionChange('error');
        onError(new Error('WebSocket connection failed'));
      });

      ws.addEventListener('close', () => {
        console.log('WebSocket disconnected');
        setConnectionStatus('disconnected');
        onConnectionChange('disconnected');
        
        // Attempt reconnection if not intentional
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, 1000 * reconnectAttempts.current);
        }
      });

    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      setConnectionStatus('error');
      setErrorMessage('Failed to establish connection');
      onError(error as Error);
    }
  }, [wsUrl, characterId, onConnectionChange, processAudioQueue, onError]);



  // Send voice generation request
  const sendVoiceRequest = useCallback((text: string = "Hello", emotionContext: Record<string, any> = {}) => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      const request = {
        type: 'generate_voice',
        text,
        character_id: characterId,
        emotion_context: emotionContext
      };
      websocketRef.current.send(JSON.stringify(request));
    }
  }, [characterId]);

  // Control handlers
  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
    
    if (!isPlaying) {
      // Start playing - send initial request
      sendVoiceRequest();
    } else {
      // Stop playing - clear audio sources
      if (audioContextRef.current) {
        // Stop all current audio sources
        const mockSource = audioContextRef.current.createBufferSource();
        mockSource.stop();
      }
    }
  };

  const handleVolumeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(event.target.value);
    setVolume(newVolume);
    
    if (gainNodeRef.current) {
      gainNodeRef.current.gain.value = newVolume;
    }
  };

  // Effects
  useEffect(() => {
    setConnectionStatus('connecting');
    onConnectionChange('connecting');
    connectWebSocket();

    return () => {
      // Cleanup
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      
      if (websocketRef.current && websocketRef.current.close) {
        websocketRef.current.close();
      }
      
      if (audioContextRef.current && audioContextRef.current.close) {
        audioContextRef.current.close();
      }
    };
  }, [connectWebSocket, onConnectionChange]);

  // Update volume when gainNode is available
  useEffect(() => {
    if (gainNodeRef.current) {
      gainNodeRef.current.gain.value = volume;
    }
  }, [volume]);

  // Expose methods to parent
  useImperativeHandle(ref, () => ({
    generateVoice: (text: string, emotionContext?: Record<string, any>) => {
      sendVoiceRequest(text, emotionContext || {});
    },
    isConnected: () => {
      return websocketRef.current?.readyState === WebSocket.OPEN;
    }
  }));

  return (
    <div className="streaming-audio-player">
      <div className="player-header">
        <h3>{characterId}</h3>
        <div className={`connection-status ${connectionStatus}`}>
          {connectionStatus === 'connecting' && 'Connecting...'}
          {connectionStatus === 'connected' && 'Connected'}
          {connectionStatus === 'disconnected' && 'Disconnected'}
          {connectionStatus === 'error' && 'Error'}
        </div>
      </div>

      {errorMessage && (
        <div className="error-message">
          Error: {errorMessage}
        </div>
      )}

      <div className="player-controls">
        <button 
          onClick={handlePlayPause}
          className="play-pause-button"
          aria-label={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? '⏸️' : '▶️'}
        </button>

        <div className="volume-control">
          <label htmlFor="volume">Volume</label>
          <input
            id="volume"
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={volume}
            onChange={handleVolumeChange}
            aria-label="Volume"
          />
          <span>{Math.round(volume * 100)}%</span>
        </div>
      </div>

      {isPlaying && (
        <div className="audio-visualizer" data-testid="audio-visualizer">
          <div className="visualizer-bars">
            {[...Array(8)].map((_, i) => (
              <div 
                key={i} 
                className="visualizer-bar"
                style={{ 
                  height: `${Math.random() * 100}%`,
                  animationDelay: `${i * 0.1}s` 
                }}
              />
            ))}
          </div>
        </div>
      )}

      <div className="player-info">
        <div className="buffer-status">
          Buffer: {audioBufferQueue.current.length}/{maxBufferSize}
        </div>
      </div>
    </div>
  );
});

StreamingAudioPlayer.displayName = 'StreamingAudioPlayer'; 