/**
 * QuadHeadStreamingPlayer Component
 * 
 * Handles real-time multimodal streaming from QuadHeadNarrativeLM model.
 * Features:
 * - WebSocket connection to quad-head streaming endpoint
 * - Real-time text token streaming and display
 * - Speech mel-spectrogram frame processing and audio playback
 * - Control signal visualization
 * - Memory update tracking
 * - Multimodal generation controls
 */

import React, { useState, useEffect, useRef, useCallback, useImperativeHandle, forwardRef } from 'react';
import styled from '@emotion/styled';

interface QuadHeadStreamResponse {
  type: string;
  text_token?: string;
  speech_frame?: number[];
  control_signal?: string;
  memory_update?: Record<string, any>;
  timestamp: number;
  finished: boolean;
  error?: string;
}

interface QuadHeadStreamRequest {
  type: string;
  text: string;
  character_id: string;
  max_length?: number;
  temperature?: number;
  speech_temperature?: number;
  emotion_context?: Record<string, any>;
  force_speech?: boolean;
  streaming?: boolean;
}

interface QuadHeadStreamingPlayerProps {
  characterId: string;
  wsUrl?: string;
  onTextToken?: (token: string) => void;
  onSpeechFrame?: (frame: number[]) => void;
  onControlSignal?: (signal: string) => void;
  onMemoryUpdate?: (update: Record<string, any>) => void;
  onConnectionChange?: (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void;
  onError?: (error: Error) => void;
  onGenerationComplete?: () => void;
}

interface QuadHeadStreamingPlayerRef {
  generateMultimodal: (text: string, options?: Partial<QuadHeadStreamRequest>) => void;
  isConnected: () => boolean;
  disconnect: () => void;
}

// Styled components
const Container = styled.div`
  background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
  border-radius: 12px;
  padding: 1.5rem;
  color: white;
  min-height: 400px;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
`;

const Title = styled.h3`
  margin: 0;
  color: #fff;
  font-size: 1.2rem;
  font-weight: 600;
`;

const StatusBadge = styled.div<{ status: string }>`
  padding: 0.3rem 0.8rem;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 500;
  background: ${props => {
    switch (props.status) {
      case 'connected': return '#10b981';
      case 'connecting': return '#f59e0b';
      case 'error': return '#ef4444';
      default: return '#6b7280';
    }
  }};
  color: white;
  text-transform: capitalize;
`;

const StreamsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1.5rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const StreamPanel = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 1rem;
`;

const StreamTitle = styled.h4`
  margin: 0 0 0.8rem 0;
  color: #fff;
  font-size: 0.9rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const TextStream = styled.div`
  font-family: 'Monaco', 'Menlo', monospace;
  font-size: 0.85rem;
  line-height: 1.4;
  color: #e5e7eb;
  min-height: 100px;
  max-height: 150px;
  overflow-y: auto;
  background: rgba(0, 0, 0, 0.3);
  padding: 0.8rem;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const ControlSignalDisplay = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
`;

const ControlChip = styled.div<{ isActive: boolean }>`
  padding: 0.3rem 0.6rem;
  background: ${props => props.isActive ? '#3b82f6' : 'rgba(255, 255, 255, 0.1)'};
  color: white;
  border-radius: 16px;
  font-size: 0.75rem;
  font-weight: 500;
  transition: all 0.3s ease;
`;

const MemoryPanel = styled.div`
  background: rgba(0, 0, 0, 0.3);
  padding: 0.8rem;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  min-height: 80px;
  font-size: 0.8rem;
  color: #d1d5db;
`;

const ControlsSection = styled.div`
  display: flex;
  gap: 1rem;
  align-items: center;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

const Input = styled.input`
  flex: 1;
  padding: 0.8rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  color: white;
  font-size: 0.9rem;
  
  &::placeholder {
    color: rgba(255, 255, 255, 0.5);
  }
  
  &:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }
`;

const Button = styled.button<{ variant?: 'primary' | 'secondary' }>`
  padding: 0.8rem 1.5rem;
  border-radius: 8px;
  border: none;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.9rem;
  
  background: ${props => props.variant === 'primary' ? '#3b82f6' : 'rgba(255, 255, 255, 0.1)'};
  color: white;
  
  &:hover {
    opacity: 0.8;
    transform: translateY(-1px);
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
`;

const SpeechVisualization = styled.div`
  display: flex;
  align-items: end;
  height: 60px;
  gap: 2px;
  background: rgba(0, 0, 0, 0.3);
  padding: 0.5rem;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const SpeechBar = styled.div<{ height: number }>`
  width: 3px;
  height: ${props => Math.max(2, props.height)}%;
  background: linear-gradient(to top, #ef4444, #f59e0b);
  border-radius: 1px;
  transition: height 0.1s ease;
`;

export const QuadHeadStreamingPlayer = forwardRef<QuadHeadStreamingPlayerRef, QuadHeadStreamingPlayerProps>(({
  characterId,
  wsUrl = 'ws://localhost:8000/api/v1/quad-head/stream',
  onTextToken,
  onSpeechFrame,
  onControlSignal,
  onMemoryUpdate,
  onConnectionChange,
  onError,
  onGenerationComplete
}, ref) => {
  // State management
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('disconnected');
  const [isGenerating, setIsGenerating] = useState(false);
  const [inputText, setInputText] = useState('');
  
  // Stream data
  const [textStream, setTextStream] = useState('');
  const [speechFrames, setSpeechFrames] = useState<number[][]>([]);
  const [currentControlSignal, setCurrentControlSignal] = useState<string>('');
  const [memoryUpdates, setMemoryUpdates] = useState<Record<string, any>>({});
  const [recentControlSignals, setRecentControlSignals] = useState<string[]>([]);
  
  // Refs
  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 3;

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
      return;
    }
    
    try {
      setConnectionStatus('connecting');
      onConnectionChange?.('connecting');
      
      const fullUrl = `${wsUrl}/${characterId}`;
      const ws = new WebSocket(fullUrl);
      websocketRef.current = ws;

      ws.onopen = () => {
        console.log('QuadHead WebSocket connected');
        setConnectionStatus('connected');
        onConnectionChange?.('connected');
        reconnectAttempts.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const response: QuadHeadStreamResponse = JSON.parse(event.data);
          
          if (response.type === 'connection_established') {
            console.log('QuadHead connection established:', response);
            return;
          }
          
          if (response.type === 'generation_step') {
            // Handle text token
            if (response.text_token) {
              setTextStream(prev => prev + response.text_token);
              onTextToken?.(response.text_token);
            }
            
            // Handle speech frame
            if (response.speech_frame) {
              setSpeechFrames(prev => [...prev.slice(-50), response.speech_frame!]); // Keep last 50 frames
              onSpeechFrame?.(response.speech_frame);
            }
            
            // Handle control signal
            if (response.control_signal) {
              setCurrentControlSignal(response.control_signal);
              setRecentControlSignals(prev => {
                const updated = [...prev, response.control_signal!];
                return updated.slice(-5); // Keep last 5 signals
              });
              onControlSignal?.(response.control_signal);
            }
            
            // Handle memory update
            if (response.memory_update) {
              setMemoryUpdates(prev => ({ ...prev, ...response.memory_update }));
              onMemoryUpdate?.(response.memory_update);
            }
          }
          
          if (response.type === 'generation_complete') {
            console.log('QuadHead generation complete');
            setIsGenerating(false);
            onGenerationComplete?.();
          }
          
          if (response.type === 'error') {
            console.error('QuadHead generation error:', response.error);
            setIsGenerating(false);
            onError?.(new Error(response.error || 'Generation failed'));
          }
          
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
          onError?.(new Error('Failed to parse server response'));
        }
      };

      ws.onerror = (error) => {
        console.error('QuadHead WebSocket error:', error);
        setConnectionStatus('error');
        onConnectionChange?.('error');
        onError?.(new Error('WebSocket connection failed'));
      };

      ws.onclose = () => {
        console.log('QuadHead WebSocket disconnected');
        setConnectionStatus('disconnected');
        onConnectionChange?.('disconnected');
        
        // Attempt reconnection
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, 1000 * reconnectAttempts.current);
        }
      };

    } catch (error) {
      console.error('Failed to create QuadHead WebSocket:', error);
      setConnectionStatus('error');
      onConnectionChange?.('error');
      onError?.(error as Error);
    }
  }, [wsUrl, characterId, onConnectionChange, onTextToken, onSpeechFrame, onControlSignal, onMemoryUpdate, onError, onGenerationComplete]);

  // Generate multimodal content
  const generateMultimodal = useCallback((text: string, options: Partial<QuadHeadStreamRequest> = {}) => {
    if (websocketRef.current?.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected');
      onError?.(new Error('Not connected to streaming service'));
      return;
    }
    
    setIsGenerating(true);
    setTextStream('');
    setSpeechFrames([]);
    setCurrentControlSignal('');
    setMemoryUpdates({});
    
    const request: QuadHeadStreamRequest = {
      type: 'generate_multimodal',
      text,
      character_id: characterId,
      max_length: 100,
      temperature: 0.8,
      speech_temperature: 0.7,
      force_speech: true,
      streaming: true,
      ...options
    };
    
    websocketRef.current.send(JSON.stringify(request));
  }, [characterId, onError]);

  // Handle input submission
  const handleSubmit = () => {
    if (inputText.trim() && !isGenerating) {
      generateMultimodal(inputText.trim());
      setInputText('');
    }
  };

  // Handle key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Expose methods via ref
  useImperativeHandle(ref, () => ({
    generateMultimodal,
    isConnected: () => websocketRef.current?.readyState === WebSocket.OPEN,
    disconnect: () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    }
  }), [generateMultimodal]);

  // Effects
  useEffect(() => {
    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, [connectWebSocket]);

  return (
    <Container>
      <Header>
        <Title>Quad-Head Multimodal Streaming</Title>
        <StatusBadge status={connectionStatus}>
          {connectionStatus}
        </StatusBadge>
      </Header>

      <StreamsGrid>
        {/* Text Generation Stream */}
        <StreamPanel>
          <StreamTitle>Text Generation</StreamTitle>
          <TextStream>
            {textStream || 'Generated text will appear here...'}
          </TextStream>
        </StreamPanel>

        {/* Speech Visualization */}
        <StreamPanel>
          <StreamTitle>Speech Generation</StreamTitle>
          <SpeechVisualization>
            {speechFrames.length > 0 ? (
              speechFrames.slice(-20).map((frame, i) => (
                <SpeechBar 
                  key={i} 
                  height={Math.abs(frame[0] || 0) * 100} 
                />
              ))
            ) : (
              <div style={{ color: 'rgba(255, 255, 255, 0.5)', fontSize: '0.8rem' }}>
                Speech visualization will appear here...
              </div>
            )}
          </SpeechVisualization>
        </StreamPanel>

        {/* Control Signals */}
        <StreamPanel>
          <StreamTitle>Control Signals</StreamTitle>
          <ControlSignalDisplay>
            {recentControlSignals.length > 0 ? (
              recentControlSignals.map((signal, i) => (
                <ControlChip 
                  key={`${signal}-${i}`} 
                  isActive={signal === currentControlSignal}
                >
                  {signal}
                </ControlChip>
              ))
            ) : (
              <div style={{ color: 'rgba(255, 255, 255, 0.5)', fontSize: '0.8rem' }}>
                Control signals will appear here...
              </div>
            )}
          </ControlSignalDisplay>
        </StreamPanel>

        {/* Memory Updates */}
        <StreamPanel>
          <StreamTitle>Memory Updates</StreamTitle>
          <MemoryPanel>
            {Object.keys(memoryUpdates).length > 0 ? (
              <pre>{JSON.stringify(memoryUpdates, null, 2)}</pre>
            ) : (
              <div style={{ color: 'rgba(255, 255, 255, 0.5)' }}>
                Memory updates will appear here...
              </div>
            )}
          </MemoryPanel>
        </StreamPanel>
      </StreamsGrid>

      <ControlsSection>
        <Input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Enter text to generate multimodal response..."
          disabled={isGenerating || connectionStatus !== 'connected'}
        />
        <Button
          variant="primary"
          onClick={handleSubmit}
          disabled={!inputText.trim() || isGenerating || connectionStatus !== 'connected'}
        >
          {isGenerating ? 'Generating...' : 'Generate'}
        </Button>
      </ControlsSection>
    </Container>
  );
}); 