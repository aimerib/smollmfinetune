import axios from 'axios';
import io, { Socket } from 'socket.io-client';

// Configuration - in production, these would come from environment variables
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

interface InferenceResponse {
  generation_text: string;
  control_tokens: Array<{
    token: string;
    confidence: number;
  }>;
  memory_vector: number[];
  memory_metadata: {
    importance: number;
    emotional_valence: number;
    context_relevance: number;
    decay_rate: number;
  };
  session_id: string;
  latency_ms: number;
}

interface SessionState {
  session_id: string;
  character_id: string;
  message_count: number;
  emotional_state: {
    current: string;
    history: Array<{
      emotion: string;
      timestamp: string;
      confidence: number;
    }>;
  };
  memory_stats: {
    total_memories: number;
    important_memories: number;
    last_memory_timestamp: string;
  };
}

class ChatService {
  private socket: Socket | null = null;
  private sessionStates: Map<string, SessionState> = new Map();

  constructor() {
    this.initializeWebSocket();
  }

  private initializeWebSocket() {
    // Initialize WebSocket connection for real-time updates
    this.socket = io(WS_URL, {
      transports: ['websocket'],
      autoConnect: false
    });

    this.socket.on('emotion_update', (data) => {
      // Handle real-time emotion updates
      window.dispatchEvent(new CustomEvent('emotion_update', { detail: data }));
    });

    this.socket.on('memory_formed', (data) => {
      // Handle memory formation events
      window.dispatchEvent(new CustomEvent('memory_formed', { detail: data }));
    });

    this.socket.on('proactive_message', (data) => {
      // Handle proactive agent messages
      window.dispatchEvent(new CustomEvent('proactive_message', { detail: data }));
    });
  }

  async startSession(characterId: string): Promise<string> {
    try {
      const response = await axios.post(`${API_BASE_URL}/sessions`, {
        character_id: characterId,
        platform: 'web',
        device_info: {
          userAgent: navigator.userAgent,
          screen: {
            width: window.screen.width,
            height: window.screen.height
          }
        }
      });

      const sessionId = response.data.session_id;
      
      // Connect WebSocket for this session
      if (this.socket) {
        this.socket.connect();
        this.socket.emit('join_session', { session_id: sessionId });
      }

      return sessionId;
    } catch (error) {
      console.error('Failed to start session:', error);
      throw error;
    }
  }

  async sendMessage(
    sessionId: string, 
    characterId: string, 
    message: string
  ): Promise<InferenceResponse> {
    try {
      const response = await axios.post(`${API_BASE_URL}/inference`, {
        session_id: sessionId,
        character_id: characterId,
        prompt: message,
        stream: false,
        include_emotions: true,
        memory_context_window: 10
      });

      return response.data;
    } catch (error) {
      console.error('Failed to send message:', error);
      throw error;
    }
  }

  async getSessionState(sessionId: string): Promise<SessionState> {
    try {
      const response = await axios.get(`${API_BASE_URL}/sessions/${sessionId}/state`);
      return response.data;
    } catch (error) {
      console.error('Failed to get session state:', error);
      throw error;
    }
  }

  async endSession(sessionId: string): Promise<void> {
    try {
      await axios.post(`${API_BASE_URL}/sessions/${sessionId}/end`);
      
      // Disconnect WebSocket for this session
      if (this.socket) {
        this.socket.emit('leave_session', { session_id: sessionId });
        this.socket.disconnect();
      }
    } catch (error) {
      console.error('Failed to end session:', error);
      throw error;
    }
  }

  // Mobile-optimized methods
  async downloadCharacterPacket(characterId: string): Promise<Blob> {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/characters/${characterId}/packet`,
        { responseType: 'blob' }
      );
      return response.data;
    } catch (error) {
      console.error('Failed to download character packet:', error);
      throw error;
    }
  }

  async enableProactiveMode(sessionId: string, enabled: boolean): Promise<void> {
    try {
      await axios.post(`${API_BASE_URL}/sessions/${sessionId}/proactive`, {
        enabled,
        check_interval_minutes: 30,
        notification_preferences: {
          push: true,
          sound: false,
          vibration: true
        }
      });
    } catch (error) {
      console.error('Failed to toggle proactive mode:', error);
      throw error;
    }
  }

  // Streaming support for mobile
  streamMessage(
    sessionId: string,
    characterId: string,
    message: string,
    onChunk: (chunk: string) => void,
    onComplete: (response: InferenceResponse) => void,
    onError: (error: Error) => void
  ) {
    if (!this.socket) {
      onError(new Error('WebSocket not initialized'));
      return;
    }

    this.socket.emit('stream_inference', {
      session_id: sessionId,
      character_id: characterId,
      prompt: message
    });

    this.socket.on('stream_chunk', (data) => {
      onChunk(data.chunk);
    });

    this.socket.on('stream_complete', (data) => {
      onComplete(data);
      // Clean up listeners
      this.socket?.off('stream_chunk');
      this.socket?.off('stream_complete');
      this.socket?.off('stream_error');
    });

    this.socket.on('stream_error', (error) => {
      onError(new Error(error.message));
      // Clean up listeners
      this.socket?.off('stream_chunk');
      this.socket?.off('stream_complete');
      this.socket?.off('stream_error');
    });
  }

  // Clean up on unmount
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

// Export singleton instance
export const chatService = new ChatService(); 