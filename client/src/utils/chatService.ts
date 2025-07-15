import axios from 'axios';

// Configuration - in production, these would come from environment variables
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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
  private sessionStates: Map<string, SessionState> = new Map();

  constructor() {
    // No WebSocket initialization - using REST API only
    console.log('ChatService initialized with REST API only');
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
      console.log('Session created:', sessionId);
      
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
      console.log('Session ended:', sessionId);
    } catch (error) {
      console.error('Failed to end session:', error);
      throw error;
    }
  }

  // Character packet download
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

  // Proactive mode toggle
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

  // Simplified streaming (no WebSocket needed for now)
  async streamMessage(
    sessionId: string,
    characterId: string,
    message: string,
    onChunk: (chunk: string) => void,
    onComplete: (response: InferenceResponse) => void,
    onError: (error: Error) => void
  ) {
    try {
      // For now, just use regular sendMessage and call onComplete
      // In the future, this could use Server-Sent Events or WebSocket streaming
      const response = await this.sendMessage(sessionId, characterId, message);
      
      // Simulate streaming by chunking the response
      const text = response.generation_text;
      const words = text.split(' ');
      
      for (let i = 0; i < words.length; i++) {
        setTimeout(() => {
          onChunk(words[i] + ' ');
          if (i === words.length - 1) {
            onComplete(response);
          }
        }, i * 50); // 50ms delay between words
      }
    } catch (error) {
      onError(error as Error);
    }
  }

  // Clean up (no WebSocket to disconnect)
  disconnect() {
    console.log('ChatService disconnected');
  }
}

// Export singleton instance
export const chatService = new ChatService(); 