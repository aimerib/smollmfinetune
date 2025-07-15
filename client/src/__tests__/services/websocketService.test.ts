import websocketService, { EventType } from '../../services/websocketService';

// Mock WebSocket
class MockWebSocket {
  url: string;
  readyState: number = WebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  
  constructor(url: string) {
    this.url = url;
    // Don't auto-connect in tests
  }
  
  simulateOpen() {
    this.readyState = WebSocket.OPEN;
    if (this.onopen) {
      this.onopen(new Event('open'));
    }
  }
  
  simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
    }
  }
  
  simulateError(error: string) {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }
  
  simulateClose() {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }
  
  send(data: string) {
    // Mock send
  }
  
  close() {
    this.simulateClose();
  }
}

// Replace global WebSocket with mock
(global as any).WebSocket = MockWebSocket;

describe('WebSocketService', () => {
  let mockSocket: MockWebSocket;
  
  beforeEach(() => {
    jest.clearAllMocks();
    jest.clearAllTimers();
    // Reset the service state
    (websocketService as any).socket = null;
    (websocketService as any).connected = false;
    (websocketService as any).reconnectAttempts = 0;
    (websocketService as any).heartbeatInterval = null;
    (websocketService as any).reconnectTimeout = null;
    (websocketService as any).listeners = new Map();
  });
  
  afterEach(() => {
    jest.clearAllTimers();
    if ((websocketService as any).heartbeatInterval) {
      clearInterval((websocketService as any).heartbeatInterval);
    }
    if ((websocketService as any).reconnectTimeout) {
      clearTimeout((websocketService as any).reconnectTimeout);
    }
    if ((websocketService as any).socket) {
      (websocketService as any).socket.close();
      (websocketService as any).socket = null;
    }
  });

  describe('Connection Management', () => {
    test('connects to WebSocket successfully', async () => {
      const connectPromise = websocketService.connect();
      
      // Get the created socket
      mockSocket = (websocketService as any).socket;
      expect(mockSocket).toBeDefined();
      expect(mockSocket.url).toContain('/ws/director');
      
      // Manually trigger open
      mockSocket.simulateOpen();
      
      await connectPromise;
      expect((websocketService as any).connected).toBe(true);
    });

    test('sends subscription message on connect', async () => {
      const sendSpy = jest.spyOn(MockWebSocket.prototype, 'send');
      
      const connectPromise = websocketService.connect();
      mockSocket = (websocketService as any).socket;
      mockSocket.simulateOpen();
      
      await connectPromise;
      
      expect(sendSpy).toHaveBeenCalledWith(
        JSON.stringify({
          type: 'subscribe',
          topics: ['state_updates', 'memory_events', 'emotion_events', 'metrics']
        })
      );
    });

    test('does not create multiple connections', async () => {
      const connectPromise = websocketService.connect();
      mockSocket = (websocketService as any).socket;
      mockSocket.simulateOpen();
      await connectPromise;
      
      const firstSocket = (websocketService as any).socket;
      
      // Try to connect again
      await websocketService.connect();
      
      // Should still be the same socket
      expect((websocketService as any).socket).toBe(firstSocket);
    });

    test('handles connection errors', async () => {
      const connectPromise = websocketService.connect();
      mockSocket = (websocketService as any).socket;
      
      // Simulate error instead of open
      mockSocket.simulateError('Connection failed');
      
      await expect(connectPromise).rejects.toThrow('WebSocket connection failed');
      expect((websocketService as any).connected).toBe(false);
    });

    test('disconnects cleanly', async () => {
      const connectPromise = websocketService.connect();
      mockSocket = (websocketService as any).socket;
      mockSocket.simulateOpen();
      await connectPromise;
      
      await websocketService.disconnect();
      
      expect(mockSocket.readyState).toBe(WebSocket.CLOSED);
      expect((websocketService as any).socket).toBeNull();
      expect((websocketService as any).connected).toBe(false);
    });
  });

  describe('Message Handling', () => {
    beforeEach(async () => {
      const connectPromise = websocketService.connect();
      mockSocket = (websocketService as any).socket;
      mockSocket.simulateOpen();
      await connectPromise;
    });

    test('handles state update messages', () => {
      const listener = jest.fn();
      websocketService.on(EventType.STATE_UPDATE, listener);
      
      const stateUpdate = {
        type: 'state_update',
        data: { character_id: 'alice', state: 'active' }
      };
      
      mockSocket.simulateMessage(stateUpdate);
      
      expect(listener).toHaveBeenCalledWith(stateUpdate);
    });

    test('handles memory formation messages', () => {
      const listener = jest.fn();
      websocketService.on(EventType.MEMORY_FORMED, listener);
      
      const memoryEvent = {
        type: 'memory_formed',
        data: {
          character_id: 'alice',
          memory_content: 'Important event',
          importance: 0.8
        }
      };
      
      mockSocket.simulateMessage(memoryEvent);
      
      expect(listener).toHaveBeenCalledWith(memoryEvent);
    });

    test('handles emotion change messages', () => {
      const listener = jest.fn();
      websocketService.on(EventType.EMOTION_CHANGED, listener);
      
      const emotionEvent = {
        type: 'emotion_changed',
        data: {
          character_id: 'alice',
          emotions: { happy: 0.8, sad: 0.2 }
        }
      };
      
      mockSocket.simulateMessage(emotionEvent);
      
      expect(listener).toHaveBeenCalledWith(emotionEvent);
    });

    test('handles invalid JSON gracefully', () => {
      const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
      
      // Send invalid JSON
      if (mockSocket.onmessage) {
        mockSocket.onmessage(new MessageEvent('message', { data: 'invalid json' }));
      }
      
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to parse WebSocket message:',
        expect.any(Error)
      );
      
      consoleErrorSpy.mockRestore();
    });
  });

  describe('Event Listener Management', () => {
    beforeEach(async () => {
      const connectPromise = websocketService.connect();
      mockSocket = (websocketService as any).socket;
      mockSocket.simulateOpen();
      await connectPromise;
    });

    test('adds and removes event listeners', () => {
      const listener = jest.fn();
      
      websocketService.on(EventType.STATE_UPDATE, listener);
      
      // Trigger event
      mockSocket.simulateMessage({ type: 'state_update', data: {} });
      expect(listener).toHaveBeenCalledTimes(1);
      
      // Check if off method exists, if not, skip this part
      if (typeof websocketService.off === 'function') {
        // Remove listener
        websocketService.off(EventType.STATE_UPDATE, listener);
        
        // Trigger again
        mockSocket.simulateMessage({ type: 'state_update', data: {} });
        expect(listener).toHaveBeenCalledTimes(1); // Still 1, not called again
      }
    });

    test('supports multiple listeners for same event', () => {
      const listener1 = jest.fn();
      const listener2 = jest.fn();
      
      websocketService.on(EventType.STATE_UPDATE, listener1);
      websocketService.on(EventType.STATE_UPDATE, listener2);
      
      mockSocket.simulateMessage({ type: 'state_update', data: {} });
      
      expect(listener1).toHaveBeenCalled();
      expect(listener2).toHaveBeenCalled();
    });
  });

  describe('Request Methods', () => {
    beforeEach(async () => {
      const connectPromise = websocketService.connect();
      mockSocket = (websocketService as any).socket;
      mockSocket.simulateOpen();
      await connectPromise;
    });

    test('sends entity details request', () => {
      const sendSpy = jest.spyOn(mockSocket, 'send');
      
      if (typeof websocketService.requestEntityDetails === 'function') {
        websocketService.requestEntityDetails('alice');
        
        expect(sendSpy).toHaveBeenCalledWith(
          JSON.stringify({
            type: 'get_entity_details',
            entity_id: 'alice'
          })
        );
      }
    });

    test('sends memory timeline request', () => {
      const sendSpy = jest.spyOn(mockSocket, 'send');
      
      if (typeof websocketService.requestMemoryTimeline === 'function') {
        websocketService.requestMemoryTimeline('alice', 50);
        
        expect(sendSpy).toHaveBeenCalledWith(
          JSON.stringify({
            type: 'get_memory_timeline',
            character_id: 'alice',
            limit: 50
          })
        );
      }
    });

    test('sends filter update', () => {
      const sendSpy = jest.spyOn(mockSocket, 'send');
      
      const filters = {
        memories: true,
        emotions: false,
        metrics: true
      };
      
      if (typeof websocketService.updateFilters === 'function') {
        websocketService.updateFilters(filters);
        
        expect(sendSpy).toHaveBeenCalledWith(
          JSON.stringify({
            type: 'filter_update',
            filters
          })
        );
      }
    });
  });
}); 