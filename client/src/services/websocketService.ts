/**
 * WebSocket Service for Director's View
 * 
 * Manages real-time communication with the FastAPI backend for live updates
 * on character states, memories, emotions, and metrics.
 */

// Event types
export enum EventType {
  STATE_UPDATE = 'state_update',
  MEMORY_FORMED = 'memory_formed',
  EMOTION_CHANGED = 'emotion_changed',
  SUBTEXT_ADDED = 'subtext_added',
  TRIPLE_HEAD_METRICS = 'triple_head_metrics',
  RELATIONSHIP_UPDATE = 'relationship_update',
}

// Interfaces
export interface StateUpdateEvent {
  event_type: EventType.STATE_UPDATE;
  timestamp: string;
  source: string;
  data: {
    entity_id: string;
    changes: Record<string, any>;
  };
}

export interface MemoryFormationEvent {
  event_type: EventType.MEMORY_FORMED;
  timestamp: string;
  source: string;
  data: {
    character_id: string;
    memory_content: string;
    importance: number;
    emotional_valence: number;
    memory_type: string;
    visualization: {
      bubble_color: string;
      bubble_size: number;
    };
  };
}

export interface EmotionChangeEvent {
  event_type: EventType.EMOTION_CHANGED;
  timestamp: string;
  source: string;
  data: {
    character_id: string;
    active_emotions: Record<string, number>;
    surprise_score: number;
    momentum: number;
  };
}

export interface TripleHeadMetricsEvent {
  event_type: EventType.TRIPLE_HEAD_METRICS;
  timestamp: string;
  source: string;
  data: {
    character_id: string;
    metrics: {
      generation_quality: number;
      control_effectiveness: number;
      memory_coherence: number;
      coordination_score: number;
    };
  };
}

export interface RelationshipUpdateEvent {
  event_type: EventType.RELATIONSHIP_UPDATE;
  timestamp: string;
  source: string;
  speaker_id: string;
  target_id: string;
  interaction_type: string;
  affinity_change: number;
  emotional_impact: string[];
  memory_significance: number;
  narrative_context?: string;
}

export interface WorldSnapshot {
  locations: Array<{
    id: string;
    name: string;
    type: string;
    custom_data: Record<string, any>;
  }>;
  characters: Array<{
    id: string;
    name: string;
    type: string;
    location: string;
    custom_data: Record<string, any>;
    memory_count: number;
    relationship_count: number;
  }>;
  total_entities: number;
  recent_subtext: Array<{
    agent_id: string;
    text: string;
    timestamp: string;
  }>;
  timestamp: string;
}

type EventCallback<T> = (event: T) => void;

class WebSocketService {
  private socket: WebSocket | null = null;
  private wsUrl: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private eventHandlers: Map<EventType, Set<EventCallback<any>>> = new Map();
  private connected = false;
  
  constructor(wsUrl: string = 'ws://localhost:8000') {
    this.wsUrl = wsUrl;
  }
  
  /**
   * Connect to the WebSocket server
   */
  async connect(): Promise<void> {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }
    
    return new Promise((resolve, reject) => {
      try {
        // Create WebSocket connection using native WebSocket
        console.log(`Connecting to WebSocket: ${this.wsUrl}/ws/director`);
        this.socket = new WebSocket(`${this.wsUrl}/ws/director`);
        
        this.socket.onopen = () => {
          console.log('WebSocket connected successfully');
          this.connected = true;
          this.reconnectAttempts = 0;
          
          // Send subscription message
          const subscribeMessage = {
            type: 'subscribe',
            topics: ['state_updates', 'memory_events', 'emotion_events', 'metrics']
          };
          console.log('Sending subscription:', subscribeMessage);
          this.socket!.send(JSON.stringify(subscribeMessage));
          
          resolve();
        };
        
        this.socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('Received WebSocket message:', data);
            this.handleMessage(data);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };
        
        this.socket.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.connected = false;
          reject(new Error('WebSocket connection failed'));
        };
        
        this.socket.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.connected = false;
          this.handleDisconnect();
        };
        
        // Start heartbeat
        this.startHeartbeat();
        
      } catch (error) {
        console.error('Failed to create WebSocket connection:', error);
        reject(error);
      }
    });
  }
  
  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.connected = false;
    }
  }
  
  /**
   * Subscribe to specific event types
   */
  on<T extends StateUpdateEvent | MemoryFormationEvent | EmotionChangeEvent | TripleHeadMetricsEvent | RelationshipUpdateEvent>(
    eventType: EventType,
    callback: EventCallback<T>
  ): () => void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, new Set());
    }
    
    const handlers = this.eventHandlers.get(eventType)!;
    handlers.add(callback);
    
    // Return unsubscribe function
    return () => {
      handlers.delete(callback);
    };
  }
  
  /**
   * Subscribe to relationship update events
   */
  onRelationshipEvent(callback: (event: RelationshipUpdateEvent) => void): () => void {
    return this.on(EventType.RELATIONSHIP_UPDATE, callback);
  }
  
  /**
   * Send a message to the server
   */
  send(message: any): void {
    if (this.socket && this.connected) {
      this.socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, message not sent:', message);
    }
  }
  
  /**
   * Request entity details
   */
  getEntityDetails(entityId: string): void {
    this.send({
      type: 'get_entity_details',
      entity_id: entityId
    });
  }
  
  /**
   * Request memory timeline
   */
  getMemoryTimeline(characterId: string, limit: number = 100): void {
    this.send({
      type: 'get_memory_timeline',
      character_id: characterId,
      limit
    });
  }
  
  /**
   * Request emotional state
   */
  getEmotionState(characterId: string): void {
    this.send({
      type: 'get_emotion_state',
      character_id: characterId
    });
  }
  
  /**
   * Set focus on specific entities
   */
  setFocus(entityIds: string[]): void {
    this.send({
      type: 'set_focus',
      entity_ids: entityIds
    });
  }
  
  /**
   * Update event filters
   */
  updateFilters(filters: Record<string, any>): void {
    this.send({
      type: 'filter_update',
      filters
    });
  }
  
  /**
   * Handle incoming messages
   */
  private handleMessage(message: any): void {
    const messageType = message.type;
    
    // Handle special message types
    if (messageType === 'initial_state') {
      this.handleInitialState(message.data);
      return;
    }
    
    if (messageType === 'entity_details') {
      // TODO: Handle entity details response
      return;
    }
    
    if (messageType === 'memory_timeline') {
      // TODO: Handle memory timeline response
      return;
    }
    
    if (messageType === 'emotion_state') {
      // TODO: Handle emotion state response
      return;
    }
    
    // Map message type to event type
    const eventTypeMap: Record<string, EventType> = {
      'state_update': EventType.STATE_UPDATE,
      'memory_formed': EventType.MEMORY_FORMED,
      'emotion_changed': EventType.EMOTION_CHANGED,
      'triple_head_metrics': EventType.TRIPLE_HEAD_METRICS,
      'relationship_update': EventType.RELATIONSHIP_UPDATE,
    };
    
    const eventType = eventTypeMap[messageType];
    if (eventType && this.eventHandlers.has(eventType)) {
      const handlers = this.eventHandlers.get(eventType)!;
      handlers.forEach(handler => {
        try {
          handler(message);
        } catch (error) {
          console.error(`Error in event handler for ${eventType}:`, error);
        }
      });
    }
  }
  
  /**
   * Handle initial state message
   */
  private handleInitialState(data: { world: WorldSnapshot }): void {
    // Emit a custom event for initial state
    if (this.eventHandlers.has(EventType.STATE_UPDATE)) {
      const handlers = this.eventHandlers.get(EventType.STATE_UPDATE)!;
      handlers.forEach(handler => {
        try {
          handler({
            event_type: EventType.STATE_UPDATE,
            timestamp: new Date().toISOString(),
            source: 'websocket',
            data: {
              entity_id: 'world',
              changes: { snapshot: data.world }
            }
          });
        } catch (error) {
          console.error('Error handling initial state:', error);
        }
      });
    }
  }
  
  /**
   * Handle disconnection and attempt reconnect
   */
  private handleDisconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect().catch(error => {
          console.error('Reconnection failed:', error);
        });
      }, delay);
    } else {
      console.error('Max reconnection attempts reached');
    }
  }
  
  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat(): void {
    setInterval(() => {
      if (this.connected) {
        this.send({ type: 'ping' });
      }
    }, 30000); // Every 30 seconds
  }
  
  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connected;
  }
}

// Export singleton instance
export const websocketService = new WebSocketService(
  process.env.REACT_APP_WS_URL || 'ws://localhost:8000'
);

export default websocketService; 