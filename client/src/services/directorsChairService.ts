/**
 * Director's Chair Service
 * 
 * Handles communication with the Director's Chair API endpoints
 * for real-time training, conversation editing, and multi-head corrections.
 */

export interface DirectorsChairSession {
  session_id: string;
  character_id: string;
  mode: string;
  training_enabled: boolean;
}

export interface ConversationTurn {
  conversation_id: string;
  user_message: string;
  assistant_response: string;
  metadata: any;
  editable: boolean;
}

export interface Correction {
  type: string;
  target_head: string;
  original_text?: string;
  corrected_text?: string;
  emotional_state?: string;
  control_tokens?: string[];
  memory_importance?: number;
  should_remember?: boolean;
  reason: string;
}

export interface PreferencePair {
  head_type: string;
  prompt?: string;
  context?: string;
  chosen_response: string;
  rejected_response: string;
  character_id: string;
  correction_reason: string;
  quality_metrics?: Record<string, number>;
  control_metrics?: Record<string, number>;
  memory_metrics?: Record<string, number>;
}

export interface TrainingStatus {
  generation_head?: {
    status: string;
    progress: number;
    queue_length?: number;
    eta?: string;
    metrics?: any;
  };
  control_head?: {
    status: string;
    progress: number;
    queue_length?: number;
    eta?: string;
    metrics?: any;
  };
  memory_head?: {
    status: string;
    progress: number;
    queue_length?: number;
    eta?: string;
    metrics?: any;
  };
  overall_progress?: number;
}

class DirectorsChairService {
  private baseUrl = 'http://localhost:8000/directors-chair';

  async startSession(sessionData: {
    character_id: string;
    mode?: string;
    training_enabled?: boolean;
  }): Promise<DirectorsChairSession> {
    const response = await fetch(`${this.baseUrl}/sessions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(sessionData),
    });
    
    if (!response.ok) {
      throw new Error('Failed to start Director\'s Chair session');
    }
    
    return response.json();
  }

  async createConversationTurn(
    sessionId: string,
    conversationData: {
      user_message: string;
      assistant_response: string;
      metadata?: any;
    }
  ): Promise<ConversationTurn> {
    const response = await fetch(`${this.baseUrl}/sessions/${sessionId}/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(conversationData),
    });
    
    if (!response.ok) {
      throw new Error('Failed to create conversation turn');
    }
    
    return response.json();
  }

  async applyCorrection(
    conversationId: string,
    correction: Correction
  ): Promise<{ correction_id: string; target_head: string; queued_for_training: boolean }> {
    const response = await fetch(`${this.baseUrl}/conversations/${conversationId}/corrections`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(correction),
    });
    
    if (!response.ok) {
      throw new Error('Failed to apply correction');
    }
    
    return response.json();
  }

  async createPreferencePair(preference: PreferencePair): Promise<any> {
    const response = await fetch(`${this.baseUrl}/preferences`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(preference),
    });
    
    if (!response.ok) {
      throw new Error('Failed to create preference pair');
    }
    
    return response.json();
  }

  async getTrainingStatus(): Promise<TrainingStatus> {
    const response = await fetch(`${this.baseUrl}/training/status`);
    
    if (!response.ok) {
      throw new Error('Failed to get training status');
    }
    
    return response.json();
  }

  async getHeadTrainingStatus(headType: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/training/${headType}/status`);
    
    if (!response.ok) {
      throw new Error(`Failed to get ${headType} head training status`);
    }
    
    return response.json();
  }

  async startTraining(config: {
    heads: string[];
    batch_size?: number;
    learning_rate?: number;
    max_training_time_minutes?: number;
    coordination_weight?: number;
  }): Promise<{ training_job_id: string; status: string; estimated_completion_time: string }> {
    const response = await fetch(`${this.baseUrl}/training/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });
    
    if (!response.ok) {
      throw new Error('Failed to start training');
    }
    
    return response.json();
  }

  async getAnalytics(headType: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/analytics/${headType}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get ${headType} analytics`);
    }
    
    return response.json();
  }

  async getCoordinationMetrics(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/analytics/coordination`);
    
    if (!response.ok) {
      throw new Error('Failed to get coordination metrics');
    }
    
    return response.json();
  }
}

export const directorsChairService = new DirectorsChairService();
export default directorsChairService; 