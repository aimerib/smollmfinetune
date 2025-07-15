import { useState, useEffect, useCallback, useRef } from 'react';

interface TrainingStatus {
  status: 'idle' | 'training' | 'paused' | 'completed' | 'error';
  phase?: 'small' | 'medium' | 'large' | 'production';
  current_step?: number;
  max_steps?: number;
  start_time?: string;
  estimated_completion?: string;
  error_message?: string;
}

interface TrainingMetrics {
  current_step: number;
  total_loss: number;
  text_loss: number;
  speech_loss: number;
  control_loss: number;
  memory_loss: number;
  alignment_loss: number;
  learning_rate: number;
  gradient_norm: number;
  samples_per_second: number;
  gpu_memory_usage: number;
  timestamp: string;
}

interface DatasetGenerationConfig {
  num_characters: number;
  conversations_per_character: number;
  multimodal_ratio: number;
  topics?: string[];
  style?: string;
}

interface TrainingConfig {
  model_size: 'small' | 'medium' | 'large';
  batch_size: number;
  learning_rate: number;
  max_steps: number;
  save_steps: number;
  eval_steps: number;
  num_characters: number;
}

interface ValidationResults {
  status: 'passed' | 'failed';
  issues: string[];
  recommendations: string[];
}

// Enhanced WebSocket message types from the diffusion trainer
interface WebSocketMessage {
  type: 'connection_established' | 'training_step' | 'validation' | 'epoch_start' | 'epoch_end' | 
        'checkpoint_saved' | 'best_model_saved' | 'training_complete' | 'validation_progress' | 'error';
  timestamp: number;
  [key: string]: any;
}

export const useDiffusionTraining = () => {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({ status: 'idle' });
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [liveUpdates, setLiveUpdates] = useState<WebSocketMessage[]>([]);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const API_BASE = 'http://localhost:8000/api/v1';
  const WS_BASE = 'ws://localhost:8765'; // Enhanced diffusion trainer WebSocket port

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }
    
    setConnectionStatus('connecting');
    
    try {
      const ws = new WebSocket(WS_BASE);
      
      ws.onopen = () => {
        console.log('🔗 Connected to diffusion training WebSocket');
        setConnectionStatus('connected');
        reconnectAttempts.current = 0;
        
        // Send ping to test connection
        ws.send(JSON.stringify({ type: 'ping' }));
      };
      
      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          console.log('📨 Training update:', message);
          
          // Add to live updates feed
          setLiveUpdates(prev => [...prev.slice(-19), message]); // Keep last 20 messages
          
          // Handle different message types
          switch (message.type) {
            case 'training_step':
              setTrainingMetrics({
                current_step: message.step || 0,
                total_loss: message.loss || 0,
                text_loss: 0, // Will be updated when available
                speech_loss: 0,
                control_loss: 0,
                memory_loss: 0,
                alignment_loss: 0,
                learning_rate: message.learning_rate || 0,
                gradient_norm: 0,
                samples_per_second: 0,
                gpu_memory_usage: message.gpu_memory || 0,
                timestamp: new Date(message.timestamp * 1000).toISOString()
              });
              setTrainingStatus(prev => ({ 
                ...prev, 
                current_step: message.step,
                status: 'training' 
              }));
              break;
              
            case 'epoch_start':
              setTrainingStatus(prev => ({ 
                ...prev, 
                status: 'training'
              }));
              break;
              
            case 'epoch_end':
              // Update with epoch completion info
              break;
              
            case 'validation':
              // Handle validation updates
              break;
              
            case 'training_complete':
              setTrainingStatus(prev => ({ 
                ...prev, 
                status: 'completed' 
              }));
              break;
              
            case 'error':
              setError(message.error || 'Training error occurred');
              setTrainingStatus(prev => ({ 
                ...prev, 
                status: 'error',
                error_message: message.error 
              }));
              break;
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setConnectionStatus('disconnected');
      };
      
      ws.onclose = (event) => {
        console.log('🔌 WebSocket disconnected:', event.code, event.reason);
        setConnectionStatus('disconnected');
        wsRef.current = null;
        
        // Attempt reconnection if training is active
        if (trainingStatus.status === 'training' && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current += 1;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          
          console.log(`🔄 Attempting reconnection ${reconnectAttempts.current}/${maxReconnectAttempts} in ${delay}ms`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, delay);
        }
      };
      
      wsRef.current = ws;
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionStatus('disconnected');
    }
  }, [trainingStatus.status]);
  
  const disconnectWebSocket = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setConnectionStatus('disconnected');
  }, []);

  // Auto-connect WebSocket when training starts
  useEffect(() => {
    if (trainingStatus.status === 'training') {
      connectWebSocket();
    } else if (trainingStatus.status === 'idle' || trainingStatus.status === 'completed') {
      disconnectWebSocket();
    }

    return () => {
      disconnectWebSocket();
    };
  }, [trainingStatus.status, connectWebSocket, disconnectWebSocket]);

  const generateDataset = useCallback(async (config: DatasetGenerationConfig) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/diffusion/dataset/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        throw new Error(`Dataset generation failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(errorMessage);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const validateModel = useCallback(async (): Promise<ValidationResults> => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/diffusion/model/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Model validation failed: ${response.statusText}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(errorMessage);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const startTraining = useCallback(async (config: TrainingConfig) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/diffusion/training/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        throw new Error(`Training start failed: ${response.statusText}`);
      }

      const result = await response.json();
      setTrainingStatus({ 
        status: 'training',
        phase: config.model_size,
        max_steps: config.max_steps,
        start_time: new Date().toISOString()
      });
      
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(errorMessage);
      setTrainingStatus({ status: 'error', error_message: errorMessage });
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const pauseTraining = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/diffusion/training/pause`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Training pause failed: ${response.statusText}`);
      }

      setTrainingStatus(prev => ({ ...prev, status: 'paused' }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(errorMessage);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const resumeTraining = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/diffusion/training/resume`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Training resume failed: ${response.statusText}`);
      }

      setTrainingStatus(prev => ({ ...prev, status: 'training' }));
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(errorMessage);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const stopTraining = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/diffusion/training/stop`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Training stop failed: ${response.statusText}`);
      }

      setTrainingStatus({ status: 'idle' });
      setTrainingMetrics(null);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(errorMessage);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getTrainingHistory = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/diffusion/training/history`);
      if (!response.ok) {
        throw new Error(`Failed to fetch training history: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch training history:', error);
      return [];
    }
  }, []);

  const getModelCheckpoints = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/diffusion/model/checkpoints`);
      if (!response.ok) {
        throw new Error(`Failed to fetch checkpoints: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Failed to fetch checkpoints:', error);
      return [];
    }
  }, []);

  const exportModel = useCallback(async (checkpointId: string, format: 'onnx' | 'tensorrt' | 'cartridge') => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/diffusion/model/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          checkpoint_id: checkpointId,
          format: format
        }),
      });

      if (!response.ok) {
        throw new Error(`Model export failed: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setError(errorMessage);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  return {
    trainingStatus,
    trainingMetrics,
    isLoading,
    error,
    connectionStatus,
    liveUpdates,
    
    // Actions
    generateDataset,
    validateModel,
    startTraining,
    pauseTraining,
    resumeTraining,
    stopTraining,
    getTrainingHistory,
    getModelCheckpoints,
    exportModel,
    
    // WebSocket controls
    connectWebSocket,
    disconnectWebSocket,
    
    // Utilities
    clearError: () => setError(null),
    clearLiveUpdates: () => setLiveUpdates([]),
  };
}; 