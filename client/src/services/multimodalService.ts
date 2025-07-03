/**
 * Multimodal Dataset Generation Service
 * 
 * Manages multimodal dataset generation including:
 * - Starting/stopping/canceling generation jobs
 * - Real-time progress tracking via WebSocket
 * - Job management and status updates
 * - Dataset download and export
 */

// API Base URL - TODO: Move to config
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';
const WS_BASE_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8001';

// Interfaces matching backend schemas
export interface MultimodalGenerationConfig {
  name: string;
  sampleCount: number;
  characterCount: number;
  narrativeTypes: string[];
  useMockTTS: boolean;
  ttsProvider: 'orpheus' | 'kokoro' | 'xtts' | 'bark';
  outputDir: string;
  batchSize: number;
  temperature: number;
}

export interface MultimodalJob {
  id: string;
  name: string;
  status: 'pending' | 'generating' | 'completed' | 'failed';
  progress: number;
  config: MultimodalGenerationConfig;
  samplesGenerated: number;
  totalSamples: number;
  currentStep: string;
  outputPath?: string;
  errorMessage?: string;
  createdAt: string;
  updatedAt: string;
  warnings?: string[];
}

export interface MultimodalProgressUpdate {
  status: string;
  progress: number;
  currentStep: string;
  samplesGenerated: number;
  totalSamples?: number;
  outputPath?: string;
  error?: string;
}

// WebSocket event callbacks
type ProgressCallback = (update: MultimodalProgressUpdate) => void;
type ErrorCallback = (error: string) => void;

class MultimodalService {
  private wsConnections: Map<string, WebSocket> = new Map();
  private progressCallbacks: Map<string, ProgressCallback> = new Map();
  private errorCallbacks: Map<string, ErrorCallback> = new Map();

  /**
   * Start a new multimodal dataset generation job
   */
  async startGeneration(config: MultimodalGenerationConfig): Promise<MultimodalJob> {
    try {
      console.log('Starting multimodal generation with config:', config);
      
      const response = await fetch(`${API_BASE_URL}/api/v1/multimodal/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // TODO: Add auth headers when auth is implemented
          // 'Authorization': `Bearer ${getAuthToken()}`
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const job: MultimodalJob = await response.json();
      console.log('Generation job started:', job);
      
      return job;
    } catch (error) {
      console.error('Failed to start generation:', error);
      throw error;
    }
  }

  /**
   * Get details of a specific generation job
   */
  async getJob(jobId: string): Promise<MultimodalJob> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/multimodal/jobs/${jobId}`, {
        headers: {
          // TODO: Add auth headers when auth is implemented
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`Failed to get job ${jobId}:`, error);
      throw error;
    }
  }

  /**
   * Get current progress of a generation job
   */
  async getProgress(jobId: string): Promise<MultimodalProgressUpdate> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/multimodal/jobs/${jobId}/progress`, {
        headers: {
          // TODO: Add auth headers when auth is implemented
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`Failed to get progress for job ${jobId}:`, error);
      throw error;
    }
  }

  /**
   * Cancel a running generation job
   */
  async cancelJob(jobId: string): Promise<void> {
    try {
      console.log(`Canceling job ${jobId}`);
      
      const response = await fetch(`${API_BASE_URL}/api/v1/multimodal/jobs/${jobId}/cancel`, {
        method: 'POST',
        headers: {
          // TODO: Add auth headers when auth is implemented
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      console.log(`Job ${jobId} canceled successfully`);
    } catch (error) {
      console.error(`Failed to cancel job ${jobId}:`, error);
      throw error;
    }
  }

  /**
   * Download completed dataset
   */
  async downloadDataset(jobId: string): Promise<Blob> {
    try {
      console.log(`Downloading dataset for job ${jobId}`);
      
      const response = await fetch(`${API_BASE_URL}/api/v1/multimodal/jobs/${jobId}/download`, {
        headers: {
          // TODO: Add auth headers when auth is implemented
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.blob();
    } catch (error) {
      console.error(`Failed to download dataset for job ${jobId}:`, error);
      throw error;
    }
  }

  /**
   * Start real-time progress monitoring via WebSocket
   */
  subscribeToProgress(
    jobId: string, 
    onProgress: ProgressCallback, 
    onError?: ErrorCallback
  ): () => void {
    try {
      // Close existing connection if any
      this.unsubscribeFromProgress(jobId);

      console.log(`Connecting to WebSocket for job ${jobId}`);
      
      const ws = new WebSocket(`${WS_BASE_URL}/api/v1/multimodal/ws/${jobId}`);
      
      ws.onopen = () => {
        console.log(`WebSocket connected for job ${jobId}`);
      };
      
      ws.onmessage = (event) => {
        try {
          const update: MultimodalProgressUpdate = JSON.parse(event.data);
          console.log(`Progress update for job ${jobId}:`, update);
          onProgress(update);
        } catch (error) {
          console.error('Failed to parse progress update:', error);
          onError?.(error as string);
        }
      };
      
      ws.onerror = (error) => {
        console.error(`WebSocket error for job ${jobId}:`, error);
        onError?.('WebSocket connection error');
      };
      
      ws.onclose = (event) => {
        console.log(`WebSocket closed for job ${jobId}:`, event.code, event.reason);
        this.wsConnections.delete(jobId);
        this.progressCallbacks.delete(jobId);
        this.errorCallbacks.delete(jobId);
      };

      // Store connections and callbacks
      this.wsConnections.set(jobId, ws);
      this.progressCallbacks.set(jobId, onProgress);
      if (onError) {
        this.errorCallbacks.set(jobId, onError);
      }

      // Return unsubscribe function
      return () => this.unsubscribeFromProgress(jobId);
      
    } catch (error) {
      console.error(`Failed to subscribe to progress for job ${jobId}:`, error);
      onError?.(error as string);
      return () => {}; // Return empty unsubscribe function
    }
  }

  /**
   * Stop progress monitoring for a job
   */
  unsubscribeFromProgress(jobId: string): void {
    const ws = this.wsConnections.get(jobId);
    if (ws) {
      ws.close();
      this.wsConnections.delete(jobId);
      this.progressCallbacks.delete(jobId);
      this.errorCallbacks.delete(jobId);
      console.log(`Unsubscribed from progress updates for job ${jobId}`);
    }
  }

  /**
   * Stop all progress monitoring
   */
  disconnectAll(): void {
    console.log('Disconnecting all WebSocket connections');
    for (const [jobId, ws] of this.wsConnections) {
      ws.close();
    }
    this.wsConnections.clear();
    this.progressCallbacks.clear();
    this.errorCallbacks.clear();
  }

  /**
   * Get list of all jobs (future feature)
   */
  async getAllJobs(): Promise<MultimodalJob[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/multimodal/jobs`, {
        headers: {
          // TODO: Add auth headers when auth is implemented
        },
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Failed to get all jobs:', error);
      throw error;
    }
  }

  /**
   * Helper method to trigger file download from blob
   */
  downloadFile(blob: Blob, filename: string): void {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }

  /**
   * Estimate generation time based on configuration
   */
  estimateGenerationTime(config: MultimodalGenerationConfig): string {
    const baseTimePerSample = config.useMockTTS ? 0.5 : 3; // seconds
    const totalTime = config.sampleCount * baseTimePerSample;
    
    if (totalTime < 60) {
      return `${Math.ceil(totalTime)} seconds`;
    } else if (totalTime < 3600) {
      return `${Math.ceil(totalTime / 60)} minutes`;
    } else {
      return `${Math.ceil(totalTime / 3600)} hours`;
    }
  }

  /**
   * Validate generation configuration
   */
  validateConfig(config: MultimodalGenerationConfig): string[] {
    const errors: string[] = [];

    if (!config.name || config.name.trim().length === 0) {
      errors.push('Dataset name is required');
    }

    if (config.sampleCount < 10 || config.sampleCount > 100000) {
      errors.push('Sample count must be between 10 and 100,000');
    }

    if (config.characterCount < 1 || config.characterCount > 100) {
      errors.push('Character count must be between 1 and 100');
    }

    if (!config.narrativeTypes || config.narrativeTypes.length === 0) {
      errors.push('At least one narrative type must be selected');
    }

    if (config.temperature < 0 || config.temperature > 2) {
      errors.push('Temperature must be between 0 and 2');
    }

    if (config.batchSize < 1 || config.batchSize > 100) {
      errors.push('Batch size must be between 1 and 100');
    }

    return errors;
  }
}

// Create singleton instance
const multimodalService = new MultimodalService();

export default multimodalService; 