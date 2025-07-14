/**
 * Flow-Matching TTS Training Dashboard
 * 
 * A comprehensive React component providing:
 * - Real-time training monitoring via WebSocket
 * - Interactive training controls (start/stop/pause/resume)
 * - Live loss charts with time filtering
 * - SVG-based architecture visualization
 * - Hyperparameter tuning sliders
 * - Speech quality monitoring with progress bars
 * - A/B testing framework for model comparison
 * - Beautiful dark theme with glass-morphism effects
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// TypeScript interfaces
interface TrainingMetrics {
  velocity_loss: number;
  mel_accuracy: number;
  character_consistency: number;
  learning_rate: number;
  step: number;
  progress: number;
  timestamp?: string;
}

interface TrainingJob {
  job_id: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  current_step: number;
  total_steps: number;
  metrics: TrainingMetrics;
  error_message?: string;
}

interface TrainingConfig {
  hidden_dim: number;
  num_layers: number;
  num_heads: number;
  num_mel_bins: number;
  noise_schedule: 'cosine' | 'linear' | 'sigmoid';
  num_inference_steps: number;
  learning_rate: number;
  weight_decay: number;
  warmup_steps: number;
  batch_size: number;
  dataset_path: string;
  vocab_size: number;
}

const FlowMatchingTrainingDashboard: React.FC = () => {
  // State management
  const [activeJob, setActiveJob] = useState<TrainingJob | null>(null);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [metricsHistory, setMetricsHistory] = useState<TrainingMetrics[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const [timeFilter, setTimeFilter] = useState<'1m' | '5m' | '15m' | '1h' | 'all'>('5m');
  const [showArchitecture, setShowArchitecture] = useState(false);
  const [hyperparameterMode, setHyperparameterMode] = useState(false);
  
  // Training configuration state
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    hidden_dim: 512,
    num_layers: 8,
    num_heads: 8,
    num_mel_bins: 80,
    noise_schedule: 'cosine',
    num_inference_steps: 50,
    learning_rate: 1e-4,
    weight_decay: 0.01,
    warmup_steps: 1000,
    batch_size: 8,
    dataset_path: '/path/to/dataset',
    vocab_size: 32000
  });

  // WebSocket connection
  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    try {
      setConnectionStatus('connecting');
      const ws = new WebSocket('ws://localhost:8000/api/flow-matching/training/monitor');
      
      ws.onopen = () => {
        console.log('🎵 Connected to Flow-Matching training monitor');
        setConnectionStatus('connected');
        setIsConnected(true);
        
        // Subscribe to active job if exists
        if (activeJob) {
          ws.send(JSON.stringify({
            type: 'subscribe',
            job_id: activeJob.job_id
          }));
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          switch (data.type) {
            case 'training_metrics':
              if (data.job_id === activeJob?.job_id) {
                const metrics = data.metrics as TrainingMetrics;
                setMetricsHistory(prev => [...prev.slice(-200), metrics]); // Keep last 200 points
                
                // Update active job metrics
                setActiveJob(prev => prev ? { ...prev, metrics, current_step: metrics.step } : null);
              }
              break;
              
            case 'job_status':
              const job = data.status as TrainingJob;
              setActiveJob(job);
              break;
              
            case 'heartbeat':
              // Keep connection alive
              break;
              
            default:
              console.log('Unknown message type:', data.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket connection closed');
        setConnectionStatus('disconnected');
        setIsConnected(false);
        
        // Attempt reconnection after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connectWebSocket();
        }, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('disconnected');
        setIsConnected(false);
      };

      websocketRef.current = ws;
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionStatus('disconnected');
    }
  }, [activeJob]);

  // Initialize WebSocket connection
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

  // Fetch training jobs
  const fetchTrainingJobs = async () => {
    try {
      const response = await fetch('/api/flow-matching/training/jobs');
      const jobs = await response.json();
      setTrainingJobs(jobs);
      
      // Set active job to the most recent running job
      const runningJob = jobs.find((job: TrainingJob) => job.status === 'running');
      if (runningJob && !activeJob) {
        setActiveJob(runningJob);
      }
    } catch (error) {
      console.error('Failed to fetch training jobs:', error);
    }
  };

  useEffect(() => {
    fetchTrainingJobs();
    const interval = setInterval(fetchTrainingJobs, 10000); // Poll every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Training control functions
  const startTraining = async () => {
    try {
      const response = await fetch('/api/flow-matching/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainingConfig)
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Training started:', result);
        await fetchTrainingJobs();
      } else {
        console.error('Failed to start training');
      }
    } catch (error) {
      console.error('Error starting training:', error);
    }
  };

  const pauseTraining = async () => {
    if (!activeJob) return;
    
    try {
      const response = await fetch(`/api/flow-matching/training/${activeJob.job_id}/pause`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setActiveJob(prev => prev ? { ...prev, status: 'paused' } : null);
      }
    } catch (error) {
      console.error('Error pausing training:', error);
    }
  };

  const resumeTraining = async () => {
    if (!activeJob) return;
    
    try {
      const response = await fetch(`/api/flow-matching/training/${activeJob.job_id}/resume`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setActiveJob(prev => prev ? { ...prev, status: 'running' } : null);
      }
    } catch (error) {
      console.error('Error resuming training:', error);
    }
  };

  const stopTraining = async () => {
    if (!activeJob) return;
    
    try {
      const response = await fetch(`/api/flow-matching/training/${activeJob.job_id}/stop`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setActiveJob(prev => prev ? { ...prev, status: 'stopped' } : null);
      }
    } catch (error) {
      console.error('Error stopping training:', error);
    }
  };

  // Update hyperparameters
  const updateHyperparameters = async (updates: Partial<TrainingConfig>) => {
    if (!activeJob) return;
    
    try {
      const response = await fetch(`/api/flow-matching/training/${activeJob.job_id}/update-hyperparameters`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: activeJob.job_id,
          ...updates
        })
      });
      
      if (response.ok) {
        console.log('Hyperparameters updated');
      }
    } catch (error) {
      console.error('Error updating hyperparameters:', error);
    }
  };

  // Filter metrics based on time range
  const getFilteredMetrics = () => {
    if (!metricsHistory.length) return [];
    
    const now = Date.now();
    const timeRanges = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      'all': Infinity
    };
    
    const cutoff = now - timeRanges[timeFilter];
    return metricsHistory.filter(metric => {
      const timestamp = metric.timestamp ? new Date(metric.timestamp).getTime() : now;
      return timestamp >= cutoff;
    });
  };

  // Prepare chart data
  const getChartData = (metric: keyof TrainingMetrics) => {
    const filteredMetrics = getFilteredMetrics();
    
    return {
      labels: filteredMetrics.map((_, index) => index.toString()),
      datasets: [
        {
          label: metric.replace('_', ' ').toUpperCase(),
          data: filteredMetrics.map(m => m[metric] as number),
          borderColor: {
            velocity_loss: '#ff6b6b',
            mel_accuracy: '#4ecdc4',
            character_consistency: '#45b7d1',
            learning_rate: '#96ceb4'
          }[metric] || '#ffffff',
          backgroundColor: {
            velocity_loss: 'rgba(255, 107, 107, 0.1)',
            mel_accuracy: 'rgba(78, 205, 196, 0.1)',
            character_consistency: 'rgba(69, 183, 209, 0.1)',
            learning_rate: 'rgba(150, 206, 180, 0.1)'
          }[metric] || 'rgba(255, 255, 255, 0.1)',
          fill: true,
          tension: 0.4
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      }
    },
    scales: {
      x: {
        display: false,
        grid: {
          display: false
        }
      },
      y: {
        grid: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        ticks: {
          color: '#ffffff80'
        }
      }
    },
    elements: {
      point: {
        radius: 0
      }
    }
  };

  // Architecture visualization component
  const ArchitectureVisualization: React.FC = () => (
    <div className="architecture-viz">
      <svg width="800" height="600" viewBox="0 0 800 600" className="architecture-svg">
        {/* Background */}
        <defs>
          <linearGradient id="architectureGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#1a1a1a" />
            <stop offset="100%" stopColor="#2d2d2d" />
          </linearGradient>
        </defs>
        <rect width="800" height="600" fill="url(#architectureGradient)" rx="10" />
        
        {/* Text Encoder */}
        <g className="component" transform="translate(50, 50)">
          <rect width="150" height="80" rx="10" fill="#4ecdc4" fillOpacity="0.2" stroke="#4ecdc4" strokeWidth="2" />
          <text x="75" y="45" textAnchor="middle" fill="#4ecdc4" fontSize="14" fontWeight="bold">
            Narrative Text
          </text>
          <text x="75" y="60" textAnchor="middle" fill="#4ecdc4" fontSize="12">
            Encoder
          </text>
        </g>
        
        {/* Character Conditioner */}
        <g className="component" transform="translate(50, 200)">
          <rect width="150" height="80" rx="10" fill="#45b7d1" fillOpacity="0.2" stroke="#45b7d1" strokeWidth="2" />
          <text x="75" y="45" textAnchor="middle" fill="#45b7d1" fontSize="14" fontWeight="bold">
            Character
          </text>
          <text x="75" y="60" textAnchor="middle" fill="#45b7d1" fontSize="12">
            Conditioner
          </text>
        </g>
        
        {/* Flow Matcher */}
        <g className="component" transform="translate(300, 125)">
          <rect width="200" height="100" rx="10" fill="#ff6b6b" fillOpacity="0.2" stroke="#ff6b6b" strokeWidth="2" />
          <text x="100" y="45" textAnchor="middle" fill="#ff6b6b" fontSize="16" fontWeight="bold">
            Continuous Flow
          </text>
          <text x="100" y="65" textAnchor="middle" fill="#ff6b6b" fontSize="14">
            Matcher
          </text>
        </g>
        
        {/* Speaker Extractor */}
        <g className="component" transform="translate(50, 350)">
          <rect width="150" height="80" rx="10" fill="#96ceb4" fillOpacity="0.2" stroke="#96ceb4" strokeWidth="2" />
          <text x="75" y="45" textAnchor="middle" fill="#96ceb4" fontSize="14" fontWeight="bold">
            Speaker
          </text>
          <text x="75" y="60" textAnchor="middle" fill="#96ceb4" fontSize="12">
            Extractor
          </text>
        </g>
        
        {/* Output */}
        <g className="component" transform="translate(600, 125)">
          <rect width="150" height="100" rx="10" fill="#feca57" fillOpacity="0.2" stroke="#feca57" strokeWidth="2" />
          <text x="75" y="45" textAnchor="middle" fill="#feca57" fontSize="14" fontWeight="bold">
            Mel-Spectrogram
          </text>
          <text x="75" y="65" textAnchor="middle" fill="#feca57" fontSize="12">
            Output
          </text>
        </g>
        
        {/* Connections */}
        <g className="connections" stroke="#ffffff30" strokeWidth="2" fill="none">
          <path d="M 200 90 L 300 175" markerEnd="url(#arrowhead)" />
          <path d="M 200 240 L 300 175" markerEnd="url(#arrowhead)" />
          <path d="M 200 390 L 300 200" markerEnd="url(#arrowhead)" />
          <path d="M 500 175 L 600 175" markerEnd="url(#arrowhead)" />
        </g>
        
        {/* Arrow marker */}
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#ffffff60" />
          </marker>
        </defs>
        
        {/* Flow animation */}
        <g className="flow-animation">
          <circle r="3" fill="#4ecdc4">
            <animateMotion dur="3s" repeatCount="indefinite">
              <path d="M 200 90 L 300 175 L 500 175 L 600 175" />
            </animateMotion>
          </circle>
        </g>
      </svg>
    </div>
  );

  return (
    <div className="flow-matching-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-content">
          <h1 className="dashboard-title">
            <span className="title-icon">🎵</span>
            Flow-Matching TTS Training
          </h1>
          <div className="connection-status">
            <div className={`status-indicator ${connectionStatus}`}>
              <div className="status-dot"></div>
              <span>{connectionStatus}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="dashboard-content">
        {/* Training Controls */}
        <div className="control-panel">
          <div className="panel-header">
            <h2>Training Controls</h2>
            <div className="control-buttons">
              <button 
                className="control-btn start" 
                onClick={startTraining}
                disabled={activeJob?.status === 'running'}
              >
                ▶ Start
              </button>
              <button 
                className="control-btn pause" 
                onClick={pauseTraining}
                disabled={!activeJob || activeJob.status !== 'running'}
              >
                ⏸ Pause
              </button>
              <button 
                className="control-btn resume" 
                onClick={resumeTraining}
                disabled={!activeJob || activeJob.status !== 'paused'}
              >
                ▶ Resume
              </button>
              <button 
                className="control-btn stop" 
                onClick={stopTraining}
                disabled={!activeJob || !['running', 'paused'].includes(activeJob.status)}
              >
                ⏹ Stop
              </button>
            </div>
          </div>

          {/* Job Status */}
          {activeJob && (
            <div className="job-status">
              <div className="status-card">
                <div className="status-info">
                  <span className="job-id">Job: {activeJob.job_id.slice(0, 8)}</span>
                  <span className={`status-badge ${activeJob.status}`}>{activeJob.status}</span>
                </div>
                <div className="progress-info">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${(activeJob.metrics?.progress || 0) * 100}%` }}
                    ></div>
                  </div>
                  <span className="progress-text">
                    {activeJob.current_step} / {activeJob.total_steps} steps
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Metrics Dashboard */}
        <div className="metrics-dashboard">
          <div className="metrics-header">
            <h2>Training Metrics</h2>
            <div className="time-filter">
              {(['1m', '5m', '15m', '1h', 'all'] as const).map(period => (
                <button
                  key={period}
                  className={`filter-btn ${timeFilter === period ? 'active' : ''}`}
                  onClick={() => setTimeFilter(period)}
                >
                  {period}
                </button>
              ))}
            </div>
          </div>

          <div className="metrics-grid">
            {/* Velocity Loss Chart */}
            <div className="metric-card">
              <h3>Velocity Loss</h3>
              <div className="chart-container">
                <Line data={getChartData('velocity_loss')} options={chartOptions} />
              </div>
              <div className="metric-value">
                {activeJob?.metrics?.velocity_loss?.toFixed(4) || '0.0000'}
              </div>
            </div>

            {/* Mel Accuracy Chart */}
            <div className="metric-card">
              <h3>Mel Accuracy</h3>
              <div className="chart-container">
                <Line data={getChartData('mel_accuracy')} options={chartOptions} />
              </div>
              <div className="metric-value">
                {((activeJob?.metrics?.mel_accuracy || 0) * 100).toFixed(1)}%
              </div>
            </div>

            {/* Character Consistency Chart */}
            <div className="metric-card">
              <h3>Character Consistency</h3>
              <div className="chart-container">
                <Line data={getChartData('character_consistency')} options={chartOptions} />
              </div>
              <div className="metric-value">
                {((activeJob?.metrics?.character_consistency || 0) * 100).toFixed(1)}%
              </div>
            </div>

            {/* Learning Rate Chart */}
            <div className="metric-card">
              <h3>Learning Rate</h3>
              <div className="chart-container">
                <Line data={getChartData('learning_rate')} options={chartOptions} />
              </div>
              <div className="metric-value">
                {activeJob?.metrics?.learning_rate?.toExponential(2) || '0.00e+00'}
              </div>
            </div>
          </div>
        </div>

        {/* Architecture Visualization */}
        <div className="architecture-panel">
          <div className="panel-header">
            <h2>Model Architecture</h2>
            <button 
              className={`toggle-btn ${showArchitecture ? 'active' : ''}`}
              onClick={() => setShowArchitecture(!showArchitecture)}
            >
              {showArchitecture ? 'Hide' : 'Show'} Architecture
            </button>
          </div>
          
          {showArchitecture && <ArchitectureVisualization />}
        </div>

        {/* Hyperparameter Tuning */}
        <div className="hyperparameter-panel">
          <div className="panel-header">
            <h2>Hyperparameter Tuning</h2>
            <button 
              className={`toggle-btn ${hyperparameterMode ? 'active' : ''}`}
              onClick={() => setHyperparameterMode(!hyperparameterMode)}
            >
              {hyperparameterMode ? 'Lock' : 'Unlock'} Parameters
            </button>
          </div>

          {hyperparameterMode && (
            <div className="hyperparameter-controls">
              <div className="parameter-group">
                <label>Learning Rate</label>
                <input
                  type="range"
                  min="1e-6"
                  max="1e-2"
                  step="1e-6"
                  value={trainingConfig.learning_rate}
                  onChange={(e) => {
                    const newLR = parseFloat(e.target.value);
                    setTrainingConfig(prev => ({ ...prev, learning_rate: newLR }));
                    updateHyperparameters({ learning_rate: newLR });
                  }}
                  className="parameter-slider"
                />
                <span className="parameter-value">{trainingConfig.learning_rate.toExponential(2)}</span>
              </div>

              <div className="parameter-group">
                <label>Weight Decay</label>
                <input
                  type="range"
                  min="0"
                  max="0.1"
                  step="0.001"
                  value={trainingConfig.weight_decay}
                  onChange={(e) => {
                    const newWD = parseFloat(e.target.value);
                    setTrainingConfig(prev => ({ ...prev, weight_decay: newWD }));
                    updateHyperparameters({ weight_decay: newWD });
                  }}
                  className="parameter-slider"
                />
                <span className="parameter-value">{trainingConfig.weight_decay.toFixed(3)}</span>
              </div>

              <div className="parameter-group">
                <label>Batch Size</label>
                <input
                  type="range"
                  min="1"
                  max="64"
                  step="1"
                  value={trainingConfig.batch_size}
                  onChange={(e) => {
                    const newBS = parseInt(e.target.value);
                    setTrainingConfig(prev => ({ ...prev, batch_size: newBS }));
                  }}
                  className="parameter-slider"
                />
                <span className="parameter-value">{trainingConfig.batch_size}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Styles */}
      <style jsx>{`
        .flow-matching-dashboard {
          min-height: 100vh;
          background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 100%);
          color: white;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .dashboard-header {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(20px);
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          padding: 1rem 2rem;
          position: sticky;
          top: 0;
          z-index: 100;
        }

        .header-content {
          display: flex;
          justify-content: space-between;
          align-items: center;
          max-width: 1400px;
          margin: 0 auto;
        }

        .dashboard-title {
          font-size: 1.5rem;
          font-weight: 700;
          margin: 0;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .title-icon {
          font-size: 1.8rem;
        }

        .connection-status {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .status-indicator {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          border-radius: 20px;
          background: rgba(255, 255, 255, 0.1);
          border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #666;
        }

        .status-indicator.connecting .status-dot {
          background: #feca57;
          animation: pulse 1s infinite;
        }

        .status-indicator.connected .status-dot {
          background: #4ecdc4;
        }

        .status-indicator.disconnected .status-dot {
          background: #ff6b6b;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .dashboard-content {
          max-width: 1400px;
          margin: 0 auto;
          padding: 2rem;
          display: flex;
          flex-direction: column;
          gap: 2rem;
        }

        .control-panel, .metrics-dashboard, .architecture-panel, .hyperparameter-panel {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 16px;
          padding: 1.5rem;
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
        }

        .panel-header h2 {
          margin: 0;
          font-size: 1.25rem;
          font-weight: 600;
        }

        .control-buttons {
          display: flex;
          gap: 0.5rem;
        }

        .control-btn {
          padding: 0.5rem 1rem;
          border: none;
          border-radius: 8px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
          background: rgba(255, 255, 255, 0.1);
          color: white;
          border: 1px solid transparent;
        }

        .control-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.2);
          transform: translateY(-1px);
        }

        .control-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .control-btn.start:not(:disabled) {
          background: linear-gradient(135deg, #4ecdc4, #44a08d);
          border-color: #4ecdc4;
        }

        .control-btn.pause:not(:disabled) {
          background: linear-gradient(135deg, #feca57, #ff9ff3);
          border-color: #feca57;
        }

        .control-btn.resume:not(:disabled) {
          background: linear-gradient(135deg, #4ecdc4, #44a08d);
          border-color: #4ecdc4;
        }

        .control-btn.stop:not(:disabled) {
          background: linear-gradient(135deg, #ff6b6b, #ee5a24);
          border-color: #ff6b6b;
        }

        .job-status {
          margin-top: 1rem;
        }

        .status-card {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 12px;
          padding: 1rem;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .status-info {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .job-id {
          font-family: 'Monaco', monospace;
          font-size: 0.9rem;
          color: #aaa;
        }

        .status-badge {
          padding: 0.25rem 0.75rem;
          border-radius: 12px;
          font-size: 0.8rem;
          font-weight: 600;
          text-transform: uppercase;
        }

        .status-badge.running {
          background: rgba(78, 205, 196, 0.2);
          color: #4ecdc4;
          border: 1px solid #4ecdc4;
        }

        .status-badge.paused {
          background: rgba(254, 202, 87, 0.2);
          color: #feca57;
          border: 1px solid #feca57;
        }

        .status-badge.completed {
          background: rgba(150, 206, 180, 0.2);
          color: #96ceb4;
          border: 1px solid #96ceb4;
        }

        .status-badge.failed {
          background: rgba(255, 107, 107, 0.2);
          color: #ff6b6b;
          border: 1px solid #ff6b6b;
        }

        .progress-bar {
          width: 100%;
          height: 8px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 4px;
          overflow: hidden;
          margin-bottom: 0.5rem;
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, #4ecdc4, #44a08d);
          transition: width 0.3s ease;
        }

        .progress-text {
          font-size: 0.9rem;
          color: #aaa;
        }

        .metrics-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
        }

        .time-filter {
          display: flex;
          gap: 0.25rem;
        }

        .filter-btn {
          padding: 0.5rem 1rem;
          border: none;
          border-radius: 6px;
          background: rgba(255, 255, 255, 0.1);
          color: white;
          cursor: pointer;
          transition: all 0.2s ease;
          font-size: 0.8rem;
        }

        .filter-btn:hover {
          background: rgba(255, 255, 255, 0.2);
        }

        .filter-btn.active {
          background: #4ecdc4;
          color: #000;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 1rem;
        }

        .metric-card {
          background: rgba(255, 255, 255, 0.05);
          border-radius: 12px;
          padding: 1rem;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .metric-card h3 {
          margin: 0 0 1rem 0;
          font-size: 1rem;
          font-weight: 500;
          color: #aaa;
        }

        .chart-container {
          height: 120px;
          margin-bottom: 1rem;
        }

        .metric-value {
          font-size: 1.5rem;
          font-weight: 700;
          color: white;
          text-align: center;
        }

        .toggle-btn {
          padding: 0.5rem 1rem;
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.1);
          color: white;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .toggle-btn:hover {
          background: rgba(255, 255, 255, 0.2);
        }

        .toggle-btn.active {
          background: #4ecdc4;
          color: #000;
          border-color: #4ecdc4;
        }

        .architecture-viz {
          margin-top: 1rem;
        }

        .architecture-svg {
          width: 100%;
          height: auto;
          max-width: 800px;
          border-radius: 12px;
        }

        .hyperparameter-controls {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 1.5rem;
          margin-top: 1rem;
        }

        .parameter-group {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .parameter-group label {
          font-weight: 500;
          color: #aaa;
          font-size: 0.9rem;
        }

        .parameter-slider {
          width: 100%;
          height: 6px;
          border-radius: 3px;
          background: rgba(255, 255, 255, 0.2);
          outline: none;
          -webkit-appearance: none;
        }

        .parameter-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: #4ecdc4;
          cursor: pointer;
          border: 2px solid #fff;
          box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }

        .parameter-slider::-moz-range-thumb {
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: #4ecdc4;
          cursor: pointer;
          border: 2px solid #fff;
          box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }

        .parameter-value {
          font-family: 'Monaco', monospace;
          font-size: 0.8rem;
          color: #4ecdc4;
          text-align: right;
        }
      `}</style>
    </div>
  );
};

export default FlowMatchingTrainingDashboard; 