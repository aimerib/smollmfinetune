import React, { useState, useEffect, useRef, useCallback } from 'react';
import '../styles/design-system.css';

// Types
interface MultimodalJob {
  id: string;
  name: string;
  status: 'queued' | 'generating' | 'completed' | 'failed' | 'paused';
  progress: number;
  currentStep: string;
  samplesGenerated: number;
  estimatedTime: string | null;
  createdAt: string;
  config: {
    sampleCount: number;
    characterCount: number;
    narrativeTypes: string[];
    useMockTTS: boolean;
    ttsProvider: string;
    outputDir: string;
    batchSize: number;
    temperature: number;
  };
  priority: 'low' | 'normal' | 'high';
  dependencies: string[];
  errorMessage?: string;
}

interface JobQueue {
  active: string[];
  waiting: string[];
  completed: string[];
  failed: string[];
  totalJobs: number;
  processingCapacity: number;
  estimatedWaitTime: string;
}

interface JobMetrics {
  totalJobs: number;
  activeJobs: number;
  completedJobs: number;
  failedJobs: number;
  averageProcessingTime: number;
  systemLoad: number;
  memoryUsage: number;
  diskUsage: number;
  networkThroughput: number;
  recentErrors: string[];
  performanceScore: number;
}

type JobAction = 'pause' | 'resume' | 'cancel' | 'delete';

interface JobManagementDashboardProps {
  onJobAction?: (action: JobAction, jobIds: string[]) => void;
  onJobReorder?: (jobId: string, newPosition: number) => void;
  onPriorityChange?: (jobId: string, priority: string) => void;
  onMetricClick?: (metricName: string) => void;
}

const JobManagementDashboard: React.FC<JobManagementDashboardProps> = ({
  onJobAction,
  onJobReorder,
  onPriorityChange,
  onMetricClick
}) => {
  const [jobs, setJobs] = useState<MultimodalJob[]>([]);
  const [selectedJobs, setSelectedJobs] = useState<string[]>([]);
  const [jobQueue, setJobQueue] = useState<JobQueue | null>(null);
  const [jobMetrics, setJobMetrics] = useState<JobMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showConfirmDialog, setShowConfirmDialog] = useState<{
    action: JobAction;
    jobIds: string[];
  } | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [draggedJob, setDraggedJob] = useState<string | null>(null);
  const [metricDetails, setMetricDetails] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const updateTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Debounced update function
  const debounceUpdate = useCallback((updateFn: () => void) => {
    if (updateTimeoutRef.current) {
      clearTimeout(updateTimeoutRef.current);
    }
    updateTimeoutRef.current = setTimeout(updateFn, 100);
  }, []);

  // Load jobs from API
  const loadJobs = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await fetch('/api/multimodal/jobs');
      
      if (!response.ok) {
        throw new Error('Failed to load jobs');
      }
      
      const data = await response.json();
      setJobs(data.jobs || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load jobs');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws/multimodal-jobs');
        wsRef.current = ws;

        ws.onopen = () => {
          setIsConnected(true);
          setError(null);
        };

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          
          switch (data.type) {
            case 'jobs_update':
              debounceUpdate(() => {
                setJobs(data.jobs);
              });
              break;
              
            case 'job_status_update':
              debounceUpdate(() => {
                setJobs(prev => prev.map(job => 
                  job.id === data.job_id 
                    ? { ...job, status: data.status }
                    : job
                ));
              });
              break;
              
            case 'job_progress':
              debounceUpdate(() => {
                setJobs(prev => prev.map(job => 
                  job.id === data.job_id 
                    ? { ...job, progress: data.progress }
                    : job
                ));
              });
              break;
              
            case 'queue_update':
              setJobQueue(data.queue);
              break;
              
            case 'job_metrics':
              setJobMetrics(data.metrics);
              break;
              
            default:
              console.log('Unknown message type:', data.type);
          }
        };

        ws.onclose = () => {
          setIsConnected(false);
          setError('Connection lost - attempting to reconnect');
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          setError('Connection lost - attempting to reconnect');
        };
      } catch (err) {
        console.error('Failed to connect WebSocket:', err);
        setError('Failed to connect to real-time updates');
      }
    };

    connectWebSocket();
    loadJobs();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, [loadJobs, debounceUpdate]);

  // Handle job selection
  const handleJobSelection = (jobId: string, selected: boolean) => {
    setSelectedJobs(prev => 
      selected 
        ? [...prev, jobId]
        : prev.filter(id => id !== jobId)
    );
  };

  // Handle select all
  const handleSelectAll = (selected: boolean) => {
    setSelectedJobs(selected ? jobs.map(job => job.id) : []);
  };

  // Handle batch job actions
  const handleBatchAction = async (action: JobAction, jobIds: string[]) => {
    if (['cancel', 'delete'].includes(action)) {
      setShowConfirmDialog({ action, jobIds });
      return;
    }

    try {
      const response = await fetch('/api/multimodal/jobs/batch-action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, job_ids: jobIds })
      });

      if (!response.ok) {
        throw new Error('Batch operation failed');
      }

      // Update jobs based on action
      switch (action) {
        case 'pause':
          setJobs(prev => prev.map(job => 
            jobIds.includes(job.id) ? { ...job, status: 'paused' } : job
          ));
          break;
        case 'resume':
          setJobs(prev => prev.map(job => 
            jobIds.includes(job.id) ? { ...job, status: 'generating' } : job
          ));
          break;
      }

      setSelectedJobs([]);
      onJobAction?.(action, jobIds);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Batch operation failed');
    }
  };

  // Handle confirmed batch action
  const handleConfirmedBatchAction = async () => {
    if (!showConfirmDialog) return;

    const { action, jobIds } = showConfirmDialog;
    setShowConfirmDialog(null);

    try {
      const response = await fetch('/api/multimodal/jobs/batch-action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, job_ids: jobIds })
      });

      if (!response.ok) {
        throw new Error('Batch operation failed');
      }

      // Update jobs based on action
      switch (action) {
        case 'cancel':
          setJobs(prev => prev.map(job => 
            jobIds.includes(job.id) ? { ...job, status: 'failed' } : job
          ));
          break;
        case 'delete':
          setJobs(prev => prev.filter(job => !jobIds.includes(job.id)));
          break;
      }

      setSelectedJobs([]);
      onJobAction?.(action, jobIds);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Batch operation failed');
    }
  };

  // Handle job reordering
  const handleJobReorder = async (jobId: string, newPosition: number) => {
    try {
      const response = await fetch('/api/multimodal/jobs/reorder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId, newPosition })
      });

      if (!response.ok) {
        throw new Error('Failed to reorder job');
      }

      onJobReorder?.(jobId, newPosition);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reorder job');
    }
  };

  // Handle priority change
  const handlePriorityChange = async (jobId: string, priority: string) => {
    try {
      const response = await fetch(`/api/multimodal/jobs/${jobId}/priority`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ priority })
      });

      if (!response.ok) {
        throw new Error('Failed to update priority');
      }

      setJobs(prev => prev.map(job => 
        job.id === jobId ? { ...job, priority: priority as any } : job
      ));

      onPriorityChange?.(jobId, priority);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update priority');
    }
  };

  // Handle metric click
  const handleMetricClick = (metricName: string) => {
    setMetricDetails(metricName);
    onMetricClick?.(metricName);
  };

  // Drag and drop handlers
  const handleDragStart = (e: React.DragEvent, jobId: string) => {
    setDraggedJob(jobId);
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };

  const handleDrop = (e: React.DragEvent, targetJobId: string) => {
    e.preventDefault();
    if (draggedJob && draggedJob !== targetJobId) {
      const targetIndex = jobs.findIndex(job => job.id === targetJobId);
      handleJobReorder(draggedJob, targetIndex);
    }
    setDraggedJob(null);
  };

  // Format time
  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    return `${minutes} minutes`;
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${Math.round(value * 100)}%`;
  };

  // Render queue visualizer
  const renderQueueVisualizer = () => (
    <div className="queue-visualizer card" aria-label="Queue visualizer">
      <h3>Queue Visualizer</h3>
      {jobQueue && (
        <div className="queue-stats">
          <div className="queue-stat">
            <span className="stat-label">Active: {jobQueue.active.length}</span>
          </div>
          <div className="queue-stat">
            <span className="stat-label">Waiting: {jobQueue.waiting.length}</span>
          </div>
          <div className="queue-stat">
            <span className="stat-label">Completed: {jobQueue.completed.length}</span>
          </div>
          <div className="queue-stat">
            <span className="stat-label">Est. Wait: {jobQueue.estimatedWaitTime}</span>
          </div>
        </div>
      )}
    </div>
  );

  // Render job table
  const renderJobTable = () => (
    <div className="job-table-container">
      <div className="job-table-header">
        <h3>Active Jobs</h3>
        <div className="table-controls">
          <label>
            <input
              type="checkbox"
              data-testid="select-all-jobs"
              aria-label="Select all jobs"
              checked={selectedJobs.length === jobs.length && jobs.length > 0}
              onChange={(e) => handleSelectAll(e.target.checked)}
            />
            Select All
          </label>
        </div>
      </div>

      <div className="job-table">
        {jobs.map((job) => (
          <div
            key={job.id}
            className={`job-row ${job.status}`}
            data-testid={`job-row-${job.id}`}
            draggable
            onDragStart={(e) => handleDragStart(e, job.id)}
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, job.id)}
          >
            <div className="job-selection">
              <input
                type="checkbox"
                data-testid={`job-checkbox-${job.id}`}
                checked={selectedJobs.includes(job.id)}
                onChange={(e) => handleJobSelection(job.id, e.target.checked)}
              />
            </div>
            
            <div className="job-drag-handle" data-testid={`drag-handle-${job.id}`}>
              ⋮⋮
            </div>
            
            <div className="job-info">
              <div className="job-name">{job.name}</div>
              <div className="job-status">{job.status}</div>
              <div className="job-progress">
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${job.progress}%` }}
                  />
                </div>
                <span className="progress-text">{job.progress}%</span>
              </div>
              <div className="job-step">{job.currentStep}</div>
            </div>
            
            <div className="job-priority">
              <select
                data-testid={`priority-select-${job.id}`}
                value={job.priority}
                onChange={(e) => handlePriorityChange(job.id, e.target.value)}
              >
                <option value="low">Low</option>
                <option value="normal">Normal</option>
                <option value="high">High</option>
              </select>
            </div>
            
            <div className="job-created">
              {new Date(job.createdAt).toLocaleDateString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  // Render batch operations panel
  const renderBatchOperations = () => (
    <div className="batch-operations card">
      <h3>Batch Operations</h3>
      <div className="batch-info">
        {selectedJobs.length === 0 ? (
          <span>No jobs selected</span>
        ) : (
          <span>{selectedJobs.length} job{selectedJobs.length > 1 ? 's' : ''} selected</span>
        )}
      </div>
      
      <div className="batch-actions">
        <button
          className="btn btn-secondary"
          disabled={selectedJobs.length === 0}
          onClick={() => handleBatchAction('pause', selectedJobs)}
        >
          Pause Selected
        </button>
        <button
          className="btn btn-secondary"
          disabled={selectedJobs.length === 0}
          onClick={() => handleBatchAction('resume', selectedJobs)}
        >
          Resume Selected
        </button>
        <button
          className="btn btn-warning"
          disabled={selectedJobs.length === 0}
          onClick={() => handleBatchAction('cancel', selectedJobs)}
        >
          Cancel Selected
        </button>
        <button
          className="btn btn-danger"
          disabled={selectedJobs.length === 0}
          onClick={() => handleBatchAction('delete', selectedJobs)}
        >
          Delete Selected
        </button>
      </div>
    </div>
  );

  // Render system metrics
  const renderSystemMetrics = () => (
    <div className="system-metrics card" aria-label="System metrics panel">
      <h3>System Metrics</h3>
      {jobMetrics && (
        <div className="metrics-grid">
          <div 
            className="metric-item"
            data-testid="metric-system-load"
            onClick={() => handleMetricClick('system-load')}
          >
            <div className="metric-label">System Load: {formatPercentage(jobMetrics.systemLoad)}</div>
            <div className="metric-bar">
              <div 
                className="metric-fill" 
                style={{ width: formatPercentage(jobMetrics.systemLoad) }}
              />
            </div>
          </div>
          
          <div 
            className="metric-item"
            data-testid="metric-memory-usage"
            onClick={() => handleMetricClick('memory-usage')}
          >
            <div className="metric-label">Memory Usage: {formatPercentage(jobMetrics.memoryUsage)}</div>
            <div className="metric-bar">
              <div 
                className="metric-fill" 
                style={{ width: formatPercentage(jobMetrics.memoryUsage) }}
              />
            </div>
          </div>
          
          <div 
            className="metric-item"
            data-testid="metric-disk-usage"
            onClick={() => handleMetricClick('disk-usage')}
          >
            <div className="metric-label">Disk Usage: {formatPercentage(jobMetrics.diskUsage)}</div>
            <div className="metric-bar">
              <div 
                className="metric-fill" 
                style={{ width: formatPercentage(jobMetrics.diskUsage) }}
              />
            </div>
          </div>
          
          <div 
            className="metric-item"
            data-testid="metric-performance-score"
            onClick={() => handleMetricClick('performance-score')}
          >
            <div className="metric-label">Performance Score: {formatPercentage(jobMetrics.performanceScore)}</div>
            <div className="metric-bar">
              <div 
                className="metric-fill" 
                style={{ width: formatPercentage(jobMetrics.performanceScore) }}
              />
            </div>
          </div>
          
          <div className="metric-item">
            <div className="metric-label">Average Processing Time: {formatTime(jobMetrics.averageProcessingTime)}</div>
          </div>
          
          <div className="metric-item">
            <div className="metric-label">Network Throughput: {jobMetrics.networkThroughput} MB/s</div>
          </div>
        </div>
      )}
    </div>
  );

  // Render confirmation dialog
  const renderConfirmDialog = () => {
    if (!showConfirmDialog) return null;

    const { action, jobIds } = showConfirmDialog;
    const actionText = action === 'cancel' ? 'Cancel' : 'Delete';
    const title = `Confirm ${actionText}ation`;

    return (
      <div className="confirm-dialog-overlay">
        <div className="confirm-dialog">
          <h3>{title}</h3>
          <p>
            Are you sure you want to {action} {jobIds.length} job{jobIds.length > 1 ? 's' : ''}?
            {action === 'delete' && ' This action cannot be undone.'}
          </p>
          <div className="dialog-actions">
            <button
              className="btn btn-secondary"
              onClick={() => setShowConfirmDialog(null)}
            >
              Cancel
            </button>
            <button
              className={`btn ${action === 'delete' ? 'btn-danger' : 'btn-warning'}`}
              onClick={handleConfirmedBatchAction}
            >
              Confirm {actionText}
            </button>
          </div>
        </div>
      </div>
    );
  };

  // Render metric details
  const renderMetricDetails = () => {
    if (!metricDetails) return null;

    return (
      <div className="metric-details-overlay">
        <div className="metric-details">
          <h3>System Load Details</h3>
          <div className="metric-detail-content">
            <div className="detail-item">
              <span className="detail-label">CPU Usage</span>
              <div className="detail-bar">
                <div className="detail-fill" style={{ width: '65%' }} />
              </div>
              <span className="detail-value">65%</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Memory Details</span>
              <div className="detail-bar">
                <div className="detail-fill" style={{ width: '72%' }} />
              </div>
              <span className="detail-value">72%</span>
            </div>
          </div>
          <button
            className="btn btn-secondary"
            onClick={() => setMetricDetails(null)}
          >
            Close
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="job-management-dashboard" aria-label="Job management dashboard">
      <div className="dashboard-header">
        <h1>Job Management Dashboard</h1>
        <div className="connection-status">
          <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? '🟢' : '🔴'}
          </span>
          <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>

      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {isLoading ? (
        <div className="loading-message">Loading jobs...</div>
      ) : (
        <div className="dashboard-content">
          <div className="dashboard-row">
            {renderQueueVisualizer()}
            {renderSystemMetrics()}
          </div>
          
          <div className="dashboard-row">
            {renderJobTable()}
          </div>
          
          <div className="dashboard-row">
            {renderBatchOperations()}
          </div>
        </div>
      )}

      {renderConfirmDialog()}
      {renderMetricDetails()}

      <style>{`
        .job-management-dashboard {
          padding: var(--space-6);
          background: var(--color-background);
          min-height: 100vh;
        }

        .dashboard-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-6);
        }

        .connection-status {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          font-size: var(--text-sm);
        }

        .status-indicator {
          font-size: var(--text-lg);
        }

        .error-message {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-4);
          background: var(--color-error-light);
          border-radius: var(--radius-md);
          margin-bottom: var(--space-4);
          color: var(--color-error-dark);
        }

        .loading-message {
          text-align: center;
          padding: var(--space-8);
          color: var(--color-text-secondary);
        }

        .dashboard-content {
          display: flex;
          flex-direction: column;
          gap: var(--space-6);
        }

        .dashboard-row {
          display: flex;
          gap: var(--space-4);
          flex-wrap: wrap;
        }

        .card {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          padding: var(--space-4);
          flex: 1;
          min-width: 300px;
        }

        .queue-visualizer {
          max-width: 400px;
        }

        .queue-stats {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: var(--space-3);
          margin-top: var(--space-4);
        }

        .queue-stat {
          text-align: center;
          padding: var(--space-2);
          background: var(--color-background);
          border-radius: var(--radius-md);
        }

        .job-table-container {
          flex: 2;
        }

        .job-table-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-4);
        }

        .job-table {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
        }

        .job-row {
          display: grid;
          grid-template-columns: 40px 30px 1fr 120px 100px;
          gap: var(--space-3);
          align-items: center;
          padding: var(--space-3);
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          cursor: pointer;
          transition: all var(--duration-normal);
        }

        .job-row:hover {
          background: var(--color-surface-hover);
          border-color: var(--color-primary-light);
        }

        .job-row.generating {
          border-left: 4px solid var(--color-warning);
        }

        .job-row.completed {
          border-left: 4px solid var(--color-success);
        }

        .job-row.failed {
          border-left: 4px solid var(--color-error);
        }

        .job-drag-handle {
          cursor: grab;
          color: var(--color-text-secondary);
          font-size: var(--text-lg);
        }

        .job-drag-handle:active {
          cursor: grabbing;
        }

        .job-info {
          display: flex;
          flex-direction: column;
          gap: var(--space-1);
        }

        .job-name {
          font-weight: var(--font-medium);
        }

        .job-status {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
          text-transform: capitalize;
        }

        .job-progress {
          display: flex;
          align-items: center;
          gap: var(--space-2);
        }

        .progress-bar {
          flex: 1;
          height: 4px;
          background: var(--color-surface);
          border-radius: var(--radius-full);
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--color-primary), var(--color-primary-light));
          transition: width var(--duration-normal);
        }

        .progress-text {
          font-size: var(--text-xs);
          color: var(--color-text-secondary);
        }

        .job-step {
          font-size: var(--text-xs);
          color: var(--color-text-secondary);
          font-style: italic;
        }

        .job-priority select {
          padding: var(--space-1) var(--space-2);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-sm);
          font-size: var(--text-sm);
        }

        .job-created {
          font-size: var(--text-xs);
          color: var(--color-text-secondary);
        }

        .batch-operations {
          max-width: 300px;
        }

        .batch-info {
          margin-bottom: var(--space-4);
          padding: var(--space-2);
          background: var(--color-background);
          border-radius: var(--radius-md);
          text-align: center;
          font-size: var(--text-sm);
        }

        .batch-actions {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
        }

        .system-metrics {
          max-width: 400px;
        }

        .metrics-grid {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
          margin-top: var(--space-4);
        }

        .metric-item {
          cursor: pointer;
          padding: var(--space-2);
          border-radius: var(--radius-md);
          transition: background var(--duration-normal);
        }

        .metric-item:hover {
          background: var(--color-surface-hover);
        }

        .metric-label {
          font-size: var(--text-sm);
          margin-bottom: var(--space-1);
        }

        .metric-bar {
          height: 6px;
          background: var(--color-surface);
          border-radius: var(--radius-full);
          overflow: hidden;
        }

        .metric-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--color-primary), var(--color-primary-light));
          transition: width var(--duration-normal);
        }

        .confirm-dialog-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }

        .confirm-dialog {
          background: var(--color-surface);
          border-radius: var(--radius-lg);
          padding: var(--space-6);
          max-width: 400px;
          width: 90%;
        }

        .confirm-dialog h3 {
          margin-bottom: var(--space-4);
        }

        .confirm-dialog p {
          margin-bottom: var(--space-6);
          color: var(--color-text-secondary);
        }

        .dialog-actions {
          display: flex;
          gap: var(--space-3);
          justify-content: flex-end;
        }

        .metric-details-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }

        .metric-details {
          background: var(--color-surface);
          border-radius: var(--radius-lg);
          padding: var(--space-6);
          max-width: 500px;
          width: 90%;
        }

        .metric-details h3 {
          margin-bottom: var(--space-4);
        }

        .metric-detail-content {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
          margin-bottom: var(--space-6);
        }

        .detail-item {
          display: flex;
          align-items: center;
          gap: var(--space-3);
        }

        .detail-label {
          font-size: var(--text-sm);
          min-width: 120px;
        }

        .detail-bar {
          flex: 1;
          height: 6px;
          background: var(--color-surface);
          border-radius: var(--radius-full);
          overflow: hidden;
        }

        .detail-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--color-primary), var(--color-primary-light));
        }

        .detail-value {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
        }

        @media (max-width: 768px) {
          .dashboard-row {
            flex-direction: column;
          }
          
          .job-row {
            grid-template-columns: 40px 1fr 100px;
          }
          
          .job-drag-handle {
            display: none;
          }
          
          .job-created {
            display: none;
          }
        }
      `}</style>
    </div>
  );
};

export default JobManagementDashboard; 