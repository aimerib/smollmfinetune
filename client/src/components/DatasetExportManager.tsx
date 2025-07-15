import React, { useState, useEffect, useRef, useCallback } from 'react';
import '../styles/design-system.css';

// Types
interface ExportConfig {
  id: string;
  name: string;
  format: 'huggingface' | 'jsonl' | 'pytorch' | 'custom';
  created_at: string;
  options: Record<string, any>;
}

interface ExportRecord {
  id: string;
  dataset_id: string;
  dataset_name: string;
  config_id: string;
  format: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at: string | null;
  file_size: number | null;
  file_path: string | null;
  progress: number;
  error_message: string | null;
}

interface ActiveExport {
  id: string;
  dataset_id: string;
  dataset_name: string;
  format: string;
  progress: number;
  current_step: string;
  estimated_completion: string;
  speed: string;
}

interface Dataset {
  id: string;
  name: string;
  created_at: string;
  sample_count: number;
  total_size: number;
}

interface DatasetExportManagerProps {
  onExportStart?: (datasetId: string, configId: string) => void;
  onExportCancel?: (exportId: string) => void;
  onExportDownload?: (exportId: string) => void;
}

const DatasetExportManager: React.FC<DatasetExportManagerProps> = ({
  onExportStart,
  onExportCancel,
  onExportDownload
}) => {
  const [exportConfigs, setExportConfigs] = useState<ExportConfig[]>([]);
  const [exportHistory, setExportHistory] = useState<ExportRecord[]>([]);
  const [activeExports, setActiveExports] = useState<ActiveExport[]>([]);
  const [availableDatasets, setAvailableDatasets] = useState<Dataset[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [selectedConfig, setSelectedConfig] = useState<string>('');
  const [selectedFormat, setSelectedFormat] = useState<string>('huggingface');
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [showBatchModal, setShowBatchModal] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState<{
    type: 'config' | 'export';
    id: string;
  } | null>(null);
  const [editingConfig, setEditingConfig] = useState<ExportConfig | null>(null);
  const [batchSelectedDatasets, setBatchSelectedDatasets] = useState<string[]>([]);
  const [configForm, setConfigForm] = useState({
    name: '',
    format: 'huggingface',
    options: {}
  });
  
  const wsRef = useRef<WebSocket | null>(null);
  const updateTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Debounced update function
  const debounceUpdate = useCallback((updateFn: () => void) => {
    if (updateTimeoutRef.current) {
      clearTimeout(updateTimeoutRef.current);
    }
    updateTimeoutRef.current = setTimeout(updateFn, 100);
  }, []);

  // Load export configurations
  const loadExportConfigs = useCallback(async () => {
    try {
      const response = await fetch('/api/multimodal/exports/configs');
      if (!response.ok) {
        throw new Error('Failed to load export configurations');
      }
      const data = await response.json();
      setExportConfigs(data.configs || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load export configurations');
    }
  }, []);

  // Load export history
  const loadExportHistory = useCallback(async () => {
    try {
      const response = await fetch('/api/multimodal/exports/history');
      if (!response.ok) {
        throw new Error('Failed to load export history');
      }
      const data = await response.json();
      setExportHistory(data.history || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load export history');
    }
  }, []);

  // Load active exports
  const loadActiveExports = useCallback(async () => {
    try {
      const response = await fetch('/api/multimodal/exports/active');
      if (!response.ok) {
        throw new Error('Failed to load active exports');
      }
      const data = await response.json();
      setActiveExports(data.active_exports || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load active exports');
    }
  }, []);

  // Load available datasets
  const loadAvailableDatasets = useCallback(async () => {
    try {
      const response = await fetch('/api/multimodal/datasets');
      if (!response.ok) {
        throw new Error('Failed to load datasets');
      }
      const data = await response.json();
      setAvailableDatasets(data.datasets || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load datasets');
    }
  }, []);

  // Initialize component
  useEffect(() => {
    const initialize = async () => {
      setIsLoading(true);
      await Promise.all([
        loadExportConfigs(),
        loadExportHistory(),
        loadActiveExports(),
        loadAvailableDatasets()
      ]);
      setIsLoading(false);
    };

    initialize();
  }, [loadExportConfigs, loadExportHistory, loadActiveExports, loadAvailableDatasets]);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws/export-monitoring');
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('Export monitoring WebSocket connected');
        };

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          
          switch (data.type) {
            case 'export_progress':
              debounceUpdate(() => {
                setActiveExports(prev => prev.map(exp => 
                  exp.id === data.export_id 
                    ? { 
                        ...exp, 
                        progress: data.progress,
                        current_step: data.current_step,
                        speed: data.speed
                      } 
                    : exp
                ));
              });
              break;
              
            case 'export_completed':
              setActiveExports(prev => prev.filter(exp => exp.id !== data.export_id));
              loadExportHistory();
              break;
              
            case 'export_failed':
              setActiveExports(prev => prev.filter(exp => exp.id !== data.export_id));
              loadExportHistory();
              break;
              
            default:
              console.log('Unknown message type:', data.type);
          }
        };

        ws.onclose = () => {
          console.log('Export monitoring WebSocket disconnected');
          setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (error) => {
          console.error('Export monitoring WebSocket error:', error);
        };
      } catch (err) {
        console.error('Failed to connect export monitoring WebSocket:', err);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, [debounceUpdate, loadExportHistory]);

  // Handle export start
  const handleExportStart = async (datasetId: string, configId: string) => {
    try {
      const config = exportConfigs.find(c => c.id === configId);
      if (!config) {
        throw new Error('Export configuration not found');
      }

      const response = await fetch('/api/multimodal/datasets/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          export_config: config
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Export failed');
      }

      const data = await response.json();
      
      // Add to active exports
      setActiveExports(prev => [...prev, {
        id: data.export_id,
        dataset_id: datasetId,
        dataset_name: availableDatasets.find(d => d.id === datasetId)?.name || 'Unknown',
        format: config.format,
        progress: 0,
        current_step: 'Starting export...',
        estimated_completion: data.estimated_completion,
        speed: '0 MB/s'
      }]);

      // Reset form
      setSelectedDataset('');
      setSelectedConfig('');
      
      onExportStart?.(datasetId, configId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    }
  };

  // Handle batch export
  const handleBatchExport = async (datasetIds: string[], configId: string) => {
    try {
      const config = exportConfigs.find(c => c.id === configId);
      if (!config) {
        throw new Error('Export configuration not found');
      }

      const response = await fetch('/api/multimodal/datasets/batch-export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_ids: datasetIds,
          export_config: config
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Batch export failed');
      }

      const data = await response.json();
      
      // Add to active exports
      data.export_ids.forEach((exportId: string, index: number) => {
        setActiveExports(prev => [...prev, {
          id: exportId,
          dataset_id: datasetIds[index],
          dataset_name: availableDatasets.find(d => d.id === datasetIds[index])?.name || 'Unknown',
          format: config.format,
          progress: 0,
          current_step: 'Starting export...',
          estimated_completion: data.estimated_completion,
          speed: '0 MB/s'
        }]);
      });

      setShowBatchModal(false);
      setBatchSelectedDatasets([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Batch export failed');
    }
  };

  // Handle export cancel
  const handleExportCancel = async (exportId: string) => {
    try {
      const response = await fetch(`/api/multimodal/exports/${exportId}/cancel`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error('Failed to cancel export');
      }

      setActiveExports(prev => prev.filter(exp => exp.id !== exportId));
      onExportCancel?.(exportId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel export');
    }
  };

  // Handle export download
  const handleExportDownload = async (exportId: string) => {
    try {
      const response = await fetch(`/api/multimodal/exports/${exportId}/download`);
      
      if (!response.ok) {
        throw new Error('Download failed');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `export-${exportId}.tar.gz`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      onExportDownload?.(exportId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
    }
  };

  // Handle export re-export
  const handleReExport = async (datasetId: string, configId: string) => {
    try {
      const response = await fetch('/api/multimodal/exports/re-export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: datasetId,
          config_id: configId
        })
      });

      if (!response.ok) {
        throw new Error('Re-export failed');
      }

      const data = await response.json();
      
      // Add to active exports
      setActiveExports(prev => [...prev, {
        id: data.export_id,
        dataset_id: datasetId,
        dataset_name: availableDatasets.find(d => d.id === datasetId)?.name || 'Unknown',
        format: exportConfigs.find(c => c.id === configId)?.format || 'unknown',
        progress: 0,
        current_step: 'Starting re-export...',
        estimated_completion: data.estimated_completion,
        speed: '0 MB/s'
      }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Re-export failed');
    }
  };

  // Handle config save
  const handleConfigSave = async () => {
    try {
      const method = editingConfig ? 'PUT' : 'POST';
      const url = editingConfig 
        ? `/api/multimodal/exports/configs/${editingConfig.id}`
        : '/api/multimodal/exports/configs';

      const response = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(configForm)
      });

      if (!response.ok) {
        throw new Error('Failed to save configuration');
      }

      setShowConfigModal(false);
      setEditingConfig(null);
      setConfigForm({ name: '', format: 'huggingface', options: {} });
      loadExportConfigs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save configuration');
    }
  };

  // Handle config delete
  const handleConfigDelete = async (configId: string) => {
    try {
      const response = await fetch(`/api/multimodal/exports/configs/${configId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error('Failed to delete configuration');
      }

      setShowDeleteDialog(null);
      loadExportConfigs();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete configuration');
    }
  };

  // Handle export delete
  const handleExportDelete = async (exportId: string) => {
    try {
      const response = await fetch(`/api/multimodal/exports/${exportId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error('Failed to delete export');
      }

      setShowDeleteDialog(null);
      loadExportHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete export');
    }
  };

  // Format file size
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'var(--color-success)';
      case 'running': return 'var(--color-warning)';
      case 'failed': return 'var(--color-error)';
      default: return 'var(--color-text-secondary)';
    }
  };

  // Calculate estimated completion time
  const calculateETA = (progress: number, speed: string) => {
    if (progress >= 100) return 'Completed';
    if (speed === '0 MB/s') return 'Calculating...';
    
    const remaining = 100 - progress;
    const speedValue = parseFloat(speed.replace(' MB/s', ''));
    const eta = Math.ceil(remaining / speedValue);
    
    return `ETA: ${eta} minutes`;
  };

  // Render export configuration section
  const renderExportConfiguration = () => (
    <div className="export-configuration card" aria-label="Export configurations">
      <div className="section-header">
        <h3>Export Configuration</h3>
        <button 
          className="btn btn-primary"
          onClick={() => setShowConfigModal(true)}
        >
          Create New Configuration
        </button>
      </div>

      <div className="config-list">
        {exportConfigs.map((config) => (
          <div key={config.id} className="config-item">
            <div className="config-header">
              <span className="config-name">{config.name}</span>
              <span className="config-format">{config.format}</span>
            </div>
            <div className="config-meta">
              <span className="config-date">{new Date(config.created_at).toLocaleDateString()}</span>
            </div>
            <div className="config-actions">
              <button 
                className="btn btn-ghost btn-sm"
                data-testid={`edit-config-${config.id}`}
                onClick={() => {
                  setEditingConfig(config);
                  setConfigForm({
                    name: config.name,
                    format: config.format,
                    options: config.options
                  });
                  setShowConfigModal(true);
                }}
              >
                Edit
              </button>
              <button 
                className="btn btn-ghost btn-sm"
                data-testid={`delete-config-${config.id}`}
                onClick={() => setShowDeleteDialog({ type: 'config', id: config.id })}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>

      <div className="export-form">
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="dataset-select">Select Dataset</label>
            <select 
              id="dataset-select"
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
            >
              <option value="">Choose a dataset</option>
              {availableDatasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.sample_count} samples)
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="config-select">Export Configuration</label>
            <select 
              id="config-select"
              value={selectedConfig}
              onChange={(e) => setSelectedConfig(e.target.value)}
            >
              <option value="">Choose a configuration</option>
              {exportConfigs.map((config) => (
                <option key={config.id} value={config.id}>
                  {config.name} ({config.format})
                </option>
              ))}
            </select>
          </div>

          <button 
            className="btn btn-primary"
            disabled={!selectedDataset || !selectedConfig}
            onClick={() => handleExportStart(selectedDataset, selectedConfig)}
          >
            Start Export
          </button>
        </div>
      </div>
    </div>
  );

  // Render active exports section
  const renderActiveExports = () => (
    <div className="active-exports card" aria-label="Active exports">
      <h3>Active Exports</h3>
      
      {activeExports.length === 0 ? (
        <div className="empty-state">
          <span>No active exports</span>
        </div>
      ) : (
        <div className="exports-list">
          {activeExports.map((export_) => (
            <div key={export_.id} className="export-item">
              <div className="export-header">
                <span className="export-name">{export_.dataset_name}</span>
                <span className="export-format">{export_.format}</span>
              </div>
              
              <div className="export-progress">
                <div className="progress-bar">
                  <div 
                    className="progress-fill" 
                    style={{ width: `${export_.progress}%` }}
                  />
                </div>
                <div className="progress-info">
                  <span className="progress-text">{export_.progress}%</span>
                  <span className="progress-speed">{export_.speed}</span>
                </div>
              </div>
              
              <div className="export-step">{export_.current_step}</div>
              <div className="export-eta">{calculateETA(export_.progress, export_.speed)}</div>
              
              <div className="export-actions">
                <button 
                  className="btn btn-secondary btn-sm"
                  data-testid={`cancel-export-${export_.id}`}
                  onClick={() => handleExportCancel(export_.id)}
                >
                  Cancel
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  // Render export history section
  const renderExportHistory = () => (
    <div className="export-history card" aria-label="Export history">
      <h3>Export History</h3>
      
      <div className="history-list">
        {exportHistory.map((record) => (
          <div key={record.id} className="history-item">
            <div className="history-header">
              <span className="history-name">{record.dataset_name}</span>
              <span 
                className="history-status"
                data-testid={`export-status-${record.status}`}
                style={{ color: getStatusColor(record.status) }}
              >
                {record.status}
              </span>
            </div>
            
            <div className="history-meta">
              <span className="history-format">{record.format}</span>
              <span className="history-date">{new Date(record.created_at).toLocaleDateString()}</span>
              {record.file_size && (
                <span className="history-size">{formatFileSize(record.file_size)}</span>
              )}
            </div>
            
            {record.error_message && (
              <div className="history-error">
                <span className="error-icon">⚠️</span>
                <span className="error-message">{record.error_message}</span>
              </div>
            )}
            
            <div className="history-actions">
              {record.status === 'completed' && (
                <button 
                  className="btn btn-primary btn-sm"
                  data-testid={`download-export-${record.id}`}
                  onClick={() => handleExportDownload(record.id)}
                >
                  Download
                </button>
              )}
              <button 
                className="btn btn-secondary btn-sm"
                data-testid={`re-export-${record.id}`}
                onClick={() => handleReExport(record.dataset_id, record.config_id)}
              >
                Re-export
              </button>
              <button 
                className="btn btn-ghost btn-sm"
                data-testid={`delete-export-${record.id}`}
                onClick={() => setShowDeleteDialog({ type: 'export', id: record.id })}
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  // Render batch export section
  const renderBatchExport = () => (
    <div className="batch-export card">
      <div className="section-header">
        <h3>Batch Export</h3>
        <button 
          className="btn btn-secondary"
          onClick={() => setShowBatchModal(true)}
        >
          Batch Export
        </button>
      </div>
      
      <div className="batch-info">
        <p>Export multiple datasets using the same configuration.</p>
      </div>
    </div>
  );

  // Render configuration modal
  const renderConfigModal = () => {
    if (!showConfigModal) return null;

    return (
      <div className="modal-overlay">
        <div className="modal">
          <div className="modal-header">
            <h3>{editingConfig ? 'Edit Configuration' : 'Create Configuration'}</h3>
            <button 
              className="btn btn-ghost"
              onClick={() => {
                setShowConfigModal(false);
                setEditingConfig(null);
                setConfigForm({ name: '', format: 'huggingface', options: {} });
              }}
            >
              ×
            </button>
          </div>
          
          <div className="modal-content">
            <div className="form-group">
              <label htmlFor="config-name">Configuration Name</label>
              <input
                id="config-name"
                type="text"
                value={configForm.name}
                onChange={(e) => setConfigForm(prev => ({ ...prev, name: e.target.value }))}
                placeholder="My Export Configuration"
              />
            </div>
            
            <div className="form-group">
              <label htmlFor="config-format">Export Format</label>
              <select 
                id="config-format"
                value={configForm.format}
                onChange={(e) => setConfigForm(prev => ({ ...prev, format: e.target.value as any }))}
              >
                <option value="huggingface">HuggingFace</option>
                <option value="jsonl">JSONL</option>
                <option value="pytorch">PyTorch</option>
                <option value="custom">Custom</option>
              </select>
            </div>

            {/* Format-specific options */}
            {configForm.format === 'huggingface' && (
              <div className="format-options">
                <h4>HuggingFace Options</h4>
                <div className="form-group">
                  <label>Train/Val/Test Split</label>
                  <div className="split-inputs">
                    <input type="number" placeholder="80" step="1" min="0" max="100" />
                    <input type="number" placeholder="10" step="1" min="0" max="100" />
                    <input type="number" placeholder="10" step="1" min="0" max="100" />
                  </div>
                </div>
                <div className="form-group">
                  <label>Tokenizer Config</label>
                  <textarea rows={3} placeholder="Enter tokenizer configuration" />
                </div>
                <div className="form-group">
                  <label>Dataset Features</label>
                  <textarea rows={3} placeholder="Define dataset features" />
                </div>
                <div className="form-group">
                  <label>
                    <input type="checkbox" />
                    Dataset Hub Upload
                  </label>
                </div>
                <div className="form-group">
                  <label>
                    <input type="checkbox" />
                    Model Card Generation
                  </label>
                </div>
                <div className="form-group">
                  <label>
                    <input type="checkbox" />
                    Feature Configuration
                  </label>
                </div>
              </div>
            )}

            {configForm.format === 'jsonl' && (
              <div className="format-options">
                <h4>JSONL Options</h4>
                <div className="form-group">
                  <label>Compression</label>
                  <select>
                    <option value="none">None</option>
                    <option value="gzip">GZIP</option>
                    <option value="bz2">BZ2</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Chunk Size</label>
                  <input type="number" placeholder="1000" min="1" />
                </div>
                <div className="form-group">
                  <label>Encoding</label>
                  <select>
                    <option value="utf-8">UTF-8</option>
                    <option value="ascii">ASCII</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>
                    <input type="checkbox" />
                    Include Audio
                  </label>
                </div>
              </div>
            )}

            {configForm.format === 'pytorch' && (
              <div className="format-options">
                <h4>PyTorch Options</h4>
                <div className="form-group">
                  <label>Tensor Format</label>
                  <select>
                    <option value="pt">PyTorch (.pt)</option>
                    <option value="pth">PyTorch (.pth)</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>DataLoader Config</label>
                  <div className="dataloader-config">
                    <input type="number" placeholder="Batch Size" />
                    <input type="number" placeholder="Num Workers" />
                  </div>
                </div>
                <div className="form-group">
                  <label>Transforms</label>
                  <textarea rows={3} placeholder="Define transforms" />
                </div>
                <div className="form-group">
                  <label>
                    <input type="checkbox" />
                    Pin Memory
                  </label>
                </div>
                <div className="form-group">
                  <label>
                    <input type="checkbox" />
                    Shuffle
                  </label>
                </div>
              </div>
            )}

            {configForm.format === 'custom' && (
              <div className="format-options">
                <h4>Custom Format Options</h4>
                <div className="form-group">
                  <label htmlFor="custom-script">Export Script</label>
                  <textarea 
                    id="custom-script"
                    rows={10}
                    placeholder="def export_custom(dataset): pass"
                  />
                </div>
                <div className="form-group">
                  <label>Output Structure</label>
                  <textarea rows={5} placeholder="Define output structure" />
                </div>
                <div className="form-group">
                  <label>Custom Export Script</label>
                  <textarea rows={5} placeholder="Enter custom export script" />
                </div>
              </div>
            )}
          </div>
          
          <div className="modal-actions">
            <button 
              className="btn btn-secondary"
              onClick={() => {
                setShowConfigModal(false);
                setEditingConfig(null);
                setConfigForm({ name: '', format: 'huggingface', options: {} });
              }}
            >
              Cancel
            </button>
            <button 
              className="btn btn-primary"
              onClick={handleConfigSave}
              disabled={!configForm.name}
            >
              Save Configuration
            </button>
          </div>
        </div>
      </div>
    );
  };

  // Render batch modal
  const renderBatchModal = () => {
    if (!showBatchModal) return null;

    return (
      <div className="modal-overlay">
        <div className="modal">
          <div className="modal-header">
            <h3>Select Multiple Datasets</h3>
            <button 
              className="btn btn-ghost"
              onClick={() => setShowBatchModal(false)}
            >
              ×
            </button>
          </div>
          
          <div className="modal-content">
            <div className="batch-datasets">
              {availableDatasets.map((dataset) => (
                <div key={dataset.id} className="batch-dataset-item">
                  <label>
                    <input
                      type="checkbox"
                      data-testid={`batch-dataset-${dataset.id}`}
                      checked={batchSelectedDatasets.includes(dataset.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setBatchSelectedDatasets(prev => [...prev, dataset.id]);
                        } else {
                          setBatchSelectedDatasets(prev => prev.filter(id => id !== dataset.id));
                        }
                      }}
                    />
                    {dataset.name} ({dataset.sample_count} samples)
                  </label>
                </div>
              ))}
            </div>

            <div className="form-group">
              <label htmlFor="batch-config">Export Configuration</label>
              <select 
                id="batch-config"
                value={selectedConfig}
                onChange={(e) => setSelectedConfig(e.target.value)}
              >
                <option value="">Choose a configuration</option>
                {exportConfigs.map((config) => (
                  <option key={config.id} value={config.id}>
                    {config.name} ({config.format})
                  </option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="modal-actions">
            <button 
              className="btn btn-secondary"
              onClick={() => setShowBatchModal(false)}
            >
              Cancel
            </button>
            <button 
              className="btn btn-primary"
              onClick={() => handleBatchExport(batchSelectedDatasets, selectedConfig)}
              disabled={batchSelectedDatasets.length === 0 || !selectedConfig}
            >
              Start Batch Export
            </button>
          </div>
        </div>
      </div>
    );
  };

  // Render delete confirmation dialog
  const renderDeleteDialog = () => {
    if (!showDeleteDialog) return null;

    const isConfig = showDeleteDialog.type === 'config';
    const title = isConfig ? 'Confirm Configuration Deletion' : 'Confirm Export Deletion';
    const message = isConfig 
      ? 'Are you sure you want to delete this configuration? This action cannot be undone.'
      : 'Are you sure you want to delete this export? This action cannot be undone.';

    return (
      <div className="modal-overlay">
        <div className="modal">
          <div className="modal-header">
            <h3>{title}</h3>
            <button 
              className="btn btn-ghost"
              onClick={() => setShowDeleteDialog(null)}
            >
              ×
            </button>
          </div>
          
          <div className="modal-content">
            <p>{message}</p>
          </div>
          
          <div className="modal-actions">
            <button 
              className="btn btn-secondary"
              onClick={() => setShowDeleteDialog(null)}
            >
              Cancel
            </button>
            <button 
              className="btn btn-danger"
              onClick={() => isConfig 
                ? handleConfigDelete(showDeleteDialog.id)
                : handleExportDelete(showDeleteDialog.id)
              }
            >
              {isConfig ? 'Delete Configuration' : 'Delete Export'}
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="dataset-export-manager" aria-label="Dataset export manager">
      <div className="dashboard-header">
        <h1>Dataset Export Manager</h1>
      </div>

      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {isLoading ? (
        <div className="loading-message">Loading export configurations...</div>
      ) : (
        <div className="dashboard-content">
          <div className="dashboard-row">
            {renderExportConfiguration()}
          </div>
          
          <div className="dashboard-row">
            {renderActiveExports()}
            {renderBatchExport()}
          </div>
          
          <div className="dashboard-row">
            {renderExportHistory()}
          </div>
        </div>
      )}

      {renderConfigModal()}
      {renderBatchModal()}
      {renderDeleteDialog()}

      <style>{`
        .dataset-export-manager {
          padding: var(--space-6);
          background: var(--color-background);
          min-height: 100vh;
        }

        .dashboard-header {
          margin-bottom: var(--space-6);
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
        }

        .card {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          padding: var(--space-6);
          flex: 1;
        }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-4);
        }

        .config-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
          margin-bottom: var(--space-6);
        }

        .config-item {
          padding: var(--space-4);
          background: var(--color-background);
          border-radius: var(--radius-md);
          border: 1px solid var(--color-border);
        }

        .config-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-2);
        }

        .config-name {
          font-weight: var(--font-medium);
          color: var(--color-text-primary);
        }

        .config-format {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
          text-transform: uppercase;
        }

        .config-meta {
          margin-bottom: var(--space-3);
        }

        .config-date {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
        }

        .config-actions {
          display: flex;
          gap: var(--space-2);
        }

        .export-form {
          margin-top: var(--space-6);
        }

        .form-row {
          display: flex;
          gap: var(--space-4);
          align-items: end;
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
          flex: 1;
        }

        .form-group label {
          font-weight: var(--font-medium);
          color: var(--color-text-primary);
        }

        .form-group input,
        .form-group select,
        .form-group textarea {
          padding: var(--space-3);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          font-size: var(--text-base);
        }

        .empty-state {
          text-align: center;
          padding: var(--space-8);
          color: var(--color-text-secondary);
        }

        .exports-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }

        .export-item {
          padding: var(--space-4);
          background: var(--color-background);
          border-radius: var(--radius-md);
          border: 1px solid var(--color-border);
        }

        .export-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-3);
        }

        .export-name {
          font-weight: var(--font-medium);
          color: var(--color-text-primary);
        }

        .export-format {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
          text-transform: uppercase;
        }

        .export-progress {
          margin-bottom: var(--space-3);
        }

        .progress-bar {
          height: 6px;
          background: var(--color-surface);
          border-radius: var(--radius-full);
          overflow: hidden;
          margin-bottom: var(--space-2);
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--color-primary), var(--color-primary-light));
          transition: width var(--duration-normal);
        }

        .progress-info {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: var(--text-sm);
        }

        .progress-text {
          font-weight: var(--font-medium);
        }

        .progress-speed {
          color: var(--color-text-secondary);
        }

        .export-step {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
          margin-bottom: var(--space-2);
        }

        .export-eta {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
          margin-bottom: var(--space-3);
        }

        .export-actions {
          display: flex;
          gap: var(--space-2);
        }

        .history-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }

        .history-item {
          padding: var(--space-4);
          background: var(--color-background);
          border-radius: var(--radius-md);
          border: 1px solid var(--color-border);
        }

        .history-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-2);
        }

        .history-name {
          font-weight: var(--font-medium);
          color: var(--color-text-primary);
        }

        .history-status {
          font-size: var(--text-sm);
          font-weight: var(--font-medium);
          text-transform: uppercase;
        }

        .history-meta {
          display: flex;
          gap: var(--space-4);
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
          margin-bottom: var(--space-3);
        }

        .history-error {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-2);
          background: var(--color-error-light);
          border-radius: var(--radius-sm);
          margin-bottom: var(--space-3);
          color: var(--color-error-dark);
          font-size: var(--text-sm);
        }

        .history-actions {
          display: flex;
          gap: var(--space-2);
        }

        .batch-info {
          color: var(--color-text-secondary);
        }

        .modal-overlay {
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

        .modal {
          background: var(--color-surface);
          border-radius: var(--radius-lg);
          padding: var(--space-6);
          max-width: 600px;
          width: 90%;
          max-height: 80vh;
          overflow-y: auto;
        }

        .modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-4);
        }

        .modal-content {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }

        .modal-actions {
          display: flex;
          gap: var(--space-2);
          justify-content: flex-end;
          margin-top: var(--space-4);
        }

        .format-options {
          padding: var(--space-4);
          background: var(--color-background);
          border-radius: var(--radius-md);
          border: 1px solid var(--color-border);
        }

        .format-options h4 {
          margin-bottom: var(--space-3);
          color: var(--color-text-primary);
        }

        .split-inputs {
          display: flex;
          gap: var(--space-2);
        }

        .split-inputs input {
          flex: 1;
        }

        .dataloader-config {
          display: flex;
          gap: var(--space-2);
        }

        .dataloader-config input {
          flex: 1;
        }

        .batch-datasets {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
          max-height: 300px;
          overflow-y: auto;
          padding: var(--space-2);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
        }

        .batch-dataset-item {
          padding: var(--space-2);
        }

        .batch-dataset-item label {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          cursor: pointer;
        }

        @media (max-width: 768px) {
          .dashboard-row {
            flex-direction: column;
          }
          
          .form-row {
            flex-direction: column;
          }
          
          .section-header {
            flex-direction: column;
            gap: var(--space-2);
          }
          
          .config-header {
            flex-direction: column;
            align-items: flex-start;
            gap: var(--space-2);
          }
          
          .export-header {
            flex-direction: column;
            align-items: flex-start;
            gap: var(--space-2);
          }
          
          .history-header {
            flex-direction: column;
            align-items: flex-start;
            gap: var(--space-2);
          }
          
          .history-meta {
            flex-direction: column;
            gap: var(--space-1);
          }
        }
      `}</style>
    </div>
  );
};

export default DatasetExportManager; 