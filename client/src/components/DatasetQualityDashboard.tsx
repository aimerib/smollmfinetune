import React, { useState, useEffect, useRef, useCallback } from 'react';
import '../styles/design-system.css';

// Types
interface QualityMetrics {
  overall_score: number;
  text_quality: {
    coherence: number;
    fluency: number;
    relevance: number;
    diversity: number;
    grammar_score: number;
    readability: number;
    toxicity_score: number;
    bias_score: number;
  };
  audio_quality: {
    clarity: number;
    naturalness: number;
    emotional_consistency: number;
    pronunciation: number;
    prosody: number;
    background_noise: number;
    volume_consistency: number;
    artifacts_score: number;
  };
  character_consistency: {
    personality_alignment: number;
    voice_consistency: number;
    behavioral_consistency: number;
    emotional_range: number;
    dialogue_style: number;
    character_arc: number;
  };
  dataset_balance: {
    narrative_type_distribution: Record<string, number>;
    character_distribution: number;
    emotion_distribution: number;
    length_distribution: number;
  };
  technical_metrics: {
    processing_time: number;
    error_rate: number;
    completion_rate: number;
    resource_efficiency: number;
    scalability_score: number;
  };
}

interface ValidationIssue {
  severity: 'low' | 'medium' | 'high';
  category: string;
  message: string;
  affected_samples: number;
  suggestions: string[];
}

interface ValidationReport {
  id: string;
  dataset_id: string;
  dataset_name: string;
  created_at: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  metrics: QualityMetrics;
  issues: ValidationIssue[];
  recommendations: string[];
  quality_trend: Array<{
    timestamp: string;
    score: number;
  }>;
}

interface ImprovementSuggestion {
  id: string;
  category: string;
  priority: 'low' | 'medium' | 'high';
  title: string;
  description: string;
  impact: string;
  effort: string;
  estimated_time: string;
  implementation_steps: string[];
}

interface DatasetQualityDashboardProps {
  onValidationRequest?: (datasetId: string) => void;
  onComparisonRequest?: (dataset1Id: string, dataset2Id: string) => void;
  onSuggestionApply?: (suggestionId: string) => void;
}

const DatasetQualityDashboard: React.FC<DatasetQualityDashboardProps> = ({
  onValidationRequest,
  onComparisonRequest,
  onSuggestionApply
}) => {
  const [qualityMetrics, setQualityMetrics] = useState<QualityMetrics | null>(null);
  const [validationHistory, setValidationHistory] = useState<ValidationReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<ValidationReport | null>(null);
  const [improvementSuggestions, setImprovementSuggestions] = useState<ImprovementSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedSuggestions, setExpandedSuggestions] = useState<Set<string>>(new Set());
  const [showValidationModal, setShowValidationModal] = useState(false);
  const [showComparisonModal, setShowComparisonModal] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [validationProgress, setValidationProgress] = useState<{
    id: string;
    progress: number;
    current_step: string;
  } | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const updateTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Debounced update function
  const debounceUpdate = useCallback((updateFn: () => void) => {
    if (updateTimeoutRef.current) {
      clearTimeout(updateTimeoutRef.current);
    }
    updateTimeoutRef.current = setTimeout(updateFn, 100);
  }, []);

  // Load quality metrics
  const loadQualityMetrics = useCallback(async () => {
    try {
      const response = await fetch('/api/multimodal/quality/metrics');
      if (!response.ok) {
        throw new Error('Failed to load quality metrics');
      }
      const data = await response.json();
      setQualityMetrics(data.metrics);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load quality metrics');
    }
  }, []);

  // Load validation history
  const loadValidationHistory = useCallback(async () => {
    try {
      const response = await fetch('/api/multimodal/quality/validation-history');
      if (!response.ok) {
        throw new Error('Failed to load validation history');
      }
      const data = await response.json();
      setValidationHistory(data.history);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load validation history');
    }
  }, []);

  // Load improvement suggestions
  const loadImprovementSuggestions = useCallback(async () => {
    try {
      const response = await fetch('/api/multimodal/quality/improvement-suggestions');
      if (!response.ok) {
        throw new Error('Failed to load improvement suggestions');
      }
      const data = await response.json();
      setImprovementSuggestions(data.suggestions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load improvement suggestions');
    }
  }, []);

  // Initialize component
  useEffect(() => {
    const initialize = async () => {
      setIsLoading(true);
      try {
        await Promise.all([
          loadQualityMetrics(),
          loadValidationHistory(),
          loadImprovementSuggestions()
        ]);
      } catch (err) {
        console.error('Failed to initialize dashboard:', err);
      } finally {
        setIsLoading(false);
      }
    };

    initialize();
  }, [loadQualityMetrics, loadValidationHistory, loadImprovementSuggestions]);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws/quality-monitoring');
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('Quality monitoring WebSocket connected');
        };

        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          
          switch (data.type) {
            case 'quality_update':
              debounceUpdate(() => {
                setQualityMetrics(data.metrics);
              });
              break;
              
            case 'validation_progress':
              setValidationProgress({
                id: data.validation_id,
                progress: data.progress,
                current_step: data.current_step
              });
              break;
              
            case 'validation_completed':
              setValidationProgress(null);
              loadValidationHistory();
              break;
              
            default:
              console.log('Unknown message type:', data.type);
          }
        };

        ws.onclose = () => {
          console.log('Quality monitoring WebSocket disconnected');
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (error) => {
          console.error('Quality monitoring WebSocket error:', error);
        };
      } catch (err) {
        console.error('Failed to connect quality monitoring WebSocket:', err);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current && typeof wsRef.current.close === 'function') {
        wsRef.current.close();
      }
      if (updateTimeoutRef.current) {
        clearTimeout(updateTimeoutRef.current);
      }
    };
  }, [debounceUpdate, loadValidationHistory]);

  // Handle validation report click
  const handleValidationReportClick = async (validationId: string) => {
    try {
      const response = await fetch(`/api/multimodal/quality/validation-report/${validationId}`);
      if (!response.ok) {
        throw new Error('Failed to load validation report');
      }
      const data = await response.json();
      setSelectedReport(data.report);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load validation report');
    }
  };

  // Handle new validation request
  const handleValidationRequest = async (datasetId: string) => {
    try {
      const response = await fetch(`/api/multimodal/datasets/${datasetId}/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Validation failed');
      }
      
      const data = await response.json();
      setValidationProgress({
        id: data.validation_id,
        progress: 0,
        current_step: 'Starting validation...'
      });
      
      setShowValidationModal(false);
      onValidationRequest?.(datasetId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Validation failed');
    }
  };

  // Handle comparison request
  const handleComparisonRequest = async (dataset1Id: string, dataset2Id: string) => {
    try {
      const response = await fetch('/api/multimodal/quality/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset1_id: dataset1Id,
          dataset2_id: dataset2Id
        })
      });
      
      if (!response.ok) {
        throw new Error('Comparison failed');
      }
      
      setShowComparisonModal(false);
      onComparisonRequest?.(dataset1Id, dataset2Id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed');
    }
  };

  // Handle improvement suggestion application
  const handleSuggestionApply = async (suggestionId: string) => {
    try {
      const response = await fetch('/api/multimodal/quality/apply-suggestion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ suggestion_id: suggestionId })
      });
      
      if (!response.ok) {
        throw new Error('Failed to apply suggestion');
      }
      
      // Refresh suggestions and metrics
      loadImprovementSuggestions();
      loadQualityMetrics();
      
      onSuggestionApply?.(suggestionId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to apply suggestion');
    }
  };

  // Handle quality trend analysis
  const handleTrendAnalysis = async () => {
    try {
      const response = await fetch('/api/multimodal/quality/trends', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ time_range: '7d' })
      });
      
      if (!response.ok) {
        throw new Error('Failed to analyze trends');
      }
      
      const data = await response.json();
      // Handle trend analysis results
      console.log('Trend analysis results:', data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze trends');
    }
  };

  // Toggle suggestion expansion
  const toggleSuggestionExpansion = (suggestionId: string) => {
    setExpandedSuggestions(prev => {
      const newSet = new Set(prev);
      if (newSet.has(suggestionId)) {
        newSet.delete(suggestionId);
      } else {
        newSet.add(suggestionId);
      }
      return newSet;
    });
  };

  // Format percentage
  const formatPercentage = (value: number) => {
    return `${Math.round(value * 100)}%`;
  };

  // Get severity color
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'var(--color-error)';
      case 'medium': return 'var(--color-warning)';
      case 'low': return 'var(--color-success)';
      default: return 'var(--color-text-secondary)';
    }
  };

  // Get priority color
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'var(--color-error)';
      case 'medium': return 'var(--color-warning)';
      case 'low': return 'var(--color-success)';
      default: return 'var(--color-text-secondary)';
    }
  };

  // Render quality overview
  const renderQualityOverview = () => (
    <div className="quality-overview card" aria-label="Quality metrics overview">
      <div className="overview-header">
        <h3>Quality Overview</h3>
        <div className="overview-actions">
          <button 
            className="btn btn-primary"
            onClick={() => setShowValidationModal(true)}
          >
            Run Quality Validation
          </button>
          <button 
            className="btn btn-secondary"
            onClick={() => setShowComparisonModal(true)}
          >
            Compare Datasets
          </button>
          <button 
            className="btn btn-secondary"
            onClick={handleTrendAnalysis}
          >
            Analyze Quality Trends
          </button>
        </div>
      </div>

      {qualityMetrics && (
        <>
          <div className="overall-score">
            <div className="score-circle">
              <div className="score-text">
                Overall Quality: {formatPercentage(qualityMetrics.overall_score)}
              </div>
            </div>
          </div>

          <div className="quality-breakdown">
            <div className="quality-category">
              <h4>Text Quality: {formatPercentage(qualityMetrics.text_quality?.coherence || 0)}</h4>
              <div className="quality-metrics">
                <div className="metric-item" data-testid="metric-coherence" onClick={() => setSelectedMetric('coherence')}>
                  <span>Coherence: {formatPercentage(qualityMetrics.text_quality?.coherence || 0)}</span>
                  <div className="metric-bar">
                    <div 
                      className="metric-fill" 
                      style={{ width: formatPercentage(qualityMetrics.text_quality?.coherence || 0) }}
                    />
                  </div>
                </div>
                <div className="metric-item">
                  <span>Fluency: {formatPercentage(qualityMetrics.text_quality?.fluency || 0)}</span>
                  <div className="metric-bar">
                    <div 
                      className="metric-fill" 
                      style={{ width: formatPercentage(qualityMetrics.text_quality?.fluency || 0) }}
                    />
                  </div>
                </div>
                <div className="metric-item">
                  <span>Relevance: {formatPercentage(qualityMetrics.text_quality?.relevance || 0)}</span>
                  <div className="metric-bar">
                    <div 
                      className="metric-fill" 
                      style={{ width: formatPercentage(qualityMetrics.text_quality?.relevance || 0) }}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="quality-category">
              <h4>Audio Quality: {formatPercentage(qualityMetrics.audio_quality?.clarity || 0)}</h4>
              <div className="quality-metrics">
                <div className="metric-item">
                  <span>Clarity: {formatPercentage(qualityMetrics.audio_quality?.clarity || 0)}</span>
                  <div className="metric-bar">
                    <div 
                      className="metric-fill" 
                      style={{ width: formatPercentage(qualityMetrics.audio_quality?.clarity || 0) }}
                    />
                  </div>
                </div>
                <div className="metric-item">
                  <span>Naturalness: {formatPercentage(qualityMetrics.audio_quality?.naturalness || 0)}</span>
                  <div className="metric-bar">
                    <div 
                      className="metric-fill" 
                      style={{ width: formatPercentage(qualityMetrics.audio_quality?.naturalness || 0) }}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="quality-category">
              <h4>Character Consistency: {formatPercentage(qualityMetrics.character_consistency?.personality_alignment || 0)}</h4>
              <div className="quality-metrics">
                <div className="metric-item">
                  <span>Personality Alignment: {formatPercentage(qualityMetrics.character_consistency?.personality_alignment || 0)}</span>
                  <div className="metric-bar">
                    <div 
                      className="metric-fill" 
                      style={{ width: formatPercentage(qualityMetrics.character_consistency?.personality_alignment || 0) }}
                    />
                  </div>
                </div>
                <div className="metric-item">
                  <span>Voice Consistency: {formatPercentage(qualityMetrics.character_consistency?.voice_consistency || 0)}</span>
                  <div className="metric-bar">
                    <div 
                      className="metric-fill" 
                      style={{ width: formatPercentage(qualityMetrics.character_consistency?.voice_consistency || 0) }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="technical-metrics">
            <h4>Technical Performance</h4>
            <div className="tech-metrics-grid">
              <div className="tech-metric">
                <span>Processing Time: {qualityMetrics.technical_metrics?.processing_time || 0}s</span>
              </div>
              <div className="tech-metric">
                <span>Error Rate: {formatPercentage(qualityMetrics.technical_metrics?.error_rate || 0)}</span>
              </div>
              <div className="tech-metric">
                <span>Completion Rate: {formatPercentage(qualityMetrics.technical_metrics?.completion_rate || 0)}</span>
              </div>
            </div>
          </div>

          <div className="dataset-balance">
            <h4>Dataset Balance</h4>
            <div className="balance-grid">
              {Object.entries(qualityMetrics.dataset_balance?.narrative_type_distribution || {}).map(([type, value]) => (
                <div key={type} className="balance-item">
                  <span>{type.charAt(0).toUpperCase() + type.slice(1).replace('_', ' ')}: {formatPercentage(value)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="quality-trend" data-testid="quality-trend-chart">
            <h4>Quality Trend</h4>
            <div className="trend-chart">
              {/* Placeholder for trend chart */}
              <div className="trend-line">
                <div className="trend-point" style={{ left: '10%', bottom: '82%' }} />
                <div className="trend-point" style={{ left: '50%', bottom: '84%' }} />
                <div className="trend-point" style={{ left: '90%', bottom: '85%' }} />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );

  // Render validation reports
  const renderValidationReports = () => (
    <div className="validation-reports card" aria-label="Validation reports list">
      <h3>Validation Reports</h3>
      
      {validationProgress && (
        <div className="validation-progress">
          <div className="progress-header">
            <span>Validation in Progress</span>
            <span>{validationProgress.progress}%</span>
          </div>
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${validationProgress.progress}%` }}
            />
          </div>
          <div className="progress-step">{validationProgress.current_step}</div>
        </div>
      )}

      <div className="reports-list">
        {validationHistory.map((report) => (
          <div 
            key={report.id}
            className="report-item"
            data-testid={`validation-report-${report.id}`}
            onClick={() => handleValidationReportClick(report.id)}
          >
            <div className="report-header">
              <span className="report-name">{report.dataset_name}</span>
              <span className="report-score">{formatPercentage(report.metrics.overall_score)}</span>
            </div>
            <div className="report-meta">
              <span className="report-date">{new Date(report.created_at).toLocaleDateString()}</span>
              <span className={`report-status ${report.status}`}>{report.status}</span>
            </div>
          </div>
        ))}
      </div>

      {selectedReport && (
        <div className="report-details-modal">
          <div className="report-details">
            <div className="report-details-header">
              <h3>Validation Report Details</h3>
              <button 
                className="btn btn-ghost"
                onClick={() => setSelectedReport(null)}
              >
                ×
              </button>
            </div>
            
            <div className="report-content">
              <div className="issues-section">
                <h4>Issues Found</h4>
                {selectedReport.issues.map((issue, index) => (
                  <div key={index} className="issue-item">
                    <div className="issue-header">
                      <span 
                        className="issue-severity"
                        data-testid={`issue-severity-${issue.severity}`}
                        style={{ color: getSeverityColor(issue.severity) }}
                      >
                        {issue.severity.toUpperCase()}
                      </span>
                      <span className="issue-category">{issue.category}</span>
                    </div>
                    <div className="issue-message">{issue.message}</div>
                    <div className="issue-affected">Affected samples: {issue.affected_samples}</div>
                    <div className="issue-suggestions">
                      <h5>Suggestions:</h5>
                      <ul>
                        {issue.suggestions.map((suggestion, idx) => (
                          <li key={idx}>{suggestion}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>

              <div className="recommendations-section">
                <h4>Recommendations</h4>
                <ul>
                  {selectedReport.recommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  // Render improvement suggestions
  const renderImprovementSuggestions = () => (
    <div className="improvement-suggestions card" aria-label="Improvement suggestions">
      <h3>Improvement Suggestions</h3>
      
      <div className="suggestions-list">
        {improvementSuggestions.map((suggestion) => (
          <div key={suggestion.id} className="suggestion-item">
            <div className="suggestion-header">
              <div className="suggestion-title">
                <span className="suggestion-name">{suggestion.title}</span>
                <span 
                  className="suggestion-priority"
                  data-testid={`suggestion-priority-${suggestion.priority}`}
                  style={{ color: getPriorityColor(suggestion.priority) }}
                >
                  {suggestion.priority.toUpperCase()}
                </span>
              </div>
              <div className="suggestion-actions">
                <button 
                  className="btn btn-ghost btn-sm"
                  data-testid={`expand-suggestion-${suggestion.id}`}
                  onClick={() => toggleSuggestionExpansion(suggestion.id)}
                >
                  {expandedSuggestions.has(suggestion.id) ? '−' : '+'}
                </button>
                <button 
                  className="btn btn-primary btn-sm"
                  data-testid={`apply-suggestion-${suggestion.id}`}
                  onClick={() => handleSuggestionApply(suggestion.id)}
                >
                  Apply
                </button>
              </div>
            </div>
            
            <div className="suggestion-description">{suggestion.description}</div>
            
            {expandedSuggestions.has(suggestion.id) && (
              <div className="suggestion-details">
                <div className="suggestion-impact">{suggestion.impact}</div>
                <div className="suggestion-effort">
                  <span>Effort: {suggestion.effort}</span>
                  <span>Estimated Time: {suggestion.estimated_time}</span>
                </div>
                <div className="suggestion-steps">
                  <h5>Implementation Steps:</h5>
                  <ol>
                    {suggestion.implementation_steps.map((step, index) => (
                      <li key={index}>{step}</li>
                    ))}
                  </ol>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  // Render validation modal
  const renderValidationModal = () => {
    if (!showValidationModal) return null;

    return (
      <div className="modal-overlay">
        <div className="modal">
          <div className="modal-header">
            <h3>Select Dataset</h3>
            <button 
              className="btn btn-ghost"
              onClick={() => setShowValidationModal(false)}
            >
              ×
            </button>
          </div>
          <div className="modal-content">
            <div className="form-group">
              <label>Dataset</label>
              <select data-testid="dataset-select">
                <option value="">Select a dataset</option>
                <option value="dataset-123">Dataset 123</option>
                <option value="dataset-456">Dataset 456</option>
              </select>
            </div>
            <div className="modal-actions">
              <button 
                className="btn btn-secondary"
                onClick={() => setShowValidationModal(false)}
              >
                Cancel
              </button>
              <button 
                className="btn btn-primary"
                onClick={() => handleValidationRequest('dataset-123')}
              >
                Start Validation
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Render comparison modal
  const renderComparisonModal = () => {
    if (!showComparisonModal) return null;

    return (
      <div className="modal-overlay">
        <div className="modal">
          <div className="modal-header">
            <h3>Select Datasets to Compare</h3>
            <button 
              className="btn btn-ghost"
              onClick={() => setShowComparisonModal(false)}
            >
              ×
            </button>
          </div>
          <div className="modal-content">
            <div className="form-group">
              <label>Dataset 1</label>
              <select data-testid="compare-dataset-1">
                <option value="">Select first dataset</option>
                <option value="dataset-123">Dataset 123</option>
                <option value="dataset-456">Dataset 456</option>
              </select>
            </div>
            <div className="form-group">
              <label>Dataset 2</label>
              <select data-testid="compare-dataset-2">
                <option value="">Select second dataset</option>
                <option value="dataset-123">Dataset 123</option>
                <option value="dataset-456">Dataset 456</option>
              </select>
            </div>
            <div className="modal-actions">
              <button 
                className="btn btn-secondary"
                onClick={() => setShowComparisonModal(false)}
              >
                Cancel
              </button>
              <button 
                className="btn btn-primary"
                onClick={() => handleComparisonRequest('dataset-123', 'dataset-456')}
              >
                Start Comparison
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Render metric details modal
  const renderMetricDetailsModal = () => {
    if (!selectedMetric) return null;

    return (
      <div className="modal-overlay">
        <div className="modal">
          <div className="modal-header">
            <h3>Coherence Details</h3>
            <button 
              className="btn btn-ghost"
              onClick={() => setSelectedMetric(null)}
            >
              ×
            </button>
          </div>
          <div className="modal-content">
            <div className="metric-details">
              <h4>Sample Analysis</h4>
              <div className="sample-breakdown">
                <div className="sample-category">
                  <span>High Coherence Samples: 78%</span>
                  <div className="sample-bar">
                    <div className="sample-fill" style={{ width: '78%' }} />
                  </div>
                </div>
                <div className="sample-category">
                  <span>Medium Coherence Samples: 18%</span>
                  <div className="sample-bar">
                    <div className="sample-fill" style={{ width: '18%' }} />
                  </div>
                </div>
                <div className="sample-category">
                  <span>Low Coherence Samples: 4%</span>
                  <div className="sample-bar">
                    <div className="sample-fill" style={{ width: '4%' }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="dataset-quality-dashboard" aria-label="Dataset quality dashboard">
      <div className="dashboard-header">
        <h1>Dataset Quality Dashboard</h1>
      </div>

      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {isLoading ? (
        <div className="loading-message">Loading quality metrics...</div>
      ) : (
        <div className="dashboard-content">
          <div className="dashboard-row">
            {renderQualityOverview()}
          </div>
          
          <div className="dashboard-row">
            {renderValidationReports()}
          </div>
          
          <div className="dashboard-row">
            {renderImprovementSuggestions()}
          </div>
        </div>
      )}

      {renderValidationModal()}
      {renderComparisonModal()}
      {renderMetricDetailsModal()}

      <style>{`
        .dataset-quality-dashboard {
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

        .overview-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-4);
        }

        .overview-actions {
          display: flex;
          gap: var(--space-2);
        }

        .overall-score {
          text-align: center;
          margin-bottom: var(--space-6);
        }

        .score-circle {
          width: 120px;
          height: 120px;
          border-radius: 50%;
          background: conic-gradient(var(--color-primary) 0deg 306deg, var(--color-surface) 306deg 360deg);
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto;
          position: relative;
        }

        .score-circle::before {
          content: '';
          position: absolute;
          width: 90px;
          height: 90px;
          border-radius: 50%;
          background: var(--color-surface);
        }

        .score-text {
          font-size: var(--text-lg);
          font-weight: var(--font-bold);
          color: var(--color-text-primary);
          z-index: 1;
        }

        .quality-breakdown {
          display: grid;
          grid-template-columns: 1fr;
          gap: var(--space-4);
          margin-bottom: var(--space-6);
        }

        .quality-category h4 {
          margin-bottom: var(--space-3);
          color: var(--color-text-primary);
        }

        .quality-metrics {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
        }

        .metric-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: var(--space-2);
          border-radius: var(--radius-md);
          cursor: pointer;
          transition: background var(--duration-normal);
        }

        .metric-item:hover {
          background: var(--color-surface-hover);
        }

        .metric-bar {
          width: 100px;
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

        .technical-metrics {
          margin-bottom: var(--space-6);
        }

        .tech-metrics-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: var(--space-4);
        }

        .tech-metric {
          padding: var(--space-3);
          background: var(--color-background);
          border-radius: var(--radius-md);
          text-align: center;
        }

        .dataset-balance {
          margin-bottom: var(--space-6);
        }

        .balance-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: var(--space-2);
        }

        .balance-item {
          padding: var(--space-2);
          background: var(--color-background);
          border-radius: var(--radius-md);
          text-align: center;
        }

        .quality-trend {
          margin-bottom: var(--space-6);
        }

        .trend-chart {
          height: 100px;
          background: var(--color-background);
          border-radius: var(--radius-md);
          position: relative;
          overflow: hidden;
        }

        .trend-line {
          position: relative;
          height: 100%;
        }

        .trend-point {
          position: absolute;
          width: 8px;
          height: 8px;
          background: var(--color-primary);
          border-radius: 50%;
          transform: translate(-50%, 50%);
        }

        .validation-progress {
          margin-bottom: var(--space-4);
          padding: var(--space-4);
          background: var(--color-background);
          border-radius: var(--radius-md);
        }

        .progress-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-2);
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

        .progress-step {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
          font-style: italic;
        }

        .reports-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
        }

        .report-item {
          padding: var(--space-4);
          background: var(--color-background);
          border-radius: var(--radius-md);
          cursor: pointer;
          transition: background var(--duration-normal);
        }

        .report-item:hover {
          background: var(--color-surface-hover);
        }

        .report-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-2);
        }

        .report-name {
          font-weight: var(--font-medium);
        }

        .report-score {
          font-size: var(--text-lg);
          font-weight: var(--font-bold);
          color: var(--color-primary);
        }

        .report-meta {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
        }

        .report-status {
          padding: var(--space-1) var(--space-2);
          border-radius: var(--radius-sm);
          font-size: var(--text-xs);
          font-weight: var(--font-medium);
          text-transform: uppercase;
        }

        .report-status.completed {
          background: var(--color-success-light);
          color: var(--color-success-dark);
        }

        .report-status.running {
          background: var(--color-warning-light);
          color: var(--color-warning-dark);
        }

        .report-status.failed {
          background: var(--color-error-light);
          color: var(--color-error-dark);
        }

        .report-details-modal {
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

        .report-details {
          background: var(--color-surface);
          border-radius: var(--radius-lg);
          padding: var(--space-6);
          max-width: 800px;
          width: 90%;
          max-height: 80vh;
          overflow-y: auto;
        }

        .report-details-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-4);
        }

        .report-content {
          display: flex;
          flex-direction: column;
          gap: var(--space-6);
        }

        .issues-section {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }

        .issue-item {
          padding: var(--space-4);
          background: var(--color-background);
          border-radius: var(--radius-md);
          border-left: 4px solid var(--color-border);
        }

        .issue-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-2);
        }

        .issue-severity {
          font-size: var(--text-sm);
          font-weight: var(--font-bold);
        }

        .issue-category {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
          text-transform: capitalize;
        }

        .issue-message {
          margin-bottom: var(--space-2);
          color: var(--color-text-primary);
        }

        .issue-affected {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
          margin-bottom: var(--space-3);
        }

        .issue-suggestions h5 {
          margin-bottom: var(--space-2);
          color: var(--color-text-primary);
        }

        .issue-suggestions ul {
          margin: 0;
          padding-left: var(--space-4);
        }

        .issue-suggestions li {
          margin-bottom: var(--space-1);
          color: var(--color-text-secondary);
        }

        .recommendations-section h4 {
          margin-bottom: var(--space-3);
          color: var(--color-text-primary);
        }

        .recommendations-section ul {
          margin: 0;
          padding-left: var(--space-4);
        }

        .recommendations-section li {
          margin-bottom: var(--space-2);
          color: var(--color-text-secondary);
        }

        .suggestions-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }

        .suggestion-item {
          padding: var(--space-4);
          background: var(--color-background);
          border-radius: var(--radius-md);
          border-left: 4px solid var(--color-border);
        }

        .suggestion-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-2);
        }

        .suggestion-title {
          display: flex;
          align-items: center;
          gap: var(--space-2);
        }

        .suggestion-name {
          font-weight: var(--font-medium);
          color: var(--color-text-primary);
        }

        .suggestion-priority {
          font-size: var(--text-xs);
          font-weight: var(--font-bold);
          padding: var(--space-1) var(--space-2);
          border-radius: var(--radius-sm);
          background: var(--color-background);
        }

        .suggestion-actions {
          display: flex;
          gap: var(--space-2);
        }

        .suggestion-description {
          color: var(--color-text-secondary);
          margin-bottom: var(--space-3);
        }

        .suggestion-details {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
          padding-top: var(--space-3);
          border-top: 1px solid var(--color-border);
        }

        .suggestion-impact {
          color: var(--color-text-primary);
          font-style: italic;
        }

        .suggestion-effort {
          display: flex;
          gap: var(--space-4);
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
        }

        .suggestion-steps h5 {
          margin-bottom: var(--space-2);
          color: var(--color-text-primary);
        }

        .suggestion-steps ol {
          margin: 0;
          padding-left: var(--space-4);
        }

        .suggestion-steps li {
          margin-bottom: var(--space-1);
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
          max-width: 500px;
          width: 90%;
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

        .form-group {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
        }

        .form-group label {
          font-weight: var(--font-medium);
          color: var(--color-text-primary);
        }

        .form-group select {
          padding: var(--space-3);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          font-size: var(--text-base);
        }

        .modal-actions {
          display: flex;
          gap: var(--space-2);
          justify-content: flex-end;
          margin-top: var(--space-4);
        }

        .metric-details {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }

        .sample-breakdown {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }

        .sample-category {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .sample-bar {
          width: 200px;
          height: 6px;
          background: var(--color-surface);
          border-radius: var(--radius-full);
          overflow: hidden;
        }

        .sample-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--color-primary), var(--color-primary-light));
        }

        @media (max-width: 768px) {
          .dashboard-row {
            flex-direction: column;
          }
          
          .overview-header {
            flex-direction: column;
            gap: var(--space-2);
          }
          
          .overview-actions {
            flex-direction: column;
            width: 100%;
          }
          
          .quality-breakdown {
            grid-template-columns: 1fr;
          }
          
          .tech-metrics-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default DatasetQualityDashboard; 