import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  LineChart,
  AreaChart,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Line,
  Area,
  Bar,
  ResponsiveContainer
} from 'recharts';

// Types
interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: {
    in: number;
    out: number;
  };
}

interface JobMetrics {
  active_jobs: number;
  queue_length: number;
  avg_processing_time: number;
  error_rate: number;
}

interface PerformanceTrend {
  timestamp: string;
  cpu: number;
  memory: number;
  response_time: number;
}

interface Bottleneck {
  id?: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  suggestions: string[];
}

interface OptimizationSuggestion {
  id?: string;
  type: string;
  priority: 'low' | 'medium' | 'high';
  description: string;
  impact: string;
  effort: string;
}

interface PerformanceAlert {
  id: string;
  type: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  timestamp: string;
}

interface ServiceMetrics {
  name: string;
  cpu: number;
  memory: number;
  disk_io: number;
}

interface PerformanceData {
  system_metrics: SystemMetrics;
  job_metrics: JobMetrics;
  performance_trends: PerformanceTrend[];
  bottlenecks: Bottleneck[];
}

const PerformanceMonitoringDashboard: React.FC = () => {
  // State management
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null);
  const [optimizationSuggestions, setOptimizationSuggestions] = useState<OptimizationSuggestion[]>([]);
  const [performanceAlerts, setPerformanceAlerts] = useState<PerformanceAlert[]>([]);
  const [serviceMetrics, setServiceMetrics] = useState<ServiceMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wsConnectionError, setWsConnectionError] = useState(false);
  
  // UI state
  const [activeTab, setActiveTab] = useState('overview');
  const [timeRange, setTimeRange] = useState('5m');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState('5s');
  
  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch performance metrics
  const fetchPerformanceMetrics = useCallback(async (timeRangeParam?: string) => {
    try {
      setError(null);
      const range = timeRangeParam || timeRange;
      const response = await fetch(`/api/performance/metrics?time_range=${range}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch performance metrics');
      }
      
      const data = await response.json();
      setPerformanceData(data);
      setLoading(false);
    } catch (err) {
      setError('Error loading performance metrics');
      setLoading(false);
    }
  }, [timeRange]);

  // Fetch optimization suggestions
  const fetchOptimizationSuggestions = async () => {
    try {
      const response = await fetch('/api/performance/optimizations/suggestions');
      if (response.ok) {
        const data = await response.json();
        setOptimizationSuggestions(data.suggestions || []);
      }
    } catch (err) {
      console.error('Failed to fetch optimization suggestions:', err);
    }
  };

  // Fetch performance alerts
  const fetchPerformanceAlerts = async () => {
    try {
      const response = await fetch('/api/performance/alerts');
      if (response.ok) {
        const data = await response.json();
        setPerformanceAlerts(data.alerts || []);
      }
    } catch (err) {
      console.error('Failed to fetch performance alerts:', err);
    }
  };

  // Fetch service breakdown
  const fetchServiceBreakdown = async () => {
    try {
      const response = await fetch('/api/performance/services/breakdown');
      if (response.ok) {
        const data = await response.json();
        setServiceMetrics(data.services || []);
      }
    } catch (err) {
      console.error('Failed to fetch service breakdown:', err);
    }
  };

  // WebSocket setup
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/performance-monitoring');
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnectionError(false);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        switch (message.type) {
          case 'metrics_update':
            setPerformanceData(prev => prev ? {
              ...prev,
              system_metrics: message.data.system_metrics || prev.system_metrics,
              job_metrics: message.data.job_metrics || prev.job_metrics
            } : null);
            break;
            
          case 'bottleneck_detected':
            setPerformanceData(prev => prev ? {
              ...prev,
              bottlenecks: [...prev.bottlenecks, message.data]
            } : null);
            // If it's a critical bottleneck, the component will automatically add role="alert"
            break;
            
          case 'trend_update':
            setPerformanceData(prev => prev ? {
              ...prev,
              performance_trends: [...prev.performance_trends, message.data]
            } : null);
            break;
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    ws.onerror = () => {
      setWsConnectionError(true);
    };

    return () => {
      if (ws && typeof ws.close === 'function') {
        ws.close();
      }
    };
  }, []);

  // Auto-refresh setup
  useEffect(() => {
    if (autoRefresh && refreshInterval) {
      const intervalMs = parseInt(refreshInterval) * 1000;
      refreshIntervalRef.current = setInterval(() => {
        fetchPerformanceMetrics();
      }, intervalMs);
    } else if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current);
      refreshIntervalRef.current = null;
    }

    return () => {
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, [autoRefresh, refreshInterval, fetchPerformanceMetrics]);

  // Initial data load
  useEffect(() => {
    fetchPerformanceMetrics();
  }, [fetchPerformanceMetrics]);

  // Handle time range change
  const handleTimeRangeChange = (newTimeRange: string) => {
    setTimeRange(newTimeRange);
    fetchPerformanceMetrics(newTimeRange);
  };

  // Handle bottleneck resolution
  const handleResolveBottleneck = async (bottleneckId: string) => {
    try {
      await fetch('/api/performance/bottlenecks/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bottleneck_id: bottleneckId })
      });
      
      // Remove resolved bottleneck from state
      setPerformanceData(prev => prev ? {
        ...prev,
        bottlenecks: prev.bottlenecks.filter(b => b.id !== bottleneckId)
      } : null);
    } catch (err) {
      console.error('Failed to resolve bottleneck:', err);
    }
  };

  // Handle optimization suggestion application
  const handleApplySuggestion = async (suggestionId: string) => {
    try {
      await fetch('/api/performance/optimizations/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ suggestion_id: suggestionId })
      });
      
      // Remove applied suggestion from state
      setOptimizationSuggestions(prev => prev.filter(s => s.id !== suggestionId));
    } catch (err) {
      console.error('Failed to apply optimization suggestion:', err);
    }
  };

  // Handle alert acknowledgment
  const handleAcknowledgeAlert = async (alertId: string) => {
    try {
      await fetch('/api/performance/alerts/acknowledge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alert_id: alertId })
      });
      
      setPerformanceAlerts(prev => prev.filter(a => a.id !== alertId));
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
    }
  };

  // Handle report export
  const handleExportReport = async () => {
    try {
      await fetch('/api/performance/reports/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: 'pdf', time_range: '24h' })
      });
    } catch (err) {
      console.error('Failed to export report:', err);
    }
  };

  // Filter bottlenecks by severity
  const filteredBottlenecks = performanceData?.bottlenecks?.filter(bottleneck => 
    severityFilter === 'all' || bottleneck.severity === severityFilter
  ) || [];

  // Format percentage
  const formatPercentage = (value: number) => `${value.toFixed(1)}%`;

  // Format time
  const formatTime = (value: number) => `${value}s`;

  if (loading) {
    return (
      <div className="performance-dashboard loading" aria-label="Performance monitoring dashboard">
        <div>Loading performance metrics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="performance-dashboard error">
        <div>Error loading performance metrics</div>
        <button onClick={() => fetchPerformanceMetrics()}>Retry</button>
      </div>
    );
  }

  return (
    <div className="performance-dashboard" aria-label="Performance monitoring dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <h1>Performance Monitoring</h1>
        
        {/* Controls */}
        <div className="dashboard-controls">
          <label>
            Time Range:
            <select 
              value={timeRange} 
              onChange={(e) => handleTimeRangeChange(e.target.value)}
              aria-label="Time range"
            >
              <option value="5m">5 minutes</option>
              <option value="1h">1 hour</option>
              <option value="24h">24 hours</option>
              <option value="7d">7 days</option>
            </select>
          </label>
          
          <label>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              aria-label="Auto refresh"
            />
            Auto Refresh
          </label>
          
          <label>
            Refresh Interval:
            <select
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(e.target.value)}
              aria-label="Refresh interval"
            >
              <option value="5s">5 seconds</option>
              <option value="10s">10 seconds</option>
              <option value="30s">30 seconds</option>
              <option value="60s">1 minute</option>
            </select>
          </label>
          
          <button onClick={() => fetchPerformanceMetrics()} aria-label="Refresh now">
            Refresh Now
          </button>
          
          <button onClick={handleExportReport} aria-label="Export report">
            Export Report
          </button>
        </div>
        
        {wsConnectionError && (
          <div className="connection-warning">
            Real-time updates unavailable
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="tab-navigation">
        <button
          className={activeTab === 'overview' ? 'active' : ''}
          onClick={() => setActiveTab('overview')}
          role="tab"
        >
          Overview
        </button>
        <button
          className={activeTab === 'resource-analysis' ? 'active' : ''}
          onClick={() => {
            setActiveTab('resource-analysis');
            if (serviceMetrics.length === 0) {
              fetchServiceBreakdown();
            }
          }}
          role="tab"
          aria-label="Resource analysis"
        >
          Resource Analysis
        </button>
        <button
          className={activeTab === 'alerts' ? 'active' : ''}
          onClick={() => {
            setActiveTab('alerts');
            if (performanceAlerts.length === 0) {
              fetchPerformanceAlerts();
            }
          }}
          role="tab"
          aria-label="Alerts"
        >
          Alerts
        </button>
        <button
          className={activeTab === 'history' ? 'active' : ''}
          onClick={() => setActiveTab('history')}
          role="tab"
          aria-label="History"
        >
          History
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="overview-tab">
          {/* System Metrics */}
          <section className="metrics-section" aria-label="System metrics overview">
            <h2>System Metrics</h2>
            <div className="metrics-grid">
              <div className="metric-card">
                <h3>CPU Usage</h3>
                <div className="metric-value">{formatPercentage(performanceData?.system_metrics.cpu_usage || 0)}</div>
              </div>
              <div className="metric-card">
                <h3>Memory Usage</h3>
                <div className="metric-value">{formatPercentage(performanceData?.system_metrics.memory_usage || 0)}</div>
              </div>
              <div className="metric-card">
                <h3>Disk Usage</h3>
                <div className="metric-value">{formatPercentage(performanceData?.system_metrics.disk_usage || 0)}</div>
              </div>
              <div className="metric-card">
                <h3>Network I/O</h3>
                <div className="metric-value">
                  In: {performanceData?.system_metrics.network_io.in || 0} MB/s<br/>
                  Out: {performanceData?.system_metrics.network_io.out || 0} MB/s
                </div>
              </div>
            </div>
          </section>

          {/* Job Performance */}
          <section className="metrics-section" aria-label="Job performance metrics">
            <h2>Job Performance</h2>
            <div className="metrics-grid">
              <div className="metric-card">
                <h3>Active Jobs</h3>
                <div className="metric-value">{performanceData?.job_metrics.active_jobs || 0}</div>
              </div>
              <div className="metric-card">
                <h3>Queue Length</h3>
                <div className="metric-value">{performanceData?.job_metrics.queue_length || 0}</div>
              </div>
              <div className="metric-card">
                <h3>Avg Processing Time</h3>
                <div className="metric-value">{formatTime(performanceData?.job_metrics.avg_processing_time || 0)}</div>
              </div>
              <div className="metric-card">
                <h3>Error Rate</h3>
                <div className="metric-value">{formatPercentage((performanceData?.job_metrics.error_rate || 0) * 100)}</div>
              </div>
            </div>
          </section>

          {/* Performance Trends */}
          <section className="chart-section">
            <h2>Performance Trends</h2>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={performanceData?.performance_trends || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="cpu" stroke="#8884d8" />
                  <Line type="monotone" dataKey="memory" stroke="#82ca9d" />
                  <Line type="monotone" dataKey="response_time" stroke="#ffc658" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={performanceData?.performance_trends || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <Tooltip />
                  <Area type="monotone" dataKey="cpu" stackId="1" stroke="#8884d8" fill="#8884d8" />
                  <Area type="monotone" dataKey="memory" stackId="1" stroke="#82ca9d" fill="#82ca9d" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* Bottleneck Detection */}
          <section className="bottlenecks-section">
            <h2>Bottleneck Detection</h2>
            
            <div className="bottleneck-controls">
              <label>
                Severity Filter:
                <select
                  value={severityFilter}
                  onChange={(e) => setSeverityFilter(e.target.value)}
                  aria-label="Severity filter"
                >
                  <option value="all">All</option>
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </label>
            </div>

            <div className="bottlenecks-list">
              {filteredBottlenecks.map((bottleneck, index) => (
                <div 
                  key={index} 
                  className={`bottleneck-item severity-${bottleneck.severity}`}
                  role={bottleneck.severity === 'critical' ? 'alert' : undefined}
                >
                  <div className="bottleneck-header">
                    <span className="bottleneck-type">{bottleneck.type}</span>
                    <span className="bottleneck-severity">{bottleneck.severity}</span>
                  </div>
                  <div className="bottleneck-description">{bottleneck.description}</div>
                  <div className="bottleneck-suggestions">
                    <h4>Suggestions:</h4>
                    <ul>
                      {bottleneck.suggestions.map((suggestion, i) => (
                        <li key={i}>{suggestion}</li>
                      ))}
                    </ul>
                  </div>
                  <button 
                    onClick={() => handleResolveBottleneck(bottleneck.id || `bottleneck-${index}`)}
                    aria-label="Resolve bottleneck"
                  >
                    Resolve Bottleneck
                  </button>
                </div>
              ))}
              
              {filteredBottlenecks.length === 0 && (
                <div className="no-bottlenecks">No bottlenecks detected</div>
              )}
            </div>
          </section>

          {/* Optimization Suggestions */}
          <section className="optimization-section">
            <h2>Optimization Suggestions</h2>
            
            <button 
              onClick={fetchOptimizationSuggestions}
              aria-label="Get optimization suggestions"
            >
              Get Optimization Suggestions
            </button>
            
            <div className="suggestions-list">
              {optimizationSuggestions.map((suggestion, index) => (
                <div key={index} className={`suggestion-item priority-${suggestion.priority}`}>
                  <div className="suggestion-header">
                    <span className="suggestion-type">{suggestion.type}</span>
                    <span className="suggestion-priority">{suggestion.priority}</span>
                  </div>
                  <div className="suggestion-description">{suggestion.description}</div>
                  <div className="suggestion-impact">Impact: {suggestion.impact}</div>
                  <div className="suggestion-effort">Effort: {suggestion.effort}</div>
                  <button 
                    onClick={() => handleApplySuggestion(suggestion.id || `suggestion-${index}`)}
                    aria-label="Apply suggestion"
                  >
                    Apply Suggestion
                  </button>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}

      {activeTab === 'resource-analysis' && (
        <div className="resource-analysis-tab">
          <h2>Resource Analysis</h2>
          
          <div className="resource-breakdown">
            <div className="resource-section">
              <h3>CPU Breakdown</h3>
            </div>
            <div className="resource-section">
              <h3>Memory Distribution</h3>
            </div>
            <div className="resource-section">
              <h3>Disk I/O Analysis</h3>
            </div>
            <div className="resource-section">
              <h3>Network Traffic</h3>
            </div>
          </div>
          
          <button 
            onClick={fetchServiceBreakdown}
            aria-label="Service breakdown"
          >
            Service Breakdown
          </button>
          
          <div className="service-metrics">
            {serviceMetrics.map((service, index) => (
              <div key={index} className="service-metric">
                <h4>{service.name}</h4>
                <div>CPU: {formatPercentage(service.cpu)}</div>
                <div>Memory: {service.memory} MB</div>
                <div>Disk I/O: {service.disk_io} MB/s</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'alerts' && (
        <div className="alerts-tab">
          <h2>Performance Alerts</h2>
          
          <button onClick={fetchPerformanceAlerts}>
            Refresh Alerts
          </button>
          
          <div className="alerts-list">
            {performanceAlerts.map((alert) => (
              <div key={alert.id} className={`alert-item severity-${alert.severity}`} role="alert">
                <div className="alert-header">
                  <span className="alert-type">{alert.type}</span>
                  <span className="alert-severity">{alert.severity}</span>
                  <span className="alert-timestamp">{alert.timestamp}</span>
                </div>
                <div className="alert-message">{alert.message}</div>
                <button 
                  onClick={() => handleAcknowledgeAlert(alert.id)}
                  aria-label="Acknowledge"
                >
                  Acknowledge
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'history' && (
        <div className="history-tab">
          <h2>Performance History</h2>
          
          <div className="history-chart">
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={performanceData?.performance_trends || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="cpu" stroke="#8884d8" />
                <Line type="monotone" dataKey="memory" stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceMonitoringDashboard; 