/**
 * Production Monitoring Dashboard for Dreamcast Platform
 * 
 * Comprehensive real-time monitoring interface with service health,
 * performance metrics, alerts, and user analytics visualization.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import './ProductionMonitoringDashboard.css';

// Types
interface PlatformMetrics {
  timestamp: string;
  api_performance: {
    response_time: number;
    requests_per_second: number;
    error_rate: number;
    active_connections: number;
    cache_hit_rate: number;
  };
  react_performance: {
    page_load_time: number;
    first_contentful_paint: number;
    active_users: number;
    bundle_size: number;
  };
  websocket_metrics: {
    active_connections: number;
    messages_per_second: number;
    connection_rate: number;
  };
  database_metrics: {
    query_time: number;
    connections_active: number;
    cache_hit_ratio: number;
    transactions_per_second: number;
  };
  voice_quality_metrics: {
    generation_latency: number;
    quality_scores: number;
    character_consistency: number;
    voice_generation_rate: number;
  };
  system_resources: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
    load_average: number;
  };
}

interface ServiceHealth {
  service_name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  response_time: number;
  error_rate: number;
  last_check: string;
  details: any;
}

interface Alert {
  level: 'info' | 'warning' | 'critical';
  message: string;
  metric: string;
  value: number;
  threshold: number;
  timestamp: string;
  service?: string;
  resolved: boolean;
}

interface UserAnalytics {
  user_engagement: {
    active_users: number;
    session_duration: number;
    bounce_rate: number;
    conversation_engagement: number;
  };
  platform_performance: {
    page_load_time: number;
    voice_generation_rate: number;
    character_creation_success: number;
  };
  feature_adoption: {
    new_feature_adoption: number;
    voice_feature_usage: number;
    journey_completion: number;
  };
}

interface MonitoringProps {
  wsUrl?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

const ProductionMonitoringDashboard: React.FC<MonitoringProps> = ({
  wsUrl = 'wss://api.dreamcast.dev/monitoring',
  autoRefresh = true,
  refreshInterval = 30000
}) => {
  // State management
  const [platformMetrics, setPlatformMetrics] = useState<PlatformMetrics | null>(null);
  const [serviceHealth, setServiceHealth] = useState<ServiceHealth[]>([]);
  const [userAnalytics, setUserAnalytics] = useState<UserAnalytics | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [metricsHistory, setMetricsHistory] = useState<PlatformMetrics[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('connecting');
  const [selectedTimeRange, setSelectedTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');
  const [healthScore, setHealthScore] = useState<number>(0);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    try {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        return;
      }

      wsRef.current = new WebSocket(wsUrl);
      
      wsRef.current.onopen = () => {
        console.log('Monitoring WebSocket connected');
        setConnectionStatus('connected');
        
        // Clear any reconnection timeout
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      wsRef.current.onclose = () => {
        console.log('Monitoring WebSocket disconnected');
        setConnectionStatus('disconnected');
        
        // Attempt to reconnect after 5 seconds
        if (autoRefresh) {
          reconnectTimeoutRef.current = setTimeout(connectWebSocket, 5000);
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('Monitoring WebSocket error:', error);
        setConnectionStatus('disconnected');
      };
      
    } catch (error) {
      console.error('Error connecting to monitoring WebSocket:', error);
      setConnectionStatus('disconnected');
    }
  }, [wsUrl, autoRefresh]);

  // Handle incoming WebSocket messages
  const handleWebSocketMessage = (data: any) => {
    switch (data.type) {
      case 'platform_metrics':
        setPlatformMetrics(data.metrics);
        updateMetricsHistory(data.metrics);
        calculateHealthScore(data.metrics);
        break;
      
      case 'service_health':
        updateServiceHealth(data.service, data.health);
        break;
      
      case 'alert':
        addAlert(data.alert);
        break;
      
      case 'user_analytics':
        setUserAnalytics(data.analytics);
        break;
      
      default:
        console.log('Unknown message type:', data.type);
    }
  };

  // Update metrics history for trending
  const updateMetricsHistory = (newMetrics: PlatformMetrics) => {
    setMetricsHistory(prev => {
      const updated = [...prev, newMetrics];
      // Keep last 100 data points
      return updated.slice(-100);
    });
  };

  // Update service health
  const updateServiceHealth = (serviceName: string, health: ServiceHealth) => {
    setServiceHealth(prev => {
      const updated = prev.filter(s => s.service_name !== serviceName);
      return [...updated, health];
    });
  };

  // Add new alert
  const addAlert = (alert: Alert) => {
    setAlerts(prev => [alert, ...prev.slice(0, 19)]); // Keep last 20 alerts
  };

  // Calculate overall health score
  const calculateHealthScore = (metrics: PlatformMetrics) => {
    try {
      const scores = [
        Math.max(0, 100 - metrics.api_performance.response_time / 20), // API response time
        Math.max(0, 100 - metrics.react_performance.page_load_time / 30), // Page load time
        Math.max(0, 100 - metrics.system_resources.cpu_usage), // CPU usage (inverted)
        Math.max(0, 100 - metrics.system_resources.memory_usage), // Memory usage (inverted)
        metrics.voice_quality_metrics.quality_scores * 100, // Voice quality
        metrics.database_metrics.cache_hit_ratio * 100 // DB cache hit ratio
      ];
      
      const avgScore = scores.reduce((sum, score) => sum + score, 0) / scores.length;
      setHealthScore(Math.round(avgScore));
    } catch (error) {
      console.error('Error calculating health score:', error);
    }
  };

  // Acknowledge alert
  const acknowledgeAlert = (alertIndex: number) => {
    setAlerts(prev => 
      prev.map((alert, index) => 
        index === alertIndex ? { ...alert, resolved: true } : alert
      )
    );
  };

  // Filter metrics by time range
  const getFilteredMetrics = () => {
    const now = new Date();
    const ranges = {
      '1h': 60 * 60 * 1000,
      '6h': 6 * 60 * 60 * 1000,
      '24h': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000
    };
    
    const cutoff = new Date(now.getTime() - ranges[selectedTimeRange]);
    
    return metricsHistory.filter(metric => 
      new Date(metric.timestamp) >= cutoff
    );
  };

  // Initialize WebSocket connection
  useEffect(() => {
    if (autoRefresh) {
      connectWebSocket();
    }
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connectWebSocket, autoRefresh]);

  // Auto-refresh interval
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      // Ping server for latest data if WebSocket is disconnected
      if (connectionStatus === 'disconnected') {
        connectWebSocket();
      }
    }, refreshInterval);
    
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, connectionStatus, connectWebSocket]);

  // Get status indicator color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#4CAF50';
      case 'degraded': return '#FF9800';
      case 'unhealthy': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  // Get health score color
  const getHealthScoreColor = (score: number) => {
    if (score >= 90) return '#4CAF50';
    if (score >= 70) return '#8BC34A';
    if (score >= 50) return '#FF9800';
    return '#F44336';
  };

  const filteredMetrics = getFilteredMetrics();

  return (
    <div className="production-monitoring-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-left">
          <h1>🎮 Dreamcast Platform Monitoring</h1>
          <div className={`connection-status ${connectionStatus}`}>
            <span className="status-indicator"></span>
            {connectionStatus === 'connected' ? 'Live' : 'Disconnected'}
          </div>
        </div>
        
        <div className="header-controls">
          <div className="time-range-selector">
            {(['1h', '6h', '24h', '7d'] as const).map(range => (
              <button
                key={range}
                className={selectedTimeRange === range ? 'active' : ''}
                onClick={() => setSelectedTimeRange(range)}
              >
                {range}
              </button>
            ))}
          </div>
          
          <div className="health-score">
            <div className="score-circle" style={{ color: getHealthScoreColor(healthScore) }}>
              {healthScore}
            </div>
            <span>Health Score</span>
          </div>
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="dashboard-grid">
        
        {/* Platform Status Overview */}
        <div className="dashboard-card platform-status">
          <h3>Platform Status</h3>
          <div className="status-grid">
            {serviceHealth.map(service => (
              <div key={service.service_name} className="service-status">
                <div 
                  className="status-dot" 
                  style={{ backgroundColor: getStatusColor(service.status) }}
                ></div>
                <div className="service-info">
                  <div className="service-name">{service.service_name}</div>
                  <div className="service-details">
                    {service.response_time.toFixed(0)}ms • {(service.error_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Real-time Metrics */}
        <div className="dashboard-card metrics-overview">
          <h3>Key Metrics</h3>
          {platformMetrics && (
            <div className="metrics-grid">
              <div className="metric-item">
                <div className="metric-value">{platformMetrics.api_performance.response_time.toFixed(0)}ms</div>
                <div className="metric-label">API Response Time</div>
              </div>
              <div className="metric-item">
                <div className="metric-value">{platformMetrics.react_performance.active_users}</div>
                <div className="metric-label">Active Users</div>
              </div>
              <div className="metric-item">
                <div className="metric-value">{platformMetrics.websocket_metrics.active_connections}</div>
                <div className="metric-label">WebSocket Connections</div>
              </div>
              <div className="metric-item">
                <div className="metric-value">{platformMetrics.voice_quality_metrics.generation_latency.toFixed(0)}ms</div>
                <div className="metric-label">Voice Generation</div>
              </div>
              <div className="metric-item">
                <div className="metric-value">{platformMetrics.system_resources.cpu_usage.toFixed(1)}%</div>
                <div className="metric-label">CPU Usage</div>
              </div>
              <div className="metric-item">
                <div className="metric-value">{platformMetrics.system_resources.memory_usage.toFixed(1)}%</div>
                <div className="metric-label">Memory Usage</div>
              </div>
            </div>
          )}
        </div>

        {/* Performance Charts */}
        <div className="dashboard-card performance-charts">
          <h3>Performance Trends</h3>
          <div className="charts-container">
            <div className="chart-section">
              <h4>API Response Time</h4>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={filteredMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleString()}
                    formatter={(value: number) => [`${value.toFixed(1)}ms`, 'Response Time']}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="api_performance.response_time" 
                    stroke="#2196F3" 
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="chart-section">
              <h4>System Resources</h4>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={filteredMetrics}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="timestamp" 
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  />
                  <YAxis />
                  <Tooltip 
                    labelFormatter={(value) => new Date(value).toLocaleString()}
                    formatter={(value: number) => [`${value.toFixed(1)}%`, '']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="system_resources.cpu_usage" 
                    stackId="1"
                    stroke="#FF5722" 
                    fill="#FF5722"
                    fillOpacity={0.6}
                    name="CPU"
                  />
                  <Area 
                    type="monotone" 
                    dataKey="system_resources.memory_usage" 
                    stackId="2"
                    stroke="#4CAF50" 
                    fill="#4CAF50"
                    fillOpacity={0.6}
                    name="Memory"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Voice Quality Metrics */}
        <div className="dashboard-card voice-metrics">
          <h3>Voice & Character Quality</h3>
          {platformMetrics && (
            <div className="voice-metrics-grid">
              <div className="metric-circle">
                <div className="circle-progress" style={{ 
                  background: `conic-gradient(#4CAF50 ${platformMetrics.voice_quality_metrics.quality_scores * 360}deg, #E0E0E0 0deg)` 
                }}>
                  <div className="circle-content">
                    <span className="circle-value">{(platformMetrics.voice_quality_metrics.quality_scores * 100).toFixed(0)}%</span>
                    <span className="circle-label">Quality</span>
                  </div>
                </div>
              </div>
              
              <div className="metric-circle">
                <div className="circle-progress" style={{ 
                  background: `conic-gradient(#2196F3 ${platformMetrics.voice_quality_metrics.character_consistency * 360}deg, #E0E0E0 0deg)` 
                }}>
                  <div className="circle-content">
                    <span className="circle-value">{(platformMetrics.voice_quality_metrics.character_consistency * 100).toFixed(0)}%</span>
                    <span className="circle-label">Consistency</span>
                  </div>
                </div>
              </div>
              
              <div className="voice-stats">
                <div className="stat-item">
                  <span className="stat-value">{platformMetrics.voice_quality_metrics.generation_latency.toFixed(0)}ms</span>
                  <span className="stat-label">Generation Latency</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">{platformMetrics.voice_quality_metrics.voice_generation_rate.toFixed(1)}/min</span>
                  <span className="stat-label">Generation Rate</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* User Analytics */}
        <div className="dashboard-card user-analytics">
          <h3>User Analytics</h3>
          {userAnalytics && (
            <div className="analytics-grid">
              <div className="analytics-section">
                <h4>Engagement</h4>
                <div className="analytics-metrics">
                  <div className="analytics-metric">
                    <span className="metric-value">{userAnalytics.user_engagement.active_users}</span>
                    <span className="metric-label">Active Users</span>
                  </div>
                  <div className="analytics-metric">
                    <span className="metric-value">{(userAnalytics.user_engagement.session_duration / 60).toFixed(1)}m</span>
                    <span className="metric-label">Avg Session</span>
                  </div>
                  <div className="analytics-metric">
                    <span className="metric-value">{(userAnalytics.user_engagement.bounce_rate * 100).toFixed(1)}%</span>
                    <span className="metric-label">Bounce Rate</span>
                  </div>
                </div>
              </div>
              
              <div className="analytics-section">
                <h4>Feature Adoption</h4>
                <ResponsiveContainer width="100%" height={120}>
                  <BarChart data={[
                    { name: 'Voice Features', value: userAnalytics.feature_adoption.voice_feature_usage * 100 },
                    { name: 'Character Creation', value: userAnalytics.platform_performance.character_creation_success * 100 },
                    { name: 'Journey Completion', value: userAnalytics.feature_adoption.journey_completion * 100 }
                  ]}>
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                    <YAxis />
                    <Tooltip formatter={(value: number) => [`${value.toFixed(1)}%`, 'Adoption']} />
                    <Bar dataKey="value" fill="#9C27B0" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>

        {/* Alerts Panel */}
        <div className="dashboard-card alerts-panel">
          <h3>Recent Alerts</h3>
          <div className="alerts-list">
            {alerts.length === 0 ? (
              <div className="no-alerts">✅ No active alerts</div>
            ) : (
              alerts.slice(0, 10).map((alert, index) => (
                <div 
                  key={index} 
                  className={`alert-item ${alert.level} ${alert.resolved ? 'resolved' : ''}`}
                >
                  <div className="alert-content">
                    <div className="alert-message">{alert.message}</div>
                    <div className="alert-details">
                      {alert.service && <span className="alert-service">{alert.service}</span>}
                      <span className="alert-time">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                  {!alert.resolved && (
                    <button 
                      className="alert-acknowledge"
                      onClick={() => acknowledgeAlert(index)}
                    >
                      ✓
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Database Performance */}
        <div className="dashboard-card database-performance">
          <h3>Database Performance</h3>
          {platformMetrics && (
            <div className="db-metrics">
              <div className="db-chart">
                <ResponsiveContainer width="100%" height={150}>
                  <LineChart data={filteredMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                      formatter={(value: number) => [`${value.toFixed(1)}ms`, 'Query Time']}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="database_metrics.query_time" 
                      stroke="#FF9800" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              <div className="db-stats">
                <div className="db-stat">
                  <span className="stat-value">{platformMetrics.database_metrics.query_time.toFixed(1)}ms</span>
                  <span className="stat-label">Avg Query Time</span>
                </div>
                <div className="db-stat">
                  <span className="stat-value">{platformMetrics.database_metrics.connections_active}</span>
                  <span className="stat-label">Active Connections</span>
                </div>
                <div className="db-stat">
                  <span className="stat-value">{(platformMetrics.database_metrics.cache_hit_ratio * 100).toFixed(1)}%</span>
                  <span className="stat-label">Cache Hit Rate</span>
                </div>
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default ProductionMonitoringDashboard; 