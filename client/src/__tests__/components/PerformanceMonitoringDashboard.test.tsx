import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import PerformanceMonitoringDashboard from '../../components/PerformanceMonitoringDashboard';

// Mock WebSocket
const mockWebSocket = {
  close: jest.fn(),
  send: jest.fn(),
  readyState: 1, // OPEN
};

global.WebSocket = jest.fn(() => mockWebSocket) as any;

// Mock fetch
global.fetch = jest.fn();

// Mock chart library
jest.mock('recharts', () => ({
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  AreaChart: ({ children }: any) => <div data-testid="area-chart">{children}</div>,
  BarChart: ({ children }: any) => <div data-testid="bar-chart">{children}</div>,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Line: () => <div data-testid="line" />,
  Area: () => <div data-testid="area" />,
  Bar: () => <div data-testid="bar" />,
  ResponsiveContainer: ({ children }: any) => <div data-testid="responsive-container">{children}</div>,
}));

describe('PerformanceMonitoringDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        system_metrics: {
          cpu_usage: 45.2,
          memory_usage: 62.1,
          disk_usage: 78.3,
          network_io: { in: 1024, out: 512 }
        },
        job_metrics: {
          active_jobs: 3,
          queue_length: 7,
          avg_processing_time: 180,
          error_rate: 0.02
        },
        performance_trends: [
          { timestamp: '2024-01-20T10:00:00Z', cpu: 40, memory: 55, response_time: 150 },
          { timestamp: '2024-01-20T10:05:00Z', cpu: 45, memory: 60, response_time: 170 },
          { timestamp: '2024-01-20T10:10:00Z', cpu: 50, memory: 65, response_time: 200 }
        ],
        bottlenecks: [
          {
            type: 'cpu_usage',
            severity: 'medium',
            description: 'CPU usage approaching threshold',
            suggestions: ['Scale horizontally', 'Optimize processing algorithms']
          }
        ]
      })
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Component Rendering', () => {
    test('renders performance monitoring dashboard', () => {
      render(<PerformanceMonitoringDashboard />);
      
      expect(screen.getByText('Performance Monitoring')).toBeInTheDocument();
      expect(screen.getByText('System Metrics')).toBeInTheDocument();
      expect(screen.getByText('Job Performance')).toBeInTheDocument();
      expect(screen.getByText('Performance Trends')).toBeInTheDocument();
      expect(screen.getByText('Bottleneck Detection')).toBeInTheDocument();
    });

    test('displays system metrics cards', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('CPU Usage')).toBeInTheDocument();
        expect(screen.getByText('45.2%')).toBeInTheDocument();
        expect(screen.getByText('Memory Usage')).toBeInTheDocument();
        expect(screen.getByText('62.1%')).toBeInTheDocument();
        expect(screen.getByText('Disk Usage')).toBeInTheDocument();
        expect(screen.getByText('78.3%')).toBeInTheDocument();
      });
    });

    test('displays job performance metrics', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Active Jobs')).toBeInTheDocument();
        expect(screen.getByText('3')).toBeInTheDocument();
        expect(screen.getByText('Queue Length')).toBeInTheDocument();
        expect(screen.getByText('7')).toBeInTheDocument();
        expect(screen.getByText('Avg Processing Time')).toBeInTheDocument();
        expect(screen.getByText('180s')).toBeInTheDocument();
        expect(screen.getByText('Error Rate')).toBeInTheDocument();
        expect(screen.getByText('2.0%')).toBeInTheDocument();
      });
    });

    test('renders performance trend charts', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Real-time Updates', () => {
    test('establishes WebSocket connection for real-time updates', () => {
      render(<PerformanceMonitoringDashboard />);
      
      expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost:8000/ws/performance-monitoring');
    });

    test('handles WebSocket metric updates', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      // Simulate WebSocket message
      const mockEvent = {
        data: JSON.stringify({
          type: 'metrics_update',
          data: {
            system_metrics: {
              cpu_usage: 55.0,
              memory_usage: 70.5,
              disk_usage: 80.1,
              network_io: { in: 2048, out: 1024 }
            }
          }
        })
      };

      if (mockWebSocket.onmessage) {
        mockWebSocket.onmessage(mockEvent);
      }

      await waitFor(() => {
        expect(screen.getByText('55.0%')).toBeInTheDocument();
        expect(screen.getByText('70.5%')).toBeInTheDocument();
        expect(screen.getByText('80.1%')).toBeInTheDocument();
      });
    });

    test('handles bottleneck detection updates', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      // Simulate WebSocket bottleneck alert
      const mockEvent = {
        data: JSON.stringify({
          type: 'bottleneck_detected',
          data: {
            type: 'memory_usage',
            severity: 'high',
            description: 'Memory usage critical',
            suggestions: ['Increase memory allocation', 'Restart services']
          }
        })
      };

      if (mockWebSocket.onmessage) {
        mockWebSocket.onmessage(mockEvent);
      }

      await waitFor(() => {
        expect(screen.getByText('Memory usage critical')).toBeInTheDocument();
        expect(screen.getByText('Increase memory allocation')).toBeInTheDocument();
      });
    });

    test('updates performance trends in real-time', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      // Simulate trend update
      const mockEvent = {
        data: JSON.stringify({
          type: 'trend_update',
          data: {
            timestamp: '2024-01-20T10:15:00Z',
            cpu: 60,
            memory: 75,
            response_time: 250
          }
        })
      };

      if (mockWebSocket.onmessage) {
        mockWebSocket.onmessage(mockEvent);
      }

      // Verify trend data is updated (chart should re-render)
      await waitFor(() => {
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Bottleneck Detection', () => {
    test('displays bottleneck alerts with severity indicators', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('CPU usage approaching threshold')).toBeInTheDocument();
        expect(screen.getByText('medium')).toBeInTheDocument();
        expect(screen.getByText('Scale horizontally')).toBeInTheDocument();
        expect(screen.getByText('Optimize processing algorithms')).toBeInTheDocument();
      });
    });

    test('allows filtering bottlenecks by severity', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const severityFilter = await screen.findByRole('combobox', { name: /severity filter/i });
      await userEvent.selectOptions(severityFilter, 'high');
      
      // Should filter out medium severity bottlenecks
      await waitFor(() => {
        expect(screen.queryByText('CPU usage approaching threshold')).not.toBeInTheDocument();
      });
    });

    test('provides bottleneck resolution actions', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const resolveButton = await screen.findByRole('button', { name: /resolve bottleneck/i });
      await userEvent.click(resolveButton);
      
      expect(global.fetch).toHaveBeenCalledWith('/api/performance/bottlenecks/resolve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bottleneck_id: expect.any(String) })
      });
    });
  });

  describe('Optimization Suggestions', () => {
    test('displays performance optimization recommendations', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          suggestions: [
            {
              type: 'scaling',
              priority: 'high',
              description: 'Consider horizontal scaling',
              impact: 'Reduce response time by 30%',
              effort: 'medium'
            },
            {
              type: 'optimization',
              priority: 'medium',
              description: 'Optimize database queries',
              impact: 'Reduce CPU usage by 15%',
              effort: 'high'
            }
          ]
        })
      });

      render(<PerformanceMonitoringDashboard />);
      
      const suggestionsButton = await screen.findByRole('button', { name: /get optimization suggestions/i });
      await userEvent.click(suggestionsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Consider horizontal scaling')).toBeInTheDocument();
        expect(screen.getByText('Reduce response time by 30%')).toBeInTheDocument();
        expect(screen.getByText('Optimize database queries')).toBeInTheDocument();
        expect(screen.getByText('Reduce CPU usage by 15%')).toBeInTheDocument();
      });
    });

    test('allows applying optimization suggestions', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      // First get suggestions
      const suggestionsButton = await screen.findByRole('button', { name: /get optimization suggestions/i });
      await userEvent.click(suggestionsButton);
      
      await waitFor(() => {
        expect(screen.getByText('Consider horizontal scaling')).toBeInTheDocument();
      });
      
      const applyButton = await screen.findByRole('button', { name: /apply suggestion/i });
      await userEvent.click(applyButton);
      
      expect(global.fetch).toHaveBeenCalledWith('/api/performance/optimizations/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ suggestion_id: expect.any(String) })
      });
    });
  });

  describe('Time Range Controls', () => {
    test('allows changing time range for metrics', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const timeRangeSelect = await screen.findByRole('combobox', { name: /time range/i });
      await userEvent.selectOptions(timeRangeSelect, '1h');
      
      expect(global.fetch).toHaveBeenCalledWith('/api/performance/metrics?time_range=1h');
    });

    test('updates charts when time range changes', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const timeRangeSelect = await screen.findByRole('combobox', { name: /time range/i });
      await userEvent.selectOptions(timeRangeSelect, '24h');
      
      await waitFor(() => {
        // Charts should re-render with new data
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
        expect(screen.getByTestId('area-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Resource Usage Analysis', () => {
    test('displays detailed resource breakdown', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const resourceTab = await screen.findByRole('tab', { name: /resource analysis/i });
      await userEvent.click(resourceTab);
      
      await waitFor(() => {
        expect(screen.getByText('CPU Breakdown')).toBeInTheDocument();
        expect(screen.getByText('Memory Distribution')).toBeInTheDocument();
        expect(screen.getByText('Disk I/O Analysis')).toBeInTheDocument();
        expect(screen.getByText('Network Traffic')).toBeInTheDocument();
      });
    });

    test('shows resource usage by service', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          services: [
            { name: 'multimodal-studio', cpu: 25.3, memory: 512, disk_io: 150 },
            { name: 'inference-engine', cpu: 35.7, memory: 1024, disk_io: 200 },
            { name: 'dataset-processor', cpu: 15.2, memory: 256, disk_io: 100 }
          ]
        })
      });

      render(<PerformanceMonitoringDashboard />);
      
      const serviceBreakdownButton = await screen.findByRole('button', { name: /service breakdown/i });
      await userEvent.click(serviceBreakdownButton);
      
      await waitFor(() => {
        expect(screen.getByText('multimodal-studio')).toBeInTheDocument();
        expect(screen.getByText('25.3%')).toBeInTheDocument();
        expect(screen.getByText('inference-engine')).toBeInTheDocument();
        expect(screen.getByText('35.7%')).toBeInTheDocument();
      });
    });
  });

  describe('Performance Alerts', () => {
    test('displays active performance alerts', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          alerts: [
            {
              id: 'alert-1',
              type: 'cpu_threshold',
              severity: 'warning',
              message: 'CPU usage above 80%',
              timestamp: '2024-01-20T10:15:00Z'
            },
            {
              id: 'alert-2',
              type: 'memory_threshold',
              severity: 'critical',
              message: 'Memory usage above 90%',
              timestamp: '2024-01-20T10:20:00Z'
            }
          ]
        })
      });

      render(<PerformanceMonitoringDashboard />);
      
      const alertsTab = await screen.findByRole('tab', { name: /alerts/i });
      await userEvent.click(alertsTab);
      
      await waitFor(() => {
        expect(screen.getByText('CPU usage above 80%')).toBeInTheDocument();
        expect(screen.getByText('Memory usage above 90%')).toBeInTheDocument();
        expect(screen.getByText('warning')).toBeInTheDocument();
        expect(screen.getByText('critical')).toBeInTheDocument();
      });
    });

    test('allows acknowledging alerts', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const alertsTab = await screen.findByRole('tab', { name: /alerts/i });
      await userEvent.click(alertsTab);
      
      await waitFor(() => {
        expect(screen.getByText('CPU usage above 80%')).toBeInTheDocument();
      });
      
      const acknowledgeButton = await screen.findByRole('button', { name: /acknowledge/i });
      await userEvent.click(acknowledgeButton);
      
      expect(global.fetch).toHaveBeenCalledWith('/api/performance/alerts/acknowledge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alert_id: 'alert-1' })
      });
    });
  });

  describe('Performance History', () => {
    test('displays historical performance data', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const historyTab = await screen.findByRole('tab', { name: /history/i });
      await userEvent.click(historyTab);
      
      await waitFor(() => {
        expect(screen.getByText('Performance History')).toBeInTheDocument();
        expect(screen.getByTestId('line-chart')).toBeInTheDocument();
      });
    });

    test('allows exporting performance reports', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const exportButton = await screen.findByRole('button', { name: /export report/i });
      await userEvent.click(exportButton);
      
      expect(global.fetch).toHaveBeenCalledWith('/api/performance/reports/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: 'pdf', time_range: '24h' })
      });
    });
  });

  describe('Auto-refresh Controls', () => {
    test('provides auto-refresh toggle', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const autoRefreshToggle = await screen.findByRole('checkbox', { name: /auto refresh/i });
      expect(autoRefreshToggle).toBeInTheDocument();
      expect(autoRefreshToggle).toBeChecked(); // Should be enabled by default
    });

    test('allows changing refresh interval', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const refreshIntervalSelect = await screen.findByRole('combobox', { name: /refresh interval/i });
      await userEvent.selectOptions(refreshIntervalSelect, '10s');
      
      // Should update the interval (tested by checking if the component re-renders)
      expect(refreshIntervalSelect).toHaveValue('10s');
    });

    test('manual refresh button works', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const refreshButton = await screen.findByRole('button', { name: /refresh now/i });
      await userEvent.click(refreshButton);
      
      // Should trigger a new fetch
      expect(global.fetch).toHaveBeenCalledWith('/api/performance/metrics?time_range=5m');
    });
  });

  describe('Error Handling', () => {
    test('handles API errors gracefully', async () => {
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('API Error'));
      
      render(<PerformanceMonitoringDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText(/error loading performance metrics/i)).toBeInTheDocument();
      });
    });

    test('handles WebSocket connection errors', async () => {
      // Mock WebSocket error
      const mockWebSocketError = {
        ...mockWebSocket,
        readyState: 3 // CLOSED
      };
      
      (global.WebSocket as jest.Mock).mockImplementationOnce(() => mockWebSocketError);
      
      render(<PerformanceMonitoringDashboard />);
      
      // Simulate connection error
      if (mockWebSocketError.onerror) {
        mockWebSocketError.onerror(new Event('error'));
      }
      
      await waitFor(() => {
        expect(screen.getByText(/real-time updates unavailable/i)).toBeInTheDocument();
      });
    });

    test('shows loading states during data fetch', () => {
      render(<PerformanceMonitoringDashboard />);
      
      expect(screen.getByText(/loading performance metrics/i)).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    test('has proper ARIA labels', () => {
      render(<PerformanceMonitoringDashboard />);
      
      expect(screen.getByLabelText('Performance monitoring dashboard')).toBeInTheDocument();
      expect(screen.getByLabelText('System metrics overview')).toBeInTheDocument();
      expect(screen.getByLabelText('Job performance metrics')).toBeInTheDocument();
    });

    test('supports keyboard navigation', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      const timeRangeSelect = await screen.findByRole('combobox', { name: /time range/i });
      timeRangeSelect.focus();
      
      fireEvent.keyDown(timeRangeSelect, { key: 'ArrowDown' });
      fireEvent.keyDown(timeRangeSelect, { key: 'Enter' });
      
      // Should be able to navigate and select options
      expect(timeRangeSelect).toHaveFocus();
    });

    test('announces important updates to screen readers', async () => {
      render(<PerformanceMonitoringDashboard />);
      
      // Simulate bottleneck detection
      const mockEvent = {
        data: JSON.stringify({
          type: 'bottleneck_detected',
          data: {
            type: 'cpu_usage',
            severity: 'critical',
            description: 'Critical CPU usage detected'
          }
        })
      };

      if (mockWebSocket.onmessage) {
        mockWebSocket.onmessage(mockEvent);
      }

      await waitFor(() => {
        expect(screen.getByRole('alert')).toBeInTheDocument();
        expect(screen.getByText('Critical CPU usage detected')).toBeInTheDocument();
      });
    });
  });
}); 