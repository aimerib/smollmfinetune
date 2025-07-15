/**
 * Tests for JobManagementDashboard Component - R6-9 Production Features
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { jest } from '@jest/globals';
import JobManagementDashboard from '../../components/JobManagementDashboard';

// Mock WebSocket
global.WebSocket = jest.fn(() => ({
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  readyState: 1, // OPEN
  onopen: null,
  onmessage: null,
  onclose: null,
  onerror: null,
})) as any;

// Mock console methods to reduce test noise
const originalConsole = { ...console };
beforeAll(() => {
  console.log = jest.fn();
  console.error = jest.fn();
});

afterAll(() => {
  Object.assign(console, originalConsole);
});

// Mock fetch for API calls
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ success: true }),
  })
) as jest.Mock;

// Test data
const mockJobs = [
  {
    id: 'job1',
    name: 'Test Dataset 1',
    status: 'generating',
    progress: 45,
    currentStep: 'Generating text samples',
    samplesGenerated: 450,
    estimatedTime: '15 minutes',
    createdAt: '2025-01-20T10:00:00Z',
    config: {
      sampleCount: 1000,
      characterCount: 5,
      narrativeTypes: ['dialogue', 'monologue'],
      useMockTTS: true,
      ttsProvider: 'orpheus',
      outputDir: 'test_dataset_1',
      batchSize: 10,
      temperature: 0.8
    },
    priority: 'high',
    dependencies: []
  },
  {
    id: 'job2',
    name: 'Test Dataset 2',
    status: 'queued',
    progress: 0,
    currentStep: 'Waiting in queue',
    samplesGenerated: 0,
    estimatedTime: '20 minutes',
    createdAt: '2025-01-20T10:15:00Z',
    config: {
      sampleCount: 2000,
      characterCount: 8,
      narrativeTypes: ['dialogue', 'action_scene'],
      useMockTTS: false,
      ttsProvider: 'xtts',
      outputDir: 'test_dataset_2',
      batchSize: 15,
      temperature: 0.7
    },
    priority: 'normal',
    dependencies: ['job1']
  },
  {
    id: 'job3',
    name: 'Test Dataset 3',
    status: 'completed',
    progress: 100,
    currentStep: 'Completed successfully',
    samplesGenerated: 1500,
    estimatedTime: null,
    createdAt: '2025-01-20T09:30:00Z',
    config: {
      sampleCount: 1500,
      characterCount: 6,
      narrativeTypes: ['dialogue', 'emotional_moment'],
      useMockTTS: true,
      ttsProvider: 'kokoro',
      outputDir: 'test_dataset_3',
      batchSize: 12,
      temperature: 0.9
    },
    priority: 'normal',
    dependencies: []
  }
];

const mockJobQueue = {
  active: ['job1'],
  waiting: ['job2'],
  completed: ['job3'],
  failed: [],
  totalJobs: 3,
  processingCapacity: 2,
  estimatedWaitTime: '5 minutes'
};

const mockJobMetrics = {
  totalJobs: 3,
  activeJobs: 1,
  completedJobs: 1,
  failedJobs: 0,
  averageProcessingTime: 1800, // 30 minutes
  systemLoad: 0.65,
  memoryUsage: 0.72,
  diskUsage: 0.45,
  networkThroughput: 125.5,
  recentErrors: [],
  performanceScore: 0.85
};

describe('JobManagementDashboard', () => {
  let mockWebSocket: any;

  beforeEach(() => {
    mockWebSocket = new WebSocket('ws://localhost:8000/ws/multimodal-jobs');
    (global.fetch as jest.Mock).mockClear();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Initial Render', () => {
    test('renders dashboard with all sections', () => {
      render(<JobManagementDashboard />);
      
      // Check main dashboard components
      expect(screen.getByText('Job Management Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Queue Visualizer')).toBeInTheDocument();
      expect(screen.getByText('Active Jobs')).toBeInTheDocument();
      expect(screen.getByText('Batch Operations')).toBeInTheDocument();
      expect(screen.getByText('System Metrics')).toBeInTheDocument();
    });

    test('loads jobs on mount', async () => {
      render(<JobManagementDashboard />);
      
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/jobs');
      });
    });

    test('establishes WebSocket connection', () => {
      render(<JobManagementDashboard />);
      
      expect(WebSocket).toHaveBeenCalledWith('ws://localhost:8000/ws/multimodal-jobs');
    });
  });

  describe('Job Queue Visualizer', () => {
    test('displays queue statistics', async () => {
      render(<JobManagementDashboard />);
      
      // Mock WebSocket message for queue update
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'queue_update',
            queue: mockJobQueue
          })
        });
      });

      await waitFor(() => {
        expect(screen.getByText('Active: 1')).toBeInTheDocument();
        expect(screen.getByText('Waiting: 1')).toBeInTheDocument();
        expect(screen.getByText('Completed: 1')).toBeInTheDocument();
        expect(screen.getByText('Est. Wait: 5 minutes')).toBeInTheDocument();
      });
    });

    test('allows job reordering', async () => {
      render(<JobManagementDashboard />);
      
      // Mock jobs in queue
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        const dragHandle = screen.getByTestId('drag-handle-job2');
        expect(dragHandle).toBeInTheDocument();
      });

      // Simulate drag and drop reorder
      const dragHandle = screen.getByTestId('drag-handle-job2');
      fireEvent.dragStart(dragHandle);
      fireEvent.dragOver(screen.getByTestId('job-row-job1'));
      fireEvent.drop(screen.getByTestId('job-row-job1'));

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/jobs/reorder', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jobId: 'job2',
            newPosition: 0
          })
        });
      });
    });

    test('allows priority changes', async () => {
      render(<JobManagementDashboard />);
      
      // Mock jobs loaded
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        const prioritySelect = screen.getByTestId('priority-select-job2');
        expect(prioritySelect).toBeInTheDocument();
      });

      // Change priority
      const prioritySelect = screen.getByTestId('priority-select-job2');
      await userEvent.selectOptions(prioritySelect, 'high');

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/jobs/job2/priority', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ priority: 'high' })
        });
      });
    });
  });

  describe('Job Table with Batch Operations', () => {
    test('displays job table with selection', async () => {
      render(<JobManagementDashboard />);
      
      // Mock jobs loaded
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        expect(screen.getByText('Test Dataset 1')).toBeInTheDocument();
        expect(screen.getByText('Test Dataset 2')).toBeInTheDocument();
        expect(screen.getByText('Test Dataset 3')).toBeInTheDocument();
      });

      // Check selection checkboxes
      const checkboxes = screen.getAllByRole('checkbox');
      expect(checkboxes).toHaveLength(4); // 3 job checkboxes + 1 select-all
    });

    test('handles select all functionality', async () => {
      render(<JobManagementDashboard />);
      
      // Mock jobs loaded
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        const selectAllCheckbox = screen.getByTestId('select-all-jobs');
        expect(selectAllCheckbox).toBeInTheDocument();
      });

      // Click select all
      const selectAllCheckbox = screen.getByTestId('select-all-jobs');
      await userEvent.click(selectAllCheckbox);

      // Check that all job checkboxes are selected
      const jobCheckboxes = screen.getAllByTestId(/job-checkbox-/);
      jobCheckboxes.forEach(checkbox => {
        expect(checkbox).toBeChecked();
      });
    });

    test('handles individual job selection', async () => {
      render(<JobManagementDashboard />);
      
      // Mock jobs loaded
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        const jobCheckbox = screen.getByTestId('job-checkbox-job1');
        expect(jobCheckbox).toBeInTheDocument();
      });

      // Select individual job
      const jobCheckbox = screen.getByTestId('job-checkbox-job1');
      await userEvent.click(jobCheckbox);

      expect(jobCheckbox).toBeChecked();

      // Check that batch operations become available
      expect(screen.getByText('1 job selected')).toBeInTheDocument();
    });

    test('handles batch pause operation', async () => {
      render(<JobManagementDashboard />);
      
      // Mock jobs loaded and select multiple
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        const job1Checkbox = screen.getByTestId('job-checkbox-job1');
        const job2Checkbox = screen.getByTestId('job-checkbox-job2');
        expect(job1Checkbox).toBeInTheDocument();
        expect(job2Checkbox).toBeInTheDocument();
      });

      // Select multiple jobs
      await userEvent.click(screen.getByTestId('job-checkbox-job1'));
      await userEvent.click(screen.getByTestId('job-checkbox-job2'));

      // Click batch pause
      const pauseButton = screen.getByText('Pause Selected');
      await userEvent.click(pauseButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/jobs/batch-action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'pause',
            job_ids: ['job1', 'job2']
          })
        });
      });
    });

    test('handles batch resume operation', async () => {
      render(<JobManagementDashboard />);
      
      // Mock jobs loaded and select multiple
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        const job1Checkbox = screen.getByTestId('job-checkbox-job1');
        expect(job1Checkbox).toBeInTheDocument();
      });

      // Select job
      await userEvent.click(screen.getByTestId('job-checkbox-job1'));

      // Click batch resume
      const resumeButton = screen.getByText('Resume Selected');
      await userEvent.click(resumeButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/jobs/batch-action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'resume',
            job_ids: ['job1']
          })
        });
      });
    });

    test('handles batch cancel operation', async () => {
      render(<JobManagementDashboard />);
      
      // Mock jobs loaded and select multiple
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        const job1Checkbox = screen.getByTestId('job-checkbox-job1');
        expect(job1Checkbox).toBeInTheDocument();
      });

      // Select job
      await userEvent.click(screen.getByTestId('job-checkbox-job1'));

      // Click batch cancel
      const cancelButton = screen.getByText('Cancel Selected');
      await userEvent.click(cancelButton);

      // Confirm cancellation
      await waitFor(() => {
        expect(screen.getByText('Confirm Cancellation')).toBeInTheDocument();
      });

      const confirmButton = screen.getByText('Confirm Cancel');
      await userEvent.click(confirmButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/jobs/batch-action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'cancel',
            job_ids: ['job1']
          })
        });
      });
    });

    test('handles batch delete operation', async () => {
      render(<JobManagementDashboard />);
      
      // Mock jobs loaded and select completed job
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        const job3Checkbox = screen.getByTestId('job-checkbox-job3');
        expect(job3Checkbox).toBeInTheDocument();
      });

      // Select completed job
      await userEvent.click(screen.getByTestId('job-checkbox-job3'));

      // Click batch delete
      const deleteButton = screen.getByText('Delete Selected');
      await userEvent.click(deleteButton);

      // Confirm deletion
      await waitFor(() => {
        expect(screen.getByText('Confirm Deletion')).toBeInTheDocument();
      });

      const confirmButton = screen.getByText('Confirm Delete');
      await userEvent.click(confirmButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/jobs/batch-action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'delete',
            job_ids: ['job3']
          })
        });
      });
    });
  });

  describe('Job Metrics Panel', () => {
    test('displays system metrics', async () => {
      render(<JobManagementDashboard />);
      
      // Mock metrics update
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'job_metrics',
            metrics: mockJobMetrics
          })
        });
      });

      await waitFor(() => {
        expect(screen.getByText('System Load: 65%')).toBeInTheDocument();
        expect(screen.getByText('Memory Usage: 72%')).toBeInTheDocument();
        expect(screen.getByText('Disk Usage: 45%')).toBeInTheDocument();
        expect(screen.getByText('Performance Score: 85%')).toBeInTheDocument();
      });
    });

    test('shows performance trends', async () => {
      render(<JobManagementDashboard />);
      
      // Mock metrics update
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'job_metrics',
            metrics: mockJobMetrics
          })
        });
      });

      await waitFor(() => {
        expect(screen.getByText('Average Processing Time: 30 minutes')).toBeInTheDocument();
        expect(screen.getByText('Network Throughput: 125.5 MB/s')).toBeInTheDocument();
      });
    });

    test('handles metric drill-down', async () => {
      render(<JobManagementDashboard />);
      
      // Mock metrics update
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'job_metrics',
            metrics: mockJobMetrics
          })
        });
      });

      await waitFor(() => {
        const systemLoadMetric = screen.getByTestId('metric-system-load');
        expect(systemLoadMetric).toBeInTheDocument();
      });

      // Click on metric for drill-down
      const systemLoadMetric = screen.getByTestId('metric-system-load');
      await userEvent.click(systemLoadMetric);

      await waitFor(() => {
        expect(screen.getByText('System Load Details')).toBeInTheDocument();
        expect(screen.getByText('CPU Usage')).toBeInTheDocument();
        expect(screen.getByText('Memory Details')).toBeInTheDocument();
      });
    });
  });

  describe('Real-time Updates', () => {
    test('handles job status updates', async () => {
      render(<JobManagementDashboard />);
      
      // Mock initial jobs
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        expect(screen.getByText('generating')).toBeInTheDocument();
      });

      // Mock job status update
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'job_status_update',
            job_id: 'job1',
            status: 'completed'
          })
        });
      });

      await waitFor(() => {
        expect(screen.getByText('completed')).toBeInTheDocument();
      });
    });

    test('handles job progress updates', async () => {
      render(<JobManagementDashboard />);
      
      // Mock initial jobs
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        expect(screen.getByText('45%')).toBeInTheDocument();
      });

      // Mock progress update
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'job_progress',
            job_id: 'job1',
            progress: 75
          })
        });
      });

      await waitFor(() => {
        expect(screen.getByText('75%')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    test('handles API errors gracefully', async () => {
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('API Error'));
      
      render(<JobManagementDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Failed to load jobs')).toBeInTheDocument();
      });
    });

    test('handles WebSocket connection errors', async () => {
      render(<JobManagementDashboard />);
      
      // Mock WebSocket error
      act(() => {
        mockWebSocket.onerror({ message: 'Connection failed' });
      });

      await waitFor(() => {
        expect(screen.getByText('Connection lost - attempting to reconnect')).toBeInTheDocument();
      });
    });

    test('handles batch operation errors', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve({ jobs: mockJobs }) })
        .mockRejectedValueOnce(new Error('Batch operation failed'));
      
      render(<JobManagementDashboard />);
      
      await waitFor(() => {
        const job1Checkbox = screen.getByTestId('job-checkbox-job1');
        expect(job1Checkbox).toBeInTheDocument();
      });

      // Select job and try batch operation
      await userEvent.click(screen.getByTestId('job-checkbox-job1'));
      await userEvent.click(screen.getByText('Pause Selected'));

      await waitFor(() => {
        expect(screen.getByText('Batch operation failed')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    test('has proper ARIA labels', async () => {
      render(<JobManagementDashboard />);
      
      expect(screen.getByLabelText('Select all jobs')).toBeInTheDocument();
      expect(screen.getByLabelText('Job management dashboard')).toBeInTheDocument();
      expect(screen.getByLabelText('Queue visualizer')).toBeInTheDocument();
      expect(screen.getByLabelText('System metrics panel')).toBeInTheDocument();
    });

    test('supports keyboard navigation', async () => {
      render(<JobManagementDashboard />);
      
      // Mock jobs loaded
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        const selectAllCheckbox = screen.getByTestId('select-all-jobs');
        expect(selectAllCheckbox).toBeInTheDocument();
      });

      // Navigate using Tab key
      await userEvent.tab();
      expect(screen.getByTestId('select-all-jobs')).toHaveFocus();
      
      await userEvent.tab();
      expect(screen.getByTestId('job-checkbox-job1')).toHaveFocus();
    });

    test('has proper focus management', async () => {
      render(<JobManagementDashboard />);
      
      // Mock jobs loaded
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: mockJobs
          })
        });
      });

      await waitFor(() => {
        const pauseButton = screen.getByText('Pause Selected');
        expect(pauseButton).toBeInTheDocument();
      });

      // Focus should be managed properly in batch operations
      await userEvent.click(screen.getByTestId('job-checkbox-job1'));
      
      const pauseButton = screen.getByText('Pause Selected');
      expect(pauseButton).not.toHaveAttribute('disabled');
    });
  });

  describe('Performance', () => {
    test('handles large job lists efficiently', async () => {
      const largeJobList = Array.from({ length: 1000 }, (_, i) => ({
        ...mockJobs[0],
        id: `job${i}`,
        name: `Test Dataset ${i}`
      }));

      render(<JobManagementDashboard />);
      
      // Mock large job list
      act(() => {
        mockWebSocket.onmessage({ 
          data: JSON.stringify({
            type: 'jobs_update',
            jobs: largeJobList
          })
        });
      });

      // Should render without performance issues
      await waitFor(() => {
        expect(screen.getByText('Test Dataset 0')).toBeInTheDocument();
      });
    });

    test('debounces WebSocket updates', async () => {
      render(<JobManagementDashboard />);
      
      // Send multiple rapid updates
      for (let i = 0; i < 10; i++) {
        act(() => {
          mockWebSocket.onmessage({ 
            data: JSON.stringify({
              type: 'job_progress',
              job_id: 'job1',
              progress: i * 10
            })
          });
        });
      }

      // Should only apply the last update
      await waitFor(() => {
        expect(screen.getByText('90%')).toBeInTheDocument();
      });
    });
  });
}); 