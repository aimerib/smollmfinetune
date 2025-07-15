/**
 * Tests for DatasetExportManager Component - R6-9 Production Features
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { jest } from '@jest/globals';
import DatasetExportManager from '../../components/DatasetExportManager';

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

// Mock URL.createObjectURL and URL.revokeObjectURL
global.URL.createObjectURL = jest.fn(() => 'mock-blob-url');
global.URL.revokeObjectURL = jest.fn();

// Mock for downloading files
const mockDownloadFile = jest.fn();
Object.defineProperty(global, 'document', {
  value: {
    createElement: jest.fn(() => ({
      href: '',
      download: '',
      click: jest.fn(),
    })),
  },
  writable: true,
});

// Test data
const mockExportConfigs = [
  {
    id: 'config-001',
    name: 'HuggingFace Standard',
    format: 'huggingface',
    created_at: '2025-01-20T10:00:00Z',
    options: {
      include_metadata: true,
      split_ratio: { train: 0.8, validation: 0.1, test: 0.1 },
      tokenizer_config: {
        max_length: 512,
        padding: 'max_length',
        truncation: true
      },
      dataset_features: {
        text: 'string',
        audio: 'audio',
        labels: 'classification_label'
      }
    }
  },
  {
    id: 'config-002',
    name: 'JSONL Export',
    format: 'jsonl',
    created_at: '2025-01-20T09:30:00Z',
    options: {
      include_audio: true,
      compression: 'gzip',
      chunk_size: 1000,
      encoding: 'utf-8'
    }
  },
  {
    id: 'config-003',
    name: 'PyTorch Dataset',
    format: 'pytorch',
    created_at: '2025-01-20T09:00:00Z',
    options: {
      tensor_format: 'pt',
      batch_size: 32,
      num_workers: 4,
      pin_memory: true,
      shuffle: true
    }
  }
];

const mockExportHistory = [
  {
    id: 'export-001',
    dataset_id: 'dataset-123',
    dataset_name: 'Test Dataset',
    config_id: 'config-001',
    format: 'huggingface',
    status: 'completed',
    created_at: '2025-01-20T10:00:00Z',
    completed_at: '2025-01-20T10:15:00Z',
    file_size: 1024000,
    file_path: '/exports/test-dataset-hf.tar.gz',
    progress: 100,
    error_message: null
  },
  {
    id: 'export-002',
    dataset_id: 'dataset-456',
    dataset_name: 'Another Dataset',
    config_id: 'config-002',
    format: 'jsonl',
    status: 'running',
    created_at: '2025-01-20T10:30:00Z',
    completed_at: null,
    file_size: null,
    file_path: null,
    progress: 65,
    error_message: null
  },
  {
    id: 'export-003',
    dataset_id: 'dataset-789',
    dataset_name: 'Failed Dataset',
    config_id: 'config-003',
    format: 'pytorch',
    status: 'failed',
    created_at: '2025-01-20T09:45:00Z',
    completed_at: null,
    file_size: null,
    file_path: null,
    progress: 25,
    error_message: 'Invalid tensor format configuration'
  }
];

const mockActiveExports = [
  {
    id: 'export-002',
    dataset_id: 'dataset-456',
    dataset_name: 'Another Dataset',
    format: 'jsonl',
    progress: 65,
    current_step: 'Converting audio files',
    estimated_completion: '2025-01-20T10:45:00Z',
    speed: '2.5 MB/s'
  }
];

const mockAvailableDatasets = [
  {
    id: 'dataset-123',
    name: 'Test Dataset',
    created_at: '2025-01-20T09:00:00Z',
    sample_count: 1000,
    total_size: 500000
  },
  {
    id: 'dataset-456',
    name: 'Another Dataset',
    created_at: '2025-01-20T08:30:00Z',
    sample_count: 2000,
    total_size: 1000000
  }
];

describe('DatasetExportManager', () => {
  beforeEach(() => {
    (global.fetch as jest.Mock).mockClear();
    mockDownloadFile.mockClear();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Initial Render', () => {
    test('renders export manager with all sections', () => {
      render(<DatasetExportManager />);
      
      expect(screen.getByText('Dataset Export Manager')).toBeInTheDocument();
      expect(screen.getByText('Export Configuration')).toBeInTheDocument();
      expect(screen.getByText('Active Exports')).toBeInTheDocument();
      expect(screen.getByText('Export History')).toBeInTheDocument();
      expect(screen.getByText('Batch Export')).toBeInTheDocument();
    });

    test('loads export configurations on mount', async () => {
      render(<DatasetExportManager />);
      
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/exports/configs');
      });
    });

    test('loads export history on mount', async () => {
      render(<DatasetExportManager />);
      
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/exports/history');
      });
    });

    test('loads available datasets on mount', async () => {
      render(<DatasetExportManager />);
      
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/datasets');
      });
    });
  });

  describe('Export Configuration', () => {
    test('displays export format options', async () => {
      render(<DatasetExportManager />);
      
      await waitFor(() => {
        expect(screen.getByText('HuggingFace')).toBeInTheDocument();
        expect(screen.getByText('JSONL')).toBeInTheDocument();
        expect(screen.getByText('PyTorch')).toBeInTheDocument();
        expect(screen.getByText('Custom')).toBeInTheDocument();
      });
    });

    test('shows format-specific configuration options', async () => {
      render(<DatasetExportManager />);
      
      const huggingFaceOption = screen.getByTestId('format-huggingface');
      await userEvent.click(huggingFaceOption);

      await waitFor(() => {
        expect(screen.getByText('Train/Val/Test Split')).toBeInTheDocument();
        expect(screen.getByText('Tokenizer Config')).toBeInTheDocument();
        expect(screen.getByText('Dataset Features')).toBeInTheDocument();
      });
    });

    test('handles configuration creation', async () => {
      render(<DatasetExportManager />);
      
      const createButton = screen.getByText('Create New Configuration');
      await userEvent.click(createButton);

      await waitFor(() => {
        expect(screen.getByText('Configuration Name')).toBeInTheDocument();
      });

      const nameInput = screen.getByLabelText('Configuration Name');
      await userEvent.type(nameInput, 'My Custom Config');

      const formatSelect = screen.getByLabelText('Export Format');
      await userEvent.selectOptions(formatSelect, 'huggingface');

      const saveButton = screen.getByText('Save Configuration');
      await userEvent.click(saveButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/exports/configs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: 'My Custom Config',
            format: 'huggingface',
            options: expect.any(Object)
          })
        });
      });
    });

    test('handles configuration editing', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ configs: mockExportConfigs })
      });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        const editButton = screen.getByTestId('edit-config-config-001');
        expect(editButton).toBeInTheDocument();
      });

      const editButton = screen.getByTestId('edit-config-config-001');
      await userEvent.click(editButton);

      await waitFor(() => {
        expect(screen.getByDisplayValue('HuggingFace Standard')).toBeInTheDocument();
      });

      const nameInput = screen.getByLabelText('Configuration Name');
      await userEvent.clear(nameInput);
      await userEvent.type(nameInput, 'Updated HuggingFace Config');

      const saveButton = screen.getByText('Save Configuration');
      await userEvent.click(saveButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/exports/configs/config-001', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: 'Updated HuggingFace Config',
            format: 'huggingface',
            options: expect.any(Object)
          })
        });
      });
    });

    test('handles configuration deletion', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ configs: mockExportConfigs })
      });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        const deleteButton = screen.getByTestId('delete-config-config-001');
        expect(deleteButton).toBeInTheDocument();
      });

      const deleteButton = screen.getByTestId('delete-config-config-001');
      await userEvent.click(deleteButton);

      await waitFor(() => {
        expect(screen.getByText('Confirm Deletion')).toBeInTheDocument();
      });

      const confirmButton = screen.getByText('Delete Configuration');
      await userEvent.click(confirmButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/exports/configs/config-001', {
          method: 'DELETE'
        });
      });
    });
  });

  describe('Export Operations', () => {
    test('handles single dataset export', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ datasets: mockAvailableDatasets })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ configs: mockExportConfigs })
        });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        const exportButton = screen.getByText('Start Export');
        expect(exportButton).toBeInTheDocument();
      });

      const datasetSelect = screen.getByLabelText('Select Dataset');
      await userEvent.selectOptions(datasetSelect, 'dataset-123');

      const configSelect = screen.getByLabelText('Export Configuration');
      await userEvent.selectOptions(configSelect, 'config-001');

      const exportButton = screen.getByText('Start Export');
      await userEvent.click(exportButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/datasets/export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_id: 'dataset-123',
            export_config: expect.any(Object)
          })
        });
      });
    });

    test('handles batch export operations', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ datasets: mockAvailableDatasets })
      });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        const batchExportButton = screen.getByText('Batch Export');
        expect(batchExportButton).toBeInTheDocument();
      });

      const batchExportButton = screen.getByText('Batch Export');
      await userEvent.click(batchExportButton);

      await waitFor(() => {
        expect(screen.getByText('Select Multiple Datasets')).toBeInTheDocument();
      });

      // Select multiple datasets
      const dataset1Checkbox = screen.getByTestId('batch-dataset-dataset-123');
      const dataset2Checkbox = screen.getByTestId('batch-dataset-dataset-456');
      
      await userEvent.click(dataset1Checkbox);
      await userEvent.click(dataset2Checkbox);

      const configSelect = screen.getByLabelText('Export Configuration');
      await userEvent.selectOptions(configSelect, 'config-001');

      const startBatchButton = screen.getByText('Start Batch Export');
      await userEvent.click(startBatchButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/datasets/batch-export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_ids: ['dataset-123', 'dataset-456'],
            export_config: expect.any(Object)
          })
        });
      });
    });

    test('handles export cancellation', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ active_exports: mockActiveExports })
      });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        const cancelButton = screen.getByTestId('cancel-export-export-002');
        expect(cancelButton).toBeInTheDocument();
      });

      const cancelButton = screen.getByTestId('cancel-export-export-002');
      await userEvent.click(cancelButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/exports/export-002/cancel', {
          method: 'POST'
        });
      });
    });
  });

  describe('Export Progress Tracking', () => {
    test('displays active export progress', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ active_exports: mockActiveExports })
      });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        expect(screen.getByText('Another Dataset')).toBeInTheDocument();
        expect(screen.getByText('65%')).toBeInTheDocument();
        expect(screen.getByText('Converting audio files')).toBeInTheDocument();
        expect(screen.getByText('2.5 MB/s')).toBeInTheDocument();
      });
    });

    test('handles real-time progress updates', async () => {
      const mockWebSocket = {
        send: jest.fn(),
        close: jest.fn(),
        onopen: null,
        onmessage: null,
        onclose: null,
        onerror: null
      };

      global.WebSocket = jest.fn(() => mockWebSocket) as any;

      render(<DatasetExportManager />);
      
      // Simulate WebSocket progress update
      act(() => {
        if (mockWebSocket.onmessage) {
          mockWebSocket.onmessage({
            data: JSON.stringify({
              type: 'export_progress',
              export_id: 'export-002',
              progress: 75,
              current_step: 'Finalizing export',
              speed: '3.2 MB/s'
            })
          });
        }
      });

      await waitFor(() => {
        expect(screen.getByText('75%')).toBeInTheDocument();
        expect(screen.getByText('Finalizing export')).toBeInTheDocument();
        expect(screen.getByText('3.2 MB/s')).toBeInTheDocument();
      });
    });

    test('handles export completion', async () => {
      const mockWebSocket = {
        send: jest.fn(),
        close: jest.fn(),
        onopen: null,
        onmessage: null,
        onclose: null,
        onerror: null
      };

      global.WebSocket = jest.fn(() => mockWebSocket) as any;

      render(<DatasetExportManager />);
      
      // Simulate export completion
      act(() => {
        if (mockWebSocket.onmessage) {
          mockWebSocket.onmessage({
            data: JSON.stringify({
              type: 'export_completed',
              export_id: 'export-002',
              file_path: '/exports/another-dataset.jsonl.gz',
              file_size: 2048000
            })
          });
        }
      });

      await waitFor(() => {
        expect(screen.getByText('Export completed successfully')).toBeInTheDocument();
      });
    });

    test('shows estimated completion time', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ active_exports: mockActiveExports })
      });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        expect(screen.getByText('ETA: 5 minutes')).toBeInTheDocument();
      });
    });
  });

  describe('Export History', () => {
    test('displays export history', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ history: mockExportHistory })
      });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        expect(screen.getByText('Test Dataset')).toBeInTheDocument();
        expect(screen.getByText('Another Dataset')).toBeInTheDocument();
        expect(screen.getByText('Failed Dataset')).toBeInTheDocument();
      });
    });

    test('shows export status indicators', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ history: mockExportHistory })
      });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        expect(screen.getByTestId('export-status-completed')).toBeInTheDocument();
        expect(screen.getByTestId('export-status-running')).toBeInTheDocument();
        expect(screen.getByTestId('export-status-failed')).toBeInTheDocument();
      });
    });

    test('handles export download', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ history: mockExportHistory })
        })
        .mockResolvedValueOnce({
          ok: true,
          blob: () => Promise.resolve(new Blob(['test data'], { type: 'application/gzip' }))
        });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        const downloadButton = screen.getByTestId('download-export-export-001');
        expect(downloadButton).toBeInTheDocument();
      });

      const downloadButton = screen.getByTestId('download-export-export-001');
      await userEvent.click(downloadButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/exports/export-001/download');
      });
    });

    test('handles export re-export', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ history: mockExportHistory })
      });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        const reExportButton = screen.getByTestId('re-export-export-001');
        expect(reExportButton).toBeInTheDocument();
      });

      const reExportButton = screen.getByTestId('re-export-export-001');
      await userEvent.click(reExportButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/exports/re-export', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset_id: 'dataset-123',
            config_id: 'config-001'
          })
        });
      });
    });

    test('handles export deletion', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ history: mockExportHistory })
      });

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        const deleteButton = screen.getByTestId('delete-export-export-001');
        expect(deleteButton).toBeInTheDocument();
      });

      const deleteButton = screen.getByTestId('delete-export-export-001');
      await userEvent.click(deleteButton);

      await waitFor(() => {
        expect(screen.getByText('Confirm Export Deletion')).toBeInTheDocument();
      });

      const confirmButton = screen.getByText('Delete Export');
      await userEvent.click(confirmButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/exports/export-001', {
          method: 'DELETE'
        });
      });
    });
  });

  describe('Format-Specific Features', () => {
    test('shows HuggingFace format options', async () => {
      render(<DatasetExportManager />);
      
      const formatSelect = screen.getByLabelText('Export Format');
      await userEvent.selectOptions(formatSelect, 'huggingface');

      await waitFor(() => {
        expect(screen.getByText('Dataset Hub Upload')).toBeInTheDocument();
        expect(screen.getByText('Model Card Generation')).toBeInTheDocument();
        expect(screen.getByText('Feature Configuration')).toBeInTheDocument();
      });
    });

    test('shows JSONL format options', async () => {
      render(<DatasetExportManager />);
      
      const formatSelect = screen.getByLabelText('Export Format');
      await userEvent.selectOptions(formatSelect, 'jsonl');

      await waitFor(() => {
        expect(screen.getByText('Compression')).toBeInTheDocument();
        expect(screen.getByText('Chunk Size')).toBeInTheDocument();
        expect(screen.getByText('Encoding')).toBeInTheDocument();
      });
    });

    test('shows PyTorch format options', async () => {
      render(<DatasetExportManager />);
      
      const formatSelect = screen.getByLabelText('Export Format');
      await userEvent.selectOptions(formatSelect, 'pytorch');

      await waitFor(() => {
        expect(screen.getByText('Tensor Format')).toBeInTheDocument();
        expect(screen.getByText('DataLoader Config')).toBeInTheDocument();
        expect(screen.getByText('Transforms')).toBeInTheDocument();
      });
    });

    test('handles custom format configuration', async () => {
      render(<DatasetExportManager />);
      
      const formatSelect = screen.getByLabelText('Export Format');
      await userEvent.selectOptions(formatSelect, 'custom');

      await waitFor(() => {
        expect(screen.getByText('Custom Export Script')).toBeInTheDocument();
        expect(screen.getByText('Output Structure')).toBeInTheDocument();
      });

      const scriptTextarea = screen.getByLabelText('Export Script');
      await userEvent.type(scriptTextarea, 'def export_custom(dataset): pass');

      expect(scriptTextarea).toHaveValue('def export_custom(dataset): pass');
    });
  });

  describe('Error Handling', () => {
    test('handles API errors gracefully', async () => {
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('API Error'));
      
      render(<DatasetExportManager />);
      
      await waitFor(() => {
        expect(screen.getByText('Failed to load export configurations')).toBeInTheDocument();
      });
    });

    test('handles export errors', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ error: 'Invalid export configuration' })
      });

      render(<DatasetExportManager />);
      
      const exportButton = screen.getByText('Start Export');
      await userEvent.click(exportButton);

      await waitFor(() => {
        expect(screen.getByText('Export failed: Invalid export configuration')).toBeInTheDocument();
      });
    });

    test('handles download errors', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ history: mockExportHistory })
        })
        .mockRejectedValueOnce(new Error('Download failed'));

      render(<DatasetExportManager />);
      
      await waitFor(() => {
        const downloadButton = screen.getByTestId('download-export-export-001');
        expect(downloadButton).toBeInTheDocument();
      });

      const downloadButton = screen.getByTestId('download-export-export-001');
      await userEvent.click(downloadButton);

      await waitFor(() => {
        expect(screen.getByText('Download failed')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    test('has proper ARIA labels', async () => {
      render(<DatasetExportManager />);
      
      expect(screen.getByLabelText('Dataset export manager')).toBeInTheDocument();
      expect(screen.getByLabelText('Export configurations')).toBeInTheDocument();
      expect(screen.getByLabelText('Active exports')).toBeInTheDocument();
      expect(screen.getByLabelText('Export history')).toBeInTheDocument();
    });

    test('supports keyboard navigation', async () => {
      render(<DatasetExportManager />);
      
      // Navigate using Tab key
      await userEvent.tab();
      expect(screen.getByLabelText('Select Dataset')).toHaveFocus();
      
      await userEvent.tab();
      expect(screen.getByLabelText('Export Configuration')).toHaveFocus();
    });

    test('has proper form labels', async () => {
      render(<DatasetExportManager />);
      
      expect(screen.getByLabelText('Select Dataset')).toBeInTheDocument();
      expect(screen.getByLabelText('Export Configuration')).toBeInTheDocument();
      expect(screen.getByLabelText('Export Format')).toBeInTheDocument();
    });
  });

  describe('Performance', () => {
    test('handles large export history efficiently', async () => {
      const largeHistory = Array.from({ length: 1000 }, (_, i) => ({
        ...mockExportHistory[0],
        id: `export-${i}`,
        dataset_name: `Dataset ${i}`
      }));

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ history: largeHistory })
      });

      render(<DatasetExportManager />);
      
      // Should render without performance issues
      await waitFor(() => {
        expect(screen.getByText('Dataset 0')).toBeInTheDocument();
      });
    });

    test('debounces real-time updates', async () => {
      const mockWebSocket = {
        send: jest.fn(),
        close: jest.fn(),
        onopen: null,
        onmessage: null,
        onclose: null,
        onerror: null
      };

      global.WebSocket = jest.fn(() => mockWebSocket) as any;

      render(<DatasetExportManager />);
      
      // Send multiple rapid progress updates
      for (let i = 0; i < 10; i++) {
        act(() => {
          if (mockWebSocket.onmessage) {
            mockWebSocket.onmessage({
              data: JSON.stringify({
                type: 'export_progress',
                export_id: 'export-002',
                progress: 60 + i,
                current_step: `Step ${i}`,
                speed: `${2 + i} MB/s`
              })
            });
          }
        });
      }

      // Should only apply the last update
      await waitFor(() => {
        expect(screen.getByText('69%')).toBeInTheDocument();
        expect(screen.getByText('Step 9')).toBeInTheDocument();
      });
    });
  });
}); 