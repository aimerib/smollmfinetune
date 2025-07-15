/**
 * Tests for DatasetQualityDashboard Component - R6-9 Production Features
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { jest } from '@jest/globals';
import DatasetQualityDashboard from '../../components/DatasetQualityDashboard';

// Mock console methods to reduce test noise
const originalConsole = { ...console };
beforeAll(() => {
  console.log = jest.fn();
  console.error = jest.fn();
});

afterAll(() => {
  Object.assign(console, originalConsole);
});

// Mock WebSocket
class MockWebSocket {
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  
  constructor(url: string) {
    // Simulate connection after a short delay
    setTimeout(() => {
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 10);
  }
  
  close() {
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }
  
  send(data: string) {
    // Mock send method
  }
}

global.WebSocket = MockWebSocket as any;

// Mock fetch for API calls
global.fetch = jest.fn((url) => {
  let responseData;
  
  if (url.includes('/api/multimodal/quality/metrics')) {
    responseData = { metrics: mockQualityMetrics };
  } else if (url.includes('/api/multimodal/quality/validation-history')) {
    responseData = { history: [mockValidationReport] };
  } else if (url.includes('/api/multimodal/quality/improvement-suggestions')) {
    responseData = { suggestions: [] };
  } else {
    responseData = { success: true };
  }
  
  return Promise.resolve({
    ok: true,
    json: () => Promise.resolve(responseData),
  });
}) as jest.Mock;

// Test data
const mockQualityMetrics = {
  overall_score: 0.85,
  text_quality: {
    coherence: 0.87,
    fluency: 0.92,
    relevance: 0.78,
    diversity: 0.83,
    grammar_score: 0.94,
    readability: 0.88,
    toxicity_score: 0.02,
    bias_score: 0.03
  },
  audio_quality: {
    clarity: 0.89,
    naturalness: 0.91,
    emotional_consistency: 0.84,
    pronunciation: 0.93,
    prosody: 0.87,
    background_noise: 0.95,
    volume_consistency: 0.88,
    artifacts_score: 0.92
  },
  character_consistency: {
    personality_alignment: 0.86,
    voice_consistency: 0.89,
    behavioral_consistency: 0.82,
    emotional_range: 0.85,
    dialogue_style: 0.88,
    character_arc: 0.84
  },
  dataset_balance: {
    narrative_type_distribution: {
      dialogue: 0.35,
      monologue: 0.25,
      action_scene: 0.20,
      emotional_moment: 0.15,
      memory_recall: 0.05
    },
    character_distribution: 0.87,
    emotion_distribution: 0.82,
    length_distribution: 0.78
  },
  technical_metrics: {
    processing_time: 1245.67,
    error_rate: 0.03,
    completion_rate: 0.97,
    resource_efficiency: 0.84,
    scalability_score: 0.79
  }
};

const mockValidationReport = {
  id: 'validation-001',
  dataset_id: 'dataset-123',
  dataset_name: 'Test Dataset',
  created_at: '2025-01-20T10:00:00Z',
  status: 'completed',
  metrics: mockQualityMetrics,
  issues: [
    {
      severity: 'high',
      category: 'text_quality',
      message: 'Low relevance score detected in 5% of samples',
      affected_samples: 47,
      suggestions: [
        'Review and improve prompt engineering',
        'Adjust narrative context parameters',
        'Implement stricter filtering criteria'
      ]
    },
    {
      severity: 'medium',
      category: 'character_consistency',
      message: 'Behavioral inconsistency in character responses',
      affected_samples: 23,
      suggestions: [
        'Refine character personality definitions',
        'Implement consistency checking during generation',
        'Add character state tracking'
      ]
    },
    {
      severity: 'low',
      category: 'dataset_balance',
      message: 'Slight imbalance in narrative type distribution',
      affected_samples: 15,
      suggestions: [
        'Adjust generation parameters for better balance',
        'Implement dynamic sampling strategy'
      ]
    }
  ],
  recommendations: [
    'Increase sample diversity to improve overall quality',
    'Implement additional quality checks during generation',
    'Consider post-processing for consistency improvements'
  ],
  quality_trend: [
    { timestamp: '2025-01-20T09:00:00Z', score: 0.82 },
    { timestamp: '2025-01-20T09:30:00Z', score: 0.84 },
    { timestamp: '2025-01-20T10:00:00Z', score: 0.85 }
  ]
};

const mockValidationHistory = [
  {
    id: 'validation-001',
    dataset_name: 'Test Dataset',
    created_at: '2025-01-20T10:00:00Z',
    overall_score: 0.85,
    status: 'completed'
  },
  {
    id: 'validation-002',
    dataset_name: 'Another Dataset',
    created_at: '2025-01-20T09:00:00Z',
    overall_score: 0.78,
    status: 'completed'
  }
];

const mockImprovementSuggestions = [
  {
    id: 'suggestion-001',
    category: 'text_quality',
    priority: 'high',
    title: 'Improve Narrative Coherence',
    description: 'Implement advanced context tracking for better story flow',
    impact: 'Expected 15% improvement in coherence score',
    effort: 'medium',
    estimated_time: '2-3 days',
    implementation_steps: [
      'Analyze current context tracking mechanisms',
      'Implement enhanced narrative state management',
      'Test and validate improvements'
    ]
  },
  {
    id: 'suggestion-002',
    category: 'character_consistency',
    priority: 'medium',
    title: 'Character Personality Refinement',
    description: 'Enhance character personality definitions for better consistency',
    impact: 'Expected 10% improvement in character consistency',
    effort: 'low',
    estimated_time: '1-2 days',
    implementation_steps: [
      'Review current character definitions',
      'Implement personality constraint checking',
      'Update character generation parameters'
    ]
  }
];

describe('DatasetQualityDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Reset and properly configure fetch mock for each test
    (global.fetch as jest.Mock).mockImplementation((url) => {
      let responseData;
      
      if (url.includes('/api/multimodal/quality/metrics')) {
        responseData = { metrics: mockQualityMetrics };
      } else if (url.includes('/api/multimodal/quality/validation-history')) {
        responseData = { history: [mockValidationReport] };
      } else if (url.includes('/api/multimodal/quality/improvement-suggestions')) {
        responseData = { 
          suggestions: [
            {
              id: 'suggestion-001',
              title: 'Improve Narrative Coherence',
              priority: 'high',
              category: 'text_quality',
              description: 'Enhance story flow and logical progression',
              estimated_impact: 'high',
              effort: 'medium',
              estimated_time: '2-3 hours',
              implementation_steps: ['Review current narratives', 'Apply coherence improvements']
            },
            {
              id: 'suggestion-002',
              title: 'Character Personality Refinement',
              priority: 'medium',
              category: 'character_consistency',
              description: 'Strengthen character personality traits',
              estimated_impact: 'medium',
              effort: 'low',
              estimated_time: '1-2 hours',
              implementation_steps: ['Analyze personality traits', 'Apply refinements']
            }
          ]
        };
      } else {
        responseData = { success: true };
      }
      
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(responseData),
      });
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Initial Render', () => {
    test('renders dashboard with all sections', () => {
      render(<DatasetQualityDashboard />);
      
      expect(screen.getByText('Dataset Quality Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Quality Overview')).toBeInTheDocument();
      expect(screen.getByText('Validation Reports')).toBeInTheDocument();
      expect(screen.getByText('Quality Metrics')).toBeInTheDocument();
      expect(screen.getByText('Improvement Suggestions')).toBeInTheDocument();
    });

    test('loads quality metrics on mount', async () => {
      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/quality/metrics');
      });
    });

    test('loads validation history on mount', async () => {
      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/quality/validation-history');
      });
    });
  });

  describe('Quality Overview', () => {
    test('displays overall quality score', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ metrics: mockQualityMetrics })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Overall Quality: 85%')).toBeInTheDocument();
      });
    });

    test('displays quality breakdown', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ metrics: mockQualityMetrics })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Text Quality: 87%')).toBeInTheDocument();
        expect(screen.getByText('Audio Quality: 89%')).toBeInTheDocument();
        expect(screen.getByText('Character Consistency: 86%')).toBeInTheDocument();
      });
    });

    test('shows quality trend chart', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ metrics: mockQualityMetrics })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(screen.getByTestId('quality-trend-chart')).toBeInTheDocument();
      });
    });
  });

  describe('Validation Reports', () => {
    test('displays validation history', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ history: mockValidationHistory })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Test Dataset')).toBeInTheDocument();
        expect(screen.getByText('Another Dataset')).toBeInTheDocument();
      });
    });

    test('handles validation report click', async () => {
      (global.fetch as jest.Mock)
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ history: mockValidationHistory })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({ report: mockValidationReport })
        });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        const reportButton = screen.getByTestId('validation-report-validation-001');
        expect(reportButton).toBeInTheDocument();
      });

      const reportButton = screen.getByTestId('validation-report-validation-001');
      await userEvent.click(reportButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/quality/validation-report/validation-001');
      });
    });

    test('displays detailed validation report', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ report: mockValidationReport })
      });

      render(<DatasetQualityDashboard />);
      
      // Simulate clicking a report
      const reportButton = screen.getByTestId('validation-report-validation-001');
      await userEvent.click(reportButton);

      await waitFor(() => {
        expect(screen.getByText('Validation Report Details')).toBeInTheDocument();
        expect(screen.getByText('Low relevance score detected in 5% of samples')).toBeInTheDocument();
        expect(screen.getByText('Behavioral inconsistency in character responses')).toBeInTheDocument();
      });
    });

    test('shows issue severity indicators', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ report: mockValidationReport })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(screen.getByTestId('issue-severity-high')).toBeInTheDocument();
        expect(screen.getByTestId('issue-severity-medium')).toBeInTheDocument();
        expect(screen.getByTestId('issue-severity-low')).toBeInTheDocument();
      });
    });
  });

  describe('Quality Metrics', () => {
    test('displays detailed metrics breakdown', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ metrics: mockQualityMetrics })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Coherence: 87%')).toBeInTheDocument();
        expect(screen.getByText('Fluency: 92%')).toBeInTheDocument();
        expect(screen.getByText('Clarity: 89%')).toBeInTheDocument();
        expect(screen.getByText('Naturalness: 91%')).toBeInTheDocument();
      });
    });

    test('shows technical performance metrics', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ metrics: mockQualityMetrics })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Processing Time: 1245.67s')).toBeInTheDocument();
        expect(screen.getByText('Error Rate: 3%')).toBeInTheDocument();
        expect(screen.getByText('Completion Rate: 97%')).toBeInTheDocument();
      });
    });

    test('displays dataset balance metrics', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ metrics: mockQualityMetrics })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Dialogue: 35%')).toBeInTheDocument();
        expect(screen.getByText('Monologue: 25%')).toBeInTheDocument();
        expect(screen.getByText('Action Scene: 20%')).toBeInTheDocument();
      });
    });

    test('handles metric drill-down', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ metrics: mockQualityMetrics })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        const coherenceMetric = screen.getByTestId('metric-coherence');
        expect(coherenceMetric).toBeInTheDocument();
      });

      const coherenceMetric = screen.getByTestId('metric-coherence');
      await userEvent.click(coherenceMetric);

      await waitFor(() => {
        expect(screen.getByText('Coherence Details')).toBeInTheDocument();
        expect(screen.getByText('Sample Analysis')).toBeInTheDocument();
      });
    });
  });

  describe('Improvement Suggestions', () => {
    test('displays improvement suggestions', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ suggestions: mockImprovementSuggestions })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Improve Narrative Coherence')).toBeInTheDocument();
        expect(screen.getByText('Character Personality Refinement')).toBeInTheDocument();
      });
    });

    test('shows suggestion priority indicators', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ suggestions: mockImprovementSuggestions })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(screen.getByTestId('suggestion-priority-high')).toBeInTheDocument();
        expect(screen.getByTestId('suggestion-priority-medium')).toBeInTheDocument();
      });
    });

    test('handles suggestion application', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ suggestions: mockImprovementSuggestions })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        const applyButton = screen.getByTestId('apply-suggestion-suggestion-001');
        expect(applyButton).toBeInTheDocument();
      });

      const applyButton = screen.getByTestId('apply-suggestion-suggestion-001');
      await userEvent.click(applyButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/quality/apply-suggestion', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ suggestion_id: 'suggestion-001' })
        });
      });
    });

    test('shows suggestion details on expand', async () => {
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ suggestions: mockImprovementSuggestions })
      });

      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        const expandButton = screen.getByTestId('expand-suggestion-suggestion-001');
        expect(expandButton).toBeInTheDocument();
      });

      const expandButton = screen.getByTestId('expand-suggestion-suggestion-001');
      await userEvent.click(expandButton);

      await waitFor(() => {
        expect(screen.getByText('Expected 15% improvement in coherence score')).toBeInTheDocument();
        expect(screen.getByText('Estimated Time: 2-3 days')).toBeInTheDocument();
        expect(screen.getByText('Analyze current context tracking mechanisms')).toBeInTheDocument();
      });
    });
  });

  describe('Quality Validation Actions', () => {
    test('handles new validation request', async () => {
      render(<DatasetQualityDashboard />);
      
      const validateButton = screen.getByText('Run Quality Validation');
      await userEvent.click(validateButton);

      await waitFor(() => {
        expect(screen.getByText('Select Dataset')).toBeInTheDocument();
      });

      const datasetSelect = screen.getByTestId('dataset-select');
      await userEvent.selectOptions(datasetSelect, 'dataset-123');

      const confirmButton = screen.getByText('Start Validation');
      await userEvent.click(confirmButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/datasets/dataset-123/validate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        });
      });
    });

    test('handles comparison between datasets', async () => {
      render(<DatasetQualityDashboard />);
      
      const compareButton = screen.getByText('Compare Datasets');
      await userEvent.click(compareButton);

      await waitFor(() => {
        expect(screen.getByText('Select Datasets to Compare')).toBeInTheDocument();
      });

      const dataset1Select = screen.getByTestId('compare-dataset-1');
      const dataset2Select = screen.getByTestId('compare-dataset-2');
      
      await userEvent.selectOptions(dataset1Select, 'dataset-123');
      await userEvent.selectOptions(dataset2Select, 'dataset-456');

      const startCompareButton = screen.getByText('Start Comparison');
      await userEvent.click(startCompareButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/quality/compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dataset1_id: 'dataset-123',
            dataset2_id: 'dataset-456'
          })
        });
      });
    });

    test('handles quality trend analysis', async () => {
      render(<DatasetQualityDashboard />);
      
      const trendButton = screen.getByText('Analyze Quality Trends');
      await userEvent.click(trendButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledWith('/api/multimodal/quality/trends', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ time_range: '7d' })
        });
      });
    });
  });

  describe('Real-time Quality Monitoring', () => {
    test('displays real-time quality updates', async () => {
      const mockWebSocket = {
        send: jest.fn(),
        close: jest.fn(),
        onopen: null,
        onmessage: null,
        onclose: null,
        onerror: null
      };

      global.WebSocket = jest.fn(() => mockWebSocket) as any;

      render(<DatasetQualityDashboard />);
      
      // Simulate WebSocket message
      act(() => {
        if (mockWebSocket.onmessage) {
          mockWebSocket.onmessage({
            data: JSON.stringify({
              type: 'quality_update',
              metrics: { overall_score: 0.87 }
            })
          });
        }
      });

      await waitFor(() => {
        expect(screen.getByText('Overall Quality: 87%')).toBeInTheDocument();
      });
    });

    test('handles real-time validation progress', async () => {
      const mockWebSocket = {
        send: jest.fn(),
        close: jest.fn(),
        onopen: null,
        onmessage: null,
        onclose: null,
        onerror: null
      };

      global.WebSocket = jest.fn(() => mockWebSocket) as any;

      render(<DatasetQualityDashboard />);
      
      // Simulate validation progress update
      act(() => {
        if (mockWebSocket.onmessage) {
          mockWebSocket.onmessage({
            data: JSON.stringify({
              type: 'validation_progress',
              validation_id: 'validation-001',
              progress: 65,
              current_step: 'Analyzing character consistency'
            })
          });
        }
      });

      await waitFor(() => {
        expect(screen.getByText('Analyzing character consistency')).toBeInTheDocument();
        expect(screen.getByText('65%')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    test('handles API errors gracefully', async () => {
      (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('API Error'));
      
      render(<DatasetQualityDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('API Error')).toBeInTheDocument();
      });
    });

    test('handles validation errors', async () => {
      // First, let the component load successfully
      render(<DatasetQualityDashboard />);
      
      // Wait for the initial load to complete
      await waitFor(() => {
        expect(screen.getByText('Run Quality Validation')).toBeInTheDocument();
      });

      // Now mock the validation request to fail
      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ error: 'Invalid dataset' })
      });
      
      const validateButton = screen.getByText('Run Quality Validation');
      await userEvent.click(validateButton);

      await waitFor(() => {
        expect(screen.getByText('Invalid dataset')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    test('has proper ARIA labels', async () => {
      render(<DatasetQualityDashboard />);
      
      expect(screen.getByLabelText('Dataset quality dashboard')).toBeInTheDocument();
      expect(screen.getByLabelText('Quality metrics overview')).toBeInTheDocument();
      expect(screen.getByLabelText('Validation reports list')).toBeInTheDocument();
      expect(screen.getByLabelText('Improvement suggestions')).toBeInTheDocument();
    });

    test('supports keyboard navigation', async () => {
      render(<DatasetQualityDashboard />);
      
      // Navigate using Tab key
      await userEvent.tab();
      expect(screen.getByText('Run Quality Validation')).toHaveFocus();
      
      await userEvent.tab();
      expect(screen.getByText('Compare Datasets')).toHaveFocus();
    });
  });

  describe('Performance', () => {
    test('handles large validation reports efficiently', async () => {
      const largeReport = {
        ...mockValidationReport,
        issues: Array.from({ length: 1000 }, (_, i) => ({
          severity: 'low',
          category: 'text_quality',
          message: `Issue ${i}`,
          affected_samples: 1,
          suggestions: [`Suggestion ${i}`]
        }))
      };

      (global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ report: largeReport })
      });

      render(<DatasetQualityDashboard />);
      
      // Should render without performance issues
      await waitFor(() => {
        expect(screen.getByText('Issue 0')).toBeInTheDocument();
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

      render(<DatasetQualityDashboard />);
      
      // Send multiple rapid updates
      for (let i = 0; i < 10; i++) {
        act(() => {
          if (mockWebSocket.onmessage) {
            mockWebSocket.onmessage({
              data: JSON.stringify({
                type: 'quality_update',
                metrics: { overall_score: 0.8 + (i * 0.01) }
              })
            });
          }
        });
      }

      // Should only apply the last update
      await waitFor(() => {
        expect(screen.getByText('Overall Quality: 89%')).toBeInTheDocument();
      });
    });
  });
}); 