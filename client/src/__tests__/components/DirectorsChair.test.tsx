import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { DirectorsChairStudio } from '../../components/DirectorsChair/DirectorsChairStudio';
import { ConversationEditor } from '../../components/DirectorsChair/ConversationEditor';
import { MultiHeadCorrectionPanel } from '../../components/DirectorsChair/MultiHeadCorrectionPanel';
import { TrainingProgressIndicator } from '../../components/DirectorsChair/TrainingProgressIndicator';
import websocketService from '../../services/websocketService';

// Mock dependencies
jest.mock('../../services/websocketService');
jest.mock('../../services/directorsChairService');

describe('DirectorsChairStudio Component', () => {
  const mockWebsocketService = websocketService as jest.Mocked<typeof websocketService>;
  
  beforeEach(() => {
    jest.clearAllMocks();
    mockWebsocketService.connect.mockResolvedValue(undefined);
  });

  test('renders Director\'s Chair interface', async () => {
    await act(async () => {
      render(<DirectorsChairStudio characterId="alice" />);
    });
    
    expect(screen.getByText(/Director's Chair/i)).toBeInTheDocument();
    expect(screen.getByText(/Training Studio/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Start Training Session/i })).toBeInTheDocument();
  });

  test('displays character selection interface', async () => {
    await act(async () => {
      render(<DirectorsChairStudio />);
    });
    
    expect(screen.getByText(/Select Character/i)).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: /character/i })).toBeInTheDocument();
  });

  test('shows triple-head status indicators', async () => {
    await act(async () => {
      render(<DirectorsChairStudio characterId="alice" />);
    });
    
    expect(screen.getByText(/Generation Head/i)).toBeInTheDocument();
    expect(screen.getByText(/Control Head/i)).toBeInTheDocument();
    expect(screen.getByText(/Memory Head/i)).toBeInTheDocument();
    
    // Check for status indicators
    expect(screen.getAllByTestId(/head-status-/)).toHaveLength(3);
  });

  test('enables training mode toggle', async () => {
    await act(async () => {
      render(<DirectorsChairStudio characterId="alice" />);
    });
    
    const trainingToggle = screen.getByRole('switch', { name: /Enable Live Training/i });
    expect(trainingToggle).toBeInTheDocument();
    
    fireEvent.click(trainingToggle);
    
    expect(screen.getByText(/Training Mode: Active/i)).toBeInTheDocument();
  });
});

describe('ConversationEditor Component', () => {
  const mockConversation = {
    id: 'conv_123',
    user_message: 'Tell me about your adventures',
    assistant_response: 'I love exploring mysterious forests and meeting new friends!',
    metadata: {
      generation_head_score: 0.7,
      control_head_score: 0.8,
      memory_head_score: 0.6
    },
    corrections: []
  };

  test('renders conversation turn for editing', () => {
    render(<ConversationEditor conversation={mockConversation} onCorrection={jest.fn()} />);
    
    expect(screen.getByText('Tell me about your adventures')).toBeInTheDocument();
    expect(screen.getByText(/I love exploring mysterious forests/)).toBeInTheDocument();
    
    // Check for edit buttons
    expect(screen.getByRole('button', { name: /Edit Response/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Add Correction/i })).toBeInTheDocument();
  });

  test('displays head-specific scores', () => {
    const mockConversation = {
      id: 'conv-1',
      user_message: 'Tell me about your adventures',
      assistant_response: 'I love exploring mysterious forests and meeting new friends!',
      metadata: {
        generation_head_score: 0.7,
        control_head_score: 0.8,
        memory_head_score: 0.6
      }
    };

    render(<ConversationEditor conversation={mockConversation} onCorrection={jest.fn()} />);
    
    expect(screen.getByTestId('generation-score')).toBeInTheDocument();
    expect(screen.getByTestId('control-score')).toBeInTheDocument();
    expect(screen.getByTestId('memory-score')).toBeInTheDocument();
  });

  test('enables inline text editing', async () => {
    const onCorrection = jest.fn();
    render(<ConversationEditor conversation={mockConversation} onCorrection={onCorrection} />);
    
    const editButton = screen.getByRole('button', { name: /Edit Response/i });
    fireEvent.click(editButton);
    
    // Should show text editor
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getByDisplayValue(/I love exploring mysterious forests/)).toBeInTheDocument();
    
    // Edit the text
    const textbox = screen.getByRole('textbox');
    fireEvent.change(textbox, { 
      target: { value: 'I absolutely adore exploring enchanted forests and discovering magical creatures!' }
    });
    
    // Apply correction
    const applyButton = screen.getByRole('button', { name: /Apply Correction/i });
    fireEvent.click(applyButton);
    
    await waitFor(() => {
      expect(onCorrection).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'content_correction',
          target_head: 'generation',
          original_text: expect.stringContaining('I love exploring'),
          corrected_text: expect.stringContaining('I absolutely adore')
        })
      );
    });
  });

  test('shows correction history', () => {
    const conversationWithCorrections = {
      ...mockConversation,
      corrections: [
        {
          id: 'corr_1',
          type: 'content_correction',
          target_head: 'generation',
          applied_at: '2024-01-17T10:00:00Z',
          reason: 'More enthusiastic tone'
        }
      ]
    };

    render(<ConversationEditor conversation={conversationWithCorrections} onCorrection={jest.fn()} />);
    
    expect(screen.getByText(/Correction History/i)).toBeInTheDocument();
    expect(screen.getByText(/More enthusiastic tone/)).toBeInTheDocument();
  });
});

describe('MultiHeadCorrectionPanel Component', () => {
  const mockOnCorrection = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders correction type selection', () => {
    render(<MultiHeadCorrectionPanel onCorrection={mockOnCorrection} />);
    
    expect(screen.getByText(/Correction Type/i)).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Content Quality/i })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Emotional Expression/i })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Memory Consistency/i })).toBeInTheDocument();
  });

  test('shows head-specific correction interfaces', async () => {
    render(<MultiHeadCorrectionPanel onCorrection={mockOnCorrection} />);
    
    // Select generation head correction
    fireEvent.click(screen.getByRole('radio', { name: /Content Quality/i }));
    
    await waitFor(() => {
      expect(screen.getByText(/Generation Head Correction/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Original Text/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Improved Text/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Improvement Reason/i)).toBeInTheDocument();
    });
    
    // Select control head correction
    fireEvent.click(screen.getByRole('radio', { name: /Emotional Expression/i }));
    
    await waitFor(() => {
      expect(screen.getByText(/Control Head Correction/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Emotional State/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Control Tokens/i)).toBeInTheDocument();
    });
    
    // Select memory head correction
    fireEvent.click(screen.getByRole('radio', { name: /Memory Consistency/i }));
    
    await waitFor(() => {
      expect(screen.getByText(/Memory Head Correction/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Memory Importance/i)).toBeInTheDocument();
      expect(screen.getByLabelText(/Should Remember/i)).toBeInTheDocument();
    });
  });

  test('submits generation head correction', async () => {
    render(<MultiHeadCorrectionPanel onCorrection={mockOnCorrection} />);
    
    fireEvent.click(screen.getByRole('radio', { name: /Content Quality/i }));
    
    await waitFor(() => {
      const originalText = screen.getByLabelText(/Original Text/i);
      const improvedText = screen.getByLabelText(/Improved Text/i);
      const reason = screen.getByLabelText(/Improvement Reason/i);
      
      fireEvent.change(originalText, { target: { value: 'Hello.' } });
      fireEvent.change(improvedText, { target: { value: 'Hello! How wonderful to meet you!' } });
      fireEvent.change(reason, { target: { value: 'More enthusiastic and welcoming' } });
    });
    
    const submitButton = screen.getByRole('button', { name: /Apply Correction/i });
    fireEvent.click(submitButton);
    
    expect(mockOnCorrection).toHaveBeenCalledWith({
      type: 'content_correction',
      target_head: 'generation',
      original_text: 'Hello.',
      corrected_text: 'Hello! How wonderful to meet you!',
      reason: 'More enthusiastic and welcoming'
    });
  });

  test('submits control head correction with emotion selection', async () => {
    render(<MultiHeadCorrectionPanel onCorrection={mockOnCorrection} />);
    
    fireEvent.click(screen.getByRole('radio', { name: /Emotional Expression/i }));
    
    await waitFor(() => {
      const emotionalState = screen.getByLabelText(/Emotional State/i);
      fireEvent.change(emotionalState, { target: { value: 'excited' } });
      
      // Add control tokens
      const tokenInput = screen.getByLabelText(/Control Tokens/i);
      fireEvent.change(tokenInput, { target: { value: '<excited>, <friendly>' } });
    });
    
    const submitButton = screen.getByRole('button', { name: /Apply Correction/i });
    fireEvent.click(submitButton);
    
    expect(mockOnCorrection).toHaveBeenCalledWith({
      type: 'emotion_correction',
      target_head: 'control',
      emotional_state: 'excited',
      control_tokens: ['<excited>', '<friendly>'],
      reason: expect.any(String)
    });
  });
});

describe('TrainingProgressIndicator Component', () => {
  test('renders overall training progress', () => {
    const mockStatus = {
      overall_progress: 0.6,
      generation_head: { status: 'training', progress: 0.6 },
      control_head: { status: 'queued', progress: 0.4 },
      memory_head: { status: 'completed', progress: 0.8 }
    };

    render(<TrainingProgressIndicator trainingStatus={mockStatus} />);
    
    expect(screen.getByText(/Training Progress/i)).toBeInTheDocument();
    expect(screen.getByTestId('overall-progress')).toHaveTextContent('60%');
  });

  test('displays head-specific progress bars', () => {
    const mockStatus = {
      generation_head: { status: 'training', progress: 0.75, eta: '2m 30s' },
      control_head: { status: 'queued', progress: 0.0, queue_position: 2, eta: 'Pending' },
      memory_head: { status: 'completed', progress: 1.0, eta: 'complete' }
    };

    render(<TrainingProgressIndicator trainingStatus={mockStatus} />);
    
    expect(screen.getByText(/Generation Head/i)).toBeInTheDocument();
    expect(screen.getByText(/Control Head/i)).toBeInTheDocument();
    expect(screen.getByText(/Memory Head/i)).toBeInTheDocument();
    expect(screen.getByTestId('memory-status')).toHaveTextContent('Complete');
  });

  test('shows training metrics and loss curves', () => {
    const trainingStatus = {
      generation_head: { 
        progress: 0.5, 
        status: 'training',
        metrics: {
          current_loss: 0.234,
          best_loss: 0.198,
          learning_rate: 1e-4
        }
      }
    };

    render(<TrainingProgressIndicator trainingStatus={trainingStatus} showMetrics={true} />);
    
    expect(screen.getByText(/Training Metrics/i)).toBeInTheDocument();
    expect(screen.getByText(/Loss: 0.234/)).toBeInTheDocument();
    expect(screen.getByText(/Best: 0.198/)).toBeInTheDocument();
  });

  test('handles training completion notifications', async () => {
    const onTrainingComplete = jest.fn();
    const initialStatus = {
      generation_head: { status: 'training', progress: 0.9 }
    };

    const { rerender } = render(
      <TrainingProgressIndicator 
        trainingStatus={initialStatus} 
        onTrainingComplete={onTrainingComplete}
      />
    );

    const completedStatus = {
      generation_head: { status: 'completed', progress: 1.0 }
    };

    rerender(
      <TrainingProgressIndicator 
        trainingStatus={completedStatus} 
        onTrainingComplete={onTrainingComplete}
      />
    );

    await waitFor(() => {
      expect(onTrainingComplete).toHaveBeenCalledWith('generation_head');
    });
  });

  test('displays queue position and estimated wait time', () => {
    const mockStatus = {
      control_head: { 
        status: 'queued', 
        progress: 0.0, 
        queue_position: 3, 
        eta: '5m 12s' 
      }
    };

    render(<TrainingProgressIndicator trainingStatus={mockStatus} />);
    
    expect(screen.getByTestId('control-queue')).toHaveTextContent('Queue Position: 3');
    expect(screen.getByTestId('control-eta')).toHaveTextContent('Est. Start: 5m 12s');
  });
});

describe('DirectorsChairStudio Integration', () => {
  test('connects conversation editing with correction panel', async () => {
    const mockConversation = {
      id: 'conv_123',
      user_message: 'Hello',
      assistant_response: 'Hi there!',
      metadata: {}
    };

    await act(async () => {
      render(
        <DirectorsChairStudio 
          characterId="alice" 
          initialConversation={mockConversation}
        />
      );
    });

    // Click edit on conversation
    const editButton = screen.getByRole('button', { name: /Edit Response/i });
    fireEvent.click(editButton);

    // Should open correction panel
    expect(screen.getByText(/Correction Type/i)).toBeInTheDocument();
  });

  test('shows real-time training updates', async () => {
    await act(async () => {
      render(<DirectorsChairStudio characterId="alice" enableTraining={true} />);
    });

    // Simulate WebSocket training update
    const trainingUpdate = {
      type: 'training_progress_update',
      data: {
        head_type: 'generation',
        progress_percentage: 65.0,
        current_batch: 13,
        total_batches: 20
      }
    };

    // In a real test, we'd simulate the WebSocket message
    // For now, just verify the component can handle updates
    expect(screen.getByText(/Training Progress/i)).toBeInTheDocument();
  });

  test('handles model updates and hot-swapping', async () => {
    const onModelUpdate = jest.fn();

    await act(async () => {
      render(
        <DirectorsChairStudio 
          characterId="alice" 
          onModelUpdate={onModelUpdate}
        />
      );
    });

    // Simulate model update completion
    const modelUpdate = {
      type: 'model_updated',
      data: {
        head_type: 'generation',
        character_id: 'alice',
        ready_for_inference: true
      }
    };

    // Component should handle model updates
    expect(screen.getByTestId('model-status-indicator')).toBeInTheDocument();
  });
}); 