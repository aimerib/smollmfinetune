import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { QuadHeadStreamingPlayer } from '../../components/QuadHeadStreamingPlayer';

// Mock WebSocket
let mockWebSocketInstance: MockWebSocket | null = null;

class MockWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;

  readyState = MockWebSocket.CONNECTING;
  url: string;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    mockWebSocketInstance = this; // Track the current instance
    
    // Simulate connection establishment
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 10);
  }

  send(data: string) {
    // Mock send - we can trigger responses here if needed
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }

  // Helper method to simulate receiving messages
  simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
    }
  }
}

// Replace global WebSocket with mock
(global as any).WebSocket = MockWebSocket;

describe('QuadHeadStreamingPlayer', () => {
  const defaultProps = {
    characterId: 'test-character',
    wsUrl: 'ws://localhost:8000/api/v1/quad-head/stream',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockWebSocketInstance = null;
  });

  test('renders with initial state', () => {
    render(<QuadHeadStreamingPlayer {...defaultProps} />);
    
    // Check main title
    expect(screen.getByText('Quad-Head Multimodal Streaming')).toBeInTheDocument();
    
    // Check stream panels
    expect(screen.getByText('Text Generation')).toBeInTheDocument();
    expect(screen.getByText('Speech Generation')).toBeInTheDocument();
    expect(screen.getByText('Control Signals')).toBeInTheDocument();
    expect(screen.getByText('Memory Updates')).toBeInTheDocument();
    
    // Check placeholder texts
    expect(screen.getByText('Generated text will appear here...')).toBeInTheDocument();
    expect(screen.getByText('Speech visualization will appear here...')).toBeInTheDocument();
    expect(screen.getByText('Control signals will appear here...')).toBeInTheDocument();
    expect(screen.getByText('Memory updates will appear here...')).toBeInTheDocument();
    
    // Check input and button
    expect(screen.getByPlaceholderText('Enter text to generate multimodal response...')).toBeInTheDocument();
    expect(screen.getByText('Generate')).toBeInTheDocument();
  });

  test('shows connecting status initially', async () => {
    render(<QuadHeadStreamingPlayer {...defaultProps} />);
    
    // Should show connecting status initially
    expect(screen.getByText('connecting')).toBeInTheDocument();
    
    // Wait for connection to establish
    await waitFor(() => {
      expect(screen.getByText('connected')).toBeInTheDocument();
    });
  });

  test('handles connection status changes', async () => {
    const onConnectionChange = jest.fn();
    render(
      <QuadHeadStreamingPlayer 
        {...defaultProps} 
        onConnectionChange={onConnectionChange}
      />
    );
    
    // Wait for connection establishment
    await waitFor(() => {
      expect(onConnectionChange).toHaveBeenCalledWith('connecting');
      expect(onConnectionChange).toHaveBeenCalledWith('connected');
    });
  });

  test('enables generate button when connected and input has text', async () => {
    render(<QuadHeadStreamingPlayer {...defaultProps} />);
    
    const input = screen.getByPlaceholderText('Enter text to generate multimodal response...');
    const button = screen.getByText('Generate');
    
    // Initially disabled (connecting)
    expect(button).toBeDisabled();
    
    // Wait for connection
    await waitFor(() => {
      expect(screen.getByText('connected')).toBeInTheDocument();
    });
    
    // Still disabled without text
    expect(button).toBeDisabled();
    
    // Enable with text
    fireEvent.change(input, { target: { value: 'Hello world' } });
    expect(button).not.toBeDisabled();
  });

  test('handles text input and submission', async () => {
    const mockWebSocket = new MockWebSocket('ws://test');
    const sendSpy = jest.spyOn(mockWebSocket, 'send');
    
    render(<QuadHeadStreamingPlayer {...defaultProps} />);
    
    const input = screen.getByPlaceholderText('Enter text to generate multimodal response...');
    const button = screen.getByText('Generate');
    
    // Wait for connection
    await waitFor(() => {
      expect(screen.getByText('connected')).toBeInTheDocument();
    });
    
    // Type text and submit
    fireEvent.change(input, { target: { value: 'Test message' } });
    fireEvent.click(button);
    
    // Input should be cleared
    expect(input).toHaveValue('');
    
    // Button should show generating state
    expect(screen.getByText('Generating...')).toBeInTheDocument();
  });

  test('handles generation completion', async () => {
    const onGenerationComplete = jest.fn();
    render(
      <QuadHeadStreamingPlayer 
        {...defaultProps} 
        onGenerationComplete={onGenerationComplete}
      />
    );
    
    // Wait for connection
    await waitFor(() => {
      expect(screen.getByText('connected')).toBeInTheDocument();
    });
    
    // Start generation
    const input = screen.getByPlaceholderText('Enter text to generate multimodal response...');
    const button = screen.getByText('Generate');
    
    fireEvent.change(input, { target: { value: 'Test' } });
    fireEvent.click(button);
    
    expect(screen.getByText('Generating...')).toBeInTheDocument();
    
    // Simulate completion
    if (mockWebSocketInstance) {
      act(() => {
        mockWebSocketInstance!.simulateMessage({
          type: 'generation_complete',
          timestamp: Date.now(),
          finished: true
        });
      });
    }
    
    // Check generation completed
    await waitFor(() => {
      expect(onGenerationComplete).toHaveBeenCalled();
      expect(screen.getByText('Generate')).toBeInTheDocument(); // Button back to normal
    });
  });

  test('handles errors gracefully', async () => {
    const onError = jest.fn();
    render(
      <QuadHeadStreamingPlayer 
        {...defaultProps} 
        onError={onError}
      />
    );
    
    // Wait for connection
    await waitFor(() => {
      expect(screen.getByText('connected')).toBeInTheDocument();
    });
    
    // Simulate error message
    if (mockWebSocketInstance) {
      act(() => {
        mockWebSocketInstance!.simulateMessage({
          type: 'error',
          error: 'Model generation failed',
          timestamp: Date.now(),
          finished: true
        });
      });
    }
    
    // Check error callback was called
    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith(expect.any(Error));
    });
  });
}); 