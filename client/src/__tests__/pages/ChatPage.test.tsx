import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import ChatPage from '../../pages/ChatPage';
import { chatService } from '../../utils/chatService';

// Mock the chatService
jest.mock('../../utils/chatService');

// Mock useParams and useSearchParams
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useParams: () => ({ sessionId: null }),
  useSearchParams: () => [new URLSearchParams({ character: 'alice' })],
}));

// Mock scrollIntoView globally
beforeAll(() => {
  Element.prototype.scrollIntoView = jest.fn();
});

describe('ChatPage Component', () => {
  const mockChatService = chatService as jest.Mocked<typeof chatService>;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Setup default mocks
    mockChatService.startSession = jest.fn().mockResolvedValue('test-session-123');
    mockChatService.sendMessage = jest.fn().mockResolvedValue({
      generation_text: 'Hello! How can I help you today?',
      control_tokens: [],
      memory_vector: [],
      memory_metadata: {},
      session_id: 'test-session-123',
    });
  });

  const renderChatPage = async () => {
    const view = render(
      <BrowserRouter>
        <ChatPage />
      </BrowserRouter>
    );
    
    // Wait for initialization to complete
    await waitFor(() => {
      expect(screen.queryByText(/Connecting with/i)).not.toBeInTheDocument();
    });
    
    return view;
  };

  test('renders chat page with character header', async () => {
    await renderChatPage();

    expect(screen.getByText('Alice')).toBeInTheDocument();
  });

  test('displays loading state initially', () => {
    render(
      <BrowserRouter>
        <ChatPage />
      </BrowserRouter>
    );

    expect(screen.getByText(/Connecting with Alice/i)).toBeInTheDocument();
  });

  test('handles message sending', async () => {
    await renderChatPage();

    const input = screen.getByPlaceholderText(/Message Alice/i);
    // Find send button by looking for button with SVG
    const buttons = screen.getAllByRole('button');
    const sendButton = buttons[buttons.length - 1]; // Last button is send button

    // Type a message
    fireEvent.change(input, { target: { value: 'Hello AI!' } });
    expect(input).toHaveValue('Hello AI!');

    // Send the message
    fireEvent.click(sendButton);

    // Check that the service was called
    await waitFor(() => {
      expect(mockChatService.sendMessage).toHaveBeenCalledWith(
        'test-session-123',
        'alice',
        'Hello AI!'
      );
    });

    // Check that the response appears
    await waitFor(() => {
      expect(screen.getByText('Hello! How can I help you today?')).toBeInTheDocument();
    });
  });

  test('disables input while message is being sent', async () => {
    await renderChatPage();

    const input = screen.getByPlaceholderText(/Message Alice/i);
    const buttons = screen.getAllByRole('button');
    const sendButton = buttons[buttons.length - 1];

    fireEvent.change(input, { target: { value: 'Test message' } });
    fireEvent.click(sendButton);

    // Input should be disabled (button disabled state is handled internally)
    expect(input).toBeDisabled();

    // Wait for response
    await waitFor(() => {
      expect(input).not.toBeDisabled();
    });
  });

  test('handles Enter key press to send message', async () => {
    await renderChatPage();

    const input = screen.getByPlaceholderText(/Message Alice/i);

    fireEvent.change(input, { target: { value: 'Enter key test' } });
    fireEvent.keyPress(input, { key: 'Enter', code: 'Enter', charCode: 13 });

    await waitFor(() => {
      expect(mockChatService.sendMessage).toHaveBeenCalledWith(
        'test-session-123',
        'alice',
        'Enter key test'
      );
    });
  });

  test('handles API error gracefully', async () => {
    mockChatService.sendMessage.mockRejectedValueOnce(new Error('Network error'));

    await renderChatPage();

    const input = screen.getByPlaceholderText(/Message Alice/i);
    const buttons = screen.getAllByRole('button');
    const sendButton = buttons[buttons.length - 1];

    fireEvent.change(input, { target: { value: 'Error test' } });
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(screen.getByText(/Sorry, I encountered an error/i)).toBeInTheDocument();
    });
  });

  test('handles session initialization error', async () => {
    // Remove the mock for this specific test  
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    mockChatService.startSession.mockRejectedValueOnce(new Error('Failed to start session'));

    render(
      <BrowserRouter>
        <ChatPage />
      </BrowserRouter>
    );

    // Since we're testing the actual component, it will log the error
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('Failed to initialize session:', expect.any(Error));
    });
    
    consoleSpy.mockRestore();
  });

  test('scrolls to bottom when new messages arrive', async () => {
    const scrollIntoViewMock = Element.prototype.scrollIntoView as jest.Mock;
    
    await renderChatPage();

    const input = screen.getByPlaceholderText(/Message Alice/i);
    const buttons = screen.getAllByRole('button');
    const sendButton = buttons[buttons.length - 1];

    fireEvent.change(input, { target: { value: 'Scroll test' } });
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(scrollIntoViewMock).toHaveBeenCalled();
    });
  });

  test('displays character emotions in messages', async () => {
    mockChatService.sendMessage.mockResolvedValueOnce({
      generation_text: 'I\'m feeling happy!',
      control_tokens: [{ token: '<emotion_happy>', confidence: 0.9 }],
      memory_vector: [],
      memory_metadata: {
        importance: 0.7,
        emotional_valence: 0.8,
        context_relevance: 0.9,
        decay_rate: 0.1
      },
      session_id: 'test-session-123',
      latency_ms: 150,
    });

    await renderChatPage();

    const input = screen.getByPlaceholderText(/Message Alice/i);
    fireEvent.change(input, { target: { value: 'How are you?' } });
    fireEvent.keyPress(input, { key: 'Enter', code: 'Enter', charCode: 13 });

    await waitFor(() => {
      expect(screen.getByText('I\'m feeling happy!')).toBeInTheDocument();
    });
  });

  test('shows typing indicator when waiting for response', async () => {
    // Create a delayed response
    let resolveMessage: (value: any) => void;
    mockChatService.sendMessage.mockImplementationOnce(() => 
      new Promise((resolve) => {
        resolveMessage = resolve;
      })
    );

    await renderChatPage();

    const input = screen.getByPlaceholderText(/Message Alice/i);
    const buttons = screen.getAllByRole('button');
    const sendButton = buttons[buttons.length - 1];

    fireEvent.change(input, { target: { value: 'Test typing' } });
    fireEvent.click(sendButton);

    // Should show typing indicator
    expect(screen.getByText(/Alice is typing/i)).toBeInTheDocument();

    // Resolve the promise
    resolveMessage!({
      generation_text: 'Response!',
      control_tokens: [],
      memory_vector: [],
      memory_metadata: {
        importance: 0.5,
        emotional_valence: 0.5,
        context_relevance: 0.5,
        decay_rate: 0.1
      },
      session_id: 'test-session-123',
      latency_ms: 100,
    });

    // Wait for response to complete
    await waitFor(() => {
      expect(screen.queryByText(/Alice is typing/i)).not.toBeInTheDocument();
    });
  });
}); 