import React from 'react';
import { render, screen } from '@testing-library/react';
import ChatMessage from '../../components/ChatMessage';

describe('ChatMessage Component', () => {
  const mockUserMessage = {
    id: 'msg-1',
    role: 'user' as const,
    content: 'Hello, how are you?',
    timestamp: new Date('2024-01-17T10:00:00Z'),
  };

  const mockCharacterMessage = {
    id: 'msg-2',
    role: 'assistant' as const,
    content: 'I am doing well, thank you!',
    emotion: 'happy',
    timestamp: new Date('2024-01-17T10:01:00Z'),
  };

  test('renders user message correctly', () => {
    render(
      <ChatMessage
        message={mockUserMessage}
        characterId="alice"
      />
    );

    const messageElement = screen.getByText(mockUserMessage.content);
    expect(messageElement).toBeInTheDocument();
  });

  test('renders character message correctly', () => {
    render(
      <ChatMessage
        message={mockCharacterMessage}
        characterId="alice"
      />
    );

    const messageElement = screen.getByText(mockCharacterMessage.content);
    expect(messageElement).toBeInTheDocument();
  });

  test('applies correct styling for user messages', () => {
    const { container } = render(
      <ChatMessage
        message={mockUserMessage}
        characterId="alice"
      />
    );

    // Check if the message container has user-specific styling
    const messageContainer = container.firstChild;
    expect(messageContainer).toHaveStyle('flex-direction: row-reverse');
  });

  test('applies correct styling for character messages', () => {
    const { container } = render(
      <ChatMessage
        message={mockCharacterMessage}
        characterId="alice"
      />
    );

    // Check if the message container has character-specific styling
    const messageContainer = container.firstChild;
    expect(messageContainer).toHaveStyle('flex-direction: row');
  });

  test('renders correct avatar for different characters', () => {
    const { rerender } = render(
      <ChatMessage
        message={mockCharacterMessage}
        characterId="alice"
      />
    );

    // Alice should have fairy emoji
    expect(screen.getByText('🧚‍♀️')).toBeInTheDocument();

    // Max should have robot emoji
    rerender(
      <ChatMessage
        message={mockCharacterMessage}
        characterId="max"
      />
    );
    expect(screen.getByText('🤖')).toBeInTheDocument();

    // Luna should have moon emoji
    rerender(
      <ChatMessage
        message={mockCharacterMessage}
        characterId="luna"
      />
    );
    expect(screen.getByText('🌙')).toBeInTheDocument();
  });

  test('renders user avatar correctly', () => {
    render(
      <ChatMessage
        message={mockUserMessage}
        characterId="alice"
      />
    );

    // User should have person emoji
    expect(screen.getByText('👤')).toBeInTheDocument();
  });

  test('handles long messages gracefully', () => {
    const longContent = 'Lorem ipsum '.repeat(50).trim(); // Remove trailing space
    const longMessage = {
      ...mockCharacterMessage,
      content: longContent,
    };
    
    render(
      <ChatMessage
        message={longMessage}
        characterId="alice"
      />
    );

    const messageElement = screen.getByText(longContent);
    expect(messageElement).toBeInTheDocument();
  });

  test('displays timestamp correctly', () => {
    const testDate = new Date('2024-01-17T10:00:00Z');
    const mockMessage = {
      ...mockUserMessage,
      timestamp: testDate,
    };
    
    render(
      <ChatMessage
        message={mockMessage}
        characterId="alice"
      />
    );

    // Check if timestamp is formatted and displayed
    // Format using the same logic as the component
    const expectedTime = testDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    expect(screen.getByText(expectedTime)).toBeInTheDocument();
  });

  test('forwards ref correctly', () => {
    const ref = React.createRef<HTMLDivElement>();
    render(
      <ChatMessage
        ref={ref}
        message={mockCharacterMessage}
        characterId="alice"
      />
    );

    expect(ref.current).toBeInstanceOf(HTMLDivElement);
  });
}); 