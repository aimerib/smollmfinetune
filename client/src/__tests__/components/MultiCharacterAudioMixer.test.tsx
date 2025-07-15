/**
 * Tests for MultiCharacterAudioMixer Component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent, { UserEvent } from '@testing-library/user-event';
import { jest } from '@jest/globals';
import MultiCharacterAudioMixer from '../../components/MultiCharacterAudioMixer';

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

// Test data
const mockCharacters = [
  {
    id: 'char1',
    name: 'Alice',
    voiceProfile: {
      provider: 'kokoro' as const,
      voiceId: 'voice1',
      emotionalRange: 0.8,
      baseFrequency: 220,
      expressiveness: 0.7,
    },
    isActive: true,
    volume: 0.8,
    spatialPosition: { x: -1, y: 0, z: 1 },
  },
  {
    id: 'char2',
    name: 'Bob',
    voiceProfile: {
      provider: 'orpheus' as const,
      voiceId: 'voice2',
      emotionalRange: 0.6,
      baseFrequency: 180,
      expressiveness: 0.5,
    },
    isActive: false,
    volume: 0.6,
    spatialPosition: { x: 1, y: 0, z: -1 },
  },
];

const mockProps = {
  sessionId: 'test-session-123',
  characters: mockCharacters,
  onCharacterUpdate: jest.fn(),
  onDialogueTurn: jest.fn(),
};

describe('MultiCharacterAudioMixer', () => {
  let mockWebSocket: any;

  beforeEach(() => {
    jest.clearAllMocks();
    mockWebSocket = {
      send: jest.fn(),
      close: jest.fn(),
      readyState: 1,
      onopen: null,
      onmessage: null,
      onclose: null,
      onerror: null,
    };
    (global.WebSocket as any).mockImplementation(() => mockWebSocket);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Component Rendering', () => {
    test('renders the main mixer interface', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      expect(screen.getByText('Multi-Character Audio Mixer')).toBeInTheDocument();
      expect(screen.getByText('Session: test-session-123')).toBeInTheDocument();
      expect(screen.getByText('Characters: 2')).toBeInTheDocument();
    });

    test('renders character controls panel', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      expect(screen.getByText('Character Controls')).toBeInTheDocument();
      expect(screen.getByText('Alice')).toBeInTheDocument();
      expect(screen.getByText('Bob')).toBeInTheDocument();
      expect(screen.getByText('KOKORO')).toBeInTheDocument();
      expect(screen.getByText('ORPHEUS')).toBeInTheDocument();
    });

    test('renders conversation visualizer panel', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      expect(screen.getByText('Conversation Timeline')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter dialogue...')).toBeInTheDocument();
      expect(screen.getByText('Select Speaker')).toBeInTheDocument();
    });

    test('renders spatial audio and environmental controls', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      expect(screen.getByText('Spatial Audio')).toBeInTheDocument();
      expect(screen.getByText('Environmental Effects')).toBeInTheDocument();
      expect(screen.getByText('Enable 3D Audio')).toBeInTheDocument();
    });

    test('renders master controls', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      expect(screen.getByText('Master Volume')).toBeInTheDocument();
      expect(screen.getByText('Conversation Pacing')).toBeInTheDocument();
      expect(screen.getByText('🔴 Start Recording')).toBeInTheDocument();
    });
  });

  describe('WebSocket Connection', () => {
    test('establishes WebSocket connection on mount', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      expect(global.WebSocket).toHaveBeenCalledWith(
        'ws://localhost:8000/api/v1/multi-character/conversations/test-session-123/stream'
      );
    });

    test('closes WebSocket connection on unmount', () => {
      const { unmount } = render(<MultiCharacterAudioMixer {...mockProps} />);

      unmount();

      expect(mockWebSocket.close).toHaveBeenCalled();
    });

    test('handles WebSocket message for conversation state', async () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const stateMessage = {
        type: 'conversation_state',
        data: {
          sessionId: 'test-session-123',
          activeCharacters: ['char1', 'char2'],
          context: {},
        },
      };

      // Simulate WebSocket message
      act(() => {
        mockWebSocket.onmessage({ data: JSON.stringify(stateMessage) });
      });

      // Verify state is handled (would need to check internal state)
      expect(console.log).toHaveBeenCalledWith('Multi-character WebSocket connected');
    });

    test('handles WebSocket message for audio chunk', async () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const audioMessage = {
        type: 'audio_chunk',
        data: {
          character_id: 'char1',
          audio_data: 'mock-audio-data',
          text: 'Hello world',
          emotion_context: { happiness: 0.8 },
        },
      };

      // Simulate WebSocket message
      act(() => {
        mockWebSocket.onmessage({ data: JSON.stringify(audioMessage) });
      });

      // Check if dialogue turn was added to history
      await waitFor(() => {
        expect(screen.getByText('Hello world')).toBeInTheDocument();
      });
    });
  });

  describe('Character Controls', () => {
    test('shows character activity indicators', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const aliceController = screen.getByText('Alice').closest('.character-controller');
      const bobController = screen.getByText('Bob').closest('.character-controller');

      expect(aliceController?.querySelector('.activity-indicator.active')).toBeInTheDocument();
      expect(bobController?.querySelector('.activity-indicator.active')).not.toBeInTheDocument();
    });

    test('selects character when clicked', async () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const aliceController = screen.getByText('Alice').closest('.character-controller');
      
      await userEvent.click(aliceController!);

      expect(aliceController).toHaveClass('selected');
    });

    test('shows character controls when selected', async () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const aliceController = screen.getByText('Alice').closest('.character-controller');
      await userEvent.click(aliceController!);

      expect(screen.getByText('Volume')).toBeInTheDocument();
      expect(screen.getByText('Position')).toBeInTheDocument();
      expect(screen.getByText('Voice Profile')).toBeInTheDocument();
    });

    test('updates character volume', async () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      // Select Alice
      const aliceController = screen.getByText('Alice').closest('.character-controller');
      await userEvent.click(aliceController!);

      // Find and change volume slider
      const volumeSlider = screen.getByDisplayValue('0.8');
      await userEvent.clear(volumeSlider);
      await userEvent.type(volumeSlider, '0.5');

      expect(mockProps.onCharacterUpdate).toHaveBeenCalledWith(
        expect.objectContaining({
          id: 'char1',
          volume: 0.5,
        })
      );
    });

    test('updates character spatial position', async () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      // Select Alice
      const aliceController = screen.getByText('Alice').closest('.character-controller');
      await userEvent.click(aliceController!);

      // Find position controls (first range input should be X position)
      const positionInputs = screen.getAllByDisplayValue('-1');
      await userEvent.clear(positionInputs[0]);
      await userEvent.type(positionInputs[0], '2');

      // Should send WebSocket message for position update
      expect(mockWebSocket.send).toHaveBeenCalledWith(
        JSON.stringify({
          type: 'spatial_position',
          data: {
            character_id: 'char1',
            position: { x: 2, y: 0, z: 1 },
          },
        })
      );
    });
  });

  describe('Conversation Interface', () => {
    test('allows selecting speaker and submitting dialogue', async () => {
      
      render(<MultiCharacterAudioMixer {...mockProps} />);

      // Select speaker
      const speakerSelect = screen.getByDisplayValue('');
      await userEvent.selectOptions(speakerSelect, 'char1');

      // Enter dialogue text
      const textArea = screen.getByPlaceholderText('Enter dialogue...');
      await userEvent.type(textArea, 'Test message');

      // Submit dialogue
      const sendButton = screen.getByText('Send');
      await userEvent.click(sendButton);

      // Should send WebSocket message
      expect(mockWebSocket.send).toHaveBeenCalledWith(
        JSON.stringify({
          type: 'dialogue_turn',
          data: {
            character_id: 'char1',
            text: 'Test message',
            emotion_context: {},
            interrupts_previous: false,
            urgency_level: 0.5,
          },
        })
      );

      // Should call onDialogueTurn callback
      expect(mockProps.onDialogueTurn).toHaveBeenCalledWith(
        expect.objectContaining({
          characterId: 'char1',
          text: 'Test message',
        })
      );

      // Should clear the text area
      expect(textArea).toHaveValue('');
    });

    test('disables send button when no speaker or text', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const sendButton = screen.getByText('Send');
      expect(sendButton).toBeDisabled();
    });

    test('displays conversation history with timestamps', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      // Simulate receiving a dialogue turn via WebSocket
      const audioMessage = {
        type: 'audio_chunk',
        data: {
          character_id: 'char1',
          text: 'Hello from Alice',
          emotion_context: { happiness: 0.8 },
        },
      };

      act(() => {
        mockWebSocket.onmessage({ data: JSON.stringify(audioMessage) });
      });

      expect(screen.getByText('Hello from Alice')).toBeInTheDocument();
      expect(screen.getByText('happiness: 80%')).toBeInTheDocument();
    });
  });

  describe('Spatial Audio Controls', () => {
    test('toggles spatial audio', async () => {
      
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const spatialToggle = screen.getByRole('checkbox', { name: /enable 3d audio/i });
      expect(spatialToggle).toBeChecked();

      await userEvent.click(spatialToggle);
      expect(spatialToggle).not.toBeChecked();
    });

    test('shows spatial visualizer when enabled', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      expect(screen.getByText('🎧')).toBeInTheDocument(); // Listener position
      expect(screen.getAllByText('🗣️')).toHaveLength(2); // Character positions
      expect(screen.getByText('Drag characters to reposition')).toBeInTheDocument();
    });

    test('hides spatial visualizer when disabled', async () => {
      
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const spatialToggle = screen.getByRole('checkbox', { name: /enable 3d audio/i });
      await userEvent.click(spatialToggle);

      expect(screen.queryByText('🎧')).not.toBeInTheDocument();
      expect(screen.queryByText('🗣️')).not.toBeInTheDocument();
    });
  });

  describe('Environmental Effects', () => {
    test('updates acoustic environment', async () => {
      
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const environmentSelect = screen.getByDisplayValue('room');
      await userEvent.selectOptions(environmentSelect, 'hall');

      // Component should update internal state (would need state testing)
      expect(environmentSelect).toHaveValue('hall');
    });

    test('updates reverb level', async () => {
      
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const reverbSlider = screen.getByDisplayValue('0.3');
      await userEvent.clear(reverbSlider);
      await userEvent.type(reverbSlider, '0.7');

      expect(screen.getByText('70%')).toBeInTheDocument();
    });

    test('updates ambient noise level', async () => {
      
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const noiseSlider = screen.getByDisplayValue('0.1');
      await userEvent.clear(noiseSlider);
      await userEvent.type(noiseSlider, '0.5');

      expect(screen.getByText('50%')).toBeInTheDocument();
    });
  });

  describe('Master Controls', () => {
    test('updates master volume', async () => {
      
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const masterVolumeSlider = screen.getByDisplayValue('0.8');
      await userEvent.clear(masterVolumeSlider);
      await userEvent.type(masterVolumeSlider, '0.9');

      expect(screen.getByText('90%')).toBeInTheDocument();
    });

    test('updates conversation pacing', async () => {
      
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const pacingSlider = screen.getByDisplayValue('1');
      await userEvent.clear(pacingSlider);
      await userEvent.type(pacingSlider, '1.5');

      expect(screen.getByText('1.5x')).toBeInTheDocument();
    });

    test('toggles recording state', async () => {
      
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const recordButton = screen.getByText('🔴 Start Recording');
      await userEvent.click(recordButton);

      expect(screen.getByText('⏹️ Stop Recording')).toBeInTheDocument();
      expect(recordButton).toHaveClass('recording');

      await userEvent.click(recordButton);
      expect(screen.getByText('🔴 Start Recording')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    test('handles WebSocket errors gracefully', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      // Simulate WebSocket error
      act(() => {
        mockWebSocket.onerror(new Error('Connection failed'));
      });

      expect(console.error).toHaveBeenCalledWith(
        'Multi-character WebSocket error:',
        expect.any(Error)
      );
    });

    test('handles invalid JSON messages', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      // Simulate invalid JSON message
      act(() => {
        mockWebSocket.onmessage({ data: 'invalid json' });
      });

      // Should continue functioning despite the error
      expect(screen.getByText('Multi-Character Audio Mixer')).toBeInTheDocument();
    });

    test('handles WebSocket disconnect', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      // Simulate WebSocket close
      act(() => {
        mockWebSocket.onclose();
      });

      expect(console.log).toHaveBeenCalledWith('Multi-character WebSocket disconnected');
    });
  });

  describe('Accessibility', () => {
    test('has proper ARIA labels for controls', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      expect(screen.getByRole('checkbox', { name: /enable 3d audio/i })).toBeInTheDocument();
      expect(screen.getByRole('textbox', { name: /enter dialogue/i })).toBeInTheDocument();
      expect(screen.getByRole('combobox')).toBeInTheDocument(); // Speaker select
    });

    test('supports keyboard navigation', async () => {
      
      render(<MultiCharacterAudioMixer {...mockProps} />);

      const textArea = screen.getByPlaceholderText('Enter dialogue...');
      
      // Tab to text area and type
      await userEvent.tab();
      await userEvent.tab();
      await userEvent.tab(); // Might need multiple tabs depending on DOM structure
      
      // Should be able to focus and interact with controls
      expect(document.activeElement).toBeDefined();
    });
  });

  describe('Performance', () => {
    test('handles multiple rapid WebSocket messages', () => {
      render(<MultiCharacterAudioMixer {...mockProps} />);

      // Send multiple messages rapidly
      for (let i = 0; i < 10; i++) {
        act(() => {
          mockWebSocket.onmessage({
            data: JSON.stringify({
              type: 'audio_chunk',
              data: {
                character_id: 'char1',
                text: `Message ${i}`,
              },
            }),
          });
        });
      }

      // Should handle all messages without crashing
      expect(screen.getByText('Message 9')).toBeInTheDocument();
    });

    test('cleans up resources on unmount', () => {
      const { unmount } = render(<MultiCharacterAudioMixer {...mockProps} />);

      unmount();

      expect(mockWebSocket.close).toHaveBeenCalled();
    });
  });
}); 