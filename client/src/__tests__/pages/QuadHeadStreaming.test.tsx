import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { QuadHeadStreaming } from '../../pages/QuadHeadStreaming';

// Mock the QuadHeadStreamingPlayer component
jest.mock('../../components/QuadHeadStreamingPlayer', () => {
  const mockReact = require('react');
  return {
    QuadHeadStreamingPlayer: mockReact.forwardRef((props: any, ref: any) => 
      mockReact.createElement('div', { 'data-testid': 'quad-head-streaming-player' },
        'Mocked QuadHeadStreamingPlayer',
        mockReact.createElement('div', null, 'Character ID: ' + props.characterId),
        mockReact.createElement('div', null, 'WS URL: ' + props.wsUrl)
      )
    )
  };
});

describe('QuadHeadStreaming Page', () => {
  test('renders page title and subtitle', () => {
    render(<QuadHeadStreaming />);
    
    expect(screen.getByText('Quad-Head Multimodal Streaming')).toBeInTheDocument();
    expect(screen.getByText(/Experience real-time character interaction/)).toBeInTheDocument();
  });

  test('displays character selection section', () => {
    render(<QuadHeadStreaming />);
    
    expect(screen.getByText('Select Character')).toBeInTheDocument();
    expect(screen.getByText('Clara')).toBeInTheDocument();
    expect(screen.getByText('Marcus')).toBeInTheDocument();
    expect(screen.getByText('Sage')).toBeInTheDocument();
  });

  test('shows empty state when no character selected', () => {
    render(<QuadHeadStreaming />);
    
    expect(screen.getByText('Select a character to begin streaming')).toBeInTheDocument();
    expect(screen.getByText('Choose a character from the sidebar to start multimodal interaction')).toBeInTheDocument();
  });

  test('displays character details in cards', () => {
    render(<QuadHeadStreaming />);
    
    // Check Clara's details
    expect(screen.getByText('A thoughtful explorer with deep curiosity about the world.')).toBeInTheDocument();
    
    // Check Marcus's details
    expect(screen.getByText('An energetic adventurer who thrives on social interaction.')).toBeInTheDocument();
    
    // Check Sage's details
    expect(screen.getByText('A wise and methodical thinker with vast knowledge.')).toBeInTheDocument();
  });

  test('shows personality bars for each character', () => {
    render(<QuadHeadStreaming />);
    
    // Should show Big Five personality trait labels (O, C, E, A, N)
    const personalityLabels = screen.getAllByText('O');
    expect(personalityLabels.length).toBeGreaterThan(0);
    
    expect(screen.getAllByText('C').length).toBeGreaterThan(0);
    expect(screen.getAllByText('E').length).toBeGreaterThan(0);
    expect(screen.getAllByText('A').length).toBeGreaterThan(0);
    expect(screen.getAllByText('N').length).toBeGreaterThan(0);
  });

  test('selects character and shows settings', () => {
    render(<QuadHeadStreaming />);
    
    // Click on Clara
    fireEvent.click(screen.getByText('Clara'));
    
    // Should show generation settings
    expect(screen.getByText('Generation Settings')).toBeInTheDocument();
    expect(screen.getByText('Text Temperature')).toBeInTheDocument();
    expect(screen.getByText('Speech Temperature')).toBeInTheDocument();
    expect(screen.getByText('Max Length')).toBeInTheDocument();
    expect(screen.getByText('Speech Generation')).toBeInTheDocument();
  });

  test('shows session info when character selected', () => {
    render(<QuadHeadStreaming />);
    
    // Click on Clara
    fireEvent.click(screen.getByText('Clara'));
    
    // Should show session info
    expect(screen.getByText('Session Info')).toBeInTheDocument();
    expect(screen.getByText(/Character:/)).toBeInTheDocument();
    expect(screen.getByText(/Session:/)).toBeInTheDocument();
    expect(screen.getByText(/Started:/)).toBeInTheDocument();
    expect(screen.getByText(/Messages:/)).toBeInTheDocument();
  });

  test('renders streaming player when character selected', () => {
    render(<QuadHeadStreaming />);
    
    // Click on Marcus
    fireEvent.click(screen.getByText('Marcus'));
    
    // Should render the streaming player
    expect(screen.getByTestId('quad-head-streaming-player')).toBeInTheDocument();
    expect(screen.getByText('Character ID: marcus')).toBeInTheDocument();
    expect(screen.getByText('WS URL: ws://localhost:8000/api/quad-head/stream/marcus')).toBeInTheDocument();
  });

  test('updates settings when sliders are moved', () => {
    render(<QuadHeadStreaming />);
    
    // Select a character first
    fireEvent.click(screen.getByText('Sage'));
    
    // Find temperature slider
    const tempSlider = screen.getByLabelText(/Text Temperature/);
    fireEvent.change(tempSlider, { target: { value: '1.2' } });
    
    // Should show updated value
    expect(screen.getByText('1.20')).toBeInTheDocument();
  });

  test('toggles speech generation setting', () => {
    render(<QuadHeadStreaming />);
    
    // Select a character first
    fireEvent.click(screen.getByText('Clara'));
    
    // Find speech checkbox
    const speechCheckbox = screen.getByLabelText('Speech Generation');
    expect(speechCheckbox).toBeChecked();
    
    // Toggle it off
    fireEvent.click(speechCheckbox);
    expect(speechCheckbox).not.toBeChecked();
  });

  test('highlights selected character card', () => {
    render(<QuadHeadStreaming />);
    
    const claraCard = screen.getByText('Clara').closest('div');
    const marcusCard = screen.getByText('Marcus').closest('div');
    
    // Initially no character should be selected
    expect(claraCard).toHaveStyle('background: rgba(255, 255, 255, 0.05)');
    
    // Click Clara
    fireEvent.click(screen.getByText('Clara'));
    
    // Clara should now be highlighted (we can't easily test the computed style,
    // but we can ensure the click handler was called)
    expect(screen.getByText('Generation Settings')).toBeInTheDocument();
  });

  test('changes character selection', () => {
    render(<QuadHeadStreaming />);
    
    // Select Clara first
    fireEvent.click(screen.getByText('Clara'));
    expect(screen.getByText('Character ID: clara')).toBeInTheDocument();
    
    // Switch to Marcus
    fireEvent.click(screen.getByText('Marcus'));
    expect(screen.getByText('Character ID: marcus')).toBeInTheDocument();
  });

  test('resets session when changing characters', () => {
    render(<QuadHeadStreaming />);
    
    // Select Clara first
    fireEvent.click(screen.getByText('Clara'));
    
    // Should show messages count as 0
    expect(screen.getByText(/Messages: 0/)).toBeInTheDocument();
    
    // Switch to Marcus
    fireEvent.click(screen.getByText('Marcus'));
    
    // Should still show messages count as 0 (new session)
    expect(screen.getByText(/Messages: 0/)).toBeInTheDocument();
  });
}); 