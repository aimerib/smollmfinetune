import React from 'react';
import { render, screen, fireEvent, waitFor, act, within } from '@testing-library/react';
import { RelationshipTimeline } from '../../../components/DirectorsView/RelationshipTimeline';
import websocketService from '../../../services/websocketService';

// Mock WebSocket service
jest.mock('../../../services/websocketService');

describe('RelationshipTimeline', () => {
  const mockWebsocketService = websocketService as jest.Mocked<typeof websocketService>;
  
  const mockEvents = [
    {
      timestamp: '2024-01-20T10:30:00Z',
      speaker_id: 'tom',
      target_id: 'clara',
      interaction_type: 'conversation',
      affinity_change: 0.1,
      emotional_impact: ['happy', 'grateful'],
      memory_significance: 0.7,
      narrative_context: 'Tom helped Clara with a difficult task'
    },
    {
      timestamp: '2024-01-20T09:15:00Z',
      speaker_id: 'clara',
      target_id: 'tom',
      interaction_type: 'support',
      affinity_change: 0.15,
      emotional_impact: ['trusting', 'content'],
      memory_significance: 0.8,
      narrative_context: 'Clara confided in Tom about her worries'
    },
    {
      timestamp: '2024-01-20T08:00:00Z',
      speaker_id: 'tom',
      target_id: 'clara',
      interaction_type: 'greeting',
      affinity_change: 0.05,
      emotional_impact: ['friendly'],
      memory_significance: 0.3,
      narrative_context: 'Morning greeting'
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    mockWebsocketService.onRelationshipEvent = jest.fn();
  });

  test('displays relationship events in chronological order', () => {
    render(<RelationshipTimeline agentPair={['tom', 'clara']} initialEvents={mockEvents} />);

    const events = screen.getAllByTestId(/timeline-event-/);
    expect(events).toHaveLength(3);

    // Check events are in reverse chronological order (newest first)
    // Since timezone conversion may affect exact times, check the narrative content instead
    const eventTexts = screen.getAllByTestId(/timeline-event-/).map(el => el.textContent);
    
    expect(eventTexts[0]).toContain('Tom helped Clara with a difficult task');
    expect(eventTexts[1]).toContain('Clara confided in Tom about her worries');
    expect(eventTexts[2]).toContain('Morning greeting');
  });

  test('shows event details including affinity changes', () => {
    render(<RelationshipTimeline agentPair={['tom', 'clara']} initialEvents={mockEvents} />);

    // Check first event details
    const firstEvent = screen.getByTestId('timeline-event-0');
    expect(within(firstEvent).getByText(/tom/)).toBeInTheDocument();
    expect(within(firstEvent).getByText(/clara/)).toBeInTheDocument();
    expect(screen.getByText('+0.10')).toBeInTheDocument();
    expect(screen.getByText('Tom helped Clara with a difficult task')).toBeInTheDocument();
  });

  test('displays emotional impact for each event', () => {
    render(<RelationshipTimeline agentPair={['tom', 'clara']} initialEvents={mockEvents} />);

    // Check emotional impacts are shown
    expect(screen.getByText('happy')).toBeInTheDocument();
    expect(screen.getByText('grateful')).toBeInTheDocument();
    expect(screen.getByText('trusting')).toBeInTheDocument();
    expect(screen.getByText('content')).toBeInTheDocument();
  });

  test('updates in real-time when new relationship events occur', async () => {
    let relationshipEventCallback: (event: any) => void = () => {};
    
    mockWebsocketService.onRelationshipEvent.mockImplementation((callback) => {
      relationshipEventCallback = callback;
      return jest.fn(); // Return unsubscribe function
    });

    render(<RelationshipTimeline agentPair={['tom', 'clara']} initialEvents={mockEvents} />);

    // Simulate new WebSocket event
    const newEvent = {
      timestamp: '2024-01-20T11:00:00Z',
      speaker_id: 'clara',
      target_id: 'tom',
      interaction_type: 'thanks',
      affinity_change: 0.2,
      emotional_impact: ['grateful', 'happy'],
      memory_significance: 0.9,
      narrative_context: 'Clara thanked Tom for his help'
    };

    act(() => {
      relationshipEventCallback({
        source: 'clara',
        target: 'tom',
        ...newEvent
      });
    });

    await waitFor(() => {
      expect(screen.getByText('Clara thanked Tom for his help')).toBeInTheDocument();
      expect(screen.getByText('+0.20')).toBeInTheDocument();
    });
  });

  test('filters events for relevant agent pair only', async () => {
    let relationshipEventCallback: (event: any) => void = () => {};
    
    mockWebsocketService.onRelationshipEvent.mockImplementation((callback) => {
      relationshipEventCallback = callback;
      return jest.fn();
    });

    render(<RelationshipTimeline agentPair={['tom', 'clara']} initialEvents={[]} />);

    // Simulate irrelevant event (different agents)
    const irrelevantEvent = {
      source: 'alice',
      target: 'bob',
      timestamp: '2024-01-20T11:00:00Z',
      speaker_id: 'alice',
      target_id: 'bob',
      interaction_type: 'conversation',
      affinity_change: 0.1,
      emotional_impact: ['neutral'],
      memory_significance: 0.5,
      narrative_context: 'Alice talked to Bob'
    };

    act(() => {
      relationshipEventCallback(irrelevantEvent);
    });

    // Event should not appear
    await waitFor(() => {
      expect(screen.queryByText('Alice talked to Bob')).not.toBeInTheDocument();
    });
  });

  test('limits displayed events to prevent overflow', () => {
    // Create 60 events
    const manyEvents = Array.from({ length: 60 }, (_, i) => {
      const date = new Date('2024-01-20T00:00:00Z');
      date.setMinutes(i * 5); // Increment by 5 minutes for each event
      return {
        timestamp: date.toISOString(),
        speaker_id: i % 2 === 0 ? 'tom' : 'clara',
        target_id: i % 2 === 0 ? 'clara' : 'tom',
        interaction_type: 'conversation',
        affinity_change: 0.01,
        emotional_impact: ['neutral'],
        memory_significance: 0.5,
        narrative_context: `Event ${i}`
      };
    });

    render(<RelationshipTimeline agentPair={['tom', 'clara']} initialEvents={manyEvents} />);

    const events = screen.getAllByTestId(/timeline-event-/);
    // Should only show last 50 events
    expect(events).toHaveLength(50);
  });

  test('indicates memory significance visually', () => {
    render(<RelationshipTimeline agentPair={['tom', 'clara']} initialEvents={mockEvents} />);

    // High significance event should have special styling
    const highSignificanceEvent = screen.getByTestId('timeline-event-1'); // Second event has 0.8 significance
    expect(highSignificanceEvent).toHaveClass('high-significance');

    // Low significance event should have different styling
    const lowSignificanceEvent = screen.getByTestId('timeline-event-2'); // Third event has 0.3 significance
    expect(lowSignificanceEvent).toHaveClass('low-significance');
  });

  test('shows interaction type icons', () => {
    render(<RelationshipTimeline agentPair={['tom', 'clara']} initialEvents={mockEvents} />);

    expect(screen.getByTestId('icon-conversation')).toBeInTheDocument();
    expect(screen.getByTestId('icon-support')).toBeInTheDocument();
    expect(screen.getByTestId('icon-greeting')).toBeInTheDocument();
  });

  test('displays affinity changes with appropriate colors', () => {
    const eventsWithNegativeChange = [
      ...mockEvents,
      {
        timestamp: '2024-01-20T11:00:00Z',
        speaker_id: 'tom',
        target_id: 'clara',
        interaction_type: 'conflict',
        affinity_change: -0.3,
        emotional_impact: ['angry', 'frustrated'],
        memory_significance: 0.9,
        narrative_context: 'Tom and Clara had a disagreement'
      }
    ];

    render(<RelationshipTimeline agentPair={['tom', 'clara']} initialEvents={eventsWithNegativeChange} />);

    const negativeChange = screen.getByText('-0.30');
    expect(negativeChange).toHaveClass('negative-change');
    
    const positiveChange = screen.getByText('+0.10');
    expect(positiveChange).toHaveClass('positive-change');
  });

  test('allows expanding event details for more information', () => {
    render(<RelationshipTimeline agentPair={['tom', 'clara']} initialEvents={mockEvents} />);

    const firstEvent = screen.getByTestId('timeline-event-0');
    const expandButton = within(firstEvent).getByRole('button', { name: /expand/i });
    
    fireEvent.click(expandButton);

    // Expanded details should appear
    expect(screen.getByText(/Memory Formation/)).toBeInTheDocument();
    expect(screen.getByText(/Significance: 0.7/)).toBeInTheDocument();
  });
}); 