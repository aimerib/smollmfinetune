import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { RelationshipPanel, RelationshipMetrics } from '../../../components/DirectorsView/RelationshipPanel';
import { RelationshipEdge } from '../../../types/relationships';

describe('RelationshipPanel', () => {
  const mockRelationshipData: RelationshipEdge = {
    source: 'tom',
    target: 'clara',
    affinity: 0.7,
    status: 'Friend',
    emotional_history: ['happy', 'grateful', 'trusting', 'content', 'excited'],
    memory_significance: 0.8,
    interaction_count: 25,
    last_interaction: '2024-01-20T10:30:00Z'
  };

  const mockHistoryEvents = [
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
    }
  ];

  test('displays selected relationship details', () => {
    render(
      <RelationshipPanel 
        selectedRelationship={mockRelationshipData}
        relationshipHistory={mockHistoryEvents}
      />
    );

    expect(screen.getByText(/tom/)).toBeInTheDocument();
    expect(screen.getByText(/clara/)).toBeInTheDocument();
    expect(screen.getByText('Friend')).toBeInTheDocument();
    expect(screen.getByText('Affinity: 0.70')).toBeInTheDocument();
    expect(screen.getByText('25 interactions')).toBeInTheDocument();
  });

  test('shows emotional timeline for relationship', () => {
    render(
      <RelationshipPanel 
        selectedRelationship={mockRelationshipData}
        relationshipHistory={mockHistoryEvents}
      />
    );

    // Check emotional history is displayed
    const emotionalTimeline = screen.getByTestId('emotional-timeline');
    expect(emotionalTimeline).toBeInTheDocument();
    
    // Recent emotions should be visible
    expect(screen.getByText('happy')).toBeInTheDocument();
    expect(screen.getByText('grateful')).toBeInTheDocument();
    expect(screen.getByText('trusting')).toBeInTheDocument();
  });

  test('displays memory significance chart', () => {
    render(
      <RelationshipPanel 
        selectedRelationship={mockRelationshipData}
        relationshipHistory={mockHistoryEvents}
      />
    );

    const memoryChart = screen.getByTestId('memory-significance-chart');
    expect(memoryChart).toBeInTheDocument();
    expect(screen.getByText('Memory Significance: 0.80')).toBeInTheDocument();
  });

  test('shows no relationship selected state', () => {
    render(
      <RelationshipPanel 
        selectedRelationship={null}
        relationshipHistory={[]}
      />
    );

    expect(screen.getByText('Select a relationship to view details')).toBeInTheDocument();
  });
});

describe('RelationshipMetrics', () => {
  const mockMetrics = {
    totalRelationships: 15,
    averageAffinity: 0.45,
    strongBonds: 5,
    conflicts: 2,
    recentChanges: [
      { pair: ['tom', 'clara'], change: 0.1, timestamp: '2024-01-20T10:30:00Z' },
      { pair: ['alice', 'bob'], change: -0.2, timestamp: '2024-01-20T10:00:00Z' }
    ],
    socialClusters: [
      { id: 'cluster-1', members: ['tom', 'clara', 'alice'], cohesion: 0.7 },
      { id: 'cluster-2', members: ['bob', 'charlie'], cohesion: 0.5 }
    ]
  };

  test('displays overall social ecosystem metrics', () => {
    render(<RelationshipMetrics metrics={mockMetrics} />);

    expect(screen.getByText('Social Ecosystem Health')).toBeInTheDocument();
    expect(screen.getByText('Total Relationships')).toBeInTheDocument();
    expect(screen.getByText('15')).toBeInTheDocument();
    
    expect(screen.getByText('Average Affinity')).toBeInTheDocument();
    expect(screen.getByText('0.45')).toBeInTheDocument();
    
    expect(screen.getByText('Strong Bonds: 5')).toBeInTheDocument();
    expect(screen.getByText('Conflicts: 2')).toBeInTheDocument();
  });

  test('shows affinity distribution chart', () => {
    render(<RelationshipMetrics metrics={mockMetrics} />);

    const distributionChart = screen.getByTestId('affinity-distribution-chart');
    expect(distributionChart).toBeInTheDocument();
  });

  test('displays social clusters visualization', () => {
    render(<RelationshipMetrics metrics={mockMetrics} />);

    const clustersViz = screen.getByTestId('social-clusters-visualization');
    expect(clustersViz).toBeInTheDocument();
    
    // Check cluster information is shown
    expect(screen.getByText(/tom, clara, alice/)).toBeInTheDocument();
    expect(screen.getByText(/Cohesion: 0.7/)).toBeInTheDocument();
  });

  test('highlights recent relationship changes', () => {
    render(<RelationshipMetrics metrics={mockMetrics} />);

    const recentChanges = screen.getByTestId('recent-changes');
    expect(recentChanges).toBeInTheDocument();
    
    // Positive change should be highlighted differently
    const positiveChange = screen.getByText(/tom → clara: \+0\.1/);
    expect(positiveChange).toHaveClass('positive-change');
    
    const negativeChange = screen.getByText(/alice → bob: -0\.2/);
    expect(negativeChange).toHaveClass('negative-change');
  });

  test('updates metrics in real-time', async () => {
    const { rerender } = render(<RelationshipMetrics metrics={mockMetrics} />);

    // Update metrics
    const updatedMetrics = {
      ...mockMetrics,
      totalRelationships: 16,
      averageAffinity: 0.48,
      strongBonds: 6
    };

    rerender(<RelationshipMetrics metrics={updatedMetrics} />);

    await waitFor(() => {
      expect(screen.getByText('16')).toBeInTheDocument();
      expect(screen.getByText('0.48')).toBeInTheDocument();
      expect(screen.getByText('Strong Bonds: 6')).toBeInTheDocument();
    });
  });
}); 