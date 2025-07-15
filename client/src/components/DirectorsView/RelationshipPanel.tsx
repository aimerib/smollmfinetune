/**
 * RelationshipPanel Component
 * 
 * Displays detailed relationship information and social ecosystem metrics
 */

import React from 'react';
import styled from '@emotion/styled';
import { motion } from 'framer-motion';
import { RelationshipEdge, RelationshipHistoryEvent, RelationshipMetricsData } from '../../types/relationships';

const PanelContainer = styled.div`
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1rem;
`;

const PanelTitle = styled.h3`
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0 0 1rem 0;
  color: #e5e7eb;
`;

const RelationshipHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
`;

const MetricCard = styled.div`
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 6px;
  padding: 1rem;
  margin-bottom: 0.75rem;
  
  h4 {
    font-size: 0.9rem;
    color: #9ca3af;
    margin: 0 0 0.5rem 0;
  }
  
  .value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #e5e7eb;
  }
`;

const EmotionalTimeline = styled.div`
  display: flex;
  gap: 0.5rem;
  margin: 1rem 0;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
  overflow-x: auto;
`;

const EmotionTag = styled.span`
  display: inline-block;
  padding: 0.25rem 0.75rem;
  background: rgba(99, 102, 241, 0.2);
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 16px;
  font-size: 0.85rem;
  color: #a5b4fc;
  white-space: nowrap;
`;

const ChangeIndicator = styled.span<{ positive: boolean }>`
  color: ${props => props.positive ? '#4ade80' : '#f87171'};
  font-weight: 600;
  
  &.positive-change {
    color: #4ade80;
  }
  
  &.negative-change {
    color: #f87171;
  }
`;

interface RelationshipPanelProps {
  selectedRelationship: RelationshipEdge | null;
  relationshipHistory: RelationshipHistoryEvent[];
}

export const RelationshipPanel: React.FC<RelationshipPanelProps> = ({
  selectedRelationship,
  relationshipHistory
}) => {
  if (!selectedRelationship) {
    return (
      <PanelContainer>
        <PanelTitle>Relationship Details</PanelTitle>
        <p style={{ color: '#6b7280', textAlign: 'center' }}>
          Select a relationship to view details
        </p>
      </PanelContainer>
    );
  }

  return (
    <PanelContainer>
      <PanelTitle>Relationship Details</PanelTitle>
      
      <RelationshipHeader>
        <div>
          <h4 style={{ margin: 0, fontSize: '1.2rem' }}>
            {selectedRelationship.source} → {selectedRelationship.target}
          </h4>
          <span style={{ color: '#9ca3af' }}>{selectedRelationship.status}</span>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div>Affinity: {selectedRelationship.affinity.toFixed(2)}</div>
          <div style={{ fontSize: '0.85rem', color: '#9ca3af' }}>
            {selectedRelationship.interaction_count} interactions
          </div>
        </div>
      </RelationshipHeader>

      <MetricCard>
        <h4>Memory Significance</h4>
        <div className="value">Memory Significance: {selectedRelationship.memory_significance.toFixed(2)}</div>
      </MetricCard>

      <div>
        <h4 style={{ fontSize: '0.95rem', marginBottom: '0.5rem' }}>Emotional History</h4>
        <EmotionalTimeline data-testid="emotional-timeline">
          {selectedRelationship.emotional_history.map((emotion, index) => (
            <EmotionTag key={index}>{emotion}</EmotionTag>
          ))}
        </EmotionalTimeline>
      </div>

      <div data-testid="memory-significance-chart">
        {/* Memory significance visualization would go here */}
      </div>
    </PanelContainer>
  );
};

interface RelationshipMetricsProps {
  metrics: RelationshipMetricsData;
}

export const RelationshipMetrics: React.FC<RelationshipMetricsProps> = ({ metrics }) => {
  return (
    <PanelContainer>
      <PanelTitle>Social Ecosystem Health</PanelTitle>
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        <MetricCard>
          <h4>Total Relationships</h4>
          <div className="value">{metrics.totalRelationships}</div>
        </MetricCard>
        
        <MetricCard>
          <h4>Average Affinity</h4>
          <div className="value">{metrics.averageAffinity.toFixed(2)}</div>
        </MetricCard>
      </div>
      
      <div style={{ marginTop: '1rem' }}>
        <div>Strong Bonds: {metrics.strongBonds}</div>
        <div>Conflicts: {metrics.conflicts}</div>
      </div>
      
      <div data-testid="affinity-distribution-chart" style={{ marginTop: '1rem' }}>
        {/* Affinity distribution chart would go here */}
      </div>
      
      <div data-testid="social-clusters-visualization" style={{ marginTop: '1rem' }}>
        {metrics.socialClusters.map(cluster => (
          <div key={cluster.id} style={{ marginBottom: '0.5rem' }}>
            <div>{cluster.members.join(', ')}</div>
            <div style={{ fontSize: '0.85rem', color: '#9ca3af' }}>
              Cohesion: {cluster.cohesion.toFixed(1)}
            </div>
          </div>
        ))}
      </div>
      
      <div data-testid="recent-changes" style={{ marginTop: '1rem' }}>
        <h4 style={{ fontSize: '0.95rem', marginBottom: '0.5rem' }}>Recent Changes</h4>
        {metrics.recentChanges.map((change, index) => (
          <div key={index} style={{ marginBottom: '0.25rem' }}>
            <ChangeIndicator 
              positive={change.change > 0}
              className={change.change > 0 ? 'positive-change' : 'negative-change'}
            >
              {change.pair[0]} → {change.pair[1]}: {change.change > 0 ? '+' : ''}{change.change.toFixed(1)}
            </ChangeIndicator>
          </div>
        ))}
      </div>
    </PanelContainer>
  );
}; 