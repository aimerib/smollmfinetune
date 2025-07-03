/**
 * RelationshipTimeline Component
 * 
 * Displays a temporal view of relationship evolution with real-time updates
 */

import React, { useState, useEffect } from 'react';
import styled from '@emotion/styled';
import { motion, AnimatePresence } from 'framer-motion';
import { format } from 'date-fns';
import websocketService from '../../services/websocketService';
import { RelationshipHistoryEvent } from '../../types/relationships';

const TimelineContainer = styled.div`
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 1.5rem;
  max-height: 600px;
  overflow-y: auto;
`;

const TimelineEvent = styled(motion.div)<{ significanceLevel: string }>`
  position: relative;
  padding: 1rem;
  margin-bottom: 1rem;
  background: rgba(255, 255, 255, 0.03);
  border-left: 3px solid ${props => 
    props.significanceLevel === 'high' ? '#8b5cf6' : 
    props.significanceLevel === 'medium' ? '#3b82f6' : 
    '#64748b'
  };
  border-radius: 4px;
  
  &.high-significance {
    border-left-color: #8b5cf6;
  }
  
  &.low-significance {
    border-left-color: #64748b;
  }
`;

const EventHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
`;

const EventTime = styled.div`
  font-size: 0.85rem;
  color: #9ca3af;
`;

const AffinityChange = styled.span<{ positive: boolean }>`
  font-weight: 600;
  font-size: 1.1rem;
  
  &.positive-change {
    color: #4ade80;
  }
  
  &.negative-change {
    color: #f87171;
  }
`;

const EmotionalImpact = styled.div`
  display: flex;
  gap: 0.5rem;
  margin-top: 0.5rem;
  flex-wrap: wrap;
`;

const EmotionTag = styled.span`
  display: inline-block;
  padding: 0.2rem 0.6rem;
  background: rgba(99, 102, 241, 0.2);
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 12px;
  font-size: 0.8rem;
  color: #a5b4fc;
`;

const ExpandButton = styled.button`
  background: none;
  border: none;
  color: #9ca3af;
  cursor: pointer;
  padding: 0.25rem;
  
  &:hover {
    color: #e5e7eb;
  }
`;

const ExpandedDetails = styled(motion.div)`
  margin-top: 0.75rem;
  padding-top: 0.75rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  font-size: 0.9rem;
  color: #d1d5db;
`;

interface RelationshipTimelineProps {
  agentPair: [string, string];
  initialEvents?: RelationshipHistoryEvent[];
}

const getInteractionIcon = (type: string) => {
  switch (type) {
    case 'conversation':
      return <span data-testid="icon-conversation">💬</span>;
    case 'support':
      return <span data-testid="icon-support">❤️</span>;
    case 'conflict':
      return <span data-testid="icon-conflict">⚠️</span>;
    case 'greeting':
      return <span data-testid="icon-greeting">👋</span>;
    default:
      return <span>💬</span>;
  }
};

export const RelationshipTimeline: React.FC<RelationshipTimelineProps> = ({
  agentPair,
  initialEvents = []
}) => {
  const [events, setEvents] = useState<RelationshipHistoryEvent[]>(initialEvents.slice(0, 50));
  const [expandedEvents, setExpandedEvents] = useState<Set<number>>(new Set());

  // Check if event is relevant to the agent pair
  const isRelevantToPair = (event: any, pair: [string, string]): boolean => {
    return (
      (event.source === pair[0] && event.target === pair[1]) ||
      (event.source === pair[1] && event.target === pair[0])
    );
  };

  // Subscribe to relationship events
  useEffect(() => {
    const unsubscribe = websocketService.onRelationshipEvent((event: any) => {
      if (isRelevantToPair(event, agentPair)) {
        const newEvent: RelationshipHistoryEvent = {
          timestamp: event.timestamp,
          speaker_id: event.speaker_id || event.source,
          target_id: event.target_id || event.target,
          interaction_type: event.interaction_type,
          affinity_change: event.affinity_change,
          emotional_impact: event.emotional_impact,
          memory_significance: event.memory_significance,
          narrative_context: event.narrative_context
        };
        
        setEvents(prev => [newEvent, ...prev.slice(0, 49)]); // Keep last 50 events
      }
    });

    return unsubscribe;
  }, [agentPair]);

  const getSignificanceLevel = (significance: number): string => {
    if (significance >= 0.7) return 'high';
    if (significance >= 0.4) return 'medium';
    return 'low';
  };

  const toggleExpanded = (index: number) => {
    setExpandedEvents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  return (
    <TimelineContainer>
      <AnimatePresence>
        {events.map((event, index) => {
          const significanceLevel = getSignificanceLevel(event.memory_significance);
          const isExpanded = expandedEvents.has(index);
          
          return (
            <TimelineEvent
              key={`${event.timestamp}-${index}`}
              data-testid={`timeline-event-${index}`}
              className={significanceLevel === 'high' ? 'high-significance' : 'low-significance'}
              significanceLevel={significanceLevel}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.3 }}
            >
              <EventHeader>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {getInteractionIcon(event.interaction_type)}
                  <span>
                    {event.speaker_id} → {event.target_id}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <EventTime>
                    {format(new Date(event.timestamp), 'HH:mm')}
                  </EventTime>
                  <AffinityChange 
                    positive={event.affinity_change > 0}
                    className={event.affinity_change > 0 ? 'positive-change' : 'negative-change'}
                  >
                    {event.affinity_change > 0 ? '+' : ''}{event.affinity_change.toFixed(2)}
                  </AffinityChange>
                  <ExpandButton 
                    onClick={() => toggleExpanded(index)}
                    role="button"
                    aria-label="expand"
                  >
                    <span style={{ 
                      transform: isExpanded ? 'rotate(180deg)' : 'none',
                      transition: 'transform 0.2s',
                      display: 'inline-block',
                      fontSize: '14px'
                    }}>
                      ▼
                    </span>
                  </ExpandButton>
                </div>
              </EventHeader>
              
              {event.narrative_context && (
                <div style={{ marginBottom: '0.5rem', color: '#e5e7eb' }}>
                  {event.narrative_context}
                </div>
              )}
              
              <EmotionalImpact>
                {event.emotional_impact.map((emotion, i) => (
                  <EmotionTag key={i}>{emotion}</EmotionTag>
                ))}
              </EmotionalImpact>
              
              {isExpanded && (
                <ExpandedDetails
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <div>Memory Formation</div>
                  <div>Significance: {event.memory_significance.toFixed(1)}</div>
                </ExpandedDetails>
              )}
            </TimelineEvent>
          );
        })}
      </AnimatePresence>
    </TimelineContainer>
  );
}; 