import React, { useEffect, useState } from 'react';
import styled from '@emotion/styled';
import { TrainingStatus } from '../../services/directorsChairService';

const Container = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 1.5rem;
`;

const Title = styled.h3`
  margin: 0 0 1.5rem 0;
  color: white;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const OverallProgress = styled.div`
  margin-bottom: 1.5rem;
`;

const ProgressLabel = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  color: rgba(255, 255, 255, 0.9);
  font-weight: 500;
`;

const ProgressBar = styled.div<{ progress: number; color?: string }>`
  width: 100%;
  height: 8px;
  background: #374151;
  border-radius: 4px;
  overflow: hidden;
  position: relative;

  &::after {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: ${props => props.progress * 100}%;
    background: ${props => props.color || '#3b82f6'};
    transition: width 0.3s ease;
  }
`;

const HeadSection = styled.div`
  margin: 1rem 0;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 6px;
`;

const HeadTitle = styled.h4`
  margin: 0 0 0.75rem 0;
  color: white;
  font-size: 0.9rem;
`;

const StatusBadge = styled.span<{ status: string }>`
  padding: 0.25rem 0.5rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 500;
  background: ${props => {
    switch (props.status) {
      case 'training': return '#d97706';
      case 'completed': return '#059669';
      case 'queued': return '#7c3aed';
      default: return '#6b7280';
    }
  }};
  color: white;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  margin-top: 0.75rem;
  font-size: 0.85rem;
`;

const MetricItem = styled.div`
  color: rgba(255, 255, 255, 0.7);
`;

const QueueInfo = styled.div`
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.6);
`;

const ExpandableSection = styled.div<{ expanded: boolean }>`
  margin-top: 1rem;
  max-height: ${props => props.expanded ? '200px' : '0'};
  overflow: hidden;
  transition: max-height 0.3s ease;
`;

const ToggleButton = styled.button`
  background: none;
  border: none;
  color: #3b82f6;
  cursor: pointer;
  font-size: 0.85rem;
  text-decoration: underline;

  &:hover {
    color: #60a5fa;
  }
`;

interface Props {
  trainingStatus: TrainingStatus;
  showMetrics?: boolean;
  onTrainingComplete?: (headType: string) => void;
}

export const TrainingProgressIndicator: React.FC<Props> = ({
  trainingStatus,
  showMetrics = false,
  onTrainingComplete
}) => {
  const [previousStatus, setPreviousStatus] = useState<TrainingStatus>({});
  const [showMetricsExpanded, setShowMetricsExpanded] = useState(showMetrics);

  useEffect(() => {
    // Check for training completion
    if (!trainingStatus || !previousStatus) return;
    
    Object.keys(trainingStatus).forEach(headType => {
      if (headType.endsWith('_head')) {
        const currentHead = trainingStatus[headType as keyof TrainingStatus];
        const previousHead = previousStatus[headType as keyof TrainingStatus];
        
        if (
          currentHead?.status === 'completed' &&
          previousHead?.status === 'training' &&
          onTrainingComplete
        ) {
          onTrainingComplete(headType);
        }
      }
    });

    setPreviousStatus(trainingStatus);
  }, [trainingStatus, previousStatus, onTrainingComplete]);

  const getOverallProgress = () => {
    const heads = ['generation_head', 'control_head', 'memory_head'] as const;
    const progresses = heads.map(head => trainingStatus?.[head]?.progress || 0);
    return progresses.reduce((sum, progress) => sum + progress, 0) / heads.length;
  };

  const getProgressColor = (status?: string) => {
    switch (status) {
      case 'training': return '#3b82f6';
      case 'completed': return '#10b981';
      case 'queued': return '#8b5cf6';
      default: return '#6b7280';
    }
  };

  const formatETA = (eta?: string) => {
    if (!eta || eta === 'pending') return 'Pending';
    if (eta === 'complete') return 'Complete';
    return eta;
  };

  const formatPercentage = (progress: number) => {
    return `${Math.round(progress * 100)}%`;
  };

  return (
    <Container>
      <Title>Training Progress</Title>

      <OverallProgress>
        <ProgressLabel>
          <span>Overall Progress</span>
          <span data-testid="overall-progress">{formatPercentage(trainingStatus?.overall_progress || getOverallProgress())}</span>
        </ProgressLabel>
        <ProgressBar progress={trainingStatus?.overall_progress || getOverallProgress()} />
      </OverallProgress>

      {/* Generation Head */}
      {trainingStatus?.generation_head && (
        <HeadSection data-testid="generation-head-section">
          <HeadTitle>Generation Head</HeadTitle>
          <ProgressLabel>
            <StatusBadge status={trainingStatus.generation_head.status}>
              {trainingStatus.generation_head.status}
            </StatusBadge>
            <span data-testid="generation-progress">{formatPercentage(trainingStatus.generation_head.progress)}</span>
          </ProgressLabel>
          <ProgressBar 
            progress={trainingStatus.generation_head.progress}
            color={getProgressColor(trainingStatus.generation_head.status)}
          />
          
          {trainingStatus.generation_head.eta && (
            <QueueInfo data-testid="generation-eta">ETA: {formatETA(trainingStatus.generation_head.eta)}</QueueInfo>
          )}

          {trainingStatus.generation_head.metrics && showMetricsExpanded && (
            <ExpandableSection expanded={showMetricsExpanded}>
              <h5>Training Metrics</h5>
              <MetricsGrid>
                <MetricItem>Loss: {trainingStatus.generation_head.metrics.current_loss || 'N/A'}</MetricItem>
                <MetricItem>Best: {trainingStatus.generation_head.metrics.best_loss || 'N/A'}</MetricItem>
              </MetricsGrid>
            </ExpandableSection>
          )}
        </HeadSection>
      )}

      {/* Control Head */}
      {trainingStatus?.control_head && (
        <HeadSection data-testid="control-head-section">
          <HeadTitle>Control Head</HeadTitle>
          <ProgressLabel>
            <StatusBadge status={trainingStatus.control_head.status}>
              {trainingStatus.control_head.status === 'queued' ? 'Queued' : trainingStatus.control_head.status}
            </StatusBadge>
            <span data-testid="control-progress">{formatPercentage(trainingStatus.control_head.progress)}</span>
          </ProgressLabel>
          <ProgressBar 
            progress={trainingStatus.control_head.progress}
            color={getProgressColor(trainingStatus.control_head.status)}
          />

          {trainingStatus.control_head.queue_position !== undefined && (
            <QueueInfo data-testid="control-queue">Queue Position: {trainingStatus.control_head.queue_position}</QueueInfo>
          )}
          
          {trainingStatus.control_head.eta && trainingStatus.control_head.status === 'queued' && (
            <QueueInfo data-testid="control-eta">Est. Start: {formatETA(trainingStatus.control_head.eta)}</QueueInfo>
          )}
        </HeadSection>
      )}

      {/* Memory Head */}
      {trainingStatus?.memory_head && (
        <HeadSection data-testid="memory-head-section">
          <HeadTitle>Memory Head</HeadTitle>
          <ProgressLabel>
            <StatusBadge status={trainingStatus.memory_head.status} data-testid="memory-status">
              {trainingStatus.memory_head.status === 'completed' ? 'Complete' : trainingStatus.memory_head.status}
            </StatusBadge>
            <span data-testid="memory-progress">{formatPercentage(trainingStatus.memory_head.progress)}</span>
          </ProgressLabel>
          <ProgressBar 
            progress={trainingStatus.memory_head.progress}
            color={getProgressColor(trainingStatus.memory_head.status)}
          />
          
          {trainingStatus.memory_head.eta && (
            <QueueInfo data-testid="memory-eta">ETA: {formatETA(trainingStatus.memory_head.eta)}</QueueInfo>
          )}
        </HeadSection>
      )}

      {showMetrics && (
        <ToggleButton 
          onClick={() => setShowMetricsExpanded(!showMetricsExpanded)}
        >
          {showMetricsExpanded ? 'Hide' : 'Show'} Detailed Metrics
        </ToggleButton>
      )}
    </Container>
  );
}; 