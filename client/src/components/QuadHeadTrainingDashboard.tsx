import React, { useState, useEffect, useRef, useCallback } from 'react';
import styled from '@emotion/styled';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

// Types
interface TrainingMetrics {
  step: number;
  epoch: number;
  total_loss: number;
  generation_loss?: number;
  control_loss?: number;
  memory_loss?: number;
  speech_loss?: number;
  learning_rate: number;
  grad_norm?: number;
  samples_per_second?: number;
  timestamp: string;
}

interface TrainingStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress: number;
  current_step: number;
  total_steps: number;
  current_epoch: number;
  total_epochs: number;
  last_metrics?: TrainingMetrics;
  error_message?: string;
  model_path?: string;
}

interface QuadHeadTrainingConfig {
  base_model_name: string;
  enable_speech_head: boolean;
  speech_mel_bins: number;
  speech_quantization_bits: number;
  dataset_path: string;
  output_dir: string;
  learning_rate: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  num_train_epochs: number;
  warmup_steps: number;
  max_seq_length: number;
  text_weight: number;
  control_weight: number;
  memory_weight: number;
  speech_weight: number;
  use_spectral_loss: boolean;
  enable_distributed: boolean;
  fp16: boolean;
  gradient_checkpointing: boolean;
  save_steps: number;
  eval_steps: number;
  logging_steps: number;
}

// Styled components
const Container = styled.div`
  background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
  border-radius: 16px;
  padding: 2rem;
  color: white;
  min-height: 100vh;
`;

const Header = styled.div`
  display: flex;
  justify-content: between;
  align-items: center;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
`;

const Title = styled.h1`
  margin: 0;
  color: #fff;
  font-size: 2rem;
  font-weight: 600;
`;

const StatusBadge = styled.div<{ status: string }>`
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-weight: 500;
  font-size: 0.875rem;
  background: ${props => {
    switch (props.status) {
      case 'running': return '#10b981';
      case 'completed': return '#3b82f6';
      case 'failed': return '#ef4444';
      case 'cancelled': return '#6b7280';
      default: return '#f59e0b';
    }
  }};
  color: white;
  text-transform: capitalize;
`;

const MetricsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const MetricCard = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1.5rem;
`;

const MetricTitle = styled.h3`
  margin: 0 0 1rem 0;
  color: #fff;
  font-size: 1.125rem;
  font-weight: 500;
`;

const MetricValue = styled.div`
  font-size: 2rem;
  font-weight: 700;
  color: #3b82f6;
  margin-bottom: 0.5rem;
`;

const MetricSubtitle = styled.div`
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.875rem;
`;

const ProgressSection = styled.div`
  margin-bottom: 2rem;
`;

const ProgressBar = styled.div<{ progress: number; color?: string }>`
  width: 100%;
  height: 12px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  overflow: hidden;
  position: relative;
  margin-bottom: 0.5rem;

  &::after {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: ${props => props.progress}%;
    background: ${props => props.color || '#3b82f6'};
    transition: width 0.3s ease;
  }
`;

const ProgressLabel = styled.div`
  display: flex;
  justify-content: space-between;
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.875rem;
`;

const ChartSection = styled.div`
  margin-bottom: 2rem;
`;

const ChartContainer = styled.div`
  background: rgba(255, 255, 255, 0.02);
  border-radius: 12px;
  padding: 1.5rem;
  height: 400px;
`;

const HeadsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
`;

const HeadCard = styled.div<{ headType: string }>`
  background: rgba(255, 255, 255, 0.05);
  border: 2px solid ${props => {
    switch (props.headType) {
      case 'generation': return '#3b82f6';
      case 'control': return '#10b981';
      case 'memory': return '#f59e0b';
      case 'speech': return '#ef4444';
      default: return 'rgba(255, 255, 255, 0.1)';
    }
  }};
  border-radius: 12px;
  padding: 1rem;
  text-align: center;
`;

const HeadTitle = styled.h4`
  margin: 0 0 0.5rem 0;
  color: #fff;
  font-size: 1rem;
  font-weight: 500;
  text-transform: capitalize;
`;

const HeadLoss = styled.div`
  font-size: 1.5rem;
  font-weight: 700;
  color: ${props => props.color || '#fff'};
`;

const ConfigPanel = styled.div`
  background: rgba(255, 255, 255, 0.02);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 2rem;
`;

const ConfigGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
`;

const ConfigItem = styled.div`
  display: flex;
  justify-content: space-between;
  padding: 0.5rem 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
`;

const ConfigLabel = styled.span`
  color: rgba(255, 255, 255, 0.7);
`;

const ConfigValue = styled.span`
  color: #fff;
  font-weight: 500;
`;

const ActionButtons = styled.div`
  display: flex;
  gap: 1rem;
  margin-top: 2rem;
`;

const Button = styled.button<{ variant?: 'primary' | 'secondary' | 'danger' }>`
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  border: none;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  
  background: ${props => {
    switch (props.variant) {
      case 'primary': return '#3b82f6';
      case 'danger': return '#ef4444';
      default: return 'rgba(255, 255, 255, 0.1)';
    }
  }};
  
  color: white;
  
  &:hover {
    opacity: 0.8;
    transform: translateY(-2px);
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
`;

interface Props {
  jobId?: string;
  config?: QuadHeadTrainingConfig;
  onStartTraining?: (config: QuadHeadTrainingConfig) => void;
  onCancelTraining?: (jobId: string) => void;
}

export const QuadHeadTrainingDashboard: React.FC<Props> = ({
  jobId,
  config,
  onStartTraining,
  onCancelTraining
}) => {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [metricsHistory, setMetricsHistory] = useState<TrainingMetrics[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // WebSocket connection for real-time updates
  const connectWebSocket = useCallback(() => {
    if (!jobId) return;

    const ws = new WebSocket(`ws://localhost:8000/api/training/quad-head/ws/${jobId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      console.log('Connected to training WebSocket');
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      
      if (message.type === 'status') {
        setTrainingStatus(message.data);
      } else if (message.type === 'metrics') {
        const metrics = message.data;
        setMetricsHistory(prev => [...prev.slice(-99), metrics]); // Keep last 100 points
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log('Disconnected from training WebSocket');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
  }, [jobId]);

  // Connect WebSocket when jobId changes
  useEffect(() => {
    if (jobId) {
      connectWebSocket();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [jobId, connectWebSocket]);

  // Get current metrics from status
  const currentMetrics = trainingStatus?.last_metrics;

  // Calculate head-specific losses
  const headLosses = currentMetrics ? [
    { name: 'Generation', loss: currentMetrics.generation_loss, color: '#3b82f6' },
    { name: 'Control', loss: currentMetrics.control_loss, color: '#10b981' },
    { name: 'Memory', loss: currentMetrics.memory_loss, color: '#f59e0b' },
    { name: 'Speech', loss: currentMetrics.speech_loss, color: '#ef4444' }
  ].filter(head => head.loss !== undefined) : [];

  // Format duration
  const formatDuration = (start?: string, end?: string) => {
    if (!start) return 'Not started';
    
    const startTime = new Date(start);
    const endTime = end ? new Date(end) : new Date();
    const diff = endTime.getTime() - startTime.getTime();
    
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((diff % (1000 * 60)) / 1000);
    
    return `${hours}h ${minutes}m ${seconds}s`;
  };

  // Handle cancel training
  const handleCancel = () => {
    if (jobId && onCancelTraining) {
      onCancelTraining(jobId);
    }
  };

  return (
    <Container>
      <Header>
        <Title>Quad-Head Training Dashboard</Title>
        {trainingStatus && (
          <StatusBadge status={trainingStatus.status}>
            {trainingStatus.status}
          </StatusBadge>
        )}
      </Header>

      {/* Overall Progress */}
      {trainingStatus && (
        <ProgressSection>
          <MetricCard>
            <MetricTitle>Training Progress</MetricTitle>
            <ProgressBar progress={trainingStatus.progress} />
            <ProgressLabel>
              <span>
                Epoch {trainingStatus.current_epoch} / {trainingStatus.total_epochs}
              </span>
              <span>{trainingStatus.progress.toFixed(1)}%</span>
            </ProgressLabel>
            <ProgressLabel>
              <span>
                Step {trainingStatus.current_step} / {trainingStatus.total_steps}
              </span>
              <span>
                Duration: {formatDuration(trainingStatus.started_at, trainingStatus.completed_at)}
              </span>
            </ProgressLabel>
          </MetricCard>
        </ProgressSection>
      )}

      {/* Key Metrics */}
      {currentMetrics && (
        <MetricsGrid>
          <MetricCard>
            <MetricTitle>Total Loss</MetricTitle>
            <MetricValue>{currentMetrics.total_loss.toFixed(4)}</MetricValue>
            <MetricSubtitle>Combined loss across all heads</MetricSubtitle>
          </MetricCard>
          
          <MetricCard>
            <MetricTitle>Learning Rate</MetricTitle>
            <MetricValue>{currentMetrics.learning_rate.toExponential(2)}</MetricValue>
            <MetricSubtitle>Current optimizer learning rate</MetricSubtitle>
          </MetricCard>
          
          <MetricCard>
            <MetricTitle>Throughput</MetricTitle>
            <MetricValue>
              {currentMetrics.samples_per_second?.toFixed(1) || 'N/A'}
            </MetricValue>
            <MetricSubtitle>Samples per second</MetricSubtitle>
          </MetricCard>
          
          <MetricCard>
            <MetricTitle>Gradient Norm</MetricTitle>
            <MetricValue>
              {currentMetrics.grad_norm?.toFixed(3) || 'N/A'}
            </MetricValue>
            <MetricSubtitle>Gradient clipping indicator</MetricSubtitle>
          </MetricCard>
        </MetricsGrid>
      )}

      {/* Head-specific Losses */}
      {headLosses.length > 0 && (
        <ChartSection>
          <MetricTitle>Head-Specific Losses</MetricTitle>
          <HeadsGrid>
            {headLosses.map((head) => (
              <HeadCard key={head.name} headType={head.name.toLowerCase()}>
                <HeadTitle>{head.name} Head</HeadTitle>
                <HeadLoss color={head.color}>
                  {head.loss?.toFixed(4) || 'N/A'}
                </HeadLoss>
              </HeadCard>
            ))}
          </HeadsGrid>
        </ChartSection>
      )}

      {/* Loss Chart */}
      {metricsHistory.length > 0 && (
        <ChartSection>
          <MetricTitle>Training Loss Over Time</MetricTitle>
          <ChartContainer>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metricsHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                <XAxis 
                  dataKey="step" 
                  stroke="rgba(255, 255, 255, 0.6)"
                />
                <YAxis stroke="rgba(255, 255, 255, 0.6)" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'rgba(0, 0, 0, 0.8)', 
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="total_loss" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  name="Total Loss"
                  dot={false}
                />
                {currentMetrics?.generation_loss !== undefined && (
                  <Line 
                    type="monotone" 
                    dataKey="generation_loss" 
                    stroke="#10b981" 
                    strokeWidth={1}
                    name="Generation"
                    dot={false}
                  />
                )}
                {currentMetrics?.control_loss !== undefined && (
                  <Line 
                    type="monotone" 
                    dataKey="control_loss" 
                    stroke="#f59e0b" 
                    strokeWidth={1}
                    name="Control"
                    dot={false}
                  />
                )}
                {currentMetrics?.memory_loss !== undefined && (
                  <Line 
                    type="monotone" 
                    dataKey="memory_loss" 
                    stroke="#8b5cf6" 
                    strokeWidth={1}
                    name="Memory"
                    dot={false}
                  />
                )}
                {currentMetrics?.speech_loss !== undefined && (
                  <Line 
                    type="monotone" 
                    dataKey="speech_loss" 
                    stroke="#ef4444" 
                    strokeWidth={1}
                    name="Speech"
                    dot={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </ChartSection>
      )}

      {/* Configuration */}
      {config && (
        <ConfigPanel>
          <MetricTitle>Training Configuration</MetricTitle>
          <ConfigGrid>
            <ConfigItem>
              <ConfigLabel>Base Model</ConfigLabel>
              <ConfigValue>{config.base_model_name}</ConfigValue>
            </ConfigItem>
            <ConfigItem>
              <ConfigLabel>Speech Head</ConfigLabel>
              <ConfigValue>{config.enable_speech_head ? 'Enabled' : 'Disabled'}</ConfigValue>
            </ConfigItem>
            <ConfigItem>
              <ConfigLabel>Learning Rate</ConfigLabel>
              <ConfigValue>{config.learning_rate}</ConfigValue>
            </ConfigItem>
            <ConfigItem>
              <ConfigLabel>Batch Size</ConfigLabel>
              <ConfigValue>{config.batch_size}</ConfigValue>
            </ConfigItem>
            <ConfigItem>
              <ConfigLabel>Epochs</ConfigLabel>
              <ConfigValue>{config.num_train_epochs}</ConfigValue>
            </ConfigItem>
            <ConfigItem>
              <ConfigLabel>Mixed Precision</ConfigLabel>
              <ConfigValue>{config.fp16 ? 'FP16' : 'FP32'}</ConfigValue>
            </ConfigItem>
          </ConfigGrid>
        </ConfigPanel>
      )}

      {/* Action Buttons */}
      <ActionButtons>
        {trainingStatus?.status === 'running' && (
          <Button variant="danger" onClick={handleCancel}>
            Cancel Training
          </Button>
        )}
        
        {trainingStatus?.status === 'completed' && trainingStatus.model_path && (
          <Button variant="primary">
            Download Model
          </Button>
        )}
        
        <Button 
          variant="secondary"
          disabled={!isConnected}
        >
          {isConnected ? '🟢 Live' : '🔴 Disconnected'}
        </Button>
      </ActionButtons>

      {/* Error Display */}
      {trainingStatus?.error_message && (
        <MetricCard style={{ marginTop: '1rem', borderColor: '#ef4444' }}>
          <MetricTitle>Error</MetricTitle>
          <MetricSubtitle style={{ color: '#ef4444' }}>
            {trainingStatus.error_message}
          </MetricSubtitle>
        </MetricCard>
      )}
    </Container>
  );
}; 