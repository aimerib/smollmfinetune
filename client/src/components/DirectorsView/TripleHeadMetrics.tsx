/**
 * Triple-Head Metrics Component
 * 
 * Visualizes the performance metrics of the triple-head model architecture:
 * Generation Head, Control Head, and Memory Head.
 */

import React from 'react';
import { Typography, Paper } from '@mui/material';
import { PsychologyAlt, ControlCamera, Memory, Hub } from '@mui/icons-material';

interface TripleHeadMetricsProps {
  metrics: Record<string, {
    generation_quality: number;
    control_effectiveness: number;
    memory_coherence: number;
    coordination_score: number;
  }> | null;
  selectedEntity: string | null;
}

interface MetricCardProps {
  icon: React.ReactNode;
  title: string;
  value: number;
  color: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ icon, title, value, color }) => (
  <Paper style={{ padding: '16px', textAlign: 'center', backgroundColor: '#1f2937' }}>
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '8px' }}>
      {React.cloneElement(icon as React.ReactElement, { 
        style: { color, fontSize: '24px' } 
      })}
    </div>
    <Typography variant="body2" color="text.secondary" gutterBottom>
      {title}
    </Typography>
    <Typography variant="h6" style={{ color, fontWeight: 'bold' }}>
      {(value * 100).toFixed(0)}%
    </Typography>
  </Paper>
);

const TripleHeadMetrics: React.FC<TripleHeadMetricsProps> = ({
  metrics,
  selectedEntity
}) => {
  // If no entity selected
  if (!selectedEntity) {
    return (
      <div>
        <Typography variant="h6" gutterBottom>📊 Triple Head Metrics</Typography>
        <Typography variant="body2" color="text.secondary">
          Select an entity to view metrics
        </Typography>
      </div>
    );
  }

  // Get metrics for selected entity
  const entityMetrics = metrics?.[selectedEntity];

  // If no metrics data for selected entity
  if (!entityMetrics) {
    return (
      <div>
        <Typography variant="h6" gutterBottom>📊 Triple Head Metrics</Typography>
        <Typography variant="body2" color="text.secondary">
          No metrics data for selected entity
        </Typography>
      </div>
    );
  }

  return (
    <div style={{ padding: '16px' }}>
      <Typography variant="h6" gutterBottom>📊 Triple Head Metrics</Typography>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
        <MetricCard
          icon={<PsychologyAlt fontSize="small" />}
          title="Generation"
          value={entityMetrics.generation_quality}
          color="#10b981"
        />
        
        <MetricCard
          icon={<ControlCamera fontSize="small" />}
          title="Control"
          value={entityMetrics.control_effectiveness}
          color="#f59e0b"
        />
        
        <MetricCard
          icon={<Memory fontSize="small" />}
          title="Memory"
          value={entityMetrics.memory_coherence}
          color="#6366f1"
        />
        
        <MetricCard
          icon={<Hub fontSize="small" />}
          title="Coordination"
          value={entityMetrics.coordination_score}
          color="#ec4899"
        />
      </div>
    </div>
  );
};

export default TripleHeadMetrics; 