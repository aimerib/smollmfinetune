/**
 * Triple-Head Metrics Component
 * 
 * Visualizes the performance metrics of the triple-head model architecture:
 * Generation Head, Control Head, and Memory Head.
 */

import React from 'react';
import { Box, Typography, CircularProgress, Paper } from '@mui/material';
import { PsychologyAlt, ControlCamera, Memory, Hub } from '@mui/icons-material';
import Plot from 'react-plotly.js';

interface TripleHeadMetricsProps {
  metrics: Record<string, {
    generation_quality: number;
    control_effectiveness: number;
    memory_coherence: number;
    coordination_score: number;
  }>;
  selectedEntity: string | null;
}

interface MetricCardProps {
  icon: React.ReactNode;
  title: string;
  value: number;
  color: string;
}

const MetricCard: React.FC<MetricCardProps> = ({ icon, title, value, color }) => {
  const percentage = value * 100;
  
  return (
    <Paper
      elevation={1}
      sx={{
        p: 2,
        backgroundColor: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: 2,
        position: 'relative',
        overflow: 'hidden',
        transition: 'all 0.3s ease',
        '&:hover': {
          backgroundColor: 'rgba(255,255,255,0.05)',
          transform: 'translateY(-2px)',
        }
      }}
    >
      <Box sx={{ position: 'relative', zIndex: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box sx={{ color, mr: 1 }}>{icon}</Box>
          <Typography variant="caption" sx={{ opacity: 0.8 }}>
            {title}
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', alignItems: 'baseline' }}>
          <Typography variant="h5" sx={{ fontWeight: 600, color }}>
            {percentage.toFixed(0)}
          </Typography>
          <Typography variant="body2" sx={{ ml: 0.5, opacity: 0.6 }}>
            %
          </Typography>
        </Box>
      </Box>
      
      {/* Background gradient based on value */}
      <Box
        sx={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: `${percentage}%`,
          background: `linear-gradient(180deg, ${color}30 0%, ${color}10 100%)`,
          transition: 'height 0.5s ease',
        }}
      />
    </Paper>
  );
};

const TripleHeadMetrics: React.FC<TripleHeadMetricsProps> = ({ metrics, selectedEntity }) => {
  const entityMetrics = selectedEntity ? metrics[selectedEntity] : null;
  
  if (!entityMetrics && selectedEntity) {
    return (
      <Box>
        <Typography variant="h6" gutterBottom>🧠 Triple-Head Metrics</Typography>
        <Typography variant="body2" color="text.secondary">
          No metrics data for selected entity
        </Typography>
      </Box>
    );
  }
  
  // If no entity selected, show aggregate metrics
  if (!selectedEntity) {
    const metricValues = Object.values(metrics);
    const hasData = metricValues.length > 0;
    
    const avgMetrics = hasData ? {
      generation_quality: metricValues.reduce((sum, m) => sum + m.generation_quality, 0) / metricValues.length,
      control_effectiveness: metricValues.reduce((sum, m) => sum + m.control_effectiveness, 0) / metricValues.length,
      memory_coherence: metricValues.reduce((sum, m) => sum + m.memory_coherence, 0) / metricValues.length,
      coordination_score: metricValues.reduce((sum, m) => sum + m.coordination_score, 0) / metricValues.length,
    } : null;
    
    if (!avgMetrics) {
      return (
        <Box>
          <Typography variant="h6" gutterBottom>🧠 Triple-Head Metrics</Typography>
          <Typography variant="body2" color="text.secondary">
            No metrics data available
          </Typography>
        </Box>
      );
    }
    
    // Radar chart for aggregate view
    const data = [{
      type: 'scatterpolar',
      r: [
        avgMetrics.generation_quality * 100,
        avgMetrics.control_effectiveness * 100,
        avgMetrics.memory_coherence * 100,
        avgMetrics.coordination_score * 100
      ],
      theta: ['Generation', 'Control', 'Memory', 'Coordination'],
      fill: 'toself',
      fillcolor: 'rgba(99, 102, 241, 0.2)',
      line: { color: '#6366f1' },
      marker: { color: '#6366f1' }
    }];
    
    const layout = {
      polar: {
        radialaxis: {
          visible: true,
          range: [0, 100],
          tickfont: { size: 10, color: '#94a3b8' },
          gridcolor: 'rgba(255,255,255,0.1)'
        },
        angularaxis: {
          tickfont: { size: 10, color: '#ffffff' },
          gridcolor: 'rgba(255,255,255,0.1)'
        },
        bgcolor: 'transparent'
      },
      showlegend: false,
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      margin: { t: 30, r: 30, b: 30, l: 30 },
      height: 200
    };
    
    return (
      <Box>
        <Typography variant="h6" gutterBottom>🧠 Triple-Head Overview</Typography>
        <Plot
          data={data}
          layout={layout}
          config={{ displayModeBar: false }}
          style={{ width: '100%' }}
        />
      </Box>
    );
  }
  
  // Show individual entity metrics
  return (
    <Box>
      <Typography variant="h6" gutterBottom>🧠 Triple-Head Metrics</Typography>
      
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: 1.5,
        }}
      >
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
      </Box>
    </Box>
  );
};

export default TripleHeadMetrics; 