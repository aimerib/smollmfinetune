/**
 * Emotional State Panel Component
 * 
 * Displays character emotional states with real-time decay visualization
 * and surprise score tracking.
 */

import React from 'react';
import { Box, Typography, LinearProgress, Chip } from '@mui/material';
import Plot from 'react-plotly.js';
import { motion } from 'framer-motion';

interface EmotionalStateProps {
  emotions: Record<string, {
    active_emotions: Record<string, number>;
    surprise_score: number;
    momentum: number;
  }>;
  selectedEntity: string | null;
}

// Emotion to color mapping
const emotionColors: Record<string, string> = {
  happy: '#fbbf24',
  sad: '#3b82f6',
  angry: '#ef4444',
  curious: '#8b5cf6',
  excited: '#f59e0b',
  worried: '#64748b',
  peaceful: '#10b981',
  contemplative: '#6366f1',
  wise: '#14b8a6',
  cheerful: '#f97316',
  ambitious: '#ec4899',
  confused: '#6b7280',
  grateful: '#84cc16',
  nervous: '#a855f7',
  focused: '#0ea5e9',
  patient: '#06b6d4',
  observant: '#8b5cf6'
};

const EmotionalStatePanel: React.FC<EmotionalStateProps> = ({ emotions, selectedEntity }) => {
  // Get emotions for selected entity or show aggregate
  const entityEmotions = selectedEntity ? emotions[selectedEntity] : null;
  
  if (!entityEmotions && selectedEntity) {
    return (
      <Box>
        <Typography variant="h6" gutterBottom>🎭 Emotional State</Typography>
        <Typography variant="body2" color="text.secondary">
          No emotional data for selected entity
        </Typography>
      </Box>
    );
  }
  
  // If no entity selected, show overview
  if (!selectedEntity) {
    const totalCharacters = Object.keys(emotions).length;
    const avgMomentum = totalCharacters > 0
      ? Object.values(emotions).reduce((sum, e) => sum + e.momentum, 0) / totalCharacters
      : 0;
    
    return (
      <Box>
        <Typography variant="h6" gutterBottom>🎭 Emotional Overview</Typography>
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            {totalCharacters} characters tracked
          </Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            Average Momentum: {(avgMomentum * 100).toFixed(0)}%
          </Typography>
        </Box>
        <Typography variant="caption" color="text.secondary">
          Select a character to see detailed emotions
        </Typography>
      </Box>
    );
  }
  
  // Render selected entity's emotions
  const activeEmotions = Object.entries(entityEmotions.active_emotions);
  const surpriseScore = entityEmotions.surprise_score;
  const momentum = entityEmotions.momentum;
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>🎭 Emotional State</Typography>
      
      {/* Active Emotions */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>Active Emotions</Typography>
        {activeEmotions.length === 0 ? (
          <Typography variant="body2" color="text.secondary">No active emotions</Typography>
        ) : (
          activeEmotions.map(([emotion, strength]) => (
            <Box key={emotion} sx={{ mb: 1.5 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                  {emotion}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {(strength * 100).toFixed(0)}%
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={strength * 100}
                sx={{
                  height: 8,
                  borderRadius: 4,
                  backgroundColor: 'rgba(255,255,255,0.1)',
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 4,
                    backgroundColor: emotionColors[emotion] || '#6366f1',
                  }
                }}
              />
            </Box>
          ))
        )}
      </Box>
      
      {/* Surprise & Momentum */}
      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Chip
            label={`Surprise: ${(surpriseScore * 100).toFixed(0)}%`}
            size="small"
            sx={{
              backgroundColor: `rgba(249, 115, 22, ${0.2 + surpriseScore * 0.8})`,
              color: '#ffffff'
            }}
          />
          <Chip
            label={`Momentum: ${(momentum * 100).toFixed(0)}%`}
            size="small"
            sx={{
              backgroundColor: `rgba(139, 92, 246, ${0.2 + momentum * 0.8})`,
              color: '#ffffff'
            }}
          />
        </Box>
      </Box>
    </Box>
  );
};

export default EmotionalStatePanel; 