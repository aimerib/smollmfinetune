/**
 * Emotional State Panel Component
 * 
 * Displays character emotional states with real-time decay visualization
 * and surprise score tracking.
 */

import React from 'react';
import { Typography, LinearProgress, Chip } from '@mui/material';
import { useDirectorsViewStore, selectCharacterEmotions } from '../../stores/directorsViewStore';

interface EmotionalStatePanelProps {
  selectedEntityId: string | null;
}

const EmotionalStatePanel: React.FC<EmotionalStatePanelProps> = ({ selectedEntityId }) => {
  const selectedEntity = useDirectorsViewStore((state) => 
    selectedEntityId 
      ? state.worldState?.characters.find(c => c.id === selectedEntityId) ||
        state.worldState?.locations.find(l => l.id === selectedEntityId)
      : null
  );

  const emotions = useDirectorsViewStore(
    selectedEntityId ? selectCharacterEmotions(selectedEntityId) : () => null
  );

  const isCharacter = selectedEntity && 'name' in selectedEntity && 'location' in selectedEntity;

  if (!selectedEntity) {
    return (
      <div>
        <Typography variant="h6" gutterBottom>🎭 Emotional State</Typography>
        <Typography variant="body2" color="text.secondary">
          Select an entity to view emotional state
        </Typography>
      </div>
    );
  }

  // If no emotion data for character (only characters have emotions)
  if (!isCharacter || !emotions) {
    return (
      <div>
        <Typography variant="h6" gutterBottom>🎭 Emotional State</Typography>
        <Typography variant="body2" color="text.secondary">
          No emotional data for selected entity
        </Typography>
      </div>
    );
  }

  // Render selected entity's emotions
  const activeEmotions = Object.entries(emotions.active_emotions);
  const surpriseScore = emotions.surprise_score;
  const momentum = emotions.momentum;

  return (
    <div style={{ padding: '16px' }}>
      <Typography variant="h6" gutterBottom>🎭 Emotional State</Typography>
      
      {/* Active Emotions */}
      <div style={{ marginBottom: '16px' }}>
        <Typography variant="subtitle2" gutterBottom>Active Emotions</Typography>
        {activeEmotions.length > 0 ? (
          activeEmotions.map(([emotion, intensity]) => (
            <div key={emotion} style={{ marginBottom: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                <Typography variant="body2" style={{ textTransform: 'capitalize' }}>
                  {emotion}
                </Typography>
                <Typography variant="caption">
                  {(intensity * 100).toFixed(0)}%
                </Typography>
              </div>
              <LinearProgress 
                variant="determinate" 
                value={intensity * 100} 
                style={{ height: '6px', borderRadius: '3px' }}
              />
            </div>
          ))
        ) : (
          <Typography variant="body2" color="text.secondary">
            No active emotions
          </Typography>
        )}
      </div>

      {/* Metrics */}
      <div style={{ marginBottom: '16px' }}>
        <Typography variant="subtitle2" gutterBottom>Metrics</Typography>
        
        <div style={{ marginBottom: '8px' }}>
          <Typography variant="body2">Surprise Score</Typography>
          <LinearProgress 
            variant="determinate" 
            value={surpriseScore * 100} 
            style={{ height: '6px', borderRadius: '3px', marginTop: '4px' }}
          />
        </div>
        
        <div style={{ marginBottom: '8px' }}>
          <Typography variant="body2">Momentum</Typography>
          <LinearProgress 
            variant="determinate" 
            value={momentum * 100} 
            style={{ height: '6px', borderRadius: '3px', marginTop: '4px' }}
          />
        </div>
      </div>

      {/* Emotional State Chip */}
      <div>
        <Chip
          label={`${activeEmotions.length} Active Emotions`}
          size="small"
          color="primary"
          variant="outlined"
        />
      </div>
    </div>
  );
};

export default EmotionalStatePanel; 