/**
 * Entity Detail Sidebar Component
 * 
 * Shows detailed information about a selected entity (character or location).
 */

import React from 'react';
import { Typography, Chip } from '@mui/material';
import { Person, LocationOn, Mood, Psychology } from '@mui/icons-material';
import { WorldSnapshot } from '../../services/websocketService';

interface EntityDetailSidebarProps {
  entityId: string;
  worldState: WorldSnapshot | null;
}

const EntityDetailSidebar: React.FC<EntityDetailSidebarProps> = ({ entityId, worldState }) => {
  if (!worldState) {
    return (
      <Typography variant="body2" color="text.secondary">
        Loading world data...
      </Typography>
    );
  }
  
  // Find entity in world state
  const character = worldState.characters.find(c => c.id === entityId);
  const location = worldState.locations.find(l => l.id === entityId);
  const entity = character || location;
  
  if (!entity) {
    return (
      <Typography variant="body2" color="text.secondary">
        Entity not found
      </Typography>
    );
  }
  
  const isCharacter = entity.type === 'character';
  
  return (
    <div style={{ padding: '16px' }}>
      {/* Entity Type Icon */}
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '16px' }}>
        {isCharacter ? (
          <Person style={{ marginRight: '8px', color: '#1976d2' }} />
        ) : (
          <LocationOn style={{ marginRight: '8px', color: '#9c27b0' }} />
        )}
        <Typography variant="h5">
          {entity.name}
        </Typography>
      </div>
      
      {/* Entity ID */}
      <Typography variant="caption" color="text.secondary" style={{ display: 'block', marginBottom: '16px' }}>
        ID: {entity.id}
      </Typography>
      
      <hr style={{ margin: '16px 0', border: 'none', borderTop: '1px solid #424242' }} />
      
      {/* Character-specific details */}
      {isCharacter && character && (
        <div>
          {/* Location */}
          <div style={{ marginBottom: '16px' }}>
            <Typography variant="subtitle2" gutterBottom>
              <LocationOn fontSize="small" style={{ verticalAlign: 'middle', marginRight: '4px' }} />
              Location
            </Typography>
            <Typography variant="body2">
              {character.location}
            </Typography>
          </div>
          
          {/* Mood */}
          {character.custom_data?.mood && (
            <div style={{ marginBottom: '16px' }}>
              <Typography variant="subtitle2" gutterBottom>
                <Mood fontSize="small" style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                Current Mood
              </Typography>
              <Chip 
                label={character.custom_data.mood}
                size="small"
                style={{ textTransform: 'capitalize' }}
              />
            </div>
          )}
          
          {/* Personality Traits */}
          {character.custom_data?.personality_traits && (
            <div style={{ marginBottom: '16px' }}>
              <Typography variant="subtitle2" gutterBottom>
                <Psychology fontSize="small" style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                Personality Traits
              </Typography>
              <div style={{ marginTop: '8px' }}>
                {Object.entries(character.custom_data.personality_traits).map(([trait, value]) => (
                  <div key={trait} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <Typography variant="body2" style={{ textTransform: 'capitalize', opacity: 0.8 }}>
                      {trait}
                    </Typography>
                    <Typography variant="caption" color="primary">
                      {typeof value === 'number' ? (value * 100).toFixed(0) : '0'}%
                    </Typography>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Stats */}
          <div style={{ marginBottom: '16px' }}>
            <Typography variant="subtitle2" gutterBottom>
              Statistics
            </Typography>
            <Typography variant="body2">
              Memories: {character.memory_count}
            </Typography>
            <Typography variant="body2">
              Relationships: {character.relationship_count}
            </Typography>
          </div>
        </div>
      )}
      
      {/* Location-specific details */}
      {!isCharacter && location && (
        <div>
          <div style={{ marginBottom: '16px' }}>
            <Typography variant="subtitle2" gutterBottom>
              Type
            </Typography>
            <Typography variant="body2">
              {location.type}
            </Typography>
          </div>
          
          {/* Characters at this location */}
          <div style={{ marginBottom: '16px' }}>
            <Typography variant="subtitle2" gutterBottom>
              Characters Present
            </Typography>
            {worldState.characters
              .filter(c => c.location === location.id)
              .map(c => (
                <Chip
                  key={c.id}
                  label={c.name}
                  size="small"
                  style={{ marginRight: '4px', marginBottom: '4px' }}
                />
              ))
            }
          </div>
        </div>
      )}
      
      {/* Additional attributes */}
      {entity.custom_data && Object.keys(entity.custom_data).length > 0 && (
        <div>
          <Typography variant="subtitle2" gutterBottom>
            Attributes
          </Typography>
          <div style={{ fontSize: '0.875rem' }}>
            <pre style={{ margin: 0, fontFamily: 'monospace' }}>
              {JSON.stringify(entity.custom_data, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};

export default EntityDetailSidebar; 