/**
 * Entity Detail Sidebar Component
 * 
 * Shows detailed information about a selected entity (character or location).
 */

import React from 'react';
import { Box, Typography, Divider, Chip } from '@mui/material';
import { LocationOn, Person, Mood, Psychology } from '@mui/icons-material';
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
    <Box>
      {/* Entity Type Icon */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        {isCharacter ? (
          <Person sx={{ mr: 1, color: 'primary.main' }} />
        ) : (
          <LocationOn sx={{ mr: 1, color: 'secondary.main' }} />
        )}
        <Typography variant="h5">
          {entity.name}
        </Typography>
      </Box>
      
      {/* Entity ID */}
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
        ID: {entity.id}
      </Typography>
      
      <Divider sx={{ mb: 2 }} />
      
      {/* Character-specific details */}
      {isCharacter && character && (
        <>
          {/* Location */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              <LocationOn fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
              Location
            </Typography>
            <Typography variant="body2">
              {character.location}
            </Typography>
          </Box>
          
          {/* Mood */}
          {character.attributes?.mood && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                <Mood fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                Current Mood
              </Typography>
              <Chip 
                label={character.attributes.mood}
                size="small"
                sx={{ textTransform: 'capitalize' }}
              />
            </Box>
          )}
          
          {/* Personality Traits */}
          {character.attributes?.personality_traits && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                <Psychology fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
                Personality Traits
              </Typography>
              <Box sx={{ mt: 1 }}>
                {Object.entries(character.attributes.personality_traits).map(([trait, value]) => (
                  <Box key={trait} sx={{ mb: 0.5 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>
                        {trait.replace('_', ' ')}
                      </Typography>
                      <Typography variant="caption" color="primary">
                        {(value * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                  </Box>
                ))}
              </Box>
            </Box>
          )}
          
          {/* Stats */}
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Statistics
            </Typography>
            <Typography variant="body2">
              Memories: {character.memory_count}
            </Typography>
            <Typography variant="body2">
              Relationships: {character.relationship_count}
            </Typography>
          </Box>
        </>
      )}
      
      {/* Location-specific details */}
      {!isCharacter && location && (
        <>
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Type
            </Typography>
            <Typography variant="body2">
              {location.type}
            </Typography>
          </Box>
          
          {/* Characters at this location */}
          <Box sx={{ mb: 2 }}>
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
                  sx={{ mr: 0.5, mb: 0.5 }}
                />
              ))
            }
          </Box>
        </>
      )}
      
      {/* Additional attributes */}
      {entity.attributes && Object.keys(entity.attributes).length > 0 && (
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Attributes
          </Typography>
          <Box sx={{ fontSize: '0.875rem' }}>
            <pre style={{ margin: 0, fontFamily: 'monospace' }}>
              {JSON.stringify(entity.attributes, null, 2)}
            </pre>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default EntityDetailSidebar; 