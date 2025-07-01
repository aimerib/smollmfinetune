/**
 * Memory Timeline Component
 * 
 * Shows a chronological timeline of character memories.
 */

import React from 'react';
import { Box, Typography } from '@mui/material';

interface MemoryTimelineProps {
  memories: Record<string, any[]>;
  selectedEntity: string | null;
}

const MemoryTimeline: React.FC<MemoryTimelineProps> = ({ memories, selectedEntity }) => {
  const entityMemories = selectedEntity ? memories[selectedEntity] : null;
  
  return (
    <Box>
      <Typography variant="h6" gutterBottom>📚 Memory Timeline</Typography>
      {entityMemories && entityMemories.length > 0 ? (
        <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
          {entityMemories.slice(0, 5).map((memory, index) => (
            <Box key={memory.id || index} sx={{ mb: 1 }}>
              <Typography variant="caption" color="text.secondary">
                {memory.memory_type} • {(memory.importance * 100).toFixed(0)}%
              </Typography>
              <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                {memory.content}
              </Typography>
            </Box>
          ))}
        </Box>
      ) : (
        <Typography variant="body2" color="text.secondary">
          {selectedEntity ? 'No memories yet' : 'Select a character to view memories'}
        </Typography>
      )}
    </Box>
  );
};

export default MemoryTimeline; 