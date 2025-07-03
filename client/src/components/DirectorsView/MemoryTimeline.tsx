/**
 * Memory Timeline Component
 * 
 * Shows a chronological timeline of character memories.
 */

import React from 'react';
import { Box, Typography } from '@mui/material';

interface Memory {
  id?: string;
  memory_type?: string;
  importance?: number;
  content?: string;
}

interface MemoryTimelineProps {
  memories: Record<string, Memory[]>;
  selectedEntity: string | null;
}

const MemoryTimeline: React.FC<MemoryTimelineProps> = ({ memories, selectedEntity }) => {
  if (!selectedEntity) {
    return (
      <Box component="div">
        <Typography variant="h6" gutterBottom>📚 Memory Timeline</Typography>
        <Typography variant="body2" color="text.secondary">
          Select a character to view memories
        </Typography>
      </Box>
    );
  }

  const entityMemories: Memory[] = memories[selectedEntity] || [];

  if (entityMemories.length === 0) {
    return (
      <Box component="div">
        <Typography variant="h6" gutterBottom>📚 Memory Timeline</Typography>
        <Typography variant="body2" color="text.secondary">
          No memories yet
        </Typography>
      </Box>
    );
  }

  return (
    <Box component="div">
      <Typography variant="h6" gutterBottom>📚 Memory Timeline</Typography>
      <Box component="div" sx={{ maxHeight: 200, overflowY: 'auto' }}>
        {entityMemories.slice(0, 5).map((memory, index) => (
          <Box key={memory.id || `memory-${index}`} component="div" sx={{ mb: 1 }}>
            <Typography variant="caption" color="text.secondary">
              {memory.memory_type || 'Unknown'} • {((memory.importance || 0) * 100).toFixed(0)}%
            </Typography>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              {memory.content || 'No content'}
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  );
};

export default MemoryTimeline; 