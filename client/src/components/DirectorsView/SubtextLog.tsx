/**
 * Subtext Log Component
 * 
 * Displays a scrollable log of character internal monologue and thoughts.
 */

import React, { useEffect, useRef } from 'react';
import { Box, Typography } from '@mui/material';
import { motion, AnimatePresence } from 'framer-motion';
import { format } from 'date-fns';

interface SubtextEntry {
  agent_id: string;
  text: string;
  timestamp: string;
}

interface SubtextLogProps {
  subtexts: SubtextEntry[];
  maxItems?: number;
}

const SubtextLog: React.FC<SubtextLogProps> = ({ subtexts, maxItems = 10 }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when new subtexts arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [subtexts]);
  
  // Limit displayed items
  const displayedSubtexts = subtexts.slice(-maxItems);
  
  return (
    <Box 
      ref={scrollRef}
      sx={{ 
        height: '100%',
        overflowY: 'auto',
        p: 2,
        '&::-webkit-scrollbar': {
          width: '8px',
        },
        '&::-webkit-scrollbar-track': {
          background: 'rgba(255,255,255,0.05)',
        },
        '&::-webkit-scrollbar-thumb': {
          background: 'rgba(255,255,255,0.2)',
          borderRadius: '4px',
        },
        '&::-webkit-scrollbar-thumb:hover': {
          background: 'rgba(255,255,255,0.3)',
        },
      }}
    >
      <Typography variant="subtitle2" sx={{ mb: 1, opacity: 0.7 }}>
        Internal Monologue
      </Typography>
      
      <AnimatePresence initial={false}>
        {displayedSubtexts.map((entry, index) => (
          <motion.div
            key={`${entry.agent_id}-${entry.timestamp}-${index}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
          >
            <Box
              sx={{
                mb: 1.5,
                p: 1,
                borderLeft: '3px solid',
                borderLeftColor: 'primary.main',
                backgroundColor: 'rgba(255,255,255,0.02)',
                borderRadius: '0 4px 4px 0',
                '&:hover': {
                  backgroundColor: 'rgba(255,255,255,0.05)',
                }
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography
                  variant="caption"
                  sx={{ 
                    color: 'primary.light',
                    fontWeight: 600,
                    textTransform: 'capitalize'
                  }}
                >
                  {entry.agent_id.replace(/_/g, ' ')}
                </Typography>
                <Typography variant="caption" sx={{ opacity: 0.5 }}>
                  {format(new Date(entry.timestamp), 'HH:mm:ss')}
                </Typography>
              </Box>
              <Typography
                variant="body2"
                sx={{
                  fontStyle: 'italic',
                  opacity: 0.9,
                  lineHeight: 1.4
                }}
              >
                {entry.text}
              </Typography>
            </Box>
          </motion.div>
        ))}
      </AnimatePresence>
      
      {displayedSubtexts.length === 0 && (
        <Typography
          variant="body2"
          sx={{
            opacity: 0.5,
            textAlign: 'center',
            mt: 4
          }}
        >
          No subtext entries yet...
        </Typography>
      )}
    </Box>
  );
};

export default SubtextLog; 