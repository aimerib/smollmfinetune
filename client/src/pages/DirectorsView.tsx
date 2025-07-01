/**
 * Director's View - Main Page Component
 * 
 * A real-time god-view debugging and storytelling console for monitoring
 * AI character simulations with triple-head architecture insights.
 */

import React, { useEffect, useState, useRef } from 'react';
import { Box, Grid, Paper, Typography, Drawer, IconButton } from '@mui/material';
import { Close as CloseIcon } from '@mui/icons-material';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import Plot from 'react-plotly.js';

// Components
import WorldVisualization from '../components/DirectorsView/WorldVisualization';
import MemoryBubbles from '../components/DirectorsView/MemoryBubbles';
import EmotionalStatePanel from '../components/DirectorsView/EmotionalStatePanel';
import EntityDetailSidebar from '../components/DirectorsView/EntityDetailSidebar';
import SubtextLog from '../components/DirectorsView/SubtextLog';
import TripleHeadMetrics from '../components/DirectorsView/TripleHeadMetrics';
import MemoryTimeline from '../components/DirectorsView/MemoryTimeline';

// Services and stores
import websocketService, { 
  EventType, 
  WorldSnapshot,
  StateUpdateEvent,
  MemoryFormationEvent,
  EmotionChangeEvent,
  TripleHeadMetricsEvent 
} from '../services/websocketService';
import { useDirectorsViewStore } from '../stores/directorsViewStore';

const DirectorsView: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  // Store access
  const {
    worldState,
    memories,
    emotions,
    metrics,
    updateWorldState,
    addMemory,
    updateEmotion,
    updateMetrics,
  } = useDirectorsViewStore();
  
  // Connect to WebSocket on mount
  useEffect(() => {
    const connectWebSocket = async () => {
      try {
        await websocketService.connect();
        setIsConnected(true);
        
        // Subscribe to events
        const unsubscribers = [
          websocketService.on<StateUpdateEvent>(EventType.STATE_UPDATE, (event) => {
            if (event.data.entity_id === 'world' && event.data.changes.snapshot) {
              updateWorldState(event.data.changes.snapshot as WorldSnapshot);
            } else {
              // Handle individual entity updates
              console.log('Entity update:', event);
            }
          }),
          
          websocketService.on<MemoryFormationEvent>(EventType.MEMORY_FORMED, (event) => {
            addMemory(event.data.character_id, {
              id: `mem_${Date.now()}`,
              content: event.data.memory_content,
              importance: event.data.importance,
              emotional_valence: event.data.emotional_valence,
              memory_type: event.data.memory_type,
              timestamp: event.timestamp,
              visualization: event.data.visualization
            });
          }),
          
          websocketService.on<EmotionChangeEvent>(EventType.EMOTION_CHANGED, (event) => {
            updateEmotion(event.data.character_id, {
              active_emotions: event.data.active_emotions,
              surprise_score: event.data.surprise_score,
              momentum: event.data.momentum
            });
          }),
          
          websocketService.on<TripleHeadMetricsEvent>(EventType.TRIPLE_HEAD_METRICS, (event) => {
            updateMetrics(event.data.character_id, event.data.metrics);
          }),
        ];
        
        // Cleanup function
        return () => {
          unsubscribers.forEach(unsubscribe => unsubscribe());
        };
        
      } catch (error) {
        console.error('Failed to connect to WebSocket:', error);
        setIsConnected(false);
      }
    };
    
    connectWebSocket();
    
    return () => {
      websocketService.disconnect();
    };
  }, []);
  
  // Handle entity selection
  const handleEntitySelect = (entityId: string) => {
    setSelectedEntity(entityId);
    setSidebarOpen(true);
    websocketService.getEntityDetails(entityId);
  };
  
  return (
    <Box sx={{ 
      display: 'flex', 
      height: '100vh', 
      backgroundColor: '#0a0a0a',
      color: '#ffffff',
      overflow: 'hidden'
    }}>
      <Grid container spacing={2} sx={{ height: '100%', p: 2 }}>
        {/* Main 3D Visualization */}
        <Grid item xs={12} md={8} sx={{ height: '100%' }}>
          <Paper 
            elevation={3} 
            sx={{ 
              height: '100%', 
              backgroundColor: '#1a1a1a',
              position: 'relative',
              overflow: 'hidden'
            }}
          >
            <Canvas
              camera={{ position: [10, 10, 10], fov: 60 }}
              style={{ background: '#0a0a0a' }}
            >
              <ambientLight intensity={0.5} />
              <pointLight position={[10, 10, 10]} />
              <WorldVisualization 
                worldState={worldState}
                onEntitySelect={handleEntitySelect}
                selectedEntity={selectedEntity}
              />
              <MemoryBubbles 
                memories={memories}
              />
              <OrbitControls 
                enablePan={true}
                enableZoom={true}
                enableRotate={true}
              />
            </Canvas>
            
            {/* Connection Status */}
            <Box
              sx={{
                position: 'absolute',
                top: 16,
                left: 16,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                backgroundColor: 'rgba(0,0,0,0.6)',
                padding: '8px 16px',
                borderRadius: 2
              }}
            >
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: isConnected ? '#4caf50' : '#f44336'
                }}
              />
              <Typography variant="body2">
                {isConnected ? 'Connected' : 'Disconnected'}
              </Typography>
            </Box>
            
            {/* Subtext Log Overlay */}
            <Box
              sx={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                maxHeight: '30%',
                backgroundColor: 'rgba(0,0,0,0.8)',
                borderTop: '1px solid rgba(255,255,255,0.1)'
              }}
            >
              <SubtextLog 
                subtexts={worldState?.recent_subtext || []}
                maxItems={5}
              />
            </Box>
          </Paper>
        </Grid>
        
        {/* Right Panel - Metrics and Controls */}
        <Grid item xs={12} md={4} sx={{ height: '100%' }}>
          <Grid container spacing={2} sx={{ height: '100%' }}>
            {/* Emotional State Panel */}
            <Grid item xs={12} sx={{ height: '40%' }}>
              <Paper 
                elevation={3} 
                sx={{ 
                  height: '100%', 
                  backgroundColor: '#1a1a1a',
                  p: 2,
                  overflow: 'auto'
                }}
              >
                <EmotionalStatePanel 
                  emotions={emotions}
                  selectedEntity={selectedEntity}
                />
              </Paper>
            </Grid>
            
            {/* Triple-Head Metrics */}
            <Grid item xs={12} sx={{ height: '30%' }}>
              <Paper 
                elevation={3} 
                sx={{ 
                  height: '100%', 
                  backgroundColor: '#1a1a1a',
                  p: 2,
                  overflow: 'auto'
                }}
              >
                <TripleHeadMetrics 
                  metrics={metrics}
                  selectedEntity={selectedEntity}
                />
              </Paper>
            </Grid>
            
            {/* Memory Timeline */}
            <Grid item xs={12} sx={{ height: '30%' }}>
              <Paper 
                elevation={3} 
                sx={{ 
                  height: '100%', 
                  backgroundColor: '#1a1a1a',
                  p: 2,
                  overflow: 'auto'
                }}
              >
                <MemoryTimeline 
                  memories={memories}
                  selectedEntity={selectedEntity}
                />
              </Paper>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
      
      {/* Entity Detail Sidebar */}
      <Drawer
        anchor="right"
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        sx={{
          '& .MuiDrawer-paper': {
            width: 400,
            backgroundColor: '#1a1a1a',
            color: '#ffffff'
          }
        }}
      >
        <Box sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">Entity Details</Typography>
            <IconButton onClick={() => setSidebarOpen(false)}>
              <CloseIcon />
            </IconButton>
          </Box>
          {selectedEntity && (
            <EntityDetailSidebar 
              entityId={selectedEntity}
              worldState={worldState}
            />
          )}
        </Box>
      </Drawer>
    </Box>
  );
};

export default DirectorsView; 