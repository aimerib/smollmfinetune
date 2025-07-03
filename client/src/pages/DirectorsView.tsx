/**
 * Director's View - Elegant Real-time Monitoring Console
 * 
 * Design Principles:
 * - Progressive Disclosure: Show complexity only when needed
 * - Visual Hierarchy: Most important info is most prominent
 * - Intuitive Interactions: Everything works as expected
 */

import React, { useEffect, useState, useRef } from 'react';
import styled from '@emotion/styled';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FiSearch, FiFilter, FiPlay, FiPause, FiSkipBack, 
  FiSkipForward, FiMaximize2, FiMinimize2, FiActivity,
  FiUser, FiMapPin, FiHeart, FiInfo, FiClock
} from 'react-icons/fi';

// Store and WebSocket
import websocketService, { EventType } from '../services/websocketService';
import { useDirectorsViewStore } from '../stores/directorsViewStore';

// Relationship Components
import { RelationshipGraph } from '../components/DirectorsView/RelationshipGraph';
import { RelationshipPanel, RelationshipMetrics } from '../components/DirectorsView/RelationshipPanel';
import { RelationshipTimeline } from '../components/DirectorsView/RelationshipTimeline';
import { RelationshipNode, RelationshipEdge } from '../types/relationships';

// Styled Components with elegant dark theme
const Container = styled.div`
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #0a0a0f;
  color: #ffffff;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
`;

const TopBar = styled.div`
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 2rem;
  background: rgba(255, 255, 255, 0.02);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
`;

const SearchBar = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  min-width: 300px;
  
  input {
    flex: 1;
    background: none;
    border: none;
    color: white;
    outline: none;
    
    &::placeholder {
      color: rgba(255, 255, 255, 0.4);
    }
  }
`;

const ViewToggle = styled.div`
  display: flex;
  gap: 0.5rem;
`;

const ToggleButton = styled.button<{ active?: boolean }>`
  padding: 0.5rem 1rem;
  background: ${props => props.active ? 'rgba(139, 92, 246, 0.2)' : 'transparent'};
  border: 1px solid ${props => props.active ? '#8b5cf6' : 'rgba(255, 255, 255, 0.1)'};
  color: ${props => props.active ? '#8b5cf6' : 'rgba(255, 255, 255, 0.6)'};
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-size: 0.875rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  &:hover {
    background: ${props => props.active ? 'rgba(139, 92, 246, 0.3)' : 'rgba(255, 255, 255, 0.05)'};
  }
`;

const MainContent = styled.div`
  flex: 1;
  display: flex;
  overflow: hidden;
`;

const WorldCanvas = styled.div`
  flex: 1;
  position: relative;
  background: radial-gradient(ellipse at center, rgba(139, 92, 246, 0.03) 0%, transparent 70%);
  overflow: hidden;
`;

const ContextPanel = styled(motion.div)<{ isOpen: boolean }>`
  width: ${props => props.isOpen ? '320px' : '0'};
  background: rgba(255, 255, 255, 0.02);
  border-left: 1px solid rgba(255, 255, 255, 0.05);
  overflow: hidden;
  transition: width 0.3s ease;
`;

const Timeline = styled.div`
  height: 80px;
  background: rgba(255, 255, 255, 0.02);
  border-top: 1px solid rgba(255, 255, 255, 0.05);
  display: flex;
  align-items: center;
  padding: 0 2rem;
  gap: 2rem;
`;

const PlaybackControls = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const IconButton = styled.button`
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: transparent;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.2);
  }
`;

const EventStream = styled.div`
  flex: 1;
  height: 50px;
  position: relative;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 25px;
  overflow: hidden;
`;

const StatusBadge = styled.div<{ type: 'online' | 'offline' }>`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.25rem 0.75rem;
  background: ${props => props.type === 'online' ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)'};
  border: 1px solid ${props => props.type === 'online' ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'};
  border-radius: 20px;
  font-size: 0.75rem;
  color: ${props => props.type === 'online' ? '#22c55e' : '#ef4444'};
`;

// 2D World Map Component
const WorldMap = styled.div`
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const Location = styled(motion.div)<{ selected?: boolean }>`
  position: absolute;
  width: 180px;
  height: 100px;
  background: ${props => props.selected 
    ? 'linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(139, 92, 246, 0.05))' 
    : 'rgba(255, 255, 255, 0.02)'};
  border: 1px solid ${props => props.selected ? '#8b5cf6' : 'rgba(255, 255, 255, 0.1)'};
  border-radius: 12px;
  padding: 1rem;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
  }
`;

const LocationName = styled.div`
  font-weight: 600;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const CharacterAvatar = styled(motion.div)<{ emotion?: string }>`
  position: absolute;
  width: 40px;
  height: 40px;
  background: ${props => {
    const emotionColors: Record<string, string> = {
      happy: '#fbbf24',
      sad: '#3b82f6',
      angry: '#ef4444',
      neutral: '#8b5cf6'
    };
    return emotionColors[props.emotion || 'neutral'] || '#8b5cf6';
  }};
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  cursor: pointer;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  z-index: 10;
  
  &:hover {
    transform: scale(1.1);
  }
`;

const MemoryBubble = styled(motion.div)<{ valence: number }>`
  position: absolute;
  padding: 0.5rem 1rem;
  background: ${props => props.valence > 0 
    ? 'rgba(251, 191, 36, 0.1)' 
    : 'rgba(59, 130, 246, 0.1)'};
  border: 1px solid ${props => props.valence > 0 
    ? 'rgba(251, 191, 36, 0.3)' 
    : 'rgba(59, 130, 246, 0.3)'};
  border-radius: 20px;
  font-size: 0.875rem;
  pointer-events: none;
  z-index: 20;
`;

const PanelSection = styled.div`
  padding: 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
`;

const SectionTitle = styled.h3`
  font-size: 0.875rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.6);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 1rem;
`;

const MetricCard = styled.div`
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 0.5rem;
`;

const MetricLabel = styled.div`
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 0.25rem;
`;

const MetricValue = styled.div`
  font-size: 1.25rem;
  font-weight: 600;
`;

const ProgressBar = styled.div<{ progress: number; color?: string }>`
  height: 4px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
  margin-top: 0.5rem;
  overflow: hidden;
  
  &::after {
    content: '';
    display: block;
    height: 100%;
    width: ${props => props.progress}%;
    background: ${props => props.color || '#8b5cf6'};
    transition: width 0.3s ease;
  }
`;

// Add new styled components after the existing ones
const ConnectionLine = styled.svg`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 5;
`;

const ActivityIndicator = styled(motion.div)`
  position: absolute;
  width: 8px;
  height: 8px;
  background: #22c55e;
  border-radius: 50%;
  box-shadow: 0 0 10px rgba(34, 197, 94, 0.5);
  pointer-events: none;
  z-index: 15;
`;

const QuickAction = styled.button`
  padding: 0.5rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 6px;
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.1);
    color: white;
  }
`;

const KeyboardHint = styled.div`
  position: fixed;
  bottom: 100px;
  right: 20px;
  padding: 0.5rem 1rem;
  background: rgba(0, 0, 0, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.6);
  backdrop-filter: blur(10px);
`;

const ZoomControls = styled.div`
  position: absolute;
  bottom: 20px;
  right: 20px;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const DirectorsView: React.FC = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [isPlaying, setIsPlaying] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [zoom, setZoom] = useState(1);
  const [showKeyboardHints, setShowKeyboardHints] = useState(false);
  const [activeFilters, setActiveFilters] = useState({
    memories: true,
    emotions: true,
    metrics: false,
    connections: true,
    relationships: false
  });
  
  // Relationship visualization state
  const [viewMode, setViewMode] = useState<'world' | 'relationships'>('world');
  const [relationshipNodes, setRelationshipNodes] = useState<RelationshipNode[]>([]);
  const [relationshipEdges, setRelationshipEdges] = useState<RelationshipEdge[]>([]);
  const [selectedRelationship, setSelectedRelationship] = useState<RelationshipEdge | null>(null);
  const [affinityFilter, setAffinityFilter] = useState({ min: -1, max: 1 });

  const { worldState, memories, updateWorldState } = useDirectorsViewStore();
  const worldMapRef = useRef<HTMLDivElement>(null);

  // Connect WebSocket
  useEffect(() => {
    const connect = async () => {
      try {
        await websocketService.connect();
        setIsConnected(true);
      } catch (error) {
        console.error('Failed to connect:', error);
      }
    };
    connect();
    
    // Subscribe to relationship events
    const unsubscribeRelationship = websocketService.on(
      EventType.RELATIONSHIP_UPDATE,
      (event: any) => {
        // Update relationship edges when we receive updates
        setRelationshipEdges(prev => {
          const updated = [...prev];
          const index = updated.findIndex(
            e => e.source === (event.speaker_id || event.source) && e.target === (event.target_id || event.target)
          );
          
          if (index >= 0) {
            // Update existing relationship
            updated[index] = {
              ...updated[index],
              affinity: updated[index].affinity + (event.affinity_change || 0),
              interaction_count: updated[index].interaction_count + 1,
              last_interaction: event.timestamp,
              emotional_history: [...updated[index].emotional_history, ...(event.emotional_impact || [])]
            };
          }
          
          return updated;
        });
      }
    );
    
    return () => {
      websocketService.disconnect();
      unsubscribeRelationship();
    };
  }, []);
  
  // Fetch relationship data when switching to relationship mode
  useEffect(() => {
    const fetchRelationshipData = async () => {
      if (viewMode === 'relationships') {
        try {
          const response = await fetch('http://localhost:8000/api/relationships/graph');
          if (response.ok) {
            const data = await response.json();
            setRelationshipNodes(data.nodes);
            setRelationshipEdges(data.edges);
          }
        } catch (error) {
          console.error('Failed to fetch relationship data:', error);
          // Fallback to extracting from world state if API fails
          if (worldState) {
            const nodes: RelationshipNode[] = worldState.characters.map(char => ({
              id: char.id,
              name: char.name,
              personality: {
                openness: char.custom_data?.personality?.openness || 0.5,
                conscientiousness: char.custom_data?.personality?.conscientiousness || 0.5,
                extraversion: char.custom_data?.personality?.extraversion || 0.5,
                agreeableness: char.custom_data?.personality?.agreeableness || 0.5,
                neuroticism: char.custom_data?.personality?.neuroticism || 0.5
              },
              emotional_state: char.custom_data?.emotional_state || { neutral: 1.0 },
              position: { x: Math.random() * 800, y: Math.random() * 600 },
              size: 40 + (char.relationship_count || 0) * 5
            }));
            
            setRelationshipNodes(nodes);
            setRelationshipEdges([]);
          }
        }
      }
    };
    
    fetchRelationshipData();
  }, [viewMode, worldState]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      switch(e.key) {
        case ' ':
          e.preventDefault();
          setIsPlaying(prev => !prev);
          break;
        case 'Escape':
          setSelectedEntity(null);
          break;
        case 'p':
          setIsPanelOpen(prev => !prev);
          break;
        case 'm':
          setActiveFilters(prev => ({ ...prev, memories: !prev.memories }));
          break;
        case 'e':
          setActiveFilters(prev => ({ ...prev, emotions: !prev.emotions }));
          break;
        case 'c':
          setActiveFilters(prev => ({ ...prev, connections: !prev.connections }));
          break;
        case 'r':
          setViewMode(prev => prev === 'relationships' ? 'world' : 'relationships');
          break;
        case '?':
          setShowKeyboardHints(prev => !prev);
          break;
        case '+':
        case '=':
          setZoom(prev => Math.min(prev + 0.1, 2));
          break;
        case '-':
          setZoom(prev => Math.max(prev - 0.1, 0.5));
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);

  // Calculate location positions in a nice layout
  const getLocationPosition = (index: number, total: number): { left: string; top: string } => {
    const angle = (index / total) * Math.PI * 2;
    const radius = 250;
    const centerX = 50;
    const centerY = 50;
    
    return {
      left: `${centerX + Math.cos(angle) * radius / 5}%`,
      top: `${centerY + Math.sin(angle) * radius / 5}%`
    };
  };

  // Get character position based on location
  const getCharacterPosition = (locationId: string, charIndex: number) => {
    const location = worldState?.locations.find(l => l.id === locationId);
    const locationIndex = worldState?.locations.findIndex(l => l.id === locationId) || 0;
    const pos = getLocationPosition(locationIndex, worldState?.locations.length || 1);
    
    const angle = (charIndex / 4) * Math.PI * 2;
    const radius = 30;
    
    return {
      left: `calc(${pos.left} + ${Math.cos(angle) * radius}px)`,
      top: `calc(${pos.top} + ${Math.sin(angle) * radius + 40}px)`
    };
  };

  // Get connection path between two characters
  const getConnectionPath = (char1Id: string, char2Id: string) => {
    const char1 = worldState?.characters.find(c => c.id === char1Id);
    const char2 = worldState?.characters.find(c => c.id === char2Id);
    
    if (!char1 || !char2) return null;
    
    const char1Index = worldState?.characters.findIndex(c => c.id === char1Id) || 0;
    const char2Index = worldState?.characters.findIndex(c => c.id === char2Id) || 0;
    
    const pos1 = getCharacterPosition(char1.location, char1Index);
    const pos2 = getCharacterPosition(char2.location, char2Index);
    
    return { pos1, pos2 };
  };

  const selectedCharacter = worldState?.characters.find(c => c.id === selectedEntity);
  const selectedLocation = worldState?.locations.find(l => l.id === selectedEntity);
  const selectedData = selectedCharacter || selectedLocation;

  return (
    <Container>
      <TopBar>
        <SearchBar>
          {React.createElement(FiSearch as React.ComponentType<any>, { size: 16 })}
          <input 
            type="text" 
            placeholder="Search entities, memories, or events..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </SearchBar>
        
        <ViewToggle>
          <ToggleButton 
            active={activeFilters.memories}
            onClick={() => setActiveFilters(prev => ({ ...prev, memories: !prev.memories }))}
            title="Toggle Memories (M)"
          >
            {React.createElement(FiInfo as React.ComponentType<any>, { size: 16 })}
            Memories
          </ToggleButton>
          <ToggleButton 
            active={activeFilters.emotions}
            onClick={() => setActiveFilters(prev => ({ ...prev, emotions: !prev.emotions }))}
            title="Toggle Emotions (E)"
          >
            {React.createElement(FiHeart as React.ComponentType<any>, { size: 16 })}
            Emotions
          </ToggleButton>
          <ToggleButton 
            active={activeFilters.connections}
            onClick={() => setActiveFilters(prev => ({ ...prev, connections: !prev.connections }))}
            title="Toggle Connections (C)"
          >
            {React.createElement(FiUser as React.ComponentType<any>, { size: 16 })}
            Connections
          </ToggleButton>
          <ToggleButton 
            active={activeFilters.metrics}
            onClick={() => setActiveFilters(prev => ({ ...prev, metrics: !prev.metrics }))}
          >
            {React.createElement(FiActivity as React.ComponentType<any>, { size: 16 })}
            Metrics
          </ToggleButton>
          <ToggleButton 
            active={viewMode === 'relationships'}
            onClick={() => setViewMode(viewMode === 'relationships' ? 'world' : 'relationships')}
            title="Toggle Relationship View (R)"
          >
            {React.createElement(FiUser as React.ComponentType<any>, { size: 16 })}
            Relationships
          </ToggleButton>
        </ViewToggle>
        
        <StatusBadge type={isConnected ? 'online' : 'offline'}>
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor' }} />
          {isConnected ? 'Connected' : 'Disconnected'}
        </StatusBadge>
      </TopBar>

      <MainContent>
        <WorldCanvas>
          {viewMode === 'world' ? (
            <WorldMap 
              ref={worldMapRef}
              style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
            >
            {/* Render Connection Lines */}
            {activeFilters.connections && (
              <ConnectionLine>
                <defs>
                  <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.2" />
                    <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.5" />
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.2" />
                  </linearGradient>
                </defs>
                {/* Example connection - in real app, this would be based on actual relationships */}
                {worldState?.characters.slice(0, 2).length === 2 && (
                  <motion.line
                    x1="40%"
                    y1="40%"
                    x2="60%"
                    y2="60%"
                    stroke="url(#connectionGradient)"
                    strokeWidth="2"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 1 }}
                  />
                )}
              </ConnectionLine>
            )}
            
            {/* Render Locations */}
            {worldState?.locations.map((location, index) => {
              const pos = getLocationPosition(index, worldState.locations.length);
              const charactersHere = worldState.characters.filter(c => c.location === location.id);
              const hasActivity = Math.random() > 0.7; // In real app, based on actual activity
              
              return (
                <Location
                  key={location.id}
                  style={{ left: pos.left, top: pos.top }}
                  selected={selectedEntity === location.id}
                  onClick={() => setSelectedEntity(location.id)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {hasActivity && (
                    <ActivityIndicator
                      style={{ top: -5, right: -5 }}
                      animate={{
                        scale: [1, 1.5, 1],
                        opacity: [1, 0.5, 1]
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity
                      }}
                    />
                  )}
                  <LocationName>
                    {React.createElement(FiMapPin as React.ComponentType<any>, { size: 14 })}
                    {location.name}
                  </LocationName>
                  <div style={{ fontSize: '0.75rem', color: 'rgba(255, 255, 255, 0.5)' }}>
                    {charactersHere.length} character{charactersHere.length !== 1 ? 's' : ''}
                  </div>
                </Location>
              );
            })}
            
            {/* Render Characters */}
            {worldState?.characters.map((character, charIndex) => {
              const pos = getCharacterPosition(character.location, charIndex);
              const emotion = character.custom_data?.mood || 'neutral';
              
              return (
                <CharacterAvatar
                  key={character.id}
                  style={{ left: pos.left, top: pos.top }}
                  emotion={emotion}
                  onClick={() => setSelectedEntity(character.id)}
                  whileHover={{ scale: 1.15 }}
                  whileTap={{ scale: 0.95 }}
                  animate={{
                    y: [0, -5, 0],
                  }}
                  transition={{
                    y: {
                      duration: 3,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }
                  }}
                >
                  {character.name.charAt(0)}
                </CharacterAvatar>
              );
            })}
            
            {/* Render Memory Bubbles */}
            <AnimatePresence>
              {activeFilters.memories && Object.entries(memories).map(([charId, charMemories]) => 
                charMemories.slice(0, 1).map(memory => {
                  const character = worldState?.characters.find(c => c.id === charId);
                  if (!character) return null;
                  const pos = getCharacterPosition(character.location, 0);
                  
                  return (
                    <MemoryBubble
                      key={memory.id}
                      valence={memory.emotional_valence}
                      initial={{ opacity: 0, y: 20, scale: 0.8 }}
                      animate={{ opacity: 1, y: -30, scale: 1 }}
                      exit={{ opacity: 0, y: -60, scale: 0.6 }}
                      transition={{ duration: 2 }}
                      style={{ 
                        left: pos.left, 
                        top: `calc(${pos.top} - 20px)`
                      }}
                    >
                      {memory.content.slice(0, 50)}...
                    </MemoryBubble>
                  );
                })
              )}
            </AnimatePresence>
          </WorldMap>
          ) : (
            // Relationship View Mode
            <RelationshipGraph
              nodes={relationshipNodes}
              edges={relationshipEdges}
              onNodeSelect={(nodeId) => setSelectedEntity(nodeId)}
              onEdgeSelect={(edge) => setSelectedRelationship(edge)}
              affinityFilter={affinityFilter}
            />
          )}
        </WorldCanvas>

        <ContextPanel isOpen={isPanelOpen}>
          {viewMode === 'relationships' ? (
            // Relationship Mode Panel
            <>
              {selectedRelationship ? (
                <RelationshipPanel 
                  selectedRelationship={selectedRelationship}
                  relationshipHistory={[]} // TODO: Fetch from WebSocket
                />
              ) : (
                <RelationshipMetrics 
                  metrics={{
                    totalRelationships: relationshipEdges.length,
                    averageAffinity: relationshipEdges.reduce((sum, edge) => sum + edge.affinity, 0) / (relationshipEdges.length || 1),
                    strongBonds: relationshipEdges.filter(e => e.affinity > 0.7).length,
                    conflicts: relationshipEdges.filter(e => e.affinity < -0.3).length,
                    recentChanges: [],
                    socialClusters: []
                  }}
                />
              )}
              
              {selectedEntity && (
                <PanelSection>
                  <SectionTitle>Character Relationships</SectionTitle>
                  <RelationshipTimeline 
                    agentPair={[selectedEntity, selectedRelationship?.target || selectedRelationship?.source || '']}
                    initialEvents={[]}
                  />
                </PanelSection>
              )}
            </>
          ) : selectedData ? (
            <>
              <PanelSection>
                <SectionTitle>
                  {selectedCharacter ? 'Character' : 'Location'} Details
                </SectionTitle>
                
                <MetricCard>
                  <MetricLabel>Name</MetricLabel>
                  <MetricValue>{selectedData.name}</MetricValue>
                </MetricCard>
                
                {selectedCharacter && (
                  <>
                    <MetricCard>
                      <MetricLabel>Current Mood</MetricLabel>
                      <MetricValue>{selectedCharacter.custom_data?.mood || 'Neutral'}</MetricValue>
                      <ProgressBar 
                        progress={75} 
                        color="#fbbf24"
                      />
                    </MetricCard>
                    
                    <MetricCard>
                      <MetricLabel>Memory Count</MetricLabel>
                      <MetricValue>{selectedCharacter.memory_count}</MetricValue>
                    </MetricCard>
                    
                    <MetricCard>
                      <MetricLabel>Relationships</MetricLabel>
                      <MetricValue>{selectedCharacter.relationship_count}</MetricValue>
                    </MetricCard>
                  </>
                )}
              </PanelSection>
              
              {selectedCharacter && (
                <PanelSection>
                  <SectionTitle>Quick Actions</SectionTitle>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <QuickAction>View Memories</QuickAction>
                    <QuickAction>Test Interaction</QuickAction>
                    <QuickAction>Edit Character</QuickAction>
                  </div>
                </PanelSection>
              )}
              
              {selectedCharacter && activeFilters.emotions && (
                <PanelSection>
                  <SectionTitle>Emotional State</SectionTitle>
                  <MetricCard>
                    <MetricLabel>Happiness</MetricLabel>
                    <ProgressBar progress={60} color="#fbbf24" />
                  </MetricCard>
                  <MetricCard>
                    <MetricLabel>Curiosity</MetricLabel>
                    <ProgressBar progress={80} color="#3b82f6" />
                  </MetricCard>
                </PanelSection>
              )}
              
              {activeFilters.metrics && (
                <PanelSection>
                  <SectionTitle>Performance Metrics</SectionTitle>
                  <MetricCard>
                    <MetricLabel>Response Quality</MetricLabel>
                    <MetricValue>92%</MetricValue>
                    <ProgressBar progress={92} />
                  </MetricCard>
                  <MetricCard>
                    <MetricLabel>Coherence Score</MetricLabel>
                    <MetricValue>88%</MetricValue>
                    <ProgressBar progress={88} color="#22c55e" />
                  </MetricCard>
                  <MetricCard>
                    <MetricLabel>Personality Match</MetricLabel>
                    <MetricValue>95%</MetricValue>
                    <ProgressBar progress={95} color="#8b5cf6" />
                  </MetricCard>
                </PanelSection>
              )}
            </>
          ) : (
            <PanelSection>
              <div style={{ textAlign: 'center', color: 'rgba(255, 255, 255, 0.4)', padding: '2rem' }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem', opacity: 0.2 }}>🎬</div>
                <div>Select an entity to view details</div>
                <div style={{ fontSize: '0.75rem', marginTop: '0.5rem' }}>
                  Press ? for keyboard shortcuts
                </div>
              </div>
            </PanelSection>
          )}
        </ContextPanel>
        
        {/* Zoom Controls */}
        <ZoomControls>
          <IconButton onClick={() => setZoom(prev => Math.min(prev + 0.1, 2))}>
            +
          </IconButton>
          <IconButton onClick={() => setZoom(1)}>
            {(zoom * 100).toFixed(0)}%
          </IconButton>
          <IconButton onClick={() => setZoom(prev => Math.max(prev - 0.1, 0.5))}>
            -
          </IconButton>
        </ZoomControls>
      </MainContent>

      <Timeline>
        <PlaybackControls>
          <IconButton onClick={() => console.log('Previous event')}>
            {React.createElement(FiSkipBack as React.ComponentType<any>, { size: 16 })}
          </IconButton>
          <IconButton onClick={() => setIsPlaying(!isPlaying)}>
            {isPlaying ? 
              React.createElement(FiPause as React.ComponentType<any>, { size: 16 }) : 
              React.createElement(FiPlay as React.ComponentType<any>, { size: 16 })
            }
          </IconButton>
          <IconButton onClick={() => console.log('Next event')}>
            {React.createElement(FiSkipForward as React.ComponentType<any>, { size: 16 })}
          </IconButton>
        </PlaybackControls>
        
        <EventStream>
          <div style={{ 
            position: 'absolute',
            left: '20%',
            top: '50%',
            transform: 'translateY(-50%)',
            width: '4px',
            height: '30px',
            background: '#8b5cf6',
            borderRadius: '2px'
          }} />
          {/* Event markers */}
          <AnimatePresence>
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                style={{
                  position: 'absolute',
                  left: `${20 + i * 15}%`,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: '2px',
                  height: '20px',
                  background: 'rgba(255, 255, 255, 0.2)',
                  borderRadius: '1px'
                }}
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.1 }}
              />
            ))}
          </AnimatePresence>
        </EventStream>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'rgba(255, 255, 255, 0.5)' }}>
          {React.createElement(FiClock as React.ComponentType<any>, { size: 14 })}
          <span style={{ fontSize: '0.875rem' }}>
            {isPlaying ? 'Live' : 'Paused'}
          </span>
        </div>
        
        <IconButton onClick={() => setIsPanelOpen(!isPanelOpen)} title="Toggle Panel (P)">
          {isPanelOpen ? 
            React.createElement(FiMinimize2 as React.ComponentType<any>, { size: 16 }) : 
            React.createElement(FiMaximize2 as React.ComponentType<any>, { size: 16 })
          }
        </IconButton>
      </Timeline>
      
      {/* Keyboard Hints */}
      <AnimatePresence>
        {showKeyboardHints && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            <KeyboardHint>
              <div style={{ marginBottom: '0.5rem', fontWeight: 600 }}>Keyboard Shortcuts</div>
              <div>Space - Play/Pause</div>
              <div>M - Toggle Memories</div>
              <div>E - Toggle Emotions</div>
              <div>C - Toggle Connections</div>
              <div>R - Toggle Relationship View</div>
              <div>P - Toggle Panel</div>
              <div>+/- - Zoom In/Out</div>
              <div>Esc - Deselect</div>
              <div style={{ marginTop: '0.5rem', opacity: 0.5 }}>Press ? to hide</div>
            </KeyboardHint>
          </motion.div>
        )}
      </AnimatePresence>
    </Container>
  );
};

export default DirectorsView; 