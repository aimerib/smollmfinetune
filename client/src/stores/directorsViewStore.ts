/**
 * Zustand Store for Director's View State Management
 */

import { create } from 'zustand';
import { WorldSnapshot } from '../services/websocketService';

// Types
interface Memory {
  id: string;
  content: string;
  importance: number;
  emotional_valence: number;
  memory_type: string;
  timestamp: string;
  visualization?: {
    bubble_color: string;
    bubble_size: number;
  };
}

interface EmotionalState {
  active_emotions: Record<string, number>;
  surprise_score: number;
  momentum: number;
  emotional_volatility: number;
  decay_states: Record<string, any>;
}

interface TripleHeadMetrics {
  generation_quality: number;
  control_effectiveness: number;
  memory_coherence: number;
  coordination_score: number;
}

interface DirectorsViewState {
  // World state
  worldState: WorldSnapshot | null;
  
  // Character memories (characterId -> memories)
  memories: Record<string, Memory[]>;
  
  // Emotional states (characterId -> state)
  emotions: Record<string, EmotionalState>;
  
  // Triple-head metrics (characterId -> metrics)
  metrics: Record<string, TripleHeadMetrics>;
  
  // Actions
  updateWorldState: (worldState: WorldSnapshot) => void;
  addMemory: (characterId: string, memory: Memory) => void;
  updateEmotion: (characterId: string, emotion: EmotionalState) => void;
  updateMetrics: (characterId: string, metrics: TripleHeadMetrics) => void;
  clearCharacterData: (characterId: string) => void;
  reset: () => void;
}

// Create store
export const useDirectorsViewStore = create<DirectorsViewState>((set) => ({
  // Initial state with mock data for testing
  worldState: {
    locations: [
      {
        id: "village_square",
        name: "Village Square", 
        type: "location",
        custom_data: { name: "Village Square" }
      },
      {
        id: "forest_path",
        name: "Forest Path",
        type: "location", 
        custom_data: { name: "Forest Path" }
      },
      {
        id: "elder_tree",
        name: "The Elder Tree",
        type: "location",
        custom_data: { name: "The Elder Tree" }
      },
      {
        id: "market",
        name: "Market",
        type: "location",
        custom_data: { name: "Market" }
      }
    ],
    characters: [
      {
        id: "clara_001",
        name: "Clara",
        type: "character",
        location: "village_square",
        custom_data: {
          name: "Clara",
          mood: "curious",
          personality_traits: {
            openness: 0.8,
            conscientiousness: 0.6,
            extraversion: 0.7,
            agreeableness: 0.9,
            neuroticism: 0.3
          }
        },
        memory_count: 5,
        relationship_count: 3
      },
      {
        id: "elder_001",
        name: "Village Elder",
        type: "character",
        location: "elder_tree", 
        custom_data: {
          name: "Village Elder",
          mood: "wise",
          personality_traits: {
            openness: 0.6,
            conscientiousness: 0.9,
            extraversion: 0.4,
            agreeableness: 0.8,
            neuroticism: 0.2
          }
        },
        memory_count: 12,
        relationship_count: 8
      },
      {
        id: "merchant_001", 
        name: "Traveling Merchant",
        type: "character",
        location: "market",
        custom_data: {
          name: "Traveling Merchant",
          mood: "cheerful",
          personality_traits: {
            openness: 0.7,
            conscientiousness: 0.7,
            extraversion: 0.9,
            agreeableness: 0.7,
            neuroticism: 0.4
          }
        },
        memory_count: 8,
        relationship_count: 6
      }
    ],
    total_entities: 7,
    recent_subtext: [
      {
        agent_id: "clara_001",
        text: "*Clara looks around the village square with curiosity*",
        timestamp: new Date().toISOString()
      },
      {
        agent_id: "elder_001", 
        text: "*The Village Elder sits quietly under the ancient tree*",
        timestamp: new Date().toISOString()
      },
      {
        agent_id: "merchant_001",
        text: "*The merchant arranges his wares with enthusiasm*", 
        timestamp: new Date().toISOString()
      }
    ],
    timestamp: new Date().toISOString()
  },
  memories: {},
  emotions: {},
  metrics: {},
  
  // Actions
  updateWorldState: (worldState) => 
    set(() => ({ worldState })),
  
  addMemory: (characterId, memory) =>
    set((state) => {
      const characterMemories = state.memories[characterId] || [];
      
      // Add new memory and keep only last 100
      const updatedMemories = [memory, ...characterMemories].slice(0, 100);
      
      return {
        memories: {
          ...state.memories,
          [characterId]: updatedMemories
        }
      };
    }),
  
  updateEmotion: (characterId, emotion) =>
    set((state) => ({
      emotions: {
        ...state.emotions,
        [characterId]: emotion
      }
    })),
  
  updateMetrics: (characterId, metrics) =>
    set((state) => ({
      metrics: {
        ...state.metrics,
        [characterId]: metrics
      }
    })),
  
  clearCharacterData: (characterId) =>
    set((state) => {
      const { [characterId]: _, ...remainingMemories } = state.memories;
      const { [characterId]: __, ...remainingEmotions } = state.emotions;
      const { [characterId]: ___, ...remainingMetrics } = state.metrics;
      
      return {
        memories: remainingMemories,
        emotions: remainingEmotions,
        metrics: remainingMetrics
      };
    }),
  
  reset: () =>
    set(() => ({
      worldState: null,
      memories: {},
      emotions: {},
      metrics: {}
    }))
}));

// Selectors
export const selectCharacterMemories = (characterId: string) => 
  (state: DirectorsViewState) => state.memories[characterId] || [];

export const selectCharacterEmotions = (characterId: string) => 
  (state: DirectorsViewState) => state.emotions[characterId] || null;

export const selectCharacterMetrics = (characterId: string) => 
  (state: DirectorsViewState) => state.metrics[characterId] || null;

export const selectCharacterByLocation = (location: string) =>
  (state: DirectorsViewState) => 
    state.worldState?.characters.filter(char => char.location === location) || [];

export const selectAllCharacters = () =>
  (state: DirectorsViewState) => state.worldState?.characters || [];

export const selectAllLocations = () =>
  (state: DirectorsViewState) => state.worldState?.locations || []; 