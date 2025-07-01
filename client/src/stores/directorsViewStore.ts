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
  // Initial state
  worldState: null,
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