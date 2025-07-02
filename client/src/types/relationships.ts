/**
 * Type definitions for relationship visualization components
 */

export interface BigFiveTraits {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

export interface EmotionalState {
  happy?: number;
  sad?: number;
  angry?: number;
  fearful?: number;
  surprised?: number;
  disgusted?: number;
  nervous?: number;
  [key: string]: number | undefined;
}

export interface RelationshipNode {
  id: string;
  name: string;
  personality: BigFiveTraits;
  emotional_state: EmotionalState;
  position: { x: number; y: number };
  size: number; // Based on total relationship count
}

export interface RelationshipEdge {
  source: string;
  target: string;
  affinity: number; // -1.0 to 1.0
  status: string; // 'Friend', 'Rival', 'Acquaintance', etc.
  emotional_history: string[];
  memory_significance: number;
  interaction_count: number;
  last_interaction: string;
}

export interface RelationshipHistoryEvent {
  timestamp: string;
  speaker_id: string;
  target_id: string;
  interaction_type: string;
  affinity_change: number;
  emotional_impact: string[];
  memory_significance: number;
  narrative_context?: string;
}

export interface RelationshipMetricsData {
  totalRelationships: number;
  averageAffinity: number;
  strongBonds: number;
  conflicts: number;
  recentChanges: Array<{
    pair: [string, string];
    change: number;
    timestamp: string;
  }>;
  socialClusters: Array<{
    id: string;
    members: string[];
    cohesion: number;
  }>;
}

export interface AffinityFilter {
  min: number;
  max: number;
} 