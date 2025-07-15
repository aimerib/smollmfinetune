/**
 * Memory Bubbles Component
 * 
 * Displays floating memory bubbles above characters when memories are formed.
 * Bubbles are colored by emotional valence and sized by importance.
 */

import React, { useRef, useState, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, Sphere, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import { animated, useSpring } from '@react-spring/three';

interface MemoryBubbleProps {
  memory: {
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
  };
  position: [number, number, number];
  onComplete: () => void;
}

// Emotion icons based on valence and type
const getEmotionIcon = (valence: number, type: string): string => {
  if (type === 'emotional') {
    if (valence > 0.5) return '💕';
    if (valence > 0) return '😊';
    if (valence > -0.5) return '😔';
    return '😢';
  }
  
  if (type === 'episodic') {
    if (valence > 0.5) return '✨';
    if (valence < -0.5) return '⚡';
    return '💭';
  }
  
  if (type === 'semantic') return '💡';
  if (type === 'procedural') return '⚙️';
  
  return '💭';
};

const MemoryBubble: React.FC<MemoryBubbleProps> = ({ memory, position, onComplete }) => {
  const groupRef = useRef<THREE.Group>(null);
  const [opacity, setOpacity] = useState(1);
  
  // Parse color from HSL string
  const parseHSLColor = (hslString: string): THREE.Color => {
    const match = hslString.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
    if (match) {
      const h = parseInt(match[1]) / 360;
      const s = parseInt(match[2]) / 100;
      const l = parseInt(match[3]) / 100;
      return new THREE.Color().setHSL(h, s, l);
    }
    return new THREE.Color('#ffffff');
  };
  
  const bubbleColor = memory.visualization?.bubble_color 
    ? parseHSLColor(memory.visualization.bubble_color)
    : new THREE.Color('#ffffff');
  
  const bubbleSize = (memory.visualization?.bubble_size || 50) / 100;
  
  // Animation spring
  const { scale, positionY } = useSpring({
    from: { scale: 0, positionY: position[1] },
    to: { scale: bubbleSize, positionY: position[1] + 3 },
    config: { tension: 120, friction: 14 }
  });
  
  // Lifetime and fade effect
  useEffect(() => {
    const lifetime = 5000; // 5 seconds
    const fadeStart = 3000; // Start fading after 3 seconds
    
    const fadeTimer = setTimeout(() => {
      setOpacity(0);
    }, fadeStart);
    
    const removeTimer = setTimeout(() => {
      onComplete();
    }, lifetime);
    
    return () => {
      clearTimeout(fadeTimer);
      clearTimeout(removeTimer);
    };
  }, [onComplete]);
  
  useFrame((state) => {
    if (groupRef.current) {
      // Gentle floating motion
      groupRef.current.position.x = position[0] + Math.sin(state.clock.elapsedTime * 0.5) * 0.2;
      groupRef.current.position.z = position[2] + Math.cos(state.clock.elapsedTime * 0.3) * 0.2;
      
      // Fade out effect
      if (opacity < 1) {
        setOpacity(prev => Math.max(0, prev - 0.02));
      }
    }
  });
  
  return (
    <animated.group 
      ref={groupRef}
      position-y={positionY}
      scale={scale}
    >
      <Billboard>
        {/* Memory bubble */}
        <Sphere args={[1, 16, 16]}>
          <meshStandardMaterial
            color={bubbleColor}
            transparent
            opacity={opacity * 0.6}
            emissive={bubbleColor}
            emissiveIntensity={0.3}
            roughness={0.1}
            metalness={0.5}
          />
        </Sphere>
        
        {/* Emotion icon */}
        <Text
          position={[0, 0, 0.5]}
          fontSize={0.5}
          anchorX="center"
          anchorY="middle"
        >
          {getEmotionIcon(memory.emotional_valence, memory.memory_type)}
        </Text>
        
        {/* Memory preview (first few words) */}
        <Text
          position={[0, 0, 1.2]}
          fontSize={0.2}
          color="white"
          anchorX="center"
          anchorY="middle"
          maxWidth={2}
        >
          {memory.content.slice(0, 30) + '...'}
        </Text>
      </Billboard>
    </animated.group>
  );
};

interface MemoryBubblesProps {
  memories: Record<string, any[]>;
}

const MemoryBubbles: React.FC<MemoryBubblesProps> = ({ memories }) => {
  const [activeBubbles, setActiveBubbles] = useState<Array<{
    id: string;
    memory: any;
    characterId: string;
    position: [number, number, number];
  }>>([]);
  
  // Track last processed memory per character
  const lastProcessedRef = useRef<Record<string, string>>({});
  
  // Process new memories
  useEffect(() => {
    const newBubbles: typeof activeBubbles = [];
    
    Object.entries(memories).forEach(([characterId, characterMemories]) => {
      if (characterMemories.length > 0) {
        const latestMemory = characterMemories[0];
        const lastProcessed = lastProcessedRef.current[characterId];
        
        // Only add if this is a new memory
        if (latestMemory.id !== lastProcessed) {
          lastProcessedRef.current[characterId] = latestMemory.id;
          
          // Get character position (this would come from world state in real implementation)
          // For now, use random positions
          const position: [number, number, number] = [
            Math.random() * 10 - 5,
            2,
            Math.random() * 10 - 5
          ];
          
          newBubbles.push({
            id: `${characterId}_${latestMemory.id}`,
            memory: latestMemory,
            characterId,
            position
          });
        }
      }
    });
    
    if (newBubbles.length > 0) {
      setActiveBubbles(prev => [...prev, ...newBubbles]);
    }
  }, [memories]);
  
  const handleBubbleComplete = (bubbleId: string) => {
    setActiveBubbles(prev => prev.filter(b => b.id !== bubbleId));
  };
  
  return (
    <>
      {activeBubbles.map(bubble => (
        <MemoryBubble
          key={bubble.id}
          memory={bubble.memory}
          position={bubble.position}
          onComplete={() => handleBubbleComplete(bubble.id)}
        />
      ))}
    </>
  );
};

export default MemoryBubbles; 