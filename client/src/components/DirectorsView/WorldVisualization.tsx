/**
 * World Visualization Component
 * 
 * 3D visualization of the game world showing locations as platforms
 * and characters as animated sprites.
 */

import React, { useRef, useState, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text, Box, Sphere, Cone } from '@react-three/drei';
import * as THREE from 'three';
import { WorldSnapshot } from '../../services/websocketService';

interface WorldVisualizationProps {
  worldState: WorldSnapshot | null;
  onEntitySelect: (entityId: string) => void;
  selectedEntity: string | null;
}

// Location platform component
const LocationPlatform: React.FC<{
  location: any;
  position: [number, number, number];
  onSelect: () => void;
  isSelected: boolean;
}> = ({ location, position, onSelect, isSelected }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  
  useFrame((state) => {
    if (meshRef.current) {
      // Gentle floating animation
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });
  
  return (
    <group position={position}>
      <Box
        ref={meshRef}
        args={[4, 0.5, 4]}
        onClick={onSelect}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <meshStandardMaterial 
          color={isSelected ? '#6366f1' : hovered ? '#4f46e5' : '#374151'}
          emissive={isSelected ? '#6366f1' : hovered ? '#4f46e5' : '#000000'}
          emissiveIntensity={isSelected ? 0.3 : hovered ? 0.1 : 0}
        />
      </Box>
      <Text
        position={[0, 1, 0]}
        fontSize={0.5}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
      >
        {location.name}
      </Text>
    </group>
  );
};

// Character sprite component
const CharacterSprite: React.FC<{
  character: any;
  position: [number, number, number];
  onSelect: () => void;
  isSelected: boolean;
}> = ({ character, position, onSelect, isSelected }) => {
  const groupRef = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);
  
  // Get personality-based color
  const getCharacterColor = () => {
    const traits = character.attributes?.personality_traits;
    if (!traits) return '#10b981';
    
    // Map personality to color
    const r = traits.extraversion || 0.5;
    const g = traits.agreeableness || 0.5;
    const b = traits.openness || 0.5;
    
    return new THREE.Color(r, g, b);
  };
  
  useFrame((state) => {
    if (groupRef.current) {
      // Rotation animation
      groupRef.current.rotation.y = state.clock.elapsedTime * 0.5;
      
      // Bobbing animation
      groupRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.1;
    }
  });
  
  return (
    <group 
      ref={groupRef}
      position={position}
      onClick={onSelect}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      {/* Character body */}
      <Cone args={[0.3, 1, 8]} position={[0, 0, 0]}>
        <meshStandardMaterial 
          color={getCharacterColor()}
          emissive={isSelected ? '#ffffff' : hovered ? '#666666' : '#000000'}
          emissiveIntensity={isSelected ? 0.3 : hovered ? 0.1 : 0}
        />
      </Cone>
      
      {/* Character head */}
      <Sphere args={[0.2]} position={[0, 0.7, 0]}>
        <meshStandardMaterial color="#fbbf24" />
      </Sphere>
      
      {/* Character name */}
      <Text
        position={[0, 1.2, 0]}
        fontSize={0.3}
        color={isSelected ? '#fbbf24' : '#ffffff'}
        anchorX="center"
        anchorY="middle"
      >
        {character.name}
      </Text>
      
      {/* Mood indicator */}
      <Text
        position={[0, -0.7, 0]}
        fontSize={0.2}
        color="#94a3b8"
        anchorX="center"
        anchorY="middle"
      >
        {character.attributes?.mood || 'neutral'}
      </Text>
    </group>
  );
};

  const WorldVisualization: React.FC<WorldVisualizationProps> = ({
  worldState,
  onEntitySelect,
  selectedEntity
}) => {
  // Calculate positions for locations in a grid
  const locationPositions = useMemo(() => {
    if (!worldState?.locations) return {};
    
    const positions: Record<string, [number, number, number]> = {};
    const gridSize = Math.ceil(Math.sqrt(worldState.locations.length));
    
    worldState.locations.forEach((location, index) => {
      const x = (index % gridSize) * 6 - (gridSize * 3);
      const z = Math.floor(index / gridSize) * 6 - (gridSize * 3);
      positions[location.id] = [x, 0, z];
    });
    
    return positions;
  }, [worldState?.locations]);
  
  // Group characters by location
  const charactersByLocation = useMemo(() => {
    if (!worldState?.characters) return {};
    
    const grouped: Record<string, any[]> = {};
    
    worldState.characters.forEach(character => {
      const loc = character.location;
      if (!grouped[loc]) {
        grouped[loc] = [];
      }
      grouped[loc].push(character);
    });
    
    return grouped;
  }, [worldState?.characters]);
  
  // Calculate character positions based on their locations
  const characterPositions = useMemo(() => {
    if (!worldState?.characters) return {};
    
    const positions: Record<string, [number, number, number]> = {};
    
    // Position characters around their location
    Object.entries(charactersByLocation).forEach(([locationId, characters]) => {
      const basePos = locationPositions[locationId] || [0, 0, 0];
      
      characters.forEach((character, index) => {
        const angle = (index / characters.length) * Math.PI * 2;
        const radius = 1.5;
        
        positions[character.id] = [
          basePos[0] + Math.cos(angle) * radius,
          1,
          basePos[2] + Math.sin(angle) * radius
        ];
      });
    });
    
    return positions;
  }, [worldState?.characters, locationPositions, charactersByLocation]);
  
  if (!worldState) {
    return (
      <Text
        position={[0, 0, 0]}
        fontSize={0.5}
        color="#94a3b8"
        anchorX="center"
        anchorY="middle"
      >
        Waiting for world data...
      </Text>
    );
  }
  
  return (
    <>
      {/* Grid floor */}
      <gridHelper args={[50, 50, '#1f2937', '#111827']} />
      
      {/* Render locations */}
      {worldState.locations.map(location => (
        <LocationPlatform
          key={location.id}
          location={location}
          position={locationPositions[location.id] || [0, 0, 0]}
          onSelect={() => onEntitySelect(location.id)}
          isSelected={selectedEntity === location.id}
        />
      ))}
      
      {/* Render characters */}
      {worldState.characters.map(character => (
        <CharacterSprite
          key={character.id}
          character={character}
          position={characterPositions[character.id] || [0, 1, 0]}
          onSelect={() => onEntitySelect(character.id)}
          isSelected={selectedEntity === character.id}
        />
      ))}
      
      {/* Connection lines between characters in same location */}
      {Object.entries(charactersByLocation).map(([locationId, characters]) => {
        if (characters.length < 2) return null;
        
        return characters.map((char1, i) => 
          characters.slice(i + 1).map(char2 => {
            const pos1 = characterPositions[char1.id];
            const pos2 = characterPositions[char2.id];
            
            if (!pos1 || !pos2) return null;
            
            const points = [
              new THREE.Vector3(...pos1),
              new THREE.Vector3(...pos2)
            ];
            
            return (
              <line key={`${char1.id}-${char2.id}`}>
                <bufferGeometry>
                  <bufferAttribute
                    attach="attributes-position"
                    count={2}
                    array={new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))}
                    itemSize={3}
                  />
                </bufferGeometry>
                <lineBasicMaterial color="#374151" opacity={0.3} transparent />
              </line>
            );
          })
        );
      })}
    </>
  );
};

export default WorldVisualization; 