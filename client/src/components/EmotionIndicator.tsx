import React from 'react';
import styled from '@emotion/styled';
import { keyframes, css } from '@emotion/react';
import { motion } from 'framer-motion';

// Living emotion animations
const heartbeat = keyframes`
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
`;

const ripple = keyframes`
  0% { transform: scale(0.8); opacity: 1; }
  100% { transform: scale(2.4); opacity: 0; }
`;

const glow = keyframes`
  0%, 100% { box-shadow: 0 0 10px rgba(139, 69, 193, 0.3); }
  50% { box-shadow: 0 0 20px rgba(139, 69, 193, 0.6); }
`;

const float = keyframes`
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-2px); }
`;

const EmotionContainer = styled(motion.div, {
  shouldForwardProp: (prop) => !['emotion', 'intensity'].includes(prop)
})<{ emotion: string; intensity: number }>`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  padding: 0.8rem;
  border-radius: 16px;
  background: rgba(15, 15, 25, 0.7);
  backdrop-filter: blur(15px);
  border: 1px solid rgba(139, 69, 193, 0.2);
  position: relative;
  overflow: hidden;
  min-width: 80px;
  
  /* Emotion-specific coloring */
  ${props => {
    const baseIntensity = Math.max(0.3, props.intensity);
    switch (props.emotion) {
      case 'mood':
        return `
          border-color: rgba(34, 197, 94, ${baseIntensity * 0.6});
          background: rgba(34, 197, 94, ${baseIntensity * 0.05});
        `;
      case 'memory':
        return `
          border-color: rgba(59, 130, 246, ${baseIntensity * 0.6});
          background: rgba(59, 130, 246, ${baseIntensity * 0.05});
        `;
      default:
        return `
          border-color: rgba(139, 69, 193, ${baseIntensity * 0.6});
          background: rgba(139, 69, 193, ${baseIntensity * 0.05});
        `;
    }
  }}
  
  /* Living background animation */
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(
      circle at center,
      ${props => {
        switch (props.emotion) {
          case 'mood': return `rgba(34, 197, 94, ${props.intensity * 0.1})`;
          case 'memory': return `rgba(59, 130, 246, ${props.intensity * 0.1})`;
          default: return `rgba(139, 69, 193, ${props.intensity * 0.1})`;
        }
      }},
      transparent 70%
    );
    ${props => css`animation: ${glow} ${2 + (1 - props.intensity) * 2}s ease-in-out infinite;`}
    border-radius: 16px;
    z-index: 0;
  }
  
  /* Intensity-based floating */
  ${props => css`animation: ${float} ${2 + (1 - props.intensity) * 3}s ease-in-out infinite;`}
`;

const IconContainer = styled(motion.div, {
  shouldForwardProp: (prop) => !['emotion', 'intensity'].includes(prop)
})<{ emotion: string; intensity: number }>`
  position: relative;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  background: linear-gradient(135deg, 
    ${props => {
      switch (props.emotion) {
        case 'mood': return `rgba(34, 197, 94, ${props.intensity * 0.3}), rgba(34, 197, 94, ${props.intensity * 0.1})`;
        case 'memory': return `rgba(59, 130, 246, ${props.intensity * 0.3}), rgba(59, 130, 246, ${props.intensity * 0.1})`;
        default: return `rgba(139, 69, 193, ${props.intensity * 0.3}), rgba(139, 69, 193, ${props.intensity * 0.1})`;
      }
    }}
  );
  color: ${props => {
    switch (props.emotion) {
      case 'mood': return `rgba(34, 197, 94, ${0.7 + props.intensity * 0.3})`;
      case 'memory': return `rgba(59, 130, 246, ${0.7 + props.intensity * 0.3})`;
      default: return `rgba(139, 69, 193, ${0.7 + props.intensity * 0.3})`;
    }
  }};
  z-index: 1;
  
  /* Heartbeat animation based on intensity */
  ${props => css`animation: ${heartbeat} ${1.5 + (1 - props.intensity) * 1.5}s ease-in-out infinite;`}
  
  /* Ripple effect */
  &::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 100%;
    height: 100%;
    border: 2px solid ${props => {
      switch (props.emotion) {
        case 'mood': return `rgba(34, 197, 94, ${props.intensity * 0.4})`;
        case 'memory': return `rgba(59, 130, 246, ${props.intensity * 0.4})`;
        default: return `rgba(139, 69, 193, ${props.intensity * 0.4})`;
      }
    }};
    border-radius: 50%;
    transform: translate(-50%, -50%);
    ${props => css`animation: ${ripple} ${2 + (1 - props.intensity) * 2}s ease-out infinite;`}
  }
`;

const IntensityBar = styled(motion.div, {
  shouldForwardProp: (prop) => !['emotion', 'intensity'].includes(prop)
})<{ emotion: string; intensity: number }>`
  width: 4px;
  height: 24px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 2px;
  position: relative;
  overflow: hidden;
  
  &::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: ${props => props.intensity * 100}%;
    background: linear-gradient(180deg,
      ${props => {
        switch (props.emotion) {
          case 'mood': return '#22c55e, rgba(34, 197, 94, 0.6)';
          case 'memory': return '#3b82f6, rgba(59, 130, 246, 0.6)';
          default: return '#8b45c1, rgba(139, 69, 193, 0.6)';
        }
      }}
    );
    border-radius: 2px;
    box-shadow: 0 0 8px ${props => {
      switch (props.emotion) {
        case 'mood': return `rgba(34, 197, 94, ${props.intensity * 0.6})`;
        case 'memory': return `rgba(59, 130, 246, ${props.intensity * 0.6})`;
        default: return `rgba(139, 69, 193, ${props.intensity * 0.6})`;
      }
    }};
    transition: height 0.5s ease-out;
  }
`;

const Label = styled(motion.div, {
  shouldForwardProp: (prop) => !['emotion'].includes(prop)
})<{ emotion: string }>`
  font-size: 0.75rem;
  font-weight: 500;
  color: ${props => {
    switch (props.emotion) {
      case 'mood': return 'rgba(34, 197, 94, 0.9)';
      case 'memory': return 'rgba(59, 130, 246, 0.9)';
      default: return 'rgba(139, 69, 193, 0.9)';
    }
  }};
  text-align: center;
  position: relative;
  z-index: 1;
  font-family: 'Inter', sans-serif;
  
  /* Subtle text glow */
  text-shadow: 0 0 10px ${props => {
    switch (props.emotion) {
      case 'mood': return 'rgba(34, 197, 94, 0.3)';
      case 'memory': return 'rgba(59, 130, 246, 0.3)';
      default: return 'rgba(139, 69, 193, 0.3)';
    }
  }};
`;

const IntensityValue = styled(motion.div, {
  shouldForwardProp: (prop) => !['emotion'].includes(prop)
})<{ emotion: string }>`
  font-size: 0.65rem;
  color: rgba(255, 255, 255, 0.6);
  position: relative;
  z-index: 1;
  font-family: 'Inter', sans-serif;
  font-weight: 600;
`;

interface EmotionIndicatorProps {
  emotion: string;
  intensity: number; // 0.0 to 1.0
  label: string;
  icon: React.ReactElement;
}

const EmotionIndicator = React.forwardRef<HTMLDivElement, EmotionIndicatorProps>(({
  emotion,
  intensity,
  label,
  icon
}, ref) => {
  const clampedIntensity = Math.max(0, Math.min(1, intensity));
  
  return (
    <EmotionContainer
      ref={ref}
      emotion={emotion}
      intensity={clampedIntensity}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.05, y: -2 }}
      transition={{
        type: "spring",
        damping: 20,
        stiffness: 300
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', position: 'relative', zIndex: 1 }}>
        <IconContainer
          emotion={emotion}
          intensity={clampedIntensity}
          whileHover={{ scale: 1.1, rotate: 5 }}
          transition={{ type: "spring", damping: 15, stiffness: 400 }}
        >
          {icon}
        </IconContainer>
        
        <IntensityBar
          emotion={emotion}
          intensity={clampedIntensity}
          initial={{ scaleY: 0 }}
          animate={{ scaleY: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
        />
      </div>
      
      <Label
        emotion={emotion}
        initial={{ opacity: 0, y: 5 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        {label}
      </Label>
      
      <IntensityValue
        emotion={emotion}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        {Math.round(clampedIntensity * 100)}%
      </IntensityValue>
    </EmotionContainer>
  );
});

EmotionIndicator.displayName = 'EmotionIndicator';

export default EmotionIndicator; 