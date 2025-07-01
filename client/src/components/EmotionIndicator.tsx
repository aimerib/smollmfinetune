import React from 'react';
import styled from '@emotion/styled';
import { motion } from 'framer-motion';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  min-width: 100px;
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.875rem;
`;

const Label = styled.span`
  font-weight: 500;
`;

const ProgressBar = styled.div`
  width: 100%;
  height: 6px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 3px;
  overflow: hidden;
  position: relative;
`;

const ProgressFill = styled(motion.div)<{ color: string }>`
  height: 100%;
  background: ${props => props.color};
  border-radius: 3px;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
      90deg,
      transparent 0%,
      rgba(255, 255, 255, 0.4) 50%,
      transparent 100%
    );
    animation: shimmer 2s infinite;
  }
  
  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
`;

const Value = styled.span`
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.6);
  margin-left: auto;
`;

interface EmotionIndicatorProps {
  emotion: string;
  intensity: number; // 0-1
  label: string;
  icon: React.ReactNode;
}

const emotionColors: Record<string, string> = {
  happy: '#F6D365',
  sad: '#4FACFE',
  angry: '#FA709A',
  curious: '#A8EDEA',
  excited: '#F093FB',
  neutral: '#E0C3FC',
  memory: '#8EC5FC'
};

const EmotionIndicator: React.FC<EmotionIndicatorProps> = ({
  emotion,
  intensity,
  label,
  icon
}) => {
  const color = emotionColors[emotion] || emotionColors.neutral;
  const percentage = Math.round(intensity * 100);

  return (
    <Container>
      <Header>
        {icon}
        <Label>{label}</Label>
        <Value>{percentage}%</Value>
      </Header>
      <ProgressBar>
        <ProgressFill
          color={color}
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </ProgressBar>
    </Container>
  );
};

export default EmotionIndicator; 