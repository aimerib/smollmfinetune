import React from 'react';
import styled from '@emotion/styled';
import { motion } from 'framer-motion';
import { FiArrowLeft, FiMoreVertical } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';

const Header = styled(motion.header)`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.5rem;
  background: rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const LeftSection = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const BackButton = styled(motion.button)`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: none;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.2);
  }
`;

const CharacterInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

// Emotion to gradient mapping
function getEmotionGradient(emotion: string): string {
  const emotionGradients: Record<string, string> = {
    happy: 'linear-gradient(135deg, #F6D365 0%, #FDA085 100%)',
    sad: 'linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%)',
    angry: 'linear-gradient(135deg, #FA709A 0%, #FEE140 100%)',
    curious: 'linear-gradient(135deg, #A8EDEA 0%, #FED6E3 100%)',
    excited: 'linear-gradient(135deg, #F093FB 0%, #F5576C 100%)',
    neutral: 'linear-gradient(135deg, #E0C3FC 0%, #8EC5FC 100%)'
  };
  
  return emotionGradients[emotion] || emotionGradients.neutral;
}

const CharacterAvatar = styled(motion.div)<{ emoji: string; emotion: string }>`
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: ${props => getEmotionGradient(props.emotion)};
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
  position: relative;
  
  &::before {
    content: '${props => props.emoji}';
  }
  
  &::after {
    content: '';
    position: absolute;
    inset: -3px;
    border-radius: 50%;
    background: ${props => getEmotionGradient(props.emotion)};
    opacity: 0.3;
    filter: blur(10px);
  }
`;

const CharacterDetails = styled.div``;

const CharacterName = styled.h2`
  font-size: 1.5rem;
  margin: 0;
  color: white;
`;

const StatusText = styled.div`
  font-size: 0.875rem;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const OnlineIndicator = styled.span`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #4caf50;
  display: inline-block;
`;

const OptionsButton = styled(motion.button)`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: none;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.2);
  }
`;

// Character data
const characterData: Record<string, { name: string; emoji: string; tagline: string }> = {
  alice: {
    name: 'Alice',
    emoji: '🧚‍♀️',
    tagline: 'Adventurer & Dreamer'
  },
  max: {
    name: 'Max',
    emoji: '🤖',
    tagline: 'Tech Enthusiast'
  },
  luna: {
    name: 'Luna',
    emoji: '🌙',
    tagline: 'Dream Guide'
  }
};

interface CharacterHeaderProps {
  characterId: string;
  emotion: string;
}

const CharacterHeader: React.FC<CharacterHeaderProps> = ({ characterId, emotion }) => {
  const navigate = useNavigate();
  const character = characterData[characterId] || characterData.alice;

  return (
    <Header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <LeftSection>
        <BackButton
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => navigate('/')}
        >
          <FiArrowLeft size={20} />
        </BackButton>
        
        <CharacterInfo>
          <CharacterAvatar
            emoji={character.emoji}
            emotion={emotion}
            animate={{ scale: [1, 1.05, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <CharacterDetails>
            <CharacterName>{character.name}</CharacterName>
            <StatusText>
              <OnlineIndicator />
              {character.tagline}
            </StatusText>
          </CharacterDetails>
        </CharacterInfo>
      </LeftSection>
      
      <OptionsButton
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
      >
        <FiMoreVertical size={20} />
      </OptionsButton>
    </Header>
  );
};

export default CharacterHeader; 