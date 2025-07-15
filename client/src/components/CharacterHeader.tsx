import React from 'react';
import styled from '@emotion/styled';
import { FiArrowLeft } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1rem;
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
`;

const LeftSection = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const BackButton = styled.button`
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: none;
  background: rgba(255, 255, 255, 0.1);
  color: rgba(255, 255, 255, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.15);
    color: white;
  }
`;

const CharacterInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
`;

const CharacterAvatar = styled.div`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #8b45c1;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.3rem;
`;

const CharacterDetails = styled.div`
  display: flex;
  flex-direction: column;
`;

const CharacterName = styled.h2`
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0;
  color: white;
`;

const CharacterStatus = styled.span`
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.6);
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.75rem;
  color: #22c55e;
`;

const StatusDot = styled.div`
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #22c55e;
  box-shadow: 0 0 4px rgba(34, 197, 94, 0.5);
`;

interface CharacterHeaderProps {
  characterId: string;
  emotion?: string;
}

const CharacterHeader: React.FC<CharacterHeaderProps> = ({ 
  characterId, 
  emotion = 'neutral' 
}) => {
  const navigate = useNavigate();
  
  const getCharacterData = (id: string) => {
    switch (id) {
      case 'alice':
        return {
          name: 'Alice',
          emoji: '🧚‍♀️',
          status: 'Curious Wanderer'
        };
      case 'max':
        return {
          name: 'Max',
          emoji: '🤖',
          status: 'Tech Enthusiast'
        };
      case 'luna':
        return {
          name: 'Luna',
          emoji: '🌙',
          status: 'Mystical Guide'
        };
      default:
        return {
          name: 'Character',
          emoji: '✨',
          status: 'AI Companion'
        };
    }
  };
  
  const character = getCharacterData(characterId);
  
  return (
    <Header>
      <LeftSection>
        <BackButton onClick={() => navigate('/')}>
          {React.createElement(FiArrowLeft as React.ComponentType<any>, { size: 16 })}
        </BackButton>
        
        <CharacterInfo>
          <CharacterAvatar>
            {character.emoji}
          </CharacterAvatar>
          
          <CharacterDetails>
            <CharacterName>{character.name}</CharacterName>
            <CharacterStatus>{character.status}</CharacterStatus>
          </CharacterDetails>
        </CharacterInfo>
      </LeftSection>
      
      <StatusIndicator>
        <StatusDot />
        Online
      </StatusIndicator>
    </Header>
  );
};

export default CharacterHeader; 