import React from 'react';
import styled from '@emotion/styled';
import { useNavigate } from 'react-router-dom';

const Container = styled.div`
  max-width: 1000px;
  margin: 0 auto;
  padding: 2rem;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
`;

const Header = styled.header`
  text-align: center;
  margin-bottom: 3rem;
  padding-top: 2rem;
`;

const Title = styled.h1`
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  color: white;
`;

const Subtitle = styled.p`
  font-size: 1.1rem;
  color: rgba(255, 255, 255, 0.7);
  max-width: 500px;
  margin: 0 auto;
`;

const CharacterGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
`;

const CharacterCard = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1.5rem;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(139, 69, 193, 0.3);
    transform: translateY(-2px);
  }
`;

const CharacterAvatar = styled.div`
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: #8b45c1;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
  margin-bottom: 1rem;
`;

const CharacterName = styled.h3`
  font-size: 1.3rem;
  margin-bottom: 0.5rem;
  color: white;
`;

const CharacterDescription = styled.p`
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.9rem;
  line-height: 1.5;
  margin-bottom: 1rem;
`;

const StartButton = styled.button`
  width: 100%;
  padding: 0.75rem;
  border: none;
  border-radius: 8px;
  background: #8b45c1;
  color: white;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: #9b55d1;
  }
`;

const characters = [
  {
    id: 'alice',
    name: 'Alice',
    emoji: '🧚‍♀️',
    description: 'A curious and adventurous soul who loves exploring magical realms and uncovering ancient mysteries.'
  },
  {
    id: 'max',
    name: 'Max',
    emoji: '🤖',
    description: 'A witty AI companion with a passion for technology, science fiction, and philosophical discussions.'
  },
  {
    id: 'luna',
    name: 'Luna',
    emoji: '🌙',
    description: 'A mystical dream guide who helps navigate the realm of sleep and uncover hidden meanings.'
  }
];

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const startChat = (characterId: string) => {
    navigate(`/chat?character=${characterId}`);
  };

  return (
    <Container>
      <Header>
        <Title>Choose Your Character</Title>
        <Subtitle>
          Select a character to begin your conversation
        </Subtitle>
      </Header>

      <CharacterGrid>
        {characters.map((character) => (
          <CharacterCard
            key={character.id}
            onClick={() => startChat(character.id)}
          >
            <CharacterAvatar>
              {character.emoji}
            </CharacterAvatar>
            
            <CharacterName>{character.name}</CharacterName>
            <CharacterDescription>{character.description}</CharacterDescription>
            
            <StartButton onClick={(e) => {
              e.stopPropagation();
              startChat(character.id);
            }}>
              Start Chat
            </StartButton>
          </CharacterCard>
        ))}
      </CharacterGrid>
    </Container>
  );
};

export default HomePage; 