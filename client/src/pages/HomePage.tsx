import React, { useState, useEffect } from 'react';
import styled from '@emotion/styled';
import { motion } from 'framer-motion';
import { FiMessageCircle, FiStar, FiHeart } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  position: relative;
  z-index: 1;
`;

const Header = styled(motion.header)`
  text-align: center;
  margin-bottom: 4rem;
`;

const Title = styled.h1`
  font-size: 4rem;
  margin-bottom: 1rem;
  background: linear-gradient(to right, #fff, rgba(255, 255, 255, 0.8));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  
  @media (max-width: 768px) {
    font-size: 2.5rem;
  }
`;

const Subtitle = styled.p`
  font-size: 1.25rem;
  color: var(--text-secondary);
  max-width: 600px;
  margin: 0 auto;
`;

const CharacterGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
`;

const CharacterCard = styled(motion.div)`
  position: relative;
  cursor: pointer;
  border-radius: 20px;
  overflow: hidden;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-5px);
  }
`;

const CardBackground = styled.div<{ gradient: string }>`
  position: absolute;
  inset: 0;
  background: ${props => props.gradient};
  opacity: 0.8;
`;

const CardContent = styled.div`
  position: relative;
  padding: 2rem;
  background: rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(20px);
  height: 100%;
  display: flex;
  flex-direction: column;
`;

const CharacterAvatar = styled.div<{ emoji: string }>`
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
  border: 2px solid rgba(255, 255, 255, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2.5rem;
  margin-bottom: 1rem;
  
  &::before {
    content: '${props => props.emoji}';
  }
`;

const CharacterName = styled.h3`
  font-size: 1.5rem;
  margin-bottom: 0.5rem;
`;

const CharacterDescription = styled.p`
  color: var(--text-secondary);
  font-size: 0.9rem;
  line-height: 1.6;
  flex: 1;
`;

const CharacterStats = styled.div`
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

const Stat = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: var(--text-secondary);
  font-size: 0.875rem;
`;

const StartButton = styled(motion.button)`
  margin-top: 1rem;
  width: 100%;
  padding: 1rem;
  border: none;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.2);
  }
`;

// Mock character data - replace with API call
const characters = [
  {
    id: 'alice',
    name: 'Alice',
    emoji: '🧚‍♀️',
    description: 'A curious and adventurous soul who loves exploring magical realms and uncovering ancient mysteries.',
    gradient: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    stats: {
      conversations: 1234,
      rating: 4.8,
      personality: 'Adventurous'
    }
  },
  {
    id: 'max',
    name: 'Max',
    emoji: '🤖',
    description: 'A witty AI companion with a passion for technology, science fiction, and philosophical discussions.',
    gradient: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    stats: {
      conversations: 892,
      rating: 4.9,
      personality: 'Intellectual'
    }
  },
  {
    id: 'luna',
    name: 'Luna',
    emoji: '🌙',
    description: 'A mystical dream guide who helps navigate the realm of sleep and uncover hidden meanings.',
    gradient: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
    stats: {
      conversations: 756,
      rating: 4.7,
      personality: 'Mystical'
    }
  }
];

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const [selectedCharacter, setSelectedCharacter] = useState<string | null>(null);

  const startChat = (characterId: string) => {
    // Generate a session ID (in production, this would come from the server)
    const sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    navigate(`/chat/${sessionId}?character=${characterId}`);
  };

  return (
    <Container>
      <Header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Title>Choose Your Character</Title>
        <Subtitle>
          Select a character to begin your conversation. Each one has a unique personality and story to share.
        </Subtitle>
      </Header>

      <CharacterGrid>
        {characters.map((character, index) => (
          <CharacterCard
            key={character.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            onClick={() => setSelectedCharacter(character.id)}
            className="glass"
          >
            <CardBackground gradient={character.gradient} />
            <CardContent>
              <CharacterAvatar emoji={character.emoji} />
              <CharacterName>{character.name}</CharacterName>
              <CharacterDescription>{character.description}</CharacterDescription>
              
              <CharacterStats>
                <Stat>
                  <FiMessageCircle />
                  {character.stats.conversations.toLocaleString()}
                </Stat>
                <Stat>
                  <FiStar />
                  {character.stats.rating}
                </Stat>
                <Stat>
                  <FiHeart />
                  {character.stats.personality}
                </Stat>
              </CharacterStats>

              <StartButton
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={(e) => {
                  e.stopPropagation();
                  startChat(character.id);
                }}
              >
                Start Conversation
              </StartButton>
            </CardContent>
          </CharacterCard>
        ))}
      </CharacterGrid>
    </Container>
  );
};

export default HomePage; 