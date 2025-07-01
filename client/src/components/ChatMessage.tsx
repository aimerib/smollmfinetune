import React from 'react';
import styled from '@emotion/styled';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

const MessageContainer = styled(motion.div)<{ isUser: boolean }>`
  display: flex;
  justify-content: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  width: 100%;
`;

const MessageBubble = styled.div<{ isUser: boolean }>`
  max-width: 70%;
  padding: 1rem 1.5rem;
  border-radius: 20px;
  background: ${props => props.isUser ? 'var(--chat-user)' : 'var(--chat-assistant)'};
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  position: relative;
  
  @media (max-width: 768px) {
    max-width: 85%;
  }
`;

const MessageContent = styled.div`
  color: white;
  font-size: 1rem;
  line-height: 1.6;
  
  p {
    margin: 0;
  }
  
  code {
    background: rgba(0, 0, 0, 0.2);
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-size: 0.9em;
  }
  
  pre {
    background: rgba(0, 0, 0, 0.3);
    padding: 1rem;
    border-radius: 8px;
    overflow-x: auto;
    margin: 0.5rem 0;
  }
`;

const MessageTime = styled.div`
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.6);
  margin-top: 0.5rem;
`;

const CharacterAvatar = styled.div<{ emoji: string }>`
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 0.5rem;
  font-size: 1.5rem;
  
  &::before {
    content: '${props => props.emoji}';
  }
`;

const MessageWrapper = styled.div<{ isUser: boolean }>`
  display: flex;
  align-items: flex-end;
  flex-direction: ${props => props.isUser ? 'row-reverse' : 'row'};
`;

const EmotionTag = styled.span<{ emotion: string }>`
  display: inline-block;
  padding: 0.2rem 0.6rem;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  font-size: 0.75rem;
  margin-top: 0.5rem;
  color: rgba(255, 255, 255, 0.8);
`;

interface ChatMessageProps {
  message: {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    emotion?: string;
    timestamp: Date;
  };
  characterId: string;
}

// Character emoji mapping
const characterEmojis: Record<string, string> = {
  alice: '🧚‍♀️',
  max: '🤖',
  luna: '🌙'
};

const ChatMessage: React.FC<ChatMessageProps> = ({ message, characterId }) => {
  const isUser = message.role === 'user';
  const characterEmoji = characterEmojis[characterId] || '🎭';

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: 'numeric', 
      minute: '2-digit'
    });
  };

  return (
    <MessageContainer
      isUser={isUser}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <MessageWrapper isUser={isUser}>
        {!isUser && (
          <CharacterAvatar emoji={characterEmoji} />
        )}
        <MessageBubble isUser={isUser}>
          <MessageContent>
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </MessageContent>
          {message.emotion && !isUser && (
            <EmotionTag emotion={message.emotion}>
              feeling {message.emotion}
            </EmotionTag>
          )}
          <MessageTime>
            {formatTime(message.timestamp)}
          </MessageTime>
        </MessageBubble>
      </MessageWrapper>
    </MessageContainer>
  );
};

export default ChatMessage; 