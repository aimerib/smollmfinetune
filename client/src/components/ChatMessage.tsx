import React from 'react';
import styled from '@emotion/styled';
import { motion } from 'framer-motion';

const MessageContainer = styled(motion.div, {
  shouldForwardProp: (prop) => !['isUser'].includes(prop)
})<{ isUser: boolean }>`
  display: flex;
  gap: 1rem;
  align-items: flex-start;
  flex-direction: ${props => props.isUser ? 'row-reverse' : 'row'};
`;

const Avatar = styled('div', {
  shouldForwardProp: (prop) => !['isUser'].includes(prop)
})<{ isUser: boolean }>`
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  flex-shrink: 0;
  
  ${props => props.isUser ? `
    background: #64748b;
  ` : `
    background: #8b45c1;
  `}
`;

const MessageBubble = styled('div', {
  shouldForwardProp: (prop) => !['isUser'].includes(prop)
})<{ isUser: boolean }>`
  max-width: 500px;
  padding: 0.75rem 1rem;
  border-radius: 16px;
  
  ${props => props.isUser ? `
    background: rgba(100, 116, 139, 0.2);
    border: 1px solid rgba(100, 116, 139, 0.3);
    margin-left: 2rem;
  ` : `
    background: rgba(139, 69, 193, 0.1);
    border: 1px solid rgba(139, 69, 193, 0.2);
    margin-right: 2rem;
  `}
`;

const MessageText = styled.div`
  font-size: 0.95rem;
  line-height: 1.5;
  color: rgba(255, 255, 255, 0.9);
`;

const MessageTime = styled.div<{ isUser: boolean }>`
  font-size: 0.7rem;
  color: rgba(255, 255, 255, 0.4);
  margin-top: 0.25rem;
  text-align: ${props => props.isUser ? 'right' : 'left'};
`;

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  emotion?: string;
  timestamp: Date;
}

interface ChatMessageProps {
  message: Message;
  characterId: string;
}

const ChatMessage = React.forwardRef<HTMLDivElement, ChatMessageProps>(({ message, characterId }, ref) => {
  const isUser = message.role === 'user';
  
  const getCharacterEmoji = (id: string) => {
    switch (id) {
      case 'alice': return '🧚‍♀️';
      case 'max': return '🤖';
      case 'luna': return '🌙';
      default: return '✨';
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <MessageContainer
      ref={ref}
      isUser={isUser}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
    >
      <Avatar isUser={isUser}>
        {isUser ? '👤' : getCharacterEmoji(characterId)}
      </Avatar>
      
      <MessageBubble isUser={isUser}>
        <MessageText>
          {message.content}
        </MessageText>
        
        <MessageTime isUser={isUser}>
          {formatTime(message.timestamp)}
        </MessageTime>
      </MessageBubble>
    </MessageContainer>
  );
});

ChatMessage.displayName = 'ChatMessage';

export default ChatMessage; 