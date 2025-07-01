import React, { useState, useEffect, useRef } from 'react';
import styled from '@emotion/styled';
import { motion, AnimatePresence } from 'framer-motion';
import { useParams, useSearchParams } from 'react-router-dom';
import { FiSend, FiSmile, FiHeart, FiActivity } from 'react-icons/fi';
import ChatMessage from '../components/ChatMessage';
import CharacterHeader from '../components/CharacterHeader';
import EmotionIndicator from '../components/EmotionIndicator';
import { chatService } from '../utils/chatService';

const Container = styled.div`
  height: 100vh;
  display: flex;
  flex-direction: column;
  position: relative;
  z-index: 1;
`;

const ChatContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  max-width: 900px;
  width: 100%;
  margin: 0 auto;
  padding: 1rem;
  gap: 1rem;
`;

const MessagesArea = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 3px;
  }
`;

const InputArea = styled(motion.div)`
  background: rgba(0, 0, 0, 0.3);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  padding: 1rem;
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const InputField = styled.input`
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: white;
  font-size: 1rem;
  font-family: 'Inter', sans-serif;
  
  &::placeholder {
    color: var(--text-muted);
  }
`;

const SendButton = styled(motion.button)`
  width: 48px;
  height: 48px;
  border-radius: 50%;
  border: none;
  background: var(--primary-gradient);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const TypingIndicator = styled(motion.div)`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  width: fit-content;
`;

const TypingDot = styled(motion.span)`
  width: 8px;
  height: 8px;
  background: white;
  border-radius: 50%;
`;

const EmotionBar = styled(motion.div)`
  position: absolute;
  top: 20px;
  right: 20px;
  display: flex;
  gap: 1rem;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(20px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  emotion?: string;
  timestamp: Date;
}

const ChatPage: React.FC = () => {
  const { sessionId } = useParams();
  const [searchParams] = useSearchParams();
  const characterId = searchParams.get('character') || 'alice';
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState('happy');
  const [memoryImportance, setMemoryImportance] = useState(0);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const sendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    try {
      // Call the inference engine
      const response = await chatService.sendMessage(
        sessionId!,
        characterId,
        inputValue
      );

      // Extract emotion from control tokens
      const emotionToken = response.control_tokens.find(
        t => t.token.includes('emotion')
      );
      const emotion = emotionToken ? 
        emotionToken.token.replace('<emotion_', '').replace('>', '') : 
        'neutral';

      setCurrentEmotion(emotion);
      setMemoryImportance(response.memory_metadata.importance || 0);

      const assistantMessage: Message = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: response.generation_text,
        emotion: emotion,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Failed to send message:', error);
      // Show error message
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <Container>
      <EmotionBar
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
      >
        <EmotionIndicator
          emotion={currentEmotion}
          intensity={0.8}
          label="Mood"
          icon={<FiSmile />}
        />
        <EmotionIndicator
          emotion="memory"
          intensity={memoryImportance}
          label="Memory"
          icon={<FiActivity />}
        />
      </EmotionBar>

      <ChatContainer>
        <CharacterHeader characterId={characterId} emotion={currentEmotion} />
        
        <MessagesArea className="glass">
          <AnimatePresence>
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                characterId={characterId}
              />
            ))}
          </AnimatePresence>
          
          {isTyping && (
            <TypingIndicator
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
            >
              <span style={{ color: 'var(--text-secondary)' }}>
                {characterId} is typing
              </span>
              <TypingDot
                animate={{ y: [0, -5, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
              />
              <TypingDot
                animate={{ y: [0, -5, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
              />
              <TypingDot
                animate={{ y: [0, -5, 0] }}
                transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
              />
            </TypingIndicator>
          )}
          
          <div ref={messagesEndRef} />
        </MessagesArea>

        <InputArea
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <InputField
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={isTyping}
          />
          <SendButton
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={sendMessage}
            disabled={!inputValue.trim() || isTyping}
          >
            <FiSend size={20} />
          </SendButton>
        </InputArea>
      </ChatContainer>
    </Container>
  );
};

export default ChatPage; 