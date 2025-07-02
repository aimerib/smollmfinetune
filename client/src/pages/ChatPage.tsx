import React, { useState, useEffect, useRef } from 'react';
import styled from '@emotion/styled';
import { motion, AnimatePresence } from 'framer-motion';
import { useParams, useSearchParams } from 'react-router-dom';
import { FiSend } from 'react-icons/fi';
import ChatMessage from '../components/ChatMessage';
import CharacterHeader from '../components/CharacterHeader';
import { chatService } from '../utils/chatService';

const Container = styled.div`
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #0f0f1e;
  color: white;
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
  
  &::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
  }
  
  &::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 3px;
  }
`;

const InputArea = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1rem;
  display: flex;
  align-items: center;
  gap: 1rem;
  
  &:focus-within {
    border-color: rgba(139, 69, 193, 0.5);
  }
`;

const InputField = styled.input`
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: white;
  font-size: 1rem;
  
  &::placeholder {
    color: rgba(255, 255, 255, 0.4);
  }
`;

const SendButton = styled.button`
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: none;
  background: #8b45c1;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: #9b55d1;
    transform: scale(1.05);
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const TypingIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem;
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.9rem;
`;

const LoadingMessage = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  color: rgba(255, 255, 255, 0.6);
  font-size: 1.1rem;
`;

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  emotion?: string;
  timestamp: Date;
}

const ChatPage: React.FC = () => {
  const { sessionId: urlSessionId } = useParams();
  const [searchParams] = useSearchParams();
  const characterId = searchParams.get('character') || 'alice';
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState('neutral');
  const [sessionId, setSessionId] = useState<string | null>(urlSessionId || null);
  const [isInitializing, setIsInitializing] = useState(true);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const characterNames = {
    alice: 'Alice',
    max: 'Max',
    luna: 'Luna'
  };

  useEffect(() => {
    const initializeSession = async () => {
      if (!sessionId) {
        try {
          const newSessionId = await chatService.startSession(characterId);
          setSessionId(newSessionId);
        } catch (error) {
          console.error('Failed to initialize session:', error);
        }
      }
      setIsInitializing(false);
    };

    initializeSession();
  }, [characterId, sessionId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || !sessionId) return;

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
      const response = await chatService.sendMessage(
        sessionId,
        characterId,
        inputValue
      );

      const emotionToken = response.control_tokens.find(
        t => t.token.includes('emotion')
      );
      const emotion = emotionToken ? 
        emotionToken.token.replace('<emotion_', '').replace('>', '') : 
        'neutral';

      setCurrentEmotion(emotion);

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
      const errorMessage: Message = {
        id: `msg-error-${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        emotion: 'confused',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
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

  if (isInitializing) {
    return (
      <Container>
        <ChatContainer>
          <LoadingMessage>
            Connecting with {characterNames[characterId as keyof typeof characterNames]}...
          </LoadingMessage>
        </ChatContainer>
      </Container>
    );
  }

  return (
    <Container>
      <ChatContainer>
        <CharacterHeader characterId={characterId} emotion={currentEmotion} />
        
        <MessagesArea>
          <AnimatePresence mode="popLayout">
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                characterId={characterId}
              />
            ))}
          </AnimatePresence>
          
          {isTyping && (
            <TypingIndicator>
              {characterNames[characterId as keyof typeof characterNames]} is typing...
            </TypingIndicator>
          )}
          
          <div ref={messagesEndRef} />
        </MessagesArea>

        <InputArea>
          <InputField
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={`Message ${characterNames[characterId as keyof typeof characterNames]}...`}
            disabled={isTyping}
          />
          <SendButton
            onClick={sendMessage}
            disabled={!inputValue.trim() || isTyping}
          >
            {React.createElement(FiSend as React.ComponentType<any>, { size: 18 })}
          </SendButton>
        </InputArea>
      </ChatContainer>
    </Container>
  );
};

export default ChatPage; 