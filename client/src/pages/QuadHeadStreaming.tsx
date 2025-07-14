/**
 * QuadHeadStreaming Page
 * 
 * Complete page for multimodal quad-head streaming experience.
 * Integrates character selection, streaming controls, and real-time multimodal outputs.
 */

import React, { useState, useEffect, useRef } from 'react';
import styled from '@emotion/styled';
import { QuadHeadStreamingPlayer } from '../components/QuadHeadStreamingPlayer';

interface Character {
  id: string;
  name: string;
  description: string;
  personality: {
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
  };
}

interface StreamingSession {
  characterId: string;
  sessionId: string;
  createdAt: Date;
  messageCount: number;
}

const PageContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  color: white;
  padding: 2rem;
`;

const Header = styled.div`
  text-align: center;
  margin-bottom: 3rem;
`;

const Title = styled.h1`
  font-size: 3rem;
  font-weight: 700;
  background: linear-gradient(45deg, #4fc3f7, #29b6f6, #03a9f4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 0.5rem;
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 2rem;
`;

const MainContent = styled.div`
  max-width: 1400px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: 350px 1fr;
  gap: 2rem;
  align-items: start;
`;

const Sidebar = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  padding: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

const SidebarSection = styled.div`
  margin-bottom: 2rem;
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const SectionTitle = styled.h3`
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #4fc3f7;
`;

const CharacterCard = styled.div<{ selected: boolean }>`
  background: ${props => props.selected ? 'rgba(79, 195, 247, 0.2)' : 'rgba(255, 255, 255, 0.05)'};
  border: 1px solid ${props => props.selected ? '#4fc3f7' : 'rgba(255, 255, 255, 0.1)'};
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 0.5rem;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(79, 195, 247, 0.1);
    border-color: #4fc3f7;
  }
`;

const CharacterName = styled.div`
  font-weight: 600;
  margin-bottom: 0.5rem;
`;

const CharacterDescription = styled.div`
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 0.5rem;
`;

const PersonalityBar = styled.div`
  font-size: 0.8rem;
  margin: 0.25rem 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const PersonalityLabel = styled.span`
  min-width: 40px;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.6);
`;

const PersonalityTrack = styled.div`
  flex: 1;
  height: 4px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 2px;
  overflow: hidden;
`;

const PersonalityFill = styled.div<{ value: number }>`
  width: ${props => props.value * 100}%;
  height: 100%;
  background: linear-gradient(90deg, #4fc3f7, #29b6f6);
  border-radius: 2px;
`;

const SettingsPanel = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 1rem;
`;

const SettingRow = styled.div`
  margin-bottom: 1rem;
  
  &:last-child {
    margin-bottom: 0;
  }
`;

const SettingLabel = styled.label`
  display: block;
  font-size: 0.9rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: rgba(255, 255, 255, 0.8);
`;

const SettingSlider = styled.input`
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.2);
  outline: none;
  -webkit-appearance: none;
  
  &::-webkit-slider-thumb {
    appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #4fc3f7;
    cursor: pointer;
    border: 2px solid white;
  }
  
  &::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #4fc3f7;
    cursor: pointer;
    border: 2px solid white;
  }
`;

const SettingValue = styled.span`
  font-size: 0.85rem;
  color: #4fc3f7;
  float: right;
`;

const SessionInfo = styled.div`
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.6);
  padding: 0.75rem;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 6px;
  border-left: 3px solid #4fc3f7;
`;

const StreamingArea = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  padding: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  min-height: 600px;
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 3rem;
  color: rgba(255, 255, 255, 0.5);
`;

export const QuadHeadStreaming: React.FC = () => {
  const [selectedCharacter, setSelectedCharacter] = useState<Character | null>(null);
  const [session, setSession] = useState<StreamingSession | null>(null);
  const [characters] = useState<Character[]>([
    {
      id: 'clara',
      name: 'Clara',
      description: 'A thoughtful explorer with deep curiosity about the world.',
      personality: {
        openness: 0.85,
        conscientiousness: 0.7,
        extraversion: 0.4,
        agreeableness: 0.8,
        neuroticism: 0.3
      }
    },
    {
      id: 'marcus',
      name: 'Marcus',
      description: 'An energetic adventurer who thrives on social interaction.',
      personality: {
        openness: 0.6,
        conscientiousness: 0.5,
        extraversion: 0.9,
        agreeableness: 0.7,
        neuroticism: 0.2
      }
    },
    {
      id: 'sage',
      name: 'Sage',
      description: 'A wise and methodical thinker with vast knowledge.',
      personality: {
        openness: 0.9,
        conscientiousness: 0.95,
        extraversion: 0.2,
        agreeableness: 0.6,
        neuroticism: 0.1
      }
    }
  ]);

  // Streaming settings
  const [temperature, setTemperature] = useState(0.8);
  const [speechTemperature, setSpeechTemperature] = useState(0.7);
  const [maxLength, setMaxLength] = useState(100);
  const [speechEnabled, setSpeechEnabled] = useState(true);

  const streamingPlayerRef = useRef<any>(null);

  useEffect(() => {
    if (selectedCharacter && !session) {
      const newSession: StreamingSession = {
        characterId: selectedCharacter.id,
        sessionId: `session_${Date.now()}`,
        createdAt: new Date(),
        messageCount: 0
      };
      setSession(newSession);
    }
  }, [selectedCharacter, session]);

  const handleCharacterSelect = (character: Character) => {
    setSelectedCharacter(character);
    setSession(null); // Reset session for new character
  };

  const handleTextGenerated = (token: string) => {
    console.log('Generated text token:', token);
  };

  const handleSpeechFrame = (frame: number[]) => {
    console.log('Generated speech frame:', frame);
  };

  const handleControlSignal = (signal: string) => {
    console.log('Control signal:', signal);
  };

  const handleMemoryUpdate = (update: Record<string, any>) => {
    console.log('Memory update:', update);
  };

  const handleGenerationComplete = () => {
    if (session) {
      setSession({
        ...session,
        messageCount: session.messageCount + 1
      });
    }
  };

  const handleError = (error: Error) => {
    console.error('Streaming error:', error);
  };

  const handleConnectionChange = (status: string) => {
    console.log('Connection status:', status);
  };

  const personalityTraits = [
    { key: 'openness', label: 'O', name: 'Openness' },
    { key: 'conscientiousness', label: 'C', name: 'Conscientiousness' },
    { key: 'extraversion', label: 'E', name: 'Extraversion' },
    { key: 'agreeableness', label: 'A', name: 'Agreeableness' },
    { key: 'neuroticism', label: 'N', name: 'Neuroticism' }
  ];

  return (
    <PageContainer>
      <Header>
        <Title>Quad-Head Multimodal Streaming</Title>
        <Subtitle>
          Experience real-time character interaction with text, speech, control signals, and memory updates
        </Subtitle>
      </Header>

      <MainContent>
        <Sidebar>
          <SidebarSection>
            <SectionTitle>Select Character</SectionTitle>
            {characters.map(character => (
              <CharacterCard
                key={character.id}
                selected={selectedCharacter?.id === character.id}
                onClick={() => handleCharacterSelect(character)}
              >
                <CharacterName>{character.name}</CharacterName>
                <CharacterDescription>{character.description}</CharacterDescription>
                
                {personalityTraits.map(trait => (
                  <PersonalityBar key={trait.key}>
                    <PersonalityLabel>{trait.label}</PersonalityLabel>
                    <PersonalityTrack>
                      <PersonalityFill 
                        value={character.personality[trait.key as keyof typeof character.personality]} 
                      />
                    </PersonalityTrack>
                  </PersonalityBar>
                ))}
              </CharacterCard>
            ))}
          </SidebarSection>

          {selectedCharacter && (
            <>
              <SidebarSection>
                <SectionTitle>Generation Settings</SectionTitle>
                <SettingsPanel>
                  <SettingRow>
                    <SettingLabel htmlFor="temperature">
                      Text Temperature
                      <SettingValue>{temperature.toFixed(2)}</SettingValue>
                    </SettingLabel>
                    <SettingSlider
                      id="temperature"
                      type="range"
                      min="0.1"
                      max="2.0"
                      step="0.1"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    />
                  </SettingRow>

                  <SettingRow>
                    <SettingLabel htmlFor="speechTemperature">
                      Speech Temperature
                      <SettingValue>{speechTemperature.toFixed(2)}</SettingValue>
                    </SettingLabel>
                    <SettingSlider
                      id="speechTemperature"
                      type="range"
                      min="0.1"
                      max="2.0"
                      step="0.1"
                      value={speechTemperature}
                      onChange={(e) => setSpeechTemperature(parseFloat(e.target.value))}
                    />
                  </SettingRow>

                  <SettingRow>
                    <SettingLabel htmlFor="maxLength">
                      Max Length
                      <SettingValue>{maxLength}</SettingValue>
                    </SettingLabel>
                    <SettingSlider
                      id="maxLength"
                      type="range"
                      min="10"
                      max="500"
                      step="10"
                      value={maxLength}
                      onChange={(e) => setMaxLength(parseInt(e.target.value))}
                    />
                  </SettingRow>

                  <SettingRow>
                    <SettingLabel htmlFor="speechEnabled">
                      Speech Generation
                    </SettingLabel>
                    <input
                      id="speechEnabled"
                      type="checkbox"
                      checked={speechEnabled}
                      onChange={(e) => setSpeechEnabled(e.target.checked)}
                      style={{ 
                        width: '18px', 
                        height: '18px', 
                        accentColor: '#4fc3f7' 
                      }}
                    />
                  </SettingRow>
                </SettingsPanel>
              </SidebarSection>

              {session && (
                <SidebarSection>
                  <SectionTitle>Session Info</SectionTitle>
                  <SessionInfo>
                    <div><strong>Character:</strong> {selectedCharacter.name}</div>
                    <div><strong>Session:</strong> {session.sessionId.slice(-8)}</div>
                    <div><strong>Started:</strong> {session.createdAt.toLocaleTimeString()}</div>
                    <div><strong>Messages:</strong> {session.messageCount}</div>
                  </SessionInfo>
                </SidebarSection>
              )}
            </>
          )}
        </Sidebar>

        <StreamingArea>
          {selectedCharacter ? (
            <QuadHeadStreamingPlayer
              ref={streamingPlayerRef}
              characterId={selectedCharacter.id}
              wsUrl={`ws://localhost:8000/api/quad-head/stream/${selectedCharacter.id}`}
              onTextGenerated={handleTextGenerated}
              onSpeechFrame={handleSpeechFrame}
              onControlSignal={handleControlSignal}
              onMemoryUpdate={handleMemoryUpdate}
              onGenerationComplete={handleGenerationComplete}
              onError={handleError}
              onConnectionChange={handleConnectionChange}
              defaultSettings={{
                temperature,
                speechTemperature,
                maxLength,
                forceSpeech: speechEnabled
              }}
            />
          ) : (
            <EmptyState>
              <h3>Select a character to begin streaming</h3>
              <p>Choose a character from the sidebar to start multimodal interaction</p>
            </EmptyState>
          )}
        </StreamingArea>
      </MainContent>
    </PageContainer>
  );
}; 