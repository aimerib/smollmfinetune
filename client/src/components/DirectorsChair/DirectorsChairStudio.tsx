import React, { useState, useEffect } from 'react';
import styled from '@emotion/styled';
import websocketService from '../../services/websocketService';
import directorsChairService, { TrainingStatus } from '../../services/directorsChairService';
import { ConversationEditor } from './ConversationEditor';
import { MultiHeadCorrectionPanel } from './MultiHeadCorrectionPanel';
import { TrainingProgressIndicator } from './TrainingProgressIndicator';

const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #0a0a0f;
  color: white;
`;

const Header = styled.div`
  padding: 1rem 2rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
`;

const Title = styled.h1`
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
`;

const Subtitle = styled.p`
  margin: 0.5rem 0 0 0;
  color: rgba(255, 255, 255, 0.7);
`;

const Content = styled.div`
  display: flex;
  flex: 1;
`;

const MainPanel = styled.div`
  flex: 1;
  padding: 2rem;
`;

const SidePanel = styled.div`
  width: 350px;
  border-left: 1px solid rgba(255, 255, 255, 0.1);
  padding: 1rem;
`;

const ControlPanel = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 2rem;
`;

const Button = styled.button`
  padding: 0.5rem 1rem;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;

  &:hover {
    background: #1d4ed8;
  }

  &:disabled {
    background: #374151;
    cursor: not-allowed;
  }
`;

const Select = styled.select`
  padding: 0.5rem;
  background: #374151;
  color: white;
  border: 1px solid #4b5563;
  border-radius: 6px;
`;

const Toggle = styled.label`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
`;

const ToggleInput = styled.input`
  width: 20px;
  height: 20px;
`;

const StatusIndicator = styled.div<{ status: string }>`
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  background: ${props => 
    props.status === 'active' ? '#059669' : 
    props.status === 'training' ? '#d97706' : '#6b7280'};
  color: white;
`;

const HeadStatusGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1rem;
  margin-bottom: 2rem;
`;

const HeadStatus = styled.div`
  padding: 1rem;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  text-align: center;
`;

interface Props {
  characterId?: string;
  initialConversation?: any;
  enableTraining?: boolean;
  onModelUpdate?: (update: any) => void;
}

export const DirectorsChairStudio: React.FC<Props> = ({
  characterId,
  initialConversation,
  enableTraining = false,
  onModelUpdate
}) => {
  const [selectedCharacter, setSelectedCharacter] = useState(characterId || '');
  const [trainingEnabled, setTrainingEnabled] = useState(enableTraining);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({});
  const [showCorrectionPanel, setShowCorrectionPanel] = useState(false);
  const [selectedConversation, setSelectedConversation] = useState(initialConversation);

  useEffect(() => {
    websocketService.connect();
    return () => websocketService.disconnect();
  }, []);

  useEffect(() => {
    if (trainingEnabled) {
      fetchTrainingStatus();
      const interval = setInterval(fetchTrainingStatus, 2000);
      return () => clearInterval(interval);
    }
  }, [trainingEnabled]);

  const fetchTrainingStatus = async () => {
    try {
      const status = await directorsChairService.getTrainingStatus();
      setTrainingStatus(status);
    } catch (error) {
      console.error('Failed to fetch training status:', error);
    }
  };

  const handleStartSession = async () => {
    if (!selectedCharacter) return;

    try {
      const session = await directorsChairService.startSession({
        character_id: selectedCharacter,
        mode: 'directors_chair',
        training_enabled: trainingEnabled
      });
      setSessionId(session.session_id);
    } catch (error) {
      console.error('Failed to start session:', error);
    }
  };

  const handleCorrection = async (correction: any) => {
    if (!selectedConversation) return;

    try {
      await directorsChairService.applyCorrection(
        selectedConversation.id,
        correction
      );
      setShowCorrectionPanel(false);
      // Refresh training status
      fetchTrainingStatus();
    } catch (error) {
      console.error('Failed to apply correction:', error);
    }
  };

  const handleEditResponse = () => {
    setShowCorrectionPanel(true);
  };

  return (
    <Container>
      <Header>
        <Title>Director's Chair</Title>
        <Subtitle>Training Studio</Subtitle>
      </Header>

      <Content>
        <MainPanel>
          <ControlPanel>
            {!selectedCharacter && (
              <>
                <span>Select Character:</span>
                <Select 
                  value={selectedCharacter}
                  onChange={(e) => setSelectedCharacter(e.target.value)}
                  aria-label="character"
                >
                  <option value="">Choose a character...</option>
                  <option value="alice">Alice</option>
                  <option value="max">Max</option>
                  <option value="luna">Luna</option>
                </Select>
              </>
            )}
            
            <Toggle>
              <ToggleInput
                type="checkbox"
                checked={trainingEnabled}
                onChange={(e) => setTrainingEnabled(e.target.checked)}
                role="switch"
                aria-label="Enable Live Training"
              />
              Enable Live Training
            </Toggle>

            {trainingEnabled && (
              <span>Training Mode: Active</span>
            )}

            <Button onClick={handleStartSession} disabled={!selectedCharacter}>
              Start Training Session
            </Button>
          </ControlPanel>

          <HeadStatusGrid>
            <HeadStatus data-testid="head-status-generation">
              <h3>Generation Head</h3>
              <StatusIndicator status={trainingStatus?.generation_head?.status || 'idle'}>
                {trainingStatus?.generation_head?.status || 'Idle'}
              </StatusIndicator>
            </HeadStatus>
            <HeadStatus data-testid="head-status-control">
              <h3>Control Head</h3>
              <StatusIndicator status={trainingStatus?.control_head?.status || 'idle'}>
                {trainingStatus?.control_head?.status || 'Idle'}
              </StatusIndicator>
            </HeadStatus>
            <HeadStatus data-testid="head-status-memory">
              <h3>Memory Head</h3>
              <StatusIndicator status={trainingStatus?.memory_head?.status || 'idle'}>
                {trainingStatus?.memory_head?.status || 'Idle'}
              </StatusIndicator>
            </HeadStatus>
          </HeadStatusGrid>

          {selectedConversation && (
            <ConversationEditor
              conversation={selectedConversation}
              onCorrection={handleCorrection}
              onEdit={handleEditResponse}
            />
          )}

          {showCorrectionPanel && (
            <MultiHeadCorrectionPanel onCorrection={handleCorrection} />
          )}
        </MainPanel>

        <SidePanel>
          <TrainingProgressIndicator
            trainingStatus={trainingStatus}
            onTrainingComplete={(headType) => {
              console.log(`Training completed for ${headType}`);
              if (onModelUpdate) {
                onModelUpdate({ head_type: headType, ready_for_inference: true });
              }
            }}
          />
          <div data-testid="model-status-indicator">
            Model Status: Ready
          </div>
        </SidePanel>
      </Content>
    </Container>
  );
}; 