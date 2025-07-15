import React, { useState } from 'react';
import styled from '@emotion/styled';

const Container = styled.div`
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  margin: 1rem 0;
  background: rgba(255, 255, 255, 0.02);
`;

const ConversationTurn = styled.div`
  padding: 1rem;
`;

const UserMessage = styled.div`
  padding: 0.75rem;
  margin-bottom: 0.5rem;
  background: rgba(59, 130, 246, 0.1);
  border-radius: 6px;
  border-left: 3px solid #3b82f6;
`;

const AssistantMessage = styled.div`
  padding: 0.75rem;
  background: rgba(16, 185, 129, 0.1);
  border-radius: 6px;
  border-left: 3px solid #10b981;
  position: relative;
`;

const ScoreDisplay = styled.div`
  display: flex;
  gap: 1rem;
  margin: 0.5rem 0;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.7);
`;

const Score = styled.span<{ score: number }>`
  color: ${props => 
    props.score >= 0.8 ? '#10b981' : 
    props.score >= 0.6 ? '#f59e0b' : '#ef4444'};
  font-weight: 500;
`;

const EditControls = styled.div`
  display: flex;
  gap: 0.5rem;
  margin-top: 0.5rem;
`;

const Button = styled.button`
  padding: 0.375rem 0.75rem;
  background: #374151;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.75rem;

  &:hover {
    background: #4b5563;
  }
`;

const PrimaryButton = styled(Button)`
  background: #2563eb;

  &:hover {
    background: #1d4ed8;
  }
`;

const TextEditor = styled.textarea`
  width: 100%;
  min-height: 80px;
  padding: 0.75rem;
  background: #1f2937;
  color: white;
  border: 1px solid #4b5563;
  border-radius: 6px;
  resize: vertical;
  font-family: inherit;

  &:focus {
    outline: none;
    border-color: #3b82f6;
  }
`;

const CorrectionHistory = styled.div`
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

const CorrectionItem = styled.div`
  padding: 0.5rem;
  margin: 0.25rem 0;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
  font-size: 0.85rem;
`;

interface Conversation {
  id: string;
  user_message: string;
  assistant_response: string;
  metadata?: {
    generation_head_score?: number;
    control_head_score?: number;
    memory_head_score?: number;
  };
  corrections?: Array<{
    id: string;
    type: string;
    target_head: string;
    applied_at: string;
    reason: string;
  }>;
}

interface Props {
  conversation: Conversation;
  onCorrection: (correction: any) => void;
  onEdit?: () => void;
}

export const ConversationEditor: React.FC<Props> = ({
  conversation,
  onCorrection,
  onEdit
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedText, setEditedText] = useState(conversation.assistant_response);

  const handleEditToggle = () => {
    if (isEditing) {
      setIsEditing(false);
      setEditedText(conversation.assistant_response);
    } else {
      setIsEditing(true);
      if (onEdit) onEdit();
    }
  };

  const handleApplyCorrection = () => {
    if (editedText !== conversation.assistant_response) {
      const correction = {
        type: 'content_correction',
        target_head: 'generation',
        original_text: conversation.assistant_response,
        corrected_text: editedText,
        reason: 'Edited response for improved quality'
      };
      onCorrection(correction);
    }
    setIsEditing(false);
  };

  const handleAddCorrection = () => {
    if (onEdit) onEdit();
  };

  return (
    <Container>
      <ConversationTurn>
        <UserMessage>
          {conversation.user_message}
        </UserMessage>

        <AssistantMessage>
          {isEditing ? (
            <TextEditor
              value={editedText}
              onChange={(e) => setEditedText(e.target.value)}
              placeholder="Edit the assistant's response..."
            />
          ) : (
            conversation.assistant_response
          )}
        </AssistantMessage>

        {conversation.metadata && (
          <ScoreDisplay>
            <span data-testid="generation-score">
              Generation: <Score score={conversation.metadata.generation_head_score || 0}>
                {conversation.metadata.generation_head_score?.toFixed(1) || '0.0'}
              </Score>
            </span>
            <span data-testid="control-score">
              Control: <Score score={conversation.metadata.control_head_score || 0}>
                {conversation.metadata.control_head_score?.toFixed(1) || '0.0'}
              </Score>
            </span>
            <span data-testid="memory-score">
              Memory: <Score score={conversation.metadata.memory_head_score || 0}>
                {conversation.metadata.memory_head_score?.toFixed(1) || '0.0'}
              </Score>
            </span>
          </ScoreDisplay>
        )}

        <EditControls>
          {isEditing ? (
            <>
              <PrimaryButton onClick={handleApplyCorrection}>
                Apply Correction
              </PrimaryButton>
              <Button onClick={handleEditToggle}>
                Cancel
              </Button>
            </>
          ) : (
            <>
              <Button onClick={handleEditToggle}>
                Edit Response
              </Button>
              <Button onClick={handleAddCorrection}>
                Add Correction
              </Button>
            </>
          )}
        </EditControls>

        {conversation.corrections && conversation.corrections.length > 0 && (
          <CorrectionHistory>
            <h4>Correction History</h4>
            {conversation.corrections.map((correction) => (
              <CorrectionItem key={correction.id}>
                <strong>{correction.type}</strong> → {correction.target_head} head
                <br />
                <span style={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  {correction.reason}
                </span>
              </CorrectionItem>
            ))}
          </CorrectionHistory>
        )}
      </ConversationTurn>
    </Container>
  );
}; 