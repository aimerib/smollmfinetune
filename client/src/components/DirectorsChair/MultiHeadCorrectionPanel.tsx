import React, { useState } from 'react';
import styled from '@emotion/styled';

const Container = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 1.5rem;
  margin: 1rem 0;
`;

const Title = styled.h3`
  margin: 0 0 1rem 0;
  color: white;
`;

const CorrectionTypeSelector = styled.div`
  margin-bottom: 1.5rem;
`;

const RadioGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
`;

const RadioOption = styled.label`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 4px;

  &:hover {
    background: rgba(255, 255, 255, 0.05);
  }
`;

const RadioInput = styled.input`
  width: 16px;
  height: 16px;
`;

const CorrectionForm = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const Label = styled.label`
  font-weight: 500;
  color: rgba(255, 255, 255, 0.9);
`;

const Input = styled.input`
  padding: 0.75rem;
  background: #1f2937;
  color: white;
  border: 1px solid #4b5563;
  border-radius: 6px;

  &:focus {
    outline: none;
    border-color: #3b82f6;
  }
`;

const TextArea = styled.textarea`
  padding: 0.75rem;
  background: #1f2937;
  color: white;
  border: 1px solid #4b5563;
  border-radius: 6px;
  min-height: 80px;
  resize: vertical;

  &:focus {
    outline: none;
    border-color: #3b82f6;
  }
`;

const Select = styled.select`
  padding: 0.75rem;
  background: #1f2937;
  color: white;
  border: 1px solid #4b5563;
  border-radius: 6px;

  &:focus {
    outline: none;
    border-color: #3b82f6;
  }
`;

const Checkbox = styled.input`
  width: 16px;
  height: 16px;
`;

const CheckboxLabel = styled.label`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
`;

const Slider = styled.input`
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 8px;
  background: #374151;
  border-radius: 4px;
  outline: none;

  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    background: #3b82f6;
    border-radius: 50%;
    cursor: pointer;
  }

  &::-moz-range-thumb {
    width: 20px;
    height: 20px;
    background: #3b82f6;
    border-radius: 50%;
    cursor: pointer;
    border: none;
  }
`;

const Button = styled.button`
  padding: 0.75rem 1.5rem;
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  margin-top: 1rem;

  &:hover {
    background: #1d4ed8;
  }

  &:disabled {
    background: #374151;
    cursor: not-allowed;
  }
`;

interface Props {
  onCorrection: (correction: any) => void;
}

export const MultiHeadCorrectionPanel: React.FC<Props> = ({ onCorrection }) => {
  const [correctionType, setCorrectionType] = useState<string>('content');
  const [formData, setFormData] = useState({
    original_text: '',
    improved_text: '',
    improvement_reason: '',
    emotional_state: '',
    control_tokens: '',
    memory_importance: 0.5,
    should_remember: false,
    reason: ''
  });

  const handleTypeChange = (type: string) => {
    setCorrectionType(type);
    // Reset form data when changing types
    setFormData({
      original_text: '',
      improved_text: '',
      improvement_reason: '',
      emotional_state: '',
      control_tokens: '',
      memory_importance: 0.5,
      should_remember: false,
      reason: ''
    });
  };

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = () => {
    let correction: any = {
      reason: formData.reason || 'Director correction'
    };

    switch (correctionType) {
      case 'content':
        correction = {
          type: 'content_correction',
          target_head: 'generation',
          original_text: formData.original_text,
          corrected_text: formData.improved_text,
          reason: formData.improvement_reason || 'Content quality improvement'
        };
        break;

      case 'emotion':
        correction = {
          type: 'emotion_correction',
          target_head: 'control',
          emotional_state: formData.emotional_state,
          control_tokens: formData.control_tokens.split(',').map(t => t.trim()).filter(t => t),
          reason: formData.reason || 'Emotional expression improvement'
        };
        break;

      case 'memory':
        correction = {
          type: 'memory_correction',
          target_head: 'memory',
          memory_importance: formData.memory_importance,
          should_remember: formData.should_remember,
          reason: formData.reason || 'Memory consistency improvement'
        };
        break;
    }

    onCorrection(correction);
  };

  const isFormValid = () => {
    switch (correctionType) {
      case 'content':
        return formData.original_text && formData.improved_text;
      case 'emotion':
        return formData.emotional_state;
      case 'memory':
        return formData.reason;
      default:
        return false;
    }
  };

  return (
    <Container>
      <Title>Correction Type</Title>
      
      <CorrectionTypeSelector>
        <RadioGroup>
          <RadioOption>
            <RadioInput
              type="radio"
              name="correction-type"
              value="content"
              checked={correctionType === 'content'}
              onChange={() => handleTypeChange('content')}
            />
            Content Quality (Generation Head)
          </RadioOption>
          
          <RadioOption>
            <RadioInput
              type="radio"
              name="correction-type"
              value="emotion"
              checked={correctionType === 'emotion'}
              onChange={() => handleTypeChange('emotion')}
            />
            Emotional Expression (Control Head)
          </RadioOption>
          
          <RadioOption>
            <RadioInput
              type="radio"
              name="correction-type"
              value="memory"
              checked={correctionType === 'memory'}
              onChange={() => handleTypeChange('memory')}
            />
            Memory Consistency (Memory Head)
          </RadioOption>
        </RadioGroup>
      </CorrectionTypeSelector>

      <CorrectionForm>
        {correctionType === 'content' && (
          <>
            <Title>Generation Head Correction</Title>
            <FormGroup>
              <Label htmlFor="original-text">Original Text</Label>
              <TextArea
                id="original-text"
                value={formData.original_text}
                onChange={(e) => handleInputChange('original_text', e.target.value)}
                placeholder="Enter the original text..."
              />
            </FormGroup>
            
            <FormGroup>
              <Label htmlFor="improved-text">Improved Text</Label>
              <TextArea
                id="improved-text"
                value={formData.improved_text}
                onChange={(e) => handleInputChange('improved_text', e.target.value)}
                placeholder="Enter the improved text..."
              />
            </FormGroup>
            
            <FormGroup>
              <Label htmlFor="improvement-reason">Improvement Reason</Label>
              <Input
                id="improvement-reason"
                value={formData.improvement_reason}
                onChange={(e) => handleInputChange('improvement_reason', e.target.value)}
                placeholder="Why is this improvement needed?"
              />
            </FormGroup>
          </>
        )}

        {correctionType === 'emotion' && (
          <>
            <Title>Control Head Correction</Title>
            <FormGroup>
              <Label htmlFor="emotional-state">Emotional State</Label>
              <Select
                id="emotional-state"
                value={formData.emotional_state}
                onChange={(e) => handleInputChange('emotional_state', e.target.value)}
              >
                <option value="">Select emotion...</option>
                <option value="excited">Excited</option>
                <option value="happy">Happy</option>
                <option value="calm">Calm</option>
                <option value="curious">Curious</option>
                <option value="thoughtful">Thoughtful</option>
                <option value="playful">Playful</option>
              </Select>
            </FormGroup>
            
            <FormGroup>
              <Label htmlFor="control-tokens">Control Tokens</Label>
              <Input
                id="control-tokens"
                value={formData.control_tokens}
                onChange={(e) => handleInputChange('control_tokens', e.target.value)}
                placeholder="e.g., <excited>, <friendly> (comma-separated)"
              />
            </FormGroup>
            
            <FormGroup>
              <Label htmlFor="emotion-reason">Reason</Label>
              <Input
                id="emotion-reason"
                value={formData.reason}
                onChange={(e) => handleInputChange('reason', e.target.value)}
                placeholder="Why this emotional correction?"
              />
            </FormGroup>
          </>
        )}

        {correctionType === 'memory' && (
          <>
            <Title>Memory Head Correction</Title>
            <FormGroup>
              <Label htmlFor="memory-importance">
                Memory Importance: {formData.memory_importance.toFixed(1)}
              </Label>
              <Slider
                id="memory-importance"
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={formData.memory_importance}
                onChange={(e) => handleInputChange('memory_importance', parseFloat(e.target.value))}
              />
            </FormGroup>
            
            <FormGroup>
              <CheckboxLabel>
                <Checkbox
                  type="checkbox"
                  checked={formData.should_remember}
                  onChange={(e) => handleInputChange('should_remember', e.target.checked)}
                />
                Should Remember This Information
              </CheckboxLabel>
            </FormGroup>
            
            <FormGroup>
              <Label htmlFor="memory-reason">Reason</Label>
              <Input
                id="memory-reason"
                value={formData.reason}
                onChange={(e) => handleInputChange('reason', e.target.value)}
                placeholder="Why this memory correction?"
              />
            </FormGroup>
          </>
        )}

        <Button onClick={handleSubmit} disabled={!isFormValid()}>
          Apply Correction
        </Button>
      </CorrectionForm>
    </Container>
  );
}; 