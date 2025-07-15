import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/design-system.css';

interface PersonalityTrait {
  name: string;
  code: string;
  value: number;
  description: string;
  color: string;
}

interface CharacterData {
  name: string;
  description: string;
  worldId: string;
  personality: PersonalityTrait[];
  backstory: string;
  goals: string;
  relationships: string;
}

const CharacterBuilder: React.FC = () => {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [hoveredTrait, setHoveredTrait] = useState<string | null>(null);
  const [character, setCharacter] = useState<CharacterData>({
    name: '',
    description: '',
    worldId: '',
    personality: [
      { name: 'Openness', code: 'O', value: 0.5, description: 'Creativity and openness to new experiences', color: 'var(--orange-400)' },
      { name: 'Conscientiousness', code: 'C', value: 0.5, description: 'Organization and dependability', color: 'var(--orange-500)' },
      { name: 'Extraversion', code: 'E', value: 0.5, description: 'Sociability and enthusiasm', color: 'var(--orange-600)' },
      { name: 'Agreeableness', code: 'A', value: 0.5, description: 'Cooperation and trust', color: 'var(--orange-700)' },
      { name: 'Neuroticism', code: 'N', value: 0.5, description: 'Emotional sensitivity and anxiety', color: 'var(--orange-800)' }
    ],
    backstory: '',
    goals: '',
    relationships: ''
  });

  const updatePersonalityTrait = (code: string, value: number) => {
    setCharacter(prev => ({
      ...prev,
      personality: prev.personality.map(trait =>
        trait.code === code ? { ...trait, value } : trait
      )
    }));
  };

  const renderStep = () => {
    switch (step) {
      case 1:
        return (
          <div className="step-content animate-slide-up">
            <h2 className="step-title">Basic Information</h2>
            <p className="step-description text-secondary">
              Let's start with the fundamentals of your character
            </p>
            
            <div className="form-group">
              <label className="form-label">Character Name</label>
              <input
                type="text"
                className="input"
                placeholder="Enter a memorable name..."
                value={character.name}
                onChange={(e) => setCharacter({ ...character, name: e.target.value })}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Description</label>
              <textarea
                className="input textarea"
                placeholder="Describe your character's appearance, demeanor, and first impression..."
                rows={4}
                value={character.description}
                onChange={(e) => setCharacter({ ...character, description: e.target.value })}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Select World</label>
              <select 
                className="input"
                value={character.worldId}
                onChange={(e) => setCharacter({ ...character, worldId: e.target.value })}
              >
                <option value="">Choose a world...</option>
                <option value="cyberpunk">Neo Tokyo 2077</option>
                <option value="fantasy">Kingdom of Aetheria</option>
                <option value="space">Galactic Federation</option>
              </select>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="step-content animate-slide-up">
            <h2 className="step-title">Personality Matrix</h2>
            <p className="step-description text-secondary">
              Define your character's psychological profile using the Big Five model
            </p>

            <div className="personality-grid">
              {character.personality.map((trait) => (
                <div 
                  key={trait.code}
                  className="trait-card"
                  onMouseEnter={() => setHoveredTrait(trait.code)}
                  onMouseLeave={() => setHoveredTrait(null)}
                >
                  <div className="trait-header">
                    <h4 className="trait-name">{trait.name}</h4>
                    <span 
                      className="trait-value"
                      style={{ color: trait.color }}
                    >
                      {Math.round(trait.value * 100)}%
                    </span>
                  </div>
                  
                  <p className="trait-description text-muted">
                    {trait.description}
                  </p>
                  
                  <div className="slider-container">
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={trait.value * 100}
                      onChange={(e) => updatePersonalityTrait(trait.code, Number(e.target.value) / 100)}
                      className="personality-slider"
                      style={{ '--slider-color': trait.color } as React.CSSProperties}
                    />
                    <div className="slider-labels">
                      <span className="text-muted">Low</span>
                      <span className="text-muted">High</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Personality Radar Chart Preview */}
            <div className="radar-preview card glass">
              <h4>Personality Overview</h4>
              <div className="radar-container">
                <svg viewBox="0 0 200 200" className="radar-chart">
                  {/* Pentagon background */}
                  <polygon
                    points="100,30 170,70 150,150 50,150 30,70"
                    fill="none"
                    stroke="var(--color-border)"
                    strokeWidth="1"
                    opacity="0.3"
                  />
                  
                  {/* Character's personality shape */}
                  <polygon
                    points={character.personality.map((trait, i) => {
                      const angle = (i * 72 - 90) * Math.PI / 180;
                      const radius = trait.value * 70;
                      const x = 100 + radius * Math.cos(angle);
                      const y = 100 + radius * Math.sin(angle);
                      return `${x},${y}`;
                    }).join(' ')}
                    fill="var(--orange-500)"
                    fillOpacity="0.3"
                    stroke="var(--orange-500)"
                    strokeWidth="2"
                    className={hoveredTrait ? 'dimmed' : ''}
                  />
                  
                  {/* Trait labels */}
                  {character.personality.map((trait, i) => {
                    const angle = (i * 72 - 90) * Math.PI / 180;
                    const x = 100 + 85 * Math.cos(angle);
                    const y = 100 + 85 * Math.sin(angle);
                    return (
                      <text
                        key={trait.code}
                        x={x}
                        y={y}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        className={`trait-label ${hoveredTrait === trait.code ? 'highlighted' : ''}`}
                        fill={hoveredTrait === trait.code ? trait.color : 'var(--color-text-secondary)'}
                        fontSize="12"
                        fontWeight="500"
                      >
                        {trait.code}
                      </text>
                    );
                  })}
                </svg>
              </div>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="step-content animate-slide-up">
            <h2 className="step-title">Character Depth</h2>
            <p className="step-description text-secondary">
              Bring your character to life with rich backstory and motivations
            </p>

            <div className="form-group">
              <label className="form-label">Backstory</label>
              <textarea
                className="input textarea"
                placeholder="What shaped them into who they are today? Key events, childhood, formative experiences..."
                rows={5}
                value={character.backstory}
                onChange={(e) => setCharacter({ ...character, backstory: e.target.value })}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Goals & Motivations</label>
              <textarea
                className="input textarea"
                placeholder="What drives them? What do they want to achieve? What are their deepest desires and fears?"
                rows={4}
                value={character.goals}
                onChange={(e) => setCharacter({ ...character, goals: e.target.value })}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Key Relationships</label>
              <textarea
                className="input textarea"
                placeholder="Important people in their life, allies, rivals, family, mentors..."
                rows={4}
                value={character.relationships}
                onChange={(e) => setCharacter({ ...character, relationships: e.target.value })}
              />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="character-builder">
      <div className="container">
        {/* Header */}
        <div className="builder-header">
          <button 
            className="btn btn-ghost"
            onClick={() => navigate('/creator')}
          >
            ← Back to Dashboard
          </button>
          
          <div className="progress-indicator">
            {[1, 2, 3].map((i) => (
              <div 
                key={i}
                className={`progress-dot ${i === step ? 'active' : ''} ${i < step ? 'completed' : ''}`}
                onClick={() => i <= step && setStep(i)}
              />
            ))}
          </div>
        </div>

        {/* Main Content */}
        <div className="builder-content">
          <div className="content-wrapper">
            {renderStep()}
          </div>

          {/* Navigation */}
          <div className="builder-navigation">
            <button 
              className="btn btn-secondary"
              onClick={() => setStep(step - 1)}
              disabled={step === 1}
            >
              Previous
            </button>
            
            {step < 3 ? (
              <button 
                className="btn btn-primary orange-glow"
                onClick={() => setStep(step + 1)}
                disabled={
                  (step === 1 && (!character.name || !character.worldId)) ||
                  (step === 2 && character.personality.some(t => t.value === 0))
                }
              >
                Next Step →
              </button>
            ) : (
              <button 
                className="btn btn-primary orange-glow"
                onClick={() => {
                  // TODO: Save character
                  navigate('/creator/characters');
                }}
              >
                <span>✨</span>
                Create Character
              </button>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .character-builder {
          min-height: 100vh;
          padding: var(--space-8) 0;
        }

        .builder-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-12);
        }

        .progress-indicator {
          display: flex;
          gap: var(--space-2);
        }

        .progress-dot {
          width: 12px;
          height: 12px;
          border-radius: var(--radius-full);
          background: var(--color-border);
          cursor: pointer;
          transition: all var(--duration-normal) var(--ease-out);
        }

        .progress-dot.active {
          background: var(--color-primary);
          transform: scale(1.2);
        }

        .progress-dot.completed {
          background: var(--color-primary);
          opacity: 0.5;
        }

        .builder-content {
          max-width: 800px;
          margin: 0 auto;
        }

        .content-wrapper {
          min-height: 500px;
          margin-bottom: var(--space-8);
        }

        .step-title {
          margin-bottom: var(--space-2);
        }

        .step-description {
          margin-bottom: var(--space-8);
        }

        .form-group {
          margin-bottom: var(--space-6);
        }

        .form-label {
          display: block;
          margin-bottom: var(--space-2);
          font-weight: var(--font-medium);
          color: var(--color-text-primary);
        }

        .textarea {
          resize: vertical;
          min-height: 100px;
        }

        /* Personality Grid */
        .personality-grid {
          display: grid;
          gap: var(--space-6);
          margin-bottom: var(--space-8);
        }

        .trait-card {
          background: var(--color-surface);
          border: 1px solid var(--color-border-subtle);
          border-radius: var(--radius-xl);
          padding: var(--space-6);
          transition: all var(--duration-normal) var(--ease-out);
        }

        .trait-card:hover {
          border-color: var(--color-border);
          transform: translateY(-2px);
        }

        .trait-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-2);
        }

        .trait-name {
          font-size: var(--text-lg);
        }

        .trait-value {
          font-size: var(--text-2xl);
          font-weight: var(--font-bold);
        }

        .trait-description {
          margin-bottom: var(--space-4);
          font-size: var(--text-sm);
        }

        .slider-container {
          position: relative;
        }

        .personality-slider {
          width: 100%;
          height: 6px;
          -webkit-appearance: none;
          appearance: none;
          background: var(--color-border-subtle);
          border-radius: var(--radius-full);
          outline: none;
          margin: var(--space-4) 0;
        }

        .personality-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 20px;
          height: 20px;
          background: var(--slider-color, var(--color-primary));
          border-radius: var(--radius-full);
          cursor: pointer;
          transition: all var(--duration-fast) var(--ease-out);
        }

        .personality-slider::-webkit-slider-thumb:hover {
          transform: scale(1.2);
          box-shadow: 0 0 0 8px rgba(249, 115, 22, 0.1);
        }

        .personality-slider::-moz-range-thumb {
          width: 20px;
          height: 20px;
          background: var(--slider-color, var(--color-primary));
          border-radius: var(--radius-full);
          cursor: pointer;
          border: none;
          transition: all var(--duration-fast) var(--ease-out);
        }

        .slider-labels {
          display: flex;
          justify-content: space-between;
          font-size: var(--text-xs);
        }

        /* Radar Preview */
        .radar-preview {
          max-width: 300px;
          margin: 0 auto;
          padding: var(--space-6);
          text-align: center;
        }

        .radar-container {
          width: 200px;
          height: 200px;
          margin: var(--space-4) auto 0;
        }

        .radar-chart {
          width: 100%;
          height: 100%;
        }

        .radar-chart polygon.dimmed {
          opacity: 0.3;
        }

        .trait-label {
          transition: all var(--duration-fast) var(--ease-out);
        }

        .trait-label.highlighted {
          font-weight: var(--font-bold);
          font-size: 14px;
        }

        /* Navigation */
        .builder-navigation {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding-top: var(--space-8);
          border-top: 1px solid var(--color-border-subtle);
        }

        /* Responsive */
        @media (max-width: 768px) {
          .builder-header {
            flex-direction: column;
            gap: var(--space-4);
          }

          .personality-grid {
            grid-template-columns: 1fr;
          }

          .radar-preview {
            margin-top: var(--space-8);
          }
        }
      `}</style>
    </div>
  );
};

export default CharacterBuilder; 