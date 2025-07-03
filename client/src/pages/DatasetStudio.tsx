import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import '../styles/design-system.css';

interface Character {
  id: string;
  name: string;
  world: string;
  isTrained: boolean;
}

interface GenerationConfig {
  characterId: string;
  numConversations: number;
  conversationLength: {
    min: number;
    max: number;
  };
  topics: string[];
  style: 'casual' | 'formal' | 'mixed';
  includeActions: boolean;
  includeThoughts: boolean;
  temperature: number;
}

interface GenerationJob {
  id: string;
  characterName: string;
  status: 'pending' | 'generating' | 'completed' | 'failed';
  progress: number;
  conversations: number;
  messages: number;
  startedAt: string;
}

interface Dataset {
  id: string;
  name: string;
  conversation_count: number;
  status: 'pending' | 'generating' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
}

interface GenerationParams {
  target_samples: number;
  temperature: number;
  batch_size: number;
  quality_mode: 'fast' | 'balanced' | 'iterative';
  topics?: string[];
}

interface GenerationProgress {
  current: number;
  total: number;
  percentage: number;
  current_topic?: string;
}

const DatasetStudio: React.FC = () => {
  const { characterId } = useParams();
  const navigate = useNavigate();
  const wsRef = useRef<WebSocket | null>(null);
  const [activeView, setActiveView] = useState<'configure' | 'jobs'>('configure');
  const [characters, setCharacters] = useState<Character[]>([]);
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [config, setConfig] = useState<GenerationConfig>({
    characterId: '',
    numConversations: 100,
    conversationLength: {
      min: 5,
      max: 15
    },
    topics: [],
    style: 'mixed',
    includeActions: true,
    includeThoughts: true,
    temperature: 0.8
  });
  const [newTopic, setNewTopic] = useState('');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState<GenerationProgress>({
    current: 0,
    total: 0,
    percentage: 0
  });
  const [params, setParams] = useState<GenerationParams>({
    target_samples: 100,
    temperature: 0.8,
    batch_size: 10,
    quality_mode: 'iterative',
    topics: []
  });

  useEffect(() => {
    if (characterId) {
      fetchDatasets();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [characterId]);

  const fetchDatasets = async () => {
    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch(`/api/v1/datasets/character/${characterId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setDatasets(data);
      }
    } catch (error) {
      console.error('Failed to fetch datasets:', error);
    }
  };

  const createDataset = async () => {
    const name = prompt('Enter dataset name:');
    if (!name) return;

    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch('/api/v1/datasets/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          character_id: characterId,
          name,
          description: `Training dataset for character`,
          generation_params: params
        })
      });

      if (response.ok) {
        const dataset = await response.json();
        setDatasets([dataset, ...datasets]);
        setSelectedDataset(dataset.id);
      }
    } catch (error) {
      console.error('Failed to create dataset:', error);
    }
  };

  const startGeneration = async () => {
    if (!selectedDataset) return;

    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch(`/api/v1/datasets/${selectedDataset}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(params)
      });

      if (response.ok) {
        setIsGenerating(true);
        connectWebSocket();
        fetchDatasets(); // Refresh to show updated status
      }
    } catch (error) {
      console.error('Failed to start generation:', error);
    }
  };

  const stopGeneration = async () => {
    if (!selectedDataset) return;

    try {
      const token = localStorage.getItem('access_token');
      await fetch(`/api/v1/datasets/${selectedDataset}/stop`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      setIsGenerating(false);
      if (wsRef.current) {
        wsRef.current.close();
      }
      fetchDatasets();
    } catch (error) {
      console.error('Failed to stop generation:', error);
    }
  };

  const connectWebSocket = () => {
    if (!selectedDataset) return;

    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
    const ws = new WebSocket(`${wsUrl}/api/v1/datasets/ws/${selectedDataset}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
        setProgress({
          current: data.current,
          total: data.total,
          percentage: data.percentage,
          current_topic: data.current_topic
        });
      } else if (data.type === 'status') {
        if (data.status === 'completed' || data.status === 'cancelled' || data.status === 'failed') {
          setIsGenerating(false);
          fetchDatasets();
        }
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    wsRef.current = ws;
  };

  const addTopic = () => {
    if (newTopic && !params.topics?.includes(newTopic)) {
      setParams({
        ...params,
        topics: [...(params.topics || []), newTopic]
      });
      setNewTopic('');
    }
  };

  const removeTopic = (topic: string) => {
    setParams({
      ...params,
      topics: params.topics?.filter(t => t !== topic) || []
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'var(--color-success)';
      case 'generating': return 'var(--color-warning)';
      case 'failed': return 'var(--color-error)';
      default: return 'var(--color-text-secondary)';
    }
  };

  const renderConfigureView = () => (
    <div className="configure-view animate-slide-up">
      <div className="config-grid">
        {/* Character Selection */}
        <div className="config-section card">
          <h3 className="section-title">Select Character</h3>
          <div className="character-select-grid">
            {characters.map((character) => (
              <div
                key={character.id}
                className={`character-option ${config.characterId === character.id ? 'selected' : ''}`}
                onClick={() => setConfig({ ...config, characterId: character.id })}
              >
                <div className="character-emoji">🎭</div>
                <div className="character-info">
                  <div className="character-name">{character.name}</div>
                  <div className="character-world text-secondary">{character.world}</div>
                </div>
                {character.isTrained && (
                  <span className="trained-badge">Trained</span>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Generation Parameters */}
        <div className="config-section card">
          <h3 className="section-title">Generation Parameters</h3>
          
          <div className="param-group">
            <label className="param-label">Number of Conversations</label>
            <input
              type="number"
              className="input"
              min="10"
              max="1000"
              value={config.numConversations}
              onChange={(e) => setConfig({ ...config, numConversations: parseInt(e.target.value) || 100 })}
            />
          </div>

          <div className="param-group">
            <label className="param-label">Conversation Length</label>
            <div className="range-inputs">
              <input
                type="number"
                className="input range-input"
                placeholder="Min"
                min="1"
                max="50"
                value={config.conversationLength.min}
                onChange={(e) => setConfig({
                  ...config,
                  conversationLength: {
                    ...config.conversationLength,
                    min: parseInt(e.target.value) || 5
                  }
                })}
              />
              <span className="range-separator">to</span>
              <input
                type="number"
                className="input range-input"
                placeholder="Max"
                min="1"
                max="50"
                value={config.conversationLength.max}
                onChange={(e) => setConfig({
                  ...config,
                  conversationLength: {
                    ...config.conversationLength,
                    max: parseInt(e.target.value) || 15
                  }
                })}
              />
              <span className="range-unit">messages</span>
            </div>
          </div>

          <div className="param-group">
            <label className="param-label">Conversation Style</label>
            <div className="style-options">
              {(['casual', 'formal', 'mixed'] as const).map((style) => (
                <button
                  key={style}
                  className={`style-option ${config.style === style ? 'active' : ''}`}
                  onClick={() => setConfig({ ...config, style })}
                >
                  {style.charAt(0).toUpperCase() + style.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div className="param-group">
            <label className="param-label">
              Temperature
              <span className="param-value">{config.temperature}</span>
            </label>
            <input
              type="range"
              className="temperature-slider"
              min="0.1"
              max="1.5"
              step="0.1"
              value={config.temperature}
              onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
            />
            <div className="slider-labels">
              <span className="text-muted">Conservative</span>
              <span className="text-muted">Creative</span>
            </div>
          </div>
        </div>

        {/* Topics */}
        <div className="config-section card">
          <h3 className="section-title">Conversation Topics</h3>
          <p className="section-description text-secondary">
            Add topics to guide the conversation generation
          </p>
          
          <div className="topic-input-group">
            <input
              type="text"
              className="input"
              placeholder="Add a topic..."
              value={newTopic}
              onChange={(e) => setNewTopic(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && addTopic()}
            />
            <button className="btn btn-primary" onClick={addTopic}>
              Add
            </button>
          </div>

          <div className="topics-list">
            {config.topics.map((topic, index) => (
              <div key={index} className="topic-tag">
                <span>{topic}</span>
                <button 
                  className="topic-remove"
                  onClick={() => removeTopic(topic)}
                >
                  ✕
                </button>
              </div>
            ))}
            {config.topics.length === 0 && (
              <p className="no-topics text-muted">No topics added. Generation will use random topics.</p>
            )}
          </div>
        </div>

        {/* Features */}
        <div className="config-section card">
          <h3 className="section-title">Additional Features</h3>
          
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={config.includeActions}
              onChange={(e) => setConfig({ ...config, includeActions: e.target.checked })}
            />
            <span className="checkbox-text">Include Actions</span>
            <span className="checkbox-description text-muted">*performs action* markers</span>
          </label>

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={config.includeThoughts}
              onChange={(e) => setConfig({ ...config, includeThoughts: e.target.checked })}
            />
            <span className="checkbox-text">Include Thoughts</span>
            <span className="checkbox-description text-muted">Internal monologue markers</span>
          </label>
        </div>
      </div>

      {/* Generate Button */}
      <div className="generate-section">
        <button 
          className="btn btn-primary btn-lg orange-glow"
          onClick={startGeneration}
          disabled={!config.characterId}
        >
          <span>🚀</span>
          Start Generation
        </button>
        <p className="estimate text-secondary">
          Estimated time: ~{Math.ceil(config.numConversations / 20)} minutes
        </p>
      </div>
    </div>
  );

  const renderJobsView = () => (
    <div className="jobs-view animate-slide-up">
      <div className="jobs-grid">
        {jobs.map((job) => (
          <div key={job.id} className="job-card card">
            <div className="job-header">
              <h4 className="job-character">{job.characterName}</h4>
              <span 
                className="job-status"
                style={{ color: getStatusColor(job.status) }}
              >
                {job.status}
              </span>
            </div>

            {job.status === 'generating' && (
              <div className="progress-section">
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${job.progress}%` }}
                  />
                </div>
                <div className="progress-text">
                  {job.progress}% - {job.conversations}/{config.numConversations} conversations
                </div>
              </div>
            )}

            <div className="job-stats">
              <div className="stat">
                <span className="stat-label">Conversations</span>
                <span className="stat-value">{job.conversations}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Messages</span>
                <span className="stat-value">{job.messages}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Started</span>
                <span className="stat-value">
                  {new Date(job.startedAt).toLocaleTimeString()}
                </span>
              </div>
            </div>

            {job.status === 'completed' && (
              <button className="btn btn-secondary btn-sm">
                <span>💾</span>
                Download Dataset
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="dataset-studio">
      <div className="container">
        {/* Header */}
        <div className="studio-header">
          <button 
            className="btn btn-ghost"
            onClick={() => navigate('/creator')}
          >
            ← Back to Dashboard
          </button>

          <div className="view-toggle">
            <button
              className={`toggle-btn ${activeView === 'configure' ? 'active' : ''}`}
              onClick={() => setActiveView('configure')}
            >
              Configure
            </button>
            <button
              className={`toggle-btn ${activeView === 'jobs' ? 'active' : ''}`}
              onClick={() => setActiveView('jobs')}
            >
              Jobs ({jobs.filter(j => j.status === 'generating').length})
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="studio-content">
          <h1 className="page-title">Dataset Studio</h1>
          <p className="page-subtitle text-secondary">
            Generate synthetic conversations to train your characters
          </p>

          {activeView === 'configure' ? renderConfigureView() : renderJobsView()}
        </div>
      </div>

      <style>{`
        .dataset-studio {
          min-height: 100vh;
          padding: var(--space-8) 0;
        }

        .studio-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-8);
        }

        .view-toggle {
          display: flex;
          background: var(--color-surface);
          border-radius: var(--radius-lg);
          padding: var(--space-1);
        }

        .toggle-btn {
          padding: var(--space-2) var(--space-4);
          background: transparent;
          border: none;
          color: var(--color-text-secondary);
          font-size: var(--text-sm);
          font-weight: var(--font-medium);
          border-radius: var(--radius-md);
          cursor: pointer;
          transition: all var(--duration-fast) var(--ease-out);
        }

        .toggle-btn.active {
          background: var(--color-surface-elevated);
          color: var(--color-text-primary);
        }

        .page-title {
          margin-bottom: var(--space-2);
        }

        .page-subtitle {
          margin-bottom: var(--space-8);
        }

        /* Configure View */
        .config-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
          gap: var(--space-6);
          margin-bottom: var(--space-8);
        }

        .config-section {
          padding: var(--space-6);
        }

        .section-title {
          margin-bottom: var(--space-4);
        }

        .section-description {
          margin-bottom: var(--space-4);
          font-size: var(--text-sm);
        }

        /* Character Selection */
        .character-select-grid {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }

        .character-option {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          padding: var(--space-3);
          background: var(--color-surface-elevated);
          border: 2px solid transparent;
          border-radius: var(--radius-lg);
          cursor: pointer;
          transition: all var(--duration-fast) var(--ease-out);
        }

        .character-option:hover {
          border-color: var(--color-border);
        }

        .character-option.selected {
          border-color: var(--color-primary);
          background: rgba(249, 115, 22, 0.1);
        }

        .character-emoji {
          font-size: var(--text-2xl);
        }

        .character-info {
          flex: 1;
        }

        .character-name {
          font-weight: var(--font-medium);
        }

        .character-world {
          font-size: var(--text-sm);
        }

        .trained-badge {
          padding: var(--space-1) var(--space-2);
          background: var(--color-success);
          color: white;
          font-size: var(--text-xs);
          font-weight: var(--font-medium);
          border-radius: var(--radius-full);
        }

        /* Parameters */
        .param-group {
          margin-bottom: var(--space-4);
        }

        .param-label {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-2);
          font-weight: var(--font-medium);
        }

        .param-value {
          color: var(--color-primary);
          font-weight: var(--font-bold);
        }

        .range-inputs {
          display: flex;
          align-items: center;
          gap: var(--space-2);
        }

        .range-input {
          width: 80px;
        }

        .range-separator {
          color: var(--color-text-secondary);
        }

        .range-unit {
          color: var(--color-text-secondary);
          font-size: var(--text-sm);
        }

        /* Style Options */
        .style-options {
          display: flex;
          gap: var(--space-2);
        }

        .style-option {
          flex: 1;
          padding: var(--space-2) var(--space-3);
          background: var(--color-surface-elevated);
          border: 1px solid var(--color-border-subtle);
          border-radius: var(--radius-md);
          color: var(--color-text-secondary);
          font-size: var(--text-sm);
          cursor: pointer;
          transition: all var(--duration-fast) var(--ease-out);
        }

        .style-option:hover {
          border-color: var(--color-border);
        }

        .style-option.active {
          background: var(--color-primary);
          border-color: var(--color-primary);
          color: white;
        }

        /* Temperature Slider */
        .temperature-slider {
          width: 100%;
          height: 6px;
          -webkit-appearance: none;
          appearance: none;
          background: var(--color-border-subtle);
          border-radius: var(--radius-full);
          outline: none;
          margin: var(--space-4) 0;
        }

        .temperature-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 20px;
          height: 20px;
          background: var(--color-primary);
          border-radius: var(--radius-full);
          cursor: pointer;
        }

        .slider-labels {
          display: flex;
          justify-content: space-between;
          font-size: var(--text-xs);
        }

        /* Topics */
        .topic-input-group {
          display: flex;
          gap: var(--space-2);
          margin-bottom: var(--space-4);
        }

        .topics-list {
          display: flex;
          flex-wrap: wrap;
          gap: var(--space-2);
        }

        .topic-tag {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-1) var(--space-3);
          background: var(--color-surface-elevated);
          border: 1px solid var(--color-border-subtle);
          border-radius: var(--radius-full);
          font-size: var(--text-sm);
        }

        .topic-remove {
          background: none;
          border: none;
          color: var(--color-text-secondary);
          cursor: pointer;
          padding: 0;
          line-height: 1;
        }

        .topic-remove:hover {
          color: var(--color-error);
        }

        .no-topics {
          font-size: var(--text-sm);
          font-style: italic;
        }

        /* Checkboxes */
        .checkbox-label {
          display: flex;
          align-items: start;
          gap: var(--space-3);
          margin-bottom: var(--space-3);
          cursor: pointer;
        }

        .checkbox-label input[type="checkbox"] {
          margin-top: 2px;
        }

        .checkbox-text {
          font-weight: var(--font-medium);
        }

        .checkbox-description {
          font-size: var(--text-sm);
          margin-left: auto;
        }

        /* Generate Section */
        .generate-section {
          text-align: center;
          padding: var(--space-8);
          border-top: 1px solid var(--color-border-subtle);
        }

        .estimate {
          margin-top: var(--space-2);
          font-size: var(--text-sm);
        }

        /* Jobs View */
        .jobs-grid {
          display: grid;
          gap: var(--space-6);
        }

        .job-card {
          padding: var(--space-6);
        }

        .job-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-4);
        }

        .job-character {
          font-size: var(--text-lg);
        }

        .job-status {
          font-size: var(--text-sm);
          font-weight: var(--font-medium);
          text-transform: uppercase;
        }

        /* Progress */
        .progress-section {
          margin-bottom: var(--space-4);
        }

        .progress-bar {
          height: 8px;
          background: var(--color-border-subtle);
          border-radius: var(--radius-full);
          overflow: hidden;
          margin-bottom: var(--space-2);
        }

        .progress-fill {
          height: 100%;
          background: var(--color-primary);
          transition: width var(--duration-normal) var(--ease-out);
        }

        .progress-text {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
        }

        /* Job Stats */
        .job-stats {
          display: flex;
          gap: var(--space-6);
          margin-bottom: var(--space-4);
        }

        .stat {
          display: flex;
          flex-direction: column;
          gap: var(--space-1);
        }

        .stat-label {
          font-size: var(--text-xs);
          color: var(--color-text-secondary);
          text-transform: uppercase;
        }

        .stat-value {
          font-size: var(--text-lg);
          font-weight: var(--font-semibold);
        }

        /* Responsive */
        @media (max-width: 768px) {
          .config-grid {
            grid-template-columns: 1fr;
          }

          .studio-header {
            flex-direction: column;
            gap: var(--space-4);
            align-items: stretch;
          }

          .job-stats {
            gap: var(--space-4);
          }
        }
      `}</style>
    </div>
  );
};

export default DatasetStudio; 