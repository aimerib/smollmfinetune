import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import multimodalService, { 
  MultimodalJob, 
  MultimodalGenerationConfig,
  MultimodalProgressUpdate
} from '../services/multimodalService';
import '../styles/design-system.css';

interface Character {
  id: string;
  name: string;
  archetype: string;
  personality: Record<string, number>;
}

// Using interfaces from multimodal service
type GenerationConfig = MultimodalGenerationConfig;
type GenerationJob = MultimodalJob;

const MultimodalStudio: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'configure' | 'jobs' | 'analysis'>('configure');
  const [config, setConfig] = useState<GenerationConfig>({
    name: 'My Multimodal Dataset',
    sampleCount: 1000,
    characterCount: 10,
    narrativeTypes: ['dialogue', 'monologue', 'action_scene'],
    useMockTTS: true,
    ttsProvider: 'orpheus',
    outputDir: 'multimodal_dataset',
    batchSize: 10,
    temperature: 0.8
  });
  
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [progressSubscriptions, setProgressSubscriptions] = useState<Map<string, () => void>>(new Map());

  const [characters, setCharacters] = useState<Character[]>([]);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [selectedCharacters, setSelectedCharacters] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const narrativeTypeOptions = [
    { id: 'dialogue', label: 'Dialogue', description: 'Character conversations and interactions', icon: '💬' },
    { id: 'monologue', label: 'Monologue', description: 'Character inner thoughts and soliloquies', icon: '🎭' },
    { id: 'action_scene', label: 'Action Scenes', description: 'Dynamic, high-energy sequences', icon: '⚔️' },
    { id: 'emotional_moment', label: 'Emotional Moments', description: 'Deep emotional expressions', icon: '💖' },
    { id: 'memory_recall', label: 'Memory Recall', description: 'Characters remembering past events', icon: '🧠' },
    { id: 'world_description', label: 'World Description', description: 'Environmental and lore exposition', icon: '🌍' }
  ];

  const ttsProviders = [
    { id: 'orpheus', label: 'Orpheus-TTS', description: 'High-quality emotion-aware synthesis', icon: '🎵' },
    { id: 'xtts', label: 'XTTS', description: 'Voice cloning with few samples', icon: '🎤' },
    { id: 'bark', label: 'Bark', description: 'Expressive speech with effects', icon: '🗣️' }
  ];

  useEffect(() => {
    // TODO: Fetch actual characters from API
    setCharacters([
      { id: '1', name: 'Aria the Mystic', archetype: 'mentor', personality: { openness: 0.9, conscientiousness: 0.7 } },
      { id: '2', name: 'Zara Stormwind', archetype: 'hero', personality: { extraversion: 0.8, agreeableness: 0.6 } },
      { id: '3', name: 'Dr. Vex', archetype: 'villain', personality: { neuroticism: 0.3, openness: 0.9 } },
      { id: '4', name: 'Luna Nightwhisper', archetype: 'trickster', personality: { openness: 0.8, agreeableness: 0.4 } }
    ]);
  }, []);

  // Cleanup WebSocket connections on unmount
  useEffect(() => {
    return () => {
      console.log('🧹 Cleaning up WebSocket connections...');
      multimodalService.disconnectAll();
    };
  }, []);

  const handleStartGeneration = async () => {
    setIsGenerating(true);
    
    try {
      // Validate configuration
      const validationErrors = multimodalService.validateConfig(config);
      if (validationErrors.length > 0) {
        alert(`Configuration errors:\n${validationErrors.join('\n')}`);
        setIsGenerating(false);
        return;
      }

      console.log('🚀 Starting real multimodal generation!', config);
      
      // Start generation via API
      const job = await multimodalService.startGeneration(config);
      
      // Add job to list and switch to jobs tab
      setJobs(prev => [job, ...prev]);
      setActiveTab('jobs');
      
      // Subscribe to real-time progress updates
      const unsubscribe = multimodalService.subscribeToProgress(
        job.id,
        (update: MultimodalProgressUpdate) => {
          console.log('📊 Progress update:', update);
          
          // Update job in list
          setJobs(prev => prev.map(existingJob => 
            existingJob.id === job.id 
              ? {
                  ...existingJob,
                  status: update.status as any,
                  progress: update.progress,
                  currentStep: update.currentStep,
                  samplesGenerated: update.samplesGenerated,
                  outputPath: update.outputPath,
                  errorMessage: update.error
                }
              : existingJob
          ));
          
          // If generation is complete, stop generating state
          if (update.status === 'completed' || update.status === 'failed') {
            setIsGenerating(false);
          }
        },
        (error: string) => {
          console.error('❌ WebSocket error:', error);
          
          // Update job with error
          setJobs(prev => prev.map(existingJob => 
            existingJob.id === job.id 
              ? {
                  ...existingJob,
                  status: 'failed',
                  errorMessage: error,
                  currentStep: 'Connection failed'
                }
              : existingJob
          ));
          
          setIsGenerating(false);
        }
      );
      
      // Store unsubscribe function
      setProgressSubscriptions(prev => new Map(prev.set(job.id, unsubscribe)));
      
      console.log('✨ Generation started successfully! Job ID:', job.id);
      
    } catch (error) {
      console.error('💥 Failed to start generation:', error);
      alert(`Failed to start generation: ${error}`);
      setIsGenerating(false);
    }
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      console.log(`🛑 Canceling job ${jobId}`);
      await multimodalService.cancelJob(jobId);
      
      // Unsubscribe from progress updates
      const unsubscribe = progressSubscriptions.get(jobId);
      if (unsubscribe) {
        unsubscribe();
        setProgressSubscriptions(prev => {
          const newMap = new Map(prev);
          newMap.delete(jobId);
          return newMap;
        });
      }
      
      // Update job status locally
      setJobs(prev => prev.map(job => 
        job.id === jobId 
          ? { ...job, status: 'failed' as any, currentStep: 'Canceled by user' }
          : job
      ));
      
      console.log(`✅ Job ${jobId} canceled successfully`);
    } catch (error) {
      console.error(`❌ Failed to cancel job ${jobId}:`, error);
      alert(`Failed to cancel job: ${error}`);
    }
  };

  const handleDownloadDataset = async (jobId: string, jobName: string) => {
    try {
      console.log(`⬇️ Downloading dataset for job ${jobId}`);
      const blob = await multimodalService.downloadDataset(jobId);
      const filename = `${jobName}_dataset.zip`;
      multimodalService.downloadFile(blob, filename);
      console.log(`✅ Dataset downloaded as ${filename}`);
    } catch (error) {
      console.error(`❌ Failed to download dataset for job ${jobId}:`, error);
      alert(`Failed to download dataset: ${error}`);
    }
  };

  const renderConfigureTab = () => (
    <div className="configure-tab">
      <div className="config-grid">
        {/* Left Column - Main Configuration */}
        <div className="config-section">
          <h3 className="section-title">Generation Settings</h3>
          
          <div className="form-group">
            <label className="form-label">Dataset Name</label>
            <div className="input-with-description">
              <input
                type="text"
                className="form-input"
                value={config.name}
                onChange={(e) => setConfig(prev => ({ ...prev, name: e.target.value }))}
                placeholder="My Multimodal Dataset"
              />
              <p className="input-description text-secondary">
                A descriptive name for your multimodal dataset
              </p>
            </div>
          </div>
          
          <div className="form-group">
            <label className="form-label">Target Samples</label>
            <div className="input-with-description">
              <input
                type="number"
                className="form-input"
                value={config.sampleCount}
                onChange={(e) => setConfig(prev => ({ ...prev, sampleCount: parseInt(e.target.value) }))}
                min="100"
                max="10000"
                step="100"
              />
              <p className="input-description text-secondary">
                Number of multimodal samples to generate (text + audio + control tokens)
              </p>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Character Count</label>
            <div className="input-with-description">
              <input
                type="number"
                className="form-input"
                value={config.characterCount}
                onChange={(e) => setConfig(prev => ({ ...prev, characterCount: parseInt(e.target.value) }))}
                min="5"
                max="50"
              />
              <p className="input-description text-secondary">
                Number of diverse characters to generate for the dataset
              </p>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Narrative Types</label>
            <div className="narrative-types-grid">
              {narrativeTypeOptions.map(type => (
                <div
                  key={type.id}
                  className={`narrative-type-card ${config.narrativeTypes.includes(type.id) ? 'selected' : ''}`}
                  onClick={() => {
                    const newTypes = config.narrativeTypes.includes(type.id)
                      ? config.narrativeTypes.filter(t => t !== type.id)
                      : [...config.narrativeTypes, type.id];
                    setConfig(prev => ({ ...prev, narrativeTypes: newTypes }));
                  }}
                >
                  <div className="type-icon">{type.icon}</div>
                  <div className="type-content">
                    <h4 className="type-title">{type.label}</h4>
                    <p className="type-description text-secondary">{type.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column - TTS Configuration */}
        <div className="config-section">
          <h3 className="section-title">Voice Synthesis</h3>
          
          <div className="form-group">
            <label className="form-label">TTS Mode</label>
            <div className="tts-mode-selector">
              <div
                className={`mode-option ${config.useMockTTS ? 'selected' : ''}`}
                onClick={() => setConfig(prev => ({ ...prev, useMockTTS: true }))}
              >
                <div className="mode-icon">⚡</div>
                <div className="mode-content">
                  <h4>Mock TTS (Fast)</h4>
                  <p className="text-secondary">Generate placeholder audio for rapid prototyping</p>
                </div>
              </div>
              <div
                className={`mode-option ${!config.useMockTTS ? 'selected' : ''}`}
                onClick={() => setConfig(prev => ({ ...prev, useMockTTS: false }))}
              >
                <div className="mode-icon">🎯</div>
                <div className="mode-content">
                  <h4>Real TTS (Quality)</h4>
                  <p className="text-secondary">Generate actual speech using TTS models</p>
                </div>
              </div>
            </div>
          </div>

          {!config.useMockTTS && (
            <div className="form-group">
              <label className="form-label">TTS Provider</label>
              <div className="tts-providers">
                {ttsProviders.map(provider => (
                  <div
                    key={provider.id}
                    className={`provider-option ${config.ttsProvider === provider.id ? 'selected' : ''}`}
                    onClick={() => setConfig(prev => ({ ...prev, ttsProvider: provider.id as any }))}
                  >
                    <div className="provider-icon">{provider.icon}</div>
                    <div className="provider-content">
                      <h4>{provider.label}</h4>
                      <p className="text-secondary">{provider.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="form-group">
            <label className="form-label">Advanced Settings</label>
            <div className="advanced-settings">
              <div className="setting-row">
                <label>Batch Size</label>
                <input
                  type="number"
                  className="form-input small"
                  value={config.batchSize}
                  onChange={(e) => setConfig(prev => ({ ...prev, batchSize: parseInt(e.target.value) }))}
                  min="1"
                  max="50"
                />
              </div>
              <div className="setting-row">
                <label>Temperature</label>
                <input
                  type="number"
                  className="form-input small"
                  value={config.temperature}
                  onChange={(e) => setConfig(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                  min="0.1"
                  max="2.0"
                  step="0.1"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Generation Summary */}
      <div className="generation-summary card">
        <h3 className="summary-title">Generation Summary</h3>
        <div className="summary-stats">
          <div className="stat-item">
            <span className="stat-value">{config.sampleCount.toLocaleString()}</span>
            <span className="stat-label">Total Samples</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{config.characterCount}</span>
            <span className="stat-label">Characters</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{config.narrativeTypes.length}</span>
            <span className="stat-label">Narrative Types</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{Math.ceil(config.sampleCount / 100)}</span>
            <span className="stat-label">Est. Minutes</span>
          </div>
        </div>
        
        <button
          className="btn btn-primary btn-lg orange-glow generate-btn"
          onClick={handleStartGeneration}
          disabled={isGenerating || config.narrativeTypes.length === 0}
        >
          <span>🚀</span>
          {isGenerating ? 'Starting Generation...' : 'Start Generation'}
        </button>
      </div>
    </div>
  );

  const renderJobsTab = () => (
    <div className="jobs-tab">
      <div className="jobs-header">
        <h3 className="section-title">Generation Jobs</h3>
        <div className="jobs-stats">
          <span className="stat-badge active">
            {jobs.filter(j => j.status === 'generating').length} Active
          </span>
          <span className="stat-badge completed">
            {jobs.filter(j => j.status === 'completed').length} Completed
          </span>
        </div>
      </div>

      <div className="jobs-list">
        {jobs.map(job => (
          <div key={job.id} className={`job-card card ${job.status}`}>
            <div className="job-header">
              <div className="job-info">
                <h4 className="job-name">{job.name}</h4>
                <span className={`job-status ${job.status}`}>
                  {job.status === 'generating' && '🔄'}
                  {job.status === 'completed' && '✅'}
                  {job.status === 'failed' && '❌'}
                  {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                </span>
              </div>
              <div className="job-meta text-secondary">
                <span>{job.createdAt}</span>
                {job.estimatedTime && <span>• {job.estimatedTime} remaining</span>}
              </div>
            </div>

            {job.status === 'generating' && (
              <div className="job-progress">
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${job.progress}%` }}
                  />
                </div>
                <div className="progress-info">
                  <span className="progress-text">{job.progress.toFixed(1)}% complete</span>
                  <span className="progress-samples">
                    {job.samplesGenerated.toLocaleString()} / {job.config.sampleCount.toLocaleString()} samples
                  </span>
                </div>
                <div className="current-step text-secondary">
                  {job.currentStep}
                </div>
                <div className="job-actions">
                  <button 
                    className="btn btn-secondary btn-sm"
                    onClick={() => handleCancelJob(job.id)}
                  >
                    🛑 Cancel Generation
                  </button>
                </div>
              </div>
            )}

            {job.status === 'completed' && (
              <div className="job-results">
                <div className="result-stats">
                  <div className="result-stat">
                    <span className="stat-value">{job.samplesGenerated.toLocaleString()}</span>
                    <span className="stat-label">Samples Generated</span>
                  </div>
                  <div className="result-stat">
                    <span className="stat-value">{job.config.narrativeTypes.length}</span>
                    <span className="stat-label">Narrative Types</span>
                  </div>
                  <div className="result-stat">
                    <span className="stat-value">{job.config.useMockTTS ? 'Mock' : job.config.ttsProvider.toUpperCase()}</span>
                    <span className="stat-label">TTS Mode</span>
                  </div>
                </div>
                <div className="job-actions">
                  <button className="btn btn-secondary btn-sm">
                    📊 View Analysis
                  </button>
                  <button 
                    className="btn btn-primary btn-sm"
                    onClick={() => handleDownloadDataset(job.id, job.name)}
                  >
                    📥 Download Dataset
                  </button>
                </div>
              </div>
            )}

            {job.errorMessage && (
              <div className="job-error">
                <span className="error-icon">⚠️</span>
                <span className="error-message">{job.errorMessage}</span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  const renderAnalysisTab = () => (
    <div className="analysis-tab">
      <h3 className="section-title">Dataset Analysis</h3>
      <div className="analysis-placeholder card">
        <div className="placeholder-content">
          <span className="placeholder-icon">📊</span>
          <h4>Dataset Analysis Coming Soon</h4>
          <p className="text-secondary">
            Comprehensive analysis of generated datasets including quality metrics, 
            narrative type distribution, and character voice consistency.
          </p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="multimodal-studio">
      <div className="container">
        {/* Header */}
        <div className="studio-header">
          <button 
            className="btn btn-ghost"
            onClick={() => navigate('/creator')}
          >
            ← Back to Dashboard
          </button>

          <div className="header-info">
            <h1 className="page-title">Multimodal Studio</h1>
            <p className="page-subtitle text-secondary">
              Generate synchronized text, audio, and control token datasets for advanced character training
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="tab-nav">
          <button
            className={`tab-button ${activeTab === 'configure' ? 'active' : ''}`}
            onClick={() => setActiveTab('configure')}
          >
            <span>⚙️</span> Configure
          </button>
          <button
            className={`tab-button ${activeTab === 'jobs' ? 'active' : ''}`}
            onClick={() => setActiveTab('jobs')}
          >
            <span>🔄</span> Jobs
            {jobs.filter(j => j.status === 'generating').length > 0 && (
              <span className="tab-badge">{jobs.filter(j => j.status === 'generating').length}</span>
            )}
          </button>
          <button
            className={`tab-button ${activeTab === 'analysis' ? 'active' : ''}`}
            onClick={() => setActiveTab('analysis')}
          >
            <span>📊</span> Analysis
          </button>
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {activeTab === 'configure' && renderConfigureTab()}
          {activeTab === 'jobs' && renderJobsTab()}
          {activeTab === 'analysis' && renderAnalysisTab()}
        </div>
      </div>

      <style>{`
        .multimodal-studio {
          min-height: 100vh;
          background: var(--color-background);
          padding-bottom: var(--space-16);
        }

        .studio-header {
          display: flex;
          align-items: flex-start;
          gap: var(--space-4);
          margin-bottom: var(--space-8);
          padding: var(--space-8) 0;
        }

        .header-info h1 {
          margin-bottom: var(--space-2);
        }

        /* Tab Navigation */
        .tab-nav {
          display: flex;
          gap: var(--space-2);
          margin-bottom: var(--space-8);
          border-bottom: 1px solid var(--color-border);
          padding-bottom: var(--space-2);
        }

        .tab-button {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-3) var(--space-4);
          background: transparent;
          border: none;
          border-radius: var(--radius-md);
          color: var(--color-text-secondary);
          cursor: pointer;
          transition: all var(--duration-normal);
          position: relative;
        }

        .tab-button:hover {
          background: var(--color-surface-hover);
          color: var(--color-text-primary);
        }

        .tab-button.active {
          background: var(--color-primary);
          color: white;
        }

        .tab-badge {
          background: var(--color-warning);
          color: white;
          font-size: var(--text-xs);
          padding: 2px 6px;
          border-radius: var(--radius-full);
          margin-left: var(--space-1);
        }

        /* Configure Tab */
        .configure-tab {
          display: flex;
          flex-direction: column;
          gap: var(--space-8);
        }

        .config-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: var(--space-8);
        }

        .config-section {
          display: flex;
          flex-direction: column;
          gap: var(--space-6);
        }

        .section-title {
          font-size: var(--text-xl);
          margin-bottom: var(--space-4);
          color: var(--color-text-primary);
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }

        .form-label {
          font-weight: var(--font-medium);
          color: var(--color-text-primary);
        }

        .input-with-description {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
        }

        .input-description {
          font-size: var(--text-sm);
          margin: 0;
        }

        /* Narrative Types Grid */
        .narrative-types-grid {
          display: grid;
          grid-template-columns: 1fr;
          gap: var(--space-3);
        }

        .narrative-type-card {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          padding: var(--space-4);
          border: 2px solid var(--color-border);
          border-radius: var(--radius-lg);
          cursor: pointer;
          transition: all var(--duration-normal);
        }

        .narrative-type-card:hover {
          border-color: var(--color-primary-light);
          background: var(--color-surface-hover);
        }

        .narrative-type-card.selected {
          border-color: var(--color-primary);
          background: var(--color-primary-light);
        }

        .type-icon {
          font-size: var(--text-2xl);
          min-width: 40px;
          text-align: center;
        }

        .type-content {
          flex: 1;
        }

        .type-title {
          margin: 0 0 var(--space-1) 0;
          font-size: var(--text-base);
          font-weight: var(--font-medium);
        }

        .type-description {
          margin: 0;
          font-size: var(--text-sm);
        }

        /* TTS Configuration */
        .tts-mode-selector {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }

        .mode-option {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          padding: var(--space-4);
          border: 2px solid var(--color-border);
          border-radius: var(--radius-lg);
          cursor: pointer;
          transition: all var(--duration-normal);
        }

        .mode-option:hover {
          border-color: var(--color-primary-light);
          background: var(--color-surface-hover);
        }

        .mode-option.selected {
          border-color: var(--color-primary);
          background: var(--color-primary-light);
        }

        .mode-icon {
          font-size: var(--text-xl);
          min-width: 32px;
          text-align: center;
        }

        .mode-content h4 {
          margin: 0 0 var(--space-1) 0;
          font-size: var(--text-base);
          font-weight: var(--font-medium);
        }

        .mode-content p {
          margin: 0;
          font-size: var(--text-sm);
        }

        .tts-providers {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
        }

        .provider-option {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          padding: var(--space-3);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          cursor: pointer;
          transition: all var(--duration-normal);
        }

        .provider-option:hover {
          border-color: var(--color-primary-light);
          background: var(--color-surface-hover);
        }

        .provider-option.selected {
          border-color: var(--color-primary);
          background: var(--color-primary-light);
        }

        .provider-icon {
          font-size: var(--text-lg);
          min-width: 24px;
        }

        .provider-content h4 {
          margin: 0 0 2px 0;
          font-size: var(--text-sm);
          font-weight: var(--font-medium);
        }

        .provider-content p {
          margin: 0;
          font-size: var(--text-xs);
        }

        /* Advanced Settings */
        .advanced-settings {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }

        .setting-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .setting-row label {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
        }

        .form-input.small {
          width: 80px;
        }

        /* Generation Summary */
        .generation-summary {
          padding: var(--space-6);
          background: linear-gradient(135deg, var(--color-surface) 0%, var(--color-surface-hover) 100%);
          border: 1px solid var(--color-border);
        }

        .summary-title {
          margin-bottom: var(--space-4);
          text-align: center;
        }

        .summary-stats {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: var(--space-4);
          margin-bottom: var(--space-6);
        }

        .stat-item {
          text-align: center;
        }

        .stat-value {
          display: block;
          font-size: var(--text-2xl);
          font-weight: var(--font-bold);
          color: var(--color-primary);
          margin-bottom: var(--space-1);
        }

        .stat-label {
          font-size: var(--text-sm);
          color: var(--color-text-secondary);
        }

        .generate-btn {
          width: 100%;
          justify-content: center;
          gap: var(--space-2);
          font-size: var(--text-lg);
          padding: var(--space-4) var(--space-6);
        }

        /* Jobs Tab */
        .jobs-tab {
          display: flex;
          flex-direction: column;
          gap: var(--space-6);
        }

        .jobs-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .jobs-stats {
          display: flex;
          gap: var(--space-2);
        }

        .stat-badge {
          padding: var(--space-1) var(--space-3);
          border-radius: var(--radius-full);
          font-size: var(--text-sm);
          font-weight: var(--font-medium);
        }

        .stat-badge.active {
          background: var(--color-warning);
          color: white;
        }

        .stat-badge.completed {
          background: var(--color-success);
          color: white;
        }

        .jobs-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }

        .job-card {
          padding: var(--space-6);
          border-left: 4px solid var(--color-border);
        }

        .job-card.generating {
          border-left-color: var(--color-warning);
        }

        .job-card.completed {
          border-left-color: var(--color-success);
        }

        .job-card.failed {
          border-left-color: var(--color-error);
        }

        .job-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: var(--space-4);
        }

        .job-info {
          display: flex;
          align-items: center;
          gap: var(--space-3);
        }

        .job-name {
          margin: 0;
          font-size: var(--text-lg);
        }

        .job-status {
          padding: var(--space-1) var(--space-2);
          border-radius: var(--radius-md);
          font-size: var(--text-xs);
          font-weight: var(--font-medium);
          display: flex;
          align-items: center;
          gap: var(--space-1);
        }

        .job-status.generating {
          background: var(--color-warning-light);
          color: var(--color-warning-dark);
        }

        .job-status.completed {
          background: var(--color-success-light);
          color: var(--color-success-dark);
        }

        .job-status.failed {
          background: var(--color-error-light);
          color: var(--color-error-dark);
        }

        .job-meta {
          font-size: var(--text-sm);
        }

        /* Job Progress */
        .job-progress {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }

        .progress-bar {
          width: 100%;
          height: 8px;
          background: var(--color-surface);
          border-radius: var(--radius-full);
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--color-primary), var(--color-primary-light));
          transition: width var(--duration-normal);
        }

        .progress-info {
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: var(--text-sm);
        }

        .progress-text {
          font-weight: var(--font-medium);
        }

        .current-step {
          font-size: var(--text-sm);
          font-style: italic;
        }

        /* Job Results */
        .job-results {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }

        .result-stats {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: var(--space-4);
        }

        .result-stat {
          text-align: center;
          padding: var(--space-3);
          background: var(--color-surface);
          border-radius: var(--radius-md);
        }

        .result-stat .stat-value {
          font-size: var(--text-lg);
          color: var(--color-success);
        }

        .result-stat .stat-label {
          font-size: var(--text-xs);
        }

        .job-actions {
          display: flex;
          gap: var(--space-2);
        }

        .job-error {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-3);
          background: var(--color-error-light);
          border-radius: var(--radius-md);
          color: var(--color-error-dark);
          font-size: var(--text-sm);
        }

        /* Analysis Tab */
        .analysis-tab {
          display: flex;
          flex-direction: column;
          gap: var(--space-6);
        }

        .analysis-placeholder {
          padding: var(--space-12);
          text-align: center;
        }

        .placeholder-content {
          max-width: 400px;
          margin: 0 auto;
        }

        .placeholder-icon {
          font-size: 4rem;
          display: block;
          margin-bottom: var(--space-4);
        }

        .placeholder-content h4 {
          margin-bottom: var(--space-2);
          color: var(--color-text-primary);
        }

        /* Responsive */
        @media (max-width: 1024px) {
          .config-grid {
            grid-template-columns: 1fr;
          }

          .summary-stats {
            grid-template-columns: repeat(2, 1fr);
          }

          .result-stats {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 768px) {
          .studio-header {
            flex-direction: column;
            align-items: stretch;
          }

          .tab-nav {
            flex-wrap: wrap;
          }

          .summary-stats {
            grid-template-columns: 1fr;
          }

          .jobs-header {
            flex-direction: column;
            align-items: stretch;
            gap: var(--space-2);
          }

          .job-header {
            flex-direction: column;
            align-items: stretch;
            gap: var(--space-2);
          }

          .job-actions {
            flex-direction: column;
          }
        }
      `}</style>
    </div>
  );
};

export default MultimodalStudio;