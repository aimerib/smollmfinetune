import React, { useState, useEffect } from 'react';
import { useDiffusionTraining } from '../hooks/useDiffusionTraining';

interface DiffusionConfig {
  model_size: 'small' | 'medium' | 'large';
  num_characters: number;
  conversations_per_character: number;
  multimodal_ratio: number;
  batch_size: number;
  learning_rate: number;
  guidance_scale: number;
  num_inference_steps: number;
  max_steps: number;
  save_steps: number;
  eval_steps: number;
  use_ema: boolean;
  cross_modal_weight: number;
}

interface WizardStep {
  id: string;
  title: string;
  description: string;
  icon: string;
}

const DiffusionTrainingWizard: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [config, setConfig] = useState<DiffusionConfig>({
    model_size: 'small',
    num_characters: 10,
    conversations_per_character: 50,
    multimodal_ratio: 0.3,
    batch_size: 4,
    learning_rate: 1e-4,
    guidance_scale: 7.5,
    num_inference_steps: 50,
    max_steps: 5000,
    save_steps: 500,
    eval_steps: 100,
    use_ema: true,
    cross_modal_weight: 0.3
  });

  const {
    trainingStatus,
    trainingMetrics,
    connectionStatus,
    liveUpdates,
    startTraining,
    pauseTraining,
    resumeTraining,
    stopTraining,
    generateDataset,
    validateModel,
    connectWebSocket,
    disconnectWebSocket,
    clearLiveUpdates
  } = useDiffusionTraining();

  const steps: WizardStep[] = [
    {
      id: 'configure',
      title: 'Model Configuration',
      description: 'Select model size and core parameters',
      icon: '⚙️'
    },
    {
      id: 'dataset',
      title: 'Dataset Generation',
      description: 'Generate multimodal training data',
      icon: '🧠'
    },
    {
      id: 'validation',
      title: 'Model Validation',
      description: 'Test architecture and fix issues',
      icon: '✅'
    },
    {
      id: 'training',
      title: 'Incremental Training',
      description: 'Small → Medium → Large scale training',
      icon: '⚡'
    },
    {
      id: 'production',
      title: 'Production Training',
      description: 'Final large-scale training run',
      icon: '🚀'
    }
  ];

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const getConnectionStatusBadge = () => {
    switch (connectionStatus) {
      case 'connected':
        return (
          <span className="px-2 py-1 bg-green-100 text-green-800 border border-green-200 rounded-full text-sm">
            🔗 Live Updates
          </span>
        );
      case 'connecting':
        return (
          <span className="px-2 py-1 bg-yellow-100 text-yellow-800 border border-yellow-200 rounded-full text-sm">
            🔄 Connecting...
          </span>
        );
      case 'disconnected':
        return (
          <span className="px-2 py-1 bg-gray-100 text-gray-600 border border-gray-300 rounded-full text-sm">
            📴 Offline
          </span>
        );
    }
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 0:
        return <ConfigurationStep config={config} setConfig={setConfig} />;
      case 1:
        return <DatasetStep config={config} generateDataset={generateDataset} />;
      case 2:
        return <ValidationStep validateModel={validateModel} />;
      case 3:
        return <TrainingStep 
          config={config} 
          trainingStatus={trainingStatus} 
          trainingMetrics={trainingMetrics}
          startTraining={startTraining}
          pauseTraining={pauseTraining}
          resumeTraining={resumeTraining}
          stopTraining={stopTraining}
        />;
      case 4:
        return <ProductionStep 
          config={config} 
          trainingStatus={trainingStatus} 
          trainingMetrics={trainingMetrics}
          startTraining={startTraining}
          pauseTraining={pauseTraining}
          resumeTraining={resumeTraining}
          stopTraining={stopTraining}
        />;
      default:
        return null;
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Progress Indicator */}
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-2xl font-bold">Diffusion Training Wizard</h2>
          <div className="flex items-center gap-3">
            {getConnectionStatusBadge()}
            {connectionStatus === 'disconnected' && trainingStatus.status === 'training' && (
              <button 
                className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
                onClick={connectWebSocket}
              >
                🔗 Reconnect
              </button>
            )}
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          {steps.map((step, index) => (
            <div
              key={step.id}
              className={`flex items-center space-x-2 ${
                index <= currentStep ? 'text-blue-600' : 'text-gray-400'
              }`}
            >
              <div
                className={`flex items-center justify-center w-8 h-8 rounded-full border-2 ${
                  index < currentStep
                    ? 'bg-blue-600 border-blue-600 text-white'
                    : index === currentStep
                    ? 'border-blue-600 text-blue-600'
                    : 'border-gray-300 text-gray-400'
                }`}
              >
                {index < currentStep ? (
                  <span>✓</span>
                ) : (
                  <span className="text-sm font-medium">{index + 1}</span>
                )}
              </div>
              <span className="text-sm font-medium hidden md:block">{step.title}</span>
              {index < steps.length - 1 && (
                <div className="w-8 h-0.5 bg-gray-300 hidden md:block" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Current Step */}
        <div className="lg:col-span-2">
          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-6">
            <div className="flex items-center space-x-3 mb-4">
              <span className="text-2xl">{steps[currentStep].icon}</span>
              <div>
                <h3 className="text-lg font-semibold">{steps[currentStep].title}</h3>
                <p className="text-sm text-gray-600">{steps[currentStep].description}</p>
              </div>
            </div>
            
            {renderCurrentStep()}
            
            {/* Navigation */}
            <div className="flex justify-between mt-6">
              <button
                onClick={prevStep}
                disabled={currentStep === 0}
                className={`px-4 py-2 rounded ${
                  currentStep === 0 
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Previous
              </button>
              <button
                onClick={nextStep}
                disabled={currentStep === steps.length - 1}
                className={`px-4 py-2 rounded ${
                  currentStep === steps.length - 1 
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                    : 'bg-blue-500 text-white hover:bg-blue-600'
                }`}
              >
                Next
              </button>
            </div>
          </div>
        </div>

        {/* Right Column - Live Updates */}
        <div className="space-y-4">
          <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-4">
            <h4 className="font-semibold mb-3">🔄 Live Updates</h4>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {liveUpdates && liveUpdates.length > 0 ? (
                liveUpdates.slice(-10).map((update, index) => (
                  <div key={index} className="text-sm p-2 bg-gray-50 rounded">
                    <span className="text-gray-500 text-xs">
                      {new Date(update.timestamp).toLocaleTimeString()}
                    </span>
                    <p className="mt-1">{update.message}</p>
                  </div>
                ))
              ) : (
                <p className="text-gray-500 text-sm">No live updates available</p>
              )}
            </div>
          </div>

          {/* Training Status */}
          {trainingStatus.status !== 'idle' && (
            <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-4">
              <h4 className="font-semibold mb-3">📊 Training Status</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm">Status:</span>
                  <span className={`text-sm font-medium ${
                    trainingStatus.status === 'training' ? 'text-green-600' : 'text-gray-600'
                  }`}>
                    {trainingStatus.status}
                  </span>
                </div>
                {trainingStatus.progress !== undefined && (
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-sm">Progress:</span>
                      <span className="text-sm font-medium">{trainingStatus.progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${trainingStatus.progress}%` }}
                      />
                    </div>
                  </div>
                )}
                {trainingStatus.eta && (
                  <div className="flex justify-between">
                    <span className="text-sm">ETA:</span>
                    <span className="text-sm font-medium">{trainingStatus.eta}</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Simplified step components
const ConfigurationStep: React.FC<any> = ({ config, setConfig }) => {
  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">Model Size</label>
        <select 
          className="w-full p-2 border border-gray-300 rounded"
          value={config.model_size}
          onChange={(e) => setConfig({...config, model_size: e.target.value})}
        >
          <option value="small">Small (Fast, Good for testing)</option>
          <option value="medium">Medium (Balanced)</option>
          <option value="large">Large (Best quality)</option>
        </select>
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Number of Characters</label>
        <input 
          type="number" 
          className="w-full p-2 border border-gray-300 rounded"
          value={config.num_characters}
          onChange={(e) => setConfig({...config, num_characters: parseInt(e.target.value)})}
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Learning Rate</label>
        <input 
          type="number" 
          step="0.00001"
          className="w-full p-2 border border-gray-300 rounded"
          value={config.learning_rate}
          onChange={(e) => setConfig({...config, learning_rate: parseFloat(e.target.value)})}
        />
      </div>
    </div>
  );
};

const DatasetStep: React.FC<any> = ({ config, generateDataset }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  
  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      await generateDataset(config);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Generate multimodal training data for your characters.
      </p>
      
      <div className="bg-blue-50 border border-blue-200 rounded p-4">
        <h4 className="font-medium mb-2">Dataset Configuration</h4>
        <p className="text-sm text-blue-700">
          • {config.num_characters} characters
          • {config.conversations_per_character} conversations per character
          • {Math.round(config.multimodal_ratio * 100)}% multimodal content
        </p>
      </div>
      
      <button
        onClick={handleGenerate}
        disabled={isGenerating}
        className={`w-full py-2 px-4 rounded ${
          isGenerating 
            ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
            : 'bg-green-500 text-white hover:bg-green-600'
        }`}
      >
        {isGenerating ? 'Generating...' : 'Generate Dataset'}
      </button>
    </div>
  );
};

const ValidationStep: React.FC<any> = ({ validateModel }) => {
  const [isValidating, setIsValidating] = useState(false);
  
  const handleValidate = async () => {
    setIsValidating(true);
    try {
      await validateModel();
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Validate your model architecture and configuration.
      </p>
      
      <button
        onClick={handleValidate}
        disabled={isValidating}
        className={`w-full py-2 px-4 rounded ${
          isValidating 
            ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
            : 'bg-blue-500 text-white hover:bg-blue-600'
        }`}
      >
        {isValidating ? 'Validating...' : 'Validate Model'}
      </button>
    </div>
  );
};

const TrainingStep: React.FC<any> = ({ 
  config, 
  trainingStatus, 
  trainingMetrics,
  startTraining, 
  pauseTraining, 
  resumeTraining, 
  stopTraining 
}) => {
  const handleStartTraining = () => {
    startTraining(config);
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Start incremental training with your configuration.
      </p>
      
      <div className="flex gap-2">
        {trainingStatus.status === 'idle' && (
          <button
            onClick={handleStartTraining}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            Start Training
          </button>
        )}
        
        {trainingStatus.status === 'training' && (
          <>
            <button
              onClick={pauseTraining}
              className="px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600"
            >
              Pause
            </button>
            <button
              onClick={stopTraining}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
            >
              Stop
            </button>
          </>
        )}
        
        {trainingStatus.status === 'paused' && (
          <>
            <button
              onClick={resumeTraining}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
            >
              Resume
            </button>
            <button
              onClick={stopTraining}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
            >
              Stop
            </button>
          </>
        )}
      </div>
    </div>
  );
};

const ProductionStep: React.FC<any> = ({ 
  config, 
  trainingStatus, 
  trainingMetrics,
  startTraining, 
  pauseTraining, 
  resumeTraining, 
  stopTraining 
}) => {
  const handleStartProduction = () => {
    startTraining({ ...config, model_size: 'large' });
  };

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">
        Final production-scale training with the large model.
      </p>
      
      <div className="bg-orange-50 border border-orange-200 rounded p-4">
        <h4 className="font-medium mb-2">⚠️ Production Training</h4>
        <p className="text-sm text-orange-700">
          This will start a large-scale training run that may take several hours.
        </p>
      </div>
      
      <div className="flex gap-2">
        {trainingStatus.status === 'idle' && (
          <button
            onClick={handleStartProduction}
            className="px-4 py-2 bg-orange-500 text-white rounded hover:bg-orange-600"
          >
            Start Production Training
          </button>
        )}
        
        {trainingStatus.status === 'training' && (
          <>
            <button
              onClick={pauseTraining}
              className="px-4 py-2 bg-yellow-500 text-white rounded hover:bg-yellow-600"
            >
              Pause
            </button>
            <button
              onClick={stopTraining}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
            >
              Stop
            </button>
          </>
        )}
        
        {trainingStatus.status === 'paused' && (
          <>
            <button
              onClick={resumeTraining}
              className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
            >
              Resume
            </button>
            <button
              onClick={stopTraining}
              className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
            >
              Stop
            </button>
          </>
        )}
      </div>
    </div>
  );
};

export default DiffusionTrainingWizard; 