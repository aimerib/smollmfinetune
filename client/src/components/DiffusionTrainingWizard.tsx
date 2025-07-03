import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, CheckCircle, Play, Pause, Square, Settings, Zap, Brain, Rocket, Wifi, WifiOff, Activity, Clock } from 'lucide-react';
import { useDiffusionTraining } from '@/hooks/useDiffusionTraining';

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
  icon: React.ReactNode;
  component: React.ComponentType<any>;
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
      icon: <Settings className="w-5 h-5" />,
      component: ConfigurationStep
    },
    {
      id: 'dataset',
      title: 'Dataset Generation',
      description: 'Generate multimodal training data',
      icon: <Brain className="w-5 h-5" />,
      component: DatasetStep
    },
    {
      id: 'validation',
      title: 'Model Validation',
      description: 'Test architecture and fix issues',
      icon: <CheckCircle className="w-5 h-5" />,
      component: ValidationStep
    },
    {
      id: 'training',
      title: 'Incremental Training',
      description: 'Small → Medium → Large scale training',
      icon: <Zap className="w-5 h-5" />,
      component: TrainingStep
    },
    {
      id: 'production',
      title: 'Production Training',
      description: 'Final large-scale training run',
      icon: <Rocket className="w-5 h-5" />,
      component: ProductionStep
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

  const currentStepComponent = steps[currentStep].component;

  // Connection status indicator
  const getConnectionStatusBadge = () => {
    switch (connectionStatus) {
      case 'connected':
        return (
          <Badge variant="default" className="bg-green-100 text-green-800 border-green-200">
            <Wifi className="w-3 h-3 mr-1" />
            Live Updates
          </Badge>
        );
      case 'connecting':
        return (
          <Badge variant="secondary" className="bg-yellow-100 text-yellow-800 border-yellow-200">
            <Activity className="w-3 h-3 mr-1 animate-pulse" />
            Connecting...
          </Badge>
        );
      case 'disconnected':
        return (
          <Badge variant="outline" className="bg-gray-100 text-gray-600 border-gray-300">
            <WifiOff className="w-3 h-3 mr-1" />
            Offline
          </Badge>
        );
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
              <Button size="sm" variant="outline" onClick={connectWebSocket}>
                <Wifi className="w-4 h-4 mr-1" />
                Reconnect
              </Button>
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
                  <CheckCircle className="w-5 h-5" />
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
        {/* Main Wizard Panel */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {steps[currentStep].icon}
                {steps[currentStep].title}
              </CardTitle>
              <p className="text-gray-600">{steps[currentStep].description}</p>
            </CardHeader>
            <CardContent>
              <currentStepComponent 
                config={config}
                setConfig={setConfig}
                trainingStatus={trainingStatus}
                trainingMetrics={trainingMetrics}
                startTraining={startTraining}
                pauseTraining={pauseTraining}
                resumeTraining={resumeTraining}
                stopTraining={stopTraining}
                generateDataset={generateDataset}
                validateModel={validateModel}
              />
            </CardContent>
          </Card>
        </div>
        
        {/* Live Updates Panel */}
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  Live Updates
                </CardTitle>
                <Button size="sm" variant="ghost" onClick={clearLiveUpdates}>
                  Clear
                </Button>
              </div>
            </CardHeader>
            <CardContent className="pt-0">
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {liveUpdates.length === 0 ? (
                  <div className="text-center text-gray-500 py-8">
                    <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No updates yet</p>
                    <p className="text-xs">Start training to see live updates</p>
                  </div>
                ) : (
                  liveUpdates.slice().reverse().map((update, index) => (
                    <div key={index} className="p-3 bg-gray-50 rounded-lg border text-sm">
                      <div className="flex items-center justify-between mb-1">
                        <Badge variant="outline" className="text-xs">
                          {update.type.replace('_', ' ')}
                        </Badge>
                        <span className="text-xs text-gray-500">
                          {new Date(update.timestamp * 1000).toLocaleTimeString()}
                        </span>
                      </div>
                      <div className="text-gray-700">
                        {update.type === 'training_step' && (
                          <div>
                            Step {update.step}: Loss {update.loss?.toFixed(4)} 
                            {update.gpu_memory && ` | GPU: ${update.gpu_memory.toFixed(1)}GB`}
                          </div>
                        )}
                        {update.type === 'epoch_start' && (
                          <div>Started epoch {update.epoch}/{update.total_epochs}</div>
                        )}
                        {update.type === 'epoch_end' && (
                          <div>Completed epoch {update.epoch} in {update.epoch_time?.toFixed(1)}s</div>
                        )}
                        {update.type === 'checkpoint_saved' && (
                          <div>Checkpoint saved at step {update.step}</div>
                        )}
                        {update.type === 'best_model_saved' && (
                          <div>🎉 New best model! Loss: {update.best_loss?.toFixed(4)}</div>
                        )}
                        {update.type === 'training_complete' && (
                          <div>✅ Training completed! {update.total_steps} steps</div>
                        )}
                        {update.type === 'connection_established' && (
                          <div>🔗 Connected to training server</div>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
      
      {/* Navigation */}
      <Card>
        <CardHeader>
          <div className="flex justify-between items-center">
            <Button
              variant="outline"
              onClick={prevStep}
              disabled={currentStep === 0}
            >
              ← Previous
            </Button>
            
            <div className="flex gap-2">
              {currentStep === steps.length - 1 ? (
                <Button onClick={() => setCurrentStep(0)} variant="default">
                  🎉 Start New Training
                </Button>
              ) : (
                <Button onClick={nextStep}>
                  Next →
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>
    </div>
  );
};

// Step Components
const ConfigurationStep: React.FC<any> = ({ config, setConfig }) => {
  const modelSizes = {
    small: { params: '~25M', memory: '~100MB', speed: 'Fast', use: 'Testing & Validation' },
    medium: { params: '~100M', memory: '~400MB', speed: 'Medium', use: 'Development' },
    large: { params: '~400M', memory: '~1.6GB', speed: 'Slow', use: 'Production' }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {Object.entries(modelSizes).map(([size, info]) => (
          <Card 
            key={size}
            className={`cursor-pointer transition-all ${
              config.model_size === size ? 'ring-2 ring-blue-500 bg-blue-50' : 'hover:shadow-md'
            }`}
            onClick={() => setConfig({ ...config, model_size: size as any })}
          >
            <CardHeader className="pb-2">
              <CardTitle className="text-lg capitalize">{size} Model</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                <div><strong>Parameters:</strong> {info.params}</div>
                <div><strong>Memory:</strong> {info.memory}</div>
                <div><strong>Speed:</strong> {info.speed}</div>
                <div><strong>Best for:</strong> {info.use}</div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Core Training Parameters</h3>
          
          <div className="space-y-2">
            <Label htmlFor="batch_size">Batch Size</Label>
            <Select 
              value={config.batch_size.toString()} 
              onValueChange={(value) => setConfig({ ...config, batch_size: parseInt(value) })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="2">2 (Safe for most GPUs)</SelectItem>
                <SelectItem value="4">4 (Recommended)</SelectItem>
                <SelectItem value="8">8 (High-end GPUs)</SelectItem>
                <SelectItem value="16">16 (Multi-GPU)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="learning_rate">Learning Rate</Label>
            <Select 
              value={config.learning_rate.toString()} 
              onValueChange={(value) => setConfig({ ...config, learning_rate: parseFloat(value) })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0.00005">5e-5 (Conservative)</SelectItem>
                <SelectItem value="0.0001">1e-4 (Recommended)</SelectItem>
                <SelectItem value="0.0002">2e-4 (Aggressive)</SelectItem>
                <SelectItem value="0.0005">5e-4 (Experimental)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="guidance_scale">Guidance Scale</Label>
            <Select 
              value={config.guidance_scale.toString()} 
              onValueChange={(value) => setConfig({ ...config, guidance_scale: parseFloat(value) })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="3.0">3.0 (Subtle guidance)</SelectItem>
                <SelectItem value="7.5">7.5 (Recommended)</SelectItem>
                <SelectItem value="10.0">10.0 (Strong guidance)</SelectItem>
                <SelectItem value="15.0">15.0 (Very strong)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Advanced Options</h3>
          
          <div className="space-y-2">
            <Label htmlFor="cross_modal_weight">Cross-Modal Alignment Weight</Label>
            <Select 
              value={config.cross_modal_weight.toString()} 
              onValueChange={(value) => setConfig({ ...config, cross_modal_weight: parseFloat(value) })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0.1">0.1 (Light alignment)</SelectItem>
                <SelectItem value="0.3">0.3 (Recommended)</SelectItem>
                <SelectItem value="0.5">0.5 (Strong alignment)</SelectItem>
                <SelectItem value="0.8">0.8 (Very strong)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="num_inference_steps">Inference Steps</Label>
            <Select 
              value={config.num_inference_steps.toString()} 
              onValueChange={(value) => setConfig({ ...config, num_inference_steps: parseInt(value) })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="20">20 (Fast)</SelectItem>
                <SelectItem value="50">50 (Recommended)</SelectItem>
                <SelectItem value="100">100 (High quality)</SelectItem>
                <SelectItem value="200">200 (Maximum quality)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="use_ema"
              checked={config.use_ema}
              onChange={(e) => setConfig({ ...config, use_ema: e.target.checked })}
              className="rounded"
            />
            <Label htmlFor="use_ema">Enable EMA (Exponential Moving Average)</Label>
          </div>
        </div>
      </div>
    </div>
  );
};

const DatasetStep: React.FC<any> = ({ config, setConfig, generateDataset }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setProgress(0);
    
    try {
      await generateDataset({
        num_characters: config.num_characters,
        conversations_per_character: config.conversations_per_character,
        multimodal_ratio: config.multimodal_ratio
      });
    } catch (error) {
      console.error('Dataset generation failed:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Dataset Configuration</h3>
          
          <div className="space-y-2">
            <Label htmlFor="num_characters">Number of Characters</Label>
            <Select 
              value={config.num_characters.toString()} 
              onValueChange={(value) => setConfig({ ...config, num_characters: parseInt(value) })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="10">10 (Quick test)</SelectItem>
                <SelectItem value="25">25 (Small scale)</SelectItem>
                <SelectItem value="50">50 (Medium scale)</SelectItem>
                <SelectItem value="100">100 (Large scale)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="conversations_per_character">Conversations per Character</Label>
            <Select 
              value={config.conversations_per_character.toString()} 
              onValueChange={(value) => setConfig({ ...config, conversations_per_character: parseInt(value) })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="25">25 (Quick validation)</SelectItem>
                <SelectItem value="50">50 (Small scale)</SelectItem>
                <SelectItem value="100">100 (Medium scale)</SelectItem>
                <SelectItem value="500">500 (Large scale)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="multimodal_ratio">Multimodal Samples Ratio</Label>
            <Select 
              value={config.multimodal_ratio.toString()} 
              onValueChange={(value) => setConfig({ ...config, multimodal_ratio: parseFloat(value) })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="0.1">10% (Text-heavy)</SelectItem>
                <SelectItem value="0.3">30% (Recommended)</SelectItem>
                <SelectItem value="0.5">50% (Balanced)</SelectItem>
                <SelectItem value="0.8">80% (Multimodal-heavy)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Estimated Dataset Size</h3>
          
          <div className="bg-gray-50 p-4 rounded-lg space-y-2">
            <div className="flex justify-between">
              <span>Total Conversations:</span>
              <Badge variant="secondary">
                {(config.num_characters * config.conversations_per_character).toLocaleString()}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span>Multimodal Samples:</span>
              <Badge variant="secondary">
                {Math.round(config.num_characters * config.conversations_per_character * config.multimodal_ratio).toLocaleString()}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span>Text-only Samples:</span>
              <Badge variant="secondary">
                {Math.round(config.num_characters * config.conversations_per_character * (1 - config.multimodal_ratio)).toLocaleString()}
              </Badge>
            </div>
            <div className="flex justify-between font-semibold pt-2 border-t">
              <span>Est. Generation Time:</span>
              <Badge>
                ~{Math.round(config.num_characters * config.conversations_per_character / 100)} hours
              </Badge>
            </div>
          </div>

          <Button 
            onClick={handleGenerate}
            disabled={isGenerating}
            className="w-full"
          >
            {isGenerating ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Generating Dataset...
              </>
            ) : (
              <>
                <Brain className="w-4 h-4 mr-2" />
                Generate Multimodal Dataset
              </>
            )}
          </Button>
        </div>
      </div>

      {isGenerating && (
        <Card>
          <CardContent className="p-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Generation Progress</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} />
              <p className="text-xs text-gray-600">
                Generating synthetic conversations with text, speech, control tokens, and memory vectors...
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

const ValidationStep: React.FC<any> = ({ validateModel }) => {
  const [isValidating, setIsValidating] = useState(false);
  const [validationResults, setValidationResults] = useState<any>(null);

  const handleValidate = async () => {
    setIsValidating(true);
    try {
      const results = await validateModel();
      setValidationResults(results);
    } catch (error) {
      console.error('Validation failed:', error);
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <p className="text-gray-600 mb-4">
          Before training, let's validate the model architecture and fix any issues.
        </p>
        
        <Button 
          onClick={handleValidate}
          disabled={isValidating}
          size="lg"
        >
          {isValidating ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
              Validating Model...
            </>
          ) : (
            <>
              <CheckCircle className="w-4 h-4 mr-2" />
              Run Model Validation
            </>
          )}
        </Button>
      </div>

      {validationResults && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {validationResults.success ? (
                <CheckCircle className="w-5 h-5 text-green-600" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-600" />
              )}
              Validation Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {validationResults.checks?.map((check: any, index: number) => (
                <div key={index} className="flex items-center gap-2">
                  {check.passed ? (
                    <CheckCircle className="w-4 h-4 text-green-600" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-red-600" />
                  )}
                  <span className={check.passed ? 'text-green-700' : 'text-red-700'}>
                    {check.description}
                  </span>
                </div>
              ))}
              
              {validationResults.issues && validationResults.issues.length > 0 && (
                <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
                  <h4 className="font-semibold text-yellow-800 mb-2">Issues Found:</h4>
                  <ul className="list-disc list-inside space-y-1 text-yellow-700">
                    {validationResults.issues.map((issue: string, index: number) => (
                      <li key={index}>{issue}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
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
  const [currentPhase, setCurrentPhase] = useState<'small' | 'medium' | 'large'>('small');
  
  const phases = [
    { 
      name: 'small', 
      title: 'Small Scale Training',
      description: 'Quick validation with 25M parameter model',
      steps: 5000,
      characters: 10
    },
    { 
      name: 'medium', 
      title: 'Medium Scale Training', 
      description: 'Development training with 100M parameter model',
      steps: 20000,
      characters: 25
    },
    { 
      name: 'large', 
      title: 'Large Scale Training',
      description: 'Full training with 400M parameter model', 
      steps: 50000,
      characters: 50
    }
  ];

  const currentPhaseData = phases.find(p => p.name === currentPhase);

  const handleStartTraining = () => {
    startTraining({
      ...config,
      model_size: currentPhase,
      max_steps: currentPhaseData?.steps || 5000,
      num_characters: currentPhaseData?.characters || 10
    });
  };

  return (
    <div className="space-y-6">
      {/* Phase Selection */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {phases.map((phase) => (
          <Card 
            key={phase.name}
            className={`cursor-pointer transition-all ${
              currentPhase === phase.name ? 'ring-2 ring-blue-500 bg-blue-50' : 'hover:shadow-md'
            }`}
            onClick={() => setCurrentPhase(phase.name as any)}
          >
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">{phase.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600 mb-2">{phase.description}</p>
              <div className="space-y-1 text-xs">
                <div><strong>Steps:</strong> {phase.steps.toLocaleString()}</div>
                <div><strong>Characters:</strong> {phase.characters}</div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Training Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="w-5 h-5" />
            {currentPhaseData?.title}
          </CardTitle>
          <p className="text-gray-600">{currentPhaseData?.description}</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            {trainingStatus?.status === 'idle' && (
              <Button onClick={handleStartTraining} className="flex items-center gap-2">
                <Play className="w-4 h-4" />
                Start Training
              </Button>
            )}
            
            {trainingStatus?.status === 'training' && (
              <Button onClick={pauseTraining} variant="outline" className="flex items-center gap-2">
                <Pause className="w-4 h-4" />
                Pause Training
              </Button>
            )}
            
            {trainingStatus?.status === 'paused' && (
              <Button onClick={resumeTraining} className="flex items-center gap-2">
                <Play className="w-4 h-4" />
                Resume Training
              </Button>
            )}
            
            {['training', 'paused'].includes(trainingStatus?.status) && (
              <Button onClick={stopTraining} variant="destructive" className="flex items-center gap-2">
                <Square className="w-4 h-4" />
                Stop Training
              </Button>
            )}
          </div>

          {/* Training Progress */}
          {trainingMetrics && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {trainingMetrics.current_step || 0}
                  </div>
                  <div className="text-sm text-gray-600">Step</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {trainingMetrics.total_loss?.toFixed(4) || '---'}
                  </div>
                  <div className="text-sm text-gray-600">Loss</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {trainingMetrics.learning_rate?.toExponential(1) || '---'}
                  </div>
                  <div className="text-sm text-gray-600">LR</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {trainingMetrics.alignment_loss?.toFixed(4) || '---'}
                  </div>
                  <div className="text-sm text-gray-600">Alignment</div>
                </div>
              </div>
              
              <Progress 
                value={(trainingMetrics.current_step / (currentPhaseData?.steps || 1)) * 100} 
                className="h-2"
              />
            </div>
          )}
        </CardContent>
      </Card>
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
  const productionConfig = {
    ...config,
    model_size: 'large',
    max_steps: 100000,
    num_characters: 100,
    conversations_per_character: 500,
    save_steps: 1000,
    eval_steps: 500
  };

  const handleStartProduction = () => {
    startTraining(productionConfig);
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-2xl font-bold mb-2">🚀 Production Training</h3>
        <p className="text-gray-600">
          Final large-scale training run with the complete dataset and large model
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Production Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-blue-600">400M</div>
              <div className="text-sm text-gray-600">Parameters</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-green-600">
                {productionConfig.num_characters}
              </div>
              <div className="text-sm text-gray-600">Characters</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-purple-600">
                {(productionConfig.num_characters * productionConfig.conversations_per_character).toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Conversations</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-orange-600">
                {productionConfig.max_steps.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Max Steps</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Production Training Controls */}
      <Card>
        <CardContent className="p-6">
          <div className="flex gap-2 justify-center">
            {trainingStatus?.status === 'idle' && (
              <Button onClick={handleStartProduction} size="lg" className="flex items-center gap-2">
                <Rocket className="w-5 h-5" />
                Start Production Training
              </Button>
            )}
            
            {trainingStatus?.status === 'training' && (
              <Button onClick={pauseTraining} variant="outline" size="lg" className="flex items-center gap-2">
                <Pause className="w-5 h-5" />
                Pause Training
              </Button>
            )}
            
            {trainingStatus?.status === 'paused' && (
              <Button onClick={resumeTraining} size="lg" className="flex items-center gap-2">
                <Play className="w-5 h-5" />
                Resume Training
              </Button>
            )}
            
            {['training', 'paused'].includes(trainingStatus?.status) && (
              <Button onClick={stopTraining} variant="destructive" size="lg" className="flex items-center gap-2">
                <Square className="w-5 h-5" />
                Stop Training
              </Button>
            )}
          </div>

          {/* Production Training Progress */}
          {trainingMetrics && (
            <div className="mt-6 space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div className="text-center">
                  <div className="text-xl font-bold text-blue-600">
                    {trainingMetrics.current_step || 0}
                  </div>
                  <div className="text-sm text-gray-600">Step</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-green-600">
                    {trainingMetrics.total_loss?.toFixed(4) || '---'}
                  </div>
                  <div className="text-sm text-gray-600">Total Loss</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-purple-600">
                    {trainingMetrics.text_loss?.toFixed(4) || '---'}
                  </div>
                  <div className="text-sm text-gray-600">Text Loss</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-orange-600">
                    {trainingMetrics.speech_loss?.toFixed(4) || '---'}
                  </div>
                  <div className="text-sm text-gray-600">Speech Loss</div>
                </div>
                <div className="text-center">
                  <div className="text-xl font-bold text-red-600">
                    {trainingMetrics.alignment_loss?.toFixed(4) || '---'}
                  </div>
                  <div className="text-sm text-gray-600">Alignment</div>
                </div>
              </div>
              
              <Progress 
                value={(trainingMetrics.current_step / productionConfig.max_steps) * 100} 
                className="h-3"
              />
              
              <div className="text-center text-sm text-gray-600">
                ETA: ~{Math.round((productionConfig.max_steps - trainingMetrics.current_step) / 100)} hours
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default DiffusionTrainingWizard; 