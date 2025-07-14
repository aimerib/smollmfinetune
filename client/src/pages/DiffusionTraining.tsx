import React from 'react';
import DiffusionTrainingWizard from '../components/DiffusionTrainingWizard';
import { useDiffusionTraining } from '../hooks/useDiffusionTraining';

const DiffusionTrainingPage: React.FC = () => {
  const { connectionStatus, connectWebSocket, disconnectWebSocket } = useDiffusionTraining();
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="container mx-auto py-8">
        {/* Page Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            🌊 Diffusion Multimodal Training
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Train cutting-edge diffusion models that generate text, speech, control tokens, and memory vectors 
            simultaneously with cross-modal attention for your character creation platform.
          </p>
        </div>

        {/* Connection Status Panel */}
        <div className="mb-8">
          <div className="bg-white border-2 border-blue-100 rounded-lg shadow-sm">
            <div className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">Enhanced Training Server</span>
                  </div>
                  
                  {connectionStatus === 'connected' && (
                    <span className="px-2 py-1 bg-green-100 text-green-800 border border-green-200 rounded-full text-sm">
                      Connected (ws://localhost:8765)
                    </span>
                  )}
                  {connectionStatus === 'connecting' && (
                    <span className="px-2 py-1 bg-yellow-100 text-yellow-800 border border-yellow-200 rounded-full text-sm">
                      Connecting...
                    </span>
                  )}
                  {connectionStatus === 'disconnected' && (
                    <span className="px-2 py-1 bg-gray-100 text-gray-600 border border-gray-300 rounded-full text-sm">
                      Disconnected
                    </span>
                  )}
                </div>
                
                <div className="flex gap-2">
                  {connectionStatus === 'disconnected' && (
                    <button 
                      className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
                      onClick={connectWebSocket}
                    >
                      Connect
                    </button>
                  )}
                  {connectionStatus === 'connected' && (
                    <button 
                      className="px-3 py-1 text-sm bg-gray-500 text-white rounded hover:bg-gray-600"
                      onClick={disconnectWebSocket}
                    >
                      Disconnect
                    </button>
                  )}
                </div>
              </div>
              
              <div className="mt-4 text-sm text-gray-600">
                <p>
                  🚀 <strong>Enhanced Features:</strong> Real-time training updates, multi-GPU support, 
                  advanced sampling strategies, custom architecture modifications, and live progress monitoring.
                </p>
                <p className="mt-1">
                  💡 <strong>Start the enhanced trainer:</strong> <code className="bg-gray-100 px-1 rounded">python scripts/enhanced_diffusion_training_demo.py</code>
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-blue-50 border border-blue-200 rounded-lg shadow-sm">
            <div className="p-6 text-center">
              <div className="text-3xl mb-2">🧠</div>
              <h3 className="font-semibold text-blue-900 mb-2">Multimodal Intelligence</h3>
              <p className="text-sm text-blue-700">
                Generate text, speech, control tokens, and memory vectors in a unified model architecture
              </p>
            </div>
          </div>
          
          <div className="bg-purple-50 border border-purple-200 rounded-lg shadow-sm">
            <div className="p-6 text-center">
              <div className="text-3xl mb-2">🔗</div>
              <h3 className="font-semibold text-purple-900 mb-2">Cross-Modal Attention</h3>
              <p className="text-sm text-purple-700">
                Advanced attention mechanisms that align and correlate different modalities
              </p>
            </div>
          </div>
          
          <div className="bg-green-50 border border-green-200 rounded-lg shadow-sm">
            <div className="p-6 text-center">
              <div className="text-3xl mb-2">⚡</div>
              <h3 className="font-semibold text-green-900 mb-2">Scalable Training</h3>
              <p className="text-sm text-green-700">
                From small validation models to large production-ready character models
              </p>
            </div>
          </div>
        </div>

        {/* Main Wizard */}
        <DiffusionTrainingWizard />
        
        {/* Footer Info */}
        <div className="mt-12 text-center">
          <div className="bg-gray-50 border border-gray-200 rounded-lg shadow-sm">
            <div className="p-6">
              <h3 className="font-semibold text-gray-900 mb-2">💡 Training Tips</h3>
              <div className="text-sm text-gray-600 space-y-1">
                <p>• Start with the <strong>small model</strong> for quick validation and testing</p>
                <p>• Use <strong>medium models</strong> for development and iterating on character personalities</p>
                <p>• Reserve <strong>large models</strong> for production-quality character voices</p>
                <p>• Monitor cross-modal alignment loss to ensure modalities are learning together</p>
                <p>• Enable EMA (Exponential Moving Average) for more stable training</p>
                <p>• 🔗 <strong>Real-time updates</strong> show live training progress via WebSocket connection</p>
                <p>• 🎯 <strong>Custom features</strong> include layer freezing, progressive unfreezing, and adaptive learning rates</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiffusionTrainingPage; 