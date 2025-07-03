import React from 'react';
import DiffusionTrainingWizard from '@/components/DiffusionTrainingWizard';
import { Card, CardContent } from '@/components/ui/card';
import { useDiffusionTraining } from '@/hooks/useDiffusionTraining';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Wifi, WifiOff, Activity, Terminal } from 'lucide-react';

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
          <Card className="bg-white border-2 border-blue-100">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <Terminal className="w-5 h-5 text-gray-600" />
                    <span className="font-semibold">Enhanced Training Server</span>
                  </div>
                  
                  {connectionStatus === 'connected' && (
                    <Badge className="bg-green-100 text-green-800 border-green-200">
                      <Wifi className="w-3 h-3 mr-1" />
                      Connected (ws://localhost:8765)
                    </Badge>
                  )}
                  {connectionStatus === 'connecting' && (
                    <Badge variant="secondary" className="bg-yellow-100 text-yellow-800 border-yellow-200">
                      <Activity className="w-3 h-3 mr-1 animate-pulse" />
                      Connecting...
                    </Badge>
                  )}
                  {connectionStatus === 'disconnected' && (
                    <Badge variant="outline" className="bg-gray-100 text-gray-600 border-gray-300">
                      <WifiOff className="w-3 h-3 mr-1" />
                      Disconnected
                    </Badge>
                  )}
                </div>
                
                <div className="flex gap-2">
                  {connectionStatus === 'disconnected' && (
                    <Button size="sm" onClick={connectWebSocket}>
                      <Wifi className="w-4 h-4 mr-1" />
                      Connect
                    </Button>
                  )}
                  {connectionStatus === 'connected' && (
                    <Button size="sm" variant="outline" onClick={disconnectWebSocket}>
                      <WifiOff className="w-4 h-4 mr-1" />
                      Disconnect
                    </Button>
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
            </CardContent>
          </Card>
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card className="bg-blue-50 border-blue-200">
            <CardContent className="p-6 text-center">
              <div className="text-3xl mb-2">🧠</div>
              <h3 className="font-semibold text-blue-900 mb-2">Multimodal Intelligence</h3>
              <p className="text-sm text-blue-700">
                Generate text, speech, control tokens, and memory vectors in a unified model architecture
              </p>
            </CardContent>
          </Card>
          
          <Card className="bg-purple-50 border-purple-200">
            <CardContent className="p-6 text-center">
              <div className="text-3xl mb-2">🔗</div>
              <h3 className="font-semibold text-purple-900 mb-2">Cross-Modal Attention</h3>
              <p className="text-sm text-purple-700">
                Advanced attention mechanisms that align and correlate different modalities
              </p>
            </CardContent>
          </Card>
          
          <Card className="bg-green-50 border-green-200">
            <CardContent className="p-6 text-center">
              <div className="text-3xl mb-2">⚡</div>
              <h3 className="font-semibold text-green-900 mb-2">Scalable Training</h3>
              <p className="text-sm text-green-700">
                From small validation models to large production-ready character models
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Main Wizard */}
        <DiffusionTrainingWizard />
        
        {/* Footer Info */}
        <div className="mt-12 text-center">
          <Card className="bg-gray-50 border-gray-200">
            <CardContent className="p-6">
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
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default DiffusionTrainingPage; 