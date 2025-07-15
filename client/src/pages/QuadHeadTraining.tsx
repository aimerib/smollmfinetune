import React from 'react';
import { QuadHeadTrainingDashboard } from '../components/QuadHeadTrainingDashboard';

const QuadHeadTraining: React.FC = () => {
  return (
    <div className="quad-head-training-page">
      <div className="page-header">
        <h1>Quad-Head Model Training</h1>
        <p>Train advanced multimodal AI models with integrated speech synthesis capabilities</p>
      </div>
      
      <QuadHeadTrainingDashboard />
    </div>
  );
};

export default QuadHeadTraining; 