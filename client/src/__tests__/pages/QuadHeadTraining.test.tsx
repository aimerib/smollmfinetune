import React from 'react';
import { render, screen } from '@testing-library/react';
import QuadHeadTraining from '../../pages/QuadHeadTraining';

// Mock the QuadHeadTrainingDashboard component since it might have complex WebSocket logic
jest.mock('../../components/QuadHeadTrainingDashboard', () => ({
  QuadHeadTrainingDashboard: () => <div data-testid="quad-head-training-dashboard">Mocked Dashboard</div>
}));

describe('QuadHeadTraining Page', () => {
  test('renders page header and dashboard', () => {
    render(<QuadHeadTraining />);
    
    // Check page header
    expect(screen.getByText('Quad-Head Model Training')).toBeInTheDocument();
    expect(screen.getByText('Train advanced multimodal AI models with integrated speech synthesis capabilities')).toBeInTheDocument();
    
    // Check dashboard is rendered
    expect(screen.getByTestId('quad-head-training-dashboard')).toBeInTheDocument();
  });
  
  test('has correct page structure', () => {
    render(<QuadHeadTraining />);
    
    const pageContainer = screen.getByText('Quad-Head Model Training').closest('.quad-head-training-page');
    expect(pageContainer).toBeInTheDocument();
    
    const headerContainer = screen.getByText('Quad-Head Model Training').closest('.page-header');
    expect(headerContainer).toBeInTheDocument();
  });
}); 