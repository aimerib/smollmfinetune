import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import CreatorLayout from './components/CreatorLayout';
import CreatorDashboard from './pages/CreatorDashboard';
import CharacterBuilder from './pages/CharacterBuilder';
import WorldBuilder from './pages/WorldBuilder';
import DatasetStudio from './pages/DatasetStudio';
import MultimodalStudio from './pages/MultimodalStudio';
import DiffusionTraining from './pages/DiffusionTraining';
import HomePage from './pages/HomePage';
import ChatPage from './pages/ChatPage';
import DirectorsView from './pages/DirectorsView';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import ProfilePage from './pages/ProfilePage';
import './styles/design-system.css';
import './App.css';

function App() {
  return (
    <Router>
      <Routes>
        {/* Public routes */}
        <Route path="/" element={<HomePage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        
        {/* Creator routes with layout */}
        <Route path="/creator" element={<CreatorLayout />}>
          <Route index element={<CreatorDashboard />} />
          <Route path="characters" element={<div>Characters List (TODO)</div>} />
          <Route path="character-builder" element={<CharacterBuilder />} />
          <Route path="worlds" element={<div>Worlds List (TODO)</div>} />
          <Route path="world-builder" element={<WorldBuilder />} />
                      <Route path="dataset-studio" element={<DatasetStudio />} />
          <Route path="multimodal-studio" element={<MultimodalStudio />} />
          <Route path="training" element={<div>Training Dashboard (TODO)</div>} />
          <Route path="diffusion-training" element={<DiffusionTraining />} />
          <Route path="projects" element={<div>All Projects (TODO)</div>} />
          <Route path="project/:id" element={<div>Project Details (TODO)</div>} />
          <Route path="new-project" element={<div>New Project (TODO)</div>} />
          <Route path="profile" element={<ProfilePage />} />
          <Route path="settings" element={<div>Settings (TODO)</div>} />
          <Route path="api-keys" element={<div>API Keys (TODO)</div>} />
        </Route>
        
        {/* Game/Runtime routes */}
        <Route path="/play" element={<CreatorLayout />}>
          <Route index element={<ChatPage />} />
        </Route>
        
        {/* Director's tools */}
        <Route path="/directors-chair" element={<CreatorLayout />}>
          <Route index element={<div>Director's Chair (TODO)</div>} />
        </Route>
        <Route path="/directors-view" element={<DirectorsView />} />
        
        {/* Catch all - redirect to creator dashboard */}
        <Route path="*" element={<Navigate to="/creator" replace />} />
      </Routes>
    </Router>
  );
}

export default App;
