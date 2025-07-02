import React from 'react';
import { Outlet } from 'react-router-dom';
import Navigation from './Navigation';
import '../styles/design-system.css';

const CreatorLayout: React.FC = () => {
  return (
    <div className="creator-layout">
      <Navigation />
      <main className="main-content">
        <Outlet />
      </main>
      
      <style>{`
        .creator-layout {
          min-height: 100vh;
          background: var(--color-background);
        }
        
        .main-content {
          margin-top: 80px; /* Account for fixed navigation */
          min-height: calc(100vh - 80px);
        }
        
        @media (max-width: 768px) {
          .main-content {
            margin-top: 64px;
            min-height: calc(100vh - 64px);
          }
        }
      `}</style>
    </div>
  );
};

export default CreatorLayout; 