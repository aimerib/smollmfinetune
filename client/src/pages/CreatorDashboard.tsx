import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import '../styles/design-system.css';

interface Project {
  id: string;
  name: string;
  worldName: string;
  characterCount: number;
  lastModified: string;
  status: 'active' | 'training' | 'ready';
}

interface QuickAction {
  icon: string;
  title: string;
  description: string;
  path: string;
  color: string;
}

const CreatorDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[]>([]);
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  useEffect(() => {
    // TODO: Fetch actual projects from API
    setProjects([
      {
        id: '1',
        name: 'Cyberpunk Dreams',
        worldName: 'Neo Tokyo 2077',
        characterCount: 5,
        lastModified: '2 hours ago',
        status: 'active'
      },
      {
        id: '2',
        name: 'Medieval Fantasy',
        worldName: 'Kingdom of Aetheria',
        characterCount: 12,
        lastModified: '1 day ago',
        status: 'training'
      },
      {
        id: '3',
        name: 'Space Opera',
        worldName: 'Galactic Federation',
        characterCount: 8,
        lastModified: '3 days ago',
        status: 'ready'
      }
    ]);
  }, []);

  const quickActions: QuickAction[] = [
    {
      icon: '🎭',
      title: 'Create Character',
      description: 'Design a new character with personality and backstory',
      path: '/creator/character-builder',
      color: 'var(--orange-500)'
    },
    {
      icon: '🌍',
      title: 'Build World',
      description: 'Craft immersive worlds with rich lore and history',
      path: '/creator/world-builder',
      color: 'var(--orange-600)'
    },
    {
      icon: '💬',
      title: 'Generate Dataset',
      description: 'Create training conversations for your characters',
      path: '/creator/dataset-studio',
      color: 'var(--orange-400)'
    },
    {
      icon: '🎬',
      title: "Director's Chair",
      description: 'Fine-tune characters through live conversation',
      path: '/directors-chair',
      color: 'var(--orange-700)'
    }
  ];

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case 'active': return 'badge-active';
      case 'training': return 'badge-training';
      case 'ready': return 'badge-ready';
      default: return '';
    }
  };

  return (
    <div className="creator-dashboard">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="container">
          <div className="hero-content">
            <h1 className="hero-title">
              Welcome to the <span className="gradient-text">Character Creation Devkit</span>
            </h1>
            <p className="hero-subtitle text-secondary">
              Transform AI characters from simple chatbots into believable digital actors
            </p>
            <button 
              className="btn btn-primary btn-lg orange-glow"
              onClick={() => navigate('/creator/new-project')}
            >
              <span>✨</span>
              Start New Project
            </button>
          </div>
        </div>
      </section>

      {/* Quick Actions */}
      <section className="quick-actions-section">
        <div className="container">
          <h2 className="section-title">Quick Actions</h2>
          <div className="grid grid-cols-4 quick-actions-grid">
            {quickActions.map((action, index) => (
              <Link
                key={index}
                to={action.path}
                className="quick-action-card card animate-slide-up"
                style={{ animationDelay: `${index * 100}ms` }}
                onMouseEnter={() => setHoveredCard(action.title)}
                onMouseLeave={() => setHoveredCard(null)}
              >
                <div 
                  className="action-icon"
                  style={{ 
                    color: action.color,
                    transform: hoveredCard === action.title ? 'scale(1.2) rotate(10deg)' : 'scale(1)'
                  }}
                >
                  {action.icon}
                </div>
                <h3 className="action-title">{action.title}</h3>
                <p className="action-description text-secondary">{action.description}</p>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Recent Projects */}
      <section className="projects-section">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Recent Projects</h2>
            <Link to="/creator/projects" className="btn btn-ghost">
              View All →
            </Link>
          </div>
          
          <div className="grid grid-cols-3 projects-grid">
            {projects.map((project) => (
              <div 
                key={project.id}
                className="project-card card card-elevated"
                onClick={() => navigate(`/creator/project/${project.id}`)}
              >
                <div className="project-header">
                  <h3 className="project-name">{project.name}</h3>
                  <span className={`badge ${getStatusBadgeClass(project.status)}`}>
                    {project.status}
                  </span>
                </div>
                
                <div className="project-meta">
                  <div className="meta-item">
                    <span className="meta-icon">🌍</span>
                    <span className="text-secondary">{project.worldName}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-icon">👥</span>
                    <span className="text-secondary">{project.characterCount} characters</span>
                  </div>
                </div>
                
                <div className="project-footer">
                  <span className="text-muted">Modified {project.lastModified}</span>
                  <button className="btn btn-ghost btn-sm">
                    Open →
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Overview */}
      <section className="stats-section">
        <div className="container">
          <div className="stats-grid grid grid-cols-4">
            <div className="stat-card glass">
              <div className="stat-value gradient-text">24</div>
              <div className="stat-label text-secondary">Total Characters</div>
            </div>
            <div className="stat-card glass">
              <div className="stat-value gradient-text">5</div>
              <div className="stat-label text-secondary">Active Worlds</div>
            </div>
            <div className="stat-card glass">
              <div className="stat-value gradient-text">1.2M</div>
              <div className="stat-label text-secondary">Conversations Generated</div>
            </div>
            <div className="stat-card glass">
              <div className="stat-value gradient-text">98%</div>
              <div className="stat-label text-secondary">Character Consistency</div>
            </div>
          </div>
        </div>
      </section>

      <style>{`
        .creator-dashboard {
          min-height: 100vh;
          background: var(--color-background);
        }

        /* Hero Section */
        .hero-section {
          padding: var(--space-20) 0 var(--space-16);
          background: radial-gradient(ellipse at top, rgba(249, 115, 22, 0.1) 0%, transparent 70%);
          position: relative;
          overflow: hidden;
        }

        .hero-section::before {
          content: '';
          position: absolute;
          top: -50%;
          left: -50%;
          width: 200%;
          height: 200%;
          background: radial-gradient(circle, var(--orange-500) 0%, transparent 70%);
          opacity: 0.03;
          animation: rotate 60s linear infinite;
        }

        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .hero-content {
          text-align: center;
          position: relative;
          z-index: 1;
        }

        .hero-title {
          font-size: var(--text-5xl);
          margin-bottom: var(--space-4);
          letter-spacing: -0.02em;
        }

        .hero-subtitle {
          font-size: var(--text-xl);
          margin-bottom: var(--space-8);
          max-width: 600px;
          margin-left: auto;
          margin-right: auto;
        }

        /* Quick Actions */
        .quick-actions-section {
          padding: var(--space-16) 0;
        }

        .section-title {
          margin-bottom: var(--space-8);
        }

        .quick-action-card {
          text-decoration: none;
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: var(--space-8);
          cursor: pointer;
          transition: all var(--duration-normal) var(--ease-out);
        }

        .quick-action-card:hover {
          transform: translateY(-4px);
          border-color: var(--color-primary);
        }

        .action-icon {
          font-size: 3rem;
          margin-bottom: var(--space-4);
          transition: transform var(--duration-normal) var(--ease-spring);
        }

        .action-title {
          font-size: var(--text-lg);
          margin-bottom: var(--space-2);
          color: var(--color-text-primary);
        }

        .action-description {
          font-size: var(--text-sm);
        }

        /* Projects Section */
        .projects-section {
          padding: var(--space-16) 0;
          background: var(--color-surface);
        }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-8);
        }

        .project-card {
          cursor: pointer;
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }

        .project-header {
          display: flex;
          justify-content: space-between;
          align-items: start;
        }

        .project-name {
          font-size: var(--text-xl);
        }

        .project-meta {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
        }

        .meta-item {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          font-size: var(--text-sm);
        }

        .meta-icon {
          font-size: var(--text-base);
        }

        .project-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: auto;
          padding-top: var(--space-4);
          border-top: 1px solid var(--color-border-subtle);
          font-size: var(--text-sm);
        }

        .btn-sm {
          padding: var(--space-1) var(--space-3);
          font-size: var(--text-xs);
        }

        /* Status Badges */
        .badge-active {
          background-color: var(--color-success);
          color: white;
          border-color: var(--color-success);
        }

        .badge-training {
          background-color: var(--color-warning);
          color: white;
          border-color: var(--color-warning);
        }

        .badge-ready {
          background-color: var(--color-info);
          color: white;
          border-color: var(--color-info);
        }

        /* Stats Section */
        .stats-section {
          padding: var(--space-16) 0;
        }

        .stat-card {
          padding: var(--space-8);
          text-align: center;
          border-radius: var(--radius-xl);
        }

        .stat-value {
          font-size: var(--text-4xl);
          font-weight: var(--font-bold);
          margin-bottom: var(--space-2);
        }

        .stat-label {
          font-size: var(--text-sm);
        }

        /* Responsive */
        @media (max-width: 768px) {
          .hero-title {
            font-size: var(--text-3xl);
          }

          .hero-subtitle {
            font-size: var(--text-base);
          }

          .quick-actions-grid {
            grid-template-columns: repeat(2, 1fr);
          }

          .projects-grid {
            grid-template-columns: 1fr;
          }

          .stats-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }
      `}</style>
    </div>
  );
};

export default CreatorDashboard; 