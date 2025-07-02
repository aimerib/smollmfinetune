import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import '../styles/design-system.css';

interface NavItem {
  label: string;
  path: string;
  icon?: string;
}

interface UserInfo {
  name: string;
  email: string;
  avatar?: string;
  role: 'creator' | 'player' | 'admin';
}

const Navigation: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isScrolled, setIsScrolled] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [notificationCount, setNotificationCount] = useState(3);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Mock user data - replace with actual auth
  const user: UserInfo = {
    name: 'Alex Morgan',
    email: 'alex@example.com',
    role: 'creator'
  };

  const navItems: NavItem[] = [
    { label: 'Dashboard', path: '/creator', icon: '🏠' },
    { label: 'Characters', path: '/creator/characters', icon: '🎭' },
    { label: 'Worlds', path: '/creator/worlds', icon: '🌍' },
    { label: 'Training', path: '/creator/training', icon: '🧠' },
    { label: 'Play', path: '/play', icon: '🎮' }
  ];

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleLogout = () => {
    // TODO: Implement actual logout
    navigate('/login');
  };

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  return (
    <>
      <nav className={`navigation ${isScrolled ? 'scrolled' : ''}`}>
        <div className="nav-container">
          {/* Logo */}
          <Link to="/creator" className="nav-logo">
            <span className="logo-icon">🎭</span>
            <span className="logo-text gradient-text">Character Devkit</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="nav-items">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`nav-item ${isActive(item.path) ? 'active' : ''}`}
              >
                {item.icon && <span className="nav-icon">{item.icon}</span>}
                <span>{item.label}</span>
              </Link>
            ))}
          </div>

          {/* Right Section */}
          <div className="nav-right">
            {/* Notifications */}
            <button className="nav-button notification-button">
              <span>🔔</span>
              {notificationCount > 0 && (
                <span className="notification-badge">{notificationCount}</span>
              )}
            </button>

            {/* User Menu */}
            <div className="user-menu-container">
              <button 
                className="user-menu-trigger"
                onClick={() => setUserMenuOpen(!userMenuOpen)}
              >
                <div className="user-avatar">
                  {user.avatar ? (
                    <img src={user.avatar} alt={user.name} />
                  ) : (
                    <span>{user.name.charAt(0)}</span>
                  )}
                </div>
                <span className="user-name">{user.name}</span>
                <span className="dropdown-arrow">▼</span>
              </button>

              {userMenuOpen && (
                <>
                  <div 
                    className="menu-backdrop" 
                    onClick={() => setUserMenuOpen(false)}
                  />
                  <div className="user-dropdown">
                    <div className="dropdown-header">
                      <div className="user-info">
                        <div className="user-name">{user.name}</div>
                        <div className="user-email text-secondary">{user.email}</div>
                      </div>
                    </div>
                    <div className="dropdown-divider" />
                    <Link to="/creator/profile" className="dropdown-item">
                      <span>👤</span> Profile
                    </Link>
                    <Link to="/creator/settings" className="dropdown-item">
                      <span>⚙️</span> Settings
                    </Link>
                    <Link to="/creator/api-keys" className="dropdown-item">
                      <span>🔑</span> API Keys
                    </Link>
                    <div className="dropdown-divider" />
                    <button onClick={handleLogout} className="dropdown-item logout">
                      <span>🚪</span> Logout
                    </button>
                  </div>
                </>
              )}
            </div>

            {/* Mobile Menu Toggle */}
            <button 
              className="mobile-menu-toggle"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <span className="hamburger"></span>
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="mobile-menu">
          <div className="mobile-menu-backdrop" onClick={() => setMobileMenuOpen(false)} />
          <div className="mobile-menu-content">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`mobile-nav-item ${isActive(item.path) ? 'active' : ''}`}
                onClick={() => setMobileMenuOpen(false)}
              >
                {item.icon && <span>{item.icon}</span>}
                <span>{item.label}</span>
              </Link>
            ))}
            <div className="mobile-menu-divider" />
            <Link to="/creator/profile" className="mobile-nav-item">
              <span>👤</span> Profile
            </Link>
            <Link to="/creator/settings" className="mobile-nav-item">
              <span>⚙️</span> Settings
            </Link>
            <button onClick={handleLogout} className="mobile-nav-item logout">
              <span>🚪</span> Logout
            </button>
          </div>
        </div>
      )}

      <style>{`
        .navigation {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          z-index: 1000;
          background: var(--color-background);
          border-bottom: 1px solid transparent;
          transition: all var(--duration-normal) var(--ease-out);
        }

        .navigation.scrolled {
          background: rgba(2, 6, 23, 0.95);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border-bottom-color: var(--color-border-subtle);
        }

        .nav-container {
          max-width: 1280px;
          margin: 0 auto;
          padding: var(--space-4) var(--space-6);
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        /* Logo */
        .nav-logo {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          text-decoration: none;
          font-size: var(--text-xl);
          font-weight: var(--font-bold);
        }

        .logo-icon {
          font-size: var(--text-2xl);
        }

        .logo-text {
          letter-spacing: -0.02em;
        }

        /* Nav Items */
        .nav-items {
          display: flex;
          gap: var(--space-1);
          margin-left: var(--space-12);
        }

        .nav-item {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-2) var(--space-4);
          text-decoration: none;
          color: var(--color-text-secondary);
          font-size: var(--text-sm);
          font-weight: var(--font-medium);
          border-radius: var(--radius-md);
          transition: all var(--duration-fast) var(--ease-out);
          position: relative;
        }

        .nav-item:hover {
          color: var(--color-text-primary);
          background: var(--color-surface);
        }

        .nav-item.active {
          color: var(--color-primary);
        }

        .nav-item.active::after {
          content: '';
          position: absolute;
          bottom: -4px;
          left: var(--space-4);
          right: var(--space-4);
          height: 2px;
          background: var(--color-primary);
          border-radius: var(--radius-full);
        }

        .nav-icon {
          font-size: var(--text-base);
        }

        /* Right Section */
        .nav-right {
          display: flex;
          align-items: center;
          gap: var(--space-4);
          margin-left: auto;
        }

        .nav-button {
          position: relative;
          padding: var(--space-2);
          background: transparent;
          border: none;
          color: var(--color-text-secondary);
          font-size: var(--text-xl);
          cursor: pointer;
          border-radius: var(--radius-md);
          transition: all var(--duration-fast) var(--ease-out);
        }

        .nav-button:hover {
          color: var(--color-text-primary);
          background: var(--color-surface);
        }

        .notification-badge {
          position: absolute;
          top: 0;
          right: 0;
          background: var(--color-primary);
          color: white;
          font-size: var(--text-xs);
          font-weight: var(--font-bold);
          padding: 2px 6px;
          border-radius: var(--radius-full);
          min-width: 18px;
          text-align: center;
        }

        /* User Menu */
        .user-menu-container {
          position: relative;
        }

        .user-menu-trigger {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-2) var(--space-3);
          background: var(--color-surface);
          border: 1px solid var(--color-border-subtle);
          border-radius: var(--radius-full);
          color: var(--color-text-primary);
          font-size: var(--text-sm);
          font-weight: var(--font-medium);
          cursor: pointer;
          transition: all var(--duration-fast) var(--ease-out);
        }

        .user-menu-trigger:hover {
          border-color: var(--color-border);
        }

        .user-avatar {
          width: 32px;
          height: 32px;
          border-radius: var(--radius-full);
          background: var(--color-primary);
          color: white;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: var(--font-bold);
          overflow: hidden;
        }

        .user-avatar img {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .dropdown-arrow {
          font-size: var(--text-xs);
          opacity: 0.5;
          transition: transform var(--duration-fast) var(--ease-out);
        }

        .user-menu-container.open .dropdown-arrow {
          transform: rotate(180deg);
        }

        /* Dropdown */
        .menu-backdrop {
          position: fixed;
          inset: 0;
          z-index: 10;
        }

        .user-dropdown {
          position: absolute;
          top: calc(100% + var(--space-2));
          right: 0;
          min-width: 240px;
          background: var(--color-surface-elevated);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          box-shadow: var(--shadow-xl);
          z-index: 11;
          animation: slide-up var(--duration-fast) var(--ease-out);
        }

        .dropdown-header {
          padding: var(--space-4);
        }

        .dropdown-divider {
          height: 1px;
          background: var(--color-border-subtle);
          margin: var(--space-2) 0;
        }

        .dropdown-item {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          padding: var(--space-3) var(--space-4);
          color: var(--color-text-secondary);
          text-decoration: none;
          font-size: var(--text-sm);
          transition: all var(--duration-fast) var(--ease-out);
          cursor: pointer;
          background: none;
          border: none;
          width: 100%;
          text-align: left;
        }

        .dropdown-item:hover {
          background: var(--color-surface);
          color: var(--color-text-primary);
        }

        .dropdown-item.logout {
          color: var(--color-error);
        }

        /* Mobile Menu */
        .mobile-menu-toggle {
          display: none;
          padding: var(--space-2);
          background: transparent;
          border: none;
          color: var(--color-text-primary);
          cursor: pointer;
        }

        .hamburger {
          display: block;
          width: 24px;
          height: 2px;
          background: currentColor;
          position: relative;
        }

        .hamburger::before,
        .hamburger::after {
          content: '';
          position: absolute;
          width: 100%;
          height: 100%;
          background: currentColor;
          left: 0;
        }

        .hamburger::before {
          top: -8px;
        }

        .hamburger::after {
          top: 8px;
        }

        .mobile-menu {
          position: fixed;
          inset: 0;
          z-index: 999;
        }

        .mobile-menu-backdrop {
          position: absolute;
          inset: 0;
          background: rgba(0, 0, 0, 0.5);
        }

        .mobile-menu-content {
          position: absolute;
          right: 0;
          top: 0;
          bottom: 0;
          width: 280px;
          background: var(--color-surface);
          padding: var(--space-6);
          overflow-y: auto;
          animation: slide-in-right var(--duration-normal) var(--ease-out);
        }

        @keyframes slide-in-right {
          from {
            transform: translateX(100%);
          }
          to {
            transform: translateX(0);
          }
        }

        .mobile-nav-item {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          padding: var(--space-3) var(--space-4);
          margin: 0 calc(var(--space-4) * -1);
          color: var(--color-text-secondary);
          text-decoration: none;
          font-size: var(--text-base);
          transition: all var(--duration-fast) var(--ease-out);
          background: none;
          border: none;
          width: calc(100% + var(--space-8));
          text-align: left;
          cursor: pointer;
        }

        .mobile-nav-item:hover {
          background: var(--color-surface-elevated);
          color: var(--color-text-primary);
        }

        .mobile-nav-item.active {
          color: var(--color-primary);
        }

        .mobile-menu-divider {
          height: 1px;
          background: var(--color-border-subtle);
          margin: var(--space-4) 0;
        }

        /* Responsive */
        @media (max-width: 768px) {
          .nav-items {
            display: none;
          }

          .user-name {
            display: none;
          }

          .mobile-menu-toggle {
            display: block;
          }
        }
      `}</style>
    </>
  );
};

export default Navigation; 