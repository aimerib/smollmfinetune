import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';

const RegisterPage: React.FC = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    // Validate passwords match
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    try {
      // TODO: Replace with actual API call to unified backend
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: formData.username,
          email: formData.email,
          password: formData.password
        }),
      });

      if (response.ok) {
        const data = await response.json();
        // Store auth token
        localStorage.setItem('auth_token', data.token);
        navigate('/creator');
      } else {
        const errorData = await response.json();
        setError(errorData.message || 'Registration failed');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="register-page">
      <div className="register-container">
        <div className="register-card">
          <div className="register-header">
            <div className="logo">
              <span className="logo-icon">🎭</span>
              <h1 className="logo-text gradient-text">Character Devkit</h1>
            </div>
            <p className="register-subtitle">Create your account and start building amazing characters</p>
          </div>

          <form onSubmit={handleSubmit} className="register-form">
            {error && (
              <div className="error-banner">
                <span>⚠️</span>
                <span>{error}</span>
              </div>
            )}

            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input
                id="username"
                name="username"
                type="text"
                value={formData.username}
                onChange={handleChange}
                placeholder="Choose a username"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                id="email"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="Enter your email"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Choose a strong password"
                required
              />
            </div>

            <div className="form-group">
              <label htmlFor="confirmPassword">Confirm Password</label>
              <input
                id="confirmPassword"
                name="confirmPassword"
                type="password"
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="Confirm your password"
                required
              />
            </div>

            <button 
              type="submit" 
              className="register-button"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Creating account...
                </>
              ) : (
                'Create Account'
              )}
            </button>
          </form>

          <div className="register-footer">
            <p>
              Already have an account?{' '}
              <Link to="/login" className="auth-link">
                Sign in
              </Link>
            </p>
          </div>
        </div>
      </div>

      <style>{`
        .register-page {
          min-height: 100vh;
          background: linear-gradient(135deg, 
            var(--color-background) 0%, 
            var(--color-surface) 100%
          );
          display: flex;
          align-items: center;
          justify-content: center;
          padding: var(--space-4);
        }

        .register-container {
          width: 100%;
          max-width: 400px;
        }

        .register-card {
          background: var(--color-surface-elevated);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-xl);
          padding: var(--space-8);
          box-shadow: var(--shadow-2xl);
        }

        .register-header {
          text-align: center;
          margin-bottom: var(--space-8);
        }

        .logo {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: var(--space-2);
          margin-bottom: var(--space-4);
        }

        .logo-icon {
          font-size: var(--text-4xl);
        }

        .logo-text {
          font-size: var(--text-2xl);
          font-weight: var(--font-bold);
          letter-spacing: -0.02em;
        }

        .register-subtitle {
          color: var(--color-text-secondary);
          font-size: var(--text-sm);
        }

        .register-form {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }

        .error-banner {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-3) var(--space-4);
          background: var(--color-error-bg);
          color: var(--color-error);
          border: 1px solid var(--color-error-border);
          border-radius: var(--radius-md);
          font-size: var(--text-sm);
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
        }

        .form-group label {
          font-size: var(--text-sm);
          font-weight: var(--font-medium);
          color: var(--color-text-primary);
        }

        .form-group input {
          padding: var(--space-3) var(--space-4);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          background: var(--color-background);
          color: var(--color-text-primary);
          font-size: var(--text-base);
          transition: border-color var(--duration-fast) var(--ease-out);
        }

        .form-group input:focus {
          outline: none;
          border-color: var(--color-primary);
          box-shadow: 0 0 0 3px var(--color-primary-bg);
        }

        .register-button {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: var(--space-2);
          padding: var(--space-3) var(--space-4);
          background: var(--color-primary);
          color: white;
          border: none;
          border-radius: var(--radius-md);
          font-size: var(--text-base);
          font-weight: var(--font-medium);
          cursor: pointer;
          transition: all var(--duration-fast) var(--ease-out);
          margin-top: var(--space-2);
        }

        .register-button:hover:not(:disabled) {
          background: var(--color-primary-hover);
          transform: translateY(-1px);
        }

        .register-button:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }

        .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid transparent;
          border-top: 2px solid currentColor;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }

        .register-footer {
          text-align: center;
          margin-top: var(--space-6);
          padding-top: var(--space-6);
          border-top: 1px solid var(--color-border-subtle);
        }

        .register-footer p {
          margin-bottom: var(--space-3);
          color: var(--color-text-secondary);
          font-size: var(--text-sm);
        }

        .auth-link {
          color: var(--color-primary);
          text-decoration: none;
          font-weight: var(--font-medium);
        }

        .auth-link:hover {
          text-decoration: underline;
        }
      `}</style>
    </div>
  );
};

export default RegisterPage; 