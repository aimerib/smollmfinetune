import React, { useState } from 'react';

const ProfilePage: React.FC = () => {
  const [userData, setUserData] = useState({
    username: 'Alex Morgan',
    email: 'alex@example.com',
    firstName: 'Alex',
    lastName: 'Morgan',
    bio: 'AI Character Creator and Storyteller',
    role: 'creator'
  });
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setSuccessMessage('');

    try {
      // TODO: Replace with actual API call to unified backend
      const response = await fetch('/api/auth/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify(userData),
      });

      if (response.ok) {
        setSuccessMessage('Profile updated successfully!');
      } else {
        throw new Error('Failed to update profile');
      }
    } catch (err) {
      console.error('Failed to update profile:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setUserData({
      ...userData,
      [e.target.name]: e.target.value
    });
  };

  return (
    <div className="profile-page">
      <div className="profile-container">
        <div className="profile-header">
          <h1>Profile Settings</h1>
          <p>Manage your account information and preferences</p>
        </div>

        <div className="profile-content">
          <div className="profile-card">
            <div className="profile-avatar-section">
              <div className="profile-avatar">
                <span>{userData.firstName.charAt(0)}{userData.lastName.charAt(0)}</span>
              </div>
              <button className="avatar-change-btn">Change Avatar</button>
            </div>

            <form onSubmit={handleSubmit} className="profile-form">
              {successMessage && (
                <div className="success-banner">
                  <span>✅</span>
                  <span>{successMessage}</span>
                </div>
              )}

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="firstName">First Name</label>
                  <input
                    id="firstName"
                    name="firstName"
                    type="text"
                    value={userData.firstName}
                    onChange={handleChange}
                    required
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="lastName">Last Name</label>
                  <input
                    id="lastName"
                    name="lastName"
                    type="text"
                    value={userData.lastName}
                    onChange={handleChange}
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label htmlFor="username">Username</label>
                <input
                  id="username"
                  name="username"
                  type="text"
                  value={userData.username}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  value={userData.email}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-group">
                <label htmlFor="bio">Bio</label>
                <textarea
                  id="bio"
                  name="bio"
                  value={userData.bio}
                  onChange={handleChange}
                  placeholder="Tell us about yourself..."
                  rows={4}
                />
              </div>

              <div className="form-actions">
                <button 
                  type="submit" 
                  className="save-button"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Saving...
                    </>
                  ) : (
                    'Save Changes'
                  )}
                </button>
              </div>
            </form>
          </div>

          <div className="danger-zone">
            <h3>Danger Zone</h3>
            <p>Irreversible and destructive actions</p>
            <button className="danger-button">
              Delete Account
            </button>
          </div>
        </div>
      </div>

      <style>{`
        .profile-page {
          padding: var(--space-6) var(--space-4);
          max-width: 800px;
          margin: 0 auto;
        }

        .profile-header {
          margin-bottom: var(--space-8);
        }

        .profile-header h1 {
          font-size: var(--text-3xl);
          font-weight: var(--font-bold);
          color: var(--color-text-primary);
          margin-bottom: var(--space-2);
        }

        .profile-header p {
          color: var(--color-text-secondary);
          font-size: var(--text-base);
        }

        .profile-content {
          display: flex;
          flex-direction: column;
          gap: var(--space-8);
        }

        .profile-card {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          padding: var(--space-6);
        }

        .profile-avatar-section {
          display: flex;
          align-items: center;
          gap: var(--space-4);
          margin-bottom: var(--space-6);
          padding-bottom: var(--space-6);
          border-bottom: 1px solid var(--color-border-subtle);
        }

        .profile-avatar {
          width: 80px;
          height: 80px;
          border-radius: var(--radius-full);
          background: var(--color-primary);
          color: white;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: var(--text-xl);
          font-weight: var(--font-bold);
        }

        .avatar-change-btn {
          padding: var(--space-2) var(--space-4);
          background: var(--color-surface-elevated);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          color: var(--color-text-primary);
          cursor: pointer;
          transition: all var(--duration-fast) var(--ease-out);
        }

        .avatar-change-btn:hover {
          background: var(--color-background);
        }

        .profile-form {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
        }

        .success-banner {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-3) var(--space-4);
          background: var(--color-success-bg);
          color: var(--color-success);
          border: 1px solid var(--color-success-border);
          border-radius: var(--radius-md);
          font-size: var(--text-sm);
        }

        .form-row {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: var(--space-4);
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

        .form-group input,
        .form-group textarea {
          padding: var(--space-3) var(--space-4);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-md);
          background: var(--color-background);
          color: var(--color-text-primary);
          font-size: var(--text-base);
          transition: border-color var(--duration-fast) var(--ease-out);
          resize: vertical;
        }

        .form-group input:focus,
        .form-group textarea:focus {
          outline: none;
          border-color: var(--color-primary);
          box-shadow: 0 0 0 3px var(--color-primary-bg);
        }

        .form-actions {
          padding-top: var(--space-4);
          border-top: 1px solid var(--color-border-subtle);
        }

        .save-button {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: var(--space-2);
          padding: var(--space-3) var(--space-6);
          background: var(--color-primary);
          color: white;
          border: none;
          border-radius: var(--radius-md);
          font-size: var(--text-base);
          font-weight: var(--font-medium);
          cursor: pointer;
          transition: all var(--duration-fast) var(--ease-out);
        }

        .save-button:hover:not(:disabled) {
          background: var(--color-primary-hover);
          transform: translateY(-1px);
        }

        .save-button:disabled {
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

        .danger-zone {
          background: var(--color-surface);
          border: 1px solid var(--color-error-border);
          border-radius: var(--radius-lg);
          padding: var(--space-6);
        }

        .danger-zone h3 {
          color: var(--color-error);
          font-size: var(--text-lg);
          font-weight: var(--font-bold);
          margin-bottom: var(--space-2);
        }

        .danger-zone p {
          color: var(--color-text-secondary);
          font-size: var(--text-sm);
          margin-bottom: var(--space-4);
        }

        .danger-button {
          padding: var(--space-2) var(--space-4);
          background: var(--color-error);
          color: white;
          border: none;
          border-radius: var(--radius-md);
          font-size: var(--text-sm);
          cursor: pointer;
          transition: all var(--duration-fast) var(--ease-out);
        }

        .danger-button:hover {
          background: var(--color-error-hover);
        }

        @media (max-width: 768px) {
          .form-row {
            grid-template-columns: 1fr;
          }

          .profile-avatar-section {
            flex-direction: column;
            text-align: center;
          }
        }
      `}</style>
    </div>
  );
};

export default ProfilePage; 