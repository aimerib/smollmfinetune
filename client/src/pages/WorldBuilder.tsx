import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import '../styles/design-system.css';

interface World {
  id: string;
  name: string;
  description: string;
  setting: string;
  rules: Record<string, any>;
  history: string;
  cultures: Record<string, any>;
  locations: Record<string, any>;
}

const WorldBuilder: React.FC = () => {
  const { worldId } = useParams();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('overview');
  const [isSaving, setIsSaving] = useState(false);
  const [world, setWorld] = useState<World>({
    id: '',
    name: '',
    description: '',
    setting: '',
    rules: {},
    history: '',
    cultures: {},
    locations: {}
  });

  useEffect(() => {
    if (worldId && worldId !== 'new') {
      fetchWorld();
    }
  }, [worldId]);

  const fetchWorld = async () => {
    try {
      const token = localStorage.getItem('access_token');
      const response = await fetch(`/api/v1/worlds/${worldId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      if (response.ok) {
        const data = await response.json();
        setWorld(data);
      }
    } catch (error) {
      console.error('Failed to fetch world:', error);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const token = localStorage.getItem('access_token');
      const projectId = localStorage.getItem('current_project_id') || '';
      
      const url = worldId === 'new' 
        ? '/api/v1/worlds/' 
        : `/api/v1/worlds/${worldId}`;
      
      const method = worldId === 'new' ? 'POST' : 'PUT';
      
      const body = worldId === 'new' 
        ? { ...world, project_id: projectId }
        : world;
      
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(body)
      });

      if (response.ok) {
        const data = await response.json();
        setWorld(data);
        if (worldId === 'new') {
          navigate(`/world/${data.id}`);
        }
      }
    } catch (error) {
      console.error('Failed to save world:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleAddRule = () => {
    const ruleName = prompt('Enter rule name:');
    if (ruleName) {
      setWorld({
        ...world,
        rules: {
          ...world.rules,
          [ruleName]: ''
        }
      });
    }
  };

  const handleAddCulture = () => {
    const cultureName = prompt('Enter culture name:');
    if (cultureName) {
      setWorld({
        ...world,
        cultures: {
          ...world.cultures,
          [cultureName]: {
            description: '',
            customs: [],
            beliefs: []
          }
        }
      });
    }
  };

  const handleAddLocation = () => {
    const locationName = prompt('Enter location name:');
    if (locationName) {
      setWorld({
        ...world,
        locations: {
          ...world.locations,
          [locationName]: {
            description: '',
            type: 'city',
            significance: ''
          }
        }
      });
    }
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                World Name
              </label>
              <input
                type="text"
                value={world.name}
                onChange={(e) => setWorld({ ...world, name: e.target.value })}
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-orange-500 focus:outline-none"
                placeholder="Enter world name..."
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Description
              </label>
              <textarea
                value={world.description}
                onChange={(e) => setWorld({ ...world, description: e.target.value })}
                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-orange-500 focus:outline-none h-32"
                placeholder="Brief description of your world..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Setting & Time Period
              </label>
              <textarea
                value={world.setting}
                onChange={(e) => setWorld({ ...world, setting: e.target.value })}
                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-orange-500 focus:outline-none h-48"
                placeholder="Describe the setting, time period, technology level, etc..."
              />
            </div>
          </div>
        );

      case 'rules':
        return (
          <div className="space-y-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-white">World Rules</h3>
              <button
                onClick={handleAddRule}
                className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors"
              >
                Add Rule
              </button>
            </div>
            
            {Object.entries(world.rules).map(([ruleName, ruleValue]) => (
              <div key={ruleName} className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="text-orange-400 font-medium">{ruleName}</h4>
                  <button
                    onClick={() => {
                      const newRules = { ...world.rules };
                      delete newRules[ruleName];
                      setWorld({ ...world, rules: newRules });
                    }}
                    className="text-red-400 hover:text-red-300"
                  >
                    Remove
                  </button>
                </div>
                <textarea
                  value={ruleValue as string || ''}
                  onChange={(e) => setWorld({
                    ...world,
                    rules: {
                      ...world.rules,
                      [ruleName]: e.target.value
                    }
                  })}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-white focus:border-orange-500 focus:outline-none"
                  placeholder="Describe this rule..."
                  rows={3}
                />
              </div>
            ))}
          </div>
        );

      case 'history':
        return (
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                World History
              </label>
              <textarea
                value={world.history}
                onChange={(e) => setWorld({ ...world, history: e.target.value })}
                className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-orange-500 focus:outline-none h-96"
                placeholder="Major historical events, eras, conflicts, discoveries..."
              />
            </div>
          </div>
        );

      case 'cultures':
        return (
          <div className="space-y-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-white">Cultures & Societies</h3>
              <button
                onClick={handleAddCulture}
                className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors"
              >
                Add Culture
              </button>
            </div>
            
            {Object.entries(world.cultures).map(([cultureName, cultureData]) => (
              <div key={cultureName} className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="text-orange-400 font-medium">{cultureName}</h4>
                  <button
                    onClick={() => {
                      const newCultures = { ...world.cultures };
                      delete newCultures[cultureName];
                      setWorld({ ...world, cultures: newCultures });
                    }}
                    className="text-red-400 hover:text-red-300"
                  >
                    Remove
                  </button>
                </div>
                <textarea
                  value={(cultureData as any).description || ''}
                  onChange={(e) => setWorld({
                    ...world,
                    cultures: {
                      ...world.cultures,
                      [cultureName]: {
                        ...(cultureData as any),
                        description: e.target.value
                      }
                    }
                  })}
                  className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-white focus:border-orange-500 focus:outline-none mb-2"
                  placeholder="Describe this culture..."
                  rows={3}
                />
              </div>
            ))}
          </div>
        );

      case 'locations':
        return (
          <div className="space-y-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-white">Important Locations</h3>
              <button
                onClick={handleAddLocation}
                className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors"
              >
                Add Location
              </button>
            </div>
            
            {Object.entries(world.locations).map(([locationName, locationData]) => (
              <div key={locationName} className="bg-slate-800 rounded-lg p-4 border border-slate-700">
                <div className="flex justify-between items-start mb-2">
                  <h4 className="text-orange-400 font-medium">{locationName}</h4>
                  <button
                    onClick={() => {
                      const newLocations = { ...world.locations };
                      delete newLocations[locationName];
                      setWorld({ ...world, locations: newLocations });
                    }}
                    className="text-red-400 hover:text-red-300"
                  >
                    Remove
                  </button>
                </div>
                <div className="space-y-2">
                  <select
                    value={(locationData as any).type || 'city'}
                    onChange={(e) => setWorld({
                      ...world,
                      locations: {
                        ...world.locations,
                        [locationName]: {
                          ...(locationData as any),
                          type: e.target.value
                        }
                      }
                    })}
                    className="px-3 py-1 bg-slate-900 border border-slate-700 rounded text-white focus:border-orange-500 focus:outline-none"
                  >
                    <option value="city">City</option>
                    <option value="town">Town</option>
                    <option value="landmark">Landmark</option>
                    <option value="region">Region</option>
                    <option value="country">Country</option>
                  </select>
                  <textarea
                    value={(locationData as any).description || ''}
                    onChange={(e) => setWorld({
                      ...world,
                      locations: {
                        ...world.locations,
                        [locationName]: {
                          ...(locationData as any),
                          description: e.target.value
                        }
                      }
                    })}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-white focus:border-orange-500 focus:outline-none"
                    placeholder="Describe this location..."
                    rows={2}
                  />
                  <input
                    type="text"
                    value={(locationData as any).significance || ''}
                    onChange={(e) => setWorld({
                      ...world,
                      locations: {
                        ...world.locations,
                        [locationName]: {
                          ...(locationData as any),
                          significance: e.target.value
                        }
                      }
                    })}
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-white focus:border-orange-500 focus:outline-none"
                    placeholder="Why is this location important?"
                  />
                </div>
              </div>
            ))}
          </div>
        );
    }
  };

  return (
    <div className="world-builder">
      <div className="container">
        {/* Header */}
        <div className="builder-header">
          <button 
            className="btn btn-ghost"
            onClick={() => navigate('/creator')}
          >
            ← Back to Dashboard
          </button>
          
          <button 
            onClick={handleSave}
            disabled={isSaving}
            className="btn btn-primary orange-glow"
          >
            <span>��</span>
            {isSaving ? 'Saving...' : 'Save World'}
          </button>
        </div>

        {/* Main Content */}
        <div className="builder-content">
          <h1 className="page-title">World Builder</h1>
          <p className="page-subtitle text-secondary">
            Create immersive worlds with rich lore and history
          </p>

          {/* Tab Navigation */}
          <div className="tab-nav">
            {(['overview', 'rules', 'history', 'cultures', 'locations'] as const).map((tab) => (
              <button
                key={tab}
                className={`tab-button ${activeTab === tab ? 'active' : ''}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="tab-container">
            {renderTabContent()}
          </div>
        </div>
      </div>

      <style>{`
        .world-builder {
          min-height: 100vh;
          padding: var(--space-8) 0;
        }

        .builder-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: var(--space-8);
        }

        .page-title {
          margin-bottom: var(--space-2);
        }

        .page-subtitle {
          margin-bottom: var(--space-8);
        }

        /* Tab Navigation */
        .tab-nav {
          display: flex;
          gap: var(--space-2);
          margin-bottom: var(--space-8);
          border-bottom: 1px solid var(--color-border-subtle);
          overflow-x: auto;
        }

        .tab-button {
          padding: var(--space-3) var(--space-6);
          background: none;
          border: none;
          color: var(--color-text-secondary);
          font-size: var(--text-sm);
          font-weight: var(--font-medium);
          cursor: pointer;
          position: relative;
          white-space: nowrap;
          transition: all var(--duration-fast) var(--ease-out);
        }

        .tab-button:hover {
          color: var(--color-text-primary);
        }

        .tab-button.active {
          color: var(--color-primary);
        }

        .tab-button.active::after {
          content: '';
          position: absolute;
          bottom: -1px;
          left: 0;
          right: 0;
          height: 2px;
          background: var(--color-primary);
        }

        .tab-container {
          min-height: 500px;
        }

        .section-title {
          margin-bottom: var(--space-2);
        }

        .section-description {
          margin-bottom: var(--space-6);
        }

        /* Rules Grid */
        .rules-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: var(--space-6);
        }

        .rule-card {
          padding: var(--space-6);
        }

        .rule-title {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          margin-bottom: var(--space-4);
        }

        .rule-icon {
          font-size: var(--text-2xl);
        }

        /* History */
        .history-input {
          font-family: var(--font-sans);
          line-height: 1.8;
        }

        .timeline-hint {
          margin-top: var(--space-6);
          padding: var(--space-4);
        }

        .hint-list {
          margin: var(--space-2) 0 0 var(--space-4);
          color: var(--color-text-secondary);
          font-size: var(--text-sm);
        }

        .hint-list li {
          margin-bottom: var(--space-1);
        }

        /* Cultures */
        .cultures-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-4);
          margin-top: var(--space-6);
        }

        .culture-card {
          padding: var(--space-4);
        }

        .culture-header {
          display: flex;
          gap: var(--space-2);
          margin-bottom: var(--space-4);
        }

        .culture-name {
          flex: 1;
          font-weight: var(--font-medium);
        }

        /* Locations */
        .locations-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
          gap: var(--space-6);
          margin-top: var(--space-6);
        }

        .location-card {
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
          padding: var(--space-4);
        }

        .location-header {
          display: flex;
          gap: var(--space-2);
        }

        .location-name {
          flex: 1;
          font-weight: var(--font-medium);
        }

        /* Responsive */
        @media (max-width: 768px) {
          .tab-nav {
            -webkit-overflow-scrolling: touch;
          }

          .rules-grid {
            grid-template-columns: 1fr;
          }

          .locations-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default WorldBuilder; 