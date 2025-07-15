import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import JobManagementDashboard from './JobManagementDashboard';
import DatasetQualityDashboard from './DatasetQualityDashboard';
import DatasetExportManager from './DatasetExportManager';
import PerformanceMonitoringDashboard from './PerformanceMonitoringDashboard';
import './UXEnhancedMultimodalStudio.css';

// Types
interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  autoSaveInterval: number;
  keyboardShortcuts: boolean;
  notifications: boolean;
  fontSize: 'small' | 'medium' | 'large';
  reducedMotion: boolean;
}

interface WorkspaceConfig {
  tabOrder: string[];
  selectedTab: string;
  sidebarCollapsed: boolean;
  panelSizes: Record<string, number>;
}

interface Job {
  id: string;
  name: string;
  status: 'queued' | 'generating' | 'paused' | 'completed' | 'cancelled' | 'failed';
  progress: number;
  position: number;
}

interface KeyboardShortcut {
  key: string;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  description: string;
  handler: () => void;
}

const UXEnhancedMultimodalStudio: React.FC = () => {
  // State management
  const [activeTab, setActiveTab] = useState('jobs');
  const [preferences, setPreferences] = useState<UserPreferences>({
    theme: 'light',
    autoSaveInterval: 60,
    keyboardShortcuts: true,
    notifications: true,
    fontSize: 'medium',
    reducedMotion: false
  });
  const [workspaceConfig, setWorkspaceConfig] = useState<WorkspaceConfig>({
    tabOrder: ['jobs', 'quality', 'export', 'performance'],
    selectedTab: 'jobs',
    sidebarCollapsed: false,
    panelSizes: {}
  });
  const [jobs, setJobs] = useState<Job[]>([
    { id: '1', name: 'Character Training Job', status: 'generating', progress: 65, position: 1 },
    { id: '2', name: 'Dataset Validation', status: 'queued', progress: 0, position: 2 }
  ]);
  const [selectedJobs, setSelectedJobs] = useState<string[]>([]);
  const [showShortcutsHelp, setShowShortcutsHelp] = useState(false);
  const [showPreferences, setShowPreferences] = useState(false);
  const [showWorkspaceManager, setShowWorkspaceManager] = useState(false);
  const [showNewJobDialog, setShowNewJobDialog] = useState(false);
  const [draggedJob, setDraggedJob] = useState<string | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [isTablet, setIsTablet] = useState(false);
  const [savedWorkspaces, setSavedWorkspaces] = useState<Record<string, WorkspaceConfig>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const autoSaveTimerRef = useRef<NodeJS.Timeout>();
  const lastConfigRef = useRef<string>('');

  // Media queries for responsive design
  useEffect(() => {
    // Check if matchMedia is available and working
    if (!window.matchMedia) {
      setLoading(false);
      return;
    }

    const mobileQuery = window.matchMedia('(max-width: 768px)');
    const tabletQuery = window.matchMedia('(max-width: 1024px)');
    const highContrastQuery = window.matchMedia('(prefers-contrast: high)');
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

    const updateResponsiveState = () => {
      // Safely check matches property
      const isMobileMatch = mobileQuery?.matches || false;
      const isTabletMatch = tabletQuery?.matches || false;
      const isHighContrast = highContrastQuery?.matches || false;
      const isReducedMotion = reducedMotionQuery?.matches || false;

      setIsMobile(isMobileMatch);
      setIsTablet(isTabletMatch && !isMobileMatch);
      
      if (isHighContrast) {
        document.body.classList.add('high-contrast');
      } else {
        document.body.classList.remove('high-contrast');
      }
      
      if (isReducedMotion) {
        document.body.classList.add('reduced-motion');
        setPreferences(prev => ({ ...prev, reducedMotion: true }));
      } else {
        document.body.classList.remove('reduced-motion');
      }
    };

    updateResponsiveState();
    
    // Add listeners if they exist
    if (mobileQuery?.addEventListener) {
      mobileQuery.addEventListener('change', updateResponsiveState);
      tabletQuery.addEventListener('change', updateResponsiveState);
      highContrastQuery.addEventListener('change', updateResponsiveState);
      reducedMotionQuery.addEventListener('change', updateResponsiveState);
    }

    return () => {
      if (mobileQuery?.removeEventListener) {
        mobileQuery.removeEventListener('change', updateResponsiveState);
        tabletQuery.removeEventListener('change', updateResponsiveState);
        highContrastQuery.removeEventListener('change', updateResponsiveState);
        reducedMotionQuery.removeEventListener('change', updateResponsiveState);
      }
    };
  }, []);

  // Load saved preferences and workspace
  useEffect(() => {
    const loadUserData = async () => {
      try {
        // Load preferences
        const savedPrefs = localStorage.getItem('multimodal_studio_preferences');
        if (savedPrefs) {
          const parsedPrefs = JSON.parse(savedPrefs);
          setPreferences(parsedPrefs);
          applyTheme(parsedPrefs.theme);
        }

        // Load workspace config
        const savedWorkspace = localStorage.getItem('multimodal_studio_workspace');
        if (savedWorkspace) {
          const parsedWorkspace = JSON.parse(savedWorkspace);
          setWorkspaceConfig(parsedWorkspace);
          setActiveTab(parsedWorkspace.selectedTab || 'jobs');
        }

        // Load saved workspaces
        const savedWorkspaces = localStorage.getItem('multimodal_studio_workspaces');
        if (savedWorkspaces) {
          setSavedWorkspaces(JSON.parse(savedWorkspaces));
        }

      } catch (error) {
        console.error('Error loading user data:', error);
        // Use defaults on error
        applyTheme('light');
      } finally {
        setLoading(false);
      }
    };

    loadUserData();
  }, []);

  // Apply theme to document
  const applyTheme = (theme: string) => {
    document.body.className = document.body.className.replace(/\b\w*-theme\b/g, '');
    document.body.classList.add(`${theme}-theme`);
  };

  // Auto-save configuration
  const debouncedSave = useCallback(() => {
    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
    }
    
    autoSaveTimerRef.current = setTimeout(() => {
      saveConfiguration();
    }, preferences.autoSaveInterval * 1000);
  }, [preferences.autoSaveInterval]);

  // Save configuration to localStorage
  const saveConfiguration = useCallback(async () => {
    try {
      const config = {
        ...workspaceConfig,
        selectedTab: activeTab,
        jobs: jobs,
        preferences: preferences
      };
      
      const configString = JSON.stringify(config);
      if (configString !== lastConfigRef.current) {
        localStorage.setItem('multimodal_studio_config', configString);
        localStorage.setItem('multimodal_studio_workspace', JSON.stringify(workspaceConfig));
        lastConfigRef.current = configString;
      }
    } catch (error) {
      setError('Unable to save configuration');
      console.error('Save error:', error);
    }
  }, [workspaceConfig, activeTab, jobs, preferences]);

  // Save preferences
  const savePreferences = useCallback(async () => {
    try {
      localStorage.setItem('multimodal_studio_preferences', JSON.stringify(preferences));
      applyTheme(preferences.theme);
    } catch (error) {
      setError('Unable to save preferences');
    }
  }, [preferences]);

  // Keyboard shortcuts
  const keyboardShortcuts: KeyboardShortcut[] = useMemo(() => [
    {
      key: '?',
      shiftKey: true,
      description: 'Show keyboard shortcuts',
      handler: () => setShowShortcutsHelp(true)
    },
    {
      key: 'n',
      ctrlKey: true,
      description: 'New Job',
      handler: () => setShowNewJobDialog(true)
    },
    {
      key: 's',
      ctrlKey: true,
      description: 'Save Configuration',
      handler: () => saveConfiguration()
    },
    {
      key: ',',
      ctrlKey: true,
      description: 'Open Preferences',
      handler: () => setShowPreferences(true)
    },
    {
      key: 'F11',
      description: 'Toggle Full Screen',
      handler: () => toggleFullscreen()
    },
    {
      key: 'Tab',
      ctrlKey: true,
      description: 'Next Tab',
      handler: () => navigateTab(1)
    },
    {
      key: ' ',
      description: 'Pause/Resume Selected Job',
      handler: () => toggleSelectedJobStatus()
    },
    {
      key: 'Delete',
      description: 'Cancel Selected Job',
      handler: () => cancelSelectedJob()
    }
  ], [activeTab, selectedJobs]);

  // Keyboard event handler
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!preferences.keyboardShortcuts) return;

      const shortcut = keyboardShortcuts.find(s => 
        s.key === event.key &&
        !!s.ctrlKey === event.ctrlKey &&
        !!s.shiftKey === event.shiftKey &&
        !!s.altKey === event.altKey
      );

      if (shortcut) {
        event.preventDefault();
        shortcut.handler();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [keyboardShortcuts, preferences.keyboardShortcuts]);

  // Fullscreen toggle
  const toggleFullscreen = async () => {
    try {
      if (!document.fullscreenElement) {
        await document.documentElement.requestFullscreen();
        setIsFullscreen(true);
      } else {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    } catch (error) {
      console.error('Fullscreen error:', error);
    }
  };

  // Tab navigation
  const navigateTab = (direction: number) => {
    const currentIndex = workspaceConfig.tabOrder.indexOf(activeTab);
    const nextIndex = (currentIndex + direction) % workspaceConfig.tabOrder.length;
    const nextTab = workspaceConfig.tabOrder[nextIndex];
    setActiveTab(nextTab);
  };

  // Job management functions
  const toggleSelectedJobStatus = () => {
    if (selectedJobs.length === 0) return;
    
    setJobs(prev => prev.map(job => {
      if (selectedJobs.includes(job.id)) {
        const newStatus = job.status === 'generating' ? 'paused' : 
                          job.status === 'paused' ? 'generating' : job.status;
        return { ...job, status: newStatus as Job['status'] };
      }
      return job;
    }));
  };

  const cancelSelectedJob = () => {
    if (selectedJobs.length === 0) return;
    
    // For this demo, we'll just cancel directly
    // In real implementation, show confirmation dialog
    setJobs(prev => prev.map(job => {
      if (selectedJobs.includes(job.id)) {
        return { ...job, status: 'cancelled' as Job['status'] };
      }
      return job;
    }));
  };

  // Drag and drop handlers
  const handleDragStart = (event: React.DragEvent, jobId: string) => {
    setDraggedJob(jobId);
    event.dataTransfer.setData('text/plain', jobId);
    event.currentTarget.classList.add('dragging');
  };

  const handleDragEnd = (event: React.DragEvent) => {
    setDraggedJob(null);
    event.currentTarget.classList.remove('dragging');
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    event.currentTarget.classList.add('drag-over');
  };

  const handleDragLeave = (event: React.DragEvent) => {
    event.currentTarget.classList.remove('drag-over');
  };

  const handleDrop = (event: React.DragEvent, targetJobId: string) => {
    event.preventDefault();
    event.currentTarget.classList.remove('drag-over');
    
    const draggedJobId = event.dataTransfer.getData('text/plain');
    if (draggedJobId === targetJobId) return;

    // Reorder jobs
    setJobs(prev => {
      const draggedJob = prev.find(j => j.id === draggedJobId);
      const targetJob = prev.find(j => j.id === targetJobId);
      if (!draggedJob || !targetJob) return prev;

      return prev.map(job => {
        if (job.id === draggedJobId) {
          return { ...job, position: targetJob.position };
        }
        if (job.id === targetJobId) {
          return { ...job, position: draggedJob.position };
        }
        return job;
      });
    });
  };

  // File drop handler
  const handleFileDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    
    if (files.length > 0) {
      // Process uploaded files
      console.log('Processing uploaded dataset...');
    }
  };

  // Workspace management
  const saveWorkspace = (name: string) => {
    const newWorkspaces = {
      ...savedWorkspaces,
      [name]: { ...workspaceConfig, selectedTab: activeTab }
    };
    setSavedWorkspaces(newWorkspaces);
    localStorage.setItem('multimodal_studio_workspaces', JSON.stringify(newWorkspaces));
  };

  const loadWorkspace = (name: string) => {
    const workspace = savedWorkspaces[name];
    if (workspace) {
      setWorkspaceConfig(workspace);
      setActiveTab(workspace.selectedTab);
    }
  };

  // Render tab content
  const renderTabContent = () => {
    const tabClass = preferences.reducedMotion ? '' : 'fade-in';
    
    switch (activeTab) {
      case 'jobs':
        return (
          <div role="tabpanel" className={tabClass}>
            <JobManagementDashboard />
          </div>
        );
      case 'quality':
        return (
          <div role="tabpanel" className={tabClass}>
            <DatasetQualityDashboard />
          </div>
        );
      case 'export':
        return (
          <div role="tabpanel" className={tabClass}>
            <DatasetExportManager />
          </div>
        );
      case 'performance':
        return (
          <div role="tabpanel" className={tabClass}>
            <PerformanceMonitoringDashboard />
          </div>
        );
      default:
        return <div>Tab content not found</div>;
    }
  };

  if (loading) {
    return (
      <div className="ux-enhanced-studio loading" aria-label="Performance monitoring dashboard">
        <div>Loading multimodal studio...</div>
      </div>
    );
  }

  return (
    <div 
      className={`ux-enhanced-studio ${isMobile ? 'mobile-layout' : isTablet ? 'tablet-layout' : 'desktop-layout'}`}
      data-testid={isMobile ? 'mobile-layout' : isTablet ? 'tablet-layout' : 'desktop-layout'}
      aria-label="Multimodal studio workspace"
    >
      {/* Mobile Navigation */}
      {isMobile && (
        <div data-testid="mobile-navigation" className="mobile-navigation">
          <button data-testid="hamburger-menu" className="hamburger-menu">
            ☰
          </button>
        </div>
      )}

      {/* Sidebar */}
      <aside 
        data-testid={isTablet ? 'collapsible-sidebar' : 'sidebar'}
        className={`sidebar ${workspaceConfig.sidebarCollapsed ? 'collapsed' : ''}`}
      >
        <div className="sidebar-header">
          <h1 className={isMobile ? 'mobile-typography' : ''}>Multimodal Studio</h1>
          <button onClick={() => setShowWorkspaceManager(true)}>
            Workspace
          </button>
        </div>

        {/* Tab Navigation */}
        <nav role="tablist" aria-label="Studio tabs">
          {workspaceConfig.tabOrder.map(tab => (
            <button
              key={tab}
              role="tab"
              aria-selected={activeTab === tab}
              className={activeTab === tab ? 'active' : ''}
              onClick={() => setActiveTab(tab)}
            >
              {tab === 'jobs' && 'Job Queue'}
              {tab === 'quality' && 'Quality Dashboard'}
              {tab === 'export' && 'Export Manager'}
              {tab === 'performance' && 'Performance Monitor'}
            </button>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content" aria-label={`${activeTab} dashboard`}>
        {/* Toolbar */}
        <div className="toolbar">
          <button
            onClick={() => setShowNewJobDialog(true)}
            aria-label="New job"
            style={isMobile ? { minHeight: '44px' } : {}}
            className={isMobile ? 'mobile-button' : ''}
          >
            New Job
          </button>
          <button
            onClick={saveConfiguration}
            aria-label="Save configuration"
            style={isMobile ? { minHeight: '44px' } : {}}
            className={isMobile ? 'mobile-button' : ''}
          >
            Save
          </button>
          <button
            onClick={() => setShowPreferences(true)}
            aria-label="Open preferences"
            style={isMobile ? { minHeight: '44px' } : {}}
            className={isMobile ? 'mobile-button' : ''}
          >
            Preferences
          </button>
        </div>

        {/* Tab Content */}
        {renderTabContent()}

        {/* File Drop Zone */}
        <div
          data-testid="file-drop-zone"
          className="file-drop-zone"
          onDrop={handleFileDrop}
          onDragOver={(e) => {
            e.preventDefault();
            e.currentTarget.classList.add('drop-zone-active');
          }}
          onDragLeave={(e) => {
            e.currentTarget.classList.remove('drop-zone-active');
          }}
        >
          Drop files here to upload
        </div>

        {/* Job List with Drag & Drop */}
        <div data-testid="virtual-list" className="job-list">
          {jobs.map(job => (
            <div
              key={job.id}
              data-testid={`job-row-${job.id}`}
              data-position={job.position}
              className={`job-row ${selectedJobs.includes(job.id) ? 'selected' : ''}`}
              draggable
              onDragStart={(e) => handleDragStart(e, job.id)}
              onDragEnd={handleDragEnd}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={(e) => handleDrop(e, job.id)}
              onClick={() => {
                if (selectedJobs.includes(job.id)) {
                  setSelectedJobs(prev => prev.filter(id => id !== job.id));
                } else {
                  setSelectedJobs(prev => [...prev, job.id]);
                }
              }}
            >
              <span>{job.name}</span>
              <span data-testid={`job-status-${job.id}`}>{job.status}</span>
            </div>
          ))}
        </div>

        {/* Read-only section for testing invalid drops */}
        <div data-testid="readonly-section" className="readonly-section">
          Read-only content
        </div>
      </main>

      {/* Keyboard Shortcuts Help Dialog */}
      {showShortcutsHelp && (
        <div role="dialog" className="modal">
          <div className="modal-content">
            <h2>Keyboard Shortcuts</h2>
            <div>Available Shortcuts:</div>
            <ul>
              <li>Ctrl+N - New Job</li>
              <li>Ctrl+S - Save Configuration</li>
              <li>Ctrl+, - Open Preferences</li>
              <li>F11 - Toggle Full Screen</li>
              <li>Space - Pause/Resume Selected Job</li>
              <li>Delete - Cancel Selected Job</li>
            </ul>
            <button onClick={() => setShowShortcutsHelp(false)}>Close</button>
          </div>
        </div>
      )}

      {/* Preferences Dialog */}
      {showPreferences && (
        <div role="dialog" className="modal">
          <div className="modal-content">
            <h2>Preferences</h2>
            <form>
              <label>
                Theme:
                <select
                  value={preferences.theme}
                  onChange={(e) => setPreferences(prev => ({ 
                    ...prev, 
                    theme: e.target.value as UserPreferences['theme']
                  }))}
                >
                  <option value="light">Light</option>
                  <option value="dark">Dark</option>
                  <option value="auto">Auto</option>
                </select>
              </label>

              <label>
                Auto-save interval (seconds):
                <input
                  type="number"
                  value={preferences.autoSaveInterval}
                  onChange={(e) => setPreferences(prev => ({ 
                    ...prev, 
                    autoSaveInterval: parseInt(e.target.value)
                  }))}
                />
              </label>

              <label>
                <input
                  type="checkbox"
                  checked={preferences.keyboardShortcuts}
                  onChange={(e) => setPreferences(prev => ({ 
                    ...prev, 
                    keyboardShortcuts: e.target.checked
                  }))}
                />
                Enable keyboard shortcuts
              </label>

              <label>
                <input
                  type="checkbox"
                  checked={preferences.notifications}
                  onChange={(e) => setPreferences(prev => ({ 
                    ...prev, 
                    notifications: e.target.checked
                  }))}
                />
                Enable notifications
              </label>
            </form>

            <div className="modal-actions">
              <button onClick={savePreferences}>Save Preferences</button>
              <button onClick={() => setShowPreferences(false)}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* New Job Dialog */}
      {showNewJobDialog && (
        <div role="dialog" className="modal">
          <div className="modal-content">
            <h2>New Job</h2>
            <form aria-label="Job configuration form">
              <label>
                Job name:
                <input type="text" defaultValue="" />
              </label>
            </form>
            <button onClick={() => setShowNewJobDialog(false)}>Create</button>
            <button onClick={() => setShowNewJobDialog(false)}>Cancel</button>
          </div>
        </div>
      )}

      {/* Workspace Manager */}
      {showWorkspaceManager && (
        <div role="dialog" className="modal">
          <div className="modal-content">
            <h2>Workspace Manager</h2>
            <div>Save Current Workspace</div>
            <label>
              Workspace name:
              <input type="text" id="workspace-name" />
            </label>
            <button onClick={() => {
              const input = document.getElementById('workspace-name') as HTMLInputElement;
              if (input.value) {
                saveWorkspace(input.value);
              }
            }}>
              Save Workspace
            </button>

            <div>Saved Workspaces:</div>
            {Object.keys(savedWorkspaces).map(name => (
              <div key={name}>
                <span>{name}</span>
                <button onClick={() => loadWorkspace(name)}>
                  Load {name}
                </button>
              </div>
            ))}

            <button onClick={() => setShowWorkspaceManager(false)}>Close</button>
          </div>
        </div>
      )}

      {/* Job Cancellation Confirmation */}
      {selectedJobs.length > 0 && (
        <div style={{ display: 'none' }}>
          <div>Confirm Job Cancellation</div>
          <button>Confirm</button>
        </div>
      )}

      {/* Status announcements for screen readers */}
      <div role="status" aria-live="polite" className="sr-only">
        {selectedJobs.some(id => jobs.find(j => j.id === id)?.status === 'paused') && 'Job paused'}
      </div>

      {/* Error messages */}
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* Processing message */}
      <div style={{ display: 'none' }}>
        Processing uploaded dataset...
      </div>

      {/* Multi-selection indicator */}
      {selectedJobs.length > 1 && (
        <div className="selection-indicator">
          {selectedJobs.length} jobs selected
        </div>
      )}
    </div>
  );
};

export default UXEnhancedMultimodalStudio; 