import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import UXEnhancedMultimodalStudio from '../../components/UXEnhancedMultimodalStudio';

// Mock dependencies
jest.mock('../../services/multimodalService');
jest.mock('../../services/websocketService');

// Mock localStorage
const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};

Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage
});

// Mock drag and drop API
const mockDataTransfer = {
  getData: jest.fn(),
  setData: jest.fn(),
  clearData: jest.fn(),
  dropEffect: 'none',
  effectAllowed: 'all',
  files: [],
  items: [],
  types: [],
};

// Mock matchMedia for responsive design tests
const mockMatchMedia = jest.fn((query) => ({
  matches: false,
  media: query,
  onchange: null,
  addListener: jest.fn(),
  removeListener: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  dispatchEvent: jest.fn(),
}));

// Store original matchMedia
const originalMatchMedia = window.matchMedia;

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: mockMatchMedia,
});

describe('UXEnhancedMultimodalStudio', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockLocalStorage.getItem.mockReturnValue(null);
    
    // Reset matchMedia mock to default behavior
    mockMatchMedia.mockImplementation((query) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: jest.fn(),
      removeListener: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));
  });

  afterAll(() => {
    // Restore original matchMedia if it existed
    if (originalMatchMedia) {
      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: originalMatchMedia,
      });
    }
  });

  describe('Keyboard Shortcuts', () => {
    test('renders keyboard shortcuts help dialog when ? is pressed', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      fireEvent.keyDown(document, { key: '?', shiftKey: true });
      
      await waitFor(() => {
        expect(screen.getByText('Keyboard Shortcuts')).toBeInTheDocument();
        expect(screen.getByText('Ctrl+N')).toBeInTheDocument();
        expect(screen.getByText('New Job')).toBeInTheDocument();
        expect(screen.getByText('Ctrl+S')).toBeInTheDocument();
        expect(screen.getByText('Save Configuration')).toBeInTheDocument();
      });
    });

    test('creates new job with Ctrl+N shortcut', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      fireEvent.keyDown(document, { key: 'n', ctrlKey: true });
      
      await waitFor(() => {
        expect(screen.getByText('New Job')).toBeInTheDocument();
        expect(screen.getByLabelText('Job configuration form')).toBeInTheDocument();
      });
    });

    test('saves configuration with Ctrl+S shortcut', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      fireEvent.keyDown(document, { key: 's', ctrlKey: true });
      
      await waitFor(() => {
        expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
          'multimodal_studio_config',
          expect.any(String)
        );
      });
    });

    test('opens preferences with Ctrl+, shortcut', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      fireEvent.keyDown(document, { key: ',', ctrlKey: true });
      
      await waitFor(() => {
        expect(screen.getByText('Preferences')).toBeInTheDocument();
        expect(screen.getByLabelText('Auto-save interval')).toBeInTheDocument();
        expect(screen.getByLabelText('Theme')).toBeInTheDocument();
      });
    });

    test('toggles full screen with F11 shortcut', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      const fullscreenSpy = jest.spyOn(document.documentElement, 'requestFullscreen').mockImplementation(() => Promise.resolve());
      
      fireEvent.keyDown(document, { key: 'F11' });
      
      await waitFor(() => {
        expect(fullscreenSpy).toHaveBeenCalled();
      });
      
      fullscreenSpy.mockRestore();
    });

    test('navigates between tabs with Ctrl+Tab shortcuts', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      // Should start on first tab
      expect(screen.getByRole('tab', { selected: true })).toHaveTextContent('Job Queue');
      
      fireEvent.keyDown(document, { key: 'Tab', ctrlKey: true });
      
      await waitFor(() => {
        expect(screen.getByRole('tab', { selected: true })).toHaveTextContent('Quality Dashboard');
      });
    });

    test('quickly pauses/resumes jobs with spacebar', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      // Select a job first
      const jobRow = await screen.findByTestId('job-row-1');
      await userEvent.click(jobRow);
      
      fireEvent.keyDown(document, { key: ' ' });
      
      await waitFor(() => {
        expect(screen.getByTestId('job-status-1')).toHaveTextContent('paused');
      });
    });

    test('supports keyboard shortcuts help overlay', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      fireEvent.keyDown(document, { key: '?', shiftKey: true });
      
      await waitFor(() => {
        expect(screen.getByRole('dialog')).toBeInTheDocument();
        expect(screen.getByText('Available Shortcuts:')).toBeInTheDocument();
        expect(screen.getByText('Ctrl+N - New Job')).toBeInTheDocument();
        expect(screen.getByText('Ctrl+S - Save Configuration')).toBeInTheDocument();
        expect(screen.getByText('Ctrl+, - Open Preferences')).toBeInTheDocument();
        expect(screen.getByText('F11 - Toggle Full Screen')).toBeInTheDocument();
        expect(screen.getByText('Space - Pause/Resume Selected Job')).toBeInTheDocument();
        expect(screen.getByText('Delete - Cancel Selected Job')).toBeInTheDocument();
      });
    });

    test('cancels selected job with Delete key', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      const jobRow = await screen.findByTestId('job-row-1');
      await userEvent.click(jobRow);
      
      fireEvent.keyDown(document, { key: 'Delete' });
      
      await waitFor(() => {
        expect(screen.getByText('Confirm Job Cancellation')).toBeInTheDocument();
      });
      
      const confirmButton = screen.getByRole('button', { name: /confirm/i });
      await userEvent.click(confirmButton);
      
      await waitFor(() => {
        expect(screen.getByTestId('job-status-1')).toHaveTextContent('cancelled');
      });
    });
  });

  describe('Drag and Drop Interface', () => {
    test('allows dragging jobs to reorder queue', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      const jobRow1 = await screen.findByTestId('job-row-1');
      const jobRow2 = await screen.findByTestId('job-row-2');
      
      // Start drag
      fireEvent.dragStart(jobRow1, {
        dataTransfer: mockDataTransfer
      });
      
      expect(mockDataTransfer.setData).toHaveBeenCalledWith('text/plain', '1');
      expect(jobRow1).toHaveClass('dragging');
      
      // Drag over another job
      fireEvent.dragOver(jobRow2, {
        dataTransfer: mockDataTransfer
      });
      
      expect(jobRow2).toHaveClass('drag-over');
      
      // Drop
      mockDataTransfer.getData.mockReturnValue('1');
      fireEvent.drop(jobRow2, {
        dataTransfer: mockDataTransfer
      });
      
      await waitFor(() => {
        expect(screen.getByTestId('job-row-1')).toHaveAttribute('data-position', '2');
        expect(screen.getByTestId('job-row-2')).toHaveAttribute('data-position', '1');
      });
    });

    test('supports file drop for dataset upload', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      const dropZone = screen.getByTestId('file-drop-zone');
      const file = new File(['dataset content'], 'dataset.jsonl', { type: 'application/json' });
      
      fireEvent.dragEnter(dropZone, {
        dataTransfer: {
          files: [file],
          types: ['Files']
        }
      });
      
      expect(dropZone).toHaveClass('drop-zone-active');
      
      fireEvent.drop(dropZone, {
        dataTransfer: {
          files: [file]
        }
      });
      
      await waitFor(() => {
        expect(screen.getByText('Processing uploaded dataset...')).toBeInTheDocument();
      });
    });

    test('provides visual feedback during drag operations', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      const jobRow = await screen.findByTestId('job-row-1');
      
      fireEvent.dragStart(jobRow);
      expect(jobRow).toHaveClass('dragging');
      
      fireEvent.dragEnd(jobRow);
      expect(jobRow).not.toHaveClass('dragging');
    });

    test('prevents invalid drop operations', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      const jobRow = await screen.findByTestId('job-row-1');
      const invalidDropTarget = screen.getByTestId('readonly-section');
      
      fireEvent.dragStart(jobRow);
      fireEvent.dragOver(invalidDropTarget);
      
      expect(invalidDropTarget).toHaveClass('drop-invalid');
    });

    test('supports dragging multiple selected jobs', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      const job1 = await screen.findByTestId('job-row-1');
      const job2 = await screen.findByTestId('job-row-2');
      
      // Multi-select jobs
      await userEvent.click(job1);
      await userEvent.click(job2, { ctrlKey: true });
      
      expect(job1).toHaveClass('selected');
      expect(job2).toHaveClass('selected');
      
      // Drag multiple
      fireEvent.dragStart(job1);
      
      await waitFor(() => {
        expect(screen.getByText('2 jobs selected')).toBeInTheDocument();
      });
    });
  });

  describe('Responsive Design', () => {
    test('adapts layout for mobile devices', async () => {
      mockMatchMedia.mockImplementation((query) => ({
        matches: query === '(max-width: 768px)',
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      }));
      
      render(<UXEnhancedMultimodalStudio />);
      
      expect(screen.getByTestId('mobile-navigation')).toBeInTheDocument();
      expect(screen.getByTestId('hamburger-menu')).toBeInTheDocument();
      expect(screen.queryByTestId('desktop-sidebar')).not.toBeInTheDocument();
    });

    test('shows tablet layout for medium screens', async () => {
      mockMatchMedia.mockImplementation((query) => ({
        matches: query === '(max-width: 1024px)' && !query.includes('768px'),
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      }));
      
      render(<UXEnhancedMultimodalStudio />);
      
      expect(screen.getByTestId('tablet-layout')).toBeInTheDocument();
      expect(screen.getByTestId('collapsible-sidebar')).toBeInTheDocument();
    });

    test('provides touch-friendly controls on mobile', async () => {
      mockMatchMedia.mockImplementation((query) => ({
        matches: query === '(max-width: 768px)',
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      }));
      
      render(<UXEnhancedMultimodalStudio />);
      
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toHaveStyle('min-height: 44px'); // Touch target size
      });
    });

    test('adjusts font sizes for readability', async () => {
      mockMatchMedia.mockImplementation((query) => ({
        matches: query === '(max-width: 768px)',
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      }));
      
      render(<UXEnhancedMultimodalStudio />);
      
      const headings = screen.getAllByRole('heading');
      headings.forEach(heading => {
        expect(heading).toHaveClass('mobile-typography');
      });
    });
  });

  describe('User Preferences Persistence', () => {
    test('saves user preferences to localStorage', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      // Open preferences
      fireEvent.keyDown(document, { key: ',', ctrlKey: true });
      
      await waitFor(() => {
        expect(screen.getByText('Preferences')).toBeInTheDocument();
      });
      
      // Change theme
      const themeSelect = screen.getByLabelText('Theme');
      await userEvent.selectOptions(themeSelect, 'dark');
      
      // Change auto-save interval
      const autoSaveInput = screen.getByLabelText('Auto-save interval');
      await userEvent.clear(autoSaveInput);
      await userEvent.type(autoSaveInput, '30');
      
      // Save preferences
      const saveButton = screen.getByRole('button', { name: /save preferences/i });
      await userEvent.click(saveButton);
      
      expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
        'multimodal_studio_preferences',
        JSON.stringify({
          theme: 'dark',
          autoSaveInterval: 30,
          keyboardShortcuts: true,
          notifications: true
        })
      );
    });

    test('loads user preferences on component mount', async () => {
      mockLocalStorage.getItem.mockReturnValue(JSON.stringify({
        theme: 'dark',
        autoSaveInterval: 60,
        keyboardShortcuts: false,
        notifications: false
      }));
      
      render(<UXEnhancedMultimodalStudio />);
      
      expect(document.body).toHaveClass('dark-theme');
      
      // Open preferences to verify values
      fireEvent.keyDown(document, { key: ',', ctrlKey: true });
      
      await waitFor(() => {
        expect(screen.getByLabelText('Theme')).toHaveValue('dark');
        expect(screen.getByLabelText('Auto-save interval')).toHaveValue(60);
        expect(screen.getByLabelText('Enable keyboard shortcuts')).not.toBeChecked();
        expect(screen.getByLabelText('Enable notifications')).not.toBeChecked();
      });
    });

    test('applies theme changes immediately', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      fireEvent.keyDown(document, { key: ',', ctrlKey: true });
      
      const themeSelect = await screen.findByLabelText('Theme');
      await userEvent.selectOptions(themeSelect, 'dark');
      
      expect(document.body).toHaveClass('dark-theme');
    });

    test('respects keyboard shortcuts preference', async () => {
      mockLocalStorage.getItem.mockReturnValue(JSON.stringify({
        keyboardShortcuts: false
      }));
      
      render(<UXEnhancedMultimodalStudio />);
      
      // Try keyboard shortcut that should be disabled
      fireEvent.keyDown(document, { key: 'n', ctrlKey: true });
      
      await waitFor(() => {
        expect(screen.queryByText('New Job')).not.toBeInTheDocument();
      });
    });
  });

  describe('Workspace Management', () => {
    test('saves workspace configuration', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      // Configure some workspace settings
      const tabOrder = ['quality', 'jobs', 'export'];
      
      fireEvent.keyDown(document, { key: 's', ctrlKey: true });
      
      await waitFor(() => {
        expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
          'multimodal_studio_workspace',
          expect.stringContaining('tabOrder')
        );
      });
    });

    test('restores workspace configuration on load', async () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'multimodal_studio_workspace') {
          return JSON.stringify({
            tabOrder: ['export', 'quality', 'jobs'],
            selectedTab: 'quality',
            sidebarCollapsed: true
          });
        }
        return null;
      });
      
      render(<UXEnhancedMultimodalStudio />);
      
      await waitFor(() => {
        expect(screen.getByRole('tab', { selected: true })).toHaveTextContent('Quality Dashboard');
        expect(screen.getByTestId('sidebar')).toHaveClass('collapsed');
      });
    });

    test('allows creating named workspace presets', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      // Open workspace manager
      const workspaceButton = screen.getByRole('button', { name: /workspace/i });
      await userEvent.click(workspaceButton);
      
      expect(screen.getByText('Save Current Workspace')).toBeInTheDocument();
      
      const nameInput = screen.getByLabelText('Workspace name');
      await userEvent.type(nameInput, 'Quality Review Setup');
      
      const saveWorkspaceButton = screen.getByRole('button', { name: /save workspace/i });
      await userEvent.click(saveWorkspaceButton);
      
      await waitFor(() => {
        expect(mockLocalStorage.setItem).toHaveBeenCalledWith(
          'multimodal_studio_workspaces',
          expect.stringContaining('Quality Review Setup')
        );
      });
    });

    test('loads named workspace presets', async () => {
      mockLocalStorage.getItem.mockImplementation((key) => {
        if (key === 'multimodal_studio_workspaces') {
          return JSON.stringify({
            'Quality Review Setup': {
              tabOrder: ['quality', 'jobs'],
              selectedTab: 'quality',
              sidebarCollapsed: false
            }
          });
        }
        return null;
      });
      
      render(<UXEnhancedMultimodalStudio />);
      
      const workspaceButton = screen.getByRole('button', { name: /workspace/i });
      await userEvent.click(workspaceButton);
      
      expect(screen.getByText('Quality Review Setup')).toBeInTheDocument();
      
      const loadButton = screen.getByRole('button', { name: /load quality review setup/i });
      await userEvent.click(loadButton);
      
      await waitFor(() => {
        expect(screen.getByRole('tab', { selected: true })).toHaveTextContent('Quality Dashboard');
      });
    });
  });

  describe('Performance Optimization', () => {
    test('debounces auto-save operations', async () => {
      jest.useFakeTimers();
      
      render(<UXEnhancedMultimodalStudio />);
      
      // Make multiple rapid changes
      const input = screen.getByLabelText('Job name');
      await userEvent.type(input, 'test');
      
      // Fast forward through debounce period
      jest.advanceTimersByTime(1000);
      
      expect(mockLocalStorage.setItem).toHaveBeenCalledTimes(1);
      
      jest.useRealTimers();
    });

    test('virtualizes large job lists for performance', async () => {
      // Mock large dataset
      const largeJobList = Array.from({ length: 1000 }, (_, i) => ({
        id: i + 1,
        name: `Job ${i + 1}`,
        status: 'queued'
      }));
      
      render(<UXEnhancedMultimodalStudio />);
      
      // Should only render visible items
      expect(screen.getAllByTestId(/job-row/)).toHaveLength(10); // Assuming 10 visible items
      expect(screen.getByTestId('virtual-list')).toBeInTheDocument();
    });

    test('provides smooth animations and transitions', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      const tabButton = screen.getByRole('tab', { name: /quality dashboard/i });
      await userEvent.click(tabButton);
      
      const tabPanel = screen.getByRole('tabpanel');
      expect(tabPanel).toHaveClass('fade-in');
    });
  });

  describe('Accessibility Enhancements', () => {
    test('provides keyboard navigation for all interactive elements', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      // Tab through interface
      await userEvent.tab();
      expect(screen.getByRole('button', { name: /new job/i })).toHaveFocus();
      
      await userEvent.tab();
      expect(screen.getByRole('button', { name: /save/i })).toHaveFocus();
    });

    test('announces status changes to screen readers', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      const jobRow = await screen.findByTestId('job-row-1');
      await userEvent.click(jobRow);
      
      fireEvent.keyDown(document, { key: ' ' });
      
      await waitFor(() => {
        expect(screen.getByRole('status')).toHaveTextContent('Job paused');
      });
    });

    test('provides proper ARIA labels and descriptions', async () => {
      render(<UXEnhancedMultimodalStudio />);
      
      expect(screen.getByLabelText('Multimodal studio workspace')).toBeInTheDocument();
      expect(screen.getByLabelText('Job queue management')).toBeInTheDocument();
      expect(screen.getByLabelText('Quality dashboard')).toBeInTheDocument();
    });

    test('supports high contrast mode', async () => {
      mockMatchMedia.mockImplementation((query) => ({
        matches: query === '(prefers-contrast: high)',
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      }));
      
      render(<UXEnhancedMultimodalStudio />);
      
      expect(document.body).toHaveClass('high-contrast');
    });

    test('respects reduced motion preference', async () => {
      mockMatchMedia.mockImplementation((query) => ({
        matches: query === '(prefers-reduced-motion: reduce)',
        media: query,
        onchange: null,
        addListener: jest.fn(),
        removeListener: jest.fn(),
        addEventListener: jest.fn(),
        removeEventListener: jest.fn(),
        dispatchEvent: jest.fn(),
      }));
      
      render(<UXEnhancedMultimodalStudio />);
      
      expect(document.body).toHaveClass('reduced-motion');
    });
  });

  describe('Error Handling', () => {
    test('gracefully handles localStorage errors', async () => {
      mockLocalStorage.setItem.mockImplementation(() => {
        throw new Error('Storage quota exceeded');
      });
      
      render(<UXEnhancedMultimodalStudio />);
      
      fireEvent.keyDown(document, { key: 's', ctrlKey: true });
      
      await waitFor(() => {
        expect(screen.getByText('Unable to save configuration')).toBeInTheDocument();
      });
    });

    test('provides fallback when preferences cannot be loaded', async () => {
      mockLocalStorage.getItem.mockImplementation(() => {
        throw new Error('Corrupted data');
      });
      
      render(<UXEnhancedMultimodalStudio />);
      
      // Should use default preferences
      expect(document.body).toHaveClass('light-theme');
    });
  });
}); 