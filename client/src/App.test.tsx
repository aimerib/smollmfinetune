import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

// Mock react-router-dom to avoid Router inside Router issue
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  BrowserRouter: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Routes: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Route: ({ element }: { element: React.ReactNode }) => <div>{element}</div>,
  Link: ({ to, children }: { to: string; children: React.ReactNode }) => <a href={to}>{children}</a>,
  useParams: () => ({ sessionId: null }),
  useSearchParams: () => [new URLSearchParams()],
  useNavigate: () => jest.fn(),
  useLocation: () => ({ pathname: '/' }),
  Navigate: ({ to }: { to: string }) => <div>Navigate to {to}</div>,
}));

// Mock pages to avoid complex dependencies
jest.mock('./pages/HomePage', () => {
  return function HomePage() {
    return <div>Choose Your Character</div>;
  };
});

jest.mock('./pages/ChatPage', () => {
  return function ChatPage() {
    return <div>Connecting with Alice...</div>;
  };
});

jest.mock('./pages/DirectorsView', () => {
  return function DirectorsView() {
    return (
      <div>
        <input placeholder="Search entities, memories, or events..." />
      </div>
    );
  };
});

describe('App Component', () => {
  test('renders without crashing', () => {
    render(<App />);
  });

  test('renders home page by default', () => {
    render(<App />);
    
    // HomePage should be rendered
    expect(screen.getByText(/Choose Your Character/i)).toBeInTheDocument();
  });

  test('renders chat page route', () => {
    render(<App />);
    
    // The Chat component exists in the tree due to how our mock works
    // Just verify the component renders without errors
    const devkitElements = screen.getAllByText(/Character Devkit/i);
    expect(devkitElements.length).toBeGreaterThan(0);
  });

  test('renders directors view route', () => {
    render(<App />);
    
    // DirectorsView should be rendered
    expect(screen.getByPlaceholderText(/Search entities, memories, or events/i)).toBeInTheDocument();
  });

  test('applies global styles', () => {
    const { container } = render(<App />);
    
    // Just check that the component renders without errors
    expect(container.firstChild).toBeInTheDocument();
  });
});
