# React Testing Guide

## Overview

This guide covers the testing practices and patterns for the React client application in the Character Creation Platform.

## Testing Stack

- **Jest**: Test runner and assertion library
- **React Testing Library**: Component testing utilities
- **TypeScript**: Type-safe tests
- **Mock Service Worker** (optional): API mocking

## Test Organization

```
client/src/__tests__/
├── components/          # Component unit tests
│   ├── ChatMessage.test.tsx
│   ├── CharacterHeader.test.tsx
│   └── EmotionIndicator.test.tsx
├── pages/              # Page integration tests
│   ├── HomePage.test.tsx
│   ├── ChatPage.test.tsx
│   └── DirectorsView.test.tsx
├── services/           # Service unit tests
│   └── websocketService.test.ts
├── hooks/              # Custom hook tests
└── utils/              # Utility function tests
```

## Running Tests

### Development Mode
```bash
# Run tests in watch mode
./scripts/run_react_tests.sh watch

# Or directly with npm
cd client && npm test
```

### CI Mode
```bash
# Run tests once with coverage
./scripts/run_react_tests.sh ci

# View coverage report
./scripts/run_react_tests.sh coverage
```

### Update Snapshots
```bash
./scripts/run_react_tests.sh update
```

## Writing Tests

### Component Testing Pattern

```typescript
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import MyComponent from '../MyComponent';

describe('MyComponent', () => {
  test('renders with default props', () => {
    render(<MyComponent />);
    expect(screen.getByText('Expected Text')).toBeInTheDocument();
  });

  test('handles user interaction', () => {
    const handleClick = jest.fn();
    render(<MyComponent onClick={handleClick} />);
    
    fireEvent.click(screen.getByRole('button'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  test('updates state correctly', async () => {
    render(<MyComponent />);
    
    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: 'New Value' }
    });
    
    await waitFor(() => {
      expect(screen.getByDisplayValue('New Value')).toBeInTheDocument();
    });
  });
});
```

### Service Testing Pattern

```typescript
import { myService } from '../myService';
import axios from 'axios';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('MyService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('fetches data successfully', async () => {
    const mockData = { id: 1, name: 'Test' };
    mockedAxios.get.mockResolvedValue({ data: mockData });

    const result = await myService.getData();
    
    expect(result).toEqual(mockData);
    expect(mockedAxios.get).toHaveBeenCalledWith('/api/data');
  });

  test('handles errors gracefully', async () => {
    mockedAxios.get.mockRejectedValue(new Error('Network error'));

    await expect(myService.getData()).rejects.toThrow('Network error');
  });
});
```

### WebSocket Testing Pattern

```typescript
class MockWebSocket {
  url: string;
  readyState: number = WebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  
  constructor(url: string) {
    this.url = url;
  }
  
  simulateOpen() {
    this.readyState = WebSocket.OPEN;
    if (this.onopen) {
      this.onopen(new Event('open'));
    }
  }
  
  simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { 
        data: JSON.stringify(data) 
      }));
    }
  }
  
  send(data: string) {
    // Mock implementation
  }
  
  close() {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }
}

(global as any).WebSocket = MockWebSocket;
```

## Best Practices

### 1. Query Priority
Use queries in this order of preference:
1. `getByRole` - Accessible, semantic
2. `getByLabelText` - Form elements
3. `getByPlaceholderText` - When no label
4. `getByText` - Non-interactive elements
5. `getByTestId` - Last resort

### 2. Async Testing
```typescript
// Wait for elements to appear
await waitFor(() => {
  expect(screen.getByText('Loaded')).toBeInTheDocument();
});

// Find elements that appear async
const element = await screen.findByText('Async Text');

// Wait for elements to disappear
await waitForElementToBeRemoved(() => screen.queryByText('Loading'));
```

### 3. User Events
```typescript
import userEvent from '@testing-library/user-event';

test('types in input field', async () => {
  const user = userEvent.setup();
  render(<MyForm />);
  
  await user.type(screen.getByRole('textbox'), 'Hello World');
  await user.click(screen.getByRole('button', { name: 'Submit' }));
  
  expect(screen.getByText('Form submitted')).toBeInTheDocument();
});
```

### 4. Mocking

#### Mock Modules
```typescript
jest.mock('../services/api', () => ({
  fetchData: jest.fn(),
  postData: jest.fn()
}));
```

#### Mock Components
```typescript
jest.mock('../components/ExpensiveComponent', () => {
  return {
    __esModule: true,
    default: () => <div>Mocked Component</div>
  };
});
```

#### Mock Hooks
```typescript
jest.mock('../hooks/useAuth', () => ({
  useAuth: () => ({
    user: { id: '123', name: 'Test User' },
    isAuthenticated: true,
    login: jest.fn(),
    logout: jest.fn()
  })
}));
```

### 5. Testing Router
```typescript
import { MemoryRouter } from 'react-router-dom';

test('navigates to different routes', () => {
  render(
    <MemoryRouter initialEntries={['/dashboard']}>
      <App />
    </MemoryRouter>
  );
  
  expect(screen.getByText('Dashboard')).toBeInTheDocument();
});
```

### 6. Testing Context
```typescript
const MockedProvider = ({ children }) => (
  <ThemeContext.Provider value={{ theme: 'dark' }}>
    {children}
  </ThemeContext.Provider>
);

test('uses context value', () => {
  render(
    <MockedProvider>
      <ThemedComponent />
    </MockedProvider>
  );
  
  expect(screen.getByTestId('container')).toHaveClass('dark-theme');
});
```

## Common Testing Scenarios

### Form Submission
```typescript
test('submits form with validation', async () => {
  const handleSubmit = jest.fn();
  render(<ContactForm onSubmit={handleSubmit} />);
  
  // Fill form
  fireEvent.change(screen.getByLabelText('Name'), {
    target: { value: 'John Doe' }
  });
  fireEvent.change(screen.getByLabelText('Email'), {
    target: { value: 'john@example.com' }
  });
  
  // Submit
  fireEvent.click(screen.getByRole('button', { name: 'Submit' }));
  
  await waitFor(() => {
    expect(handleSubmit).toHaveBeenCalledWith({
      name: 'John Doe',
      email: 'john@example.com'
    });
  });
});
```

### Error States
```typescript
test('displays error message on failure', async () => {
  const error = new Error('Failed to load');
  mockedService.getData.mockRejectedValue(error);
  
  render(<DataDisplay />);
  
  await waitFor(() => {
    expect(screen.getByText(/failed to load/i)).toBeInTheDocument();
  });
});
```

### Loading States
```typescript
test('shows loading indicator while fetching', async () => {
  render(<AsyncComponent />);
  
  // Loading state
  expect(screen.getByText(/loading/i)).toBeInTheDocument();
  
  // Wait for content
  await waitFor(() => {
    expect(screen.queryByText(/loading/i)).not.toBeInTheDocument();
    expect(screen.getByText('Content loaded')).toBeInTheDocument();
  });
});
```

## Coverage Goals

- **Statements**: 80%+
- **Branches**: 75%+
- **Functions**: 80%+
- **Lines**: 80%+

Focus coverage on:
- Business logic
- User interactions
- Error handling
- Edge cases

## Debugging Tests

### Debug Output
```typescript
// Print the DOM
screen.debug();

// Print specific element
screen.debug(screen.getByRole('button'));

// Use prettyDOM for better formatting
import { prettyDOM } from '@testing-library/react';
console.log(prettyDOM(container));
```

### Common Issues

1. **Element not found**: Check if element is rendered conditionally or async
2. **Multiple elements found**: Use more specific queries or `getAllBy*`
3. **State not updating**: Wrap in `act()` or use `waitFor()`
4. **Timer issues**: Use `jest.useFakeTimers()` and `jest.runAllTimers()`

## Integration with CI/CD

The React tests are integrated into the platform CI pipeline:

```yaml
# GitHub Actions example
- name: Run React Tests
  run: |
    cd client
    npm ci
    npm test -- --coverage --watchAll=false
```

## Resources

- [React Testing Library Docs](https://testing-library.com/docs/react-testing-library/intro/)
- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Testing Best Practices](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library) 