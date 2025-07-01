# Character Chat Client - React App

A beautiful, mobile-first React application for chatting with AI characters. Features real-time emotions, memory formation visualization, and stunning glassmorphism UI.

## 🌟 Features

- **Beautiful UI**: Glassmorphism design with animated backgrounds
- **Real-time Chat**: WebSocket-powered conversations with instant responses
- **Emotion Visualization**: See character emotions change in real-time
- **Memory Formation**: Watch as characters form memories during conversations
- **Mobile-First**: Responsive design that looks great on all devices
- **Character Selection**: Choose from multiple characters with unique personalities

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm
- The inference server running (see main project README)

### Installation

```bash
# From the project root
cd client
npm install
```

### Development

```bash
# Start the development server
npm start
```

The app will open at [http://localhost:3000](http://localhost:3000)

### Environment Variables

Create a `.env` file in the client directory:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

## 🏗️ Architecture

### Directory Structure
```
src/
├── components/       # Reusable UI components
│   ├── ChatMessage.tsx
│   ├── CharacterHeader.tsx
│   └── EmotionIndicator.tsx
├── pages/           # Route pages
│   ├── HomePage.tsx
│   └── ChatPage.tsx
├── utils/           # Services and utilities
│   └── chatService.ts
├── styles/          # Global styles
│   └── globals.css
└── App.tsx         # Main app component
```

### Key Technologies
- **React 18** with TypeScript
- **Framer Motion** for animations
- **Emotion/styled** for styling
- **Socket.io** for WebSocket communication
- **Axios** for REST API calls
- **React Router** for navigation

## 📱 Mobile Optimization

The client is optimized for mobile devices with:
- Touch-friendly interface
- Responsive layouts
- Efficient data usage
- WebSocket reconnection handling
- Progressive enhancement

## 🎨 Customization

### Adding New Characters

Edit `src/pages/HomePage.tsx`:

```typescript
const characters = [
  {
    id: 'your-character',
    name: 'Character Name',
    emoji: '🎭',
    description: 'Character description',
    gradient: 'linear-gradient(135deg, #color1 0%, #color2 100%)',
    stats: {
      conversations: 0,
      rating: 5.0,
      personality: 'Trait'
    }
  }
];
```

### Emotion Colors

Edit `src/components/EmotionIndicator.tsx`:

```typescript
const emotionColors: Record<string, string> = {
  happy: '#F6D365',
  sad: '#4FACFE',
  // Add more emotions
};
```

## 🔌 API Integration

The client connects to the inference engine via:
- REST API for messages and session management
- WebSocket for real-time updates
- Event system for emotion and memory updates

## 🚢 Production Build

```bash
# Create optimized production build
npm run build

# Serve the build locally
npx serve -s build
```

## 🔧 Troubleshooting

### WebSocket Connection Issues
- Ensure the inference server is running
- Check CORS settings on the server
- Verify environment variables are correct

### Performance Issues
- Enable React production mode
- Check network latency
- Monitor WebSocket reconnection attempts

## 🎯 Future Enhancements

- [ ] Voice input/output integration
- [ ] Character customization UI
- [ ] Conversation history
- [ ] Multi-character scenes
- [ ] Offline mode with character packets
- [ ] React Native mobile app

## 📄 License

Part of the Character Creation Platform - see main project license.
