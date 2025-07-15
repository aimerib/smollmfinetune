# Director's View - Real-Time AI Character Monitoring

## 🎬 Overview

The Director's View is a revolutionary real-time monitoring and debugging console for AI character simulations. It provides a "god view" of your digital world, visualizing character states, memory formation, emotional changes, and triple-head model performance in an elegant 2D interface.

## 🏗️ Architecture

We've split from the monolithic Streamlit architecture to a modern FastAPI + React stack:

### Backend (FastAPI)
- **WebSocket Support**: Real-time bidirectional communication
- **Event-Driven Architecture**: Pub/sub pattern for decoupled services
- **RESTful APIs**: For state snapshots and data queries
- **Async Python**: High-performance concurrent operations

### Frontend (React + TypeScript)
- **Clean 2D Visualization**: Elegant, performant 2D world representation
- **WebSocket Client**: Real-time updates without polling
- **Zustand**: Lightweight state management
- **Emotion/styled-components**: Beautiful, themeable UI
- **Framer Motion**: Smooth, meaningful animations

## ✨ Features

### 🌍 2D World Visualization
- Clean circular layout showing locations and characters
- Character indicators with personality-based styling
- Real-time position updates with smooth transitions
- Click-to-select for detailed information
- Activity indicators for ongoing interactions

### 💭 Memory Formation Display
- Floating memory bubbles that rise from characters
- Color-coded by emotional valence (warm/cool gradients)
- Size indicates importance
- Fade animation based on memory persistence
- Memory count tracking per character

### 🎭 Emotional State Tracking
- Real-time emotion indicators with visual feedback
- Current mood display in context panel
- Emotional transitions during conversations
- Personality trait visualization

### 🧠 Triple-Head Architecture Monitoring
- **Generation Head**: Content quality, coherence scores
- **Control Head**: Emotional appropriateness, control token usage
- **Memory Head**: Formation rate, retrieval patterns
- **Cross-Head Coordination**: Visualization of head interactions

### 📜 Subtext Log
- Real-time internal monologue display
- Character thoughts and decision-making process
- Timestamp tracking for forensic analysis

### ⌨️ Keyboard Shortcuts
- `Space`: Play/Pause simulation
- `M`: Toggle memory display
- `E`: Toggle emotion display
- `T`: Toggle metrics
- `C`: Toggle connection lines
- `Z/X`: Zoom in/out
- `?`: Show keyboard shortcuts

## 🚧 Current Integration Status

**Important**: The Director's View currently operates with simulated data for demonstration purposes. Full integration with actual character inference is in progress.

### What's Working:
- ✅ Beautiful, responsive UI
- ✅ WebSocket connection infrastructure
- ✅ Event-driven architecture
- ✅ State management and updates
- ✅ Smooth animations and transitions

### What's Pending:
- ⏳ Connection to real character inference
- ⏳ Actual memory formation from narrative engine
- ⏳ Real emotion state from control tokens
- ⏳ Live triple-head metrics
- ⏳ Integration with trained models

See our [Integration Guide](./character-docs/docs/directors-view-integration.md) for detailed plans.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Quick Start

1. **Clone and navigate to the project**
   ```bash
   cd /path/to/character-creation-platform
   ```

2. **Run the Director's View**
   ```bash
   ./start-directors-view.sh
   ```

   This script will:
   - Create Python virtual environment
   - Install FastAPI dependencies
   - Start the API backend on http://localhost:8000
   - Install React dependencies
   - Start the React frontend on http://localhost:3001
   - Begin world simulation with demo characters

3. **Open your browser**
   - Navigate to http://localhost:3001/directors-view
   - The 2D world will load with demo characters
   - Watch as simulated events demonstrate the interface!

### Manual Setup

If you prefer to run services separately:

**Backend:**
```bash
cd api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd client
npm install
npm start
```

## 🔌 API Endpoints

### REST Endpoints
- `GET /api/state/snapshot` - Get complete world state
- `GET /api/state/entities/{id}` - Get entity details
- `GET /api/memories/{character_id}` - Get character memories
- `GET /api/emotions/{character_id}` - Get emotional state

### WebSocket
- `ws://localhost:8000/ws/director` - Main WebSocket for real-time updates

## 🧪 Testing

**Backend Tests:**
```bash
cd api
pytest tests/
```

**Frontend Tests:**
```bash
cd client
npm test
```

## 📁 Project Structure

```
api/
├── app/
│   ├── main.py              # FastAPI application
│   ├── routers/             # REST endpoints
│   ├── websocket/           # WebSocket handlers
│   ├── services/            # Business logic
│   └── models/              # Pydantic models
└── tests/                   # API tests

client/
├── src/
│   ├── pages/               # React pages
│   ├── components/          # Reusable components
│   ├── services/            # API clients
│   └── stores/              # Zustand stores
└── public/                  # Static assets
```

## 🎮 Using the Director's View

1. **Navigate the World**: Click and drag to pan, use zoom controls or scroll
2. **Select Entities**: Click on characters for detailed information
3. **Monitor Activity**: Watch for activity indicators during interactions
4. **Track State**: Observe the context panel for character details
5. **Toggle Filters**: Use top bar to show/hide different data layers
6. **Control Playback**: Use timeline controls to pause/resume

## 🔮 Future Enhancements

- **Real Model Integration**: Connect to actual character inference
- **Time Scrubbing**: Replay past conversations
- **Multi-Character Orchestration**: Watch characters interact
- **Advanced Filtering**: Complex event queries
- **Performance Profiling**: Model inference metrics
- **Export Tools**: Save visualizations and reports

## 🤝 Contributing

This is a cutting-edge feature pushing the boundaries of AI character visualization. Contributions are welcome! Please follow the TDD methodology established in the codebase.

## 📜 License

Part of the Character Creation Platform - see main project license.

---

*"The Director's View transforms invisible AI processes into a living, breathing world you can see, understand, and direct."* 