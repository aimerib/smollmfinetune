# Director's View - Real-Time AI Character Monitoring

## 🎬 Overview

The Director's View is a revolutionary real-time monitoring and debugging console for AI character simulations. It provides a "god view" of your digital world, visualizing character states, memory formation, emotional changes, and triple-head model performance in an immersive 3D interface.

## 🏗️ Architecture

We've split from the monolithic Streamlit architecture to a modern FastAPI + React stack:

### Backend (FastAPI)
- **WebSocket Support**: Real-time bidirectional communication
- **Event-Driven Architecture**: Pub/sub pattern for decoupled services
- **RESTful APIs**: For state snapshots and data queries
- **Async Python**: High-performance concurrent operations

### Frontend (React + TypeScript)
- **Three.js**: Immersive 3D world visualization
- **Socket.io Client**: Real-time WebSocket connection
- **Zustand**: Lightweight state management
- **Material-UI**: Beautiful, responsive components
- **Plotly**: Advanced data visualization

## ✨ Features

### 🌍 3D World Visualization
- Interactive node-graph showing locations and characters
- Character sprites colored by personality traits
- Real-time position updates and movement animations
- Click-to-select entities for detailed information

### 💭 Memory Formation Bubbles
- Floating 3D bubbles appear when memories form
- Color-coded by emotional valence (red=negative, green=positive)
- Size indicates importance
- Emoji overlays for quick memory type recognition
- Fade animation based on memory persistence

### 🎭 Emotional State Tracking
- Real-time emotion bars with decay visualization
- Surprise score history with sparkline charts
- Emotional momentum indicators
- Active emotion tokens display

### 🧠 Triple-Head Architecture Monitoring
- **Generation Head**: Content quality, coherence scores
- **Control Head**: Emotional appropriateness, control token usage
- **Memory Head**: Formation rate, retrieval patterns
- **Cross-Head Coordination**: Visualization of head interactions

### 📜 Subtext Log
- Real-time internal monologue display
- Character thoughts and motivations
- Scrollable history with timestamps

### 📊 Memory Timeline
- Chronological view of formed memories
- Filter by memory type (episodic, semantic, emotional, procedural)
- Click to view full memory details

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
   - Navigate to http://localhost:3001
   - The 3D world will load with demo characters
   - Watch as characters move, form memories, and experience emotions!

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

1. **Navigate the World**: Use mouse to orbit, zoom, and pan the 3D scene
2. **Select Entities**: Click on characters or locations for details
3. **Monitor Memories**: Watch for floating bubbles when memories form
4. **Track Emotions**: Observe the emotion panel for state changes
5. **Read Subtext**: Follow character thoughts in the bottom panel
6. **Analyze Metrics**: Use the triple-head dashboard for model insights

## 🔮 Future Enhancements

- **VR Support**: Immersive virtual reality mode
- **Time Scrubbing**: Replay past events
- **Multi-User Collaboration**: Shared viewing sessions
- **Advanced Filtering**: Complex event queries
- **Performance Profiling**: Model inference metrics
- **Export Tools**: Save visualizations and reports

## 🤝 Contributing

This is a cutting-edge feature pushing the boundaries of AI character visualization. Contributions are welcome! Please follow the TDD methodology established in the codebase.

## 📜 License

Part of the Character Creation Platform - see main project license.

---

*"The Director's View transforms invisible AI processes into a living, breathing world you can see, understand, and direct."* 