# Character Creation Devkit - Quick Start Guide

Welcome to the new Character Creation Devkit! This guide will help you get up and running quickly.

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Start all services with Docker
./start-devkit.sh

# Or build fresh images first
./start-devkit.sh build
```

### Option 2: Local Development

```bash
# Start services locally (requires Redis installed)
./start-devkit.sh local
```

## 🌐 Access Points

Once running, access the platform at:

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/api/docs
- **Celery Monitoring**: http://localhost:5555

## 🏗️ Architecture Overview

### Frontend (React)
- Modern, responsive UI with stunning slate & orange theme
- Creator Dashboard, Character Builder, World Builder, Dataset Studio
- Real-time updates via WebSocket connections

### Backend (FastAPI)
- RESTful API with automatic documentation
- JWT authentication
- Redis caching for performance
- Async request handling

### Background Jobs (Celery)
- Dataset generation
- Model training (SFT, DPO, GRPO)
- Character export

### Infrastructure
- **Redis**: Caching, real-time updates, job queue
- **SQLite**: Development database (PostgreSQL ready)
- **Docker Compose**: Orchestration for all services

## 🎨 Key Features

### Creator Dashboard
- Overview of all projects, worlds, and characters
- Quick actions for common tasks
- Real-time statistics

### Character Builder
- Visual personality editor with Big Five traits
- Interactive radar chart visualization
- Step-by-step character creation wizard

### World Builder
- Comprehensive world creation with lore, rules, history
- Culture and location management
- Tabbed interface for organization

### Dataset Studio
- Configure synthetic conversation generation
- Real-time job monitoring
- Download generated datasets

## 🔐 Authentication

1. Register a new account at `/api/v1/auth/register`
2. Login to get JWT tokens
3. Use the access token for authenticated requests

## 📝 API Examples

### Create a Character
```bash
curl -X POST http://localhost:8000/api/v1/characters \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Aria Blackwood",
    "world_id": "world-uuid",
    "personality": {
      "openness": 0.8,
      "conscientiousness": 0.6,
      "extraversion": 0.7,
      "agreeableness": 0.5,
      "neuroticism": 0.3
    }
  }'
```

### Start Dataset Generation
```bash
curl -X POST http://localhost:8000/api/v1/datasets \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "character_id": "character-uuid",
    "num_conversations": 100,
    "style": "mixed"
  }'
```

## 🛠️ Development

### Frontend Development
```bash
cd client
npm install
npm start
```

### Backend Development
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Running Tests
```bash
# Python tests
./scripts/run_tests.sh fast

# React tests
./scripts/run_react_tests.sh ci
```

## 📚 Next Steps

1. Create your first world in the World Builder
2. Design a character with unique personality traits
3. Generate synthetic conversations in Dataset Studio
4. Train your character model
5. Export as a runtime packet for deployment

## 🆘 Troubleshooting

### Services won't start
- Ensure Docker and Docker Compose are installed
- Check if ports 3000, 8000, 5555, 6379 are available
- Run `docker-compose -f docker-compose.dev.yml logs` for details

### Can't connect to Redis
- For local mode, ensure Redis is installed: `brew install redis` (macOS)
- Check if Redis is running: `redis-cli ping`

### Database issues
- The database is created automatically on first run
- To reset: delete `backend/character_devkit.db`

## 📖 Documentation

- [Platform Migration Architecture](PLATFORM_MIGRATION_ARCHITECTURE.md)
- [UI Showcase](PLATFORM_UI_SHOWCASE.md)
- [API Documentation](http://localhost:8000/api/docs)

## 🎉 Ready to Create!

You're all set! Start creating amazing AI characters with rich personalities and immersive worlds. Happy creating! 