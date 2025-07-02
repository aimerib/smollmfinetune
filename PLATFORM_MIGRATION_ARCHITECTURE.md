# Character Creation Devkit - Platform Migration Architecture

## Overview

This document outlines the architecture for migrating the Character Creation Devkit from a Streamlit-based application to a modern React + FastAPI stack with Redis, Celery, and SQLite.

## Architecture Stack

### Frontend (React)
- **Framework**: React 18 with TypeScript
- **Routing**: React Router v6
- **State Management**: Zustand (lightweight, TypeScript-friendly)
- **UI Framework**: Custom design system (Slate & Orange theme)
- **Real-time**: WebSocket connections for live updates
- **Styling**: CSS-in-JS with CSS variables for theming

### Backend (FastAPI)
- **Framework**: FastAPI with Python 3.11+
- **Authentication**: JWT tokens with refresh mechanism
- **API Design**: RESTful + WebSocket endpoints
- **Validation**: Pydantic models for request/response
- **Documentation**: Auto-generated OpenAPI/Swagger

### Data Layer
- **Database**: SQLite (development) → PostgreSQL (production)
- **ORM**: SQLAlchemy 2.0 with async support
- **Migrations**: Alembic for schema management
- **Caching**: Redis for session management and real-time data

### Task Queue
- **Queue**: Celery with Redis as broker
- **Workers**: Separate workers for:
  - Dataset generation
  - Model training
  - Character evaluation
  - Export operations

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Reverse Proxy**: Nginx for static files and API routing
- **Process Manager**: Supervisor for Celery workers
- **Monitoring**: Prometheus + Grafana

## Migration Plan

### Phase 1: Foundation (Current)
✅ Design system and UI components
✅ Creator dashboard and navigation
✅ Basic routing structure
- [ ] Authentication system
- [ ] FastAPI base setup
- [ ] Database models
- [ ] Redis integration

### Phase 2: Core Features
- [ ] Character management API
- [ ] World builder API
- [ ] Dataset generation with Celery
- [ ] Training pipeline integration
- [ ] Real-time training status

### Phase 3: Advanced Features
- [ ] Director's Chair real-time training
- [ ] Character testing interface
- [ ] Export/Import functionality
- [ ] Collaboration features
- [ ] Analytics dashboard

### Phase 4: Production Ready
- [ ] Performance optimization
- [ ] Security hardening
- [ ] Deployment automation
- [ ] Monitoring and logging
- [ ] Documentation

## API Structure

```
/api/v1/
├── auth/
│   ├── login
│   ├── logout
│   ├── refresh
│   └── profile
├── characters/
│   ├── list
│   ├── create
│   ├── update/{id}
│   ├── delete/{id}
│   └── export/{id}
├── worlds/
│   ├── list
│   ├── create
│   ├── update/{id}
│   └── delete/{id}
├── datasets/
│   ├── generate
│   ├── status/{job_id}
│   └── download/{id}
├── training/
│   ├── start
│   ├── status/{job_id}
│   ├── logs/{job_id}
│   └── cancel/{job_id}
└── ws/
    ├── chat/{character_id}
    ├── directors-chair/{session_id}
    └── training-status/{job_id}
```

## Database Schema

### Core Tables
- **users**: Authentication and profile
- **projects**: User projects/workspaces
- **worlds**: World definitions and lore
- **characters**: Character profiles and personalities
- **datasets**: Generated conversation datasets
- **training_jobs**: Training job tracking
- **models**: Trained model metadata
- **chat_sessions**: Runtime chat history

### Relationships
```sql
users 1:N projects
projects 1:N worlds
worlds 1:N characters
characters 1:N datasets
characters 1:N training_jobs
training_jobs 1:1 models
characters 1:N chat_sessions
```

## React Component Architecture

```
src/
├── components/          # Reusable UI components
│   ├── common/         # Buttons, inputs, cards
│   ├── character/      # Character-specific components
│   ├── world/          # World-building components
│   └── training/       # Training UI components
├── pages/              # Route-level components
│   ├── creator/        # Creator platform pages
│   ├── runtime/        # Game/chat interface
│   └── admin/          # Admin dashboard
├── services/           # API client services
│   ├── api.ts          # Base API configuration
│   ├── auth.ts         # Authentication
│   ├── characters.ts   # Character CRUD
│   └── training.ts     # Training operations
├── stores/             # Zustand state stores
│   ├── authStore.ts
│   ├── characterStore.ts
│   └── trainingStore.ts
└── hooks/              # Custom React hooks
    ├── useWebSocket.ts
    ├── useTraining.ts
    └── useCharacter.ts
```

## Celery Task Architecture

```python
# tasks/dataset_generation.py
@celery.task(bind=True)
def generate_dataset(self, character_id: str, config: dict):
    """Generate synthetic conversations for character"""
    
# tasks/training.py
@celery.task(bind=True)
def train_character(self, character_id: str, dataset_id: str):
    """Fine-tune model for character"""
    
# tasks/export.py
@celery.task(bind=True)
def export_runtime_packet(self, character_id: str):
    """Export character as runtime packet"""
```

## Security Considerations

1. **Authentication**: JWT with secure httpOnly cookies
2. **Authorization**: Role-based access control (RBAC)
3. **API Security**: Rate limiting, CORS configuration
4. **Data Validation**: Pydantic models for all inputs
5. **File Upload**: Virus scanning, size limits
6. **WebSocket**: Token-based authentication
7. **Database**: Parameterized queries, encryption at rest

## Performance Optimization

1. **Frontend**:
   - Code splitting and lazy loading
   - Virtual scrolling for large lists
   - Optimistic UI updates
   - Service worker for offline support

2. **Backend**:
   - Redis caching for frequently accessed data
   - Database query optimization
   - Async endpoints where possible
   - Connection pooling

3. **Infrastructure**:
   - CDN for static assets
   - Horizontal scaling for API servers
   - Dedicated GPU nodes for training
   - Load balancing with health checks

## Monitoring & Observability

1. **Application Metrics**:
   - API response times
   - Error rates
   - Active users
   - Training job statistics

2. **Infrastructure Metrics**:
   - CPU/Memory usage
   - Database performance
   - Redis hit rates
   - Celery queue lengths

3. **Business Metrics**:
   - Characters created
   - Training hours
   - User engagement
   - Export usage

## Deployment Strategy

### Development
```bash
docker-compose up -d
```

### Production
```bash
# Kubernetes deployment
kubectl apply -f k8s/

# Or traditional deployment
ansible-playbook deploy.yml
```

## Migration Timeline

- **Week 1-2**: Foundation setup, authentication
- **Week 3-4**: Character management migration
- **Week 5-6**: World builder migration
- **Week 7-8**: Dataset generation migration
- **Week 9-10**: Training pipeline migration
- **Week 11-12**: Director's Chair migration
- **Week 13-14**: Testing and optimization
- **Week 15-16**: Production deployment

## Success Metrics

1. **Performance**: 10x faster page loads
2. **Scalability**: Support 1000+ concurrent users
3. **Reliability**: 99.9% uptime
4. **User Experience**: 50% reduction in task completion time
5. **Developer Experience**: 80% test coverage

## Next Steps

1. Set up FastAPI project structure
2. Implement authentication system
3. Create database models
4. Build character management API
5. Migrate dataset generation to Celery
6. Implement WebSocket connections
7. Complete UI component library
8. Conduct user testing
9. Deploy to staging environment
10. Production rollout 