"""
FastAPI Backend for Character Creation Platform Director's View

This API provides real-time WebSocket connections and REST endpoints for
monitoring the triple-head model, character states, and memory formation.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from prometheus_client import make_asgi_app

# Import routers
from app.routers import state, memories, emotions, metrics, inference, directors_chair, relationships
from app.websocket.manager import websocket_manager
from app.services.event_bus import event_bus
from app.services.state_service import state_service
from app.services.memory_service import memory_service
from app.services.emotion_service import emotion_service

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Director's View API")
    
    # Initialize services
    await event_bus.start()
    await state_service.initialize()
    await memory_service.initialize()
    await emotion_service.initialize()
    
    # Start background tasks
    asyncio.create_task(websocket_manager.heartbeat_loop())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Director's View API")
    await event_bus.stop()
    await websocket_manager.disconnect_all()


# Create FastAPI app
app = FastAPI(
    title="Character Creation Devkit - Director's View API",
    description="Real-time monitoring and control interface for AI character simulations",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(state.router, prefix="/api/state", tags=["state"])
app.include_router(memories.router, prefix="/api/memories", tags=["memories"])
app.include_router(emotions.router, prefix="/api/emotions", tags=["emotions"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])
app.include_router(inference.router, tags=["inference"])  # No prefix for direct endpoints
app.include_router(directors_chair.router, tags=["directors-chair"])  # Directors Chair endpoints
app.include_router(relationships.router, prefix="/api", tags=["relationships"])  # Relationship visualization

# WebSocket endpoint
from app.websocket.director import director_websocket
app.add_websocket_route("/ws/director", director_websocket)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "directors-view-api",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error("Unhandled exception", exc_info=exc, path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if app.debug else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    ) 