from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from backend.app.config import settings
from backend.app.database import create_tables
from backend.app.routers import (
    characters, worlds, datasets, multimodal, training, inference, 
    websocket, voice_streaming, quad_head_training, quad_head_streaming,
    flow_matching
)
from backend.app.redis_client import get_redis_pool
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting up Character Creation Devkit API...")
    
    # Create database tables
    await create_tables()
    logger.info("Database tables created/verified")
    
    # Initialize Redis pool
    await get_redis_pool()
    logger.info("Redis connection pool initialized")
    
    # Initialize inference engine
    from backend.app.routers.inference import engine
    await engine.initialize()
    logger.info("Inference engine initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    
    # Shutdown inference engine
    from backend.app.routers.inference import engine
    await engine.shutdown()
    logger.info("Inference engine shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include all routers
app.include_router(characters.router)
app.include_router(worlds.router)
app.include_router(datasets.router)
app.include_router(multimodal.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(websocket.router)
app.include_router(voice_streaming.router)
app.include_router(quad_head_training.router)
app.include_router(quad_head_streaming.router)
app.include_router(flow_matching.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/api/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    ) 