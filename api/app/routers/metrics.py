"""
Metrics Router

REST endpoints for system metrics and monitoring.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.websocket.manager import websocket_manager
from app.services.event_bus import event_bus

router = APIRouter()


@router.get("/websocket")
async def get_websocket_metrics():
    """Get WebSocket connection statistics"""
    try:
        stats = websocket_manager.get_connection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eventbus")
async def get_eventbus_metrics():
    """Get event bus statistics"""
    try:
        stats = event_bus.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "services": {
            "websocket": {
                "connections": len(websocket_manager.active_connections)
            },
            "event_bus": {
                "running": event_bus.running,
                "queue_size": event_bus.event_queue.qsize()
            }
        }
    } 