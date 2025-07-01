"""
State Router

REST endpoints for accessing world state information.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional

from app.services.state_service import state_service

router = APIRouter()


@router.get("/snapshot")
async def get_world_snapshot():
    """Get a complete snapshot of the world state"""
    try:
        snapshot = await state_service.get_world_snapshot()
        return snapshot
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}")
async def get_entity_details(entity_id: str):
    """Get detailed information about a specific entity"""
    try:
        details = await state_service.get_entity_details(entity_id)
        
        if "error" in details:
            raise HTTPException(status_code=404, detail=details["error"])
        
        return details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities/{entity_id}/update")
async def update_entity_state(entity_id: str, changes: Dict[str, Any]):
    """Update an entity's state"""
    try:
        await state_service.update_entity_state(entity_id, changes)
        return {"status": "success", "entity_id": entity_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities")
async def query_entities(
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    location: Optional[str] = Query(None, description="Filter by location")
):
    """Query entities with filters"""
    try:
        # For now, get all entities and filter
        snapshot = await state_service.get_world_snapshot()
        
        result = {
            "characters": snapshot.get("characters", []),
            "locations": snapshot.get("locations", [])
        }
        
        # Apply filters
        if entity_type == "character":
            result["locations"] = []
        elif entity_type == "location":
            result["characters"] = []
        
        if location:
            result["characters"] = [
                char for char in result["characters"]
                if char.get("location") == location
            ]
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 