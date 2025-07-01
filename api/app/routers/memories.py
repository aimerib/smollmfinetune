"""
Memories Router

REST endpoints for accessing character memories.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.services.memory_service import memory_service

router = APIRouter()


@router.get("/{character_id}")
async def get_character_memories(
    character_id: str,
    limit: Optional[int] = Query(100, description="Maximum number of memories to return"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type")
):
    """Get memories for a specific character"""
    try:
        memories = await memory_service.get_character_memories(character_id, limit=limit)
        
        # Filter by type if requested
        if memory_type:
            memories = [m for m in memories if m.get("memory_type") == memory_type]
        
        return {
            "character_id": character_id,
            "count": len(memories),
            "memories": memories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{character_id}/add")
async def add_character_memory(character_id: str, memory: dict):
    """Add a new memory for a character"""
    try:
        await memory_service.add_memory(character_id, memory)
        return {"status": "success", "character_id": character_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 