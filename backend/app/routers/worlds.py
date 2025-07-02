from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from ..database import get_db
from ..auth import get_current_user
from ..models import User, World, Project
from ..schemas import (
    WorldBase, WorldCreate, WorldUpdate, WorldResponse, 
    WorldListResponse
)
from ..redis_client import redis_client
import json

router = APIRouter(prefix="/api/worlds", tags=["worlds"])

@router.post("/", response_model=WorldResponse)
async def create_world(
    world: WorldCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new world"""
    # Verify project ownership
    project = db.query(Project).filter(
        Project.id == world.project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or unauthorized"
        )
    
    # Create world
    db_world = World(
        name=world.name,
        description=world.description,
        project_id=world.project_id,
        setting=world.setting,
        rules=world.rules,
        history=world.history,
        cultures=world.cultures,
        locations=world.locations
    )
    
    db.add(db_world)
    db.commit()
    db.refresh(db_world)
    
    # Clear cache
    await redis_client.delete(f"worlds:project:{world.project_id}")
    
    return db_world

@router.get("/{world_id}", response_model=WorldResponse)
async def get_world(
    world_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific world"""
    # Check cache first
    cached = await redis_client.get(f"world:{world_id}")
    if cached:
        return json.loads(cached)
    
    # Query database
    world = db.query(World).join(Project).filter(
        World.id == world_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not world:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="World not found"
        )
    
    # Cache for 5 minutes
    await redis_client.setex(
        f"world:{world_id}",
        300,
        json.dumps({
            "id": world.id,
            "name": world.name,
            "description": world.description,
            "project_id": world.project_id,
            "setting": world.setting,
            "rules": world.rules,
            "history": world.history,
            "cultures": world.cultures,
            "locations": world.locations,
            "created_at": world.created_at.isoformat(),
            "updated_at": world.updated_at.isoformat()
        })
    )
    
    return world

@router.get("/project/{project_id}", response_model=List[WorldListResponse])
async def list_project_worlds(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all worlds in a project"""
    # Verify project ownership
    project = db.query(Project).filter(
        Project.id == project_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or unauthorized"
        )
    
    # Check cache
    cached = await redis_client.get(f"worlds:project:{project_id}")
    if cached:
        return json.loads(cached)
    
    # Query worlds
    worlds = db.query(World).filter(
        World.project_id == project_id
    ).order_by(World.updated_at.desc()).all()
    
    # Format response
    result = [{
        "id": world.id,
        "name": world.name,
        "description": world.description,
        "character_count": len(world.characters),
        "created_at": world.created_at.isoformat(),
        "updated_at": world.updated_at.isoformat()
    } for world in worlds]
    
    # Cache for 1 minute
    await redis_client.setex(
        f"worlds:project:{project_id}",
        60,
        json.dumps(result)
    )
    
    return result

@router.put("/{world_id}", response_model=WorldResponse)
async def update_world(
    world_id: str,
    world_update: WorldUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a world"""
    # Get world with authorization check
    world = db.query(World).join(Project).filter(
        World.id == world_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not world:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="World not found"
        )
    
    # Update fields
    update_data = world_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(world, field, value)
    
    world.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(world)
    
    # Clear cache
    await redis_client.delete(f"world:{world_id}")
    await redis_client.delete(f"worlds:project:{world.project_id}")
    
    return world

@router.delete("/{world_id}")
async def delete_world(
    world_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a world"""
    # Get world with authorization check
    world = db.query(World).join(Project).filter(
        World.id == world_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not world:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="World not found"
        )
    
    project_id = world.project_id
    
    # Delete world (cascades to characters)
    db.delete(world)
    db.commit()
    
    # Clear cache
    await redis_client.delete(f"world:{world_id}")
    await redis_client.delete(f"worlds:project:{project_id}")
    
    return {"message": "World deleted successfully"}

@router.post("/{world_id}/duplicate", response_model=WorldResponse)
async def duplicate_world(
    world_id: str,
    new_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Duplicate a world"""
    # Get original world
    original = db.query(World).join(Project).filter(
        World.id == world_id,
        Project.owner_id == current_user.id
    ).first()
    
    if not original:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="World not found"
        )
    
    # Create duplicate
    duplicate = World(
        name=new_name,
        description=f"Copy of {original.description}" if original.description else None,
        project_id=original.project_id,
        setting=original.setting,
        rules=original.rules,
        history=original.history,
        cultures=original.cultures,
        locations=original.locations
    )
    
    db.add(duplicate)
    db.commit()
    db.refresh(duplicate)
    
    # Clear cache
    await redis_client.delete(f"worlds:project:{original.project_id}")
    
    return duplicate 