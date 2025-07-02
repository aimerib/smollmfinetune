from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from typing import List, Optional
from app.database import get_db
from app.models import Character, World, User
from app.schemas import CharacterCreate, CharacterUpdate, CharacterResponse
from app.auth import get_current_user
from app.redis_client import RedisCache

router = APIRouter(prefix="/characters", tags=["characters"])

@router.get("/", response_model=List[CharacterResponse])
async def list_characters(
    world_id: Optional[str] = Query(None, description="Filter by world ID"),
    is_trained: Optional[bool] = Query(None, description="Filter by training status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List characters accessible to the current user."""
    
    # Build query
    query = select(Character).join(World).where(
        World.project_id.in_(
            select(User.projects).where(User.id == current_user.id)
        )
    )
    
    # Apply filters
    if world_id:
        query = query.where(Character.world_id == world_id)
    if is_trained is not None:
        query = query.where(Character.is_trained == is_trained)
    
    # Execute query with pagination
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    characters = result.scalars().all()
    
    return characters

@router.post("/", response_model=CharacterResponse)
async def create_character(
    character_data: CharacterCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new character."""
    
    # Verify user has access to the world
    world_query = select(World).where(
        World.id == character_data.world_id,
        World.project_id.in_(
            select(User.projects).where(User.id == current_user.id)
        )
    )
    result = await db.execute(world_query)
    world = result.scalar_one_or_none()
    
    if not world:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="World not found or access denied"
        )
    
    # Create character
    character = Character(
        name=character_data.name,
        description=character_data.description,
        world_id=character_data.world_id,
        openness=character_data.personality.openness,
        conscientiousness=character_data.personality.conscientiousness,
        extraversion=character_data.personality.extraversion,
        agreeableness=character_data.personality.agreeableness,
        neuroticism=character_data.personality.neuroticism,
        backstory=character_data.backstory,
        goals=character_data.goals,
        relationships=character_data.relationships,
        traits=character_data.traits,
        voice_style=character_data.voice_style
    )
    
    db.add(character)
    await db.commit()
    await db.refresh(character)
    
    # Invalidate cache
    await RedisCache.delete("characters", f"user_{current_user.id}")
    
    return character

@router.get("/{character_id}", response_model=CharacterResponse)
async def get_character(
    character_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific character."""
    
    # Check cache first
    cached = await RedisCache.get("character", character_id)
    if cached:
        return CharacterResponse(**cached)
    
    # Query with access check
    query = select(Character).join(World).where(
        Character.id == character_id,
        World.project_id.in_(
            select(User.projects).where(User.id == current_user.id)
        )
    )
    result = await db.execute(query)
    character = result.scalar_one_or_none()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found or access denied"
        )
    
    # Cache the result
    character_dict = {
        "id": character.id,
        "name": character.name,
        "description": character.description,
        "world_id": character.world_id,
        "openness": character.openness,
        "conscientiousness": character.conscientiousness,
        "extraversion": character.extraversion,
        "agreeableness": character.agreeableness,
        "neuroticism": character.neuroticism,
        "backstory": character.backstory,
        "goals": character.goals,
        "relationships": character.relationships,
        "traits": character.traits,
        "voice_style": character.voice_style,
        "is_trained": character.is_trained,
        "adapter_path": character.adapter_path,
        "model_version": character.model_version,
        "created_at": character.created_at.isoformat(),
        "updated_at": character.updated_at.isoformat()
    }
    await RedisCache.set("character", character_id, character_dict)
    
    return character

@router.patch("/{character_id}", response_model=CharacterResponse)
async def update_character(
    character_id: str,
    character_update: CharacterUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a character."""
    
    # Get character with access check
    query = select(Character).join(World).where(
        Character.id == character_id,
        World.project_id.in_(
            select(User.projects).where(User.id == current_user.id)
        )
    )
    result = await db.execute(query)
    character = result.scalar_one_or_none()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found or access denied"
        )
    
    # Update fields
    update_data = character_update.model_dump(exclude_unset=True)
    
    # Handle personality traits separately
    if "personality" in update_data:
        personality = update_data.pop("personality")
        if personality:
            character.openness = personality.openness
            character.conscientiousness = personality.conscientiousness
            character.extraversion = personality.extraversion
            character.agreeableness = personality.agreeableness
            character.neuroticism = personality.neuroticism
    
    # Update other fields
    for field, value in update_data.items():
        setattr(character, field, value)
    
    await db.commit()
    await db.refresh(character)
    
    # Invalidate cache
    await RedisCache.delete("character", character_id)
    
    return character

@router.delete("/{character_id}")
async def delete_character(
    character_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a character."""
    
    # Get character with access check
    query = select(Character).join(World).where(
        Character.id == character_id,
        World.project_id.in_(
            select(User.projects).where(User.id == current_user.id)
        )
    )
    result = await db.execute(query)
    character = result.scalar_one_or_none()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found or access denied"
        )
    
    # Delete character (cascades to related records)
    await db.delete(character)
    await db.commit()
    
    # Invalidate cache
    await RedisCache.delete("character", character_id)
    
    return {"message": "Character deleted successfully"}

@router.post("/{character_id}/export")
async def export_character(
    character_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Export a character as a runtime packet."""
    
    # Verify access
    query = select(Character).join(World).where(
        Character.id == character_id,
        World.project_id.in_(
            select(User.projects).where(User.id == current_user.id)
        )
    )
    result = await db.execute(query)
    character = result.scalar_one_or_none()
    
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Character not found or access denied"
        )
    
    # Queue export task
    from app.tasks import export_character_packet
    task = export_character_packet.delay(character_id)
    
    return {
        "message": "Export queued",
        "task_id": task.id,
        "character_id": character_id
    } 