"""
Emotions Router

REST endpoints for accessing character emotional states.
"""

from fastapi import APIRouter, HTTPException

from app.services.emotion_service import emotion_service

router = APIRouter()


@router.get("/{character_id}")
async def get_character_emotions(character_id: str):
    """Get current emotional state for a character"""
    try:
        emotions = await emotion_service.get_character_emotions(character_id)
        return {
            "character_id": character_id,
            **emotions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{character_id}/update")
async def update_character_emotion(
    character_id: str,
    emotion: str,
    strength: float,
    decay_rate: float = 0.05
):
    """Update a specific emotion for a character"""
    try:
        if not 0 <= strength <= 1:
            raise HTTPException(status_code=400, detail="Strength must be between 0 and 1")
        
        if not 0 < decay_rate <= 1:
            raise HTTPException(status_code=400, detail="Decay rate must be between 0 and 1")
        
        await emotion_service.update_emotion(character_id, emotion, strength, decay_rate)
        return {"status": "success", "character_id": character_id, "emotion": emotion}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 