# narrative_engine/data_schema.py
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional

class Turn(BaseModel):
    """Represents a single turn in the conversation."""
    sender: Literal["user", "assistant"]
    text: str = Field(description="The natural language text of the turn.")
    
    # The 'channel' tag is crucial for the dual-head loss function.
    # It tells the trainer whether to use the text head or action head for this turn.
    channel: Literal["text", "action"] = Field(
        default="text",
        description="The output channel for an assistant's turn."
    )
    
    # An assistant turn can optionally include a structured action.
    # This is denormalized from the 'text' field which would contain its JSON representation.
    action: Optional[Dict] = Field(
        default=None, 
        description="A structured tool call action, if any."
    )

class DatasetSample(BaseModel):
    """
    Defines the schema for a single training sample for the Narrative-LLM.
    This structure provides all necessary context for stateful, multi-persona training.
    """
    session_id: str = Field(description="Unique identifier for the conversational session.")
    
    persona_mix: Dict[str, float] = Field(
        description="A mapping of persona adapter names to their weights. e.g., {'DetectiveNoir': 0.7, 'CosmicHorror': 0.3}"
    )
    
    memory_slots: List[str] = Field(
        description="A list of key memories retrieved for this point in the conversation."
    )
    
    turns: List[Turn] = Field(
        description="The sequence of conversational turns.",
        min_length=1
    )

    class Config:
        extra = 'forbid' # Forbid any extra fields to ensure strict schema adherence. 