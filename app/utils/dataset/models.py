from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


@dataclass
class GenerationConfig:
    """Enhanced configuration for dataset generation"""
    use_vllm_optimization: bool = True
    quality_threshold: float = 0.7
    enable_progressive_refinement: bool = True
    max_refinement_iterations: int = 2
    batch_config: Optional[Dict[str, Any]] = None
    judge_batch_size: int = 20
    diversity_weight: float = 0.3
    enable_real_time_filtering: bool = True
    
    # Generation parameters
    max_tokens: int = 4096
    temperature: float = 0.8
    top_p: float = 0.9
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1
    
    # Feature flags
    use_premium_generation: bool = False
    enable_nsfw: bool = False
    nsfw_probability: float = 0.3
    enable_temporal_prompts: bool = True
    temporal_probability: float = 0.4
    enable_intelligent_temporal: bool = True
    enable_scenario_generation: bool = True
    scenario_probability: float = 0.5
    enable_intimate_scenarios: bool = False
    intimate_probability: float = 0.2
    enable_multi_turn: bool = True
    multi_turn_probability: float = 0.3
    enable_exploration_prompts: bool = True
    exploration_probability: float = 0.4
    enable_greeting_prompts: bool = True
    greeting_probability: float = 0.2
    enable_factual_qa: bool = False
    factual_qa_probability: float = 0.1
    
    # Quality control
    diversity_threshold: float = 0.7
    max_retries: int = 3
    batch_size: int = 5
    refinement_iterations: int = 2
    enable_quality_filtering: bool = True
    filter_threshold: float = 0.6


class QualityLevel(Enum):
    """Quality levels for generation"""
    BASIC = "basic"
    ENHANCED = "enhanced" 
    PREMIUM = "premium"


class QAPair(BaseModel):
    """A single question–answer pair used for factual QA datasets."""
    question: str = Field(..., description="User question to reveal the fact")
    answer: str = Field(..., description="Character's first-person answer confirming the fact")