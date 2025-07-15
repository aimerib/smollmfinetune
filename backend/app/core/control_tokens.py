import json
from typing import Dict, Any

def load_control_tokens(filepath: str = "app/utils/control_tokens.json") -> Dict[str, Any]:
    """
    Loads control tokens from a JSON file.
    For now, returns a hardcoded dictionary.
    """
    # In the future, this could load from a file.
    # For now, return a dictionary that might be expected.
    return {
        "emotions": ["joy", "sadness", "anger", "fear", "surprise", "disgust", "happiness", "tired"],
        "paces": ["very_slow", "slow", "normal", "fast", "very_fast"],
        "tones": ["dramatic", "emotional", "expressive", "neutral", "calm", "matter_of_fact"]
    } 