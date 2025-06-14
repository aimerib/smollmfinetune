import streamlit as st
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters in text generation"""
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0  # 0 means disabled
    repetition_penalty: float = 1.1
    max_tokens: int = 400
    min_tokens: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls"""
        config = {
            'temperature': self.temperature,
            'top_p': self.top_p,
            'repetition_penalty': self.repetition_penalty,
            'max_tokens': self.max_tokens
        }
        if self.top_k > 0:
            config['top_k'] = self.top_k
        if self.min_tokens > 0:
            config['min_tokens'] = self.min_tokens
        return config


# Model-specific presets
MODEL_PRESETS = {
    'SmolLM2': {
        'name': 'SmolLM2 Optimized',
        'temperature': 0.8,
        'top_p': 0.9,
        'repetition_penalty': 1.05,
        'max_tokens': 400
    },
    'Mistral': {
        'name': 'Mistral Optimized',
        'temperature': 0.7,
        'top_p': 0.95,
        'repetition_penalty': 1.1,
        'max_tokens': 500
    },
    'Llama': {
        'name': 'Llama Optimized',
        'temperature': 0.8,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
        'max_tokens': 400
    },
    'Qwen': {
        'name': 'Qwen Optimized',
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.05,
        'max_tokens': 350
    }
}


def get_model_preset(model_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Get preset configuration for a specific model"""
    if not model_name:
        return None
    
    model_lower = model_name.lower()
    
    for preset_key, preset_config in MODEL_PRESETS.items():
        if preset_key.lower() in model_lower:
            return preset_config
    
    return None


def render_sampling_config_ui(current_config: SamplingConfig, 
                             model_name: Optional[str] = None,
                             key_prefix: str = "sampling",
                             use_expander: bool = True) -> SamplingConfig:
    """Render sampling configuration UI in Streamlit"""
    
    def render_controls():
        # Model preset selection
        preset = get_model_preset(model_name)
        if preset:
            st.info(f"💡 **{preset['name']}** preset available")
            if st.button(f"Apply {preset['name']} Preset", key=f"{key_prefix}_apply_preset"):
                return SamplingConfig(**{k: v for k, v in preset.items() if k != 'name'})
        
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=current_config.temperature,
                step=0.1,
                help="Higher = more random, Lower = more focused",
                key=f"{key_prefix}_temperature"
            )
            
            top_p = st.slider(
                "Top-p (Nucleus Sampling)",
                min_value=0.1,
                max_value=1.0,
                value=current_config.top_p,
                step=0.05,
                help="Probability mass for nucleus sampling",
                key=f"{key_prefix}_top_p"
            )
            
            repetition_penalty = st.slider(
                "Repetition Penalty",
                min_value=1.0,
                max_value=1.5,
                value=current_config.repetition_penalty,
                step=0.01,
                help="Penalize repetitive text",
                key=f"{key_prefix}_rep_penalty"
            )
        
        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=50,
                max_value=2000,
                value=current_config.max_tokens,
                step=50,
                help="Maximum response length",
                key=f"{key_prefix}_max_tokens"
            )
            
            use_top_k = st.checkbox(
                "Enable Top-K Sampling",
                value=current_config.top_k > 0,
                help="Limit to top K most likely tokens",
                key=f"{key_prefix}_use_top_k"
            )
            
            if use_top_k:
                top_k = st.number_input(
                    "Top-K Value",
                    min_value=1,
                    max_value=100,
                    value=max(1, current_config.top_k),
                    step=5,
                    help="Number of top tokens to consider",
                    key=f"{key_prefix}_top_k"
                )
            else:
                top_k = 0
            
            min_tokens = st.number_input(
                "Min Tokens",
                min_value=0,
                max_value=200,
                value=current_config.min_tokens,
                step=10,
                help="Minimum response length (0 = disabled)",
                key=f"{key_prefix}_min_tokens"
            )
        
        return SamplingConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            min_tokens=min_tokens
        )
    

    return render_controls()


def validate_sampling_config(config: SamplingConfig) -> tuple[bool, list[str]]:
    """Validate sampling configuration and return errors if any"""
    errors = []
    
    if config.temperature <= 0 or config.temperature > 2:
        errors.append("Temperature must be between 0.1 and 2.0")
    
    if config.top_p <= 0 or config.top_p > 1:
        errors.append("Top-p must be between 0.1 and 1.0")
    
    if config.repetition_penalty < 1.0 or config.repetition_penalty > 2.0:
        errors.append("Repetition penalty must be between 1.0 and 2.0")
    
    if config.max_tokens < 10 or config.max_tokens > 4000:
        errors.append("Max tokens must be between 10 and 4000")
    
    if config.top_k < 0 or config.top_k > 200:
        errors.append("Top-k must be between 0 and 200")
    
    if config.min_tokens < 0 or config.min_tokens >= config.max_tokens:
        errors.append("Min tokens must be less than max tokens")
    
    return len(errors) == 0, errors


def get_quality_optimized_config() -> SamplingConfig:
    """Get configuration optimized for quality over speed"""
    return SamplingConfig(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        max_tokens=500,
        min_tokens=20
    )


def get_speed_optimized_config() -> SamplingConfig:
    """Get configuration optimized for speed over quality"""
    return SamplingConfig(
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.05,
        max_tokens=300,
        min_tokens=10
    )


def get_creative_config() -> SamplingConfig:
    """Get configuration optimized for creative/diverse outputs"""
    return SamplingConfig(
        temperature=1.0,
        top_p=0.85,
        repetition_penalty=1.15,
        max_tokens=400,
        min_tokens=15
    )


def get_conservative_config() -> SamplingConfig:
    """Get configuration optimized for consistent/conservative outputs"""
    return SamplingConfig(
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
        max_tokens=350,
        min_tokens=20
    ) 