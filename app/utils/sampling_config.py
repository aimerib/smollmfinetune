"""
Model-specific sampling configuration and presets.
Different models work best with different sampling parameters.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import streamlit as st

@dataclass
class SamplingConfig:
    """Configuration for sampling parameters"""
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 200
    seed: Optional[int] = None
    
    # Advanced parameters
    eta_cutoff: Optional[float] = None
    epsilon_cutoff: Optional[float] = None
    typical_p: Optional[float] = None
    tfs: Optional[float] = None
    mirostat_mode: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result

# Model-specific sampling presets
MODEL_SAMPLING_PRESETS = {
    # RpR Model (ArliAI QwQ-32B-ArliAI-RpR-v4)
    "ArliAI/QwQ-32B-ArliAI-RpR-v4": {
        "name": "QwQ-32B RpR Optimized",
        "description": "Optimized for RpR (Roleplay Reasoning) model - high creativity with controlled output",
        "config": SamplingConfig(
            temperature=1.0,
            min_p=0.02,
            top_k=40,
            top_p=0.95,  # Higher top_p for more diversity
            repetition_penalty=1.05,
            max_tokens=2048,
        ),
        "use_cases": ["Roleplay", "Character conversations", "Creative writing"]
    },
    
    # PersonalityEngine model
    "PocketDoc/Dans-PersonalityEngine-V1.3.0-24b": {
        "name": "PersonalityEngine Balanced",
        "description": "Balanced settings for character personality modeling",
        "config": SamplingConfig(
            temperature=0.9,
            top_p=0.92,
            top_k=60,
            min_p=0.05,
            repetition_penalty=1.08,
            max_tokens=300,
        ),
        "use_cases": ["Character modeling", "Personality consistency", "Dialogue"]
    },
    
    # Llama models
    "meta-llama/Llama-3.1-70B-Instruct": {
        "name": "Llama 3.1 Instruct",
        "description": "Conservative settings for instruction following",
        "config": SamplingConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            max_tokens=512,
        ),
        "use_cases": ["Instructions", "Q&A", "Analysis"]
    },
    
    "meta-llama/Llama-3.2-3B-Instruct": {
        "name": "Llama 3.2 Small",
        "description": "Optimized for smaller Llama models",
        "config": SamplingConfig(
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            min_p=0.03,
            repetition_penalty=1.1,
            max_tokens=256,
        ),
        "use_cases": ["General chat", "Simple tasks", "Fast responses"]
    },
    
    # Mistral models
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "name": "Mistral 7B Instruct",
        "description": "Balanced settings for Mistral instruction model",
        "config": SamplingConfig(
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=400,
        ),
        "use_cases": ["Instructions", "Coding", "Analysis"]
    },
    
    # Qwen models  
    "Qwen/Qwen2.5-7B-Instruct": {
        "name": "Qwen 2.5 Instruct",
        "description": "Optimized for Qwen's instruction following",
        "config": SamplingConfig(
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            max_tokens=512,
        ),
        "use_cases": ["Instructions", "Reasoning", "Multilingual"]
    },
    
    # Phi models
    "microsoft/Phi-3.5-mini-instruct": {
        "name": "Phi 3.5 Mini",
        "description": "Efficient settings for Phi small models",
        "config": SamplingConfig(
            temperature=0.8,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            max_tokens=300,
        ),
        "use_cases": ["Chat", "Simple reasoning", "Code assistance"]
    },
    
    # SmolLM models
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "name": "SmolLM Instruct",
        "description": "Optimized for small language models",
        "config": SamplingConfig(
            temperature=0.9,
            top_p=0.95,
            top_k=30,
            min_p=0.05,
            repetition_penalty=1.15,
            max_tokens=200,
        ),
        "use_cases": ["Basic chat", "Simple tasks", "Fast responses"]
    },
}

# Generic presets for common use cases
GENERIC_SAMPLING_PRESETS = {
    "creative_writing": {
        "name": "Creative Writing",
        "description": "High creativity, diverse outputs for creative tasks",
        "config": SamplingConfig(
            temperature=1.1,
            top_p=0.95,
            top_k=50,
            min_p=0.02,
            repetition_penalty=1.05,
            max_tokens=512,
        ),
        "use_cases": ["Stories", "Poetry", "Creative roleplay"]
    },
    
    "conservative": {
        "name": "Conservative",
        "description": "Low temperature, focused outputs for factual tasks",
        "config": SamplingConfig(
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.1,
            max_tokens=256,
        ),
        "use_cases": ["Facts", "Analysis", "Summaries"]
    },
    
    "balanced": {
        "name": "Balanced",
        "description": "Moderate creativity with good coherence",
        "config": SamplingConfig(
            temperature=0.8,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            max_tokens=300,
        ),
        "use_cases": ["General chat", "Q&A", "Dialogue"]
    },
    
    "roleplay": {
        "name": "Roleplay Optimized",
        "description": "High creativity with character consistency",
        "config": SamplingConfig(
            temperature=1.0,
            top_p=0.92,
            min_p=0.03,
            top_k=50,
            repetition_penalty=1.08,
            max_tokens=400,
        ),
        "use_cases": ["Character roleplay", "Interactive fiction", "Dialogue"]
    },
    
    "dataset_generation": {
        "name": "Dataset Generation",
        "description": "High diversity for synthetic data generation",
        "config": SamplingConfig(
            temperature=0.9,
            top_p=0.95,
            top_k=60,
            min_p=0.04,
            repetition_penalty=1.05,
            max_tokens=400,
        ),
        "use_cases": ["Training data", "Diverse samples", "Data augmentation"]
    }
}

def get_model_preset(model_name: str) -> Optional[Dict[str, Any]]:
    """Get sampling preset for a specific model"""
    # Check exact match first
    if model_name in MODEL_SAMPLING_PRESETS:
        return MODEL_SAMPLING_PRESETS[model_name]
    
    # Check partial matches for flexibility
    for preset_model, preset in MODEL_SAMPLING_PRESETS.items():
        if any(part in model_name.lower() for part in preset_model.lower().split('/')):
            return preset
    
    return None

def get_available_presets(model_name: str = None) -> Dict[str, Dict[str, Any]]:
    """Get all available presets, with model-specific ones first if applicable"""
    presets = {}
    
    # Add model-specific preset if available
    model_preset = get_model_preset(model_name) if model_name else None
    if model_preset:
        presets[f"model_specific_{model_name}"] = model_preset
    
    # Add generic presets
    presets.update(GENERIC_SAMPLING_PRESETS)
    
    return presets

def render_sampling_config_ui(
    current_config: Optional[SamplingConfig] = None,
    model_name: str = None,
    key_prefix: str = ""
) -> SamplingConfig:
    """Render UI for sampling configuration with model-specific presets"""
    
    # Get available presets
    presets = get_available_presets(model_name)
    
    # Preset selection
    preset_options = ["Custom"] + list(presets.keys())
    preset_labels = ["Custom"] + [presets[key]["name"] for key in presets.keys()]
    
    selected_preset_idx = st.selectbox(
        "Sampling Preset",
        range(len(preset_options)),
        format_func=lambda i: preset_labels[i],
        help="Choose a preset optimized for your model or use case",
        key=f"{key_prefix}_preset"
    )
    
    selected_preset_key = preset_options[selected_preset_idx]
    
    # Show preset description if not custom
    if selected_preset_key != "Custom":
        preset_info = presets[selected_preset_key]
        st.info(f"**{preset_info['name']}**: {preset_info['description']}")
        
        # Show use cases
        if 'use_cases' in preset_info:
            use_cases = ", ".join(preset_info['use_cases'])
            st.caption(f"Best for: {use_cases}")
        
        # Use preset config as default
        default_config = preset_info["config"]
    else:
        # Use provided config or defaults
        default_config = current_config or SamplingConfig()
    
    # Render parameter controls
    st.markdown("#### Sampling Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Core Parameters**")
        temperature = st.slider(
            "Temperature", 0.1, 2.0, default_config.temperature, 0.05,
            help="Controls randomness. Higher = more creative",
            key=f"{key_prefix}_temp"
        )
        
        top_p = st.slider(
            "Top-p (Nucleus)", 0.1, 1.0, default_config.top_p, 0.01,
            help="Cumulative probability cutoff",
            key=f"{key_prefix}_top_p"
        )
        
        max_tokens = st.number_input(
            "Max Tokens", 10, 4096, default_config.max_tokens,
            help="Maximum response length",
            key=f"{key_prefix}_max_tokens"
        )
    
    with col2:
        st.markdown("**Advanced Parameters**")
        
        # Top-k with optional None
        top_k_enabled = st.checkbox(
            "Enable Top-k", value=default_config.top_k is not None,
            key=f"{key_prefix}_top_k_enabled"
        )
        if top_k_enabled:
            top_k = st.number_input(
                "Top-k", 1, 200, default_config.top_k or 50,
                help="Consider only top-k tokens",
                key=f"{key_prefix}_top_k"
            )
        else:
            top_k = None
        
        # Min-p with optional None
        min_p_enabled = st.checkbox(
            "Enable Min-p", value=default_config.min_p is not None,
            key=f"{key_prefix}_min_p_enabled"
        )
        if min_p_enabled:
            min_p = st.slider(
                "Min-p", 0.0, 0.5, default_config.min_p or 0.05, 0.01,
                help="Minimum probability threshold",
                key=f"{key_prefix}_min_p"
            )
        else:
            min_p = None
        
        repetition_penalty = st.slider(
            "Repetition Penalty", 0.5, 2.0, default_config.repetition_penalty, 0.05,
            help="Penalize repeated tokens",
            key=f"{key_prefix}_rep_penalty"
        )
    
    with col3:
        st.markdown("**Penalty Parameters**")
        
        frequency_penalty = st.slider(
            "Frequency Penalty", -2.0, 2.0, default_config.frequency_penalty, 0.1,
            help="Penalize frequent tokens",
            key=f"{key_prefix}_freq_penalty"
        )
        
        presence_penalty = st.slider(
            "Presence Penalty", -2.0, 2.0, default_config.presence_penalty, 0.1,
            help="Penalize repeated concepts",
            key=f"{key_prefix}_pres_penalty"
        )
        
        # Seed configuration
        use_random_seed = st.checkbox(
            "Random Seed", value=default_config.seed is None,
            help="Use random seed for each generation",
            key=f"{key_prefix}_random_seed"
        )
        
        if not use_random_seed:
            seed = st.number_input(
                "Seed", 0, 2**31-1, default_config.seed or 42,
                help="Fixed seed for reproducible results",
                key=f"{key_prefix}_seed"
            )
        else:
            seed = None
    
    # Advanced parameters (in expander to reduce clutter)
    with st.expander("🔬 Advanced Sampling Parameters"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Typical-p
            typical_p_enabled = st.checkbox(
                "Enable Typical-p", value=default_config.typical_p is not None,
                key=f"{key_prefix}_typical_p_enabled"
            )
            if typical_p_enabled:
                typical_p = st.slider(
                    "Typical-p", 0.1, 1.0, default_config.typical_p or 0.95, 0.01,
                    key=f"{key_prefix}_typical_p"
                )
            else:
                typical_p = None
            
            # TFS (Tail Free Sampling)
            tfs_enabled = st.checkbox(
                "Enable TFS", value=default_config.tfs is not None,
                key=f"{key_prefix}_tfs_enabled"
            )
            if tfs_enabled:
                tfs = st.slider(
                    "TFS", 0.1, 1.0, default_config.tfs or 1.0, 0.01,
                    key=f"{key_prefix}_tfs"
                )
            else:
                tfs = None
        
        with col_b:
            # Eta cutoff
            eta_cutoff_enabled = st.checkbox(
                "Enable Eta Cutoff", value=default_config.eta_cutoff is not None,
                key=f"{key_prefix}_eta_enabled"
            )
            if eta_cutoff_enabled:
                eta_cutoff = st.slider(
                    "Eta Cutoff", 0.0, 1.0, default_config.eta_cutoff or 0.0, 0.01,
                    key=f"{key_prefix}_eta"
                )
            else:
                eta_cutoff = None
            
            # Epsilon cutoff
            epsilon_cutoff_enabled = st.checkbox(
                "Enable Epsilon Cutoff", value=default_config.epsilon_cutoff is not None,
                key=f"{key_prefix}_epsilon_enabled"
            )
            if epsilon_cutoff_enabled:
                epsilon_cutoff = st.slider(
                    "Epsilon Cutoff", 0.0, 1.0, default_config.epsilon_cutoff or 0.0, 0.01,
                    key=f"{key_prefix}_epsilon"
                )
            else:
                epsilon_cutoff = None
    
    # Create and return the configuration
    return SamplingConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        seed=seed,
        eta_cutoff=eta_cutoff,
        epsilon_cutoff=epsilon_cutoff,
        typical_p=typical_p,
        tfs=tfs,
    )

def validate_sampling_config(config: SamplingConfig) -> List[str]:
    """Validate sampling configuration and return list of warnings"""
    warnings = []
    
    if config.temperature < 0.1:
        warnings.append("Very low temperature may produce repetitive outputs")
    elif config.temperature > 1.5:
        warnings.append("Very high temperature may produce incoherent outputs")
    
    if config.top_p < 0.1:
        warnings.append("Very low top-p may be too restrictive")
    
    if config.top_k is not None and config.top_k < 10:
        warnings.append("Very low top-k may be too restrictive")
    
    if config.repetition_penalty > 1.5:
        warnings.append("High repetition penalty may produce unnatural text")
    elif config.repetition_penalty < 0.8:
        warnings.append("Low repetition penalty may cause excessive repetition")
    
    if config.max_tokens > 2048:
        warnings.append("Very long responses may be slow to generate")
    
    return warnings 