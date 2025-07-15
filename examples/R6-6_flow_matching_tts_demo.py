#!/usr/bin/env python3
"""
R6-6: Flow-Matching TTS Demo
============================

This script demonstrates the advanced custom speech architecture implemented in R6-6,
including character conditioning, narrative-aware attention, and real-time voice synthesis.

Usage:
    python examples/R6-6_flow_matching_tts_demo.py

Features Demonstrated:
- Flow-matching TTS architecture
- Character voice conditioning
- Narrative context integration
- Real-time inference
- Zero-shot voice cloning
- Emotional voice morphing
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.app.services.voice.flow_matching_tts import (
    NarrativeFlowMatchingTTS,
    FlowMatchingTrainer,
    SpeakerEmbeddingExtractor,
    CharacterConditioner
)

def demo_basic_flow_matching():
    """Demonstrate basic flow-matching TTS functionality."""
    print("🎵 Flow-Matching TTS Basic Demo")
    print("=" * 50)
    
    # Initialize the model
    model = NarrativeFlowMatchingTTS(
        hidden_dim=768,
        num_mel_bins=80,
        num_heads=12
    )
    
    # Create sample inputs
    text_tokens = torch.randint(0, 1000, (1, 50))  # Batch=1, seq_len=50
    character_id = "alice"
    narrative_context = torch.randn(1, 768)  # Story context embedding
    
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Input text tokens: {text_tokens.shape}")
    print(f"✓ Character ID: {character_id}")
    print(f"✓ Narrative context: {narrative_context.shape}")
    
    # Forward pass for inference
    model.eval()
    with torch.no_grad():
        mel_output = model(
            text_tokens=text_tokens,
            character_id=character_id, 
            narrative_context=narrative_context
        )
    
    print(f"✓ Generated mel-spectrogram: {mel_output.shape}")
    print(f"✓ Output range: [{mel_output.min():.3f}, {mel_output.max():.3f}]")
    print()

def demo_character_conditioning():
    """Demonstrate character-specific voice conditioning."""
    print("🎭 Character Voice Conditioning Demo")
    print("=" * 50)
    
    # Initialize character conditioner
    conditioner = CharacterConditioner(hidden_dim=768)
    
    # Define character personalities (Big Five traits)
    characters = {
        "alice": {
            "openness": 0.8,
            "conscientiousness": 0.7,
            "extraversion": 0.6,
            "agreeableness": 0.9,
            "neuroticism": 0.3
        },
        "bob": {
            "openness": 0.4,
            "conscientiousness": 0.9,
            "extraversion": 0.3,
            "agreeableness": 0.6,
            "neuroticism": 0.7
        }
    }
    
    text_hidden = torch.randn(1, 50, 768)  # Sample text embeddings
    
    for char_name, personality in characters.items():
        # Apply character conditioning
        conditioned_output = conditioner(text_hidden, char_name, personality)
        
        print(f"✓ Character: {char_name}")
        print(f"  Personality: {personality}")
        print(f"  Conditioned output: {conditioned_output.shape}")
        print(f"  Output norm: {conditioned_output.norm().item():.3f}")
        print()

def demo_emotional_voice_morphing():
    """Demonstrate emotion-aware voice morphing."""
    print("😊 Emotional Voice Morphing Demo")
    print("=" * 50)
    
    model = NarrativeFlowMatchingTTS(hidden_dim=768, num_mel_bins=80)
    
    # Sample text
    text_tokens = torch.randint(0, 1000, (1, 30))
    character_id = "alice"
    
    # Different emotional states
    emotions = {
        "happy": {"joy": 0.9, "sadness": 0.1, "anger": 0.0, "fear": 0.1},
        "sad": {"joy": 0.1, "sadness": 0.8, "anger": 0.0, "fear": 0.3},
        "angry": {"joy": 0.0, "sadness": 0.2, "anger": 0.9, "fear": 0.1},
        "excited": {"joy": 0.8, "sadness": 0.0, "anger": 0.0, "fear": 0.0}
    }
    
    model.eval()
    with torch.no_grad():
        for emotion_name, emotion_state in emotions.items():
            # Create narrative context with emotional state
            narrative_context = torch.randn(1, 768)
            
            # Generate voice with emotional conditioning
            mel_output = model(
                text_tokens=text_tokens,
                character_id=character_id,
                narrative_context=narrative_context,
                emotional_state=emotion_state
            )
            
            print(f"✓ Emotion: {emotion_name}")
            print(f"  State: {emotion_state}")
            print(f"  Generated mel: {mel_output.shape}")
            print(f"  Spectral centroid: {torch.mean(mel_output).item():.3f}")
            print()

def demo_zero_shot_voice_cloning():
    """Demonstrate zero-shot voice cloning capabilities."""
    print("🎙️ Zero-Shot Voice Cloning Demo")
    print("=" * 50)
    
    # Initialize speaker embedding extractor
    extractor = SpeakerEmbeddingExtractor(
        embedding_dim=256,
        num_mel_bins=80
    )
    
    # Simulate reference audio (mel-spectrogram)
    reference_mel = torch.randn(1, 80, 100)  # 1 sec of audio
    
    print(f"✓ Reference audio: {reference_mel.shape}")
    
    # Extract speaker embedding
    speaker_embedding = extractor.extract_speaker_embedding(reference_mel)
    print(f"✓ Speaker embedding extracted: {speaker_embedding.shape}")
    print(f"✓ Embedding norm: {speaker_embedding.norm().item():.3f}")
    
    # Use embedding for voice adaptation
    model = NarrativeFlowMatchingTTS(hidden_dim=768, num_mel_bins=80)
    text_tokens = torch.randint(0, 1000, (1, 40))
    
    model.eval()
    with torch.no_grad():
        # Generate with original voice
        original_mel = model(
            text_tokens=text_tokens,
            character_id="alice",
            narrative_context=torch.randn(1, 768)
        )
        
        # Generate with cloned voice
        cloned_mel = model(
            text_tokens=text_tokens,
            character_id="alice",
            narrative_context=torch.randn(1, 768),
            speaker_embedding=speaker_embedding
        )
    
    print(f"✓ Original voice: {original_mel.shape}")
    print(f"✓ Cloned voice: {cloned_mel.shape}")
    print(f"✓ Voice difference: {torch.abs(original_mel - cloned_mel).mean().item():.3f}")
    print()

def demo_training_workflow():
    """Demonstrate the training workflow."""
    print("🏋️ Training Workflow Demo")
    print("=" * 50)
    
    # Initialize trainer
    model = NarrativeFlowMatchingTTS(hidden_dim=512, num_mel_bins=80)  # Smaller for demo
    trainer = FlowMatchingTrainer(
        model=model,
        learning_rate=1e-4,
        warmup_steps=100,
        max_steps=1000
    )
    
    print(f"✓ Trainer initialized")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Learning rate: {trainer.learning_rate}")
    
    # Simulate training batch
    batch = {
        "text_tokens": torch.randint(0, 1000, (4, 32)),  # Batch of 4
        "target_mel": torch.randn(4, 80, 64),
        "character_ids": ["alice", "bob", "clara", "alice"],
        "narrative_contexts": torch.randn(4, 512)
    }
    
    print(f"✓ Training batch prepared: {len(batch['character_ids'])} samples")
    
    # Simulate training step
    model.train()
    loss = trainer.training_step(batch)
    
    print(f"✓ Training step completed")
    print(f"✓ Loss: {loss:.4f}")
    
    # Check training stats
    stats = trainer.get_training_stats()
    print(f"✓ Training stats: {stats}")
    print()

def demo_realtime_inference():
    """Demonstrate real-time streaming inference."""
    print("⚡ Real-Time Streaming Inference Demo")
    print("=" * 50)
    
    model = NarrativeFlowMatchingTTS(hidden_dim=768, num_mel_bins=80)
    model.eval()
    
    # Simulate streaming text tokens
    streaming_text = [
        torch.randint(0, 1000, (1, 10)),  # "Hello there"
        torch.randint(0, 1000, (1, 12)),  # "how are you"
        torch.randint(0, 1000, (1, 8)),   # "today?"
    ]
    
    character_id = "alice"
    narrative_context = torch.randn(1, 768)
    
    print(f"✓ Streaming text simulation: {len(streaming_text)} chunks")
    
    with torch.no_grad():
        for i, text_chunk in enumerate(streaming_text):
            # Generate mel chunk
            mel_chunk = model(
                text_tokens=text_chunk,
                character_id=character_id,
                narrative_context=narrative_context
            )
            
            print(f"  Chunk {i+1}: {text_chunk.shape} → {mel_chunk.shape}")
            
            # Simulate real-time processing delay
            import time
            time.sleep(0.1)
    
    print(f"✓ Streaming inference completed")
    print()

def main():
    """Run all demos."""
    print("🎵 Flow-Matching TTS Demonstration Suite")
    print("🚀 R6-6: Advanced Custom Speech Architecture")
    print("=" * 60)
    print()
    
    try:
        # Check PyTorch availability
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print()
        
        # Run demos
        demo_basic_flow_matching()
        demo_character_conditioning()
        demo_emotional_voice_morphing() 
        demo_zero_shot_voice_cloning()
        demo_training_workflow()
        demo_realtime_inference()
        
        print("🎉 All demos completed successfully!")
        print()
        print("Next Steps:")
        print("- Check out the React training dashboard at http://localhost:3000/flow-matching")
        print("- Use the FastAPI endpoints at http://localhost:8000/docs")
        print("- Explore character voice customization in the Creator Dashboard")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure you have PyTorch installed and the flow-matching modules available.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 