#!/usr/bin/env python3
"""
Generate Multimodal Dataset for NarrativeLM Pretraining

This script demonstrates how to generate a synthetic multimodal dataset
for bootstrapping the pretraining of the quad-head NarrativeLM.
"""

import asyncio
import argparse
import logging
from pathlib import Path
import json
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from narrative_engine.synthetic_multimodal_dataset import (
    MultimodalDatasetGenerator,
    SyntheticGenerationConfig
)
from narrative_engine.tts_integration import TTSOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def generate_dataset_with_real_tts(config: SyntheticGenerationConfig):
    """Generate dataset using real TTS integration"""
    
    # Initialize TTS orchestrator
    tts_orchestrator = TTSOrchestrator()
    
    # Modify the dataset generator to use real TTS
    generator = MultimodalDatasetGenerator(config)
    
    # Replace the mock speech synthesizer with real TTS
    original_synthesizer = generator.speech_synthesizer
    
    async def real_synthesize_speech(text, character, emotion_tags):
        """Use real TTS instead of mock synthesis"""
        audio, sr = await tts_orchestrator.synthesize_character_voice(
            text=text,
            character=character,
            emotion_tags=emotion_tags
        )
        return audio, sr
    
    # Monkey patch the synthesizer
    generator.speech_synthesizer.synthesize_speech = real_synthesize_speech
    
    # Generate the dataset
    logger.info("Starting multimodal dataset generation with real TTS...")
    samples = await generator.generate_dataset()
    
    return samples


def analyze_dataset(samples):
    """Analyze the generated dataset and print statistics"""
    
    logger.info("\n=== Dataset Analysis ===")
    logger.info(f"Total samples: {len(samples)}")
    
    # Analyze narrative types
    narrative_types = {}
    for sample in samples:
        nt = sample.narrative_context["type"]
        narrative_types[nt] = narrative_types.get(nt, 0) + 1
    
    logger.info("\nNarrative type distribution:")
    for nt, count in narrative_types.items():
        logger.info(f"  {nt}: {count} ({count/len(samples)*100:.1f}%)")
    
    # Analyze control tokens
    control_token_counts = {}
    for sample in samples:
        for token in sample.control_sequence:
            control_token_counts[token] = control_token_counts.get(token, 0) + 1
    
    logger.info("\nTop 10 control tokens:")
    sorted_tokens = sorted(control_token_counts.items(), key=lambda x: x[1], reverse=True)
    for token, count in sorted_tokens[:10]:
        logger.info(f"  {token}: {count}")
    
    # Analyze memory generation
    memory_samples = sum(1 for s in samples if s.memory_importance > 0)
    logger.info(f"\nMemory samples: {memory_samples} ({memory_samples/len(samples)*100:.1f}%)")
    
    # Analyze speech data
    total_frames = sum(len(s.mel_frames) for s in samples)
    avg_frames = total_frames / len(samples)
    logger.info(f"\nAverage mel frames per sample: {avg_frames:.1f}")
    
    # Estimate total duration (assuming 256 hop length at 22050 Hz)
    hop_length = 256
    sample_rate = 22050
    frame_duration = hop_length / sample_rate  # seconds per frame
    total_duration = total_frames * frame_duration / 60  # minutes
    logger.info(f"Estimated total audio duration: {total_duration:.1f} minutes")


def create_training_manifest(samples, output_dir: Path):
    """Create training manifest for the multimodal model"""
    
    manifest = {
        "version": "1.0",
        "dataset_type": "multimodal_narrative",
        "num_samples": len(samples),
        "heads": {
            "generation": True,
            "control": True,
            "memory": True,
            "speech": True
        },
        "speech_config": {
            "n_mels": 80,
            "quantization_bits": 4,
            "sample_rate": 22050,
            "hop_length": 256
        },
        "chunks": []
    }
    
    # List all chunk files
    chunk_files = sorted(output_dir.glob("chunk_*.json"))
    for chunk_file in chunk_files:
        manifest["chunks"].append(chunk_file.name)
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Training manifest saved to {manifest_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic multimodal dataset for NarrativeLM"
    )
    parser.add_argument(
        "--num-samples", 
        type=int, 
        default=1000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--num-characters", 
        type=int, 
        default=20,
        help="Number of unique characters"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="multimodal_dataset",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--use-real-tts", 
        action="store_true",
        help="Use real TTS providers instead of mock synthesis"
    )
    parser.add_argument(
        "--save-audio", 
        action="store_true",
        help="Save audio files for validation"
    )
    parser.add_argument(
        "--narrative-types",
        nargs="+",
        default=["dialogue", "monologue", "emotional_moment", "action_scene"],
        help="Types of narrative content to generate"
    )
    parser.add_argument(
        "--tts-model",
        choices=["orpheus", "xtts", "bark"],
        default="orpheus",
        help="TTS model to use"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = SyntheticGenerationConfig(
        num_samples=args.num_samples,
        num_characters=args.num_characters,
        narrative_types=args.narrative_types,
        output_dir=Path(args.output_dir),
        save_audio=args.save_audio,
        tts_model=args.tts_model
    )
    
    logger.info("=== Multimodal Dataset Generation Configuration ===")
    logger.info(f"Samples: {config.num_samples}")
    logger.info(f"Characters: {config.num_characters}")
    logger.info(f"Narrative types: {config.narrative_types}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"TTS model: {config.tts_model}")
    logger.info(f"Real TTS: {args.use_real_tts}")
    
    # Generate dataset
    if args.use_real_tts:
        samples = await generate_dataset_with_real_tts(config)
    else:
        generator = MultimodalDatasetGenerator(config)
        samples = await generator.generate_dataset()
    
    # Analyze dataset
    analyze_dataset(samples)
    
    # Create training manifest
    create_training_manifest(samples, config.output_dir)
    
    logger.info("\n=== Dataset Generation Complete ===")
    logger.info(f"Dataset saved to: {config.output_dir}")
    
    # Print example usage for training
    print("\nTo use this dataset for training:")
    print(f"python scripts/run_multimodal_sft.py \\")
    print(f"    --dataset-path {config.output_dir}/manifest.json \\")
    print(f"    --model-name HuggingFaceTB/SmolLM2-135M-Instruct \\")
    print(f"    --output-dir multimodal_narrative_model")


if __name__ == "__main__":
    asyncio.run(main()) 