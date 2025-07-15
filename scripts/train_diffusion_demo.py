#!/usr/bin/env python3
"""
Demonstration training script for the diffusion multimodal model

This script shows how to:
1. Create a diffusion training manager
2. Generate synthetic multimodal data
3. Train the model
4. Generate samples

Run this to validate the complete training pipeline.
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.narrative_engine.diffusion_trainer import create_diffusion_trainer
from backend.app.narrative_engine.diffusion_config import get_small_config, get_medium_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_batch(config, batch_size=4, device='cpu'):
    """Create synthetic multimodal batch for training demonstration"""
    
    # Generate synthetic data matching the expected format
    text_embeddings = torch.randn(
        batch_size, 
        config.modalities.text_max_sequence_length, 
        config.modalities.text_embedding_dim,
        device=device
    )
    
    speech_features = torch.randn(
        batch_size,
        config.modalities.speech_max_frames,
        config.modalities.speech_mel_bins,
        device=device
    )
    
    control_embeddings = torch.randn(
        batch_size,
        config.modalities.control_max_tokens,
        config.modalities.control_embedding_dim,
        device=device
    )
    
    memory_vectors = torch.randn(
        batch_size,
        config.modalities.memory_max_vectors,
        config.modalities.memory_embedding_dim + config.modalities.memory_metadata_dim,
        device=device
    )
    
    character_ids = torch.randint(1, 100, (batch_size,), device=device)
    
    return {
        'text_embeddings': text_embeddings,
        'speech_features': speech_features,
        'control_embeddings': control_embeddings,
        'memory_vectors': memory_vectors,
        'character_ids': character_ids
    }


def create_synthetic_dataset(config, num_batches=10, batch_size=4, device='cpu'):
    """Create a synthetic dataset for demonstration"""
    dataset = []
    
    for i in range(num_batches):
        batch = create_synthetic_batch(config, batch_size, device)
        dataset.append(batch)
    
    logger.info(f"Created synthetic dataset with {num_batches} batches of {batch_size} samples each")
    return dataset


def train_demo(config_name="small", num_epochs=2, num_batches=10, batch_size=4, device=None):
    """Demonstration training loop"""
    
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"🚀 Starting Diffusion Training Demo")
    logger.info(f"Config: {config_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batches per epoch: {num_batches}")
    logger.info(f"Batch size: {batch_size}")
    
    # Create trainer
    trainer = create_diffusion_trainer(config_name, device=device)
    trainer.setup_training()
    
    config = trainer.diffusion_config
    
    # Print model summary
    summary = trainer.get_training_summary()
    logger.info(f"📊 Model Summary:")
    logger.info(f"  Parameters: {summary['model_parameters']:,}")
    logger.info(f"  Hidden size: {config.transformer.hidden_size}")
    logger.info(f"  Layers: {config.transformer.num_layers}")
    logger.info(f"  EMA enabled: {config.use_ema}")
    
    # Create synthetic dataset
    train_dataset = create_synthetic_dataset(config, num_batches, batch_size, device)
    val_dataset = create_synthetic_dataset(config, max(2, num_batches // 5), batch_size, device)
    
    # Training loop
    logger.info("\n🎯 Starting Training Loop")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        trainer.model.train()
        epoch_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataset):
            metrics = trainer.train_step(batch)
            epoch_train_loss += metrics['total_loss']
            
            if batch_idx % 5 == 0:  # Log every 5 batches
                logger.info(f"  Batch {batch_idx}/{len(train_dataset)}: "
                          f"Loss={metrics['total_loss']:.4f}, "
                          f"Text={metrics['text_loss']:.4f}, "
                          f"Speech={metrics['speech_loss']:.4f}, "
                          f"Control={metrics['control_loss']:.4f}, "
                          f"Memory={metrics['memory_loss']:.4f}, "
                          f"Align={metrics['alignment_loss']:.4f}")
        
        avg_train_loss = epoch_train_loss / len(train_dataset)
        
        # Validation phase
        trainer.model.eval()
        epoch_val_loss = 0.0
        
        for batch in val_dataset:
            val_metrics = trainer.validate_step(batch)
            epoch_val_loss += val_metrics['val_total_loss']
        
        avg_val_loss = epoch_val_loss / len(val_dataset)
        
        logger.info(f"  📈 Epoch Summary:")
        logger.info(f"    Train Loss: {avg_train_loss:.4f}")
        logger.info(f"    Val Loss: {avg_val_loss:.4f}")
        logger.info(f"    Learning Rate: {trainer.lr_scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint if best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trainer.save_checkpoint()
            logger.info(f"    ✅ New best model saved!")
        
        trainer.current_epoch += 1
    
    logger.info(f"\n🎉 Training Complete!")
    logger.info(f"Final training loss: {avg_train_loss:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    return trainer


def generation_demo(trainer, num_samples=2):
    """Demonstrate sample generation"""
    logger.info(f"\n🎨 Generation Demo")
    
    # Generate unconditional samples
    logger.info("Generating unconditional samples...")
    unconditional_samples = trainer.generate_samples(
        batch_size=num_samples,
        character_ids=None,
        guidance_scale=1.0,  # No guidance for unconditional
        num_inference_steps=20
    )
    
    logger.info(f"✅ Generated unconditional samples:")
    logger.info(f"  Text: {unconditional_samples['text'].shape}")
    logger.info(f"  Speech: {unconditional_samples['speech'].shape}")
    logger.info(f"  Control: {unconditional_samples['control'].shape}")
    logger.info(f"  Memory: {unconditional_samples['memory'].shape}")
    
    # Generate character-conditioned samples
    character_ids = torch.tensor([1, 2], device=trainer.device)[:num_samples]
    logger.info(f"Generating character-conditioned samples (IDs: {character_ids.tolist()})...")
    
    conditional_samples = trainer.generate_samples(
        batch_size=num_samples,
        character_ids=character_ids,
        guidance_scale=7.5,  # Strong guidance for character conditioning
        num_inference_steps=20
    )
    
    logger.info(f"✅ Generated character-conditioned samples:")
    logger.info(f"  Text: {conditional_samples['text'].shape}")
    logger.info(f"  Speech: {conditional_samples['speech'].shape}")
    logger.info(f"  Control: {conditional_samples['control'].shape}")
    logger.info(f"  Memory: {conditional_samples['memory'].shape}")
    
    return unconditional_samples, conditional_samples


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description="Diffusion Multimodal Training Demo")
    parser.add_argument("--config", default="small", choices=["small", "medium"], 
                       help="Model configuration")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batches", type=int, default=10, help="Batches per epoch")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    parser.add_argument("--no-generation", action="store_true", help="Skip generation demo")
    
    args = parser.parse_args()
    
    try:
        # Training demo
        trainer = train_demo(
            config_name=args.config,
            num_epochs=args.epochs,
            num_batches=args.batches,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Generation demo (unless skipped)
        if not args.no_generation:
            generation_demo(trainer, num_samples=2)
        
        logger.info("\n🌟 Demo completed successfully!")
        logger.info("The diffusion multimodal architecture is ready for production training!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 