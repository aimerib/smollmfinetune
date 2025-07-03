#!/usr/bin/env python3
"""
Enhanced Diffusion Training Demo

Demonstrates the advanced features added to the diffusion trainer:
- Real-time WebSocket updates
- Multi-GPU training support  
- Advanced sampling strategies
- Custom architecture modifications
- Progressive unfreezing
- Adaptive learning rates

Usage:
    python scripts/enhanced_diffusion_training_demo.py
"""

import asyncio
import logging
import torch
import json
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from narrative_engine.diffusion_config import get_small_config, get_medium_config
from narrative_engine.diffusion_trainer import DiffusionTrainingManager

def create_dummy_dataloader(batch_size: int = 4, num_batches: int = 10):
    """Create a dummy dataloader for demonstration"""
    class DummyDataset:
        def __init__(self, num_batches, batch_size):
            self.num_batches = num_batches
            self.batch_size = batch_size
        
        def __len__(self):
            return self.num_batches
        
        def __iter__(self):
            for i in range(self.num_batches):
                # Create dummy multimodal data
                batch = {
                    'text': torch.randn(self.batch_size, 512, 768),  # [batch, seq_len, embed_dim]
                    'speech': torch.randn(self.batch_size, 100, 80),  # [batch, frames, mel_bins]
                    'control': torch.randn(self.batch_size, 32, 256),  # [batch, tokens, control_dim]
                    'memory': torch.randn(self.batch_size, 64, 512),  # [batch, vectors, memory_dim]
                    'character_ids': torch.randint(0, 10, (self.batch_size,))
                }
                yield batch
    
    return DummyDataset(num_batches, batch_size)

def custom_architecture_hook_example(trainer, **kwargs):
    """Example custom architecture hook"""
    step = kwargs.get('step', 0)
    if step % 100 == 0:
        logger.info(f"🪝 Custom hook executed at step {step}")
        
        # Example: Print gradient norms
        total_norm = 0
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        logger.info(f"📊 Gradient norm: {total_norm:.4f}")

async def enhanced_training_demo():
    """Comprehensive demo of enhanced training features"""
    
    logger.info("🚀 Starting Enhanced Diffusion Training Demo")
    
    # 1. Setup configuration
    config = get_small_config()
    logger.info(f"📋 Using config: {config.transformer.hidden_size}D model")
    
    # 2. Create trainer with WebSocket support
    trainer = DiffusionTrainingManager(
        config=config,
        output_dir="demo_output/enhanced_training",
        use_wandb=False,  # Set to True if you have wandb configured
        enable_websocket=True,
        websocket_port=8765
    )
    
    # 3. Setup model
    trainer.setup_model()
    logger.info(f"🔧 Model created with {sum(p.numel() for p in trainer.model.parameters())} parameters")
    
    # 4. Demonstrate custom architecture modifications
    logger.info("🎯 Setting up custom architecture modifications...")
    
    # Custom loss weights (emphasize text and control)
    trainer.set_custom_loss_weights(
        text_weight=2.0,
        speech_weight=1.0,
        control_weight=1.5,
        memory_weight=1.0,
        alignment_weight=0.5
    )
    
    # Add custom architecture hook
    trainer.add_architecture_hook("gradient_monitoring", custom_architecture_hook_example)
    
    # Progressive unfreezing schedule
    trainer.unfreeze_schedule = {
        50: ["transformer.layers.0", "transformer.layers.1"],  # Unfreeze first 2 layers at step 50
        100: ["transformer.layers.2", "transformer.layers.3"], # Unfreeze next 2 layers at step 100
    }
    
    # Enable adaptive learning rate
    trainer.adaptive_lr_enabled = True
    
    # 5. Demonstrate layer freezing
    logger.info("❄️  Demonstrating layer freezing...")
    trainer.freeze_layers(["transformer.layers.0", "transformer.layers.1"])
    
    # 6. Create dummy data
    train_dataloader = create_dummy_dataloader(batch_size=2, num_batches=20)
    val_dataloader = create_dummy_dataloader(batch_size=2, num_batches=5)
    
    # 7. Start training with all enhancements
    logger.info("🏋️  Starting enhanced training...")
    
    try:
        await trainer.train_async(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=2,
            save_every=25,
            validate_every=15,
            log_every=5
        )
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    # 8. Demonstrate advanced sampling
    logger.info("🎲 Demonstrating advanced sampling strategies...")
    
    # Create dummy logits for sampling demo
    dummy_logits = torch.randn(1, 1000)  # [batch_size, vocab_size]
    
    # Nucleus sampling
    nucleus_logits = trainer.nucleus_sampling(dummy_logits.clone(), top_p=0.9, temperature=0.8)
    logger.info(f"🌰 Nucleus sampling: {(nucleus_logits > float('-inf')).sum().item()} tokens kept")
    
    # Top-k sampling
    topk_logits = trainer.top_k_sampling(dummy_logits.clone(), top_k=50, temperature=0.8)
    logger.info(f"🔝 Top-k sampling: {(topk_logits > float('-inf')).sum().item()} tokens kept")
    
    # Adaptive sampling
    adaptive_logits = trainer.adaptive_sampling(dummy_logits.clone(), strategy="adaptive")
    logger.info(f"🧠 Adaptive sampling: {(adaptive_logits > float('-inf')).sum().item()} tokens kept")
    
    # 9. Generate training summary
    summary = trainer.get_training_summary()
    logger.info("📊 Training Summary:")
    for key, value in summary.items():
        if key != 'diffusion_config':  # Skip the large config dict
            logger.info(f"  {key}: {value}")
    
    logger.info("✅ Enhanced Diffusion Training Demo completed!")

def sync_training_demo():
    """Synchronous version for simpler usage"""
    logger.info("🔄 Running synchronous training demo...")
    
    # Create trainer without WebSocket for simpler demo
    config = get_small_config()
    trainer = DiffusionTrainingManager(
        config=config,
        output_dir="demo_output/sync_training",
        enable_websocket=False  # Disable WebSocket for sync demo
    )
    
    trainer.setup_model()
    
    # Setup custom features
    trainer.set_custom_loss_weights(text_weight=1.5, control_weight=1.2)
    trainer.freeze_layers(["transformer.layers.0"])
    
    # Create data and train
    train_dataloader = create_dummy_dataloader(batch_size=2, num_batches=10)
    
    # Use synchronous training (no WebSocket updates)
    logger.info("🏃 Running synchronous training...")
    for epoch in range(1):
        for batch_idx, batch in enumerate(train_dataloader):
            metrics = trainer.train_step(batch)
            if batch_idx % 3 == 0:
                logger.info(f"Step {trainer.global_step}: Loss={metrics['total_loss']:.4f}")
    
    logger.info("✅ Synchronous demo completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Diffusion Training Demo")
    parser.add_argument("--mode", choices=["async", "sync"], default="async",
                       help="Demo mode: async (with WebSocket) or sync (simple)")
    parser.add_argument("--websocket-port", type=int, default=8765,
                       help="WebSocket port for real-time updates")
    
    args = parser.parse_args()
    
    if args.mode == "async":
        # Run async demo with WebSocket updates
        logger.info("🌐 Running async demo with WebSocket updates")
        logger.info(f"💡 Connect to ws://localhost:{args.websocket_port} to see live updates")
        asyncio.run(enhanced_training_demo())
    else:
        # Run simple sync demo
        sync_training_demo() 