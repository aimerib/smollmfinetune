"""
Training manager for diffusion multimodal model

Integrates with the existing TrainingManager infrastructure while providing
diffusion-specific training logic and callbacks.
"""

import torch
import torch.nn.functional as F
import logging
import wandb
from typing import Dict, Any, Optional, Tuple, List, Callable
from pathlib import Path
import json
from dataclasses import asdict
import asyncio
import time
import websockets
from websockets.server import WebSocketServerProtocol
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn

from .diffusion_model import DiffusionMultimodalModel, create_diffusion_model
from .diffusion_config import DiffusionMultimodalConfig, get_small_config, get_medium_config, get_large_config
# Remove problematic relative imports for now
# from ..training import TrainingManager  
# from ..config import NarrativeLLMConfig

logger = logging.getLogger(__name__)


# ============================================================================
# WebSocket Live Updates System
# ============================================================================

class TrainingWebSocketManager:
    """Manages WebSocket connections for real-time training updates"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.connections: set[WebSocketServerProtocol] = set()
        self.server = None
        self.is_running = False
        
    async def start_server(self):
        """Start WebSocket server"""
        if self.is_running:
            return
            
        try:
            self.server = await websockets.serve(
                self.handle_connection,
                "localhost", 
                self.port
            )
            self.is_running = True
            logger.info(f"🔗 WebSocket server started on ws://localhost:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def stop_server(self):
        """Stop WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.is_running = False
            logger.info("🔌 WebSocket server stopped")
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        self.connections.add(websocket)
        logger.info(f"📱 New WebSocket connection from {websocket.remote_address}")
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connection_established",
                "message": "Connected to diffusion training updates",
                "timestamp": time.time()
            }))
            
            # Keep connection alive
            async for message in websocket:
                # Handle client messages if needed
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                except json.JSONDecodeError:
                    pass
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connections.discard(websocket)
            logger.info(f"📱 WebSocket connection closed")
    
    async def broadcast_update(self, update: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        if not self.connections:
            return
            
        message = json.dumps({
            **update,
            "timestamp": time.time()
        })
        
        # Send to all connections (remove dead ones)
        dead_connections = set()
        for websocket in self.connections.copy():
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                dead_connections.add(websocket)
        
        # Clean up dead connections
        self.connections -= dead_connections

class DiffusionTrainingManager:
    """Training manager for diffusion multimodal models"""
    
    def __init__(self, 
                 config: DiffusionMultimodalConfig,
                 output_dir: str = "training_output/diffusion",
                 use_wandb: bool = False,
                 enable_websocket: bool = True,
                 websocket_port: int = 8765):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # WebSocket integration
        self.websocket_manager = TrainingWebSocketManager(websocket_port) if enable_websocket else None
        self.executor = ThreadPoolExecutor(max_workers=1) if enable_websocket else None
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="diffusion-multimodal",
                config=asdict(self.config)
            )

    def setup_model(self):
        """Create and initialize the model"""
        self.model = create_diffusion_model(self.config)
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: 1.0
        )
        
        logger.info(f"Model setup complete:")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Weight decay: {self.config.weight_decay}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Apply pre-step architecture hooks
        self.apply_architecture_hooks("pre_step", batch=batch, step=self.global_step)
        
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Extract data
        clean_text = batch['text']
        clean_speech = batch['speech']
        clean_control = batch['control']
        clean_memory = batch['memory']
        character_ids = batch.get('character_ids')
        
        # Generate noise
        noise_text = torch.randn_like(clean_text)
        noise_speech = torch.randn_like(clean_speech)
        noise_control = torch.randn_like(clean_control)
        noise_memory = torch.randn_like(clean_memory)
        
        # Sample timesteps
        batch_size = clean_text.shape[0]
        timesteps = torch.randint(
            0, self.config.scheduler.num_train_timesteps,
            (batch_size,), device=self.device
        )
        
        # Add noise
        noisy_text = self.model.noise_scheduler.add_noise(clean_text, noise_text, timesteps, 'text')
        noisy_speech = self.model.noise_scheduler.add_noise(clean_speech, noise_speech, timesteps, 'speech')
        noisy_control = self.model.noise_scheduler.add_noise(clean_control, noise_control, timesteps, 'control')
        noisy_memory = self.model.noise_scheduler.add_noise(clean_memory, noise_memory, timesteps, 'memory')
        
        # Classifier-free guidance training
        if self.config.guidance_probability > 0:
            # Randomly drop character conditioning for classifier-free guidance
            guidance_mask = torch.rand(batch_size, device=self.device) < self.config.guidance_probability
            if character_ids is not None:
                character_ids = character_ids.clone()
                character_ids[guidance_mask] = -1  # Use -1 to indicate no character conditioning
        
        # Forward pass
        output = self.model(
            noisy_text=noisy_text,
            noisy_speech=noisy_speech,
            noisy_control=noisy_control,
            noisy_memory=noisy_memory,
            timesteps=timesteps,
            character_ids=character_ids,
            target_text=noise_text,
            target_speech=noise_speech,
            target_control=noise_control,
            target_memory=noise_memory
        )
        
        # Apply custom loss weights if configured
        if hasattr(self, 'custom_loss_weights'):
            total_loss = self.apply_custom_loss_weights(output)
        else:
            total_loss = output.total_loss
        
        # Apply gradient accumulation
        total_loss = total_loss / self.config.gradient_accumulation_steps
        
        # Apply mid-step architecture hooks
        self.apply_architecture_hooks("mid_step", 
                                    loss=total_loss, 
                                    output=output, 
                                    step=self.global_step)
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )
        
        # Optimizer step (if accumulation is complete)
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Apply post-step architecture hooks
            self.apply_architecture_hooks("post_step", step=self.global_step)
            
            # Progressive unfreezing check
            if hasattr(self, 'unfreeze_schedule'):
                self.progressive_unfreezing(self.global_step, self.unfreeze_schedule)
            
            # Adaptive learning rate check
            if hasattr(self, 'adaptive_lr_enabled') and self.adaptive_lr_enabled:
                self.adaptive_learning_rate(total_loss.item())
        
        self.global_step += 1
        
        # Return metrics
        return {
            'total_loss': total_loss.item() * self.config.gradient_accumulation_steps,  # Unscale for logging
            'text_loss': output.text_loss.item(),
            'speech_loss': output.speech_loss.item(),
            'control_loss': output.control_loss.item(),
            'memory_loss': output.memory_loss.item(),
            'alignment_loss': output.alignment_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step"""
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Extract data
            clean_text = batch['text_embeddings']
            clean_speech = batch['speech_features']
            clean_control = batch['control_embeddings']
            clean_memory = batch['memory_vectors']
            character_ids = batch.get('character_ids', None)
            
            # Sample random timesteps
            batch_size = clean_text.shape[0]
            timesteps = torch.randint(
                0, self.config.scheduler.num_train_timesteps,
                (batch_size,), device=self.device
            )
            
            # Add noise
            noise_text = torch.randn_like(clean_text)
            noise_speech = torch.randn_like(clean_speech)
            noise_control = torch.randn_like(clean_control)
            noise_memory = torch.randn_like(clean_memory)
            
            noisy_text = self.model.noise_scheduler.add_noise(clean_text, noise_text, timesteps, 'text')
            noisy_speech = self.model.noise_scheduler.add_noise(clean_speech, noise_speech, timesteps, 'speech')
            noisy_control = self.model.noise_scheduler.add_noise(clean_control, noise_control, timesteps, 'control')
            noisy_memory = self.model.noise_scheduler.add_noise(clean_memory, noise_memory, timesteps, 'memory')
            
            # Forward pass
            output = self.model(
                noisy_text=noisy_text,
                noisy_speech=noisy_speech,
                noisy_control=noisy_control,
                noisy_memory=noisy_memory,
                timesteps=timesteps,
                character_ids=character_ids,
                clean_text=clean_text,
                clean_speech=clean_speech,
                clean_control=clean_control,
                clean_memory=clean_memory,
                return_loss=True
            )
            
            return {
                'val_total_loss': output.total_loss.item(),
                'val_text_loss': output.text_loss.item(),
                'val_speech_loss': output.speech_loss.item(),
                'val_control_loss': output.control_loss.item(),
                'val_memory_loss': output.memory_loss.item(),
                'val_alignment_loss': output.alignment_loss.item()
            }
    
    def save_checkpoint(self, save_path: Optional[str] = None):
        """Save training checkpoint"""
        if save_path is None:
            save_path = self.output_dir / f"checkpoint_step_{self.global_step}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'lr_scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'diffusion_config': asdict(self.config),
            'current_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
        
        # Save config separately for easy loading
        config_path = save_path.parent / "diffusion_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'lr_scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        self.global_step = checkpoint.get('current_step', 0)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resumed at step {self.global_step}, epoch {self.current_epoch}")
    
    def generate_samples(self, 
                        batch_size: int = 4,
                        character_ids: Optional[torch.Tensor] = None,
                        guidance_scale: float = None,
                        num_inference_steps: int = None) -> Dict[str, torch.Tensor]:
        """Generate samples using the trained model"""
        
        self.model.eval()
        
        with torch.no_grad():
            samples = self.model.generate(
                batch_size=batch_size,
                character_ids=character_ids,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                device=self.device
            )
        
        return samples
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return {
            'current_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'diffusion_config': asdict(self.config),
            'device': self.device,
            'output_dir': str(self.output_dir)
        }

    # ============================================================================
    # Multi-GPU Support
    # ============================================================================
    
    def setup_multi_gpu(self):
        """Setup multi-GPU training with DataParallel or DistributedDataParallel"""
        if torch.cuda.device_count() > 1:
            logger.info(f"🚀 Using {torch.cuda.device_count()} GPUs for training")
            
            # Use DataParallel for simplicity (DistributedDataParallel is more complex)
            self.model = nn.DataParallel(self.model)
            
            # Adjust batch size and learning rate for multi-GPU
            effective_batch_size = self.config.batch_size * torch.cuda.device_count()
            logger.info(f"📊 Effective batch size: {effective_batch_size}")
            
            return True
        return False
    
    # ============================================================================
    # Enhanced Training Loop with WebSocket Integration
    # ============================================================================
    
    async def train_async(self, 
                         train_dataloader,
                         val_dataloader=None,
                         num_epochs: int = 10,
                         save_every: int = 1000,
                         validate_every: int = 500,
                         log_every: int = 100):
        """Main training loop with async WebSocket updates"""
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
        
        # Setup multi-GPU if available
        self.setup_multi_gpu()
        
        # Start WebSocket server
        if self.websocket_manager:
            await self.websocket_manager.start_server()
        
        try:
            # Training loop
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # Send epoch start update
                if self.websocket_manager:
                    await self.websocket_manager.broadcast_update({
                        "type": "epoch_start",
                        "epoch": epoch,
                        "total_epochs": num_epochs
                    })
                
                # Training phase
                self.model.train()
                epoch_losses = []
                
                for batch_idx, batch in enumerate(train_dataloader):
                    # Training step
                    metrics = self.train_step(batch)
                    epoch_losses.append(metrics['total_loss'])
                    
                    # Logging
                    if self.global_step % log_every == 0:
                        avg_loss = sum(epoch_losses[-log_every:]) / len(epoch_losses[-log_every:])
                        
                        log_data = {
                            "type": "training_step",
                            "step": self.global_step,
                            "epoch": epoch,
                            "batch": batch_idx,
                            "loss": avg_loss,
                            "learning_rate": metrics['learning_rate'],
                            "gpu_memory": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                        }
                        
                        logger.info(f"Step {self.global_step}: Loss={avg_loss:.4f}, LR={metrics['learning_rate']:.2e}")
                        
                        # Send WebSocket update
                        if self.websocket_manager:
                            await self.websocket_manager.broadcast_update(log_data)
                        
                        # Wandb logging
                        if self.use_wandb:
                            wandb.log(log_data, step=self.global_step)
                    
                    # Validation
                    if val_dataloader and self.global_step % validate_every == 0:
                        val_metrics = await self.validate_async(val_dataloader)
                        
                        # Send validation update
                        if self.websocket_manager:
                            await self.websocket_manager.broadcast_update({
                                "type": "validation",
                                "step": self.global_step,
                                "epoch": epoch,
                                "val_loss": val_metrics['total_loss'],
                                "val_metrics": val_metrics
                            })
                        
                        # Save best model
                        if val_metrics['total_loss'] < self.best_loss:
                            self.best_loss = val_metrics['total_loss']
                            self.save_checkpoint(self.output_dir / "best_model.pt")
                            
                            if self.websocket_manager:
                                await self.websocket_manager.broadcast_update({
                                    "type": "best_model_saved",
                                    "step": self.global_step,
                                    "best_loss": self.best_loss
                                })
                    
                    # Save checkpoint
                    if self.global_step % save_every == 0:
                        self.save_checkpoint()
                        
                        if self.websocket_manager:
                            await self.websocket_manager.broadcast_update({
                                "type": "checkpoint_saved",
                                "step": self.global_step,
                                "epoch": epoch
                            })
                
                # End of epoch
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                
                logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s, avg loss: {avg_epoch_loss:.4f}")
                
                # Send epoch end update
                if self.websocket_manager:
                    await self.websocket_manager.broadcast_update({
                        "type": "epoch_end",
                        "epoch": epoch,
                        "epoch_time": epoch_time,
                        "avg_loss": avg_epoch_loss,
                        "total_steps": self.global_step
                    })
            
            # Training complete
            if self.websocket_manager:
                await self.websocket_manager.broadcast_update({
                    "type": "training_complete",
                    "total_epochs": num_epochs,
                    "total_steps": self.global_step,
                    "best_loss": self.best_loss
                })
                
        finally:
            # Clean up WebSocket server
            if self.websocket_manager:
                await self.websocket_manager.stop_server()
    
    def train(self, *args, **kwargs):
        """Synchronous wrapper for train_async"""
        return asyncio.run(self.train_async(*args, **kwargs))
    
    async def validate_async(self, val_dataloader) -> Dict[str, float]:
        """Async validation with progress updates"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                metrics = self.validate_step(batch)
                val_losses.append(metrics['total_loss'])
                
                # Send progress update every 10 batches
                if batch_idx % 10 == 0 and self.websocket_manager:
                    await self.websocket_manager.broadcast_update({
                        "type": "validation_progress",
                        "batch": batch_idx,
                        "total_batches": len(val_dataloader),
                        "current_loss": metrics['total_loss']
                    })
        
        avg_metrics = {
            'total_loss': sum(val_losses) / len(val_losses),
            'num_batches': len(val_losses)
        }
        
        return avg_metrics

    # ============================================================================
    # Advanced Sampling Strategies
    # ============================================================================
    
    def nucleus_sampling(self, logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
        """Apply nucleus (top-p) sampling to logits"""
        if temperature != 1.0:
            logits = logits / temperature
            
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Create mask for tokens to keep (cumulative probability <= top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Set logits to -inf for tokens to remove
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def top_k_sampling(self, logits: torch.Tensor, top_k: int = 50, temperature: float = 1.0) -> torch.Tensor:
        """Apply top-k sampling to logits"""
        if temperature != 1.0:
            logits = logits / temperature
            
        # Get top-k values and indices
        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        
        # Create mask for tokens to keep
        indices_to_remove = torch.ones_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=top_k_indices, value=False)
        
        # Set logits to -inf for tokens to remove
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def adaptive_sampling(self, logits: torch.Tensor, 
                         strategy: str = "nucleus", 
                         temperature: float = 1.0,
                         top_p: float = 0.9,
                         top_k: int = 50,
                         min_tokens_to_keep: int = 1) -> torch.Tensor:
        """Apply adaptive sampling strategy based on confidence"""
        if strategy == "nucleus":
            return self.nucleus_sampling(logits, top_p, temperature)
        elif strategy == "top_k":
            return self.top_k_sampling(logits, top_k, temperature)
        elif strategy == "adaptive":
            # Use nucleus for high-confidence, top-k for low-confidence
            confidence = torch.softmax(logits, dim=-1).max(dim=-1)[0]
            high_confidence = confidence > 0.7
            
            result_logits = logits.clone()
            if high_confidence.any():
                result_logits[high_confidence] = self.nucleus_sampling(
                    logits[high_confidence], top_p, temperature
                )
            if (~high_confidence).any():
                result_logits[~high_confidence] = self.top_k_sampling(
                    logits[~high_confidence], top_k, temperature
                )
            return result_logits
        else:
            # Default temperature scaling
            return logits / temperature if temperature != 1.0 else logits

    # ============================================================================
    # Custom Architecture Modifications
    # ============================================================================
    
    def freeze_layers(self, layer_patterns: List[str]):
        """Freeze specific layers based on name patterns"""
        frozen_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += 1
            
            # Check if this parameter matches any freeze pattern
            should_freeze = any(pattern in name for pattern in layer_patterns)
            
            if should_freeze:
                param.requires_grad = False
                frozen_params += 1
                logger.info(f"❄️  Frozen parameter: {name}")
        
        logger.info(f"🧊 Frozen {frozen_params}/{total_params} parameters")
        
        # Update optimizer to only include trainable parameters
        if self.optimizer:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
    
    def unfreeze_layers(self, layer_patterns: List[str]):
        """Unfreeze specific layers based on name patterns"""
        unfrozen_params = 0
        
        for name, param in self.model.named_parameters():
            # Check if this parameter matches any unfreeze pattern
            should_unfreeze = any(pattern in name for pattern in layer_patterns)
            
            if should_unfreeze and not param.requires_grad:
                param.requires_grad = True
                unfrozen_params += 1
                logger.info(f"🔥 Unfrozen parameter: {name}")
        
        logger.info(f"🔓 Unfrozen {unfrozen_params} parameters")
    
    def set_custom_loss_weights(self, 
                               text_weight: float = 1.0,
                               speech_weight: float = 1.0, 
                               control_weight: float = 1.0,
                               memory_weight: float = 1.0,
                               alignment_weight: float = 0.1):
        """Set custom loss weights for different modalities"""
        self.custom_loss_weights = {
            'text': text_weight,
            'speech': speech_weight,
            'control': control_weight,
            'memory': memory_weight,
            'alignment': alignment_weight
        }
        logger.info(f"🎯 Custom loss weights set: {self.custom_loss_weights}")
    
    def apply_custom_loss_weights(self, output) -> torch.Tensor:
        """Apply custom loss weights to model output"""
        if not hasattr(self, 'custom_loss_weights'):
            return output.total_loss
        
        weighted_loss = (
            output.text_loss * self.custom_loss_weights['text'] +
            output.speech_loss * self.custom_loss_weights['speech'] +
            output.control_loss * self.custom_loss_weights['control'] +
            output.memory_loss * self.custom_loss_weights['memory'] +
            output.alignment_loss * self.custom_loss_weights['alignment']
        )
        
        return weighted_loss
    
    def add_architecture_hook(self, hook_name: str, hook_fn: Callable):
        """Add custom architecture hooks for model modifications"""
        if not hasattr(self, 'architecture_hooks'):
            self.architecture_hooks = {}
        
        self.architecture_hooks[hook_name] = hook_fn
        logger.info(f"🪝 Added architecture hook: {hook_name}")
    
    def apply_architecture_hooks(self, stage: str, **kwargs):
        """Apply architecture hooks at specific training stages"""
        if not hasattr(self, 'architecture_hooks'):
            return
        
        for hook_name, hook_fn in self.architecture_hooks.items():
            if stage in hook_name or 'all' in hook_name:
                try:
                    hook_fn(self, **kwargs)
                except Exception as e:
                    logger.warning(f"⚠️  Hook {hook_name} failed: {e}")
    
    def progressive_unfreezing(self, current_step: int, unfreeze_schedule: Dict[int, List[str]]):
        """Progressive unfreezing based on training steps"""
        for step_threshold, layer_patterns in unfreeze_schedule.items():
            if current_step == step_threshold:
                self.unfreeze_layers(layer_patterns)
                logger.info(f"🎯 Progressive unfreezing at step {current_step}")
    
    def adaptive_learning_rate(self, current_loss: float, patience: int = 5):
        """Adaptive learning rate based on loss plateaus"""
        if not hasattr(self, 'loss_history'):
            self.loss_history = []
            self.plateau_counter = 0
            self.best_loss_lr = float('inf')
        
        self.loss_history.append(current_loss)
        
        # Check for plateau (no improvement for patience steps)
        if current_loss < self.best_loss_lr:
            self.best_loss_lr = current_loss
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
        
        # Reduce learning rate on plateau
        if self.plateau_counter >= patience:
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = current_lr * 0.5
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            logger.info(f"📉 Learning rate reduced: {current_lr:.2e} → {new_lr:.2e}")
            self.plateau_counter = 0

def create_diffusion_trainer(config_name: str = "small", **kwargs) -> DiffusionTrainingManager:
    """Factory function to create diffusion trainer with predefined configs"""
    
    config_map = {
        "small": get_small_config,
        "medium": get_medium_config,
        "large": get_large_config
    }
    
    if config_name not in config_map:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(config_map.keys())}")
    
    config = config_map[config_name]()
    
    # Override any config parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")
    
    return DiffusionTrainingManager(config)
