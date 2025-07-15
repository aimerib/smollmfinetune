#!/usr/bin/env python
"""
Launch the Production Inference Engine server.

This script starts the high-performance inference server with:
- FastAPI endpoints for generation
- Triple-head model support
- Hot-swappable adapters
- Memory integration
- WebSocket support for real-time updates
"""

import asyncio
import logging
import sys
from pathlib import Path
import uvicorn
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.app.inference_engine import ProductionInferenceEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def start_server(host: str = "0.0.0.0", port: int = 8000, 
                      config: dict = None):
    """Start the inference server"""
    
    # Create engine with configuration
    engine = ProductionInferenceEngine(config)
    
    # Initialize engine
    logger.info("Initializing inference engine...")
    await engine.initialize()
    
    # Log configuration
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Configuration: {engine.config}")
    
    # Start uvicorn server
    config = uvicorn.Config(
        app=engine.app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        ws_ping_interval=20,
        ws_ping_timeout=20
    )
    
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    finally:
        # Cleanup
        logger.info("Shutting down engine...")
        await engine.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Production Inference Engine Server"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)"
    )
    
    parser.add_argument(
        "--enable-vllm",
        action="store_true",
        help="Enable vLLM optimization"
    )
    
    parser.add_argument(
        "--enable-triple-head",
        action="store_true", 
        help="Enable triple-head model features"
    )
    
    parser.add_argument(
        "--gpu-memory-threshold",
        type=float,
        default=0.85,
        help="GPU memory threshold for auto-recovery (default: 0.85)"
    )
    
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=300,
        help="Attention cache TTL in seconds (default: 300)"
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        "max_concurrent": args.max_concurrent,
        "enable_vllm": args.enable_vllm,
        "enable_triple_head": args.enable_triple_head,
        "gpu_memory_threshold": args.gpu_memory_threshold,
        "cache_ttl": args.cache_ttl
    }
    
    # Print startup banner
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║         Production Inference Engine for Narrative-LLM     ║
    ║                                                           ║
    ║  Features:                                                ║
    ║  • Triple-head model support (Generation + Control + Memory) ║
    ║  • Hot-swappable character adapters                      ║
    ║  • GPU-optimized inference with attention caching        ║
    ║  • Real-time memory formation and retrieval              ║
    ║  • WebSocket support for mobile clients                  ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    logger.info(f"Starting server with configuration: {config}")
    
    # Run async server
    try:
        asyncio.run(start_server(args.host, args.port, config))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 