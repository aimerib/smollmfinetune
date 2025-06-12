#!/usr/bin/env python3
"""
Model Cache Pre-loader Script
=============================

This script pre-downloads and caches HuggingFace models to avoid rate limiting
during dataset generation. Run this before starting your application.

Usage:
    python cache-model.py --model ArliAI/QwQ-32B-ArliAI-RpR-v4
    python cache-model.py --list-popular
    python cache-model.py --cache-info
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_cache_dirs():
    """Setup HuggingFace cache directories"""
    if Path("/workspace").exists():
        cache_dir = "/workspace/.cache/vllm_hf"
    else:
        cache_dir = f"{os.path.expanduser('~')}/.cache/huggingface"
    
    # Set environment variables
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    
    # Create directories
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Cache directory: {cache_dir}")
    return cache_dir

def cache_model(model_name: str, cache_dir: str):
    """Download and cache a model"""
    logger.info(f"🔽 Downloading model: {model_name}")
    
    try:
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
        
        # Download model files
        logger.info("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        logger.info("📥 Downloading config...")
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        
        logger.info("📥 Downloading model weights (this may take a while)...")
        # Just download the model files, don't load into memory
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True
        )
        
        logger.info(f"✅ Successfully cached model: {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to cache model {model_name}: {e}")
        return False

def list_cached_models(cache_dir: str):
    """List all cached models"""
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        logger.info("📭 No cache directory found")
        return
    
    # Look for model directories
    model_dirs = []
    for item in cache_path.rglob("*"):
        if item.is_dir() and (item / "config.json").exists():
            # Calculate size
            size_bytes = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            size_gb = size_bytes / (1024**3)
            model_dirs.append((str(item.relative_to(cache_path)), size_gb))
    
    if model_dirs:
        logger.info(f"📦 Found {len(model_dirs)} cached models:")
        for model_path, size_gb in sorted(model_dirs):
            logger.info(f"   {model_path} ({size_gb:.1f} GB)")
    else:
        logger.info("📭 No cached models found")

def get_popular_models() -> List[str]:
    """Get list of popular models for caching"""
    return [
        "PocketDoc/Dans-PersonalityEngine-V1.3.0-24b",
        "ArliAI/QwQ-32B-ArliAI-RpR-v4", 
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "NousResearch/Hermes-3-Llama-3.1-70B",
        "Qwen/Qwen2.5-72B-Instruct",
        "microsoft/Phi-3-medium-4k-instruct"
    ]

def test_offline_mode(model_name: str):
    """Test if a model can be loaded in offline mode"""
    logger.info(f"🔍 Testing offline access for: {model_name}")
    
    # Force offline mode
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        logger.info(f"✅ Model available offline: {model_name}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Model not available offline: {e}")
        return False
    finally:
        # Remove offline mode
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)

def main():
    parser = argparse.ArgumentParser(description="HuggingFace Model Cache Manager")
    parser.add_argument("--model", type=str, help="Model to cache (HuggingFace model ID)")
    parser.add_argument("--list-popular", action="store_true", help="List popular models")
    parser.add_argument("--cache-info", action="store_true", help="Show cache information")
    parser.add_argument("--test-offline", type=str, help="Test if a model is available offline")
    parser.add_argument("--cache-popular", action="store_true", help="Cache all popular models")
    
    args = parser.parse_args()
    
    # Setup cache directory
    cache_dir = setup_cache_dirs()
    
    if args.list_popular:
        popular_models = get_popular_models()
        logger.info("📋 Popular models:")
        for i, model in enumerate(popular_models, 1):
            logger.info(f"   {i:2d}. {model}")
        return
    
    if args.cache_info:
        list_cached_models(cache_dir)
        return
    
    if args.test_offline:
        test_offline_mode(args.test_offline)
        return
    
    if args.cache_popular:
        popular_models = get_popular_models()
        logger.info(f"🔽 Caching {len(popular_models)} popular models...")
        
        success_count = 0
        for i, model in enumerate(popular_models, 1):
            logger.info(f"📥 [{i}/{len(popular_models)}] Caching: {model}")
            if cache_model(model, cache_dir):
                success_count += 1
        
        logger.info(f"🎉 Successfully cached {success_count}/{len(popular_models)} models")
        return
    
    if args.model:
        cache_model(args.model, cache_dir)
        return
    
    # No arguments provided, show help
    parser.print_help()
    logger.info("\n💡 Examples:")
    logger.info("   python cache-model.py --model ArliAI/QwQ-32B-ArliAI-RpR-v4")
    logger.info("   python cache-model.py --list-popular")
    logger.info("   python cache-model.py --cache-info")
    logger.info("   python cache-model.py --test-offline ArliAI/QwQ-32B-ArliAI-RpR-v4")

if __name__ == "__main__":
    main() 