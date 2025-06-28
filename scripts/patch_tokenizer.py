#!/usr/bin/env python3
"""
Tokenizer patching script for control tokens.

This script loads a base model's tokenizer, adds control tokens from core_data/tokens.json,
and saves the patched tokenizer to a cached directory for reuse.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers library not found. Please install with: pip install transformers")
    exit(1)

logger = logging.getLogger(__name__)


def load_control_tokens(tokens_file: str = "core_data/tokens.json") -> List[str]:
    """Load control tokens from JSON file"""
    tokens_path = Path(tokens_file)
    
    if not tokens_path.exists():
        logger.error(f"Tokens file not found: {tokens_path}")
        return []
    
    try:
        with open(tokens_path, 'r') as f:
            tokens_data = json.load(f)
        
        # Extract just the token strings
        tokens = [item["token"] for item in tokens_data if "token" in item]
        logger.info(f"Loaded {len(tokens)} control tokens from {tokens_file}")
        return tokens
        
    except Exception as e:
        logger.error(f"Failed to load tokens: {e}")
        return []


def patch_tokenizer(base_model_name: str, tokens_file: str = "core_data/tokens.json", 
                   cache_dir: str = ".cache/tokenizers") -> str:
    """
    Patch a tokenizer with control tokens and cache the result.
    
    Args:
        base_model_name: Name of the base model (e.g., "meta-llama/Llama-2-7b-chat-hf")
        tokens_file: Path to tokens JSON file
        cache_dir: Directory to cache patched tokenizers
        
    Returns:
        Path to the patched tokenizer
    """
    # Load control tokens
    control_tokens = load_control_tokens(tokens_file)
    if not control_tokens:
        raise ValueError("No control tokens found")
    
    # Create cache directory
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Generate cache name
    safe_model_name = base_model_name.replace("/", "_").replace("-", "_")
    patched_name = f"{safe_model_name}_patched"
    patched_path = cache_path / patched_name
    
    # Check if already patched
    if patched_path.exists():
        logger.info(f"Using cached patched tokenizer: {patched_path}")
        return str(patched_path)
    
    try:
        # Load base tokenizer
        logger.info(f"Loading base tokenizer: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Get current vocabulary
        current_vocab = tokenizer.get_vocab()
        
        # Find tokens that need to be added
        tokens_to_add = []
        for token in control_tokens:
            if token not in current_vocab:
                tokens_to_add.append(token)
        
        if tokens_to_add:
            logger.info(f"Adding {len(tokens_to_add)} new tokens: {tokens_to_add}")
            
            # Add the new tokens
            tokenizer.add_tokens(tokens_to_add)
            
            # Save patched tokenizer
            tokenizer.save_pretrained(patched_path)
            logger.info(f"Saved patched tokenizer to: {patched_path}")
            
            # Verify tokens were added correctly
            new_vocab = tokenizer.get_vocab()
            for token in tokens_to_add:
                if token not in new_vocab:
                    logger.warning(f"Token {token} was not properly added to vocabulary")
        else:
            logger.info("All control tokens already present in tokenizer")
            # Still save a copy for consistency
            tokenizer.save_pretrained(patched_path)
        
        return str(patched_path)
        
    except Exception as e:
        logger.error(f"Failed to patch tokenizer: {e}")
        raise


def verify_patched_tokenizer(patched_path: str, control_tokens: List[str]) -> bool:
    """Verify that a patched tokenizer contains all control tokens"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(patched_path)
        vocab = tokenizer.get_vocab()
        
        missing_tokens = []
        for token in control_tokens:
            if token not in vocab:
                missing_tokens.append(token)
        
        if missing_tokens:
            logger.error(f"Missing tokens in patched tokenizer: {missing_tokens}")
            return False
        
        # Test encoding/decoding round trip
        test_text = " ".join(control_tokens[:3])  # Test first 3 tokens
        encoded = tokenizer.encode(test_text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded)
        
        if test_text.strip() != decoded.strip():
            logger.warning(f"Round-trip test failed: '{test_text}' -> '{decoded}'")
            return False
        
        logger.info(f"Patched tokenizer verification passed: {len(control_tokens)} tokens")
        return True
        
    except Exception as e:
        logger.error(f"Tokenizer verification failed: {e}")
        return False


def main():
    """Command-line interface for tokenizer patching"""
    parser = argparse.ArgumentParser(description="Patch tokenizer with control tokens")
    parser.add_argument("model_name", help="Base model name (e.g., meta-llama/Llama-2-7b-chat-hf)")
    parser.add_argument("--tokens-file", default="core_data/tokens.json", 
                       help="Path to tokens JSON file")
    parser.add_argument("--cache-dir", default=".cache/tokenizers",
                       help="Cache directory for patched tokenizers")
    parser.add_argument("--verify", action="store_true",
                       help="Verify the patched tokenizer")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Patch the tokenizer
        patched_path = patch_tokenizer(
            base_model_name=args.model_name,
            tokens_file=args.tokens_file,
            cache_dir=args.cache_dir
        )
        
        print(f"✅ Patched tokenizer saved to: {patched_path}")
        
        # Verify if requested
        if args.verify:
            control_tokens = load_control_tokens(args.tokens_file)
            if verify_patched_tokenizer(patched_path, control_tokens):
                print("✅ Tokenizer verification passed")
            else:
                print("❌ Tokenizer verification failed")
                exit(1)
                
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)


if __name__ == "__main__":
    main() 