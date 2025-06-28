#!/usr/bin/env python3
"""
Aggregate preference logs from character directories into a unified dataset for RLHF training.

This script scans character directories for preference_logs.ndjson files and combines them
into a single HuggingFace dataset suitable for TRL trainers.
"""
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datasets import Dataset
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_preference_files(base_path: Path, character_name: Optional[str] = None) -> List[Path]:
    """
    Find all preference_logs.ndjson files in the given path.
    
    Args:
        base_path: Base directory to search (e.g., content/worlds)
        character_name: Optional specific character to search for
        
    Returns:
        List of paths to preference files
    """
    preference_files = []
    
    if character_name:
        # Look for specific character
        pattern = f"**/characters/{character_name}/preference_logs.ndjson"
    else:
        # Find all preference files
        pattern = "**/characters/*/preference_logs.ndjson"
    
    for pref_file in base_path.glob(pattern):
        preference_files.append(pref_file)
        logger.info(f"Found preference file: {pref_file}")
    
    return preference_files


def load_preferences_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load preference entries from a single NDJSON file.
    
    Args:
        file_path: Path to preference_logs.ndjson
        
    Returns:
        List of preference dictionaries
    """
    preferences = []
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        
                        # Validate required fields
                        if all(k in data for k in ['prompt', 'chosen', 'rejected']):
                            # Add metadata
                            data['source_file'] = str(file_path)
                            data['character'] = file_path.parent.name
                            data['world'] = file_path.parents[3].name
                            preferences.append(data)
                        else:
                            logger.warning(f"Missing required fields in {file_path}:{line_num}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error in {file_path}:{line_num} - {e}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
    
    return preferences


def process_rejected_options(preferences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process preferences with multiple rejected options.
    
    Args:
        preferences: List of preference dictionaries
        
    Returns:
        Processed preferences with single rejected option per entry
    """
    processed = []
    
    for pref in preferences:
        rejected = pref.get('rejected', [])
        
        if isinstance(rejected, list) and len(rejected) > 0:
            # Create separate entries for each rejected option
            # This gives us more training samples
            for rej_option in rejected:
                if rej_option:  # Skip empty strings
                    new_pref = pref.copy()
                    new_pref['rejected'] = rej_option
                    processed.append(new_pref)
        elif isinstance(rejected, str) and rejected:
            # Single rejected option
            processed.append(pref)
        else:
            logger.warning(f"Skipping preference with invalid rejected field")
    
    return processed


def aggregate_preferences(
    base_path: Path,
    character_name: Optional[str] = None,
    min_length: int = 10,
    max_duplicates: int = 1
) -> Dataset:
    """
    Aggregate all preference data into a HuggingFace dataset.
    
    Args:
        base_path: Base directory to search
        character_name: Optional specific character
        min_length: Minimum response length in characters
        max_duplicates: Maximum times to include duplicate prompts
        
    Returns:
        HuggingFace Dataset ready for training
    """
    # Find all preference files
    pref_files = find_preference_files(base_path, character_name)
    
    if not pref_files:
        raise ValueError(f"No preference files found in {base_path}")
    
    # Load all preferences
    all_preferences = []
    for pref_file in pref_files:
        preferences = load_preferences_from_file(pref_file)
        all_preferences.extend(preferences)
    
    logger.info(f"Loaded {len(all_preferences)} raw preference entries")
    
    # Process rejected options
    processed_preferences = process_rejected_options(all_preferences)
    logger.info(f"Processed into {len(processed_preferences)} training samples")
    
    # Filter by length
    filtered_preferences = [
        p for p in processed_preferences
        if len(p['chosen']) >= min_length and len(p['rejected']) >= min_length
    ]
    logger.info(f"After length filtering: {len(filtered_preferences)} samples")
    
    # Handle duplicates
    if max_duplicates > 0:
        # Count prompt occurrences
        prompt_counts = defaultdict(int)
        deduplicated = []
        
        for pref in filtered_preferences:
            prompt = pref['prompt']
            if prompt_counts[prompt] < max_duplicates:
                deduplicated.append(pref)
                prompt_counts[prompt] += 1
        
        filtered_preferences = deduplicated
        logger.info(f"After deduplication: {len(filtered_preferences)} samples")
    
    # Create dataset
    if not filtered_preferences:
        raise ValueError("No valid preferences after filtering")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(filtered_preferences)
    
    # Create HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    # Log statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total samples: {len(dataset)}")
    logger.info(f"Unique prompts: {df['prompt'].nunique()}")
    logger.info(f"Characters included: {df['character'].unique().tolist()}")
    logger.info(f"Average chosen length: {df['chosen'].str.len().mean():.1f} chars")
    logger.info(f"Average rejected length: {df['rejected'].str.len().mean():.1f} chars")
    
    return dataset


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Aggregate preference logs for RLHF training"
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("content/worlds"),
        help="Base path to search for preference files (default: content/worlds)"
    )
    parser.add_argument(
        "--character",
        type=str,
        help="Specific character name to aggregate (optional)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/preferences.arrow"),
        help="Output path for aggregated dataset (default: datasets/preferences.arrow)"
    )
    parser.add_argument(
        "--format",
        choices=["arrow", "json", "csv"],
        default="arrow",
        help="Output format (default: arrow)"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum response length in characters (default: 10)"
    )
    parser.add_argument(
        "--max-duplicates",
        type=int,
        default=1,
        help="Maximum times to include duplicate prompts (default: 1, 0=unlimited)"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.0,
        help="Train/test split ratio (default: 0.0 = no split)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Aggregate preferences
        dataset = aggregate_preferences(
            args.base_path,
            args.character,
            args.min_length,
            args.max_duplicates
        )
        
        # Split if requested
        if args.split_ratio > 0:
            split_dataset = dataset.train_test_split(test_size=args.split_ratio)
            train_dataset = split_dataset['train']
            test_dataset = split_dataset['test']
            
            logger.info(f"\nSplit dataset: {len(train_dataset)} train, {len(test_dataset)} test")
            
            # Save based on format
            if args.format == "arrow":
                train_path = args.output.with_suffix('.train.arrow')
                test_path = args.output.with_suffix('.test.arrow')
                train_dataset.save_to_disk(str(train_path.parent / train_path.stem))
                test_dataset.save_to_disk(str(test_path.parent / test_path.stem))
            elif args.format == "json":
                train_path = args.output.with_suffix('.train.json')
                test_path = args.output.with_suffix('.test.json')
                train_dataset.to_json(str(train_path))
                test_dataset.to_json(str(test_path))
            else:  # csv
                train_path = args.output.with_suffix('.train.csv')
                test_path = args.output.with_suffix('.test.csv')
                train_dataset.to_csv(str(train_path))
                test_dataset.to_csv(str(test_path))
            
            logger.info(f"Saved train dataset to: {train_path}")
            logger.info(f"Saved test dataset to: {test_path}")
            
        else:
            # Save full dataset
            if args.format == "arrow":
                dataset.save_to_disk(str(args.output.parent / args.output.stem))
                logger.info(f"Saved dataset to: {args.output.parent / args.output.stem}")
            elif args.format == "json":
                dataset.to_json(str(args.output.with_suffix('.json')))
                logger.info(f"Saved dataset to: {args.output.with_suffix('.json')}")
            else:  # csv
                dataset.to_csv(str(args.output.with_suffix('.csv')))
                logger.info(f"Saved dataset to: {args.output.with_suffix('.csv')}")
        
        logger.info("\nPreference aggregation complete!")
        
    except Exception as e:
        logger.error(f"Error aggregating preferences: {e}")
        raise


if __name__ == "__main__":
    main() 