#!/usr/bin/env python3
"""
Dataset Registration CLI Tool

Command-line interface for registering datasets in the versioning system.
Computes SHA256 hashes, counts samples, and creates/updates dataset.lock files.

Usage:
    python scripts/dataset_register.py path/to/dataset --name "My Dataset"
    python scripts/dataset_register.py datasets/character_chat/ --name "character_chat"
    python scripts/dataset_register.py datasets/*/  --name-from-path  # Bulk registration
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add app to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.dataset_versioning import DatasetVersioning, DatasetManifest


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI tool"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def register_single_dataset(
    versioning: DatasetVersioning,
    dataset_path: str,
    name: str,
    source_url: Optional[str] = None,
    verbose: bool = False
) -> bool:
    """
    Register a single dataset.
    
    Args:
        versioning: DatasetVersioning instance
        dataset_path: Path to dataset
        name: Dataset name
        source_url: Optional source URL
        verbose: Whether to show verbose output
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"📊 Registering dataset: {dataset_path}")
        
        # Check if path exists
        path_obj = Path(dataset_path)
        if not path_obj.exists():
            print(f"❌ Error: Dataset path does not exist: {dataset_path}")
            return False
        
        # Register the dataset
        manifest = versioning.register(dataset_path, name, source_url)
        
        # Show results
        print(f"✅ Dataset registered successfully!")
        print(f"   Name: {manifest.name}")
        print(f"   Hash: {manifest.hash_sha256[:16]}...")
        print(f"   Samples: {manifest.num_samples:,}")
        print(f"   Created: {manifest.created_at}")
        
        if verbose:
            print(f"   Full Hash: {manifest.hash_sha256}")
            print(f"   Source URL: {manifest.source_url}")
            print(f"   Schema Version: {manifest.schema_version}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error registering dataset {dataset_path}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def register_bulk_datasets(
    versioning: DatasetVersioning,
    pattern: str,
    name_from_path: bool = False,
    name_prefix: str = "",
    verbose: bool = False
) -> int:
    """
    Register multiple datasets using a glob pattern.
    
    Args:
        versioning: DatasetVersioning instance
        pattern: Glob pattern for dataset paths
        name_from_path: Whether to derive names from paths
        name_prefix: Prefix for generated names
        verbose: Whether to show verbose output
        
    Returns:
        Number of datasets successfully registered
    """
    from glob import glob
    
    paths = glob(pattern)
    if not paths:
        print(f"❌ No datasets found matching pattern: {pattern}")
        return 0
    
    print(f"🔍 Found {len(paths)} datasets matching pattern: {pattern}")
    
    success_count = 0
    
    for path in sorted(paths):
        path_obj = Path(path)
        
        if not path_obj.exists():
            print(f"⚠️  Skipping non-existent path: {path}")
            continue
        
        # Generate name from path if requested
        if name_from_path:
            if path_obj.is_file():
                name = name_prefix + path_obj.stem
            else:
                name = name_prefix + path_obj.name
        else:
            name = name_prefix + path_obj.name
        
        if register_single_dataset(versioning, path, name, verbose=verbose):
            success_count += 1
        
        print()  # Add spacing between registrations
    
    print(f"📈 Bulk registration complete: {success_count}/{len(paths)} datasets registered")
    return success_count


def list_registered_datasets(root_path: str = "datasets") -> None:
    """
    List all registered datasets in a directory tree.
    
    Args:
        root_path: Root directory to search for dataset.lock files
    """
    import json
    
    root = Path(root_path)
    if not root.exists():
        print(f"❌ Root path does not exist: {root_path}")
        return
    
    lock_files = list(root.rglob("dataset.lock"))
    
    if not lock_files:
        print(f"📂 No registered datasets found in {root_path}")
        return
    
    print(f"📋 Registered datasets in {root_path}:")
    print("-" * 80)
    print(f"{'Name':<25} {'Hash':<16} {'Samples':<10} {'Created':<20}")
    print("-" * 80)
    
    for lock_file in sorted(lock_files):
        try:
            with open(lock_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            name = data.get('name', 'Unknown')[:24]
            hash_short = data.get('hash_sha256', 'Unknown')[:12]
            samples = data.get('num_samples', 0)
            created = data.get('created_at', 'Unknown')[:19]
            
            print(f"{name:<25} {hash_short:<16} {samples:<10,} {created:<20}")
            
        except Exception as e:
            print(f"❌ Error reading {lock_file}: {e}")
    
    print(f"\nTotal: {len(lock_files)} registered datasets")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Dataset Registration Tool - Version and track datasets with SHA256 hashes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register single dataset
  python scripts/dataset_register.py datasets/character_chat/ --name "Character Chat Dataset"
  
  # Register with custom source URL
  python scripts/dataset_register.py data.json --name "training_data" --source-url "s3://bucket/data.json"
  
  # Bulk register all datasets in a directory
  python scripts/dataset_register.py "datasets/*/" --name-from-path --name-prefix "dataset_"
  
  # List all registered datasets
  python scripts/dataset_register.py --list
  
  # List datasets in custom directory
  python scripts/dataset_register.py --list --root-path "my_datasets/"
        """
    )
    
    # Main arguments
    parser.add_argument('path', nargs='?', 
                       help='Path to dataset file or directory (or glob pattern for bulk registration)')
    
    parser.add_argument('--name', '-n', 
                       help='Human-readable name for the dataset')
    
    parser.add_argument('--source-url', '-s',
                       help='Source URL for the dataset (optional)')
    
    # Bulk registration options
    parser.add_argument('--name-from-path', action='store_true',
                       help='Generate dataset names from file/directory names')
    
    parser.add_argument('--name-prefix', default='',
                       help='Prefix to add to generated names (used with --name-from-path)')
    
    # Listing options
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all registered datasets')
    
    parser.add_argument('--root-path', default='datasets',
                       help='Root path for listing datasets (default: datasets)')
    
    # General options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Configure logging
    setup_logging(args.verbose)
    
    # Handle listing mode
    if args.list:
        list_registered_datasets(args.root_path)
        return
    
    # Validate required arguments for registration
    if not args.path:
        parser.error("Dataset path is required (or use --list to list datasets)")
    
    # Check if this is bulk registration (contains wildcards)
    is_bulk = '*' in args.path or '?' in args.path or '[' in args.path
    
    if is_bulk:
        if not args.name_from_path and not args.name:
            parser.error("For bulk registration, either --name-from-path or --name must be specified")
    else:
        if not args.name:
            parser.error("Dataset name is required (use --name or -n)")
    
    # Initialize versioning system
    versioning = DatasetVersioning()
    
    try:
        if is_bulk:
            # Bulk registration
            success_count = register_bulk_datasets(
                versioning,
                args.path,
                args.name_from_path,
                args.name_prefix,
                args.verbose
            )
            
            if success_count == 0:
                sys.exit(1)
        
        else:
            # Single dataset registration
            success = register_single_dataset(
                versioning,
                args.path,
                args.name,
                args.source_url,
                args.verbose
            )
            
            if not success:
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Registration interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 