#!/usr/bin/env python3
"""
Dataset Lock Consistency Checker

CI script that validates dataset consistency by checking that all datasets
have valid dataset.lock files and that the content hashes match.

Used in GitHub Actions to fail PRs when datasets change without lock file updates.

Usage:
    python scripts/check_dataset_lock.py                    # Check all datasets
    python scripts/check_dataset_lock.py --path datasets/   # Check specific path
    python scripts/check_dataset_lock.py --changed-only     # Check only changed files (for CI)
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Set, Tuple, Optional

# Add app to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.dataset_versioning import DatasetVersioning, ConsistencyResult


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CI tool"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def get_changed_files() -> Set[str]:
    """
    Get list of changed files from git.
    
    Returns:
        Set of changed file paths relative to repo root
    """
    try:
        # Get changed files in current PR/commit
        # Try multiple git commands to handle different CI scenarios
        
        # First try: compare with main branch
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'origin/main...HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip():
                return set(result.stdout.strip().split('\n'))
        except subprocess.CalledProcessError:
            pass
        
        # Second try: get staged files
        try:
            result = subprocess.run(
                ['git', 'diff', '--cached', '--name-only'],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip():
                return set(result.stdout.strip().split('\n'))
        except subprocess.CalledProcessError:
            pass
        
        # Third try: get unstaged changes
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only'],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip():
                return set(result.stdout.strip().split('\n'))
        except subprocess.CalledProcessError:
            pass
        
        # Last resort: compare with HEAD~1
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD~1'],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout.strip():
            return set(result.stdout.strip().split('\n'))
        else:
            return set()
            
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to get changed files from git: {e}")
        return set()


def find_dataset_directories(root_path: str) -> List[str]:
    """
    Find all directories that contain dataset files.
    
    Args:
        root_path: Root directory to search
        
    Returns:
        List of dataset directories
    """
    root = Path(root_path)
    if not root.exists():
        return []
    
    dataset_dirs = set()
    
    # Look for common dataset file patterns
    dataset_patterns = ['*.json', '*.jsonl', '*.csv', '*.parquet', '*.arrow']
    
    for pattern in dataset_patterns:
        for file_path in root.rglob(pattern):
            # Skip hidden files and common non-dataset files
            if (file_path.name.startswith('.') or 
                file_path.name == 'dataset.lock' or
                'checkpoint' in str(file_path).lower() or
                '__pycache__' in str(file_path)):
                continue
            
            dataset_dirs.add(str(file_path.parent))
    
    return sorted(list(dataset_dirs))


def identify_changed_datasets(changed_files: Set[str], dataset_dirs: List[str]) -> List[str]:
    """
    Identify which dataset directories have changed files.
    
    Args:
        changed_files: Set of changed file paths
        dataset_dirs: List of all dataset directories
        
    Returns:
        List of dataset directories that have changes
    """
    changed_datasets = set()
    
    for changed_file in changed_files:
        changed_path = Path(changed_file)
        
        # Check if the changed file is in any dataset directory
        for dataset_dir in dataset_dirs:
            dataset_path = Path(dataset_dir)
            
            # Check if changed file is within this dataset directory
            try:
                changed_path.relative_to(dataset_path)
                changed_datasets.add(dataset_dir)
                break
            except ValueError:
                # Path is not within this dataset directory
                continue
    
    return sorted(list(changed_datasets))


def check_single_dataset(versioning: DatasetVersioning, dataset_path: str, verbose: bool = False) -> Tuple[bool, str]:
    """
    Check consistency of a single dataset.
    
    Args:
        versioning: DatasetVersioning instance
        dataset_path: Path to dataset directory
        verbose: Whether to show verbose output
        
    Returns:
        Tuple of (is_consistent, error_message)
    """
    try:
        result = versioning.check_consistency(dataset_path)
        
        if result.is_consistent:
            if verbose:
                print(f"✅ {dataset_path}: Consistent (hash: {result.actual_hash[:12]}...)")
            return True, ""
        else:
            error_msg = f"❌ {dataset_path}: {result.error_message or 'Hash mismatch'}"
            if result.expected_hash and result.actual_hash:
                error_msg += f"\n   Expected: {result.expected_hash[:16]}..."
                error_msg += f"\n   Actual:   {result.actual_hash[:16]}..."
            print(error_msg)
            return False, error_msg
            
    except Exception as e:
        error_msg = f"❌ {dataset_path}: Error during check - {e}"
        print(error_msg)
        return False, error_msg


def suggest_fix_command(dataset_path: str) -> str:
    """
    Generate a command suggestion to fix a dataset consistency issue.
    
    Args:
        dataset_path: Path to the inconsistent dataset
        
    Returns:
        Suggested fix command
    """
    dataset_name = Path(dataset_path).name
    return f"python scripts/dataset_register.py {dataset_path} --name {dataset_name}"


def main():
    """Main CI check entry point"""
    parser = argparse.ArgumentParser(
        description='Dataset Lock Consistency Checker - Validate dataset versioning for CI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script is designed to be used in CI/CD pipelines to ensure dataset
integrity and prevent silent dataset drift.

Exit codes:
  0 - All datasets are consistent
  1 - One or more datasets are inconsistent
  2 - Script error (e.g., invalid arguments)

Examples:
  # Check all datasets
  python scripts/check_dataset_lock.py
  
  # Check specific path
  python scripts/check_dataset_lock.py --path datasets/character_chat/
  
  # Check only datasets that changed in current PR (for CI)
  python scripts/check_dataset_lock.py --changed-only
  
  # Verbose output for debugging
  python scripts/check_dataset_lock.py --verbose
        """
    )
    
    parser.add_argument('--path', default='datasets',
                       help='Path to check for datasets (default: datasets)')
    
    parser.add_argument('--changed-only', action='store_true',
                       help='Only check datasets that have changed files (for CI)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('--fail-fast', action='store_true',
                       help='Exit immediately on first consistency failure')
    
    args = parser.parse_args()
    
    # Configure logging
    setup_logging(args.verbose)
    
    try:
        # Initialize versioning system
        versioning = DatasetVersioning()
        
        # Determine which datasets to check
        if args.changed_only:
            print("🔍 Checking only datasets with changed files...")
            
            changed_files = get_changed_files()
            if not changed_files:
                print("ℹ️  No changed files detected - skipping dataset checks")
                return
            
            if args.verbose:
                print(f"Changed files: {sorted(changed_files)}")
            
            all_dataset_dirs = find_dataset_directories(args.path)
            dataset_dirs = identify_changed_datasets(changed_files, all_dataset_dirs)
            
            if not dataset_dirs:
                print("ℹ️  No dataset directories have changed files")
                return
            
            print(f"📊 Found {len(dataset_dirs)} dataset directories with changes:")
            for dataset_dir in dataset_dirs:
                print(f"   - {dataset_dir}")
        
        else:
            print(f"🔍 Checking all datasets in {args.path}...")
            dataset_dirs = find_dataset_directories(args.path)
            
            if not dataset_dirs:
                print(f"ℹ️  No dataset directories found in {args.path}")
                return
            
            print(f"📊 Found {len(dataset_dirs)} dataset directories")
        
        # Check each dataset
        print("\n" + "="*60)
        print("DATASET CONSISTENCY CHECK")
        print("="*60)
        
        failed_datasets = []
        total_checked = 0
        
        for dataset_dir in dataset_dirs:
            total_checked += 1
            is_consistent, error_msg = check_single_dataset(versioning, dataset_dir, args.verbose)
            
            if not is_consistent:
                failed_datasets.append((dataset_dir, error_msg))
                
                if args.fail_fast:
                    print(f"\n💥 Failing fast due to consistency error in {dataset_dir}")
                    break
        
        # Report results
        print("\n" + "="*60)
        
        if failed_datasets:
            print(f"❌ CONSISTENCY CHECK FAILED")
            print(f"   {len(failed_datasets)}/{total_checked} datasets are inconsistent")
            
            print(f"\n🔧 To fix these issues, run:")
            for dataset_path, _ in failed_datasets:
                print(f"   {suggest_fix_command(dataset_path)}")
            
            print(f"\n💡 Or run this to fix all datasets:")
            print(f"   python scripts/dataset_register.py \"datasets/*/\" --name-from-path")
            
            sys.exit(1)
        
        else:
            print(f"✅ CONSISTENCY CHECK PASSED")
            print(f"   All {total_checked} datasets are consistent")
            print(f"   Dataset versioning is up to date! 🎉")
    
    except KeyboardInterrupt:
        print("\n⚠️  Check interrupted by user")
        sys.exit(2)
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main() 