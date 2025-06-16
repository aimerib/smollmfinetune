import json
import logging
import os
import tempfile
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_dataset_path(character: Dict[str, Any]) -> str:
    """Get the file path for a character's dataset"""
    char_name = character.get("name", "unknown")
    safe_name = "".join(c for c in char_name if c.isalnum() or c in " -_").strip()
    safe_name = safe_name.replace(" ", "_")
    
    # Create datasets directory if it doesn't exist
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    return str(datasets_dir / f"{safe_name}_dataset.json")


def save_dataset(character: Dict[str, Any], dataset: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save dataset to disk with optional metadata"""
    try:
        dataset_path = get_dataset_path(character)
        
        # Extract system prompt info from dataset if not provided in metadata
        if metadata is None:
            metadata = {}
        
        # Check if dataset uses consistent system prompts
        if dataset and len(dataset) > 0:
            first_system = dataset[0].get('messages', [{}])[0].get('content', '') if dataset[0].get('messages') else ''
            all_same_system = all(
                sample.get('messages', [{}])[0].get('content', '') == first_system 
                for sample in dataset 
                if sample.get('messages')
            )
            if all_same_system and 'system_prompt_config' not in metadata:
                metadata['system_prompt_config'] = {
                    'type': 'custom' if first_system else 'none',
                    'prompt': first_system
                }
            elif 'system_prompt_config' not in metadata:
                metadata['system_prompt_config'] = {
                    'type': 'temporal',
                    'prompt': None
                }
        
        with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(dataset_path), encoding="utf-8") as tmp_f:
            json.dump({
                'character': character,
                'dataset': dataset,
                'created_at': _time.time(),
                'sample_count': len(dataset),
                'metadata': metadata
            }, tmp_f, indent=2, ensure_ascii=False)
            tmp_path = tmp_f.name

        os.replace(tmp_path, dataset_path)
        logger.info(
            f"💾 Saved dataset with {len(dataset)} samples to {dataset_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save dataset: {e}")


def load_dataset(character: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Load existing dataset from disk"""
    try:
        dataset_path = get_dataset_path(character)
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dataset = data.get('dataset', [])
            logger.info(
                f"📂 Loaded existing dataset with {len(dataset)} samples from {dataset_path}")
            return dataset
        return None
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return None
        

def load_dataset_with_metadata(character: Dict[str, Any]) -> Optional[tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """Load existing dataset and metadata from disk"""
    try:
        dataset_path = get_dataset_path(character)
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            dataset = data.get('dataset', [])
            metadata = data.get('metadata', {})
            logger.info(
                f"📂 Loaded existing dataset with {len(dataset)} samples and metadata from {dataset_path}")
            return dataset, metadata
        return None
    except Exception as e:
        logger.error(f"❌ Failed to load dataset with metadata: {e}")
        return None


def get_dataset_info(character: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get dataset metadata without loading full dataset"""
    try:
        dataset_path = get_dataset_path(character)
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            metadata = data.get('metadata', {})
            return {
                'exists': True,
                'sample_count': data.get('sample_count', 0),
                'created_at': data.get('created_at', 'unknown'),
                'path': dataset_path,
                'system_prompt_config': metadata.get('system_prompt_config', {})
            }
        return {'exists': False}
    except Exception as e:
        logger.error(f"❌ Failed to get dataset info: {e}")
        return {'exists': False}


def delete_dataset(character: Dict[str, Any]) -> bool:
    """Delete stored dataset"""
    try:
        dataset_path = get_dataset_path(character)
        if os.path.exists(dataset_path):
            os.remove(dataset_path)
            logger.info(f"🗑️ Deleted dataset at {dataset_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Failed to delete dataset: {e}")
        return False


def export_dataset(character: Dict[str, Any]) -> Optional[str]:
    """Return the raw JSON string of a character's dataset for download.

    This is primarily used by the Streamlit UI to feed into a download
    button.  If no dataset exists, returns ``None``.
    """
    dataset_path = get_dataset_path(character)
    if not os.path.exists(dataset_path):
        logger.warning("No dataset found to export.")
        return None
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to export dataset: {e}")
        return None


def import_dataset_from_bytes(character: Dict[str, Any], raw_bytes: bytes,
                              merge_mode: str = "replace") -> bool:
    """Import a dataset JSON (bytes) for the given character.

    merge_mode:
      - "replace"  : overwrite any existing dataset
      - "append"   : append new samples (deduplicated) to existing dataset
    Returns ``True`` on success.
    """
    try:
        data = json.loads(raw_bytes.decode("utf-8"))

        # Validate structure
        if "dataset" not in data or not isinstance(data["dataset"], list):
            raise ValueError(
                "Invalid dataset file: missing 'dataset' list")

        imported_dataset = data["dataset"]

        if merge_mode == "append":
            existing = load_dataset(character) or []
            # Very naive deduplication by hashing messages tuple
            seen = {json.dumps(s, sort_keys=True) for s in existing}
            for sample in imported_dataset:
                key = json.dumps(sample, sort_keys=True)
                if key not in seen:
                    existing.append(sample)
                    seen.add(key)
            merged_dataset = existing
        else:
            merged_dataset = imported_dataset

        save_dataset(character, merged_dataset)
        logger.info(
            f"Imported dataset with {len(imported_dataset)} samples (mode={merge_mode})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to import dataset: {e}")
        return False