"""
Dataset Export Manager for Multimodal Studio

Provides comprehensive dataset export capabilities including:
- Multi-format export support (HuggingFace, JSONL, PyTorch, Custom)
- Export configuration management
- Batch export operations
- Real-time progress tracking
- Export history and management
"""

import asyncio
import json
import uuid
import zipfile
import tarfile
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import shutil

logger = logging.getLogger(__name__)

class ExportFormat(str, Enum):
    HUGGINGFACE = "huggingface"
    JSONL = "jsonl"
    PYTORCH = "pytorch"
    CUSTOM = "custom"

class ExportStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ExportConfig:
    """Configuration for dataset export"""
    id: str
    name: str
    user_id: str
    format: ExportFormat
    options: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['format'] = self.format.value
        return data

@dataclass
class ExportJob:
    """Export job representation"""
    id: str
    dataset_id: str
    config_id: str
    user_id: str
    status: ExportStatus
    progress: float
    file_path: Optional[str]
    file_size: Optional[int]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'config_id': self.config_id,
            'user_id': self.user_id,
            'status': self.status.value,
            'progress': self.progress,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'metadata': self.metadata
        }

class ExportEngine:
    """Core export processing engine"""
    
    def __init__(self):
        self.export_base_path = Path("data/exports")
        self.export_base_path.mkdir(parents=True, exist_ok=True)
        
    async def export_dataset(
        self,
        dataset_id: str,
        config: ExportConfig,
        progress_callback: Optional[callable] = None
    ) -> tuple[str, int]:
        """Export dataset in specified format"""
        
        # Load dataset
        if progress_callback:
            await progress_callback(10.0, "Loading dataset...")
        
        dataset_path = f"data/datasets/{dataset_id}/dataset.jsonl"
        dataset = await self._load_dataset(dataset_path)
        
        if not dataset:
            raise ValueError(f"Could not load dataset {dataset_id}")
        
        # Create export directory
        export_dir = self.export_base_path / f"{dataset_id}_{config.id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if progress_callback:
            await progress_callback(20.0, f"Exporting to {config.format.value} format...")
        
        if config.format == ExportFormat.HUGGINGFACE:
            output_path = await self._export_huggingface(dataset, export_dir, config.options, progress_callback)
        elif config.format == ExportFormat.JSONL:
            output_path = await self._export_jsonl(dataset, export_dir, config.options, progress_callback)
        elif config.format == ExportFormat.PYTORCH:
            output_path = await self._export_pytorch(dataset, export_dir, config.options, progress_callback)
        elif config.format == ExportFormat.CUSTOM:
            output_path = await self._export_custom(dataset, export_dir, config.options, progress_callback)
        else:
            raise ValueError(f"Unsupported export format: {config.format}")
        
        # Get file size
        file_size = output_path.stat().st_size
        
        if progress_callback:
            await progress_callback(100.0, "Export completed")
        
        return str(output_path), file_size
    
    async def _load_dataset(self, dataset_path: str) -> Optional[List[Dict[str, Any]]]:
        """Load dataset from file"""
        try:
            path = Path(dataset_path)
            if not path.exists():
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix == '.jsonl':
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_path}: {e}")
            return None
    
    async def _export_huggingface(
        self,
        dataset: List[Dict[str, Any]],
        export_dir: Path,
        options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Export in HuggingFace Datasets format"""
        
        # Create HuggingFace dataset structure
        hf_dir = export_dir / "huggingface_dataset"
        hf_dir.mkdir(exist_ok=True)
        
        # Create dataset_info.json
        dataset_info = {
            "citation": options.get("citation", ""),
            "description": options.get("description", "Multimodal character dataset"),
            "features": {
                "text": {"dtype": "string", "_type": "Value"},
                "character": {"dtype": "string", "_type": "Value"},
                "metadata": {"dtype": "string", "_type": "Value"}
            },
            "homepage": options.get("homepage", ""),
            "license": options.get("license", ""),
            "size_in_bytes": 0,
            "splits": {
                "train": {
                    "name": "train",
                    "num_bytes": 0,
                    "num_examples": len(dataset)
                }
            },
            "version": {"version_str": "1.0.0", "major": 1, "minor": 0, "patch": 0}
        }
        
        if progress_callback:
            await progress_callback(40.0, "Creating HuggingFace structure...")
        
        # Save dataset_info.json
        with open(hf_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Convert and save data
        hf_data = []
        for i, sample in enumerate(dataset):
            hf_sample = {
                "text": sample.get("text", ""),
                "character": json.dumps(sample.get("character", {})),
                "metadata": json.dumps({k: v for k, v in sample.items() if k not in ["text", "character"]})
            }
            hf_data.append(hf_sample)
            
            if progress_callback and i % 100 == 0:
                progress = 40.0 + (i / len(dataset)) * 40.0
                await progress_callback(progress, f"Processing sample {i}/{len(dataset)}")
        
        # Save train split
        with open(hf_dir / "train.jsonl", 'w') as f:
            for sample in hf_data:
                f.write(json.dumps(sample) + '\n')
        
        if progress_callback:
            await progress_callback(90.0, "Creating archive...")
        
        # Create zip archive
        output_path = export_dir / "huggingface_dataset.zip"
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in hf_dir.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(hf_dir))
        
        return output_path
    
    async def _export_jsonl(
        self,
        dataset: List[Dict[str, Any]],
        export_dir: Path,
        options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Export in JSONL format"""
        
        output_path = export_dir / "dataset.jsonl"
        
        # Apply any transformations specified in options
        include_fields = options.get("include_fields", None)
        exclude_fields = options.get("exclude_fields", [])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(dataset):
                # Filter fields if specified
                if include_fields:
                    filtered_sample = {k: v for k, v in sample.items() if k in include_fields}
                else:
                    filtered_sample = {k: v for k, v in sample.items() if k not in exclude_fields}
                
                f.write(json.dumps(filtered_sample) + '\n')
                
                if progress_callback and i % 100 == 0:
                    progress = 30.0 + (i / len(dataset)) * 60.0
                    await progress_callback(progress, f"Writing sample {i}/{len(dataset)}")
        
        return output_path
    
    async def _export_pytorch(
        self,
        dataset: List[Dict[str, Any]],
        export_dir: Path,
        options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Export in PyTorch format"""
        
        # Create PyTorch dataset structure
        torch_dir = export_dir / "pytorch_dataset"
        torch_dir.mkdir(exist_ok=True)
        
        if progress_callback:
            await progress_callback(40.0, "Creating PyTorch structure...")
        
        # Create dataset.py file
        dataset_py = '''
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path

class CharacterDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'text': sample.get('text', ''),
            'character': sample.get('character', {}),
            'metadata': {k: v for k, v in sample.items() if k not in ['text', 'character']}
        }
'''
        
        with open(torch_dir / "dataset.py", 'w') as f:
            f.write(dataset_py)
        
        # Save data as JSONL
        data_path = torch_dir / "data.jsonl"
        with open(data_path, 'w') as f:
            for i, sample in enumerate(dataset):
                f.write(json.dumps(sample) + '\n')
                
                if progress_callback and i % 100 == 0:
                    progress = 50.0 + (i / len(dataset)) * 30.0
                    await progress_callback(progress, f"Writing sample {i}/{len(dataset)}")
        
        # Create README
        readme_content = '''# PyTorch Character Dataset

This dataset can be used with PyTorch for training character models.

## Usage

```python
from dataset import CharacterDataset

dataset = CharacterDataset('data.jsonl')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    texts = batch['text']
    characters = batch['character']
    metadata = batch['metadata']
    # Your training code here
```
'''
        
        with open(torch_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        if progress_callback:
            await progress_callback(90.0, "Creating archive...")
        
        # Create tar.gz archive
        output_path = export_dir / "pytorch_dataset.tar.gz"
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(torch_dir, arcname='pytorch_dataset')
        
        return output_path
    
    async def _export_custom(
        self,
        dataset: List[Dict[str, Any]],
        export_dir: Path,
        options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Export in custom format based on options"""
        
        format_type = options.get("format_type", "json")
        
        if format_type == "csv":
            return await self._export_csv(dataset, export_dir, options, progress_callback)
        elif format_type == "xml":
            return await self._export_xml(dataset, export_dir, options, progress_callback)
        else:
            # Default to JSON
            output_path = export_dir / "dataset.json"
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
            return output_path
    
    async def _export_csv(
        self,
        dataset: List[Dict[str, Any]],
        export_dir: Path,
        options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Export as CSV format"""
        import csv
        
        output_path = export_dir / "dataset.csv"
        
        # Get all unique field names
        all_fields = set()
        for sample in dataset:
            all_fields.update(sample.keys())
        
        field_names = sorted(list(all_fields))
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            
            for i, sample in enumerate(dataset):
                # Convert complex objects to JSON strings
                csv_row = {}
                for field in field_names:
                    value = sample.get(field, '')
                    if isinstance(value, (dict, list)):
                        csv_row[field] = json.dumps(value)
                    else:
                        csv_row[field] = str(value)
                
                writer.writerow(csv_row)
                
                if progress_callback and i % 100 == 0:
                    progress = 30.0 + (i / len(dataset)) * 60.0
                    await progress_callback(progress, f"Writing sample {i}/{len(dataset)}")
        
        return output_path
    
    async def _export_xml(
        self,
        dataset: List[Dict[str, Any]],
        export_dir: Path,
        options: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Export as XML format"""
        import xml.etree.ElementTree as ET
        
        output_path = export_dir / "dataset.xml"
        
        root = ET.Element("dataset")
        
        for i, sample in enumerate(dataset):
            sample_elem = ET.SubElement(root, "sample", {"id": str(i)})
            
            for key, value in sample.items():
                elem = ET.SubElement(sample_elem, key)
                if isinstance(value, (dict, list)):
                    elem.text = json.dumps(value)
                else:
                    elem.text = str(value)
            
            if progress_callback and i % 100 == 0:
                progress = 30.0 + (i / len(dataset)) * 60.0
                await progress_callback(progress, f"Writing sample {i}/{len(dataset)}")
        
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        return output_path

class DatasetExportManager:
    """Main export management service"""
    
    def __init__(self):
        self.engine = ExportEngine()
        self.configs_storage_path = Path("data/export_configs")
        self.configs_storage_path.mkdir(parents=True, exist_ok=True)
        self.export_configs: Dict[str, ExportConfig] = {}
        self.export_jobs: Dict[str, ExportJob] = {}
        self.user_exports: Dict[str, List[str]] = {}  # user_id -> [export_job_ids]
        
    async def create_config(
        self,
        user_id: str,
        name: str,
        format: str,
        options: Dict[str, Any]
    ) -> ExportConfig:
        """Create new export configuration"""
        config_id = str(uuid.uuid4())
        
        config = ExportConfig(
            id=config_id,
            name=name,
            user_id=user_id,
            format=ExportFormat(format),
            options=options,
            created_at=datetime.utcnow()
        )
        
        self.export_configs[config_id] = config
        await self._save_config(config)
        
        logger.info(f"Created export config {config_id} for user {user_id}")
        return config
    
    async def get_user_configs(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all export configurations for user"""
        configs = []
        for config in self.export_configs.values():
            if config.user_id == user_id:
                configs.append(config.to_dict())
        
        # Sort by creation time, most recent first
        configs.sort(key=lambda x: x['created_at'], reverse=True)
        return configs
    
    async def start_export(
        self,
        dataset_id: str,
        config_id: str,
        user_id: str
    ) -> ExportJob:
        """Start dataset export"""
        config = self.export_configs.get(config_id)
        if not config or config.user_id != user_id:
            raise ValueError("Export configuration not found")
        
        job_id = str(uuid.uuid4())
        
        job = ExportJob(
            id=job_id,
            dataset_id=dataset_id,
            config_id=config_id,
            user_id=user_id,
            status=ExportStatus.QUEUED,
            progress=0.0,
            file_path=None,
            file_size=None,
            created_at=datetime.utcnow()
        )
        
        self.export_jobs[job_id] = job
        
        # Add to user exports
        if user_id not in self.user_exports:
            self.user_exports[user_id] = []
        self.user_exports[user_id].append(job_id)
        
        logger.info(f"Started export job {job_id} for dataset {dataset_id}")
        return job
    
    async def process_export(self, job_id: str) -> None:
        """Process export job in background"""
        job = self.export_jobs.get(job_id)
        if not job:
            return
        
        config = self.export_configs.get(job.config_id)
        if not config:
            job.status = ExportStatus.FAILED
            job.error_message = "Export configuration not found"
            return
        
        try:
            job.status = ExportStatus.PROCESSING
            job.started_at = datetime.utcnow()
            
            # Progress callback
            async def update_progress(progress: float, message: str = ""):
                job.progress = progress
                job.metadata['current_step'] = message
            
            # Run export
            file_path, file_size = await self.engine.export_dataset(
                job.dataset_id,
                config,
                update_progress
            )
            
            job.status = ExportStatus.COMPLETED
            job.file_path = file_path
            job.file_size = file_size
            job.completed_at = datetime.utcnow()
            job.progress = 100.0
            
            logger.info(f"Export job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Export job {job_id} failed: {e}")
            job.status = ExportStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
    
    async def get_user_exports(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get export jobs for user"""
        exports = []
        
        user_job_ids = self.user_exports.get(user_id, [])
        for job_id in user_job_ids:
            job = self.export_jobs.get(job_id)
            if job and (not status or job.status.value == status):
                exports.append(job.to_dict())
        
        # Sort by creation time, most recent first
        exports.sort(key=lambda x: x['created_at'], reverse=True)
        
        return exports[:limit]
    
    async def get_export(self, export_id: str, user_id: str) -> Optional[ExportJob]:
        """Get specific export job"""
        job = self.export_jobs.get(export_id)
        if job and job.user_id == user_id:
            return job
        return None
    
    async def cancel_export(self, export_id: str, user_id: str) -> bool:
        """Cancel export job"""
        job = self.export_jobs.get(export_id)
        if not job or job.user_id != user_id:
            return False
        
        if job.status in [ExportStatus.COMPLETED, ExportStatus.FAILED, ExportStatus.CANCELLED]:
            return False
        
        job.status = ExportStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        job.error_message = "Cancelled by user"
        
        logger.info(f"Export job {export_id} cancelled by user {user_id}")
        return True
    
    async def delete_export(self, export_id: str, user_id: str) -> bool:
        """Delete export job and associated files"""
        job = self.export_jobs.get(export_id)
        if not job or job.user_id != user_id:
            return False
        
        # Delete export file if exists
        if job.file_path:
            try:
                file_path = Path(job.file_path)
                if file_path.exists():
                    if file_path.is_dir():
                        shutil.rmtree(file_path)
                    else:
                        file_path.unlink()
                        
                    # Also delete parent directory if empty
                    parent_dir = file_path.parent
                    if parent_dir.exists() and not any(parent_dir.iterdir()):
                        parent_dir.rmdir()
                        
            except Exception as e:
                logger.error(f"Failed to delete export file {job.file_path}: {e}")
        
        # Remove from collections
        del self.export_jobs[export_id]
        if user_id in self.user_exports:
            try:
                self.user_exports[user_id].remove(export_id)
            except ValueError:
                pass
        
        logger.info(f"Export job {export_id} deleted")
        return True
    
    async def _save_config(self, config: ExportConfig) -> None:
        """Save export configuration to storage"""
        config_file = self.configs_storage_path / f"{config.id}.json"
        try:
            with open(config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config {config.id}: {e}")
    
    async def cleanup_old_exports(self, days: int = 30) -> None:
        """Cleanup old export files and jobs"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        jobs_to_delete = []
        for job_id, job in self.export_jobs.items():
            if (job.completed_at and 
                job.completed_at < cutoff_date and 
                job.status in [ExportStatus.COMPLETED, ExportStatus.FAILED, ExportStatus.CANCELLED]):
                jobs_to_delete.append(job_id)
        
        for job_id in jobs_to_delete:
            job = self.export_jobs[job_id]
            await self.delete_export(job_id, job.user_id)
        
        logger.info(f"Cleaned up {len(jobs_to_delete)} old export jobs") 